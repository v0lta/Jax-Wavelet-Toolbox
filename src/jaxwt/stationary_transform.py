"""Code for stationary wavelet transforms."""

#
# Created on Fri Aug 04 2023
# Copyright (c) 2023 Moritz Wolter
#

from typing import List, Optional, Union

import jax
import jax.numpy as jnp
import pywt

from .conv_fwt import (
    _get_filter_arrays,
    _postprocess_result_list_dec1d,
    _preprocess_array_dec1d,
    _preprocess_result_list_rec1d,
)
from .utils import _as_wavelet, _unfold_axes


def swt(
    data: jnp.ndarray,
    wavelet: Union[pywt.Wavelet, str],
    level: Optional[int] = None,
    axis: int = -1,
    precision: str = "highest",
) -> List[jnp.ndarray]:
    """Compute a multilevel 1d stationary wavelet transform.

    Args:
        data (jnp.ndarray): The input data of shape [batch_size, time].
            This function assumes a trailing input dimension with
            a length divisible by two.
        wavelet (Union[Wavelet, str]): The wavelet to use.
        level (Optional[int], optional): The number of levels to compute
        axis (int): Compute the transform over this axis instead of the
            last one. Defaults to -1.
        precision (str): For the desired precision, choose "fastest", "high" or "highest".
            Defaults to "highest".

    Returns:
        List[jnp.ndarray]: A list containing the wavelet coefficients.
            The coefficients are in pywt order:
            [cA_n, cD_n, cD_n-1, …, cD2, cD1].
            A denotes approximation and D detail coefficients.
            The ordering is identical to the ``wavedec`` function.
            Equivalent to pywt.swt with trim_approx=True.

    Raises:
        ValueError: If the axis argument is not an integer.

    Example:
        >>> import jax, jaxwt
        >>> import jax.numpy as jnp
        >>> signal = jax.random.randint(
        >>>     jax.random.PRNGKey(42), [1, 10], 0, 9).astype(jnp.float32)
        >>> jaxwt.swt(signal, "haar", level=2)
    """
    if axis != -1:
        if isinstance(axis, int):
            data = data.swapaxes(axis, -1)
        else:
            raise ValueError("swt transforms a single axis.")

    wavelet = _as_wavelet(wavelet)
    data, ds = _preprocess_array_dec1d(data)

    dec_lo, dec_hi, _, _ = _get_filter_arrays(wavelet, flip=True, dtype=data.dtype)
    filt_len = dec_lo.shape[-1]
    filt = jnp.stack([dec_lo, dec_hi], 0)

    if level is None:
        level = pywt.swt_max_level(data.shape[-1])

    result_list = []
    res_lo = data
    for current_level in range(level):
        dilation = 2**current_level
        padl, padr = dilation * (filt_len // 2 - 1), dilation * (filt_len // 2)
        res_lo = jnp.pad(
            res_lo, [(0, 0)] * (data.ndim - 1) + [(padl, padr)], mode="wrap"
        )
        res = jax.lax.conv_general_dilated(
            lhs=res_lo,  # lhs = NCH image tensor
            rhs=filt,  # rhs = OIH conv kernel tensor
            padding=[(0, 0)],
            window_strides=[1],
            rhs_dilation=[dilation],
            dimension_numbers=("NCT", "OIT", "NCT"),
            precision=jax.lax.Precision(precision),
        )
        res_lo, res_hi = jnp.split(res, 2, 1)
        # Trim_approx == False
        # result_list.append((res_lo.squeeze(1), res_hi.squeeze(1)))
        result_list.append(res_hi.squeeze(1))
    result_list.append(res_lo.squeeze(1))

    if ds:
        result_list = _postprocess_result_list_dec1d(result_list, ds)

    if axis != -1:
        swap = []
        for coeff in result_list:
            swap.append(coeff.swapaxes(axis, -1))
        result_list = swap

    return result_list[::-1]


def _conv_transpose_dedilate(
    conv_res: jnp.ndarray,
    rec_filt: jnp.ndarray,
    dilation: int,
    length: int,
    precision: Optional[str] = "highest",
) -> jnp.ndarray:
    """Undo the forward dilated convolution from the analysis transform.

    Args:
        conv_res (jnp.ndarray): The dilated coeffcients
            of shape [batch, 2, length].
        rec_filt (jnp.ndarray): The reconstruction filter pair
            of shape [1, 2, filter_length].
        dilation (int): The dilation factor.
        length (int): The signal length.
        precision (str): Defaults to "highest".

    Returns:
        jnp.ndarray: The convolution result.

    """
    recs = []
    to_conv_t_list = [
        conv_res[..., fl : (fl + dilation * rec_filt.shape[-1]) : dilation]
        for fl in range(length)
    ]
    to_conv_t = jnp.concatenate(to_conv_t_list, 0)
    rec = jax.lax.conv_transpose(
        lhs=to_conv_t,
        rhs=rec_filt,
        padding=[(0, 0)],
        strides=[1],
        dimension_numbers=("NCH", "OIH", "NCH"),
        precision=jax.lax.Precision(precision),
    )
    rec = rec / 2.0
    recs = jnp.split(rec, len(to_conv_t_list))
    return jnp.concatenate(recs, -1)


def iswt(
    coeffs: List[jnp.ndarray],
    wavelet: Union[pywt.Wavelet, str],
    axis: int = -1,
    precision: Optional[str] = "highest",
) -> jnp.ndarray:
    """Compute an inverse stationary wavelet transform.

    Args:
        coeffs (List[jnp.ndarray]): The coefficients as computed by the analysis code.
        wavelet (Union[pywt.Wavelet, str]): The wavelet used by the transform.
        axis (int): Transform this axis instead of the last one. Defaults to -1.
        precision (Optional[str]): Precision value for the underlying lax convolution code.
            Defaults to "highest".

    Raises:
        ValueError: If the axis argument is not an integer.

    Returns:
        jnp.ndarray: The reconstruction of the original signal.


    Example:
        >>> import jax, jaxwt
        >>> import jax.numpy as jnp
        >>> signal = jax.random.randint(
                jax.random.PRNGKey(42), [1, 10], 0, 9).astype(jnp.float32)
        >>> jaxwt.iswt(jaxwt.swt(signal, "haar", level=2), "haar")
    """
    if axis != -1:
        swap = []
        if isinstance(axis, int):
            for coeff in coeffs:
                swap.append(coeff.swapaxes(axis, -1))
            coeffs = swap
        else:
            raise ValueError("iswt transforms a single axis only.")

    ds = None
    length = coeffs[0].shape[-1]
    if coeffs[0].ndim > 2:
        coeffs, ds = _preprocess_result_list_rec1d(coeffs)

    wavelet = _as_wavelet(wavelet)
    # unlike pytorch lax's transpose conv requires filter flips.
    _, _, rec_lo, rec_hi = _get_filter_arrays(wavelet, flip=True, dtype=coeffs[0].dtype)
    filt_len = rec_lo.shape[-1]
    rec_filt = jnp.stack([rec_lo, rec_hi], 1)

    res_lo = coeffs[0]
    for c_pos, res_hi in enumerate(coeffs[1:]):
        dilation = 2 ** (len(coeffs[1:]) - c_pos - 1)
        res_lo = jnp.stack([res_lo, res_hi], 1)
        padl, padr = dilation * (filt_len // 2), dilation * (filt_len // 2 - 1)
        res_lo = jnp.pad(
            res_lo, [(0, 0)] * (res_lo.ndim - 1) + [(padl, padr)], mode="wrap"
        )
        res_lo = _conv_transpose_dedilate(res_lo, rec_filt, dilation, length, precision)
        res_lo = res_lo.squeeze(1)

    if ds:
        res_lo = _unfold_axes(res_lo, ds, 1)
    if axis != -1:
        res_lo = res_lo.swapaxes(axis, -1)

    return res_lo
