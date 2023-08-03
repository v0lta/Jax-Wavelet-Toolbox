"""Code for stationary wavelet transforms."""
from typing import List, Optional, Union

import jax
import jax.numpy as jnp
import pywt

from .conv_fwt import (
    _get_filter_arrays,  # _fwt_unpad,
    _postprocess_array_dec1d,
    _preprocess_array_dec1d,
)
from .utils import _as_wavelet, _fold_axes, _unfold_axes


def _swt(
    data: jnp.ndarray,
    wavelet: Union[pywt.Wavelet, str],
    level: Optional[int] = None,
    precision: str = "highest",
) -> List[jnp.ndarray]:
    """Compute a multilevel 1d stationary wavelet transform.

    Args:
        data (torch.Tensor): The input data of shape [batch_size, time].
        wavelet (Union[Wavelet, str]): The wavelet to use.
        level (Optional[int], optional): The number of levels to compute

    Returns:
        List[torch.Tensor]: Same as wavedec.
        Equivalent to pywt.swt with trim_approx=True.
    """
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
        result_list = _postprocess_array_dec1d(result_list, ds)

    return result_list[::-1]


def _iswt(
    coeffs: List[jnp.ndarray],
    wavelet: pywt.Wavelet,
    precision: str = "highest",
) -> jnp.ndarray:
    ds = None
    if coeffs[0].ndim > 2:
        fold_coeffs = []
        ds = list(coeffs[0].shape)
        for uf_coeff in coeffs:
            f_coeff, _ = _fold_axes(uf_coeff, 1)
            fold_coeffs.append(f_coeff)
        coeffs = fold_coeffs

    wavelet = _as_wavelet(wavelet)
    # unlike pytorch lax's transpose conv requires filter flips.
    _, _, rec_lo, rec_hi = _get_filter_arrays(wavelet, flip=True, dtype=coeffs[0].dtype)
    filt_len = rec_lo.shape[-1]
    filt = jnp.stack([rec_lo, rec_hi], 1)

    res_lo = coeffs[0]
    for c_pos, res_hi in enumerate(coeffs[1:]):
        dilation = 2 ** (len(coeffs[1:]) // 2 - c_pos)
        padl, padr = dilation * (filt_len // 2), dilation * (filt_len // 2 - 1)
        res_lo = jnp.stack([res_lo, res_hi], 1)
        res_lo = jnp.pad(
            res_lo, [(0, 0)] * (res_lo.ndim - 1) + [(padl, padr)], mode="wrap"
        )
        res_lo = res_lo[..., ::2]
        res_lo = jax.lax.conv_general_dilated(
            lhs=res_lo,
            rhs=filt,
            padding=[(0, 0)],
            lhs_dilation=[2],
            rhs_dilation=[dilation],
            window_strides=[1],
            dimension_numbers=("NCH", "OIH", "NCH"),
            precision=jax.lax.Precision(precision),
        )
        # res_lo = _fwt_unpad(res_lo, filt_len, c_pos, coeffs)
        # res_lo = res_lo[..., ::2]
        res_lo = res_lo.squeeze(1)

    if ds:
        res_lo = _unfold_axes(res_lo, ds, 1)
    return res_lo
