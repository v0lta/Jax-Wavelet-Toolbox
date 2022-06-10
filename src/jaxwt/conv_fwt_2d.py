"""Two dimensional convolution based fast wavelet transforms."""
#
# Created on Thu Jun 12 2020
# Copyright (c) 2020 Moritz Wolter
#
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pywt
from jax.config import config

from .conv_fwt import _get_filter_arrays
from .utils import Wavelet

config.update("jax_enable_x64", True)


def wavedec2(
    data: jnp.ndarray,
    wavelet: Wavelet,
    level: Optional[int] = None,
    mode: str = "reflect",
) -> List[Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]]:
    """Compute the two dimensional wavelet analysis transform on the last two dimensions of the input data array.

    Args:
        data (jnp.array): Jax array containing the data to be transformed. Assumed shape:
                         [batch size, hight, width].
        wavelet (Wavelet): A namedtouple containing the filters for the transformation.
        level (int): The max level to be used, if not set as many levels as possible
                               will be used. Defaults to None.
        mode (str): The desired padding mode. Choose reflect or symmetric.
            Defaults to reflect.

    Returns:
        list: The wavelet coefficients in a nested list.
            The coefficients are in pywt order. That is:
            [cAn, (cHn, cVn, cDn), … (cH1, cV1, cD1)].
            A denotes approximation, H horizontal, V vertical
            and D diagonal coefficients.

    Examples:
        >>> import pywt, scipy.misc
        >>> import jaxwt as jwt
        >>> import jax.numpy as jnp
        >>> face = jnp.transpose(scipy.misc.face(), [2, 0, 1]).astype(jnp.float64)
        >>> jwt.wavedec2(face, pywt.Wavelet("haar"), level=2, mode="reflect")
    """
    data = jnp.expand_dims(data, 1)
    dec_lo, dec_hi, _, _ = _get_filter_arrays(wavelet, flip=True)
    dec_filt = construct_2d_filt(lo=dec_lo, hi=dec_hi)

    if mode == "zero":
        # translate pywt to numpy.
        mode = "constant"

    if level is None:
        level = pywt.dwtn_max_level(
            [data.shape[-1], data.shape[-2]], pywt.Wavelet("MyWavelet", wavelet)
        )

    result_lst: List[
        Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
    ] = []
    res_ll = data
    for _ in range(level):
        res_ll = _fwt_pad2d(res_ll, len(wavelet), mode=mode)
        res = jax.lax.conv_general_dilated(
            lhs=res_ll,  # lhs = NCHw image tensor
            rhs=dec_filt,  # rhs = OIHw conv kernel tensor
            padding="VALID",
            window_strides=[2, 2],
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        res_ll, res_lh, res_hl, res_hh = jnp.split(res, 4, 1)
        result_lst.append((res_lh.squeeze(1), res_hl.squeeze(1), res_hh.squeeze(1)))
    result_lst.append(res_ll.squeeze(1))
    result_lst.reverse()
    return result_lst


def waverec2(
    coeffs: List[Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]],
    wavelet: Wavelet,
) -> jnp.ndarray:
    """Compute a two dimensional synthesis wavelet transfrom.

       Use it to reconstruct the original input image from the wavelet coefficients.

    Args:
        coeffs (list): The input coefficients, typically the output of wavedec2.
        wavelet (Wavelet): The named tuple contining the filters used to compute the analysis transform.

    Returns:
        jnp.array: Reconstruction of the original input data array of shape [batch, height, width].

    Raises:
        ValueError: If `coeffs` is not in the shape as it is returned from `wavedec2`.

    Example:
        >>> import pywt, scipy.misc
        >>> import jaxwt as jwt
        >>> import jax.numpy as jnp
        >>> face = jnp.transpose(scipy.misc.face(), [2, 0, 1]).astype(jnp.float64)
        >>> transformed = jwt.wavedec2(face, pywt.Wavelet("haar"), level=2, mode="reflect")
        >>> jwt.waverec2(transformed, pywt.Wavelet("haar"))


    """
    if not isinstance(coeffs[0], jnp.ndarray):
        raise ValueError(
            "First element of coeffs must be the approximation coefficient tensor."
        )
    _, _, rec_lo, rec_hi = _get_filter_arrays(wavelet, flip=True, dtype=coeffs[0].dtype)
    filt_len = rec_lo.shape[-1]
    rec_filt = construct_2d_filt(lo=rec_lo, hi=rec_hi)
    rec_filt = jnp.transpose(rec_filt, [1, 0, 2, 3])

    res_ll = jnp.expand_dims(coeffs[0], 1)
    for c_pos, res_lh_hl_hh in enumerate(coeffs[1:]):
        res_ll = jnp.concatenate(
            [
                res_ll,
                jnp.expand_dims(res_lh_hl_hh[0], 1),
                jnp.expand_dims(res_lh_hl_hh[1], 1),
                jnp.expand_dims(res_lh_hl_hh[2], 1),
            ],
            1,
        )
        res_ll = jax.lax.conv_transpose(
            lhs=res_ll,
            rhs=rec_filt,
            padding="VALID",
            strides=[2, 2],
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        # remove the padding
        padl = (2 * filt_len - 3) // 2
        padr = (2 * filt_len - 3) // 2
        padt = (2 * filt_len - 3) // 2
        padb = (2 * filt_len - 3) // 2
        if c_pos < len(coeffs) - 2:
            pred_len = res_ll.shape[-1] - (padl + padr)
            next_len = coeffs[c_pos + 2][0].shape[-1]
            pred_len2 = res_ll.shape[-2] - (padt + padb)
            next_len2 = coeffs[c_pos + 2][0].shape[-2]
            if next_len != pred_len:
                padr += 1
                pred_len = res_ll.shape[-1] - (padl + padr)
                assert (
                    next_len == pred_len
                ), "padding error, please open an issue on github "
            if next_len2 != pred_len2:
                padb += 1
                pred_len2 = res_ll.shape[-2] - (padt + padb)
                assert (
                    next_len2 == pred_len2
                ), "padding error, please open an issue on github "
        # print('padding', padt, padb, padl, padr)
        if padt > 0:
            res_ll = res_ll[..., padt:, :]
        if padb > 0:
            res_ll = res_ll[..., :-padb, :]
        if padl > 0:
            res_ll = res_ll[..., padl:]
        if padr > 0:
            res_ll = res_ll[..., :-padr]
    return res_ll.squeeze(1)


def construct_2d_filt(lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
    """Construct 2d filters from 1d inputs using outer products.

    Args:
        lo (jnp.array): 1d lowpass input filter of size [1, length].
        hi (jnp.array): 1d highpass input filter of size [1, length].

    Returns
        jnp.array: 2d filter arrays of shape [4, 1, length, length].
    """
    ll = jnp.outer(lo, lo)
    lh = jnp.outer(hi, lo)
    hl = jnp.outer(lo, hi)
    hh = jnp.outer(hi, hi)
    filt = jnp.stack([ll, lh, hl, hh], 0)
    filt = jnp.expand_dims(filt, 1)
    return filt


def _fwt_pad2d(data: jnp.ndarray, filt_len: int, mode: str = "reflect") -> jnp.ndarray:
    padr = 0
    padl = 0
    padt = 0
    padb = 0
    if filt_len > 2:
        # we pad half of the total requried padding on each side.
        padr += (2 * filt_len - 3) // 2
        padl += (2 * filt_len - 3) // 2
        padt += (2 * filt_len - 3) // 2
        padb += (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data.shape[-1] % 2 != 0:
        padr += 1
    if data.shape[-2] % 2 != 0:
        padb += 1

    data = jnp.pad(data, ((0, 0), (0, 0), (padt, padb), (padl, padr)), mode)
    return data
