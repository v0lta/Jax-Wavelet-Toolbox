"""Code for 2D-stationary wavelet transforms."""
from typing import List, Optional, Union

import jax
import jax.numpy as jnp
import pywt

from .conv_fwt import _get_filter_arrays
from .conv_fwt_2d import (
    _preprocess_array_dec2d,
    _postprocess_result_list_dec2d,
    _preprocess_result_list_rec2d,
    _construct_2d_filt
)
from .utils import _as_wavelet, _unfold_axes


def swt2(
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
    data, ds = _preprocess_array_dec2d(data)

    dec_lo, dec_hi, _, _ = _get_filter_arrays(wavelet, flip=True, dtype=data.dtype)
    filt_len = dec_lo.shape[-1]
    dec_filt = _construct_2d_filt(lo=dec_lo, hi=dec_hi)

    if level is None:
        level = pywt.dwtn_max_level(
            [data.shape[-1], data.shape[-2]], wavelet
        )


    result_list = []
    res_lo = data
    for current_level in range(level):
        dilation = 2**current_level
        padl = (dilation * ( filt_len - 1)) // 4 
        padr = (dilation * ( filt_len - 1)) // 4 + 1
        res_lo = jnp.pad(
            res_lo, [(0, 0)] * (data.ndim - 2) + [(padl, padr), (padl, padr)],
            mode="wrap"
        )
        res = jax.lax.conv_general_dilated(
            lhs=res_lo,
            rhs=dec_filt,
            padding=[(0, 0), (0, 0)],
            window_strides=[1, 1],
            rhs_dilation=[dilation, dilation],
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            precision=jax.lax.Precision(precision),
        )
        res_ll, res_lh, res_hl, res_hh = jnp.split(res, 4, 1)
        result_list.append((res_lh.squeeze(1), res_hl.squeeze(1), res_hh.squeeze(1)))
    result_list.append(res_ll.squeeze(1))
    result_list.reverse()

    if ds:
        result_list = _postprocess_result_list_dec2d(result_list, ds)  # type: ignore

    return result_list