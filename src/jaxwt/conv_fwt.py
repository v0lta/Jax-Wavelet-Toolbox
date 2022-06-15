"""Convolution based fast wavelet transforms."""

# -*- coding: utf-8 -*-

# Created on Thu Jun 11 2020
# Copyright (c) 2020 Moritz Wolter
#
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pywt

from .utils import Wavelet


def wavedec(
    data: jnp.ndarray,
    wavelet: Wavelet,
    level: Optional[int] = None,
    mode: str = "reflect",
) -> List[jnp.ndarray]:
    """Compute the one dimensional analysis wavelet transform of the last dimension.

    Args:
        data (jnp.array): Input data array of shape [batch, channels, time]
        wavelet (Wavelet): The named tuple containing the wavelet filter arrays.
        level (int): Max scale level to be used, of none as many levels as possible are
                     used. Defaults to None.
        mode: The padding used to extend the input signal. Choose reflect or symmetric.
            Defaults to reflect.

    Returns:
        list: List containing the wavelet coefficients.
            The coefficients are in pywt order:
            [cA_n, cD_n, cD_n-1, …, cD2, cD1].
            A denotes approximation and D detail coefficients.

    Examples:
        >>> import pywt
        >>> import jaxwt as jwt
        >>> import jax.numpy as np
        >>> # generate an input of even length.
        >>> data = jnp.array([0., 1., 2., 3, 4, 5, 5, 4, 3, 2, 1, 0])
        >>> jwt.wavedec(data, pywt.Wavelet('haar'),
                        mode='reflect', level=2)
    """
    if len(data.shape) == 1:
        # add channel and batch dimension.
        data = jnp.expand_dims(jnp.expand_dims(data, 0), 0)
    if len(data.shape) == 2:
        # add the channel dimension.
        data = jnp.expand_dims(data, 1)

    dec_lo, dec_hi, _, _ = _get_filter_arrays(wavelet, flip=True, dtype=data.dtype)
    filt_len = dec_lo.shape[-1]
    filt = jnp.stack([dec_lo, dec_hi], 0)

    if level is None:
        level = pywt.dwt_max_level(data.shape[-1], filt_len)

    result_lst = []
    res_lo = data
    for _ in range(level):
        res_lo = _fwt_pad(res_lo, len(wavelet.dec_lo), mode=mode)
        res = jax.lax.conv_general_dilated(
            lhs=res_lo,  # lhs = NCH image tensor
            rhs=filt,  # rhs = OIH conv kernel tensor
            padding="VALID",
            window_strides=[
                2,
            ],
            dimension_numbers=("NCH", "OIH", "NCH"),
        )
        res_lo, res_hi = jnp.split(res, 2, 1)
        result_lst.append(res_hi)
    result_lst.append(res_lo)
    result_lst.reverse()
    return result_lst


def waverec(coeffs: List[jnp.ndarray], wavelet: Wavelet) -> jnp.ndarray:
    """Reconstruct the original signal in one dimension.

    Args:
        coeffs (list): Wavelet coefficients, typically produced by the wavedec function.
        wavelet (Wavelet): The named tuple containing the wavelet filters used to evaluate
                              the decomposition.

    Returns:
        jnp.array: Reconstruction of the original data.

    Examples:
        >>> import pywt
        >>> import jaxwt as jwt
        >>> import jax.numpy as np
        >>> # generate an input of even length.
        >>> data = jnp.array([0., 1., 2., 3, 4, 5, 5, 4, 3, 2, 1, 0])
        >>> transformed = jwt.wavedec(data, pywt.Wavelet('haar'),
                          mode='reflect', level=2)
        >>> jwt.waverec(transformed, pywt.Wavelet('haar'))
    """
    # lax's transpose conv requires filter flips in contrast to pytorch.
    _, _, rec_lo, rec_hi = _get_filter_arrays(wavelet, flip=True, dtype=coeffs[0].dtype)
    filt_len = rec_lo.shape[-1]
    filt = jnp.stack([rec_lo, rec_hi], 1)

    res_lo = coeffs[0]
    for c_pos, res_hi in enumerate(coeffs[1:]):
        # print('shapes', res_lo.shape, res_hi.shape)
        res_lo = jnp.concatenate([res_lo, res_hi], 1)
        res_lo = jax.lax.conv_transpose(
            lhs=res_lo,
            rhs=filt,
            padding="VALID",
            strides=[
                2,
            ],
            dimension_numbers=("NCH", "OIH", "NCH"),
        )
        res_lo = _fwt_unpad(res_lo, filt_len, c_pos, coeffs)
    return res_lo


def _fwt_unpad(
    res_lo: jnp.ndarray, filt_len: int, c_pos: int, coeffs: List[jnp.ndarray]
) -> jnp.ndarray:
    padr = 0
    padl = 0
    if filt_len > 2:
        padr += (2 * filt_len - 3) // 2
        padl += (2 * filt_len - 3) // 2
    if c_pos < len(coeffs) - 2:
        pred_len = res_lo.shape[-1] - (padl + padr)
        nex_len = coeffs[c_pos + 2].shape[-1]
        if nex_len != pred_len:
            padl += 1
            pred_len = res_lo.shape[-1] - padl
            # assert nex_len == pred_len, 'padding error, please open an issue on github '
    if padl == 0:
        res_lo = res_lo[..., padr:]
    else:
        res_lo = res_lo[..., padr:-padl]
    return res_lo


def _fwt_pad(data: jnp.ndarray, filt_len: int, mode: str = "reflect") -> jnp.ndarray:
    """Pad an input to ensure our fwts are invertible.

    Args:
        data (jnp.array): The input array.
        filt_len (int): The length of the wavelet filters
        mode (str): How to pad. Defaults to "reflect".

    Returns:
        jnp.array: A padded version of the input data array.
    """
    # pad to we see all filter positions and pywt compatability.
    # convolution output length:
    # see https://arxiv.org/pdf/1603.07285.pdf section 2.3:
    # floor([data_len - filt_len]/2) + 1
    # should equal pywt output length
    # floor((data_len + filt_len - 1)/2)
    # => floor([data_len + total_pad - filt_len]/2) + 1
    #    = floor((data_len + filt_len - 1)/2)
    # (data_len + total_pad - filt_len) + 2 = data_len + filt_len - 1
    # total_pad = 2*filt_len - 3

    if mode == "zero":
        # translate pywt to numpy.
        mode = "constant"

    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data.shape[-1] % 2 != 0:
        padr += 1

    data = jnp.pad(data, ((0, 0), (0, 0), (padl, padr)), mode)
    return data


def _get_filter_arrays(
    wavelet: Wavelet, flip: bool, dtype: jnp.dtype = jnp.float64  # type: ignore
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract the filter coefficients from an input wavelet object.

    Args:
        wavelet (Wavelet): A pywt-style input wavelet.
        flip (bool): If true flip the input coefficients.
        dtype: The desired precision. Defaults to jnp.float64 .

    Returns:
        tuple: The dec_lo, dec_hi, rec_lo and rec_hi
            filter coefficients as jax arrays.
    """

    def create_array(filter: Union[List[float], jnp.ndarray]) -> jnp.ndarray:
        if flip:
            if type(filter) is jnp.ndarray:
                return jnp.expand_dims(jnp.flip(filter), 0)
            else:
                return jnp.expand_dims(jnp.array(filter[::-1]), 0)
        else:
            if type(filter) is jnp.ndarray:
                return jnp.expand_dims(filter, 0)
            else:
                return jnp.expand_dims(jnp.array(filter), 0)

    if isinstance(wavelet, str):
        wavelet = pywt.Wavelet(wavelet)
    elif type(wavelet) is pywt.Wavelet:
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    else:
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet
    dec_lo = create_array(dec_lo).astype(dtype)
    dec_hi = create_array(dec_hi).astype(dtype)
    rec_lo = create_array(rec_lo).astype(dtype)
    rec_hi = create_array(rec_hi).astype(dtype)
    return dec_lo, dec_hi, rec_lo, rec_hi
