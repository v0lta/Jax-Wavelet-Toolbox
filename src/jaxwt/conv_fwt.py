"""Convolution based fast wavelet transforms."""

# -*- coding: utf-8 -*-

# Created on Thu Jun 11 2020
# Copyright (c) 2020 Moritz Wolter
#

import jax
import jax.numpy as np
import pywt

from .utils import Wavelet


def dwt(data: np.array, wavelet: Wavelet, mode="reflect") -> tuple:
    """Single level discrete analysis wavelet transform.

    Args:
        data (np.array): The input data array
        wavelet (Wavelet): The wavelet to use. The fields
            "dec_lo", "dec_hi", "rec_lo" and "rec_hi" must be defined.
        mode (str): The desired way to pad. Defaults to "reflect".

    Returns:
        tuple: The low and high-pass filtered coefficients,
            res_lo and res_hi.

    # TODO: add the shapes.
    """
    dec_lo, dec_hi, _, _ = get_filter_arrays(wavelet, flip=True)
    filt = np.stack([dec_lo, dec_hi], 0)
    res_lo = _fwt_pad(data, len(wavelet.dec_lo), mode)
    res = jax.lax.conv_general_dilated(
        lhs=res_lo,  # lhs = NCH image tensor
        rhs=filt,  # rhs = OIH conv kernel tensor
        padding="VALID",
        window_strides=[
            2,
        ],
        dimension_numbers=("NCH", "OIH", "NCH"),
    )
    res_lo, res_hi = np.split(res, 2, 1)
    return res_lo, res_hi


def idwt(coeff_lst: list, wavelet: Wavelet) -> np.array:
    """Single level synthesis transform.

    Args:
        coeff_lst (list): A list of wavelet coefficients.
        wavelet (Wavelet): The wavelet used in the analysis transform.
            Jwt follows pywt convention, the fields
            "dec_lo", "dec_hi", "rec_lo" and "rec_hi" must be defined.

    Returns:
        np.array: A reconstruction of the original input.
    """
    _, _, rec_lo, rec_hi = get_filter_arrays(wavelet, flip=True)
    filt_len = rec_lo.shape[-1]
    filt = np.stack([rec_lo, rec_hi], 1)
    res_lo = np.concatenate([coeff_lst[0], coeff_lst[1]], 1)
    rec = jax.lax.conv_transpose(
        lhs=res_lo,
        rhs=filt,
        padding="VALID",
        strides=[
            2,
        ],
        dimension_numbers=("NCH", "OIH", "NCH"),
    )
    rec = _fwt_unpad(rec, filt_len, 0, coeff_lst)
    return rec


def wavedec(
    data: np.array, wavelet: Wavelet, level: int = None, mode: str = "reflect"
) -> list:
    """Compute the one dimensional analysis wavelet transform of the last dimension.

    Args:
        data (np.array): Input data array of shape [batch, channels, time]
        wavelet (Wavelet): The named tuple containing the wavelet filter arrays.
        level (int): Max scale level to be used, of none as many levels as possible are
                     used. Defaults to None.
        mode: The padding used to extend the input signal. Choose reflect or symmetric.
            Defaults to reflect.

    Returns:
        list: List containing the wavelet coefficients.
    """
    if len(data.shape) == 1:
        data = np.expand_dims(np.expand_dims(data, 0), 0)

    if mode == "zero":
        # translate pywt to numpy.
        mode = "constant"

    dec_lo, dec_hi, _, _ = get_filter_arrays(wavelet, flip=True, dtype=data.dtype)
    filt_len = dec_lo.shape[-1]
    filt = np.stack([dec_lo, dec_hi], 0)

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
        res_lo, res_hi = np.split(res, 2, 1)
        result_lst.append(res_hi)
    result_lst.append(res_lo)
    result_lst.reverse()
    return result_lst


def waverec(coeffs: list, wavelet: Wavelet) -> np.array:
    """Reconstruct the original signal in one dimension.

    Args:
        coeffs (list): Wavelet coefficients, typically produced by the wavedec function.
        wavelet (Wavelet): The named tuple containing the wavelet filters used to evaluate
                              the decomposition.

    Returns:
        np.array: Reconstruction of the original data.
    """
    # lax's transpose conv requires filter flips in contrast to pytorch.
    _, _, rec_lo, rec_hi = get_filter_arrays(wavelet, flip=True, dtype=coeffs[0].dtype)
    filt_len = rec_lo.shape[-1]
    filt = np.stack([rec_lo, rec_hi], 1)

    res_lo = coeffs[0]
    for c_pos, res_hi in enumerate(coeffs[1:]):
        # print('shapes', res_lo.shape, res_hi.shape)
        res_lo = np.concatenate([res_lo, res_hi], 1)
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


def _fwt_unpad(res_lo, filt_len, c_pos, coeffs):
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


def _fwt_pad(data: np.array, filt_len: int, mode: str = "reflect") -> np.array:
    """Pad an input to ensure our fwts are invertible.

    Args:
        data (np.array): The input array.
        filt_len (int): The length of the wavelet filters
        mode (str): How to pad. Defaults to "reflect".

    Returns:
        np.array: A padded version of the input data array.
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

    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data.shape[-1] % 2 != 0:
        padr += 1

    data = np.pad(data, ((0, 0), (0, 0), (padl, padr)), mode)
    return data


def get_filter_arrays(
    wavelet: Wavelet, flip: bool, dtype: np.dtype = np.float64
) -> tuple:
    """Extract the filter coefficients from an input wavelet object.

    Args:
        wavelet (Wavelet): A pywt-style input wavelet.
        flip (bool): If true flip the input coefficients.
        dtype: The desired precision. Defaults to np.float64 .

    Returns:
        tuple: The dec_lo, dec_hi, rec_lo and rec_hi
            filter coefficients as jax arrays.
    """

    def create_array(filter):
        if flip:
            if type(filter) is np.array:
                return np.expand_dims(np.flip(filter), 0)
            else:
                return np.expand_dims(np.array(filter[::-1]), 0)
        else:
            if type(filter) is np.array:
                return np.expand_dims(filter, 0)
            else:
                return np.expand_dims(np.array(filter), 0)

    if type(wavelet) is pywt.Wavelet:
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    else:
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet
    dec_lo = create_array(dec_lo).astype(dtype)
    dec_hi = create_array(dec_hi).astype(dtype)
    rec_lo = create_array(rec_lo).astype(dtype)
    rec_hi = create_array(rec_hi).astype(dtype)
    return dec_lo, dec_hi, rec_lo, rec_hi
