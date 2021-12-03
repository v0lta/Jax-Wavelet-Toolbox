"""Convolution based fast wavelet transforms."""

# -*- coding: utf-8 -*-

# Created on Thu Jun 11 2020
# Copyright (c) 2020 Moritz Wolter
#

import click
import jax
import jax.numpy as np
import pywt

from ._lorenz import generate_lorenz
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
        mode: The padding used to extend the input signal. Default: reflect.

    Returns:
        list: List containing the wavelet coefficients.
    """
    if len(data.shape) == 1:
        data = np.expand_dims(np.expand_dims(data, 0), 0)

    if mode == "zero":
        # translate pywt to numpy.
        mode = "constant"

    dec_lo, dec_hi, _, _ = get_filter_arrays(wavelet, flip=True)
    filt_len = dec_lo.shape[-1]
    filt = np.stack([dec_lo, dec_hi], 0)

    if level is None:
        # scales = pywt.dwt_max_level(data.shape[-1], filt_len)
        level = dwt_max_level(data.shape[-1], filt_len)

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
    _, _, rec_lo, rec_hi = get_filter_arrays(wavelet, flip=True)
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
    padr = 0
    padl = 0
    if filt_len > 2:
        # we pad half of the total requried padding on each side.
        padr += (2 * filt_len - 3) // 2
        padl += (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data.shape[-1] % 2 != 0:
        padr += 1

    data = np.pad(data, ((0, 0), (0, 0), (padl, padr)), mode)
    return data


def get_filter_arrays(wavelet: Wavelet, flip: bool) -> tuple:
    """Extract the filter coefficients from an input wavelet object.

    Args:
        wavelet (Wavelet): A pywt-style input wavelet.
        flip (bool): If true flip the input coefficients.

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
    dec_lo = create_array(dec_lo)
    dec_hi = create_array(dec_hi)
    rec_lo = create_array(rec_lo)
    rec_hi = create_array(rec_hi)
    return dec_lo, dec_hi, rec_lo, rec_hi


def dwt_max_level(data_len: int, filt_len: int) -> int:
    """Compute maximum number of possible levels.

    Args:
        data_len (int): Length of the input data.
        filt_len (int): Length of the wavelet filters.

    Returns:
        int: The number of possible levels.
    """
    return np.floor(np.log2(data_len / (filt_len - 1.0))).astype(np.int32)


@click.command()
@click.option("-o", "--output")
def main(output):
    """Run a test example."""
    # import os
    # os.environ["DISPLAY"] = ":1"
    # import matplotlib
    # matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    wavelet = pywt.Wavelet("haar")
    # ---- Test haar wavelet analysis and synthesis on lorenz signal. -----
    lorenz = np.transpose(np.expand_dims(generate_lorenz()[:, 0], -1), [1, 0])
    data = np.expand_dims(lorenz, 0)

    coeff = wavedec(data, wavelet)
    cat_coeff = np.concatenate(coeff, axis=-1)
    pywt_coeff = pywt.wavedec(lorenz, wavelet, mode="reflect")
    pywt_cat_coeff = np.concatenate(pywt_coeff, axis=-1)
    err = np.mean(np.abs(cat_coeff - pywt_cat_coeff))
    print("coefficient", err)

    rest_data = waverec(coeff, wavelet)
    err = np.mean(np.abs(rest_data - data))
    plt.plot(rest_data[0, 0, :])
    plt.plot(data[0, 0, :])
    if output is None:
        plt.show()
    else:
        plt.savefig(output)
    print(err, "done")


if __name__ == "__main__":
    main()
