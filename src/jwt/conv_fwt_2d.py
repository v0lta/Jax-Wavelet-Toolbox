"""Two dimensional convolution based fast wavelet transforms."""
#
# Created on Thu Jun 12 2020
# Copyright (c) 2020 Moritz Wolter
#

from typing import Optional

import click
import jax
import jax.numpy as np
import pywt

from .conv_fwt import get_filter_arrays
from .utils import Wavelet, flatten_2d_coeff_lst


def wavedec2(data: np.array, wavelet: Wavelet, level: int = None) -> list:
    """Compute the two dimensional wavelet analysis transform on the last two dimensions of the input data array.

    Args:
        data (np.array): Jax array containing the data to be transformed. Assumed shape:
                         [batch size, channels, hight, width].
        wavelet (Wavelet): A namedtouple containing the filters for the transformation.
        level (int): The max level to be used, if not set as many levels as possible
                               will be used. Defaults to None.

    Returns:
        list: The wavelet coefficients in a nested list.
    """
    dec_lo, dec_hi, _, _ = get_filter_arrays(wavelet, flip=True)
    dec_filt = construct_2d_filt(lo=dec_lo, hi=dec_hi)
    # filt_len = dec_lo.shape[-1]

    if level is None:
        level = pywt.dwtn_max_level(
            [data.shape[-1], data.shape[-2]], pywt.Wavelet("MyWavelet", wavelet)
        )

    result_lst = []
    res_ll = data
    for _ in range(level):
        res_ll = _fwt_pad2d(res_ll, len(wavelet.dec_lo))
        res = jax.lax.conv_general_dilated(
            lhs=res_ll,  # lhs = NCHw image tensor
            rhs=dec_filt,  # rhs = OIHw conv kernel tensor
            padding="VALID",
            window_strides=[2, 2],
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

        res_ll, res_lh, res_hl, res_hh = np.split(res, 4, 1)
        result_lst.append((res_lh, res_hl, res_hh))
    result_lst.append(res_ll)
    result_lst.reverse()
    return result_lst


def waverec2(coeffs: list, wavelet: Wavelet) -> np.array:
    """Compute a two dimensional synthesis wavelet transfrom.

       Use it to reconstruct the original input image from the wavelet coefficients.

    Args:
        coeffs (list): The input coefficients, typically the output of wavedec2.
        wavelet (Wavelet): The named touple contining the filters used to compute the analysis transform.

    Returns:
        np.array: Reconstruction of the original input data array of shape [batch, channel, height, width].
    """
    _, _, rec_lo, rec_hi = get_filter_arrays(wavelet, flip=True)
    filt_len = rec_lo.shape[-1]
    rec_filt = construct_2d_filt(lo=rec_lo, hi=rec_hi)
    rec_filt = np.transpose(rec_filt, [1, 0, 2, 3])

    res_ll = coeffs[0]
    for c_pos, res_lh_hl_hh in enumerate(coeffs[1:]):
        res_ll = np.concatenate(
            [res_ll, res_lh_hl_hh[0], res_lh_hl_hh[1], res_lh_hl_hh[2]], 1
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
    return res_ll


def construct_2d_filt(lo, hi):
    """Construct 2d filters from 1d inputs."""
    ll = np.outer(lo, lo)
    lh = np.outer(hi, lo)
    hl = np.outer(lo, hi)
    hh = np.outer(hi, hi)
    filt = np.stack([ll, lh, hl, hh], 0)
    filt = np.expand_dims(filt, 1)
    return filt


def _fwt_pad2d(data: np.array, filt_len: int, mode="reflect") -> np.array:
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

    data = np.pad(data, ((0, 0), (0, 0), (padt, padb), (padl, padr)), mode)
    return data


@click.command()
@click.option("-o", "--output")
@click.option("--level", type=int)
def main(output, level: Optional[int]):
    """Run some toy examples."""
    import matplotlib.pyplot as plt
    import scipy.misc

    # os.environ["DISPLAY"] = ":1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # matplotlib.use('Qt5Agg')
    face = np.transpose(scipy.misc.face(), [2, 0, 1]).astype(np.float32)
    face = face[:, 128 : (512 + 128), 256 : (512 + 256)]
    face_exd = np.expand_dims(np.array(face), 1)
    wavelet = pywt.Wavelet("haar")
    jax_wavelet = Wavelet(
        wavelet.dec_lo, wavelet.dec_hi, wavelet.rec_lo, wavelet.rec_hi
    )

    print(f"Using level: {level}")
    coeff2d_pywt = pywt.wavedec2(face, wavelet, mode="reflect", level=level)
    coeff2d = wavedec2(face_exd, jax_wavelet, level=level)
    recss2d = waverec2(coeff2d, jax_wavelet)

    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = np.concatenate(flatten_2d_coeff_lst(coeff2d), -1)
    errc = np.mean(np.abs(flat_lst - flat_lst2))
    print("coefficient error", errc)
    print("done")

    print("err pywt", np.mean(np.abs(pywt.waverec2(coeff2d, wavelet) - face_exd)))
    print("err", np.mean(np.abs(recss2d - face_exd)))
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(
        np.transpose(recss2d[:, 0, :, :], [1, 2, 0]) / np.max(np.abs(recss2d))
    )
    errimg = np.abs(recss2d - face_exd)
    axes[1].imshow(np.transpose(errimg[:, 0, :, :], [1, 2, 0]) / np.max(np.abs(errimg)))
    if output is None:
        plt.show()
    else:
        plt.savefig(output)


if __name__ == "__main__":
    main()
