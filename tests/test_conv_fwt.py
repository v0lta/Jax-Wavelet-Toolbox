"""Convolution fast wavelet transform test code."""

#
# Created on Thu Jun 11 2020
# Copyright (c) 2020 Moritz Wolter
#

import jax.numpy as np
import pywt
from jax.config import config
from src.jwt._lorenz import generate_lorenz
from src.jwt.conv_fwt import wavedec, waverec

config.update("jax_enable_x64", True)


def test_haar_fwt_ifwt_16_float32():
    """Test Haar wavelet analysis and synthesis on 16 sample signal."""
    wavelet = pywt.Wavelet("haar")
    data = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
        ]
    ).astype(np.float32)
    data = np.expand_dims(np.expand_dims(data, 0), 0)
    coeffs = pywt.wavedec(data, wavelet, level=2)
    coeffs2 = wavedec(data, wavelet, level=2)
    cat_coeffs = np.concatenate(coeffs, -1)
    cat_coeffs2 = np.concatenate(coeffs2, -1)
    err = np.mean(np.abs(cat_coeffs - cat_coeffs2))
    assert err < 1e-4
    rest_data = waverec(coeffs2, wavelet)
    err = np.mean(np.abs(rest_data - data))
    assert err < 1e-4


def fwt_ifwt_lorenz(wavelet, mode: str = "reflect"):
    """Test wavelet analysis and synthesis on lorenz signal."""
    lorenz = np.transpose(
        np.expand_dims(generate_lorenz(tmax=1.27)[:, 0], -1), [1, 0]
    ).astype(np.float64)
    data = np.expand_dims(lorenz, 0)
    coeff = wavedec(data, wavelet, mode=mode)
    pywt_coeff = pywt.wavedec(lorenz, wavelet, mode=mode)
    jwt_cat_coeff = np.concatenate(coeff, axis=-1).squeeze()
    pywt_cat_coeff = np.concatenate(pywt_coeff, axis=-1).squeeze()
    err = np.mean(np.abs(jwt_cat_coeff - pywt_cat_coeff))
    print(
        "wavelet: {}, mode: {},    coefficient-error: {:2.2e}".format(
            wavelet.name, mode, err
        )
    )
    assert np.allclose(jwt_cat_coeff, pywt_cat_coeff)
    rec_data = waverec(coeff, wavelet)
    err = np.mean(np.abs(rec_data - data))
    print(
        "wavelet: {}, mode: {}, reconstruction-error: {:2.2e}".format(
            wavelet.name, mode, err
        )
    )
    assert np.allclose(rec_data, data)


def test_conv_fwt():
    """Run all tests."""
    for wavelet_str in ("haar", "db2", "sym4"):
        for boundary in ["reflect", "symmetric"]:
            wavelet = pywt.Wavelet(wavelet_str)
            fwt_ifwt_lorenz(wavelet, mode=boundary)


if __name__ == "__main__":
    test_haar_fwt_ifwt_16_float32()
    test_conv_fwt()
