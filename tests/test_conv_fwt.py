"""Convolution fast wavelet transform test code."""

#
# Created on Thu Jun 11 2020
# Copyright (c) 2020 Moritz Wolter
#

import pytest
import jax.numpy as np
import pywt
from jax.config import config
from src.jaxwt._lorenz import generate_lorenz
from src.jaxwt.conv_fwt import wavedec, waverec

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
    coeffs_pywt = pywt.wavedec(data, wavelet, level=2)
    coeffs_jaxwt = wavedec(data, wavelet, level=2)
    cat_coeffs_pywt = np.concatenate(coeffs_pywt, -1)
    cat_coeffs_jaxwt = np.concatenate(coeffs_jaxwt, -1)
    assert np.allclose(cat_coeffs_pywt, cat_coeffs_jaxwt)
    reconstructed_data = waverec(coeffs_jaxwt, wavelet)
    assert np.allclose(reconstructed_data, data)


@pytest.mark.parametrize("wavelet", ["haar", "db2", "db3", "sym4"])
@pytest.mark.parametrize("mode", ["reflect", "symmetric"])
@pytest.mark.parametrize("level", [1, 2, None])
def test_fwt_ifwt_lorenz(wavelet, level, mode):
    """Test wavelet analysis and synthesis on lorenz signal."""
    wavelet = pywt.Wavelet(wavelet)
    lorenz = np.transpose(
        np.expand_dims(generate_lorenz(tmax=1.27)[:, 0], -1), [1, 0]
    ).astype(np.float64)
    data = np.expand_dims(lorenz, 0)
    coeff = wavedec(data, wavelet, mode=mode, level=level)
    pywt_coeff = pywt.wavedec(lorenz, wavelet, mode=mode, level=level)
    jwt_cat_coeff = np.concatenate(coeff, axis=-1).squeeze()
    pywt_cat_coeff = np.concatenate(pywt_coeff, axis=-1).squeeze()
    assert np.allclose(jwt_cat_coeff, pywt_cat_coeff)
    rec_data = waverec(coeff, wavelet)
    assert np.allclose(rec_data, data)
