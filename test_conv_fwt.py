#
# Created on Thu Jun 11 2020
# Copyright (c) 2020 Moritz Wolter
#


import jax
import jax.numpy as np
import pywt
from lorenz import generate_lorenz
from conv_fwt import analysis_fwt, synthesis_fwt

def test_haar_fwt_ifwt_16():
    # ---- Test harr wavelet analysis and synthesis on 16 sample signal. -----
    wavelet = pywt.Wavelet('haar')
    data = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.])
    data = np.expand_dims(np.expand_dims(data, 0), 0)
    coeffs = pywt.wavedec(data, wavelet, level=2)
    coeffs2 = analysis_fwt(data, wavelet, scales=2)
    cat_coeffs = np.concatenate(coeffs, -1)
    cat_coeffs2 = np.concatenate(coeffs2, -1)
    err = np.mean(np.abs(cat_coeffs - cat_coeffs2))
    assert err < 1e-5
    rest_data = synthesis_fwt(coeffs2, wavelet, scales=2)
    err = np.mean(np.abs(rest_data - data))
    assert err < 1e-5

def fwt_ifwt_lorenz(wavelet):
    # ---- Test wavelet analysis and synthesis on lorenz signal. -----
    lorenz = np.transpose(np.expand_dims(generate_lorenz()[:, 0], -1), [1, 0])
    data = np.expand_dims(lorenz, 0)
    coeff = analysis_fwt(data, wavelet)
    pywt_coeff = pywt.wavedec(lorenz, wavelet, mode='reflect')
    cat_coeff = np.concatenate(coeff, axis=-1)
    pywt_cat_coeff = np.concatenate(pywt_coeff, axis=-1)
    err = np.mean(np.abs(cat_coeff - pywt_cat_coeff))
    assert err < 1e-5
    rest_data = synthesis_fwt(coeff, wavelet)
    err = np.mean(np.abs(rest_data - data))
    assert err < 1e-5

def test_haar_fwt_ifwt_lorenz():
    # ---- Test haar wavelet analysis and synthesis on lorenz signal. -----
    wavelet = pywt.Wavelet('haar')
    fwt_ifwt_lorenz(wavelet)

def test_db3_fwt_ifwt_lorenz():
    # ---- Test db3 wavelet analysis and synthesis on lorenz signal. -----
    wavelet = pywt.Wavelet('db3')
    fwt_ifwt_lorenz(wavelet)

def test_db4_fwt_ifwt_lorenz():
    # ---- Test db4 wavelet analysis and synthesis on lorenz signal. -----
    wavelet = pywt.Wavelet('db4')
    fwt_ifwt_lorenz(wavelet)

def test_db8_fwt_ifwt_lorenz():
    # ---- Test db8 wavelet analysis and synthesis on lorenz signal. -----
    wavelet = pywt.Wavelet('db8')
    fwt_ifwt_lorenz(wavelet)

