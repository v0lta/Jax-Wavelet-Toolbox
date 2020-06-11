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

def test_haar_fwt_ifwt_lorenz():
    wavelet = pywt.Wavelet('haar')
    # ---- Test harr wavelet analysis and synthesis on lorenz signal. -----
    lorenz = np.transpose(np.expand_dims(generate_lorenz()[:, 0], -1), [1, 0])
    data = np.expand_dims(lorenz, 0)

    coeff = analysis_fwt(data, wavelet)
    cat_coeff = np.concatenate(coeff, axis=-1)
    pywt_coeff = pywt.wavedec(lorenz, wavelet, mode='reflect')
    pywt_cat_coeff = np.concatenate(pywt_coeff, axis=-1)
    err = np.mean(np.abs(cat_coeff - pywt_cat_coeff))
    assert err < 1e-5

    rest_data = synthesis_fwt(coeff, wavelet)
    err = np.mean(np.abs(rest_data - data))
    assert err < 1e-5