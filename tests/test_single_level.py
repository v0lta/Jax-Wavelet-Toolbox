"""Test the single fwt and ifwt code."""
# Created on Thu Jun 11 2020
# Copyright (c) 2020 Moritz Wolter
#

import jax.numpy as np
import pywt
from jax.config import config
from src.jaxwt._lorenz import generate_lorenz
from src.jaxwt.conv_fwt import dwt, idwt

config.update("jax_enable_x64", True)


def dwt_idwt_lorenz(wavelet, mode="reflect"):
    """Test single level wavelet analysis and synthesis on lorenz signal."""
    lorenz = np.transpose(
        np.expand_dims(generate_lorenz(tmax=0.99)[:, 0], -1), [1, 0]
    ).astype(np.float64)
    data = np.expand_dims(lorenz, 0)
    coeff = dwt(data, wavelet=wavelet, mode=mode)
    pywt_coeff = pywt.dwt(lorenz, wavelet, mode=mode)
    cat_coeff = np.concatenate(coeff, axis=-1)
    pywt_cat_coeff = np.concatenate(pywt_coeff, axis=-1)
    err = np.mean(np.abs(cat_coeff - pywt_cat_coeff))
    print("coeff error {}, {}: {}".format(wavelet.name, mode, err))
    assert np.allclose(cat_coeff, pywt_cat_coeff)
    rest_data = idwt(coeff, wavelet)
    err = np.mean(np.abs(rest_data - data))
    print("rec error   {}, {}: {}".format(wavelet.name, mode, err))
    assert np.allclose(rest_data, data)


def test_run_all():
    """Run all tests."""
    for wavelet_str in ["haar", "db2", "db4", "db8"]:
        for boundary in ["reflect", "symmetric"]:
            wavelet = pywt.Wavelet(wavelet_str)
            dwt_idwt_lorenz(wavelet, mode=boundary)


if __name__ == "__main__":
    test_run_all()
