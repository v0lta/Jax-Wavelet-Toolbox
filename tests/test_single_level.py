#
# Created on Thu Jun 11 2020
# Copyright (c) 2020 Moritz Wolter
#

import pywt
import jax.numpy as np

from src.jwt.conv_fwt import dwt, idwt
from src.jwt._lorenz import generate_lorenz


def dwt_idwt_lorenz(wavelet, mode="reflect"):
    # ---- Test single level wavelet analysis and synthesis on lorenz signal. -----
    lorenz = np.transpose(np.expand_dims(generate_lorenz(tmax=0.99)[:, 0], -1), [1, 0])
    data = np.expand_dims(lorenz, 0)
    coeff = dwt(data, wavelet=wavelet, mode=mode)
    pywt_coeff = pywt.dwt(lorenz, wavelet, mode=mode)
    cat_coeff = np.concatenate(coeff, axis=-1)
    pywt_cat_coeff = np.concatenate(pywt_coeff, axis=-1)
    err = np.mean(np.abs(cat_coeff - pywt_cat_coeff))
    print("coeff error {}, {}: {}".format(wavelet.name, mode, err))
    assert np.allclose(cat_coeff, pywt_cat_coeff, atol=1e-5)
    rest_data = idwt(coeff, wavelet)
    err = np.mean(np.abs(rest_data - data))
    print("rec error   {}, {}: {}".format(wavelet.name, mode, err))
    assert np.allclose(rest_data, data)


def test():
    for wavelet_str in ["haar", "db2", "db4", "db8"]:
        for boundary in ["reflect", "symmetric"]:
            wavelet = pywt.Wavelet(wavelet_str)
            dwt_idwt_lorenz(wavelet, mode=boundary)


if __name__ == "__main__":
    test()
