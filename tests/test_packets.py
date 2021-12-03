"""Test the wavelet packet code."""
#
# Created on Fri Jun 19 2020
# Copyright (c) 2020 Moritz Wolter
#

import jax.numpy as np
import numpy as nnp
import pywt
from src.jwt._lorenz import generate_lorenz
from src.jwt.packets import WaveletPacket


def run_packets_lorenz(wavelet, level=2, mode="reflect"):
    """Test wavelet analysis and synthesis on lorenz signal."""
    lorenz = generate_lorenz(tmax=0.99)[:, 0]
    jwp = WaveletPacket(lorenz, wavelet, mode=mode)
    nodes = jwp.get_level(level)
    jnp_lst = []
    for node in nodes:
        jnp_lst.append(jwp[node])
    jres = np.stack(jnp_lst)

    np_lorenz = nnp.array(lorenz._value)
    wp = pywt.WaveletPacket(np_lorenz, wavelet=wavelet, mode=mode)
    nodes = [node.path for node in wp.get_level(level, "freq")]
    np_lst = []
    for node in nodes:
        np_lst.append(wp[node].data)
    res = np.stack(np_lst)

    err = np.mean(np.abs(jres - res))
    print("wavelet: {}, level: {}, error: {:2.2e}".format(wavelet.name, level, err))
    assert np.allclose(jres, res, atol=1e-5, rtol=1e-4)


def test_packets():
    """Run all tests."""
    for wavelet_str in ("haar", "db2", "db3"):
        for level in (2, 4):
            wavelet = pywt.Wavelet(wavelet_str)
            run_packets_lorenz(wavelet, level=level)


if __name__ == "__main__":
    test_packets()
