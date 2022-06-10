"""Test the wavelet packet code."""
#
# Created on Fri Jun 19 2020
# Copyright (c) 2020 Moritz Wolter
#

import jax.numpy as np
import numpy as nnp
import pytest
import pywt

from src.jaxwt._lorenz import generate_lorenz
from src.jaxwt.packets import WaveletPacket


@pytest.mark.parametrize("wavelet", ("haar", "db2", "db3"))
@pytest.mark.parametrize("level", (2, 4))
def test_packets_lorenz(wavelet, level, mode="reflect"):
    """Test wavelet analysis and synthesis on lorenz signal."""
    wavelet = pywt.Wavelet(wavelet)
    lorenz = generate_lorenz(tmax=0.99)[:, 0].astype(np.float64)
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
    assert np.allclose(jres, res)
