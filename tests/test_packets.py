#
# Created on Fri Jun 19 2020
# Copyright (c) 2020 Moritz Wolter
#

import jax.numpy as np
import numpy as nnp
import pywt
from jaxlets.utils import Wavelet
from jaxlets.lorenz import generate_lorenz
from jaxlets.packets import WaveletPacket


def packets_lorenz(wavelet, level=2, mode='reflect'):
    # ---- Test wavelet analysis and synthesis on lorenz signal. -----
    lorenz = generate_lorenz()[:, 0]
    jwp = WaveletPacket(lorenz, wavelet, mode=mode)
    nodes = jwp.get_level(level)
    jnp_lst = []
    for node in nodes:
        jnp_lst.append(jwp[node])
    jres = np.stack(jnp_lst)

    np_lorenz = nnp.array(lorenz._value)
    wp = pywt.WaveletPacket(np_lorenz, wavelet=wavelet,
                            mode=mode)
    nodes = [node.path for node in wp.get_level(level, 'freq')]
    np_lst = []
    for node in nodes:
        np_lst.append(wp[node].data)
    res = np.stack(np_lst)

    err =  np.mean(np.abs(jres - res))
    assert err < 1e-4


def test_haar():
    wavelet = pywt.Wavelet('haar')
    packets_lorenz(wavelet, level=2)

def test_db2():
    wavelet = pywt.Wavelet('db2')
    packets_lorenz(wavelet, level=2)

def test_db4():
    wavelet = pywt.Wavelet('db4')
    packets_lorenz(wavelet, level=5)

