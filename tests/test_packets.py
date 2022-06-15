"""Test the wavelet packet code."""
#
# Created on Fri Jun 19 2020
# Copyright (c) 2020 Moritz Wolter
#

from itertools import product

import jax.numpy as np
import numpy as nnp
import pytest
import pywt

from src.jaxwt.packets import WaveletPacket, WaveletPacket2D


# TODO: add more input shapes. and level none
@pytest.mark.parametrize("input_shape", ((2, 256), (3, 255), (1,244)))
@pytest.mark.parametrize("wavelet", ("haar", "db2", "db3"))
@pytest.mark.parametrize("level", (2, 4))
@pytest.mark.parametrize("mode", ("reflect", "symmetric", "zero"))
def test_packets_1d(input_shape, wavelet, level, mode):
    """Test the wavelet packet analysis code."""
    wavelet = pywt.Wavelet(wavelet)
    input_data = np.array(nnp.random.randn(*input_shape))
    jwp = WaveletPacket(input_data, wavelet, mode=mode)
    nodes = jwp.get_level(level)
    jnp_lst = []
    for node in nodes:
        jnp_lst.append(jwp[node])
    jres = np.stack(jnp_lst)

    np_input_data = nnp.array(input_data._value)
    wp = pywt.WaveletPacket(np_input_data, wavelet=wavelet, mode=mode)
    nodes = [node.path for node in wp.get_level(level, "freq")]
    np_lst = []
    for node in nodes:
        np_lst.append(wp[node].data)
    res = np.stack(np_lst)
    assert np.allclose(jres, res)


@pytest.mark.parametrize("input_shape", ((2, 32, 32), (3, 33, 33), (1, 32, 33)))
@pytest.mark.parametrize("wavelet", ("haar", "db2"))
@pytest.mark.parametrize("level", (2, 3))
@pytest.mark.parametrize("mode", ("reflect", "symmetric", "zero"))
def test_packets_2d(input_shape, wavelet, level, mode):
    """Test the wavelet packet analysis code."""
    wavelet = pywt.Wavelet(wavelet)
    input_data = np.array(nnp.random.randn(*input_shape))
    jwp = WaveletPacket2D(input_data, wavelet, mode=mode)
    jnp_list = []
    wp_keys = list(product(["a", "h", "v", "d"], repeat=level))
    for node in wp_keys:
        jnp_list.append(jwp["".join(node)])
    jres = np.stack(jnp_list)

    np_input_data = nnp.array(input_data._value)
    wp = pywt.WaveletPacket2D(np_input_data, wavelet=wavelet, mode=mode)
    np_lst = []
    for node in wp_keys:
        np_lst.append(wp["".join(node)].data)
    res = np.stack(np_lst)
    assert np.allclose(jres, res)
