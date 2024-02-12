"""Test the wavelet packet code."""

#
# Created on Fri Jun 19 2020
# Copyright (c) 2020 Moritz Wolter
#

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pywt

from src.jaxwt.packets import WaveletPacket, WaveletPacket2D

jax.config.update("jax_enable_x64", True)


# TODO: add more input shapes. and level none
@pytest.mark.parametrize("input_shape", ((2, 256), (3, 255), (1, 244)))
@pytest.mark.parametrize("wavelet", ("haar", "db2", "db3"))
@pytest.mark.parametrize("level", (2, 4))
@pytest.mark.parametrize("mode", ("reflect", "symmetric", "zero"))
def test_packets_1d(input_shape, wavelet, level, mode):
    """Test the wavelet packet analysis code."""
    wavelet = pywt.Wavelet(wavelet)
    input_data = jnp.array(np.random.randn(*input_shape))
    jwp = WaveletPacket(input_data, wavelet, mode=mode)
    nodes = jwp.get_level(level)
    jnp_lst = []
    for node in nodes:
        jnp_lst.append(jwp[node])
    jres = jnp.stack(jnp_lst)

    np_input_data = np.array(input_data._value)
    wp = pywt.WaveletPacket(np_input_data, wavelet=wavelet, mode=mode)
    nodes = [node.path for node in wp.get_level(level, "freq")]
    np_lst = []
    for node in nodes:
        np_lst.append(wp[node].data)
    res = jnp.stack(np_lst)
    assert jnp.allclose(jres, res)


@pytest.mark.parametrize("input_shape", ((1, 32), (3, 255), (1, 244)))
@pytest.mark.parametrize("base_key", ["a", "d"])
@pytest.mark.parametrize("wavelet", ("haar", "db2", "db3"))
@pytest.mark.parametrize("level", (1, 2, 4))
@pytest.mark.parametrize("mode", ("reflect", "zero"))
def test_inverse_packets_1d(input_shape, wavelet, level, mode, base_key):
    """Test 1d packet inversion."""
    wavelet = pywt.Wavelet(wavelet)
    input_data = jnp.array(np.random.randn(*input_shape))
    jwp = WaveletPacket(input_data, wavelet, mode=mode, max_level=level)
    np_input_data = np.array(input_data._value)
    wp = pywt.WaveletPacket(np_input_data, wavelet=wavelet, mode=mode, maxlevel=level)

    wp[base_key * level].data *= 0
    jwp[base_key * level] *= 0

    wp.reconstruct(update=True)
    jwp.reconstruct()

    assert jnp.allclose(wp[""].data, jwp[""][:, : input_shape[-1]])


@pytest.mark.parametrize(
    "input_shape", ((2, 32, 32), (3, 33, 33), (1, 32, 33), (3, 2, 32, 32))
)
@pytest.mark.parametrize("wavelet", ("haar", "db2"))
@pytest.mark.parametrize("level", (2, 3))
@pytest.mark.parametrize("mode", ("reflect", "symmetric", "zero"))
def test_packets_2d(input_shape, wavelet, level, mode):
    """Test the wavelet packet analysis code."""
    wavelet = pywt.Wavelet(wavelet)
    input_data = jnp.array(np.random.randn(*input_shape))
    jwp = WaveletPacket2D(input_data, wavelet, mode=mode)
    jnp_list = []
    wp_keys = list(product(["a", "h", "v", "d"], repeat=level))
    for node in wp_keys:
        jnp_list.append(jwp["".join(node)])
    jres = jnp.stack(jnp_list)

    np_input_data = np.array(input_data._value)
    wp = pywt.WaveletPacket2D(np_input_data, wavelet=wavelet, mode=mode)
    np_lst = []
    for node in wp_keys:
        np_lst.append(wp["".join(node)].data)
    res = jnp.stack(np_lst)
    assert jnp.allclose(jres, res)


@pytest.mark.parametrize("level", [1, 3])
@pytest.mark.parametrize("base_key", ["a", "h", "d"])
@pytest.mark.parametrize("size", [(1, 32, 32), (2, 31, 64)])
@pytest.mark.parametrize("wavelet", ["db1", "db2", "sym4"])
def test_inverse_packet_2d(level, base_key, size, wavelet):
    """Test the 2d reconstruction code."""
    signal = np.random.randn(size[0], size[1], size[2])
    mode = "reflect"
    wp = pywt.WaveletPacket2D(signal, wavelet, mode=mode, maxlevel=level)
    jaxwp = WaveletPacket2D(jnp.array(signal), wavelet, mode=mode, max_level=level)
    wp[base_key * level].data *= 0
    jaxwp[base_key * level] *= 0
    wp.reconstruct(update=True)
    jaxwp.reconstruct()
    assert jnp.allclose(wp[""].data, jaxwp[""][:, : size[1], : size[2]])
