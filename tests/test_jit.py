"""Test jit compilation."""
#
# Copyright (c) 2023 Moritz Wolter
#
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pywt

import src.jaxwt as jaxwt
from tests._lorenz import generate_lorenz

WaveletTuple = namedtuple("Wavelet", ["dec_lo", "dec_hi", "rec_lo", "rec_hi"])


def _to_wavelet_tuple(wavelet: pywt.Wavelet) -> WaveletTuple:
    return WaveletTuple(
        jnp.array(wavelet.dec_lo),
        jnp.array(wavelet.dec_hi),
        jnp.array(wavelet.rec_lo),
        jnp.array(wavelet.rec_hi),
    )


@pytest.mark.slow
@pytest.mark.parametrize("wavelet_string", ["db1", "db4", "sym5"])
@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("length", [64, 65])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_conv_fwt_jit(wavelet_string, level, length, batch_size, dtype):
    """Test jitting a convolution fwt, for various levels and padding options."""
    data = generate_lorenz().transpose()[:batch_size, :length].astype(dtype)

    wavelet = pywt.Wavelet(wavelet_string)
    # pywt.Wavelets do not compile with jax.jit
    wavelet = _to_wavelet_tuple(wavelet)
    jit_wavedec = jax.jit(jaxwt.wavedec, static_argnames=["level"])
    coeff = jit_wavedec(data, wavelet, level=level)
    jit_waverec = jax.jit(jaxwt.waverec)
    res = jit_waverec(coeff, wavelet)
    assert jnp.allclose(data, res[:, : data.shape[-1]])


@pytest.mark.parametrize("level", [1, 2])
def test_conv_fwt_jit_2d(level):
    """Test the jit compilation feature for the wavedec2 function."""
    data = jnp.array(np.random.randn(10, 64, 64)).astype(jnp.float64)
    wavelet = pywt.Wavelet("db2")
    wavelet = _to_wavelet_tuple(wavelet)
    jit_wavedec2 = jax.jit(jaxwt.wavedec2, static_argnames=["level"])
    coeff = jit_wavedec2(data, wavelet, level=level)
    jit_waverec2 = jax.jit(jaxwt.waverec2)
    rec = jit_waverec2(coeff, wavelet=wavelet)
    assert np.allclose(rec, data)
