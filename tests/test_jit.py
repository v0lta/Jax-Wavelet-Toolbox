"""Test jit compilation."""

#
# Copyright (c) 2023 Moritz Wolter
#
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pywt

import src.jaxwt as jaxwt
from src.jaxwt.utils import create_wavelet_named_tuple
from tests._lorenz import generate_lorenz

jax.config.update("jax_enable_x64", True)


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
    wavelet = create_wavelet_named_tuple(wavelet, dtype=dtype)
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
    wavelet = create_wavelet_named_tuple(wavelet, jnp.float64)
    jit_wavedec2 = jax.jit(jaxwt.wavedec2, static_argnames=["level"])
    coeff = jit_wavedec2(data, wavelet, level=level)
    jit_waverec2 = jax.jit(jaxwt.waverec2)
    rec = jit_waverec2(coeff, wavelet=wavelet)
    assert np.allclose(rec, data)
