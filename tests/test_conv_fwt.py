"""Convolution fast wavelet transform test code."""

#
# Created on Thu Jun 11 2020
# Copyright (c) 2020 Moritz Wolter
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pywt
from jax import random
from jax.config import config

from src.jaxwt.conv_fwt import wavedec, waverec
from tests._lorenz import generate_lorenz

config.update("jax_enable_x64", True)


def test_haar_fwt_ifwt_16_float32():
    """Test Haar wavelet analysis and synthesis on 16 sample signal."""
    wavelet = pywt.Wavelet("haar")
    data = jnp.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
        ]
    ).astype(jnp.float32)
    data = jnp.expand_dims(data, 0)
    coeffs_pywt = pywt.wavedec(data, wavelet, level=2)
    coeffs_jaxwt = wavedec(data, wavelet, level=2)
    cat_coeffs_pywt = jnp.concatenate(coeffs_pywt, -1)
    cat_coeffs_jaxwt = jnp.concatenate(coeffs_jaxwt, -1)
    assert jnp.allclose(cat_coeffs_pywt, cat_coeffs_jaxwt)
    reconstructed_data = waverec(coeffs_jaxwt, wavelet)
    assert jnp.allclose(reconstructed_data, data)


@pytest.mark.parametrize("wavelet", ["haar", "db2", "db3", "sym4"])
@pytest.mark.parametrize("mode", ["reflect", "symmetric", "zero"])
@pytest.mark.parametrize("tmax", [1.27, 1.26])
@pytest.mark.parametrize("level", [1, 2, None])
def test_fwt_ifwt_lorenz(wavelet, level, mode, tmax):
    """Test wavelet analysis and synthesis on lorenz signal."""
    wavelet = pywt.Wavelet(wavelet)
    lorenz = jnp.transpose(
        jnp.expand_dims(generate_lorenz(tmax=tmax)[:, 0], -1), [1, 0]
    ).astype(jnp.float64)
    coeff = wavedec(lorenz, wavelet, mode=mode, level=level)
    pywt_coeff = pywt.wavedec(lorenz, wavelet, mode=mode, level=level)
    jwt_cat_coeff = jnp.concatenate(coeff, axis=-1).squeeze()
    pywt_cat_coeff = jnp.concatenate(pywt_coeff, axis=-1).squeeze()
    assert jnp.allclose(jwt_cat_coeff, pywt_cat_coeff)
    rec_data = waverec(coeff, wavelet)
    assert jnp.allclose(rec_data[..., : lorenz.shape[-1]], lorenz)


@pytest.mark.parametrize("wavelet", ["db2", "sym4"])
@pytest.mark.parametrize("mode", ["symmetric", "reflect", "zero"])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("level", [2, None])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_batch_fwt_ifwt(wavelet, mode, batch_size, level, dtype: jnp.dtype):
    """Test the batched version of the fwt."""
    if dtype == jnp.float32:
        atol = 1e-4
    else:
        atol = 1e-8

    wavelet = pywt.Wavelet(wavelet)
    random_dat = jnp.array(np.random.randn(batch_size, 100)).astype(dtype)
    coeff = wavedec(random_dat, wavelet, mode=mode, level=level)
    rec_data = waverec(coeff, wavelet)
    assert jnp.allclose(rec_data[..., : random_dat.shape[-1]], random_dat, atol=atol)


@pytest.mark.parametrize("level", [1, 2, 3, None])
@pytest.mark.parametrize("shape", [(64,), (1, 64), (3, 2, 64), (4, 3, 2, 64)])
def test_multi_batch_fwt(level, shape):
    """Test 1d conv support for multiple inert batch dimensions."""
    key = random.PRNGKey(42)
    data = jax.random.normal(key, shape, jnp.float64)

    jaxwt_coeff = wavedec(data, "haar", level=level)
    pywt_coeff = pywt.wavedec(np.array(data), "haar", level=level)

    test = []
    for jaxwtc, pywtc in zip(jaxwt_coeff, pywt_coeff):
        test.append(np.allclose(jaxwtc, pywtc))
    assert all(test)

    rec = waverec(jaxwt_coeff, "haar")
    assert np.allclose(data, rec)


@pytest.mark.parametrize("axis", [-1, 0, 1, 2])
def test_axis_arg(axis):
    """Ensure the axis argument works as expected."""
    key = random.PRNGKey(42)
    data = jax.random.normal(key, [16, 16, 16], jnp.float64)

    jaxwtcs = wavedec(data, "haar", level=2, axis=axis)
    pywtcs = pywt.wavedec(data, "haar", level=2, axis=axis)
    test = []
    for jaxwtc, pywtc in zip(jaxwtcs, pywtcs):
        test.append(np.allclose(jaxwtc, pywtc))
    assert all(test)

    rec = waverec(jaxwtcs, "haar", axis=axis)
    assert np.allclose(data, rec)
