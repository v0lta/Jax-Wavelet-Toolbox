"""Test 3d transform support."""

#
# Created on Fri Aug 04 2023
# Copyright (c) 2023 Moritz Wolter
#

from typing import List

import jax
import jax.numpy as jnp
import pytest
import pywt

from src.jaxwt.conv_fwt_3d import wavedec3, waverec3

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def _compare_coeffs(jaxwt_coeff, pywt_coeff):
    test_list = []
    for jaxwtc, pywtc in zip(jaxwt_coeff, pywt_coeff):
        if isinstance(jaxwtc, jnp.ndarray):
            test_list.append(jnp.allclose(jaxwtc, pywtc))
        else:
            test_list.extend(
                tuple(
                    jnp.allclose(jaxwtc[key], pywtce) for key, pywtce in jaxwtc.items()
                )
            )
    return test_list


@pytest.mark.parametrize(
    "size", [[5, 32, 32, 32], [4, 3, 32, 32, 32], [1, 1, 1, 32, 32, 32]]
)
@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("wavelet", ["haar", "sym3", "db4"])
@pytest.mark.parametrize("mode", ["zero", "symmetric", "reflect"])
def test_multidim_input(size: List[int], level: int, wavelet: str, mode: str):
    """Ensure correct folding of multidimensional inputs."""
    key = jax.random.PRNGKey(42)
    data = jax.random.uniform(key, size).astype(jnp.float64)

    jaxwt_coeff = wavedec3(data, wavelet, level=level, mode=mode)
    pywt_coeff = pywt.wavedecn(data, wavelet, level=level, mode=mode, axes=[-3, -2, -1])
    test_list = _compare_coeffs(jaxwt_coeff, pywt_coeff)
    assert all(test_list)

    rec = waverec3(jaxwt_coeff, wavelet)

    assert jnp.allclose(data, rec)


@pytest.mark.parametrize("axes", [[0, 2, 1], [-3, -2, -1]])
def test_axes_arg(axes):
    """Test axes argument support."""
    key = jax.random.PRNGKey(41)
    data = jax.random.uniform(key, [32, 32, 32, 32, 32]).astype(jnp.float64)
    jaxwt_coeff = wavedec3(data, "db3", level=2, axes=axes)
    pywt_coeff = pywt.wavedecn(data, "db3", level=2, axes=axes)
    test_list = _compare_coeffs(jaxwt_coeff, pywt_coeff)
    assert all(test_list)

    rec = waverec3(jaxwt_coeff, "db3", axes=axes)
    assert jnp.allclose(data, rec)


def test_axis_error_axes_count():
    """Check the error for too many axes."""
    with pytest.raises(ValueError):
        data = jax.random.uniform(jax.random.PRNGKey(42), [32, 32, 32, 32])
        wavedec3(data, "haar", 1, axes=(1, 2, 3, 4))


def test_axis_error_axes_rep():
    """Check the error for axes repetition."""
    with pytest.raises(ValueError):
        data = jax.random.uniform(jax.random.PRNGKey(42), [32, 32, 32, 32])
        wavedec3(data, "haar", 1, axes=(1, 2, 2))


def test_broken_input():
    """Check the error for too many axes."""
    with pytest.raises(ValueError):
        data = jax.random.uniform(jax.random.PRNGKey(42), [32, 32])
        wavedec3(data, "haar", 1)
