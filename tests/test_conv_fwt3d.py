"""Test 3d transform support."""

#
# Created on Fri Aug 04 2023
# Copyright (c) 2023 Moritz Wolter
#

from functools import partial
from typing import List

import chex
import jax
import jax.numpy as jnp
import pytest
import pywt
from absl.testing import parameterized

from jaxwt.conv_fwt_3d import wavedec3, waverec3

jax.config.update("jax_enable_x64", True)


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


class TestConv3D(parameterized.TestCase):
    """Tests fort the two-dimensional fwt code."""

    @chex.all_variants(with_pmap=False)
    @parameterized.product(
        size=[[5, 32, 32, 32], [4, 3, 32, 32, 32], [1, 1, 1, 32, 32, 32]],
        level=[1, 2, None],
        wavelet=["haar", "sym3"],
        mode=["zero", "reflect"],
    )
    def test_multidim_input(self, size: List[int], level: int, wavelet: str, mode: str):
        """Ensure correct folding of multidimensional inputs."""
        key = jax.random.PRNGKey(42)
        data = jax.random.uniform(key, size).astype(jnp.float64)

        wavedec3_variants = self.variant(
            partial(wavedec3, wavelet=wavelet, level=level, mode=mode)
        )
        jaxwt_coeff = wavedec3_variants(data)
        pywt_coeff = pywt.wavedecn(
            data, wavelet=wavelet, level=level, mode=mode, axes=[-3, -2, -1]
        )
        test_list = _compare_coeffs(jaxwt_coeff, pywt_coeff)
        assert all(test_list)

        waverec3_variants = self.variant(partial(waverec3, wavelet=wavelet))
        rec = waverec3_variants(jaxwt_coeff)
        assert jnp.allclose(data, rec)


class TestAxesArg(parameterized.TestCase):
    """Test axes argument support."""

    @chex.all_variants(with_pmap=False, with_jit=False, without_jit=True)
    @parameterized.product(axes=[[1, 2, 3], [-3, -2, -1]])
    def test_axes_arg(self, axes):
        """Run test."""
        key = jax.random.PRNGKey(41)
        data = jax.random.uniform(key, [1, 16, 16, 16, 16]).astype(jnp.float64)

        wavedec3_variants = self.variant(
            partial(wavedec3, wavelet="db2", level=2, axes=axes)
        )
        jaxwt_coeff = wavedec3_variants(data)
        pywt_coeff = pywt.wavedecn(data, "db2", level=2, axes=axes)
        test_list = _compare_coeffs(jaxwt_coeff, pywt_coeff)
        assert all(test_list)

        waverec3_variants = self.variant(partial(waverec3, wavelet="db2", axes=axes))
        rec = waverec3_variants(jaxwt_coeff)
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
