"""2d Convolution fast wavelet transform test code."""

#
# Copyright (c) 2023 Moritz Wolter
#

from functools import partial
from typing import List

import chex
import jax
import jax.numpy as jnp
import pytest
import pywt
import scipy.datasets
from absl.testing import parameterized

from jaxwt.conv_fwt_2d import wavedec2, waverec2
from jaxwt.utils import flatten_2d_coeff_lst

jax.config.update("jax_enable_x64", True)


class TestConv2D(parameterized.TestCase):
    """Tests fort the two-dimensional fwt code."""

    @chex.all_variants(with_pmap=False)
    @parameterized.product(
        mode=["symmetric", "zero"],
        wavelet=["haar", "db3", "sym4"],
        level=[1, 2, None],
        size=[(64, 64), (65, 65), (47, 45), (45, 47)],
        dtype=[jnp.float64, jnp.float32],
    )
    def test_conv_2d(
        self, wavelet: str, level: int, size: tuple, mode: str, dtype: jnp.dtype
    ):
        """Test the 2d convolution based fwt."""
        if dtype == jnp.float32:
            atol = 1e-3
        else:
            atol = 1e-8

        wavelet = pywt.Wavelet(wavelet)
        face = jnp.transpose(scipy.datasets.face(), [2, 0, 1]).astype(dtype)
        face = face[:, 128 : (128 + size[0]), 256 : (256 + size[1])]

        coeff2d_pywt = pywt.wavedec2(face, wavelet, mode=mode, level=level)
        all_variant_wavedec2 = self.variant(
            partial(wavedec2, wavelet=wavelet, level=level, mode=mode)
        )
        coeff2d = all_variant_wavedec2(face)
        # test pywt compatability
        pywt_flat_list = jnp.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
        jwt_flat_list = jnp.concatenate(flatten_2d_coeff_lst(coeff2d), -1)
        assert jnp.allclose(pywt_flat_list, jwt_flat_list, atol=atol)

        # test invertability
        all_variant_waverec2 = self.variant(partial(waverec2, wavelet=wavelet))
        reconstruction_2d = all_variant_waverec2(coeff2d)[..., : size[0], : size[1]]
        assert jnp.allclose(reconstruction_2d, face, atol=atol)


def _compare_coeffs(jaxwt_coeff, pywt_coeff):
    test_list = []
    for jaxwtc, pywtc in zip(jaxwt_coeff, pywt_coeff):
        if isinstance(jaxwtc, jnp.ndarray):
            test_list.append(jnp.allclose(jaxwtc, pywtc))
        else:
            test_list.extend(
                tuple(
                    jnp.allclose(jaxwtce, pywtce)
                    for jaxwtce, pywtce in zip(jaxwtc, pywtc)
                )
            )
    return test_list


class TestMultiDimInput(parameterized.TestCase):
    """Test the multi-dimensional input handling."""

    @chex.all_variants(with_pmap=False)
    @parameterized.product(size=[[5, 4, 64, 64], [4, 3, 2, 32, 32], [1, 1, 1, 16, 16]])
    def test_multidim_input(self, size: List[int]):
        """Run the test."""
        key = jax.random.PRNGKey(42)
        data = jax.random.uniform(key, size).astype(jnp.float64)

        all_variants_wavedec2 = self.variant(partial(wavedec2, wavelet="db2", level=3))
        jaxwt_coeff = all_variants_wavedec2(data)
        pywt_coeff = pywt.wavedec2(data, "db2", level=3)

        test_list = _compare_coeffs(jaxwt_coeff, pywt_coeff)
        assert all(test_list)

        all_variants_waverec2 = self.variant(partial(waverec2, wavelet="db2"))
        rec = all_variants_waverec2(jaxwt_coeff)
        assert jnp.allclose(data, rec)


class TestAxisArgument(parameterized.TestCase):
    """Test the axis argument. Does not work with jit."""

    @chex.all_variants(with_pmap=False, with_jit=False, without_jit=True)
    @parameterized.product(axes=[(-2, -1), (-1, -2), (-3, -2), (0, 1), (1, 0)])
    def test_axis_argument(self, axes):
        """Ensure the axes argument works as expected."""
        key = jax.random.PRNGKey(42)
        data = jax.random.uniform(key, [32, 32, 32, 32]).astype(jnp.float64)
        axes = jnp.array(axes)

        all_variants_wavedec2 = self.variant(
            partial(wavedec2, wavelet="db2", level=3, axes=axes)
        )
        jaxwt_coeff = all_variants_wavedec2(data)
        pywt_coeff = pywt.wavedec2(data, "db2", level=3, axes=axes)
        test_list = _compare_coeffs(jaxwt_coeff, pywt_coeff)
        assert all(test_list)

        all_variants_waverec2 = self.variant(
            partial(waverec2, wavelet="db2", axes=axes)
        )
        rec = all_variants_waverec2(jaxwt_coeff)
        assert jnp.allclose(data, rec)


def test_axis_error_axes_count():
    """Check the error for too many axes."""
    with pytest.raises(ValueError):
        data = jax.random.uniform(jax.random.PRNGKey(42), [32, 32, 32, 32])
        wavedec2(data, "haar", 1, axes=(1, 2, 3))


def test_axis_error_axes_rep():
    """Check the error for axes repetition."""
    with pytest.raises(ValueError):
        data = jax.random.uniform(jax.random.PRNGKey(42), [32, 32, 32, 32])
        wavedec2(data, "haar", 1, axes=(2, 2))
