"""Convolution fast wavelet transform test code."""

#
# Created on Thu Jun 11 2020
# Copyright (c) 2020 Moritz Wolter
#

from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pywt
from absl.testing import parameterized
from jax import random

from jaxwt.conv_fwt import wavedec, waverec
from tests._lorenz import generate_lorenz

jax.config.update("jax_enable_x64", True)


class TestHaar(parameterized.TestCase):
    """Test Haar wavelet analysis and synthesis on 16 sample signal."""

    @chex.all_variants(with_pmap=True)
    def test_haar(self):
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
        all_variants_wavedec = self.variant(
            partial(wavedec, wavelet=wavelet, level=2)
        )
        coeffs_jaxwt = all_variants_wavedec(data)
        cat_coeffs_pywt = jnp.concatenate(coeffs_pywt, -1)
        cat_coeffs_jaxwt = jnp.concatenate(coeffs_jaxwt, -1)
        assert jnp.allclose(cat_coeffs_pywt, cat_coeffs_jaxwt)
        all_variants_waverec = self.variant(partial(waverec, wavelet=wavelet))
        reconstructed_data = all_variants_waverec(coeffs_jaxwt)
        assert jnp.allclose(reconstructed_data, data)


class TestInvert(parameterized.TestCase):
    """Test 1d inversion."""

    @chex.all_variants(with_pmap=True)
    @parameterized.product(
        wavelet=["haar", "db2", "db3"],
        mode=["reflect", "symmetric", "zero"],
        tmax=[1.27, 1.26],
        level=[1, None],
    )
    def test_fwt_ifwt_lorenz(self, wavelet, level, mode, tmax):
        """Test wavelet analysis and synthesis on lorenz signal."""
        wavelet = pywt.Wavelet(wavelet)
        lorenz = jnp.transpose(
            jnp.expand_dims(generate_lorenz(tmax=tmax)[:, 0], -1), [1, 0]
        ).astype(jnp.float64)
        all_variants_wavedec = self.variant(
            partial(wavedec, wavelet=wavelet, mode=mode, level=level)
        )
        coeff = all_variants_wavedec(lorenz)
        pywt_coeff = pywt.wavedec(lorenz, wavelet, mode=mode, level=level)
        jwt_cat_coeff = jnp.concatenate(coeff, axis=-1).squeeze()
        pywt_cat_coeff = jnp.concatenate(pywt_coeff, axis=-1).squeeze()
        assert jnp.allclose(jwt_cat_coeff, pywt_cat_coeff)
        all_variants_waverec = self.variant(partial(waverec, wavelet=wavelet))
        rec_data = all_variants_waverec(coeff)
        assert jnp.allclose(rec_data[..., : lorenz.shape[-1]], lorenz)


class TestBatch(parameterized.TestCase):
    """Test the batched version of the fwt."""

    @chex.all_variants(with_pmap=True)
    @parameterized.product(
        wavelet=["db2", "sym4"],
        mode=["reflect", "symmetric"],
        batch_size=[1, 3],
        level=[2, None],
        dtype=[jnp.float64, jnp.float32],
    )
    def test_batch_fwt_ifwt(self, wavelet, mode, batch_size, level, dtype: jnp.dtype):
        """Test the batched version of the fwt."""
        if dtype == jnp.float32:
            atol = 1e-4
        else:
            atol = 1e-8

        wavelet = pywt.Wavelet(wavelet)
        random_dat = jnp.array(np.random.randn(batch_size, 100)).astype(dtype)
        variant_wavedec = self.variant(
            partial(wavedec, wavelet=wavelet, mode=mode, level=level)
        )
        coeff = variant_wavedec(random_dat)
        variant_waverec = self.variant(partial(waverec, wavelet=wavelet))
        rec_data = variant_waverec(coeff)
        assert jnp.allclose(
            rec_data[..., : random_dat.shape[-1]], random_dat, atol=atol
        )


class TestMultiBatch(parameterized.TestCase):
    """Test 1d conv support for multiple inert batch dimensions."""

    @chex.all_variants(with_pmap=True)
    @parameterized.product(
        level=[1, 2, 3, None], shape=[(64,), (1, 64), (3, 2, 64), (4, 3, 2, 64)]
    )
    def test_multi_batch_fwt(self, level, shape):
        """Run the test."""
        key = random.PRNGKey(42)
        data = jax.random.normal(key, shape, jnp.float64)

        all_variants_wavedec = self.variant(
            partial(wavedec, wavelet="haar", level=level)
        )
        jaxwt_coeff = all_variants_wavedec(data)

        pywt_coeff = pywt.wavedec(np.array(data), "haar", level=level)

        test = []
        for jaxwtc, pywtc in zip(jaxwt_coeff, pywt_coeff):
            test.append(jnp.allclose(jaxwtc, pywtc))
        assert all(test)
        all_variants_waverec = self.variant(partial(waverec, wavelet="haar"))
        rec = all_variants_waverec(jaxwt_coeff)
        assert np.allclose(data, rec)


class TestAxisArg(parameterized.TestCase):
    """Test the axis argument."""

    @chex.all_variants(with_pmap=True)
    @parameterized.product(axis=[-1, 0, 1, 2])
    def test_axis_arg(self, axis):
        """Ensure the axis argument works as expected."""
        key = random.PRNGKey(42)
        data = jax.random.normal(key, [16, 16, 16], jnp.float64)

        all_variants_wavedec = self.variant(
            partial(wavedec, wavelet="haar", level=2, axis=axis)
        )
        jaxwtcs = all_variants_wavedec(data)
        pywtcs = pywt.wavedec(data, "haar", level=2, axis=axis)
        test = []
        for jaxwtc, pywtc in zip(jaxwtcs, pywtcs):
            test.append(jnp.allclose(jaxwtc, pywtc))
        assert all(test)

        all_variants_waverec = self.variant(partial(waverec, wavelet="haar", axis=axis))
        rec = all_variants_waverec(jaxwtcs)
        assert np.allclose(data, rec)
