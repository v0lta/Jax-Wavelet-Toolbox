"""Test utility code."""

#
# Copyright (c) 2023 Moritz Wolter
#

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax import random

from src.jaxwt.utils import _fold_axes, _unfold_axes


class TestFold(parameterized.TestCase):
    """Test the folding and unfolding functions."""

    @chex.variants(
        without_jit=True, with_device=True, without_device=True, with_jit=False
    )
    @parameterized.parameters([1, 2, 3])
    def test_fold(self, keep_no):
        """Run the test."""
        key = random.PRNGKey(42)
        data = jax.random.normal(key, (4, 3, 2, 5), jnp.float64)
        keep_no = jnp.array(keep_no)
        folded, ds = self.variant(_fold_axes)(data, keep_no)
        restored = self.variant(_unfold_axes)(folded, ds, keep_no)

        assert np.allclose(restored, data)
