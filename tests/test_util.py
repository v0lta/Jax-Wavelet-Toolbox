"""Test utility code."""

#
# Copyright (c) 2023 Moritz Wolter
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from src.jaxwt.utils import _fold_axes, _unfold_axes


@pytest.mark.parametrize("keep_no", [1, 2, 3])
def test_fold(keep_no):
    """Try the folding functions."""
    key = random.PRNGKey(42)
    data = jax.random.normal(key, (4, 3, 2, 5), jnp.float64)

    folded, ds = _fold_axes(data, keep_no)
    restored = _unfold_axes(folded, ds, keep_no)

    assert np.allclose(restored, data)
