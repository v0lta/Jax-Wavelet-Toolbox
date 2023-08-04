"""Test the stationary wavelet transformation code."""

import jax
import jax.numpy as jnp
import pytest
import pywt
from jax.config import config

from src.jaxwt.stationary_transform_2d import swt2

config.update("jax_enable_x64", True)


@pytest.mark.slow
@pytest.mark.parametrize("size", [[1, 32, 32], [3, 2, 32, 32], [3, 2, 1, 32, 32]])
@pytest.mark.parametrize("wavelet", ["db1", "db2", "db3", "sym4"])
@pytest.mark.parametrize("level", [1, 2, None])
def test_swt_2d(level, size, wavelet):
    """Test the 1d swt."""
    key = jax.random.PRNGKey(42)
    signal = jax.random.randint(key, size, 0, 9).astype(jnp.float64)
    ptwt_coeff = swt2(signal, wavelet, level=level)
    pywt_coeff = pywt.swt2(signal, wavelet, level, trim_approx=True, norm=False)
    test_list = []
    for a, b in zip(ptwt_coeff, pywt_coeff):
        if isinstance(a, jnp.ndarray):
            test_list.append(jnp.allclose(a, b))
        else:
            test_list.extend([jnp.allclose(ael, bel) for ael, bel in zip(a, b)])
    assert all(test_list)

    #rec = iswt2(ptwt_coeff, wavelet)
    #assert jnp.allclose(rec, signal)