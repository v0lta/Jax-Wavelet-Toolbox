from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pywt
from jax.config import config

from src.jaxwt.conv_fwt_3d import wavedec3, waverec3

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


@pytest.mark.parametrize(
    "size", [[5, 32, 32, 32], [4, 3, 32, 32, 32], [1, 1, 1, 32, 32, 32]]
)
@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("axes", [[-3, -2, -1]])
def test_multidim_input(size: List[int], axes: List[int], level: int):
    """Ensure correct folding of multidimensional inputs."""
    key = jax.random.PRNGKey(42)
    data = jax.random.uniform(key, size).astype(jnp.float64)

    jaxwt_coeff = wavedec3(data, "db2", level=level)
    pywt_coeff = pywt.wavedecn(data, "db2", level=level, axes=axes)

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
    assert all(test_list)

    rec = waverec3(jaxwt_coeff, "db2")

    assert jnp.allclose(data, rec)
