"""2d Convolution fast wavelet transform test code."""
#
# Copyright (c) 2023 Moritz Wolter
#

from typing import List

import jax
import jax.numpy as jnp
import pytest
import pywt
import scipy.datasets
from jax.config import config

from src.jaxwt.conv_fwt_2d import wavedec2, waverec2
from src.jaxwt.utils import flatten_2d_coeff_lst

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


@pytest.mark.parametrize("mode", ["symmetric", "reflect", "zero"])
@pytest.mark.parametrize("wavelet", ["haar", "db2", "db3", "sym4"])
@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("size", [(65, 65), (64, 64), (47, 45), (45, 47)])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_conv_2d(wavelet: str, level: int, size: tuple, mode: str, dtype: jnp.dtype):
    """Runs tests of the two-dimensional fwt code."""
    if dtype == jnp.float32:
        atol = 1e-3
    else:
        atol = 1e-8

    wavelet = pywt.Wavelet(wavelet)
    face = jnp.transpose(scipy.datasets.face(), [2, 0, 1]).astype(dtype)
    face = face[:, 128 : (128 + size[0]), 256 : (256 + size[1])]

    coeff2d_pywt = pywt.wavedec2(face, wavelet, mode=mode, level=level)
    coeff2d = wavedec2(face, wavelet, level=level, mode=mode)
    # test pywt compatability
    pywt_flat_list = jnp.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    jwt_flat_list = jnp.concatenate(flatten_2d_coeff_lst(coeff2d), -1)
    assert jnp.allclose(pywt_flat_list, jwt_flat_list, atol=atol)

    # test invertability
    reconstruction_2d = waverec2(coeff2d, wavelet)[..., : size[0], : size[1]]
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


@pytest.mark.parametrize("size", [[5, 4, 64, 64], [4, 3, 2, 32, 32], [1, 1, 1, 16, 16]])
def test_multidim_input(size: List[int]):
    """Ensure correct folding of multidimensional inputs."""
    key = jax.random.PRNGKey(42)
    data = jax.random.uniform(key, size).astype(jnp.float64)

    jaxwt_coeff = wavedec2(data, "db2", level=3)
    pywt_coeff = pywt.wavedec2(data, "db2", level=3)

    test_list = _compare_coeffs(jaxwt_coeff, pywt_coeff)
    assert all(test_list)
    rec = waverec2(jaxwt_coeff, "db2")
    assert jnp.allclose(data, rec)


@pytest.mark.parametrize("axes", [(-2, -1), (-1, -2), (-3, -2), (0, 1), (1, 0)])
def test_axis_argument(axes):
    """Ensure the axes argument works as expected."""
    key = jax.random.PRNGKey(42)
    data = jax.random.uniform(key, [32, 32, 32, 32]).astype(jnp.float64)

    jaxwt_coeff = wavedec2(data, "db2", level=3, axes=axes)
    pywt_coeff = pywt.wavedec2(data, "db2", level=3, axes=axes)
    test_list = _compare_coeffs(jaxwt_coeff, pywt_coeff)
    assert all(test_list)

    rec = waverec2(jaxwt_coeff, "db2", axes=axes)
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
