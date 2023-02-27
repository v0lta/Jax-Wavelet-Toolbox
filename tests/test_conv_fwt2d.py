"""2d Convolution fast wavelet transform test code."""
import jax.numpy as jnp
import pytest
import pywt
import scipy.datasets
from jax.config import config

from src.jaxwt.conv_fwt_2d import wavedec2, waverec2
from src.jaxwt.utils import flatten_2d_coeff_lst

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


@pytest.mark.parametrize("mode", ["symmetric", "reflect"])
@pytest.mark.parametrize("wavelet", ["haar", "db2", "db3", "sym4"])
@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("size", [(65, 65), (64, 64), (47, 45), (45, 47)])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_conv_2d(wavelet: str, level: int, size: tuple, mode: str, dtype: jnp.dtype):
    """Run a specific test."""
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
