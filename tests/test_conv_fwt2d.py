"""2d Convolution fast wavelet transform test code."""
import jax.numpy as np
import pytest
import pywt
import scipy
from jax.config import config

from src.jaxwt.conv_fwt_2d import wavedec2, waverec2
from src.jaxwt.utils import flatten_2d_coeff_lst

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("mode", ["symmetric", "reflect"])
@pytest.mark.parametrize("wavelet", ["haar", "db2", "db3", "sym4"])
@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("size", [(65, 65), (64, 64), (47, 45), (45, 47)])
def test_conv_2d(wavelet: str, level: int, size: tuple, mode: str):
    """Run a specific test."""
    wavelet = pywt.Wavelet(wavelet)
    face = np.transpose(scipy.misc.face(), [2, 0, 1]).astype(np.float64)
    face = face[:, 128 : (128 + size[0]), 256 : (256 + size[1])]

    coeff2d_pywt = pywt.wavedec2(face, wavelet, mode=mode, level=level)
    coeff2d = wavedec2(face, wavelet, level=level, mode=mode)
    # test pywt compatability
    pywt_flat_list = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    jwt_flat_list = np.concatenate(flatten_2d_coeff_lst(coeff2d), -1)
    errc = np.max(np.abs(pywt_flat_list - jwt_flat_list))
    assert np.allclose(pywt_flat_list, jwt_flat_list)

    # test invertability
    reconstruction_2d = waverec2(coeff2d, wavelet)[..., : size[0], : size[1]]
    err = np.max(np.abs(reconstruction_2d - face))
    print(
        "{}, {}, {}, coefficients: {:2.2e}, reconstruction, {:2.2e}".format(
            wavelet.name, str(level).center(4), size, errc, err
        )
    )
    assert np.allclose(reconstruction_2d, face)
