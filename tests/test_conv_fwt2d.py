"""2d Convolution fast wavelet transform test code."""
import jax.numpy as np
import pywt
import scipy
from jax.config import config
from src.jwt.conv_fwt_2d import wavedec2, waverec2
from src.jwt.utils import flatten_2d_coeff_lst

config.update("jax_enable_x64", True)


def run_2dtest(wavelet: str, level: int, size: tuple):
    """Run a specific test."""
    wavelet = pywt.Wavelet(wavelet)
    face = np.transpose(scipy.misc.face(), [2, 0, 1]).astype(np.float64)
    face = face[:, 128 : (128 + size[0]), 256 : (256 + size[1])]
    face_extended = np.expand_dims(np.array(face), 1)

    coeff2d_pywt = pywt.wavedec2(face, wavelet, mode="reflect", level=level)
    coeff2d = wavedec2(face_extended, wavelet, level=level)
    # test pywt compatability
    pywt_flat_list = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    jwt_flat_list = np.concatenate(flatten_2d_coeff_lst(coeff2d), -1)
    errc = np.max(np.abs(pywt_flat_list - jwt_flat_list))
    assert np.allclose(pywt_flat_list, jwt_flat_list)

    # test invertability
    reconstruction_2d = waverec2(coeff2d, wavelet)[..., : size[0], : size[1]]
    err = np.max(np.abs(reconstruction_2d - face_extended))
    print(
        "{}, {}, {}, coefficients: {:2.2e}, reconstruction, {:2.2e}".format(
            wavelet.name, str(level).center(4), size, errc, err
        )
    )
    assert np.allclose(reconstruction_2d, face_extended)


def test_2d():
    """Go through various test cases."""
    for wavelet in ["haar", "db3", "db4", "sym2", "sym5"]:
        for level in [1, 2, None]:
            for size in [(65, 65), (64, 64), (47, 45), (45, 47), (32, 32)]:
                run_2dtest(wavelet, level, size)


if __name__ == "__main__":
    test_2d()
