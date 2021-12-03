"""2d Convolution fast wavelet transform test code."""

import jax.numpy as np
import pywt
import scipy
from src.jwt.conv_fwt_2d import wavedec2, waverec2
from src.jwt.utils import flatten_2d_coeff_lst


def run_2dtest(wavelet: str, level: int, size: tuple):
    """Run a specific test."""
    wavelet = pywt.Wavelet(wavelet)
    face = np.transpose(scipy.misc.face(), [2, 0, 1]).astype(np.float32)
    face = face[:, 128 : (512 + size[0]), 256 : (512 + size[1])]
    face_exd = np.expand_dims(np.array(face), 1)

    coeff2d_pywt = pywt.wavedec2(face, wavelet, mode="reflect", level=level)
    coeff2d = wavedec2(face_exd, wavelet, level=level)
    # test pywt compatability
    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = np.concatenate(flatten_2d_coeff_lst(coeff2d), -1)
    errc = np.mean(np.abs(flat_lst - flat_lst2))
    assert errc < 5e-4

    # test invertability
    recss2d = waverec2(coeff2d, wavelet)
    err = np.mean(np.abs(recss2d - face_exd))
    print(
        "{}, {}, {}, coefficient error: {:2.2e}, reconstruction err, {:2.2e}".format(
            wavelet.name, str(level).center(4), size, errc, err
        )
    )
    assert np.allclose(recss2d, face_exd, atol=1e-3)


def test_2d():
    """Go through various test cases."""
    for wavelet in ["haar", "db3", "db4", "sym2", "sym5"]:
        for level in [1, 2, None]:
            for size in [(32, 32), (64, 64)]:
                run_2dtest(wavelet, level, size)


if __name__ == "__main__":
    test_2d()
