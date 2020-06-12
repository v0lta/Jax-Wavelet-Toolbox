import pywt
import scipy
import jax.numpy as np
from conv_fwt_2d import wavedec2, waverec2


def flatten_2d_coeff_lst(coeff_lst_2d, flatten_arrays=True):
    flat_coeff_lst = []
    for coeff in coeff_lst_2d:
        if type(coeff) is tuple:
            for c in coeff:
                if flatten_arrays:
                    flat_coeff_lst.append(c.flatten())
                else:
                    flat_coeff_lst.append(c)
        else:
            if flatten_arrays:
                flat_coeff_lst.append(coeff.flatten())
            else:
                flat_coeff_lst.append(coeff)
    return flat_coeff_lst


def run_2dtest(wavelet):
    face = np.transpose(scipy.misc.face(), [2, 0, 1]).astype(np.float32)
    face_exd = np.expand_dims(np.array(face), 1)
    
    coeff2d_pywt = pywt.wavedec2(face, wavelet, mode='reflect', level=5)
    coeff2d = wavedec2(face_exd, wavelet, scales=5)
    # test pywt compatability
    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = np.concatenate(flatten_2d_coeff_lst(coeff2d), -1)
    # print([c.shape for c in flatten_2d_coeff_lst(coeff2d_pywt, flatten_tensors=False)])
    # print([list(c.shape) for c in flatten_2d_coeff_lst(coeff2d, flatten_tensors=False)])
    errc = np.mean(np.abs(flat_lst - flat_lst2))
    assert errc < 1e-4

    # test invertability
    recss2d = waverec2(coeff2d, wavelet)
    err = np.mean(np.abs(recss2d -  face_exd))
    assert err < 1e-4

def test_2d_haar(wavelet=pywt.Wavelet('haar')):
    run_2dtest(wavelet)

def test_2d_db2(wavelet=pywt.Wavelet('db2')):
    run_2dtest(wavelet)

def test_2d_db3(wavelet=pywt.Wavelet('db3')):
    run_2dtest(wavelet)