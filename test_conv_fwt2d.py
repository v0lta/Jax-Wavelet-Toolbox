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


def run_2dtest(wavelet, level=None):
    face = np.transpose(scipy.misc.face(), [2, 0, 1]).astype(np.float32)
    face = face[:, 128:(512+128), 256:(512+256)]
    face_exd = np.expand_dims(np.array(face), 1)
    
    coeff2d_pywt = pywt.wavedec2(face, wavelet, mode='reflect', level=level)
    coeff2d = wavedec2(face_exd, wavelet, scales=level)
    # test pywt compatability
    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = np.concatenate(flatten_2d_coeff_lst(coeff2d), -1)
    # print([c.shape for c in flatten_2d_coeff_lst(coeff2d_pywt, flatten_tensors=False)])
    # print([list(c.shape) for c in flatten_2d_coeff_lst(coeff2d, flatten_tensors=False)])
    errc = np.mean(np.abs(flat_lst - flat_lst2))
    assert errc < 5e-4

    # test invertability
    recss2d = waverec2(coeff2d, wavelet)
    err = np.mean(np.abs(recss2d -  face_exd))
    assert err < 5e-4

def test_2d_haar_l1(wavelet=pywt.Wavelet('haar')):
    run_2dtest(wavelet, level=1)

def test_2d_haar_l2(wavelet=pywt.Wavelet('haar')):
    run_2dtest(wavelet, level=2)

def test_2d_haar_l3(wavelet=pywt.Wavelet('haar')):
    run_2dtest(wavelet, level=3)

def test_2d_haar_lmax(wavelet=pywt.Wavelet('haar')):
    run_2dtest(wavelet, level=None)

def test_2d_db2(wavelet=pywt.Wavelet('db2')):
    run_2dtest(wavelet)

def test_2d_db3(wavelet=pywt.Wavelet('db3')):
    run_2dtest(wavelet)

def test_2d_db4(wavelet=pywt.Wavelet('db4')):
    run_2dtest(wavelet)

def test_2d_sym5(wavelet=pywt.Wavelet('sym5')):
    run_2dtest(wavelet)
    