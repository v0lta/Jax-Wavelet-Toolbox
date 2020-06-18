import jax.numpy as np
import pywt

from src.jaxlets.conv_fwt import dwt, idwt
from src.jaxlets.lorenz import generate_lorenz
from src.jaxlets.utils import JaxWavelet

def dwt_idwt_lorenz(wavelet, mode='reflect'):
    jax_wavelet = JaxWavelet(wavelet.dec_lo, wavelet.dec_hi,
                             wavelet.rec_lo, wavelet.rec_hi)
    # ---- Test single level wavelet analysis and synthesis on lorenz signal. -----
    lorenz = np.transpose(np.expand_dims(generate_lorenz()[:, 0], -1), [1, 0])
    data = np.expand_dims(lorenz, 0)
    coeff = dwt(data, wavelet=jax_wavelet, mode=mode)
    pywt_coeff = pywt.dwt(lorenz, wavelet, mode=mode)
    cat_coeff = np.concatenate(coeff, axis=-1)
    pywt_cat_coeff = np.concatenate(pywt_coeff, axis=-1)
    err = np.mean(np.abs(cat_coeff - pywt_cat_coeff))
    assert err < 1e-4
    rest_data = idwt(coeff, jax_wavelet)
    err = np.mean(np.abs(rest_data - data))
    assert err < 1e-4

def test_haar_reflect():
    wavelet = pywt.Wavelet('haar')
    dwt_idwt_lorenz(wavelet, mode='reflect')

def test_haar_symmetric():
    wavelet = pywt.Wavelet('haar')
    dwt_idwt_lorenz(wavelet, mode='symmetric')

def test_db2_reflect():
    wavelet = pywt.Wavelet('db2')
    dwt_idwt_lorenz(wavelet, mode='reflect')

def test_db2_symmetric():
    wavelet = pywt.Wavelet('db2')
    dwt_idwt_lorenz(wavelet, mode='symmetric')

def test_db4_reflect():
    wavelet = pywt.Wavelet('db4')
    dwt_idwt_lorenz(wavelet, mode='reflect')

def test_db4_symmetric():
    wavelet = pywt.Wavelet('db4')
    dwt_idwt_lorenz(wavelet, mode='symmetric')

def test_db8_reflect():
    wavelet = pywt.Wavelet('db8')
    dwt_idwt_lorenz(wavelet, mode='reflect')

def test_db8_symmetric():
    wavelet = pywt.Wavelet('db8')
    dwt_idwt_lorenz(wavelet, mode='symmetric')