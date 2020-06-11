import pywt
import jax
import jax.numpy as np


def fwt_pad(data, wavelet):
    pad = 0
    if data.shape[-1] % 2 != 0:
        pad += 1

    data = np.pad(data, ((0, 0), (0, 0), (0, pad)), 'reflect')
    return data


def get_filter_tensors(wavelet, flip):
    def create_tensor(filter):
        if flip:
            if type(filter) is np.array:
                return np.expand_dims(np.flip(filter), 0)
            else:
                return np.expand_dims(np.array(filter[::-1]), 0)
        else:
            if type(filter) is np.array:
                return np.expand_dims(filter, 0)
            else:
                return np.expand_dims(np.array(filter), 0)
    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo = create_tensor(dec_lo)
    dec_hi = create_tensor(dec_hi)
    rec_lo = create_tensor(rec_lo)
    rec_hi = create_tensor(rec_hi)
    return dec_lo, dec_hi, rec_lo, rec_hi


def analysis_fwt(data, wavelet, scales: int = None):

    dec_lo, dec_hi, _, _ = get_filter_tensors(wavelet, flip=True)
    filt_len = dec_lo.shape[-1]
    dec_lo = np.array(dec_lo[::-1])
    dec_hi = np.array(dec_hi[::-1])
    filt = np.stack([dec_lo, dec_hi], 0)

    if scales is None:
        scales = pywt.dwt_max_level(data.shape[-1], filt_len)

    result_lst = []
    res_lo = data
    for _ in range(scales):
        res_lo = fwt_pad(res_lo, wavelet)
        res = jax.lax.conv(lhs=res_lo, # lhs = NCW image tensor
                           rhs=filt,   # rhs = OIW conv kernel tensor
                           padding='VALID', window_strides=[2,])
        res_lo, res_hi = np.split(res, 2, 1)
        result_lst.append(res_hi)
    result_lst.append(res_lo)
    return result_lst[::-1]


def synthesis_fwt(data, wavelet, scales: int= None):
    pass



if __name__ == '__main__':
    from lorenz import generate_lorenz
    import os
    os.environ["DISPLAY"] = ":1"
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    wavelet = pywt.Wavelet('haar')

    data = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.])
    data = np.expand_dims(np.expand_dims(data, 0), 0)
    coeffs = pywt.wavedec(data, wavelet, level=2)
    coeffs2 = analysis_fwt(data, wavelet, scales=2)
    cat_coeffs = np.concatenate(coeffs, -1)
    cat_coeffs2 = np.concatenate(coeffs2, -1)
    err = np.sum(np.abs(cat_coeffs - cat_coeffs2))
    print(err, 'done')