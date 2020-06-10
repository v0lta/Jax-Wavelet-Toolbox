import pywt
import jax
import jax.numpy as np


def fwt_pad(data, wavelet):
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


def forward_fwt(data, wavelet, scales: int = None):

    dec_lo, dec_hi, _, _ = get_filter_tensors(wavelet, flip=True)
    filt_len = dec_lo.shape[-1]
    dec_lo = np.expand_dims(np.array(dec_lo[::-1]), 0)
    dec_hi = np.expand_dims(np.array(dec_hi[::-1]), 0)
    filt = np.stack([dec_lo, dec_hi], 0)

    if scales is None:
        scales = pywt.dwt_max_level(data.shape[-1], filt_len)

    result_lst = []
    res_lo = data
    for _ in range(scales):
        #if filt_len > 2:
        res_lo = fwt_pad(res_lo, wavelet)
        res = jax.lax.conv(lhs=res_lo, # lhs = NCHW image tensor
                           rhs=filt,   # rhs = OIHW conv kernel tensor
                           padding='SAME', window_strides=[1, 2])
        res_lo, res_hi = np.split(res, 2, 1)
        result_lst.append(res_hi.squeeze(1))
    result_lst.append(res_lo.squeeze(1))
    return result_lst[::-1]



if __name__ == '__main__':
    from lorenz import generate_lorenz
    import matplotlib.pyplot as plt

    data = np.expand_dims(np.expand_dims(generate_lorenz(), 0), 0)
    data = np.transpose(data, axes=[0, 1, 3, 2])
    wavelet = pywt.Wavelet('haar')
    coeff = forward_fwt(data, wavelet)
    cat_coeff = np.concatenate(coeff, axis=-1)
    plt.plot(cat_coeff[0, 0, :])
    pywt_coeff = pywt.wavedec(data, wavelet)
    pywt_cat_coeff = np.concatenate(pywt_coeff, axis=-1)
    plt.plot(pywt_cat_coeff[0, 0, 0, :])
    plt.show()
    print('err', np.sum(np.abs(cat_coeff - pywt_cat_coeff[0])))
    print('done')