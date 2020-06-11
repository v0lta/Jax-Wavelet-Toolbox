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
    # dec_lo = np.array(dec_lo[::-1])
    # dec_hi = np.array(dec_hi[::-1])
    filt = np.stack([dec_lo, dec_hi], 0)

    if scales is None:
        scales = pywt.dwt_max_level(data.shape[-1], filt_len)

    result_lst = []
    res_lo = data
    for _ in range(scales):
        res_lo = fwt_pad(res_lo, wavelet)
        res = jax.lax.conv_general_dilated(
            lhs=res_lo, # lhs = NCH image tensor
            rhs=filt,   # rhs = OIH conv kernel tensor
            padding='VALID', window_strides=[2,],
            dimension_numbers=('NCH', 'OIH', 'NCH'))
        res_lo, res_hi = np.split(res, 2, 1)
        result_lst.append(res_hi)
    result_lst.append(res_lo)
    return result_lst[::-1]


def synthesis_fwt(coeffs, wavelet, scales: int= None):
    # lax's transpose conv requires filter flips in contrast to pytorch.
    _, _, rec_lo, rec_hi = get_filter_tensors(wavelet, flip=True) 
    filt_len = rec_lo.shape[-1]
    filt = np.stack([rec_lo, rec_hi], 1)

    res_lo = coeffs[0]
    for c_pos, res_hi in enumerate(coeffs[1:]):
        # print('shapes', res_lo.shape, res_hi.shape)
        res_lo = np.concatenate([res_lo, res_hi], 1)
        res_lo = jax.lax.conv_transpose(lhs=res_lo, rhs=filt, padding='VALID',
                                        strides=[2,],
                                        dimension_numbers=('NCH', 'OIH', 'NCH'))

        # remove the padding
        # padl = (2*filt_len - 3)//2
        # padr = (2*filt_len - 3)//2
        padl = 0
        if c_pos < len(coeffs)-2:
            pred_len = res_lo.shape[-1]
            nex_len = coeffs[c_pos+2].shape[-1]
            if nex_len != pred_len:
                padl += 1
                pred_len = res_lo.shape[-1] - padl
                assert nex_len == pred_len, 'padding error, please open an issue on github '
                res_lo = res_lo[..., :-padl]
    return res_lo



if __name__ == '__main__':
    from lorenz import generate_lorenz
    import os
    os.environ["DISPLAY"] = ":1"
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    wavelet = pywt.Wavelet('haar')
    # ---- Test harr wavelet analysis and synthesis on lorenz signal. -----
    lorenz = np.transpose(np.expand_dims(generate_lorenz()[:, 0], -1), [1, 0])
    data = np.expand_dims(lorenz, 0)

    coeff = analysis_fwt(data, wavelet)
    cat_coeff = np.concatenate(coeff, axis=-1)
    pywt_coeff = pywt.wavedec(lorenz, wavelet, mode='reflect')
    pywt_cat_coeff = np.concatenate(pywt_coeff, axis=-1)
    err = np.mean(np.abs(cat_coeff - pywt_cat_coeff))
    assert err < 1e-6

    rest_data = synthesis_fwt(coeff, wavelet, scales=2)
    err = np.mean(np.abs(rest_data - data))
    plt.plot(rest_data[0, 0, :])
    plt.plot(data[0, 0, :])
    plt.show()
    # assert err < 1e-6
    print('done')