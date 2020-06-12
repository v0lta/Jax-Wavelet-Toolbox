#
# Created on Thu Jun 12 2020
# Copyright (c) 2020 Moritz Wolter
#
import pywt
import jax
import jax.numpy as np
from conv_fwt import fwt_pad
from conv_fwt import get_filter_arrays



def construct_2d_filt(lo, hi):
    """ Construct 2d filters from 1d inputs."""
    ll = np.outer(lo, lo)
    lh = np.outer(hi, lo)
    hl = np.outer(lo, hi)
    hh = np.outer(hi, hi)
    filt = np.stack([ll, lh, hl, hh], 0)
    filt = np.expand_dims(filt, 1)
    return filt


def fwt_pad2d(data: np.array, wavelet, mode='reflect') -> np.array:
    filt_len = len(wavelet.dec_lo)
    padr = 0
    padl = 0
    padt = 0
    padb = 0
    if filt_len > 2:
        # we pad half of the total requried padding on each side.
        padr += (2*filt_len - 3)//2
        padl += (2*filt_len - 3)//2
        padt += (2*filt_len - 3)//2
        padb += (2*filt_len - 3)//2

    # pad to even singal length.
    if data.shape[-1] % 2 != 0:
        padr += 1
    if data.shape[-2] % 2 != 0:
        padb += 1

    data = np.pad(data, ((0, 0), (0, 0), (padt, padb), (padl, padr)),
                  mode)
    return data


def wavedec2(data, wavelet, scales: int=None):
    """ 2d non-seperated fwt """
    # dec_lo, dec_hi, _, _ = wavelet.filter_bank
    # filt_len = len(dec_lo)
    # dec_lo = torch.tensor(dec_lo[::-1]).unsqueeze(0)
    # dec_hi = torch.tensor(dec_hi[::-1]).unsqueeze(0)
    dec_lo, dec_hi, _, _ = get_filter_arrays(wavelet, flip=True)
    dec_filt = construct_2d_filt(lo=dec_lo, hi=dec_hi)

    if scales is None:
        scales = pywt.dwtn_max_level([data.shape[-1], data.shape[-2]], wavelet)

    result_lst = []
    res_ll = data
    for _ in range(scales):
        res_ll = fwt_pad2d(res_ll, wavelet)
        # res = torch.nn.functional.conv2d(res_ll, dec_filt, stride=2)
        res = jax.lax.conv_general_dilated(
            lhs=res_ll, # lhs = NCHw image tensor
            rhs=dec_filt,   # rhs = OIHw conv kernel tensor
            padding='VALID', window_strides=[2,2],
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'))

        res_ll, res_lh, res_hl, res_hh = np.split(res, 4, 1)
        result_lst.append((res_lh, res_hl, res_hh))
    result_lst.append(res_ll)
    result_lst.reverse()
    return result_lst


def waverec2(coeffs, wavelet):
    """ 2d non separated ifwt"""
    _, _, rec_lo, rec_hi = get_filter_arrays(wavelet, flip=True)
    filt_len = rec_lo.shape[-1]
    rec_filt = construct_2d_filt(lo=rec_lo, hi=rec_hi)
    rec_filt = np.transpose(rec_filt, [1, 0, 2, 3])

    res_ll = coeffs[0]
    for c_pos, res_lh_hl_hh in enumerate(coeffs[1:]):
        print(res_ll.shape)
        res_ll = np.concatenate([res_ll, res_lh_hl_hh[0], res_lh_hl_hh[1], res_lh_hl_hh[2]], 1)
        res_ll = jax.lax.conv_transpose(lhs=res_ll, rhs=rec_filt,
                                        padding='VALID',
                                        strides=[2,2],
                                        dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
        # rec_filt_rot = np.rot90(np.rot90(rec_filt, axes=[0, 1]), axes=[0, 1])
        # res_ll = jax.lax.conv_general_dilated(
        #      lhs=res_ll, # lhs = NCHw image tensor
        #      rhs=rec_filt_rot,   # rhs = OIHw conv kernel tensor
        #      padding=((1, 1), (1, 1)), 
        #      window_strides=[1, 1], lhs_dilation=[2, 2],
        #      dimension_numbers=('NCHW', 'OIHW', 'NCHW'))

        # remove the padding
        padl = (2*filt_len - 3)//2
        padr = (2*filt_len - 3)//2
        padt = (2*filt_len - 3)//2
        padb = (2*filt_len - 3)//2
        if c_pos < len(coeffs)-2:
            pred_len = res_ll.shape[-1] - (padl + padr)
            next_len = coeffs[c_pos+2][0].shape[-1]
            pred_len2 = res_ll.shape[-2] - (padt + padb)
            next_len2 = coeffs[c_pos+2][0].shape[-2]
            if next_len != pred_len:
                padr += 1
                pred_len = res_ll.shape[-1] - (padl + padr)
                assert next_len == pred_len, 'padding error, please open an issue on github '
            if next_len2 != pred_len2:
                padb += 1
                pred_len2 = res_ll.shape[-2] - (padt + padb)
                assert next_len2 == pred_len2, 'padding error, please open an issue on github '
        if padt > 0:
            res_ll = res_ll[..., padt:, :]
        if padb > 0:
            res_ll = res_ll[..., :-padb, :]
        if padl > 0:
            res_ll = res_ll[..., padl:]
        if padr > 0:
            res_ll = res_ll[..., :-padr]
    return res_ll

if __name__ == '__main__':
    import os
    os.environ["DISPLAY"] = ":1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import jax.numpy as np
    import scipy.misc
    from test_conv_fwt2d import flatten_2d_coeff_lst
    
    face = np.transpose(scipy.misc.face(), [2, 0, 1]).astype(np.float32)
    face_exd = np.expand_dims(np.array(face), 1)
    wavelet = pywt.Wavelet('haar')

    coeff2d_pywt = pywt.wavedec2(face, wavelet, mode='reflect', level=1)
    coeff2d = wavedec2(face_exd, wavelet, scales=1)
    recss2d = waverec2(coeff2d, wavelet)

    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = np.concatenate(flatten_2d_coeff_lst(coeff2d), -1)
    errc = np.mean(np.abs(flat_lst - flat_lst2))
    print('coefficient error', errc)
    print('done')

    print('err pywt', np.mean(np.abs(pywt.waverec2(coeff2d, wavelet) - face_exd)))
    print('err', np.mean(np.abs(recss2d -  face_exd)))
    plt.imshow(np.transpose(recss2d[:, 0,:,:], [1, 2, 0])/np.max(np.abs(recss2d)))
    plt.show()
    errimg = np.abs(recss2d -  face_exd)
    plt.imshow(np.transpose(errimg[:, 0,:,:], [1, 2, 0])/np.max(np.abs(errimg)))
    plt.show()
    print('done')