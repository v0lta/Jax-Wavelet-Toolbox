import pywt
import jax
import jax.numpy as np

def analysis_fwt_2d(data, wavelet, scales: int = None):
    # TODO write me
    # dec_lo, dec_hi, _, _ = get_filter_tensors(wavelet, flip=True)
    # filt_len = dec_lo.shape[-1]
    # # dec_lo = np.array(dec_lo[::-1])
    # # dec_hi = np.array(dec_hi[::-1])
    # filt = np.stack([dec_lo, dec_hi], 0)

    # if scales is None:
    #     scales = pywt.dwt_max_level(data.shape[-1], filt_len)

    # result_lst = []
    # res_lo = data
    # for _ in range(scales):
    #     res_lo = fwt_pad(res_lo, wavelet)
    #     res = jax.lax.conv_general_dilated(
    #         lhs=res_lo, # lhs = NCH image tensor
    #         rhs=filt,   # rhs = OIH conv kernel tensor
    #         padding='VALID', window_strides=[2,],
    #         dimension_numbers=('NCH', 'OIH', 'NCH'))
    #     res_lo, res_hi = np.split(res, 2, 1)
    #     result_lst.append(res_hi)
    # result_lst.append(res_lo)
    # return result_lst[::-1]
    pass