#
# Created on Fri Jun 19 2020
# Copyright (c) 2020 Moritz Wolter
#
import jax
import jax.numpy as np
import collections
from jaxlets.conv_fwt import dwt_max_level, get_filter_arrays, fwt_pad


class WaveletPacket(collections.UserDict):

    def __init__(self, data: np.array, wavelet, mode: str='reflect'):
        """Create a wavelet packet decomposition object
        Args:
            data (np.array): The input data array.
            wavelet (pywt.Wavelet or JaxWavelet): The wavelet used for the decomposition.
            mode ([str]): The desired padding method
        """        
        self.input_data = data
        self.wavelet = wavelet
        self.mode = mode 
        self.nodes = {}
        self.data = None
        self._wavepacketdec(self.input_data, wavelet, mode=mode)

    def get_level(self, level):
        return self.get_graycode_order(level)

    def get_graycode_order(self, level, x='a', y='d'):
        graycode_order = [x, y]
        for i in range(level - 1):
            graycode_order = [x + path for path in graycode_order] + \
                            [y + path for path in graycode_order[::-1]]
        return graycode_order


    def recursive_dwt(self, data, filt, mode , level, max_level, path):
        self.data[path] = data
        if level < max_level:
            data = fwt_pad(data, filt_len=filt.shape[-1])
            res = jax.lax.conv_general_dilated(
                lhs=data,  # lhs = NCH image tensor
                rhs=filt,  # rhs = OIH conv kernel tensor
                padding='VALID', window_strides=[2, ],
                dimension_numbers=('NCH', 'OIH', 'NCH'))
            res_lo, res_hi = np.split(res, 2, 1)
            return self.recursive_dwt(res_lo, filt, mode, level+1, max_level, path + 'a'), \
                   self.recursive_dwt(res_hi, filt, mode, level+1, max_level, path + 'd')
        else:
            self.data[path] = data


    def _wavepacketdec(self, data, wavelet, level=None, mode='reflect'):
        self.data = {}
        dec_lo, dec_hi, _, _ = get_filter_arrays(wavelet, flip=True)
        filt_len = dec_lo.shape[-1]
        filt = np.stack([dec_lo, dec_hi], 0)

        if level is None:
            # scales = pywt.dwt_max_level(data.shape[-1], filt_len)
            level = dwt_max_level(data.shape[-1], filt_len)
        self.recursive_dwt(data, filt, mode, level=0, max_level=level, path='')




if __name__ == '__main__':
    import pywt
    import numpy as nnp
    import matplotlib.pyplot as plt
    import scipy.signal as signal
    import os
    os.environ["DISPLAY"] = ":1"
    import matplotlib
    matplotlib.use('Qt5Agg')

    # t = [1,   2, 3,  4,  5,  6,  7]
    w = [56., 40., 8., 24., 48., 40., 16.]
    wavelet = pywt.Wavelet('haar')
    data = np.expand_dims(np.expand_dims(np.array(w), 0), 0)
    jwp = WaveletPacket(data, wavelet, mode='reflect')
    nodes = jwp.get_level(2)
    jnp_lst = []
    for node in nodes:
        jnp_lst.append(np.squeeze(jwp[node]))
    res = np.stack(jnp_lst)

    wp = pywt.WaveletPacket(data=nnp.array(w), wavelet='haar',
                            mode='reflect')
    nodes = [node.path for node in wp.get_level(2, 'freq')]
    np_lst = []
    for node in nodes:
        np_lst.append(wp[node].data)
    viz = np.stack(np_lst)

    print(res)
    print(viz)
    print(np.round(res))
    print(np.round(viz))
    print('err', np.mean(np.abs(res - viz)))
    print('stop')


    t = np.linspace(0, 10, 5001)
    w = signal.chirp(t, f0=.00001, f1=20, t1=10, method='linear')

    plt.plot(t, w)
    plt.title("Linear Chirp, f(0)=6, f(10)=1")
    plt.xlabel('t (sec)')
    plt.show()

    wp = pywt.WaveletPacket(data=w, wavelet='db4',
                            mode='symmetric')
    print('maxlevel', wp.maxlevel)
    print([node.path for node in wp.get_level(2, 'natural')])
    nodes = [node.path for node in wp.get_level(7, 'freq')]
    np_lst = []
    for node in nodes:
        np_lst.append(wp[node].data)
    viz = np.stack(np_lst)
    plt.imshow(viz[:20, :])
    plt.show()


    
    # plt.imshow(np.log(np.abs(viz)+0.01))
    # plt.show()
    # print(wp['aa'].data)
    # print(wp['ad'].data)
    # print(wp['dd'].data)
    # print(wp['da'].data)


    # x = np.linspace(0, 1, num=512)
    # data = np.sin(250 * np.pi * x**2)

    # wavelet = 'db2'
    # level = 4
    # order = "freq"  # other option is "normal"
    # interpolation = 'nearest'

    # # Construct wavelet packet
    # wp = pywt.WaveletPacket(data, wavelet, 'symmetric', maxlevel=level)
    # nodes = wp.get_level(level, order=order)
    # labels = [n.path for n in nodes]
    # values = np.array([n.data for n in nodes], 'd')
    # values = abs(values)

    # # Show signal and wavelet packet coefficients
    # fig = plt.figure()
    # fig.subplots_adjust(hspace=0.2, bottom=.03, left=.07, right=.97, top=.92)
    # ax = fig.add_subplot(2, 1, 1)
    # ax.set_title("linchirp signal")
    # ax.plot(x, data, 'b')
    # ax.set_xlim(0, x[-1])

    # ax = fig.add_subplot(2, 1, 2)
    # ax.set_title("Wavelet packet coefficients at level %d" % level)
    # ax.imshow(values, interpolation=interpolation,  aspect="auto",
    #         origin="lower", extent=[0, 1, 0, len(values)])
    # ax.set_yticks(np.arange(0.5, len(labels) + 0.5), labels)

    # # Show spectrogram and wavelet packet coefficients
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(211)
    # ax2.specgram(data, NFFT=64, noverlap=32, Fs=2,
    #             interpolation='bilinear')
    # ax2.set_title("Spectrogram of signal")
    # ax3 = fig2.add_subplot(212)
    # ax3.imshow(values, origin='upper', extent=[-1, 1, -1, 1],
    #         interpolation='nearest')
    # ax3.set_title("Wavelet packet coefficients")
    # plt.show()
