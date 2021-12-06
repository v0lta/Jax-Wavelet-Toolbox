"""Compute wavelet packets using jwt."""

#
# Created on Fri Jun 19 2020
# Copyright (c) 2020 Moritz Wolter
#
import collections

import jax
import jax.numpy as np
import pywt

from .conv_fwt import _fwt_pad, get_filter_arrays


class WaveletPacket(collections.UserDict):
    """A wavelet packet tree."""

    def __init__(self, data: np.array, wavelet, mode: str = "reflect"):
        """Create a wavelet packet decomposition object.

        Args:
            data (np.array): The input data array of shape [time].
            wavelet (pywt.Wavelet or JaxWavelet): The wavelet used for the decomposition.
            mode (str): The desired padding method
        """
        self.input_data = np.expand_dims(np.expand_dims(data, 0), 0)
        self.wavelet = wavelet
        self.mode = mode
        self.nodes = {}
        self.data = None
        self._wavepacketdec(self.input_data, wavelet, mode=mode)

    def get_level(self, level: int) -> list:
        """Return the graycodes for a given level.

        Args:
            level (int): The required depth of the tree.

        Returns:
            list: A list with the node names.
        """
        return self._get_graycode_order(level)

    def _get_graycode_order(self, level, x="a", y="d"):
        graycode_order = [x, y]
        for _ in range(level - 1):
            graycode_order = [x + path for path in graycode_order] + [
                y + path for path in graycode_order[::-1]
            ]
        return graycode_order

    def _recursive_dwt(self, data, filt, mode, level, max_level, path):
        self.data[path] = np.squeeze(data)
        if level < max_level:
            data = _fwt_pad(data, filt_len=filt.shape[-1])
            res = jax.lax.conv_general_dilated(
                lhs=data,  # lhs = NCH image tensor
                rhs=filt,  # rhs = OIH conv kernel tensor
                padding="VALID",
                window_strides=[
                    2,
                ],
                dimension_numbers=("NCH", "OIH", "NCH"),
            )
            res_lo, res_hi = np.split(res, 2, 1)
            return self._recursive_dwt(
                res_lo, filt, mode, level + 1, max_level, path + "a"
            ), self._recursive_dwt(res_hi, filt, mode, level + 1, max_level, path + "d")
        else:
            self.data[path] = np.squeeze(data)

    def _wavepacketdec(self, data, wavelet, level=None, mode="reflect"):
        self.data = {}
        dec_lo, dec_hi, _, _ = get_filter_arrays(wavelet, flip=True)
        filt_len = dec_lo.shape[-1]
        filt = np.stack([dec_lo, dec_hi], 0)

        if level is None:
            level = pywt.dwt_max_level(data.shape[-1], filt_len)
        self._recursive_dwt(data, filt, mode, level=0, max_level=level, path="")
