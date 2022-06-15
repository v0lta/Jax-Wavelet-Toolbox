"""Compute wavelet packets using jwt."""

#
# Created on Fri Jun 19 2020
# Copyright (c) 2020 Moritz Wolter
#
import collections
from typing import TYPE_CHECKING, List, Optional, Union

import jax
import jax.numpy as jnp
import pywt

from .conv_fwt import _fwt_pad, _get_filter_arrays
from .conv_fwt_2d import wavedec2
from .utils import Wavelet, _as_wavelet

if TYPE_CHECKING:
    BaseDict = collections.UserDict[str, jnp.ndarray]
else:
    BaseDict = collections.UserDict


class WaveletPacket(BaseDict):
    """A wavelet packet tree."""

    def __init__(
        self,
        data: jnp.ndarray,
        wavelet: Wavelet,
        mode: str = "reflect",
        max_level: Optional[int] = None,
    ):
        """Create a wavelet packet decomposition object.

        Args:
            data (jnp.array): The input data array of shape [batch_size, time].
            wavelet (Wavelet): The wavelet used for the decomposition.
            mode (str): The desired padding method. Choose i.e.
                "reflect", "symmetric" or "zero". Defaults to "reflect".
        """
        if len(data.shape) == 1:
            self.input_data = jnp.expand_dims(jnp.expand_dims(data, 0), 0)
        elif len(data.shape) == 2:
            self.input_data = jnp.expand_dims(data, 1)

        self.wavelet = wavelet
        self.mode = mode
        self.data = {}

        dec_lo, dec_hi, _, _ = _get_filter_arrays(wavelet, flip=True)
        filt_len = dec_lo.shape[-1]
        filt = jnp.stack([dec_lo, dec_hi], 0)

        if max_level is None:
            max_level = pywt.dwt_max_level(data.shape[-1], filt_len)
        self._recursive_dwt(
            self.input_data, filt, level=0, max_level=max_level, path=""
        )

    def get_level(self, level: int) -> List[str]:
        """Return the graycodes for a given level.

        Args:
            level (int): The required depth of the tree.

        Returns:
            list: A list with the node names.
        """
        return self._get_graycode_order(level)

    def _get_graycode_order(self, level: int, x: str = "a", y: str = "d") -> List[str]:
        graycode_order = [x, y]
        for _ in range(level - 1):
            graycode_order = [x + path for path in graycode_order] + [
                y + path for path in graycode_order[::-1]
            ]
        return graycode_order

    def _recursive_dwt(
        self,
        data: jnp.ndarray,
        filt: jnp.ndarray,
        level: int,
        max_level: int,
        path: str,
    ) -> None:
        self.data[path] = jnp.squeeze(data, 1)
        if level < max_level:
            data = _fwt_pad(data, filt_len=filt.shape[-1], mode=self.mode)
            res = jax.lax.conv_general_dilated(
                lhs=data,  # lhs = NCH image tensor
                rhs=filt,  # rhs = OIH conv kernel tensor
                padding="VALID",
                window_strides=[
                    2,
                ],
                dimension_numbers=("NCH", "OIH", "NCH"),
            )
            res_lo, res_hi = jnp.split(res, 2, 1)
            self._recursive_dwt(res_lo, filt, level + 1, max_level, path + "a")
            self._recursive_dwt(res_hi, filt, level + 1, max_level, path + "d")
        else:
            self.data[path] = jnp.squeeze(data, 1)


class WaveletPacket2D(BaseDict):
    """A wavelet packet tree."""

    def __init__(
        self,
        data: jnp.ndarray,
        wavelet: Union[str, pywt.Wavelet],
        mode: str = "reflect",
        max_level: Optional[int] = None,
    ):
        """Create a 2D-wavelet packet decomposition object.

        Args:
            data (jnp.array): The input data array of shape [batch_size, height, width].
            wavelet (Wavelet): The wavelet used for the decomposition.
            mode (str): The desired padding method. Choose i.e.
                "reflect", "symmetric" or "zero". Defaults to "reflect".
        """
        self.input_data = data
        self.wavelet: pywt.Wavelet = _as_wavelet(wavelet)
        self.mode = mode
        self.data = {}
        if max_level is None:
            self.max_level = pywt.dwt_max_level(
                min(self.input_data.shape[-2:]), self.wavelet.dec_len
            )
        else:
            self.max_level = max_level
        self._recursive_dwt2d(self.input_data, level=0, path="")

    def get_level(self, level: int) -> List[str]:
        """Return the graycodes for a given level.

        Args:
            level (int): The required depth of the tree.

        Returns:
            list: A list with the node names.
        """
        return self._get_graycode_order(level)

    def _get_graycode_order(self, level: int, x: str = "a", y: str = "d") -> List[str]:
        graycode_order = [x, y]
        for _ in range(level - 1):
            graycode_order = [x + path for path in graycode_order] + [
                y + path for path in graycode_order[::-1]
            ]
        return graycode_order

    def _recursive_dwt2d(
        self,
        data: jnp.ndarray,
        level: int,
        path: str,
    ) -> None:
        self.data[path] = data
        if level < self.max_level:
            result_a, (result_h, result_v, result_d) = wavedec2(
                data, self.wavelet, 1, mode=self.mode
            )
            # assert for type checking
            assert not isinstance(result_a, tuple)
            self._recursive_dwt2d(result_a, level + 1, path + "a")
            self._recursive_dwt2d(result_h, level + 1, path + "h")
            self._recursive_dwt2d(result_v, level + 1, path + "v")
            self._recursive_dwt2d(result_d, level + 1, path + "d")
        else:
            self.data[path] = data
