"""Compute wavelet packets using jwt."""

#
# Created on Fri Jun 19 2020
# Copyright (c) 2020 Moritz Wolter
#
import collections
from itertools import product
from typing import TYPE_CHECKING, List, Optional, Union

# import jax
import jax.numpy as jnp
import pywt

from .conv_fwt import wavedec, waverec
from .conv_fwt_2d import wavedec2, waverec2
from .utils import _as_wavelet

if TYPE_CHECKING:
    BaseDict = collections.UserDict[str, jnp.ndarray]
else:
    BaseDict = collections.UserDict


class WaveletPacket(BaseDict):
    """A wavelet packet tree."""

    def __init__(
        self,
        data: jnp.ndarray,
        wavelet: pywt.Wavelet,
        mode: str = "reflect",
        max_level: Optional[int] = None,
    ):
        """Create a wavelet packet decomposition object.

        Args:
            data (jnp.ndarray): The input data array of shape [batch_size, time].
            wavelet (pywt.Wavelet): The wavelet used for the decomposition.
            mode (str): The desired padding method. Choose i.e.
                "reflect", "symmetric" or "zero". Defaults to "reflect".

        Example:
            >>> import pywt
            >>> import jax.numpy as jnp
            >>> from jaxwt import WaveletPacket
            >>> import scipy.signal as signal
            >>> wavelet = pywt.Wavelet("db4")
            >>> t = jnp.linspace(0, 10, 5001)
            >>> w = signal.chirp(t, f0=0.00001,
            >>>                  f1=20, t1=10, method="linear")
            >>> wp = WaveletPacket(data=w, wavelet=wavelet)
            >>> nodes = wp.get_level(7)
            >>> jnp_lst = []
            >>> for node in nodes:
            >>>     jnp_lst.append(wp[node])
            >>> viz = jnp.concatenate(jnp_lst)
        """
        if len(data.shape) == 1:
            self.input_data = jnp.expand_dims(jnp.expand_dims(data, 0), 0)
        elif len(data.shape) == 2:
            self.input_data = jnp.expand_dims(data, 1)

        self.wavelet = _as_wavelet(wavelet)
        self.mode = mode
        self.data = {}
        if max_level is None:
            self.max_level = pywt.dwt_max_level(data.shape[-1], self.wavelet.dec_len)
        else:
            self.max_level = max_level
        self._recursive_dwt(self.input_data, level=0, path="")

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
        if level == 0:
            return [""]
        else:
            return graycode_order

    def _recursive_dwt(
        self,
        data: jnp.ndarray,
        level: int,
        path: str,
    ) -> None:
        self.data[path] = jnp.squeeze(data, 1)
        if level < self.max_level:
            res_lo, res_hi = wavedec(data, self.wavelet, 1, mode=self.mode)
            self._recursive_dwt(res_lo, level + 1, path + "a")
            self._recursive_dwt(res_hi, level + 1, path + "d")
        else:
            self.data[path] = jnp.squeeze(data, 1)

    def reconstruct(self) -> "WaveletPacket":
        """Recursively reconstruct the input starting from the leaf nodes.

        Reconstruction replaces the input-data originally assigned to this object.

        Note:
           Only changes to leaf node data impacts the results,
           since changes in all other nodes will be replaced with
           a reconstruction from the leafs.

        Example:
            >>> import jaxwt as jwt
            >>> import jax
            >>> key = jax.random.PRNGKey(0)
            >>> input_data = jax.random.normal(key, (1, 24))
            >>> jwp = jwt.WaveletPacket(input_data, "haar", max_level=2)
            >>> jwp["a" * 2] *= 0
            >>> jwp.reconstruct()
            >>> print(jwp[""])
        """
        if self.max_level is None:
            self.max_level = pywt.dwt_max_level(
                self[""].shape[-1], self.wavelet.dec_len
            )

        for level in reversed(range(self.max_level)):
            for node in self.get_level(level):
                data_a = self[node + "a"]
                data_b = self[node + "d"]
                rec = waverec([data_a, data_b], self.wavelet)
                self[node] = rec
        return self


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

        Example code illustrating the use of this class is available at:
        https://github.com/v0lta/Jax-Wavelet-Toolbox/tree/packet-patch/examples/deepfake_analysis

        Args:
            data (jnp.ndarray): The input data array of shape [batch_size, height, width].
            wavelet (pywt.Wavelet or str): The wavelet used for the decomposition.
            mode (str): The desired padding method. Choose i.e.
                "reflect", "symmetric" or "zero". Defaults to "reflect".
            max_level (int, optional): Choose the desired decomposition level.
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

    def _get_natural_order(self, level: int) -> List[str]:
        """Get the natural ordering for a given decomposition level.

        Args:
            level (int): The decomposition level.

        Returns:
            list: A list with the filter order strings.
        """
        return ["".join(p) for p in list(product(["a", "h", "v", "d"], repeat=level))]

    def reconstruct(self) -> "WaveletPacket2D":
        """Recursively reconstruct the input starting from the leaf nodes.

        Note:
           Only changes to leaf node data impacts the results,
           since changes in all other nodes will be replaced with
           a reconstruction from the leafs.
        """
        if self.max_level is None:
            self.max_level = pywt.dwt_max_level(
                min(self[""].shape[-2:]), self.wavelet.dec_len
            )

        for level in reversed(range(self.max_level)):
            for node in self._get_natural_order(level):
                data_a = self[node + "a"]
                data_h = self[node + "h"]
                data_v = self[node + "v"]
                data_d = self[node + "d"]
                rec = waverec2([data_a, (data_h, data_v, data_d)], self.wavelet)
                self[node] = rec
        return self
