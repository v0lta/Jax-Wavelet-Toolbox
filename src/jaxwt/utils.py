"""Various utility functions."""

#
# Copyright (c) 2023 Moritz Wolter
#
from typing import Any, NamedTuple, Union

import jax.numpy as jnp
import numpy as np
import pywt

__all__ = ["flatten_2d_coeff_lst"]


class WaveletNamedTuple(NamedTuple):
    """A jax jit compatible wavelet object."""

    dec_lo: jnp.ndarray
    dec_hi: jnp.ndarray
    rec_lo: jnp.ndarray
    rec_hi: jnp.ndarray
    name: str = "unnamed"

    @property
    def dec_len(self) -> int:
        return len(self.dec_lo)

    @property
    def rec_len(self) -> int:
        return len(self.rec_lo)

    def __len__(self) -> int:
        return len(self.dec_lo)

    def as_pywt(self) -> pywt.Wavelet:
        return pywt.Wavelet(self.name)


def flatten_2d_coeff_lst(
    coeff_list_2d: list[
        Union[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
    ],
    flatten_arrays: bool = True,
) -> list[jnp.ndarray]:
    """Flattens a list of array tuples into a single list.

    Args:
        coeff_list_2d (list): A pywt-style coefficient list.
        flatten_arrays (bool): If true,
             2d array are flattened. Defaults to True.

    Returns:
        list: A single 1-d list with all original elements.
    """
    flat_coeff_lst = []
    for coeff in coeff_list_2d:
        if isinstance(coeff, tuple):
            for c in coeff:
                if flatten_arrays:
                    flat_coeff_lst.append(c.flatten())
                else:
                    flat_coeff_lst.append(c)
        else:
            if flatten_arrays:
                flat_coeff_lst.append(coeff.flatten())
            else:
                flat_coeff_lst.append(coeff)
    return flat_coeff_lst


def _as_wavelet(
    wavelet: Union[WaveletNamedTuple, str, pywt.Wavelet],
    dtype: jnp.dtype[Any] = jnp.float64,
) -> WaveletNamedTuple:
    """Ensure the input argument to be a pywt wavelet compatible object.

    Args:
        wavelet (Wavelet or str): The input argument, which is either a
            pywt wavelet compatible object or a valid pywt wavelet name string.

    Returns:
        WaveletNamedTuple: Our Jax-compatible Wavelet object.

    Raises:
        ValueError: If the input argument is neither
            a proper Wavelet object nor a string
    """
    if isinstance(wavelet, WaveletNamedTuple):
        return wavelet
    elif isinstance(wavelet, str):
        return create_wavelet_named_tuple(pywt.Wavelet(wavelet), dtype)
    elif isinstance(wavelet, pywt.Wavelet):
        return create_wavelet_named_tuple(wavelet, dtype)
    else:
        raise ValueError("Invalid wavelet input.")


def _fold_axes(data: jnp.ndarray, keep_no: int) -> tuple[jnp.ndarray, list[int]]:
    """Fold unchanged leading dimensions into a single batch dimension.

    Args:
        data (jnp.ndarray): The input data array.
        keep_no (int): The number of dimensions to keep.

    Returns:
        Tuple[jnp.ndarray, list[int]]:
            The folded result array, and the shape of the original input.
    """
    dshape = list(data.shape)
    return jnp.reshape(data, [np.prod(dshape[:-keep_no])] + dshape[-keep_no:]), dshape


def _unfold_axes(data: jnp.ndarray, ds: list[int], keep_no: int) -> jnp.ndarray:
    """Unfold i.e. [batch*channel, height, widht] into [batch, channel, height, width]."""
    return jnp.reshape(data, ds[:-keep_no] + list(data.shape[-keep_no:]))


def _adjust_padding_at_reconstruction(
    res_size: int, coeff_size: int, pad_end: int, pad_start: int
) -> tuple[int, int]:
    pred_size = res_size - (pad_start + pad_end)
    next_size = coeff_size
    if next_size == pred_size:
        pass
    elif next_size == pred_size - 1:
        pad_end += 1
    else:
        raise AssertionError(
            "padding error, please check if dec as well as rec wavelets \
             and axes are identical."
        )
    return pad_end, pad_start


def _check_if_array(array: Any) -> Union[jnp.ndarray, np.ndarray[Any, Any]]:
    if not (isinstance(array, jnp.ndarray) or isinstance(array, np.ndarray)):
        raise ValueError(
            "First element of coeffs must be the approximation coefficient array."
        )
    return array


def _check_axes_argument(axes: list[int]) -> None:
    if len(set(axes)) != len(axes):
        raise ValueError("Cant transform the same axis twice.")


def _get_transpose_order(
    axes: list[int], data_shape: list[int]
) -> tuple[list[int], list[int]]:
    axes = list(map(lambda a: a + len(data_shape) if a < 0 else a, axes))
    all_axes = list(range(len(data_shape)))
    remove_transformed = list(filter(lambda a: a not in axes, all_axes))
    return remove_transformed, axes


def _swap_axes(data: jnp.ndarray, axes: list[int]) -> jnp.ndarray:
    _check_axes_argument(axes)
    front, back = _get_transpose_order(axes, list(data.shape))
    return jnp.transpose(data, front + back)


def _undo_swap_axes(data: jnp.ndarray, axes: list[int]) -> jnp.ndarray:
    _check_axes_argument(axes)
    front, back = _get_transpose_order(axes, list(data.shape))
    restore_sorted = jnp.argsort(jnp.array(front + back))
    return jnp.transpose(data, list(restore_sorted))


def create_wavelet_named_tuple(
    wavelet: Union[pywt.Wavelet, str], dtype: jnp.dtype[Any] = jnp.float64
) -> WaveletNamedTuple:
    """Create a WaveletNamedTuple from a pywt.Wavelet object.

    Conversion is required to take advante of JAX's JIT compilation.

    Args:
        wavelet (pywt.Wavelet): The pywt.Wavelet or wavelet-str
            to create the WaveletNamedTuple from. # noqa DAR003
        dtype (jnp.dtype, optional): The data type to use for the arrays in the WaveletNamedTuple.
        Defaults to jnp.float64.

    Returns:
        WaveletNamedTuple: A Jax-JIT-compatible WaveletNamedTuple.
    """
    if isinstance(wavelet, str):
        wavelet = pywt.Wavelet(wavelet)

    return WaveletNamedTuple(
        jnp.array(wavelet.dec_lo, dtype=dtype),
        jnp.array(wavelet.dec_hi, dtype=dtype),
        jnp.array(wavelet.rec_lo, dtype=dtype),
        jnp.array(wavelet.rec_hi, dtype=dtype),
        name=wavelet.name,
    )


def _get_filter_arrays(
    wavelet: WaveletNamedTuple, flip: bool
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract the filter coefficients from an input wavelet object.

    Args:
        wavelet (WaveletNamedTuple): A pywt-style input wavelet.
        flip (bool): If true flip the input coefficients.

    Returns:
        tuple: The dec_lo, dec_hi, rec_lo and rec_hi
            filter coefficients as jax arrays.
    """

    def _prepare_filter(filter: Union[list[float], jnp.ndarray]) -> jnp.ndarray:
        """Prepare the filters for further downstream processing."""
        if flip:
            if type(filter) is jnp.ndarray:
                return jnp.expand_dims(jnp.flip(filter), 0)
            else:
                return jnp.expand_dims(jnp.array(filter[::-1]), 0)
        else:
            if type(filter) is jnp.ndarray:
                return jnp.expand_dims(filter, 0)
            else:
                return jnp.expand_dims(jnp.array(filter), 0)

    dec_lo = _prepare_filter(wavelet.dec_lo)
    dec_hi = _prepare_filter(wavelet.dec_hi)
    rec_lo = _prepare_filter(wavelet.rec_lo)
    rec_hi = _prepare_filter(wavelet.rec_hi)
    return dec_lo, dec_hi, rec_lo, rec_hi
