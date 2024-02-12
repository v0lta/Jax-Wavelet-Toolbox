"""Various utility functions."""

#
# Copyright (c) 2023 Moritz Wolter
#
from collections import namedtuple
from typing import Any, List, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pywt

__all__ = ["flatten_2d_coeff_lst"]

Wavelet = namedtuple("Wavelet", ["dec_lo", "dec_hi", "rec_lo", "rec_hi"])


def flatten_2d_coeff_lst(
    coeff_list_2d: List[
        Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
    ],
    flatten_arrays: bool = True,
) -> List[jnp.ndarray]:
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


def _as_wavelet(wavelet: Union[Wavelet, str]) -> pywt.Wavelet:
    """Ensure the input argument to be a pywt wavelet compatible object.

    Args:
        wavelet (Wavelet or str): The input argument, which is either a
            pywt wavelet compatible object or a valid pywt wavelet name string.

    Returns:
        pywt.Wavelet: the input wavelet object or the pywt wavelet object
            described by the input str.
    """
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet


def _fold_axes(data: jnp.ndarray, keep_no: int) -> Tuple[jnp.ndarray, List[int]]:
    """Fold unchanged leading dimensions into a single batch dimension.

    Args:
        data (jnp.ndarray): The input data array.
        keep_no (int): The number of dimensions to keep.

    Returns:
        Tuple[jnp.ndarray, List[int]]:
            The folded result array, and the shape of the original input.
    """
    dshape = list(data.shape)
    return jnp.reshape(data, [np.prod(dshape[:-keep_no])] + dshape[-keep_no:]), dshape


def _unfold_axes(data: jnp.ndarray, ds: List[int], keep_no: int) -> jnp.ndarray:
    """Unfold i.e. [batch*channel, height, widht] into [batch, channel, height, width]."""
    return jnp.reshape(data, ds[:-keep_no] + list(data.shape[-keep_no:]))


def _adjust_padding_at_reconstruction(
    res_size: int, coeff_size: int, pad_end: int, pad_start: int
) -> Tuple[int, int]:
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


def _check_if_array(array: Any) -> jnp.ndarray:
    if not isinstance(array, jnp.ndarray):
        raise ValueError(
            "First element of coeffs must be the approximation coefficient tensor."
        )
    return array


def _check_axes_argument(axes: List[int]) -> None:
    if len(set(axes)) != len(axes):
        raise ValueError("Cant transform the same axis twice.")


def _get_transpose_order(
    axes: List[int], data_shape: List[int]
) -> Tuple[List[int], List[int]]:
    axes = list(map(lambda a: a + len(data_shape) if a < 0 else a, axes))
    all_axes = list(range(len(data_shape)))
    remove_transformed = list(filter(lambda a: a not in axes, all_axes))
    return remove_transformed, axes


def _swap_axes(data: jnp.ndarray, axes: List[int]) -> jnp.ndarray:
    _check_axes_argument(axes)
    front, back = _get_transpose_order(axes, list(data.shape))
    return jnp.transpose(data, front + back)


def _undo_swap_axes(data: jnp.ndarray, axes: List[int]) -> jnp.ndarray:
    _check_axes_argument(axes)
    front, back = _get_transpose_order(axes, list(data.shape))
    restore_sorted = jnp.argsort(jnp.array(front + back))
    return jnp.transpose(data, list(restore_sorted))
