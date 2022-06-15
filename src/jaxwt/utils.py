"""Various utility functions."""

# -*- coding: utf-8 -*-
from collections import namedtuple
from typing import List, Tuple, Union

import jax.numpy as jnp
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
