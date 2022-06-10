"""Various utility functions."""

# -*- coding: utf-8 -*-
from collections import namedtuple
from typing import List, Tuple, Union

import jax.numpy as np

__all__ = ["flatten_2d_coeff_lst"]

Wavelet = namedtuple("Wavelet", ["dec_lo", "dec_hi", "rec_lo", "rec_hi"])


def flatten_2d_coeff_lst(
    coeff_list_2d: List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    flatten_arrays: bool = True,
) -> List[np.ndarray]:
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
