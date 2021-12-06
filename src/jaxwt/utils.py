"""Various utility functions."""

# -*- coding: utf-8 -*-

from collections import namedtuple

__all__ = ["flatten_2d_coeff_lst"]

Wavelet = namedtuple("Wavelet", ["dec_lo", "dec_hi", "rec_lo", "rec_hi"])


def flatten_2d_coeff_lst(coeff_list_2d: list, flatten_arrays: bool = True) -> list:
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
        if type(coeff) is tuple:
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
