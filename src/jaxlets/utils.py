# -*- coding: utf-8 -*-

from collections import namedtuple

__all__ = [
    'JaxWavelet',
    'flatten_2d_coeff_lst',
]

JaxWavelet = namedtuple('JaxWavelet', ['dec_lo', 'dec_hi', 'rec_lo', 'rec_hi'])


def flatten_2d_coeff_lst(coeff_lst_2d, flatten_arrays=True):
    flat_coeff_lst = []
    for coeff in coeff_lst_2d:
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
