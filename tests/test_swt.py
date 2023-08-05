"""Test the stationary wavelet transformation code."""
#
# Created on Fri Aug 04 2023
# Copyright (c) 2023 Moritz Wolter
#

import jax
import jax.numpy as jnp
import pytest
import pywt
from jax.config import config

from src.jaxwt.conv_fwt import _get_filter_arrays
from src.jaxwt.stationary_transform import _conv_transpose_dedilate, iswt, swt
from src.jaxwt.utils import _as_wavelet

config.update("jax_enable_x64", True)


@pytest.mark.slow
@pytest.mark.parametrize("size", [[1, 32], [3, 2, 32], [3, 2, 1, 32]])
@pytest.mark.parametrize("wavelet", ["db1", "db2", "db3", "sym4"])
@pytest.mark.parametrize("level", [1, 2, None])
def test_swt_1d(level, size, wavelet):
    """Test the 1d swt."""
    key = jax.random.PRNGKey(42)
    signal = jax.random.randint(key, size, 0, 9).astype(jnp.float64)
    jaxwt_coeff = swt(signal, wavelet, level=level)
    pywt_coeff = pywt.swt(signal, wavelet, level, trim_approx=True, norm=False)
    test_list = []
    for a, b in zip(jaxwt_coeff, pywt_coeff):
        test_list.extend([jnp.allclose(ael, bel) for ael, bel in zip(a, b)])
    assert all(test_list)

    rec = iswt(jaxwt_coeff, wavelet)
    assert jnp.allclose(rec, signal)


@pytest.mark.parametrize("axis", [-1, 0, 1, 2])
def test_axis_arg(axis):
    """Test swt axis argument support."""
    key = jax.random.PRNGKey(41)
    signal = jax.random.randint(key, [32, 32, 32], 0, 9).astype(jnp.float64)
    jaxwt_coeff = swt(signal, "haar", level=2, axis=axis)
    pywt_coeff = pywt.swt(signal, "haar", 2, trim_approx=True, norm=False, axis=axis)
    test_list = []
    for a, b in zip(jaxwt_coeff, pywt_coeff):
        test_list.extend([jnp.allclose(ael, bel) for ael, bel in zip(a, b)])
    assert all(test_list)

    rec = iswt(jaxwt_coeff, "haar", axis=axis)
    assert jnp.allclose(rec, signal)


@pytest.mark.parametrize("wavelet_str", ["db1", "db2", "db3", "sym4"])
def test_inverse_dilation(wavelet_str):
    """Test transposed dilated convolutions."""
    precision = "highest"
    level = 2
    dilation = 2**level
    length = 32
    print(wavelet_str, dilation)

    wavelet = _as_wavelet(wavelet_str)
    data = jnp.expand_dims(jnp.arange(length).astype(jnp.float64), (0, 1))
    data = jnp.concatenate([data, data + 1], 0)

    dec_lo, dec_hi, _, _ = _get_filter_arrays(wavelet, flip=True, dtype=data.dtype)
    filt_len = dec_lo.shape[-1]
    filt = jnp.stack([dec_lo, dec_hi], 0)

    padl, padr = dilation * (filt_len // 2 - 1), dilation * (filt_len // 2)
    datap = jnp.pad(data, [(0, 0)] * (data.ndim - 1) + [(padl, padr)], mode="wrap")
    conv = jax.lax.conv_general_dilated(
        lhs=datap,  # lhs = NCH image tensor
        rhs=filt,  # rhs = OIH conv kernel tensor
        padding=[(0, 0)],
        window_strides=[1],
        rhs_dilation=[dilation],
        dimension_numbers=("NCT", "OIT", "NCT"),
        precision=jax.lax.Precision(precision),
    )

    # unlike pytorch lax's transpose conv requires filter flips.
    _, _, rec_lo, rec_hi = _get_filter_arrays(wavelet, flip=True, dtype=data.dtype)
    filt_len = rec_lo.shape[-1]
    rec_filt = jnp.stack([rec_lo, rec_hi], 1)
    padl, padr = dilation * (filt_len // 2), dilation * (filt_len // 2 - 1)
    conv = jnp.pad(conv, [(0, 0)] * (data.ndim - 1) + [(padl, padr)], mode="wrap")

    rec = _conv_transpose_dedilate(conv, rec_filt, dilation, length, "highest")

    assert jnp.allclose(rec, data)
