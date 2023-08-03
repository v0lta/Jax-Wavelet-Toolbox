"""Test the stationary wavelet transformation code."""

import jax
import jax.numpy as jnp
import pytest
import pywt
from jax.config import config

from src.jaxwt._stationary_transform import _swt  # _iswt
from src.jaxwt.conv_fwt import _get_filter_arrays
from src.jaxwt.utils import _as_wavelet

config.update("jax_enable_x64", True)


@pytest.mark.slow
@pytest.mark.parametrize("size", [32])
@pytest.mark.parametrize(
    "wavelet", ["db1", "db2", "db3"]
)  # TODO: explore lonnger wavelets.
@pytest.mark.parametrize("level", [1, 2, None])
def test_swt_1d(level, size, wavelet):
    """Test the 1d swt."""
    signal = jnp.expand_dims(jnp.arange(size).astype(jnp.float64), 0)
    ptwt_coeff = _swt(signal, wavelet, level=level)
    pywt_coeff = pywt.swt(signal, wavelet, level, trim_approx=True, norm=False)
    test_list = []
    for a, b in zip(ptwt_coeff, pywt_coeff):
        test_list.extend([jnp.allclose(ael, bel) for ael, bel in zip(a, b)])
    assert all(test_list)

    # rec = _iswt(ptwt_coeff, wavelet)
    # assert jnp.allclose(rec, signal)
    # pass


@pytest.mark.parametrize("wavelet_str", ["db1", "db2", "db3"])
def test_inverse_dilation(wavelet_str):
    """Test transposed dilated convolutions."""
    precision = "highest"
    level = 2
    dilation = 2**level
    length = 32
    print(wavelet_str, dilation)

    wavelet = _as_wavelet(wavelet_str)
    data = jnp.expand_dims(jnp.arange(length).astype(jnp.float64), (0, 1))

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

    recs = []
    for fl in range(length):
        to_conv_t = conv[..., fl : (fl + dilation * filt_len) : dilation]
        rec = jax.lax.conv_transpose(
            lhs=to_conv_t,
            rhs=rec_filt,
            padding=[(0, 0)],
            strides=[1],
            dimension_numbers=("NCH", "OIH", "NCH"),
            precision=jax.lax.Precision(precision),
        )
        recs.append(rec / 2.0)
    print(" ")
    rec = jnp.concatenate(recs, -1)
    print(rec)

    assert jnp.allclose(rec, data)
