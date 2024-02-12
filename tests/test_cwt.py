"""Test the continuous transformation code."""

#
# Copyright (c) 2023 Moritz Wolter
#
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pywt
from scipy import signal

jax.config.update("jax_enable_x64", True)

from src.jaxwt.continuous_transform import cwt

continuous_wavelets = [
    "cgau1",
    "cgau2",
    "cgau3",
    "cgau4",
    "cgau5",
    "cgau6",
    "cgau7",
    "cgau8",
    "gaus1",
    "gaus2",
    "gaus3",
    "gaus4",
    "gaus5",
    "gaus6",
    "gaus7",
    "gaus8",
    "mexh",
    "morl",
]


@pytest.mark.parametrize("scales", [np.arange(1, 16), 5.0, np.arange(1, 15)])
@pytest.mark.parametrize("samples", [31, 32])
@pytest.mark.parametrize("wavelet", continuous_wavelets)
def test_cwt(
    wavelet: str, samples: int, scales: Union[np.ndarray, np.ndarray, float]
) -> None:
    """Test the cwt implementation for various wavelets."""
    t = np.linspace(-1, 1, samples, endpoint=False)
    sig = signal.chirp(t, f0=1, f1=50, t1=10, method="linear").astype(np.float64)
    cwtmatr, freqs = pywt.cwt(data=sig, scales=scales, wavelet=wavelet)
    cwtmatr_jax, freqs_jax = cwt(jnp.array(sig), jnp.array(scales), wavelet)
    assert jnp.allclose(cwtmatr_jax, cwtmatr)
    assert jnp.allclose(freqs, freqs_jax)


@pytest.mark.parametrize("wavelet", continuous_wavelets)
def test_cwt_batched(wavelet):
    """Test batched transforms."""
    sig = np.random.randn(10, 200)
    widths = np.arange(1, 30)
    cwtmatr, freqs = pywt.cwt(sig, widths, wavelet)
    cwtmatr_jax, freqs_jax = cwt(jnp.array(sig), widths, wavelet)
    assert jnp.allclose(cwtmatr_jax, cwtmatr)
    assert jnp.allclose(freqs, freqs_jax)
