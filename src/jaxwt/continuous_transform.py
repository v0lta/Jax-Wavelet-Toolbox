"""Jax compatible cwt code."""
from typing import Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax.numpy.fft import fft, ifft
from pywt import ContinuousWavelet, DiscreteContinuousWavelet, Wavelet
from pywt._functions import integrate_wavelet, scale2frequency


def _next_fast_len(n: int) -> int:
    """Round up size to the nearest power of two.

    Given a number of samples `n`, returns the next power of two
    following this number to take advantage of FFT speedup.
    This fallback is less efficient than `scipy.fftpack.next_fast_len`
    """
    return int(2 ** jnp.ceil(jnp.log2(n)))


def cwt(
    data: jnp.ndarray,
    scales: Union[np.ndarray, jnp.ndarray],  # type: ignore
    wavelet: Union[ContinuousWavelet, str],
    sampling_period: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the single dimensional continuous wavelet transform.

    This function is a jax port of pywt.cwt as found at:
    https://github.com/PyWavelets/pywt/blob/master/pywt/_cwt.py

    Args:
        data (jnp.ndarray): The ijnput tensor of shape [batch_size, time].
        scales (np.ndarray or jnp.array):
            The wavelet scales to use. One can use
            ``f = pywt.scale2frequency(wavelet, scale)/sampling_period`` to determine
            what physical frequency, ``f``. Here, ``f`` is in hertz when the
            ``sampling_period`` is given in seconds.
            wavelet (str or Wavelet of ContinuousWavelet): The wavelet to work with.
        wavelet (ContinuousWavelet or str): The continuous wavelet to work with.
        sampling_period (float): Sampling period for the frequencies output (optional).
            The values computed for ``coefs`` are independent of the choice of
            ``sampling_period`` (i.e. ``scales`` is not scaled by the sampling
            period).

    Raises:
        ValueError: If a scale is too small for the input signal.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple with the transformation matrix
            and frequencies in this order.

    Example:
        >>> import jaxwt as jwt
        >>> import jax.numpy as jnp
        >>> import scipy.signal as signal
        >>> t = jnp.linspace(-2, 2, 800, endpoint=False)
        >>> sig = signal.chirp(t, f0=1, f1=12, t1=2, method="linear")
        >>> widths = jnp.arange(1, 31)
        >>> cwtmatr, freqs = jwt.cwt(
        >>>     jnp.array(sig), widths, "mexh",
        >>>     sampling_period=(4 / 800) * jnp.pi
        >>> )
    """
    # accept array_like ijnput; make a copy to ensure a contiguous array
    if not isinstance(wavelet, (ContinuousWavelet, Wavelet)):
        wavelet = DiscreteContinuousWavelet(wavelet)
    if type(scales) is jnp.ndarray:
        scales = np.array(scales)
    elif jnp.isscalar(scales):
        scales = jnp.array([scales])
    # if not jnp.isscalar(axis):
    #    raise jnp.AxisError("axis must be a scalar.")

    precision = 10
    int_psi, x = integrate_wavelet(wavelet, precision=precision)
    if type(wavelet) is ContinuousWavelet:
        int_psi = jnp.conj(int_psi) if wavelet.complex_cwt else int_psi
    int_psi = jnp.array(int_psi)

    # convert int_psi, x to the same precision as the data
    # x = jnp.asarray(x, dtype=data.cpu().numpy().real.dtype)
    x = jnp.array(x)

    size_scale0 = -1
    fft_data = None

    out = []
    for scale in scales:
        step = x[1] - x[0]
        j = jnp.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
        j = j.astype(int)  # floor
        if j[-1] >= len(int_psi):
            j = jnp.extract(j < len(int_psi), j)
        int_psi_scale = jnp.flip(int_psi[j], 0)

        # The padding is selected for:
        # - optimal FFT complexity
        # - to be larger than the two signals length to avoid circular
        #   convolution
        size_scale = _next_fast_len(data.shape[-1] + len(int_psi_scale) - 1)
        if size_scale != size_scale0:
            # Must recompute fft_data when the padding size changes.
            fft_data = fft(data, size_scale, axis=-1)
        size_scale0 = size_scale
        fft_wav = fft(int_psi_scale, size_scale, axis=-1)
        conv = ifft(fft_wav * fft_data, axis=-1)
        conv = conv[..., : data.shape[-1] + len(int_psi_scale) - 1]

        coef = -jnp.sqrt(scale) * jnp.diff(conv, axis=-1)

        # transform axis is always -1
        d = (coef.shape[-1] - data.shape[-1]) / 2.0
        if d > 0:
            coef = coef[..., int(jnp.floor(d)) : -int(jnp.ceil(d))]
        elif d < 0:
            raise ValueError("Selected scale of {} too small.".format(scale))

        out.append(coef)
    out_tensor = jnp.stack(out)
    if type(wavelet) is Wavelet:
        out_tensor = out_tensor.real
    else:
        out_tensor = out_tensor if wavelet.complex_cwt else out_tensor.real

    frequencies = scale2frequency(wavelet, scales, precision)
    if jnp.isscalar(frequencies):
        frequencies = jnp.array([frequencies])
    frequencies /= sampling_period
    return out_tensor, frequencies
