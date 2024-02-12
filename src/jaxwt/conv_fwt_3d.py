"""Three-dimensional transformation support."""

#
# Created on Fri Aug 4 2023
# Copyright (c) 2023 Moritz Wolter
#
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pywt

from .conv_fwt import _check_if_array, _get_filter_arrays
from .utils import (
    _adjust_padding_at_reconstruction,
    _as_wavelet,
    _check_axes_argument,
    _fold_axes,
    _swap_axes,
    _undo_swap_axes,
    _unfold_axes,
)


def wavedec3(
    data: jnp.ndarray,
    wavelet: Union[pywt.Wavelet, str],
    mode: str = "symmetric",
    level: Optional[int] = None,
    axes: Tuple[int, int, int] = (-3, -2, -1),
    precision: str = "highest",
) -> List[Union[jnp.ndarray, Dict[str, jnp.ndarray]]]:
    """Compute the three-dimensional wavelet analysis transform on the last three \
       dimensions of the input data array.

    Args:
        data (jnp.ndarray): Jax array containing the data to be transformed.
            A possible input shape would be [batch size, channels, height, width].
        wavelet (Union[pywt.Wavelet, str]): A wavelet object or str for the transformation.
        mode (str): The desired padding mode. Choose reflect, symmetric or zero.
            Defaults to symmetric.
        level (int): The max level to be used, if not set as many levels as possible
                               will be used. Defaults to None.
        axes (Tuple[int, int, int]): Compute the transform over these axes instead of the
            last three. Defaults to (-3, -2, -1).
        precision (str): For desired precision, choose "fastest", "high" or "highest".
            Defaults to "highest".

    Returns:
        list: A list with the lll coefficients and dictionaries
        with the filter order strings::

            ("aad", "ada", "add", "daa", "dad", "dda", "ddd")

        as keys. With a for the low pass or approximation filter and
        d for the high-pass or detail filter.

    Raises:
        ValueError: If the input has less than three dimensions.
        ValueError: If the axes tuple does not have three elements or contains a repetition.

    Examples:
        >>> import pywt
        >>> import jaxwt as jwt
        >>> import jax
        >>> data = jax.random.uniform(jax.random.PRNGKey(42),
        >>>                           [3, 16, 16, 16])
        >>> jwt.wavedec3(data, "haar", level=2)
    """
    ds = None
    wavelet = _as_wavelet(wavelet)

    if tuple(axes) != (-3, -2, -1):
        if len(axes) != 3:
            raise ValueError("3d transforms work with three axes.")
        else:
            data = _swap_axes(data, list(axes))

    if len(data.shape) == 3:
        data = jnp.expand_dims(data, 1)
    elif len(data.shape) >= 4:
        data, ds = _fold_axes(data, 3)
        data = jnp.expand_dims(data, 1)
    else:
        raise ValueError(
            "Wavedec3 needs at least  \
                         three input dimensions to work."
        )

    dec_lo, dec_hi, _, _ = _get_filter_arrays(wavelet, flip=True, dtype=data.dtype)
    dec_filt = _construct_3d_filt(lo=dec_lo, hi=dec_hi)

    if mode == "zero":
        # translate pywt to numpy.
        mode = "constant"

    if level is None:
        level = pywt.dwtn_max_level(
            [data.shape[-3], data.shape[-2], data.shape[-1]],
            pywt.Wavelet("MyWavelet", wavelet),
        )

    result_list: List[Union[jnp.ndarray, Dict[str, jnp.ndarray]]] = []
    res_lll = data
    for _ in range(level):
        if res_lll.ndim == 4:
            res_lll = jnp.expand_dims(res_lll, 1)
        res_lll = _fwt_pad3d(res_lll, len(wavelet), mode=mode)
        res = jax.lax.conv_general_dilated(
            lhs=res_lll,  # lhs = NCHw image tensor
            rhs=dec_filt,  # rhs = OIHw conv kernel tensor
            padding="VALID",
            window_strides=[2, 2, 2],
            dimension_numbers=("NCDHW", "OIDHW", "NCDHW"),
            precision=jax.lax.Precision(precision),
        )
        res_lll, res_llh, res_lhl, res_lhh, res_hll, res_hlh, res_hhl, res_hhh = [
            sr.squeeze(1) for sr in jnp.split(res, 8, 1)
        ]
        result_list.append(
            {
                "aad": res_llh,
                "ada": res_lhl,
                "add": res_lhh,
                "daa": res_hll,
                "dad": res_hlh,
                "dda": res_hhl,
                "ddd": res_hhh,
            }
        )
    result_list.append(res_lll)
    result_list.reverse()

    if ds:
        f_unfold_axes3 = partial(_unfold_axes, ds=ds, keep_no=3)
        result_list = jax.tree_util.tree_map(f_unfold_axes3, result_list)

    if tuple(axes) != (-3, -2, -1):
        to_tree = partial(_undo_swap_axes, axes=axes)
        result_list = jax.tree_util.tree_map(to_tree, result_list)

    return result_list


def waverec3(
    coeffs: List[Union[jnp.ndarray, Dict[str, jnp.ndarray]]],
    wavelet: Union[pywt.Wavelet, str],
    axes: Tuple[int, int, int] = (-3, -2, -1),
    precision: str = "highest",
) -> jnp.ndarray:
    """Compute a three-dimensional synthesis wavelet transform.

       Use it to reconstruct the original input image from the wavelet coefficients.

    Args:
        coeffs (list): The input coefficients, typically the output of wavedec3.
        wavelet (Union[pywt.Wavelet, str]): The wavelet we want.
        axes (Tuple[int, int, int]): Transform these axes instead of the
            last three. Defaults to (-3, -2, -1).
        precision (str): For desired precision, choose "fastest", "high" or "highest".
            Defaults to "highest".

    Returns:
        jnp.ndarray: Reconstruction of the original input data array.
            For example of shape [batch, channels, height, width].

    Raises:
        ValueError: If the axes list does not have three elements
            or contains a repetition.

    Example:
        >>> import pywt
        >>> import jaxwt as jwt
        >>> import jax
        >>> data = jax.random.uniform(jax.random.PRNGKey(42),
        >>>                           [3, 16, 16, 16])
        >>> rec = jwt.waverec3(jwt.wavedec3(data, "haar", level=2), "haar")
        >>> jax.numpy.allclose(data, rec)

    """
    wavelet = _as_wavelet(wavelet)

    if tuple(axes) != (-3, -2, -1):
        if len(axes) != 3:
            raise ValueError("3d transforms work with three axes.")
        else:
            _check_axes_argument(list(axes))
            to_tree = partial(_swap_axes, axes=list(axes))
            coeffs = jax.tree_util.tree_map(to_tree, coeffs)

    ds = None
    if _check_if_array(coeffs[0]).ndim > 4:
        ds = list(_check_if_array(coeffs[0]).shape)
        _fold_axes_keep3 = partial(_fold_axes, keep_no=3)
        _tree_fold = lambda array: _fold_axes_keep3(array)[0]  # noqa: E731
        coeffs = jax.tree_util.tree_map(_tree_fold, coeffs)

    _, _, rec_lo, rec_hi = _get_filter_arrays(
        wavelet, flip=True, dtype=_check_if_array(coeffs[0]).dtype
    )
    filt_len = rec_lo.shape[-1]
    rec_filt = _construct_3d_filt(lo=rec_lo, hi=rec_hi)
    rec_filt = jnp.transpose(rec_filt, [1, 0, 2, 3, 4])

    res_lll = jnp.expand_dims(_check_if_array(coeffs[0]), 1)

    coeff_dicts = coeffs[1:]
    for c_pos, coeff_dict in enumerate(coeffs[1:]):
        res_lll = jnp.concatenate(
            [
                res_lll,
                jnp.expand_dims(coeff_dict["aad"], 1),
                jnp.expand_dims(coeff_dict["ada"], 1),
                jnp.expand_dims(coeff_dict["add"], 1),
                jnp.expand_dims(coeff_dict["daa"], 1),
                jnp.expand_dims(coeff_dict["dad"], 1),
                jnp.expand_dims(coeff_dict["dda"], 1),
                jnp.expand_dims(coeff_dict["ddd"], 1),
            ],
            1,
        )
        res_lll = jax.lax.conv_transpose(
            lhs=res_lll,
            rhs=rec_filt,
            padding="VALID",
            strides=[2, 2, 2],
            dimension_numbers=("NCDHW", "OIDHW", "NCDHW"),
            precision=jax.lax.Precision(precision),
        )
        # remove the padding
        padl = (2 * filt_len - 3) // 2
        padr = (2 * filt_len - 3) // 2
        padt = (2 * filt_len - 3) // 2
        padb = (2 * filt_len - 3) // 2
        padfr = (2 * filt_len - 3) // 2
        padba = (2 * filt_len - 3) // 2

        if c_pos + 1 < len(coeff_dicts):
            padr, padl = _adjust_padding_at_reconstruction(
                res_lll.shape[-1], coeff_dicts[c_pos + 1]["aad"].shape[-1], padr, padl
            )
            padb, padt = _adjust_padding_at_reconstruction(
                res_lll.shape[-2], coeff_dicts[c_pos + 1]["aad"].shape[-2], padb, padt
            )
            padba, padfr = _adjust_padding_at_reconstruction(
                res_lll.shape[-3], coeff_dicts[c_pos + 1]["aad"].shape[-3], padba, padfr
            )
        # print('padding', padt, padb, padl, padr)
        if padt > 0:
            res_lll = res_lll[..., padt:, :]
        if padb > 0:
            res_lll = res_lll[..., :-padb, :]
        if padl > 0:
            res_lll = res_lll[..., padl:]
        if padr > 0:
            res_lll = res_lll[..., :-padr]
        if padfr > 0:
            res_lll = res_lll[..., padfr:, :, :]
        if padba > 0:
            res_lll = res_lll[..., :-padba, :, :]

    res_lll = res_lll.squeeze(1)

    if ds:
        res_lll = _unfold_axes(res_lll, ds, 3)
    if axes != (-3, -2, -1):
        res_lll = _undo_swap_axes(res_lll, list(axes))
    return res_lll


def _construct_3d_filt(lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
    """Construct three-dimensional filters using outer products.

    Args:
        lo (jnp.ndarray): Low-pass input filter.
        hi (jnp.ndarray): High-pass input filter

    Returns:
        jnp.ndarray: Stacked 3d filters of dimension::

        [8, 1, length, height, width].

        The four filters are ordered ll, lh, hl, hh.

    """
    dim_size = lo.shape[-1]
    size = [dim_size] * 3
    lll = jnp.reshape(jnp.outer(lo, jnp.outer(lo, lo)), size)
    llh = jnp.reshape(jnp.outer(lo, jnp.outer(lo, hi)), size)
    lhl = jnp.reshape(jnp.outer(lo, jnp.outer(hi, lo)), size)
    lhh = jnp.reshape(jnp.outer(lo, jnp.outer(hi, hi)), size)
    hll = jnp.reshape(jnp.outer(hi, jnp.outer(lo, lo)), size)
    hlh = jnp.reshape(jnp.outer(hi, jnp.outer(lo, hi)), size)
    hhl = jnp.reshape(jnp.outer(hi, jnp.outer(hi, lo)), size)
    hhh = jnp.reshape(jnp.outer(hi, jnp.outer(hi, hi)), size)
    filt = jnp.stack([lll, llh, lhl, lhh, hll, hlh, hhl, hhh], 0)
    filt = jnp.expand_dims(filt, 1)
    return filt


def _fwt_pad3d(
    data: jnp.ndarray, filt_len: int, mode: str = "symmetric"
) -> jnp.ndarray:
    padr = 0
    padl = 0
    padt = 0
    padb = 0
    padfr = 0
    padba = 0

    if filt_len > 2:
        # we pad half of the total requried padding on each side.
        padr += (2 * filt_len - 3) // 2
        padl += (2 * filt_len - 3) // 2
        padt += (2 * filt_len - 3) // 2
        padb += (2 * filt_len - 3) // 2
        padfr += (2 * filt_len - 3) // 2
        padba += (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data.shape[-1] % 2 != 0:
        padr += 1
    if data.shape[-2] % 2 != 0:
        padb += 1
    if data.shape[-3] % 2 != 0:
        padba += 1

    if mode == "zero":
        # translate pywt to numpy.
        mode = "constant"

    data = jnp.pad(
        data, ((0, 0), (0, 0), (padfr, padba), (padt, padb), (padl, padr)), mode
    )
    return data
