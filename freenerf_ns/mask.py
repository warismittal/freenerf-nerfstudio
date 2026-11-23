"""Frequency-annealing mask of FreeNeRF (paper Eq. 6), in pure NumPy.

Official FreeNeRF (and its DietNeRF-based Blender code) treats the encoding vector as consecutive groups
of `in_dim` entries in this order:

    group 0 : the raw input x
    group 1 : sin(band 0)      group 2 : cos(band 0)
    group 3 : sin(band 1)      group 4 : cos(band 1)   ...

A pointer moves through the groups as training advances:  ptr = n_groups * t / T + 1.
Groups left of int(ptr) are fully visible, the group at int(ptr) is visible with weight
(ptr - int(ptr)) (the "ramp"), everything to the right is hidden. After T (or slightly before, once
ptr reaches n_groups) the mask is all ones.

nerfstudio's NeRFEncoding emits [sin block | cos block | raw input] with a dimension-major layout inside
each block, so we do not index by position; we compute each column's *group* and look the weight up.
`check_layout` verifies the assumed layout against the real encoder at start-up.
"""
from __future__ import annotations

import math

import numpy as np


def band_groups(in_dim: int, num_freq: int, include_input: bool = True, reverse: bool = False):
    """Group index (0 .. 2*num_freq) for every output column of nerfstudio's NeRFEncoding.

    Returns (groups[int64, in_dim*(2*num_freq+1)], n_groups). `reverse=True` opens the finest
    band first (arm X1); the raw input is always group 0.
    """
    if not include_input:
        raise ValueError("FreeNeRF masking requires include_input=True (raw coordinates are group 0).")
    block = in_dim * num_freq
    f = np.arange(block) % num_freq          # column c of a block belongs to band c % F (dim-major layout)
    if reverse:
        f = num_freq - 1 - f
    groups = np.concatenate([1 + 2 * f, 2 + 2 * f, np.zeros(in_dim, dtype=np.int64)])
    return groups.astype(np.int64), 2 * num_freq + 1


def visible_pointer(step: int, end_step: int, n_groups: int) -> float:
    if end_step <= 0 or step >= end_step:
        return float(n_groups)
    return min(n_groups * float(step) / float(end_step) + 1.0, float(n_groups))


def group_weights(step: int, end_step: int, n_groups: int, mode: str = "ramp") -> np.ndarray:
    """Weight in [0,1] for each group. mode: 'none' | 'ramp' (paper Eq. 6) | 'step' (hard-step form)."""
    if mode == "none":
        return np.ones(n_groups, dtype=np.float32)
    if mode not in ("ramp", "step"):
        raise ValueError(f"unknown mask mode {mode!r}")
    ptr = visible_pointer(step, end_step, n_groups)
    ip = int(math.floor(ptr))
    g = np.arange(n_groups)
    w = (g < ip).astype(np.float32)
    if mode == "ramp":
        w = w + (g == ip).astype(np.float32) * np.float32(ptr - ip)
    return np.clip(w, 0.0, 1.0).astype(np.float32)


def column_weights(groups: np.ndarray, n_groups: int, step: int, end_step: int, mode: str = "ramp") -> np.ndarray:
    return group_weights(step, end_step, n_groups, mode)[groups]


def check_layout(encode, in_dim: int, num_freq: int, include_input: bool = True) -> None:
    """Verify that `encode` (batch,in_dim -> batch,D array-like) has the layout `band_groups` assumes.

    Probes: x = 0 must give [0...0 | 1...1 | 0...0] (sin block, cos block, raw);
    x = (eps,0,0) must excite only dimension 0 of the sin block, with magnitude growing with the band.
    Raises AssertionError with a clear message otherwise. Cheap; run once at model construction.
    """
    D = in_dim * num_freq
    eps = 1e-4
    probe = np.zeros((2, in_dim), dtype=np.float32)
    probe[1, 0] = eps
    out = np.asarray(encode(probe), dtype=np.float64)
    expected = 2 * D + (in_dim if include_input else 0)
    assert out.shape[-1] == expected, f"encoding width {out.shape[-1]} != expected {expected}"
    zero, e = out[0], out[1]
    assert np.abs(zero[:D]).max() < 1e-6, "sin block is not first: layout differs from [sin | cos | raw]"
    assert np.abs(zero[D:2 * D] - 1.0).max() < 1e-6, "cos block is not second: layout differs from [sin | cos | raw]"
    if include_input:
        assert np.abs(zero[2 * D:]).max() < 1e-6, "raw input is not last: layout differs from [sin | cos | raw]"
        assert np.allclose(e[2 * D:], probe[1], atol=1e-7), "raw input columns are not the input"
    s0 = e[:num_freq]
    assert s0[0] > 0 and np.all(np.diff(s0) > 0), "sin block is not dimension-major with increasing frequency"
    assert np.abs(e[num_freq:D]).max() < 1e-6, "sin block is not dimension-major (other dims responded)"
