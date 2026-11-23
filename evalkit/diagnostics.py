"""Failure-signature diagnostics (proposal Section 8.2): G, Omega_M, B, R_hi."""
from __future__ import annotations

import numpy as np


def generalisation_gap(psnr_train: float, psnr_test: float) -> float:
    return float(psnr_train - psnr_test)


def omega_M(weights, m: int) -> float:
    """Mean opacity inside the first M samples of training rays: floater mass next to the training cameras.
    weights: [R, S] fine-pass weights sorted near -> far."""
    w = np.asarray(weights, dtype=np.float64)
    return float(w[:, :m].sum(axis=1).mean())


def background_opacity(acc, alpha_gt) -> float:
    """B: mean accumulated opacity where the ground truth is empty (floaters seen from novel views; synthetic only)."""
    acc = np.asarray(acc, dtype=np.float64)
    empty = np.asarray(alpha_gt) <= 0
    if not empty.any():
        return float("nan")
    return float(acc[empty].mean())


def radial_power(img2d) -> np.ndarray:
    """Radially averaged power spectrum; index = radius in cycles per image (Nyquist = H/2)."""
    a = np.asarray(img2d, dtype=np.float64)
    a = a - a.mean()
    f = np.fft.fftshift(np.fft.fft2(a))
    p = f.real ** 2 + f.imag ** 2
    h, w = p.shape
    yy, xx = np.indices(p.shape)
    r = np.rint(np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)).astype(int)
    tot = np.bincount(r.ravel(), p.ravel())
    cnt = np.bincount(r.ravel())
    return tot / np.maximum(cnt, 1)


def r_hi(pred, gt, frac: float = 0.25, eps: float = 1e-12) -> float:
    """Geometric-mean ratio of rendered to true power above `frac` of the Nyquist frequency, averaged over views.
    < 1: over-smoothing, > 1: spurious detail. pred, gt: [V,H,W,3] or [H,W,3]."""
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if pred.ndim == 3:
        pred, gt = pred[None], gt[None]
    vals = []
    for p, g in zip(pred, gt):
        pp, pg = radial_power(p.mean(axis=-1)), radial_power(g.mean(axis=-1))
        nyq = min(p.shape[0], p.shape[1]) // 2
        lo = int(np.ceil(frac * nyq))
        k = np.arange(lo, nyq + 1)
        vals.append(np.exp(np.mean(np.log((pp[k] + eps) / (pg[k] + eps)))))
    return float(np.mean(vals))
