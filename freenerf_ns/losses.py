"""Regularisers written with only operators shared by NumPy arrays and torch tensors,
so the exact same code is unit-tested on CPU/NumPy and used on GPU/torch."""
from __future__ import annotations


def occlusion_loss(density, m: int):
    """FreeNeRF occlusion term, Eq. (7): mean over rays of (1/K) * sum_{k<=M} sigma_k.

    density: [R, K] raw densities of the fine pass, samples sorted near -> far.
    """
    k = density.shape[-1]
    m = max(0, min(int(m), k))
    return (density[..., :m].sum(axis=-1) / k).mean()


def _min2(a, b):
    return (a + b - abs(a - b)) * 0.5


def depth_smoothness_loss(depth, acc=None):
    """RegNeRF depth smoothness, Eq. (9), on S x S patches.

    depth: [P, S, S]. If `acc` ([P, S, S], accumulated opacity) is given, each neighbour pair is weighted by the
    smaller of its two opacities (gradient-stopped): empty background pixels then do not fight the object
    silhouette. This is a documented adaptation for Blender objects on a white background (RegNeRF was designed
    for scenes with no empty pixels); pass acc=None for the literal paper form.
    """
    d00 = depth[:, :-1, :-1]
    dx = d00 - depth[:, :-1, 1:]
    dy = d00 - depth[:, 1:, :-1]
    if acc is None:
        sq = dx * dx + dy * dy
    else:
        a = acc.detach() if hasattr(acc, "detach") else acc
        a00 = a[:, :-1, :-1]
        wx = _min2(a00, a[:, :-1, 1:])
        wy = _min2(a00, a[:, 1:, :-1])
        sq = wx * dx * dx + wy * dy * dy
    return sq.sum(axis=-1).sum(axis=-1).mean()
