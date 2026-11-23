"""Gap-recovery ratio, two-level paired bootstrap, view-equivalence and seed-noise statistics."""
from __future__ import annotations

import numpy as np


def rho_psnr(reg, van, dense) -> float:
    gap = dense - van
    return float("nan") if abs(gap) < 1e-9 else float((reg - van) / gap)


def rho_lpips(reg, van, dense) -> float:
    gap = van - dense
    return float("nan") if abs(gap) < 1e-12 else float((van - reg) / gap)


def bootstrap(arms: dict, stat_fn, n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    """Two-level, paired bootstrap.

    arms: name -> array [S_k, V] of per-view values (seeds x views). All arms share the same V views.
    Each resample draws seeds with replacement *within each arm* (seeds are not paired across arms) and one set of
    view indices shared by all arms (views are paired). stat_fn receives {name: scalar mean}.
    """
    rng = np.random.default_rng(seed)
    mats = {k: np.atleast_2d(np.asarray(v, dtype=np.float64)) for k, v in arms.items()}
    v = next(iter(mats.values())).shape[1]
    assert all(m.shape[1] == v for m in mats.values()), "arms must be evaluated on the same views"
    out = np.empty(n_boot)
    for b in range(n_boot):
        vi = rng.integers(0, v, v)
        vals = {}
        for k, m in mats.items():
            si = rng.integers(0, m.shape[0], m.shape[0])
            vals[k] = m[np.ix_(si, vi)].mean()
        out[b] = stat_fn(vals)
    return out


def percentile_ci(samples, level: float = 0.95):
    s = np.asarray(samples, dtype=np.float64)
    s = s[~np.isnan(s)]
    if len(s) == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))


def rho_with_ci(reg, van, dense, kind: str = "psnr", n_boot: int = 10_000, seed: int = 0) -> dict:
    """rho and its 95% interval; numerator and denominator are resampled jointly, never combined from separate intervals."""
    f = rho_psnr if kind == "psnr" else rho_lpips
    point = f(np.mean(reg), np.mean(van), np.mean(dense))
    boots = bootstrap({"reg": reg, "van": van, "dense": dense}, lambda d: f(d["reg"], d["van"], d["dense"]), n_boot, seed)
    lo, hi = percentile_ci(boots)
    return {"rho": point, "lo": lo, "hi": hi}


def diff_with_ci(a, b, n_boot: int = 10_000, seed: int = 0) -> dict:
    """mean(a) - mean(b) with a paired two-level bootstrap interval."""
    point = float(np.mean(a) - np.mean(b))
    boots = bootstrap({"a": a, "b": b}, lambda d: d["a"] - d["b"], n_boot, seed)
    lo, hi = percentile_ci(boots)
    return {"diff": point, "lo": lo, "hi": hi}


def n_equivalent(target: float, ns, values):
    """View count at which vanilla NeRF reaches `target` PSNR, interpolated linearly in log N inside the measured range.
    Returns (value, status) with status in {'in', 'below', 'above'}; outside the range the nearest N is returned."""
    ns = np.asarray(ns, dtype=np.float64)
    order = np.argsort(ns)
    ns, vals = ns[order], np.maximum.accumulate(np.asarray(values, dtype=np.float64)[order])  # enforce monotone in N
    if target <= vals[0]:
        return float(ns[0]), "below"
    if target >= vals[-1]:
        return float(ns[-1]), "above"
    return float(np.exp(np.interp(target, vals, np.log(ns)))), "in"


def seed_noise(per_seed_means) -> dict:
    x = np.asarray(per_seed_means, dtype=np.float64)
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
    return {"mean": float(x.mean()), "sd": sd, "resolvable": 2 * sd}


def minimum_resolvable_difference(seed_sd: float, view_sd: float, n_seeds: int, n_views: int, z: float = 1.96) -> float:
    """Smallest arm difference the design can resolve (two arms, paired views): z * sqrt(2*seed_var/S + 2*view_var/V)."""
    return float(z * np.sqrt(2 * seed_sd ** 2 / max(n_seeds, 1) + 2 * view_sd ** 2 / max(n_views, 1)))
