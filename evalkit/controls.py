"""Negative controls and known-answer checks (proposal Section 8.4). Pure NumPy; each returns a plain verdict."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import diagnostics, metrics, stats
from .io import load_json, sha256_file


def render_weights(sigma, delta):
    """Eq. (2): w_i = T_i (1 - exp(-sigma_i delta_i)), T_i = exp(-sum_{j<i} sigma_j delta_j). sigma, delta: [R,S]."""
    a = np.asarray(sigma, dtype=np.float64) * np.asarray(delta, dtype=np.float64)
    T = np.exp(-(np.cumsum(a, axis=-1) - a))
    return T * (1.0 - np.exp(-a))


# --- C1 pose permutation ------------------------------------------------------------------------------------------
def c1_pass(test_psnr: float, const_bg_psnr: float, tol: float = 2.0) -> bool:
    """With the image-pose pairing shuffled, test PSNR must collapse to within `tol` dB of a constant-background image."""
    return bool(test_psnr <= const_bg_psnr + tol)


# --- C3 overfit ---------------------------------------------------------------------------------------------------
def c3_pass(train_psnr: float, minimum: float = 30.0) -> bool:
    return bool(train_psnr >= minimum)


# --- C4 dense sanity ----------------------------------------------------------------------------------------------
def c4_pass(dense_psnr: float, reference: float = 32.54, tol: float = 2.0) -> bool:
    return bool(abs(dense_psnr - reference) <= tol)


def c4_monotone(psnr_by_n: dict) -> bool:
    """PSNR of vanilla NeRF must not decrease with the number of views."""
    ns = sorted(psnr_by_n)
    return all(psnr_by_n[a] <= psnr_by_n[b] + 1e-9 for a, b in zip(ns, ns[1:]))


# --- C5 leakage ---------------------------------------------------------------------------------------------------
def verify_hashes(manifest_dir) -> list:
    """Compare every manifest against manifests/HASHES.txt; returns the list of problems (empty = ok)."""
    d = Path(manifest_dir)
    hashes = d / "HASHES.txt"
    if not hashes.exists():
        return ["HASHES.txt missing"]
    problems = []
    for line in hashes.read_text().splitlines():
        if not line.strip():
            continue
        name, h = line.split()
        p = d / name
        if not p.exists():
            problems.append(f"{name} missing")
        elif sha256_file(p) != h:
            problems.append(f"{name} changed")
    return problems


def check_manifest(m: dict, n_train_total: int = 100, n_test_total: int = 200) -> list:
    """Structural checks of one manifest. (Train and test indices refer to different folders, so numeric overlap is not
    leakage; leakage is prevented physically by check_sparse_dir.)"""
    problems = []
    tr, te = m["train"], m["test"]
    if len(set(tr)) != len(tr):
        problems.append("duplicate training indices")
    if len(set(te)) != len(te):
        problems.append("duplicate test indices")
    if any(not (0 <= i < n_train_total) for i in tr):
        problems.append("training index out of range")
    if any(not (0 <= i < n_test_total) for i in te):
        problems.append("test index out of range")
    return problems


def check_sparse_dir(sparse_dir, manifest: dict) -> list:
    """C5 for the physical isolation design: the trainer's dataset folder must contain exactly the manifest's training
    images, no `test/` directory, and no image outside `train/`. Returns the list of problems (empty = ok)."""
    d = Path(sparse_dir)
    problems = []
    if (d / "test").exists():
        problems.append("a test/ directory exists in the training dataset")
    files = sorted(p.name for p in (d / "train").glob("*.png")) if (d / "train").exists() else []
    expected = sorted(f"r_{i}.png" for i in manifest["train"])
    if files != expected:
        problems.append(f"train/ holds {len(files)} images, manifest lists {len(expected)}")
    extra = [p.name for p in d.rglob("*.png") if p.parent.name != "train"]
    if extra:
        problems.append(f"images outside train/: {extra[:3]}")
    for split in ("train", "val", "test"):
        tj = d / f"transforms_{split}.json"
        if tj.exists():
            frames = load_json(tj)["frames"]
            if any(not f["file_path"].lstrip("./").startswith("train/") for f in frames):
                problems.append(f"transforms_{split}.json references images outside train/")
    return problems


# --- C7 seed noise ------------------------------------------------------------------------------------------------
def c7_resolved(gain: float, seed_sd: float) -> bool:
    """A gain below twice the seed standard deviation is 'unresolved'."""
    return bool(gain >= 2 * seed_sd)


# --- C8 metric identities ----------------------------------------------------------------------------------------
def c8_metric_identities(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 0.9, (64, 64, 3))
    out = {}
    delta = 0.05
    out["psnr_exact"] = math.isclose(metrics.psnr(x, x + delta), -20 * math.log10(delta), rel_tol=1e-12)
    out["ssim_identity"] = math.isclose(metrics.ssim(x, x), 1.0, abs_tol=1e-9)
    out["psnr_identity_inf"] = metrics.psnr(x, x) == float("inf")
    out["rho_zero_for_vanilla"] = abs(stats.rho_psnr(25.0, 25.0, 31.0)) < 1e-12
    out["rho_one_for_dense"] = abs(stats.rho_psnr(31.0, 25.0, 31.0) - 1.0) < 1e-12
    return out


# --- C9 diagnostic identities ------------------------------------------------------------------------------------
def c9_diagnostic_identities(m: int = 20, k: int = 192, rays: int = 16) -> dict:
    delta = np.full((rays, k), 4.0 / k)
    billboard = np.zeros((rays, k))
    billboard[:, 3] = 1e4                      # opaque textured plane at sample 3: the billboard solution of Proposition 1
    out = {"billboard_omega_near_1": abs(diagnostics.omega_M(render_weights(billboard, delta), m) - 1.0) < 1e-6}
    empty_w = render_weights(np.zeros((rays, k)), delta)
    out["empty_omega_zero"] = diagnostics.omega_M(empty_w, m) == 0.0
    out["empty_background_opacity_zero"] = diagnostics.background_opacity(np.zeros((4, 8, 8)), np.zeros((4, 8, 8))) == 0.0
    img = np.random.default_rng(0).uniform(0, 1, (64, 64, 3))
    out["r_hi_identity_one"] = abs(diagnostics.r_hi(img, img) - 1.0) < 1e-9
    return out
