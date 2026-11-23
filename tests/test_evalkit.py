import math
import tempfile
from pathlib import Path

import numpy as np

from evalkit import controls, diagnostics, metrics, stats
from evalkit.run_eval import evaluate_export
from freenerf_ns import datasets
from tests.helpers import make_fake_blender


def test_C8_metric_and_rho_identities():
    res = controls.c8_metric_identities()
    assert all(res.values()), res


def test_C9_diagnostic_identities():
    res = controls.c9_diagnostic_identities()
    assert all(res.values()), res


def test_psnr_foreground_mask_and_average():
    gt = np.zeros((8, 8, 3)); pred = gt.copy(); pred[:4] += 0.1
    mask = np.zeros((8, 8), bool); mask[:4] = True
    assert math.isclose(metrics.psnr(pred, gt, mask), 20.0, rel_tol=1e-9)           # error only inside the mask
    assert metrics.psnr(pred, gt) > metrics.psnr(pred, gt, mask)                     # whole-image PSNR is inflated by empty space
    assert math.isclose(metrics.average_metric(20.0, 0.75, 0.2), (0.01 * 0.5 * 0.2) ** (1 / 3), rel_tol=1e-9)


def test_ssim_drops_with_noise_and_blur_lowers_R_hi():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    base = gaussian_filter(rng.uniform(0, 1, (64, 64, 3)), (1, 1, 0)) * 0.5 + 0.25
    noisy = np.clip(base + rng.normal(0, 0.1, base.shape), 0, 1)
    assert metrics.ssim(base, noisy) < 0.9 and metrics.ssim(base, base) > 0.999
    blurred = gaussian_filter(base, (2.0, 2.0, 0))
    assert diagnostics.r_hi(blurred, base) < 1.0                                    # over-smoothing
    assert diagnostics.r_hi(noisy, base) > 1.0                                      # spurious high-frequency detail


def test_omega_and_background_opacity():
    w = np.zeros((10, 50)); w[:, 5] = 0.4; w[:, 30] = 0.6
    assert math.isclose(diagnostics.omega_M(w, 20), 0.4)
    acc = np.full((2, 4, 4), 0.2); alpha = np.ones((2, 4, 4)); alpha[:, :2] = 0
    assert math.isclose(diagnostics.background_opacity(acc, alpha), 0.2)
    assert math.isnan(diagnostics.background_opacity(acc, np.ones((2, 4, 4))))     # real scenes: no empty pixels


def test_rho_endpoints_and_undefined_gap():
    assert stats.rho_psnr(20, 20, 31) == 0 and stats.rho_psnr(31, 20, 31) == 1
    assert math.isclose(stats.rho_psnr(24.26, 13.93, 31.01), 0.6048, abs_tol=1e-3)   # the indicative published value in the proposal
    assert math.isnan(stats.rho_psnr(25, 20, 20))
    assert stats.rho_lpips(0.3, 0.3, 0.1) == 0 and stats.rho_lpips(0.1, 0.3, 0.1) == 1


def test_bootstrap_ci_covers_truth_and_widens_with_noise():
    rng = np.random.default_rng(1)
    van = 20 + rng.normal(0, 0.5, (3, 25)); reg = 26 + rng.normal(0, 0.5, (3, 25)); dense = 31 + rng.normal(0, 0.5, (1, 25))
    r = stats.rho_with_ci(reg, van, dense, "psnr", n_boot=600, seed=0)
    assert r["lo"] < r["rho"] < r["hi"] and abs(r["rho"] - 6 / 11) < 0.05 and r["hi"] - r["lo"] < 0.3
    noisy = stats.rho_with_ci(reg + rng.normal(0, 3, reg.shape), van, dense, "psnr", n_boot=600, seed=0)
    assert noisy["hi"] - noisy["lo"] > r["hi"] - r["lo"]
    d = stats.diff_with_ci(reg, van, n_boot=600)
    assert d["lo"] > 4 and abs(d["diff"] - 6) < 0.3


def test_bootstrap_is_paired_over_views():
    base = np.random.default_rng(0).normal(25, 5, 25)                               # large view-to-view spread
    a = np.stack([base + 2.0]); b = np.stack([base])
    d = stats.diff_with_ci(a, b, n_boot=400)
    assert d["hi"] - d["lo"] < 1e-9                                                  # perfect pairing cancels the view variance


def test_n_equivalent_interpolates_in_log_n():
    ns, vals = [4, 8, 16, 32, 100], [14.0, 20.0, 24.0, 27.0, 31.0]
    v, st = stats.n_equivalent(26.0, ns, vals)
    expected = math.exp(math.log(16) + (26 - 24) / (27 - 24) * (math.log(32) - math.log(16)))
    assert st == "in" and math.isclose(v, expected, rel_tol=1e-9)
    assert stats.n_equivalent(10.0, ns, vals) == (4.0, "below") and stats.n_equivalent(40.0, ns, vals) == (100.0, "above")
    assert stats.n_equivalent(26.0, ns, [14.0, 20.0, 19.0, 27.0, 31.0])[1] == "in"  # non-monotone sweep is made monotone


def test_seed_noise_and_resolvable_difference():
    s = stats.seed_noise([24.0, 24.4, 23.6])
    assert math.isclose(s["sd"], 0.4) and controls.c7_resolved(1.0, s["sd"]) and not controls.c7_resolved(0.5, s["sd"])
    assert stats.minimum_resolvable_difference(0.4, 1.0, 3, 25) > stats.minimum_resolvable_difference(0.4, 1.0, 3, 100)


def test_control_decisions():
    assert controls.c1_pass(11.0, 10.5) and not controls.c1_pass(20.0, 10.5)
    assert controls.c3_pass(31.2) and not controls.c3_pass(24.0)
    assert controls.c4_pass(31.5) and not controls.c4_pass(25.0)
    assert controls.c4_monotone({4: 14, 8: 20, 100: 31}) and not controls.c4_monotone({4: 14, 8: 20, 100: 19})


def test_evaluate_export_end_to_end_on_fake_scene():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        make_fake_blender(t / "raw", ("lego",), size=16)
        datasets.synthetic_manifests(t / "raw", t / "man", scenes=("lego",))
        man = __import__("json").loads((t / "man" / "lego_n8.json").read_text())
        from evalkit.io import load_gt
        gt, _ = load_gt(t / "raw", "lego", "test", man["test"])
        trgt, _ = load_gt(t / "raw", "lego", "train", man["train"])
        ex = t / "export"; ex.mkdir()
        rng = np.random.default_rng(0)
        np.save(ex / "test_rgb.npy", np.clip(gt + rng.normal(0, 0.05, gt.shape), 0, 1).astype(np.float32))
        np.save(ex / "test_acc.npy", np.full(gt.shape[:3], 0.1, np.float32))
        np.save(ex / "train_rgb.npy", trgt.astype(np.float32))              # perfect training views -> a generalisation gap
        w = np.zeros((64, 192), np.float32); w[:, 4] = 0.5
        np.save(ex / "train_rays_w.npy", w)
        r = evaluate_export(ex, t / "raw", "lego", man, M=20, lpips_nets=())
        assert len(r["test"]["psnr"]) == 25 and 20 < np.mean(r["test"]["psnr"]) < 35
        assert r["diag"]["G"] > 50                                           # train capped at 100 dB, test ~ 26 dB
        assert math.isclose(r["diag"]["omega_M"], 0.5)
        assert 0.09 < r["diag"]["B"] < 0.11 and len(r["ang_dist"]) == 25
        assert r["controls"]["C3_train_psnr_ge_30"] is True
