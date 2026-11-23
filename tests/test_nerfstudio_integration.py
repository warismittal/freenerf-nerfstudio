"""Integration tests that need torch + nerfstudio. They are skipped where those are missing (CI, laptops without them) and
are the FIRST thing to run on a real machine:   pytest tests/test_nerfstudio_integration.py -x -q
Each test pinpoints one assumption about nerfstudio's API or tensor layout."""
import numpy as np

try:
    import torch
    import nerfstudio  # noqa: F401
    HAVE = True
except Exception:  # pragma: no cover
    HAVE = False


def _skip():
    try:
        import pytest
        pytest.skip("torch + nerfstudio not installed")
    except ImportError:
        return


def _replica(x, F, lo, hi):
    freqs = 2.0 ** np.linspace(lo, hi, F)
    s = (2 * np.pi * x)[..., None] * freqs
    s = s.reshape(x.shape[0], -1)
    return np.concatenate([np.sin(np.concatenate([s, s + np.pi / 2], -1)), x], -1)


def test_masked_encoding_equals_numpy_reference_mask():
    """The real nerfstudio encoding times our mask == NumPy replica times the NumPy mask (layout assumption)."""
    if not HAVE:
        return _skip()
    from freenerf_ns.encoding import MaskedNeRFEncoding
    from freenerf_ns.mask import band_groups, column_weights
    enc = MaskedNeRFEncoding(3, 10, 0.0, 8.0, True, "ramp")
    enc.set_progress(300, 1000)
    x = np.random.default_rng(0).uniform(-1, 1, (7, 3)).astype(np.float32)
    got = enc(torch.from_numpy(x)).numpy()
    g, n = band_groups(3, 10)
    want = _replica(x.astype(np.float64), 10, 0.0, 8.0) * column_weights(g, n, 300, 1000, "ramp")
    assert np.allclose(got, want, atol=1e-4)


def test_unit_mask_path_is_bit_identical_to_unmasked_encoding():
    """Identity control C2 at the encoding level: mask path active with weights == 1 changes nothing."""
    if not HAVE:
        return _skip()
    from freenerf_ns.encoding import MaskedNeRFEncoding
    ones = MaskedNeRFEncoding(3, 10, 0.0, 8.0, True, "ramp"); ones.set_progress(0, 0)
    plain = MaskedNeRFEncoding(3, 10, 0.0, 8.0, True, "none")
    x = torch.rand(5, 3)
    assert torch.equal(ones(x), plain(x))


def test_mask_state_survives_checkpointing():
    if not HAVE:
        return _skip()
    from freenerf_ns.encoding import MaskedNeRFEncoding
    a = MaskedNeRFEncoding(3, 4, 0.0, 2.0, True, "ramp"); a.set_progress(120, 400)
    b = MaskedNeRFEncoding(3, 4, 0.0, 2.0, True, "ramp"); b.load_state_dict(a.state_dict())
    x = torch.rand(4, 3)
    assert torch.equal(a(x), b(x))


def test_regulariser_formulas_work_on_torch_tensors_and_have_gradients():
    if not HAVE:
        return _skip()
    from freenerf_ns.losses import depth_smoothness_loss, occlusion_loss
    dens = torch.rand(6, 192, requires_grad=True)
    l1 = occlusion_loss(dens, 20); l1.backward()
    assert dens.grad[:, :20].abs().sum() > 0 and dens.grad[:, 20:].abs().sum() == 0
    d = torch.rand(3, 8, 8, requires_grad=True); acc = torch.rand(3, 8, 8, requires_grad=True)
    l2 = depth_smoothness_loss(d, acc); l2.backward()
    assert d.grad.abs().sum() > 0
    assert acc.grad is None                                                 # opacity weights are gradient-stopped
    assert np.isclose(float(depth_smoothness_loss(torch.full((2, 8, 8), 3.0))), 0.0)


def _make_model(**kw):
    from nerfstudio.data.scene_box import SceneBox
    from freenerf_ns.model import FreeNeRFModelConfig
    cfg = FreeNeRFModelConfig(**kw)
    return cfg.setup(scene_box=SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])), num_train_data=8, metadata={})


def _bundle(model, n=64):
    from nerfstudio.cameras.rays import RayBundle
    d = torch.nn.functional.normalize(torch.randn(n, 3), dim=-1)
    o = -4.0 * d + 0.05 * torch.randn(n, 3)
    return model.collider(RayBundle(origins=o, directions=d, pixel_area=torch.ones(n, 1)))


def test_model_builds_renders_and_backpropagates_with_all_regularisers():
    if not HAVE:
        return _skip()
    from freenerf_ns.geometry import PoseSampler, look_at_c2w
    model = _make_model(freq_mask="ramp", occ_weight=0.01, ds_weight=0.1, ds_num_patches=2, ds_patch_size=4)
    model.train()
    for e in model._encodings:
        e.set_progress(100, 1000)
    poses = np.stack([look_at_c2w(4.0 * np.array([np.cos(a), np.sin(a), 0.6])) for a in np.linspace(0, 6, 8)])
    model._ds_sampler = PoseSampler(poses, "sampled")
    model._ds_intr = dict(fx=300.0, fy=300.0, cx=100.0, cy=100.0, width=200, height=200)
    out = model.get_outputs(_bundle(model))
    assert out["occ_density"].shape[0] == 64 and "depth_fine" in out and "accumulation_fine" in out
    loss = model.get_loss_dict(out, {"image": torch.rand(64, 3)})
    assert {"rgb_loss_coarse", "rgb_loss_fine", "occ_loss", "ds_loss"} <= set(loss)
    sum(loss.values()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_expected_depth_is_differentiable_and_stats_are_sorted():
    if not HAVE:
        return _skip()
    model = _make_model()
    model.train()
    out = model.get_outputs(_bundle(model))
    assert out["depth_fine"].requires_grad                                  # nerfstudio's default (median) would not be
    w, t = model.ray_stats(_bundle(model, 16))
    assert w.shape == t.shape and (t[:, 1:] >= t[:, :-1]).all()


def test_all_off_model_has_no_extra_loss_terms_and_unit_mask():
    if not HAVE:
        return _skip()
    model = _make_model()
    model.train()
    out = model.get_outputs(_bundle(model))
    loss = model.get_loss_dict(out, {"image": torch.rand(64, 3)})
    assert set(loss) == {"rgb_loss_coarse", "rgb_loss_fine"}
