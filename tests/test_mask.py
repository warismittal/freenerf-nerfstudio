import numpy as np

from freenerf_ns.mask import band_groups, check_layout, column_weights, group_weights, visible_pointer


def replica_encode(x, F, min_e, max_e, include_input=True):
    """NumPy replica of nerfstudio NeRFEncoding.forward: [sin block | cos block | raw], dimension-major blocks."""
    x = np.asarray(x, dtype=np.float64)
    freqs = 2.0 ** np.linspace(min_e, max_e, F)
    scaled = (2 * np.pi * x)[..., None] * freqs
    scaled = scaled.reshape(x.shape[0], -1)
    enc = np.sin(np.concatenate([scaled, scaled + np.pi / 2.0], axis=-1))
    return np.concatenate([enc, x], axis=-1) if include_input else enc


def test_group_structure_position_encoding():
    g, n = band_groups(3, 10)
    assert n == 21 and len(g) == 63                      # matches the official 63-dim encoding = 21 groups of 3
    assert sorted(set(g.tolist())) == list(range(21))
    assert (g[-3:] == 0).all()                           # raw coordinates are group 0
    assert g[0] == 1 and g[10] == 1 and g[20] == 1       # sin of band 0 for x, y, z
    assert g[30] == 2                                    # cos of band 0 starts the cos block


def test_mask_starts_with_raw_only_and_ends_all_ones():
    g, n = band_groups(3, 10)
    w0 = column_weights(g, n, 0, 1000, "ramp")
    assert (w0[-3:] == 1).all() and w0[:-3].sum() == 0   # only raw coordinates visible at step 0
    for step in (1000, 1500, 10**6):
        assert (column_weights(g, n, step, 1000, "ramp") == 1).all()
    assert visible_pointer(0, 1000, n) == 1.0


def test_mask_range_and_monotone_in_time():
    g, n = band_groups(3, 4)
    prev = column_weights(g, n, 0, 500, "ramp")
    for step in range(1, 600, 7):
        w = column_weights(g, n, step, 500, "ramp")
        assert w.min() >= 0 and w.max() <= 1
        assert (w >= prev - 1e-7).all()                  # non-decreasing in t
        prev = w


def test_ramp_has_single_fractional_group_and_step_has_none():
    g, n = band_groups(3, 10)
    w = group_weights(300, 1000, n, "ramp")
    frac = [(i, v) for i, v in enumerate(w) if 0 < v < 1]
    assert len(frac) <= 1
    ws = group_weights(300, 1000, n, "step")
    assert set(np.unique(ws)) <= {0.0, 1.0}
    assert (ws <= w + 1e-7).all()                        # hard step is the floor of the ramp


def test_no_large_jumps():
    g, n = band_groups(3, 10)
    prev = column_weights(g, n, 0, 10000, "ramp")
    for step in range(1, 10001, 10):
        w = column_weights(g, n, step, 10000, "ramp")
        assert np.abs(w - prev).max() < 0.05
        prev = w


def test_reverse_opens_finest_band_first_and_keeps_raw():
    g, n = band_groups(3, 10, reverse=True)
    w = column_weights(g, n, 250, 1000, "ramp")
    assert (w[-3:] == 1).all()
    finest_sin_cols = [9, 19, 29]                        # band 9 for x, y, z in the sin block
    coarsest_sin_cols = [0, 10, 20]
    assert w[finest_sin_cols].min() >= w[coarsest_sin_cols].max()
    fwd = column_weights(*band_groups(3, 10), 250, 1000, "ramp")
    assert fwd[coarsest_sin_cols].min() >= fwd[finest_sin_cols].max()


def test_none_mode_is_identity():
    g, n = band_groups(3, 10)
    assert (column_weights(g, n, 0, 1000, "none") == 1).all()


def test_check_layout_accepts_nerfstudio_layout():
    check_layout(lambda x: replica_encode(x, 10, 0.0, 8.0), 3, 10)
    check_layout(lambda x: replica_encode(x, 4, 0.0, 2.0), 3, 4)


def test_check_layout_rejects_interleaved_layout():
    def interleaved(x):                                   # a plausible-but-wrong layout (sin/cos interleaved per band)
        x = np.asarray(x, dtype=np.float64)
        out = []
        for f in range(10):
            out += [np.sin(2 * np.pi * 2.0 ** f * x), np.cos(2 * np.pi * 2.0 ** f * x)]
        return np.concatenate(out + [x], axis=-1)
    try:
        check_layout(interleaved, 3, 10)
    except AssertionError:
        return
    raise AssertionError("check_layout accepted a wrong layout")


def test_mask_gates_the_intended_physical_band():
    """Hidden bands must contribute exactly zero to the encoding, visible raw coordinates stay untouched."""
    x = np.random.default_rng(0).uniform(-1, 1, (5, 3))
    enc = replica_encode(x, 10, 0.0, 8.0)
    g, n = band_groups(3, 10)
    w = column_weights(g, n, 0, 1000, "ramp")
    masked = enc * w
    assert np.allclose(masked[:, -3:], x)
    assert np.abs(masked[:, :-3]).max() == 0
