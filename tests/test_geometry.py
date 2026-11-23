import math

import numpy as np

from freenerf_ns.geometry import (PoseSampler, angular_distance_deg, compute_M, fps, fps_extend, look_at_c2w,
                                  orbit_poses, patch_rays)


def blender_like_poses(n=100, seed=0, radius=4.0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        az, el = rng.uniform(0, 2 * np.pi), rng.uniform(0.1, 1.2)
        p = radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        out.append(look_at_c2w(p))
    return np.stack(out)


def test_compute_M_is_close_to_paper_value_on_blender_geometry():
    m = compute_M(near=2.0, far=6.0, cam_radius=4.03, object_radius=1.5, num_samples=192)
    assert 15 <= m <= 30                                           # paper uses 20 for Blender
    assert compute_M(2.0, 6.0, 3.0, 1.5, 192) == 1                 # object reaches the near plane -> minimal window


def test_look_at_is_rigid_right_handed_and_faces_origin():
    c = look_at_c2w([1.0, 2.0, 3.0])
    R = c[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and math.isclose(np.linalg.det(R), 1.0)
    fwd = -R[:, 2]
    assert np.allclose(fwd, -np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1, 2, 3]))


def test_fps_is_nested_and_spread_out():
    pos = blender_like_poses()[:, :3, 3]
    s8 = fps(pos, 8, seed=0)
    s16 = fps_extend(pos, s8, 16)
    s32 = fps_extend(pos, s16, 32)
    assert s16[:8] == s8 and s32[:16] == s16 and len(set(s32)) == 32
    rnd = list(np.random.default_rng(1).choice(100, 8, replace=False))
    def min_pair(idx):
        p = pos[idx]; d = np.linalg.norm(p[:, None] - p[None], axis=-1); return d[np.triu_indices(len(idx), 1)].min()
    assert min_pair(s8) >= min_pair(rnd)


def test_angular_distance_zero_for_identical_and_positive_otherwise():
    pos = blender_like_poses()[:, :3, 3]
    assert np.allclose(angular_distance_deg(pos[:5], pos[:5]), 0, atol=1e-5)
    assert (angular_distance_deg(pos[:5], pos[50:60]) > 0).all()


def test_pose_sampler_modes_respect_training_envelope():
    train = blender_like_poses(8)
    rng = np.random.default_rng(0)
    s = PoseSampler(train, "sampled")
    poses = s.sample(200, rng)
    r = np.linalg.norm(poses[:, :3, 3], axis=1)
    assert np.allclose(r, s.radius)
    el = np.arcsin(poses[:, 2, 3] / r)
    assert el.min() >= s.elev_range[0] - 1e-9 and el.max() <= s.elev_range[1] + 1e-9
    assert (PoseSampler(train, "train").sample(10, rng)[:, :3, 3].shape == (10, 3))
    it = PoseSampler(train, "interp").sample(20, rng)
    ri = np.linalg.norm(it[:, :3, 3], axis=1)
    assert ri.min() >= np.linalg.norm(train[:, :3, 3], axis=1).min() - 1e-6
    for m in it:
        assert np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-9)


def test_orbit_has_constant_radius_and_full_circle():
    train = blender_like_poses(8)
    o = orbit_poses(train, 36)
    r = np.linalg.norm(o[:, :3, 3], axis=1)
    assert np.allclose(r, r[0]) and len(o) == 36
    assert np.allclose(o[:, 2, 3], o[0, 2, 3])                      # constant elevation


def test_patch_rays_geometry():
    c2w = np.stack([np.eye(4)])
    rng = np.random.default_rng(0)
    o, d = patch_rays(c2w, 500.0, 500.0, 200.0, 200.0, 400, 400, 8, rng)
    assert o.shape == (64, 3) and d.shape == (64, 3)
    assert np.allclose(np.linalg.norm(d, axis=1), 1, atol=1e-5)
    assert (d[:, 2] < 0).all()                                      # OpenGL: camera looks down -z
    # a patch exactly at the principal point looks along -z
    class Fixed:
        def integers(self, lo, hi, size): return np.full(size, 196)
    o2, d2 = patch_rays(c2w, 500.0, 500.0, 200.0, 200.0, 400, 400, 8, Fixed())
    centre = d2.reshape(8, 8, 3)[3:5, 3:5].mean(axis=(0, 1))
    assert np.allclose(centre / np.linalg.norm(centre), [0, 0, -1], atol=1e-3)
