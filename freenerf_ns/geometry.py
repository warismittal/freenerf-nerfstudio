"""Camera / scene geometry helpers (NumPy). OpenGL camera convention, as in Blender and nerfstudio:
the camera looks down its local -z axis, +y is up, +x is right."""
from __future__ import annotations

import numpy as np


def compute_M(near: float, far: float, cam_radius: float, object_radius: float, num_samples: int, safety: float = 0.8) -> int:
    """Number of near-camera samples M that lie in front of the object (occlusion window).

    Assumes samples are spread uniformly in [near, far]; the object's nearest surface is at
    cam_radius - object_radius from the origin-centred scene. A safety factor keeps M below the first real
    surface (FreeNeRF's DTU ablation shows PSNR collapsing when M reaches the object).
    """
    free = cam_radius - object_radius - near
    if free <= 0:
        return 1
    return max(1, int(safety * num_samples * free / (far - near)))


def look_at_c2w(pos, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    f = np.asarray(target, dtype=np.float64) - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, dtype=np.float64))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, pos
    return m


def fps_extend(points: np.ndarray, initial, k: int) -> list:
    """Greedy farthest-point sampling that starts from the indices in `initial` (so subsets are nested)."""
    pts = np.asarray(points, dtype=np.float64)
    sel = list(initial)
    d = np.full(len(pts), np.inf)
    for i in sel:
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    while len(sel) < k:
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return sel


def fps(points: np.ndarray, k: int, seed: int = 0) -> list:
    start = int(np.random.default_rng(seed).integers(len(points)))
    return fps_extend(points, [start], k)


def angular_distance_deg(train_pos: np.ndarray, test_pos: np.ndarray) -> np.ndarray:
    """Angle (degrees) between each test camera and its nearest training camera, seen from the scene centre."""
    a = train_pos / np.linalg.norm(train_pos, axis=1, keepdims=True)
    b = test_pos / np.linalg.norm(test_pos, axis=1, keepdims=True)
    cos = np.clip(b @ a.T, -1.0, 1.0)
    return np.degrees(np.arccos(cos.max(axis=1)))


def _slerp_dirs(a, b, t):
    om = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    if om < 1e-6:
        return a
    return (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / np.sin(om)


class PoseSampler:
    """Draw camera poses for the depth-smoothness patches (RegNeRF samples poses the field has not been fitted to).

    mode 'sampled': object scenes. Azimuth uniform, elevation uniform over the range spanned by the training
                    cameras, radius = mean training radius, looking at the origin (z-up).
    mode 'interp' : any scene. Spherical interpolation between random pairs of training cameras
                    (direction slerp, linear radius, rotation slerp). Orientation-agnostic.
    mode 'train'  : the training poses themselves (control arm X2).
    """

    def __init__(self, train_c2w: np.ndarray, mode: str = "sampled"):
        self.c2w = np.asarray(train_c2w, dtype=np.float64)
        pos = self.c2w[:, :3, 3]
        r = np.linalg.norm(pos, axis=1)
        self.radius = float(r.mean())
        el = np.arcsin(np.clip(pos[:, 2] / r, -1.0, 1.0))
        self.elev_range = (float(el.min()), float(el.max()))
        self.mode = mode

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.zeros((n, 4, 4))
        if self.mode == "train":
            idx = rng.integers(0, len(self.c2w), n)
            return self.c2w[idx].copy()
        if self.mode == "sampled":
            az = rng.uniform(0, 2 * np.pi, n)
            el = rng.uniform(self.elev_range[0], self.elev_range[1], n)
            for i in range(n):
                p = self.radius * np.array([np.cos(el[i]) * np.cos(az[i]), np.cos(el[i]) * np.sin(az[i]), np.sin(el[i])])
                out[i] = look_at_c2w(p)
            return out
        if self.mode == "interp":
            from scipy.spatial.transform import Rotation, Slerp

            for i in range(n):
                a, b = rng.choice(len(self.c2w), 2, replace=False)
                t = rng.uniform(0.0, 1.0)
                pa, pb = self.c2w[a, :3, 3], self.c2w[b, :3, 3]
                ra, rb = np.linalg.norm(pa), np.linalg.norm(pb)
                d = _slerp_dirs(pa / ra, pb / rb, t)
                p = d * ((1 - t) * ra + t * rb)
                rot = Slerp([0, 1], Rotation.from_matrix([self.c2w[a, :3, :3], self.c2w[b, :3, :3]]))(t)
                m = np.eye(4)
                m[:3, :3] = rot.as_matrix()
                m[:3, 3] = p
                out[i] = m
            return out
        raise ValueError(f"unknown pose mode {self.mode!r}")


def orbit_poses(train_c2w: np.ndarray, n_frames: int) -> np.ndarray:
    """Circular orbit at the mean training radius and mean training elevation (z-up object scenes)."""
    pos = np.asarray(train_c2w)[:, :3, 3]
    r = np.linalg.norm(pos, axis=1)
    radius = r.mean()
    el = float(np.arcsin(np.clip(pos[:, 2] / r, -1, 1)).mean())
    out = []
    for az in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
        p = radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        out.append(look_at_c2w(p))
    return np.stack(out)


def patch_rays(c2w: np.ndarray, fx: float, fy: float, cx: float, cy: float, width: int, height: int, size: int,
               rng: np.random.Generator):
    """Rays for one random S x S pixel patch per pose. Returns origins, directions as [P*S*S, 3] (row-major per patch)."""
    p = len(c2w)
    v0 = rng.integers(0, height - size + 1, p)
    u0 = rng.integers(0, width - size + 1, p)
    ar = np.arange(size)
    py = (v0[:, None, None] + ar[None, :, None] + 0.5) * np.ones((1, 1, size))
    px = (u0[:, None, None] + ar[None, None, :] + 0.5) * np.ones((1, size, 1))
    dcam = np.stack([(px - cx) / fx, -(py - cy) / fy, -np.ones_like(px)], axis=-1)  # [P,S,S,3]
    dw = np.einsum("pab,pstb->psta", np.asarray(c2w)[:, :3, :3], dcam)
    dw /= np.linalg.norm(dw, axis=-1, keepdims=True)
    org = np.broadcast_to(np.asarray(c2w)[:, None, None, :3, 3], dw.shape)
    return org.reshape(-1, 3).astype(np.float32), dw.reshape(-1, 3).astype(np.float32)
