"""Calibrated OpenGL camera rays and RGB targets for a prepared scene."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def image_downscale(camera, requested):
    if requested is None:
        return max(1, math.ceil(max(camera["w"], camera["h"]) / 400))
    if requested < 1:
        raise ValueError("downscale must be positive")
    return requested


def _invert_lens(xd, yd, camera):
    """Invert radial plus tangential OpenCV distortion before the OpenGL axis flip."""
    k1, k2, k3 = (camera[k] for k in ("k1", "k2", "k3"))
    p1, p2 = camera["p1"], camera["p2"]
    if not any((k1, k2, k3, p1, p2)):
        return xd, yd
    x, y = xd.copy(), yd.copy()
    for iteration in range(16):
        r2 = x*x + y*y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        ex = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x) - xd
        ey = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y - yd
        if max(np.max(np.abs(ex)), np.max(np.abs(ey))) < 1e-8:
            return x, y
        slope = k1 + r2 * (2*k2 + 3*r2*k3)
        jxx = radial + 2*x*x*slope + 2*p1*y + 6*p2*x
        jxy = 2*x*y*slope + 2*p1*x + 2*p2*y
        jyy = radial + 2*y*y*slope + 6*p1*y + 2*p2*x
        det = jxx*jyy - jxy*jxy
        if not np.isfinite(det).all() or np.any(det <= 1e-10):
            break
        x -= (jyy*ex - jxy*ey) / det
        y -= (jxx*ey - jxy*ex) / det
    raise ValueError("Lens inversion failed. Check camera calibration or use calibrated undistorted images.")


def _direction_grid(camera, width, height):
    # Integer downscaling may round height and width differently.
    sx, sy = width / camera["w"], height / camera["h"]
    col = (np.arange(width) + 0.5 - camera["cx"]*sx) / (camera["fx"]*sx)
    row = (np.arange(height) + 0.5 - camera["cy"]*sy) / (camera["fy"]*sy)
    grid = np.empty((height, width, 3), dtype=np.float32)
    for top in range(0, height, 32):
        bottom = min(top+32, height)
        xd, yd = np.broadcast_arrays(col[None, :], row[top:bottom, None])
        x, y = _invert_lens(xd, yd, camera)
        grid[top:bottom, :, 0] = x
        grid[top:bottom, :, 1] = -y
        grid[top:bottom, :, 2] = -1
    return torch.from_numpy(grid)


def _world_rays(pose, camera_dirs):
    if pose.ndim == 2:
        directions = camera_dirs @ pose[:3, :3].T
        return pose[:3, 3].expand_as(directions), directions
    directions = (pose[..., :3, :3] @ camera_dirs.unsqueeze(-1)).squeeze(-1)
    return pose[..., :3, 3].expand_as(directions), directions


def _blender_orbit(phase):
    angle, elevation = 2*math.pi*phase, math.pi/6
    back = np.array([math.cos(elevation)*math.cos(angle), math.cos(elevation)*math.sin(angle),
                     math.sin(elevation)])
    right = np.array([-math.sin(angle), math.cos(angle), 0])
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, np.cross(back, right), back))
    pose[:3, 3] = 4 * back
    return pose


def _interpolate_pose(first, second, weight):
    """Translate linearly and rotate along the shortest arc in SO(3)."""
    rotation = first[:3, :3]
    relative = rotation.T @ second[:3, :3]
    omega = np.array([relative[2, 1]-relative[1, 2], relative[0, 2]-relative[2, 0],
                      relative[1, 0]-relative[0, 1]]) / 2
    sine = np.linalg.norm(omega)
    cosine = np.clip((np.trace(relative) - 1) / 2, -1, 1)
    angle = np.arctan2(sine, cosine) * weight
    if sine > 1e-7:
        axis = omega / sine
    elif cosine < 0:
        # The relative rotation is 180 degrees. Its fixed eigenvector is the axis.
        _, eigenvectors = np.linalg.eigh((relative + relative.T) / 2)
        axis = eigenvectors[:, -1]
    else:
        axis = np.array([1., 0., 0.])
        angle = 0.
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    turn = np.eye(3) + np.sin(angle)*skew + (1 - np.cos(angle))*(skew @ skew)
    pose = np.eye(4)
    pose[:3, :3] = rotation @ turn
    pose[:3, 3] = (1-weight)*first[:3, 3] + weight*second[:3, 3]
    return pose


class SceneDataset:
    """Load RGB lazily. A shared camera-direction grid serves all views of this scene."""

    def __init__(self, manifest, split, downscale=1):
        path = Path(manifest)
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        meta = self.manifest
        if meta.get("version") != 1:
            raise ValueError("Unsupported scene manifest version. Regenerate it with sparse_nerf.prepare.")
        if not isinstance(downscale, int) or downscale < 1:
            raise ValueError("downscale must be a positive integer factor")
        dense, sparse, test = (meta["splits"][s] for s in ("dense", "sparse", "test"))
        for ids in (dense, sparse, test):
            if (not ids or len(ids) != len(set(ids))
                    or min(ids) < 0 or max(ids) >= len(meta["frames"])):
                raise ValueError("Manifest contains an empty, repeated or invalid frame index")
        if len(sparse) != 8 or not set(sparse).issubset(dense) or set(dense).intersection(test):
            raise ValueError("Manifest must have 8 sparse views nested in dense and a disjoint test split")
        self.split = split
        self.name = meta["name"]
        self.source = path.parent / meta["source"]
        self.frames = [meta["frames"][i] for i in meta["splits"][split]]
        self.poses = torch.tensor([f["transform_matrix"] for f in self.frames], dtype=torch.float32)
        self.near, self.far = meta["near"], meta["far"]
        self.white_background = meta["white_background"]
        camera = meta["camera"]
        self.width = camera["w"] // downscale
        self.height = camera["h"] // downscale
        if min(self.height, self.width) < 1:
            raise ValueError(f"downscale={downscale} makes the image smaller than 1 pixel")
        self.camera_dirs = _direction_grid(camera, self.width, self.height)

    def __len__(self):
        return len(self.frames)

    def frame_id(self, i):
        return self.frames[i]["file_path"]

    def image_uint8(self, i):
        path = self.source / self.frame_id(i)
        with Image.open(path) as image:
            original_size = (self.manifest["camera"]["w"], self.manifest["camera"]["h"])
            if image.size != original_size:
                raise ValueError(f"{path}: image dimensions have changed since scene preparation")
            if "A" in image.getbands() or "transparency" in image.info:
                background = (255, 255, 255, 255) if self.white_background else (0, 0, 0, 255)
                image = Image.alpha_composite(Image.new("RGBA", image.size, background), image.convert("RGBA"))
            image = image.convert("RGB")
            if image.size != (self.width, self.height):
                image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
            return torch.from_numpy(np.array(image))

    def image(self, i):
        return self.image_uint8(i).float().div_(255)

    def rays(self, i, rows, cols):
        """Accept a single camera index or one integer camera index per pixel."""
        return _world_rays(self.poses[i], self.camera_dirs[rows, cols])

    def frame(self, i):
        origins, dirs = _world_rays(self.poses[i], self.camera_dirs)
        return origins, dirs, self.image(i)

    def video_rays(self, i, n_frames):
        """Return H x W x 3 rays on a deterministic scene-specific novel-view path."""
        if self.split != "test" or not 0 <= i < n_frames:
            raise ValueError("Video rendering requires the test split and a valid video frame index")
        if self.manifest["format"] == "blender":
            pose = _blender_orbit(i / n_frames)
        else:
            # Capture order is retained. Do not jump across the endpoints of an open path.
            t = i * (len(self) - 1) / max(1, n_frames-1)
            j = min(int(t), len(self)-2)
            pose = _interpolate_pose(self.poses[j].numpy(), self.poses[j+1].numpy(), t-j)
        return _world_rays(torch.tensor(pose, dtype=torch.float32), self.camera_dirs)
