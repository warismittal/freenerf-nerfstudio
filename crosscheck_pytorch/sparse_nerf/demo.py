"""Generate a small ray-traced sphere fixture with known Blender-format cameras.

    python -m sparse_nerf.demo --output data/demo
    python -m sparse_nerf.prepare --format blender --source data/demo --output manifests/demo.json --dense-count 10

The fixture provides 10 train and 3 held-out views at 64 x 64 for pipeline checks.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def _view(azimuth, elevation):
    back = np.array([math.cos(elevation)*math.cos(azimuth),
                     math.cos(elevation)*math.sin(azimuth), math.sin(elevation)])
    right = np.array([-math.sin(azimuth), math.cos(azimuth), 0.])
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, np.cross(back, right), back])
    pose[:3, 3] = 4*back
    return pose


def _image(pose, fov, size=64):
    # Solve |origin + t * ray|^2 = 1 for the nearest positive intersection.
    samples = 2*size
    focal = samples / (2*math.tan(fov/2))
    yy, xx = np.indices((samples, samples), dtype=float) + 0.5
    camera_rays = np.stack([(xx - samples/2)/focal, (samples/2 - yy)/focal, -np.ones_like(yy)], axis=-1)
    rays = camera_rays @ pose[:3, :3].T
    origin = pose[:3, 3]
    a = np.sum(rays*rays, axis=-1)
    b = rays @ origin
    discriminant = b*b - a*(origin @ origin - 1)
    hit = discriminant >= 0
    t = (-b[hit] - np.sqrt(discriminant[hit])) / a[hit]
    normals = origin + t[:, None]*rays[hit]

    # Fixed object-space texture and a stationary directional light across views.
    x, y, z = normals.T
    angle = np.arctan2(y, x)
    ribbon = np.sin(4 * angle + 3*z) > 0.35
    albedo = np.where(ribbon[:, None], [0.90, 0.42, 0.16], [0.12, 0.57, 0.66])
    patch = normals @ np.array([0.55, -0.40, 0.73]) > 0.87
    albedo[patch] = [0.95, 0.83, 0.40]
    light = np.array([0.6, -0.2, 0.9])
    shading = 0.4 + 0.6*np.clip(normals @ (light / np.linalg.norm(light)), 0, 1)
    rgba = np.zeros((samples, samples, 4), dtype=np.uint8)
    rgba[hit, :3] = np.rint(255*shading[:, None]*albedo).astype(np.uint8)
    rgba[hit, 3] = 255
    return Image.fromarray(rgba).resize((size, size), Image.Resampling.LANCZOS)


def generate(output):
    root = Path(output)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(f"{root} already exists and is not an empty directory. Choose a new demo path.")
    root.mkdir(parents=True, exist_ok=True)
    fov = math.radians(40)
    for split, count, phase in [("train", 10, 0.0), ("test", 3, 0.37)]:
        (root / split).mkdir()
        frames = []
        for i in range(count):
            azimuth = 2*math.pi*(i+phase)/count
            elevation = math.radians(24 + 12*math.sin(2*azimuth))
            pose = _view(azimuth, elevation)
            file_path = f"{split}/r_{i}"
            with (root / (file_path + ".png")).open("xb") as out:
                _image(pose, fov).save(out, format="PNG")
            frames.append(dict(file_path=file_path, transform_matrix=pose.tolist()))
        with (root / f"transforms_{split}.json").open("x", encoding="utf-8") as out:
            json.dump(dict(camera_angle_x=fov, frames=frames), out, indent=2)
            out.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="An empty or new scene directory")
    args = parser.parse_args()
    try:
        generate(args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Demo generation failed: {exc}\n")
    print(f"Wrote 10 train and 3 test RGBA views to {args.output}")


if __name__ == "__main__":
    main()
