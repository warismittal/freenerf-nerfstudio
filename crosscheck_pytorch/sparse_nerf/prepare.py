"""Freeze an RGB dataset, calibrated camera poses and nested view selections."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from PIL import Image


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pose(values, name):
    pose = np.asarray(values, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{name}: expected a finite 4 x 4 camera-to-world matrix")
    rotation = pose[:3, :3]
    if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)):
        raise ValueError(f"{name}: camera matrix must be a right-handed rigid transform")
    return pose


def _camera(kind, metadata, width, height):
    if kind == "blender":
        angle = float(metadata["camera_angle_x"])
        if not 0 < angle < math.pi:
            raise ValueError("camera_angle_x must be in radians between 0 and pi")
        focal = width / (2 * math.tan(angle / 2))
        return dict(w=width, h=height, fx=focal, fy=focal, cx=width/2, cy=height/2,
                    k1=0., k2=0., k3=0., p1=0., p2=0.)

    model = metadata.get("camera_model", "OPENCV")
    if model not in {"OPENCV", "PINHOLE", "SIMPLE_PINHOLE"}:
        raise ValueError(f"Unsupported camera model {model}. Use a perspective camera.")
    if any(float(metadata.get(key, 0)) != 0 for key in ("k4", "k5", "k6", "s1", "s2", "s3", "s4")):
        raise ValueError("Unsupported lens coefficients. Use the four-parameter OPENCV camera.")
    if (width, height) != (metadata["w"], metadata["h"]):
        raise ValueError("Image dimensions differ from transforms.json. Use the original processed images.")
    camera = dict(w=width, h=height, fx=float(metadata["fl_x"]), fy=float(metadata["fl_y"]),
                  cx=float(metadata["cx"]), cy=float(metadata["cy"]))
    camera.update({key: float(metadata.get(key, 0.0)) for key in ("k1", "k2", "k3", "p1", "p2")})
    if not np.isfinite(list(camera.values())).all() or min(camera["fx"], camera["fy"]) <= 0:
        raise ValueError("Camera intrinsics must be finite with positive focal lengths")
    return camera


def _load_frames(source, filename, kind, frames, seen, camera):
    meta = _read_json(source / filename)
    entries = meta["frames"]
    if kind == "nerfstudio":
        # ns-process-data writes incrementally numbered output filenames.
        def order(frame):
            return [int(part) if part.isdigit() else part.lower()
                    for part in re.split(r"(\d+)", frame["file_path"])]
        entries = sorted(entries, key=order)
    indices = []
    for entry in entries:
        path = source / entry["file_path"]
        if kind == "blender" and not path.suffix:
            path = path.with_suffix(".png")
        path = path.resolve()
        name = path.relative_to(source).as_posix()
        if path in seen:
            raise ValueError(f"Repeated image path across views or splits: {name}")
        seen.add(path)
        with Image.open(path) as image:
            calibration = _camera(kind, {**meta, **entry}, *image.size)
        if camera and calibration != camera:
            raise ValueError(f"{name}: all views must share one calibration and resolution")
        camera.update(calibration)
        indices.append(len(frames))
        frames.append(dict(file_path=name, transform_matrix=_pose(entry["transform_matrix"], name)))
    if not indices:
        raise ValueError(f"No registered frames in {filename}")
    return indices


def _coverage_subset(indices, frames, count):
    if count > len(indices):
        raise ValueError(f"Requested {count} training views but only {len(indices)} are available")
    if count == len(indices):
        return list(indices)
    centers = np.stack([frames[i]["transform_matrix"][:3, 3] for i in indices])
    # Start with the first eligible view. Each subsequent camera fills the largest gap.
    chosen = []
    distances = np.full(len(indices), np.inf)
    current = 0
    for _ in range(count):
        chosen.append(current)
        distances = np.minimum(distances, np.square(centers - centers[current]).sum(axis=1))
        distances[chosen] = -np.inf
        current = int(distances.argmax())
    return sorted(indices[i] for i in chosen)


def _center_capture(frames, dense):
    # For an inward-facing capture the object lies near the least-squares
    # intersection of the viewing axes. Only dense training poses enter this fit.
    poses = np.stack([frames[i]["transform_matrix"] for i in dense])
    axes = -poses[:, :3, 2]
    origins = poses[:, :3, 3]
    projections = np.eye(3) - axes[:, :, None] * axes[:, None, :]
    system = projections.mean(axis=0)
    if np.linalg.cond(system) > 1e5:
        raise ValueError("The cameras look in parallel directions. Capture an inward-facing object orbit.")
    center = np.linalg.solve(system, (projections @ origins[:, :, None]).mean(axis=0)[:, 0])
    radius = np.median(np.linalg.norm(center - origins, axis=1))
    in_front = np.sum((center - origins) * axes, axis=1)
    if radius <= 1e-8 or np.median(in_front) <= 0:
        raise ValueError("Cannot locate an object in front of these cameras. Check camera poses and orbit.")
    scale = 4.0 / radius
    for frame in frames:
        pose = frame["transform_matrix"]
        pose[:3, 3] = (pose[:3, 3] - center) * scale
    return dict(center=center.tolist(), scale=scale)


def prepare(source, output, kind, *, dense_count=100, test_count=20, near=None, far=None, name=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if dense_count <= 8:
        raise ValueError("The dense reference needs more than eight training views")
    frames, seen, camera = [], set(), {}
    normalization = dict(center=[0., 0., 0.], scale=1.)
    if kind == "blender":
        pool = _load_frames(source, "transforms_train.json", kind, frames, seen, camera)
        test = _load_frames(source, "transforms_test.json", kind, frames, seen, camera)
        default_bounds = 2.0, 6.0
    elif kind == "nerfstudio":
        indices = _load_frames(source, "transforms.json", kind, frames, seen, camera)
        if test_count < 2 or len(indices) < test_count + dense_count:
            raise ValueError(f"Need >=2 test views and at least dense-count + test-count "
                             f"({dense_count + test_count}) registered images; found {len(indices)}")
        # One held-out frame in the middle of each equal-size temporal interval.
        test = [((2 * j + 1) * len(indices)) // (2 * test_count) for j in range(test_count)]
        test_indices = set(test)
        pool = [i for i in indices if i not in test_indices]
        default_bounds = 0.2, 8.0
    else:
        raise ValueError(f"Unknown data format: {kind}")
    dense = _coverage_subset(pool, frames, dense_count)
    sparse = _coverage_subset(dense, frames, 8)
    if kind == "nerfstudio":
        normalization = _center_capture(frames, dense)
    near = default_bounds[0] if near is None else near
    far = default_bounds[1] if far is None else far
    if not np.isfinite([near, far]).all() or not 0 < near < far:
        raise ValueError("Ray bounds must be finite and satisfy 0 < near < far")
    for frame in frames:
        frame["transform_matrix"] = frame["transform_matrix"].tolist()
    try:
        source_reference = Path(os.path.relpath(source, output.parent)).as_posix()
    except ValueError:  # Two different Windows drives
        source_reference = source.as_posix()
    manifest = dict(version=1, name=name or source.name, format=kind, source=source_reference,
                    camera=camera, near=near, far=far, white_background=kind == "blender",
                    normalization=normalization, frames=frames,
                    splits=dict(dense=dense, sparse=sparse, test=test))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, encoding="utf-8", delete=False) as tmp:
        temp_path = Path(tmp.name)
        try:
            json.dump(manifest, tmp, indent=2, allow_nan=False)
            tmp.write("\n")
        except BaseException:
            tmp.close()
            temp_path.unlink(missing_ok=True)
            raise
    try:
        temp_path.replace(output)
    finally:
        temp_path.unlink(missing_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=["blender", "nerfstudio"], required=True)
    parser.add_argument("--source", type=Path, required=True, help="Scene directory containing transforms JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output manifest JSON")
    parser.add_argument("--name", help="Scene label (defaults to the source directory name)")
    parser.add_argument("--dense-count", type=int, default=100)
    parser.add_argument("--test-count", type=int, default=20, help="Self-capture only. Blender uses its full test split.")
    parser.add_argument("--near", type=float, help="Near ray bound in normalized scene units")
    parser.add_argument("--far", type=float, help="Far ray bound in normalized scene units")
    args = parser.parse_args()
    try:
        result = prepare(args.source, args.output, args.format, dense_count=args.dense_count,
                         test_count=args.test_count, near=args.near, far=args.far, name=args.name)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")
    counts = ", ".join(f"{key}={len(ids)}" for key, ids in result["splits"].items())
    print(f"{args.output} ({counts}, near={result['near']}, far={result['far']})")


if __name__ == "__main__":
    main()
