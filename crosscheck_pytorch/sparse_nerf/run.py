"""Shared checkpoint loading, provenance checks and chunked full-image rendering."""

import hashlib
import json
from pathlib import Path

import torch

from .model import ModelConfig, NeRFRenderer


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this interpreter. Check the torch installation and GPU driver.")
    return device


def read_manifest(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    if torch.device(device).type == "cuda":
        # Use the same inference precision in the standalone and all-in-one commands.
        torch.set_float32_matmul_precision("highest")
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    path = (run_dir / metadata["manifest_path"]).resolve()
    manifest, digest = read_manifest(path)
    if digest != metadata["manifest_hash"]:
        raise ValueError(f"Manifest has changed since training: {path}")
    metadata["manifest_path"] = str(path)
    state = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    if state["step"] != metadata["training_config"]["steps"] or state["run_id"] != metadata["run_id"]:
        raise ValueError("Evaluation requires this run's completed checkpoint. Resume training first.")
    model = NeRFRenderer(ModelConfig(**metadata["training_config"]["model"]))
    model.load_state_dict(state["model"])
    return manifest, model.to(device).eval(), metadata


@torch.inference_mode()
def render_image(renderer, origins, directions, near, far, *, regularized, white_background, chunk, device):
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if directions.shape != origins.shape or directions.ndim != 3 or directions.shape[-1] != 3:
        raise ValueError("Expected matching H x W x 3 ray origins and directions")
    output = torch.empty(directions.shape, dtype=torch.float32, device="cpu").view(-1, 3)
    flat_o, flat_d = origins.reshape(-1, 3), directions.reshape(-1, 3)
    for start in range(0, len(flat_d), chunk):
        stop = start + chunk
        result = renderer(flat_o[start:stop].to(device), flat_d[start:stop].to(device), near, far,
                          regularized=regularized, progress=1.0, randomized=False,
                          white_background=white_background)
        output[start:stop] = result["rgb"].cpu()
    return output.view(directions.shape)
