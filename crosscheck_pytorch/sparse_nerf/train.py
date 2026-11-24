"""Train one scene and protocol. Re-run with --resume after an interruption."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
from uuid import uuid4

import torch
from torch.nn import functional as F

from .data import SceneDataset, image_downscale
from .model import ModelConfig, NeRFRenderer
from .run import read_manifest, select_device


PROTOCOLS = ("dense-vanilla", "sparse-vanilla", "sparse-regularized")


@contextmanager
def training_lock(run):
    """OS releases the advisory lock even if the training process is killed."""
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".train.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another trainer holds the lock on {run}") from exc
        yield


def save_state(run, model, optimizer, step, run_id, device):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, run_id=run_id,
                 cpu_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
    temp = run / "checkpoint.tmp"
    torch.save(state, temp)
    temp.replace(run / "checkpoint.pt")


def _restore(run, model, optimizer, run_id, device):
    state = torch.load(run / "checkpoint.pt", weights_only=True, map_location="cpu")
    if state["run_id"] != run_id:
        raise ValueError("The checkpoint and run.json belong to different runs")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    return state["step"]


def train(args):
    run, manifest_path = args.run.resolve(), args.manifest.resolve()
    device = select_device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Training supports CPU and CUDA")
    if min(args.steps, args.batch_size, args.save_every, args.log_every) <= 0:
        raise ValueError("Steps, batch size, log and save intervals must be positive")
    if not all(math.isfinite(x) and x > 0 for x in (args.lr, args.lr_final)):
        raise ValueError("Learning rates must be positive and finite")
    manifest, digest = read_manifest(manifest_path)
    downscale = image_downscale(manifest["camera"], args.downscale)
    split = "dense" if args.protocol == "dense-vanilla" else "sparse"
    dataset = SceneDataset(manifest_path, split, downscale)
    config = ModelConfig(width=args.width, depth=args.depth, coarse_samples=args.coarse_samples,
                         fine_samples=args.fine_samples, anneal_fraction=args.anneal_fraction)
    settings = dict(model=asdict(config), steps=args.steps, batch_size=args.batch_size,
                    lr=args.lr, lr_final=args.lr_final, downscale=downscale)
    try:
        stored_path = Path(os.path.relpath(manifest_path, run)).as_posix()
    except ValueError:  # Across Windows drives.
        stored_path = manifest_path.as_posix()
    meta = dict(scene=dataset.name, protocol=args.protocol, seed=args.seed, manifest_hash=digest,
                manifest_path=stored_path, training_config=settings, train_views=len(dataset),
                downscale=downscale, width=dataset.width, height=dataset.height, near=dataset.near,
                far=dataset.far, white_background=dataset.white_background, device_type=device.type)
    with training_lock(run):
        meta_path, checkpoint = run / "run.json", run / "checkpoint.pt"
        if meta_path.exists():
            if not args.resume:
                raise FileExistsError(f"{run} already has a run. Supply --resume or choose a new directory.")
            recorded = json.loads(meta_path.read_text(encoding="utf-8"))
            for key, value in meta.items():
                if recorded.get(key) != value:
                    raise ValueError(f"Cannot resume: {key} differs from run.json. Use the original settings.")
            meta = recorded
        else:
            if checkpoint.exists():
                raise ValueError("Found a checkpoint without run.json. Choose a new run directory.")
            meta.update(run_id=uuid4().hex, torch_version=str(torch.__version__),
                        hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
            temp = run / "run.json.tmp"
            temp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            temp.replace(meta_path)

        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        model = NeRFRenderer(config).to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        step = _restore(run, model, optimizer, meta["run_id"], device) if checkpoint.exists() else 0
        if step == args.steps:
            print(f"Completed checkpoint already exists in {run}")
            return False
        # Uniform sampling over the selected images and pixels. Cache RGB bytes on the CPU.
        images = torch.empty((len(dataset), dataset.height, dataset.width, 3), dtype=torch.uint8)
        for i in range(len(dataset)):
            images[i] = dataset.image_uint8(i)
        regularized = args.protocol == "sparse-regularized"
        print(f"{dataset.name} | {args.protocol} | {len(dataset)} views | {dataset.width}x{dataset.height} "
              f"| {device} | starting from step {step}", flush=True)
        started, start_step = time.monotonic(), step
        for step in range(start_step + 1, args.steps + 1):
            progress = (step - 1) / (args.steps - 1) if args.steps > 1 else 1.0
            lr = args.lr * (args.lr_final / args.lr) ** progress
            for group in optimizer.param_groups:
                group["lr"] = lr
            view = torch.randint(len(dataset), (args.batch_size,))
            row = torch.randint(dataset.height, (args.batch_size,))
            col = torch.randint(dataset.width, (args.batch_size,))
            origins, directions = dataset.rays(view, row, col)
            target = images[view, row, col].to(device, dtype=torch.float32).div_(255)
            prediction = model(origins.to(device), directions.to(device), dataset.near, dataset.far,
                               progress=progress, regularized=regularized, randomized=True,
                               white_background=dataset.white_background)
            mse = F.mse_loss(prediction["rgb"], target)
            loss = mse + F.mse_loss(prediction["rgb_coarse"], target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == start_step + 1 or step % args.log_every == 0 or step == args.steps:
                rate = (step - start_step) / max(1e-6, time.monotonic() - started)
                psnr = -10 * math.log10(max(1e-12, mse.item()))
                print(f"step {step}/{args.steps}  loss {loss.item():.6f}  train-batch PSNR {psnr:.2f} dB  "
                      f"{rate:.2f} steps/s  ~{(args.steps - step)/rate/3600:.2f} h left", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                save_state(run, model, optimizer, step, meta["run_id"], device)
        return True


def add_settings(parser):
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=512, help="rays per optimizer step")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-final", type=float, default=5e-5)
    parser.add_argument("--downscale", type=int, help="integer image reduction (default: fit longest edge in 400)")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--coarse-samples", type=int, default=64)
    parser.add_argument("--fine-samples", type=int, default=128, help="additional importance samples per ray")
    parser.add_argument("--anneal-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--resume", action="store_true")
    add_settings(parser)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
