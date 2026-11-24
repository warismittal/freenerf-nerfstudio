"""Experimental arms -> nerfstudio command line (pure Python, unit-tested).

Arms B and C inherit D's frozen hyper-parameters so that each cell of the 2x2 factorial differs from D in exactly one
term; tuning them separately would confound the ablation with tuning luck (proposal Section 7.3).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Defaults used before tuning (paper values / proposal grid midpoints). `scripts/freeze_config.py` overwrites these.
DEFAULTS: Dict[str, Dict[str, Any]] = {
    "A": {"lr": 5e-4},
    "D": {"freq_end_frac": 0.7, "occ_weight": 0.01, "occ_num_samples": 20},
    "E": {"ds_weight": 0.1, "ds_patch_size": 8},
}

ARMS = ["A", "Aprime", "B", "C", "D", "E", "X1", "X2", "C2"]


@dataclass
class ArmSpec:
    arm: str
    method: str                                   # nerfstudio method name
    model_flags: Dict[str, Any] = field(default_factory=dict)
    lr: Optional[float] = None                    # None => method default (arm A')
    note: str = ""


def merge_frozen(frozen: Optional[dict]) -> dict:
    out = copy.deepcopy(DEFAULTS)
    for arm, vals in (frozen or {}).items():
        if arm in out and isinstance(vals, dict):
            out[arm].update(vals)
    return out


def arm_spec(arm: str, frozen: Optional[dict] = None, scene_kind: str = "synthetic") -> ArmSpec:
    f = merge_frozen(frozen)
    D, E, A = f["D"], f["E"], f["A"]
    pose_src = "interp" if scene_kind == "real" else "sampled"
    occ = {"occ_weight": D["occ_weight"], "occ_num_samples": D["occ_num_samples"]}
    freq = {"freq_mask": "ramp", "freq_end_frac": D["freq_end_frac"]}
    ds = {"ds_weight": E["ds_weight"], "ds_patch_size": E["ds_patch_size"], "ds_pose_source": pose_src}
    if arm == "A":
        return ArmSpec(arm, "freenerf", {}, A["lr"], "vanilla NeRF, tuned learning rate")
    if arm == "Aprime":
        return ArmSpec(arm, "sparse-vanilla-default", {}, None, "nerfstudio defaults, untuned")
    if arm == "B":
        return ArmSpec(arm, "freenerf", dict(freq), A["lr"], "frequency curriculum only")
    if arm == "C":
        return ArmSpec(arm, "freenerf", dict(occ), A["lr"], "occlusion term only")
    if arm == "D":
        return ArmSpec(arm, "freenerf", {**freq, **occ}, A["lr"], "FreeNeRF")
    if arm == "E":
        return ArmSpec(arm, "freenerf", dict(ds), A["lr"], "depth smoothness (RegNeRF term only)")
    if arm == "X1":
        return ArmSpec(arm, "freenerf", {"freq_mask": "reverse", "freq_end_frac": D["freq_end_frac"]}, A["lr"],
                       "reversed curriculum (finest band first)")
    if arm == "X2":
        return ArmSpec(arm, "freenerf", {**ds, "ds_pose_source": "train"}, A["lr"], "smoothness at training poses")
    if arm == "C2":
        return ArmSpec(arm, "freenerf", {"freq_mask": "ramp", "freq_end_frac": 0.0, "occ_weight": 0.0}, A["lr"],
                       "identity control: mask path active but fixed to 1, no occlusion; must equal arm A")
    raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    return str(v)


def flag(name: str) -> str:
    return "--" + name.replace("_", "-")


# Scenes trained with a random background colour (gt alpha kept, blended with the same random colour in the loss).
# Chair collapsed to an empty, all-white field under the default white background in every run, including the
# 100-view reference (training loss flat at the "predict white" value from step ~1.4k): the object covers ~16% of
# each image, so explaining the image as background is a strong local minimum. Random backgrounds remove that
# minimum. Both arms of a scene always share the setting, so within-scene comparisons stay paired.
RANDOM_BG_SCENES = ("chair",)


def train_command(spec: ArmSpec, *, data_dir: str, out_dir: str, run_id: str, iters: int, seed: int,
                  scene_kind: str = "synthetic", rays: int = 4096, device_type: Optional[str] = None,
                  eval_chunk: Optional[int] = None, amp: bool = False, random_bg: bool = False) -> List[str]:
    cmd = ["ns-train", spec.method, "--data", data_dir, "--output-dir", out_dir, "--experiment-name", run_id,
           "--timestamp", "main", "--max-num-iterations", str(iters), "--machine.seed", str(seed), "--vis", "tensorboard"]
    if spec.method == "freenerf":
        cmd += ["--optimizers.fields.scheduler.max-steps", str(iters)]
        if spec.lr is not None:
            cmd += ["--optimizers.fields.optimizer.lr", _fmt(spec.lr)]
    for k, v in spec.model_flags.items():
        cmd += ["--pipeline.model." + k.replace("_", "-"), _fmt(v)]
    if scene_kind == "real":
        cmd += ["--pipeline.model.background-color", "last_sample"]
    elif random_bg:
        cmd += ["--pipeline.model.background-color", "random"]
    if rays != 4096:                      # lite mode: smaller ray batches fit in 8 GB of memory
        cmd += ["--pipeline.datamanager.train-num-rays-per-batch", str(rays)]
    if device_type:                       # 'mps' on Apple Silicon, 'cpu' on Intel Macs
        cmd += ["--machine.device-type", device_type]
    if amp:                               # fp16 autocast + grad scaling (nerfstudio trainer); ~2x on T4 tensor cores
        cmd += ["--mixed-precision", "True"]
    if eval_chunk:                        # rays rendered at once when exporting images (memory!)
        cmd += ["--pipeline.model.eval-num-rays-per-chunk", str(eval_chunk)]
    cmd.append("blender-data")
    if random_bg and scene_kind != "real":
        cmd += ["--alpha-color", "None"]     # keep RGBA so the loss can blend gt with the same random colour
    return cmd
