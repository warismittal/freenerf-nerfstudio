"""FreeNeRF / RegNeRF-smoothness as a drop-in subclass of nerfstudio's vanilla NeRF. Requires nerfstudio + torch.

One model, one switch per idea, so that every experimental arm differs from vanilla in exactly the flags it enables:
    freq_mask   none | ramp | step | reverse      (arms B, D | step variant | X1)
    occ_weight  lambda_occ, occ_num_samples = M    (arms C, D)
    ds_weight   lambda_DS, ds_patch_size = S       (arms E, X2)
With everything off the model is vanilla NeRF (arm A).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Type

import numpy as np
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig

from .encoding import MaskedNeRFEncoding
from .geometry import PoseSampler, patch_rays
from .losses import depth_smoothness_loss, occlusion_loss


@dataclass
class FreeNeRFModelConfig(VanillaModelConfig):
    _target: Type = field(default_factory=lambda: FreeNeRFModel)
    freq_mask: Literal["none", "ramp", "step", "reverse"] = "none"
    """Frequency curriculum: none (vanilla), ramp (paper Eq. 6), step (hard-step variant), reverse (finest band first)."""
    freq_end_frac: float = 0.7
    """Curriculum length T as a fraction of max_num_iterations (0.0 => mask fixed to 1, used by identity control C2)."""
    freq_end_step: Optional[int] = None
    """Absolute T; overrides freq_end_frac when set."""
    occ_weight: float = 0.0
    """lambda_occ of the occlusion term (0 disables)."""
    occ_num_samples: int = 20
    """M: number of near-camera samples the occlusion term acts on."""
    ds_weight: float = 0.0
    """lambda_DS of the depth-smoothness term (0 disables)."""
    ds_patch_size: int = 8
    ds_num_patches: int = 32
    ds_pose_source: Literal["sampled", "interp", "train"] = "sampled"
    ds_alpha_weight: bool = True
    """Weight neighbour pairs by opacity (Blender-on-white adaptation; False = literal RegNeRF)."""


class FreeNeRFModel(NeRFModel):
    config: FreeNeRFModelConfig

    # ------------------------------------------------------------------ construction
    def populate_modules(self):
        super().populate_modules()
        cfg = self.config
        mode = "none" if cfg.freq_mask == "none" else ("step" if cfg.freq_mask == "step" else "ramp")
        reverse = cfg.freq_mask == "reverse"
        pos = MaskedNeRFEncoding(3, 10, 0.0, 8.0, True, mode, reverse)
        direction = MaskedNeRFEncoding(3, 4, 0.0, 2.0, True, mode, reverse)
        self.field_coarse = NeRFField(position_encoding=pos, direction_encoding=direction)
        self.field_fine = NeRFField(position_encoding=pos, direction_encoding=direction)
        # Differentiable, normalised expected depth (nerfstudio's default renderer is the non-differentiable median).
        self.renderer_depth = DepthRenderer(method="expected")
        self._encodings = [pos, direction]
        self._collect_stats = False
        self._fine_cache = None
        self._ds_sampler: Optional[PoseSampler] = None
        self._ds_intr = None
        self._ds_rng = np.random.default_rng(0)

        # nerfstudio calls `field_fine.forward(...)` directly, which bypasses nn.Module hooks, so we wrap the
        # instance attribute to capture the fine-pass samples and densities needed by the occlusion term.
        orig_forward = self.field_fine.forward

        def forward_and_cache(ray_samples, *args, **kwargs):
            out = orig_forward(ray_samples, *args, **kwargs)
            self._fine_cache = (ray_samples, out)
            return out

        self.field_fine.forward = forward_and_cache

    # ------------------------------------------------------------------ rendering
    def get_outputs(self, ray_bundle: RayBundle):
        self._fine_cache = None
        outputs = super().get_outputs(ray_bundle)
        if self._fine_cache is not None and (self.training or self._collect_stats):
            samples, fo = self._fine_cache
            density = fo[FieldHeadNames.DENSITY]                       # [R, S, 1]
            starts = samples.frustums.starts                           # [R, S, 1]
            order = torch.argsort(starts[..., 0], dim=-1)              # near -> far, regardless of sampler details
            outputs["occ_density"] = torch.gather(density[..., 0], 1, order)
            if self._collect_stats:
                w = samples.get_weights(density)[..., 0]
                mid = ((starts + samples.frustums.ends) * 0.5)[..., 0]
                outputs["w_fine"] = torch.gather(w, 1, order)
                outputs["t_fine"] = torch.gather(mid, 1, order)
        return outputs

    @torch.no_grad()
    def ray_stats(self, ray_bundle: RayBundle):
        """Fine-pass weights [R,S] and sample depths [R,S] (sorted near->far) for diagnostics such as Omega_M."""
        self._collect_stats = True
        try:
            out = self.forward(ray_bundle)
        finally:
            self._collect_stats = False
        return out["w_fine"], out["t_fine"]

    # ------------------------------------------------------------------ losses
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss = super().get_loss_dict(outputs, batch, metrics_dict)
        cfg = self.config
        if cfg.occ_weight > 0 and "occ_density" in outputs:
            loss["occ_loss"] = cfg.occ_weight * occlusion_loss(outputs["occ_density"], cfg.occ_num_samples)
        if cfg.ds_weight > 0 and self.training:
            loss["ds_loss"] = cfg.ds_weight * self._depth_smoothness_term()
        return loss

    def _depth_smoothness_term(self):
        if self._ds_sampler is None:
            raise RuntimeError("depth smoothness is enabled but the pose sampler was not initialised "
                               "(it is built in get_training_callbacks from the training cameras).")
        cfg = self.config
        S, P = cfg.ds_patch_size, cfg.ds_num_patches
        c2w = self._ds_sampler.sample(P, self._ds_rng)
        o, d = patch_rays(c2w, size=S, rng=self._ds_rng, **self._ds_intr)
        dev = self.device
        bundle = RayBundle(origins=torch.from_numpy(o).to(dev), directions=torch.from_numpy(d).to(dev),
                           pixel_area=torch.ones(o.shape[0], 1, device=dev))
        if self.collider is not None:
            bundle = self.collider(bundle)
        out = self.get_outputs(bundle)
        depth = out["depth_fine"].reshape(P, S, S)
        acc = out["accumulation_fine"].reshape(P, S, S)
        return depth_smoothness_loss(depth, acc if cfg.ds_alpha_weight else None)

    # ------------------------------------------------------------------ training callbacks
    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)
        cfg = self.config
        trainer = getattr(training_callback_attributes, "trainer", None)
        max_iters = trainer.config.max_num_iterations if trainer is not None else None
        end = cfg.freq_end_step
        if end is None and max_iters is not None:
            end = int(cfg.freq_end_frac * max_iters)
        if cfg.freq_mask != "none" and end is None:
            raise RuntimeError("Cannot determine the curriculum length T: set --pipeline.model.freq-end-step.")

        def update_mask(step: int):
            for enc in self._encodings:
                enc.set_progress(step, end if end is not None else 1)

        callbacks.append(TrainingCallback(where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                                          func=update_mask))
        if cfg.ds_weight > 0:
            cams = training_callback_attributes.pipeline.datamanager.train_dataset.cameras
            n = int(cams.camera_to_worlds.shape[0])
            c2w = np.concatenate([cams.camera_to_worlds.detach().cpu().numpy(),
                                  np.tile(np.array([[[0, 0, 0, 1.0]]]), (n, 1, 1))], axis=1)
            self._ds_sampler = PoseSampler(c2w, cfg.ds_pose_source)
            self._ds_intr = dict(fx=float(cams.fx[0].item()), fy=float(cams.fy[0].item()), cx=float(cams.cx[0].item()),
                                 cy=float(cams.cy[0].item()), width=int(cams.width[0].item()),
                                 height=int(cams.height[0].item()))
            seed = trainer.config.machine.seed if trainer is not None else 0
            self._ds_rng = np.random.default_rng(10_000 + int(seed))
        return callbacks
