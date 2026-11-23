"""nerfstudio method registrations (entry points in pyproject.toml). Requires nerfstudio.

  ns-train freenerf               --data <sparse dataset>  [--pipeline.model.<flag> ...] blender-data
  ns-train sparse-vanilla-default --data <sparse dataset>  blender-data       (arm A': nerfstudio defaults)
"""
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from .model import FreeNeRFModelConfig

_MAX_ITERS = 30_000


def _trainer(name: str, model_config, optimizers) -> TrainerConfig:
    return TrainerConfig(
        method_name=name,
        # Evaluation images are never read during training (steps set beyond any run): the trainer must not touch test data.
        steps_per_eval_batch=1000,
        steps_per_eval_image=10**9,
        steps_per_eval_all_images=10**9,
        steps_per_save=5000,
        max_num_iterations=_MAX_ITERS,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=model_config,
        ),
        optimizers=optimizers,
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    )


freenerf_method = MethodSpecification(
    config=_trainer(
        "freenerf",
        FreeNeRFModelConfig(eval_num_rays_per_chunk=1 << 15),
        {  # original NeRF schedule: Adam, lr 5e-4 -> 5e-5 exponential decay
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=_MAX_ITERS),
            }
        },
    ),
    description="Vanilla NeRF + optional FreeNeRF frequency annealing / occlusion term and RegNeRF depth smoothness.",
)

sparse_vanilla_default = MethodSpecification(
    config=_trainer(
        "sparse-vanilla-default",
        VanillaModelConfig(_target=NeRFModel, eval_num_rays_per_chunk=1 << 15),
        {"fields": {"optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-8), "scheduler": None}},
    ),
    description="Arm A': nerfstudio's own vanilla-nerf defaults on the sparse manifest dataset (untuned baseline).",
)
