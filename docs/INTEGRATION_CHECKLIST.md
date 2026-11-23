# nerfstudio integration checklist

Everything that does not touch nerfstudio or a GPU is unit-tested (see README). The parts below were written against
nerfstudio's documented API **without being able to execute them** where this repo was authored (no torch, no network).
Run `bash scripts/smoke_test.sh` first; it exercises every item. If something fails, this table says where to look.

| # | Assumption | Where | How it is checked | Symptom if wrong, and fix |
|---|---|---|---|---|
| 1 | `NeRFEncoding` output layout is `[sin block | cos block | raw input]`, dimension-major inside each block | `mask.py::band_groups`, `encoding.py` | `check_layout` runs at model construction; integration test 1 compares against a NumPy reference | `AssertionError: ... layout differs ...` at start-up. Update `band_groups` to the new layout (the weights logic does not change) |
| 2 | `include_input=True` appends the raw coordinates last | same | same | same |
| 3 | `NeRFEncoding(in_dim, num_frequencies, min_freq_exp, max_freq_exp, include_input)` signature | `encoding.py` | integration test 1 | `TypeError` at construction: adapt the `super().__init__` call |
| 4 | `NeRFModel.populate_modules` creates `field_coarse`, `field_fine`, `renderer_depth`, `collider`; `NeRFField(position_encoding=, direction_encoding=)` | `model.py::populate_modules` | integration tests 5-7 | `AttributeError`/`TypeError`: compare with `nerfstudio/models/vanilla_nerf.py` of your version and mirror its `populate_modules` |
| 5 | `field_fine.forward(...)` is called by `get_outputs` (the instance-attribute wrapper catches both `forward()` and `__call__`) | `model.py` | integration test 5 asserts `occ_density` exists | no `occ_density` in outputs: the fine field is not `self.field_fine`; adjust the wrapper target |
| 6 | `DepthRenderer(method="expected")` exists and is differentiable | `model.py` | integration test 6 (`requires_grad`) | `TypeError`: older versions use `DepthRenderer(method=...)` positional; or compute expected depth yourself from `w_fine`, `t_fine` |
| 7 | `samples.frustums.starts/ends` and `samples.get_weights(density)` | `model.py::get_outputs` | integration tests 5-6 | `AttributeError`: rename per your version |
| 8 | `RayBundle(origins, directions, pixel_area)` is a valid constructor; `model.collider(bundle)` fills near/far | `model.py`, `export.py` | integration tests 5-6 | `TypeError`: add the extra required field for your version |
| 9 | `TrainingCallback(where_to_run=[...BEFORE_TRAIN_ITERATION], func=f)` calls `f(step=step)`; `TrainingCallbackAttributes` has `.trainer` and `.pipeline` | `model.py::get_training_callbacks` | smoke run with `--pipeline.model.freq-mask ramp` | mask never advances (check `step_buf` in a checkpoint) or `AttributeError`: use `attrs.config`/`attrs.pipeline` as your version exposes |
| 10 | `pipeline.datamanager.train_dataset.cameras` exposes `camera_to_worlds`, `fx`, `cx`, `width`, ... | `model.py` (depth smoothness only) | smoke run with `--pipeline.model.ds-weight 0.1` | `AttributeError`: print `dir(pipeline.datamanager)` and adjust the lookup |
| 11 | Method registration through the `nerfstudio.method_configs` entry point | `pyproject.toml`, `config.py` | `ns-train --help` lists `freenerf` and `sparse-vanilla-default` | not listed: `pip install -e .` again; check the entry-point names |
| 12 | Config classes and field names (`TrainerConfig`, `VanillaPipelineConfig`, `ExponentialDecaySchedulerConfig(lr_final, max_steps)`, `steps_per_eval_all_images`, ...) | `config.py` | `import freenerf_ns.config` | `TypeError` naming the unknown field: delete or rename it |
| 13 | CLI flags: `--pipeline.model.freq-mask`, `--optimizers.fields.scheduler.max-steps`, `--machine.seed`, `--timestamp main`, `--experiment-name`, trailing `blender-data` | `arms.py::train_command` | smoke test; `tests/test_arms_plan.py` pins the strings | tyro prints the valid options; fix the string in `arms.py` |
| 14 | Output path `outputs/<experiment>/<method>/main/config.yml` | `scripts/run_arm.py` | smoke test | export cannot find the config: `ls outputs/<run_id>/*/*/` and adjust |
| 15 | `eval_setup(path, test_mode="inference")` returns `(config, pipeline, checkpoint_path, step)` | `export.py` | smoke test step 4 | `ValueError: not enough values`: adapt the unpacking |
| 16 | `Cameras(..., fx=float, ...)`, `cameras.generate_rays(camera_indices=i, keep_shape=True)`, `model.get_outputs_for_camera_ray_bundle` returns `rgb_fine/depth_fine/accumulation_fine` | `export.py` | smoke test step 4 | `KeyError: rgb_fine`: print `out.keys()` and map |
| 17 | The Blender parser reads `transforms_{train,val,test}.json` and image width from the file | `datasets.py::make_sparse` | smoke test | `FileNotFoundError`: the parser version wants another split name; write that file too |
| 18 | `ns-download-data blender` writes `data/blender/<scene>` somewhere under `data/` | `scripts/get_data.sh` | script searches for `lego/` | download manually (message printed) |

## Design decisions you should know about (put them in the report's "deviations" section)

1. **No custom data parser.** The trainer is given a folder that physically contains only the training views
   (`datasets.make_sparse`), and `val`/`test` JSON files point at those same training images. This is stronger than a parser that
   "raises on a test index": test images are not there. Control C5 verifies it (`controls.check_sparse_dir`).
2. **Expected depth** (`DepthRenderer(method="expected")`) replaces nerfstudio's median depth, so depth is differentiable for the
   smoothness term and consistent with the exported depth maps and videos.
3. **Opacity-weighted depth smoothness** (`ds_alpha_weight=True`, default): neighbour pairs are weighted by the smaller of their
   opacities. Without it the term fights the object silhouette on white-background Blender scenes. Set it to False for the literal RegNeRF form.
4. **Arm E is the RegNeRF depth-smoothness term only** (no sampling-space annealing, no appearance term): name it that way in the report.
5. **The real scene is converted to the Blender layout** (`datasets.convert_real_scene`) so it runs through the same code path:
   poses are re-centred, cameras rescaled to radius 4 (the default near/far of 2..6 then apply), images resized to width 400.
   Check the first render in the viewer. Lens distortion is ignored (a warning is printed if it is large).
6. **The 8 DietNeRF training indices** in `freenerf_ns/datasets.py::DIETNERF_8` were written from memory of the FreeNeRF/DietNeRF Blender
   scripts. **Verify them against github.com/ajayjain/DietNeRF before freezing the manifests**, or pass `--dietnerf-indices`.
7. **Sample window M** defaults to 20 (the paper's Blender value). `geometry.compute_M` gives the geometry-based value
   (about 24 for Blender with 192 samples and a 0.8 safety factor); pick it per scene before `config-frozen` and record it.
