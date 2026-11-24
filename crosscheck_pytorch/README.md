> **Cross-check implementation (Tanishq).** A separate plain-PyTorch NeRF with the FreeNeRF frequency mask, written independently of the nerfstudio method in the repository root. Commands below run from this folder. Its Lego GPU renders (75k steps, seed 42) are in `../results/crosscheck_pytorch/lego_s42/`.

# Sparse-view NeRF with a frequency regularisation prior

Train the same hierarchical NeRF on 100 views and on eight views. Then train the eight-view version again with **FreeNeRF frequency annealing**. The experiment measures how much of the observed 100-to-eight-view gap this one intervention recovers on Lego, Chair, and an object you photograph. Each scene gets three checkpoints, held-out PSNR and LPIPS scores, and two matched-camera movies.

| Run | Training images | Encoding during training | What it measures |
| --- | ---: | --- | --- |
| `dense-vanilla` | 100 | All frequencies available | Dense reference at the same compute budget |
| `sparse-vanilla` | 8 | All frequencies available | Effect of reducing the input views |
| `sparse-regularized` | Same 8 | Low frequencies first, all frequencies available by halfway | Effect of the prior |

The renderer and optimiser are implemented in PyTorch. Nerfstudio supplies the capture processing and COLMAP camera export. This is a single-component reproduction of Yang, Pavone and Wang's FreeNeRF. The paper also studies near-camera density regularisation. Our experiment isolates frequency annealing. Its camera subset, training budget, evaluation split, and LPIPS backbone are local protocol choices, so use the dense and sparse scores from **your own runs** as references. [FreeNeRF paper](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_FreeNeRF_Improving_Few-Shot_Neural_Rendering_With_Free_Frequency_Regularization_CVPR_2023_paper.html) · [Experimental details in the supplement](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Yang_FreeNeRF_Improving_Few-Shot_CVPR_2023_supplemental.pdf)

## What is already in this workspace

Lego and Chair are downloaded under `data/blender`, with frozen manifests in `manifests/`. A **one-seed CPU pilot** ran all three protocols on each scene at 80 × 80, 1,000 steps and 96 rays per update. Each of the six models was evaluated on all 200 official test cameras. The full 400-pixel, 50,000-step experiment and the self-captured scene remain to be run. The pilot's small dense advantage on Lego makes its recovery percentage particularly sensitive to the denominator.

| Scene | 100-view vanilla PSNR / LPIPS | 8-view vanilla PSNR / LPIPS | 8-view annealed PSNR / LPIPS | Pilot movies |
| --- | --- | --- | --- | --- |
| Lego | 12.228 / 0.7145 | 12.134 / 0.6875 | 14.857 / 0.7142 | [Vanilla](results/cpu-pilot/lego-8-vanilla.mp4) · [Annealed](results/cpu-pilot/lego-8-annealed.mp4) |
| Chair | 17.461 / 0.6490 | 15.853 / 0.6366 | 16.715 / 0.6201 | [Vanilla](results/cpu-pilot/chair-8-vanilla.mp4) · [Annealed](results/cpu-pilot/chair-8-annealed.mp4) |

On Chair, the prior recovered 53.6% of the PSNR gap and improved LPIPS by 0.0165. On Lego it improved PSNR by 2.723 dB and worsened LPIPS by 0.0267. Dense vanilla had worse LPIPS than eight-view vanilla on both scenes, leaving no positive dense LPIPS advantage to recover in this pilot. The full-budget runs test whether those effects persist. [Pilot report](results/cpu-pilot/report.md) · [Matched held-out renders](results/cpu-pilot/heldout-views.png) · [Unrounded paired results](results/cpu-pilot/paired.csv)

To reproduce the measured pilot on a new machine, follow steps 1, 3 and 5 for Lego and Chair, then run the command below.

```text
python -m sparse_nerf.experiment --manifests manifests/lego.json manifests/chair.json --output runs/cpu-pilot --device cpu --steps 1000 --batch-size 96 --width 48 --depth 4 --coarse-samples 12 --fine-samples 8 --downscale 10 --chunk 512 --frames 36 --fps 12 --save-every 250
```

## 1. Set up the training machine

Run the commands from the directory containing this README. The benchmark is designed for an NVIDIA CUDA GPU. The CPU path works for development and the demo. When moving to a GPU machine, copy the source, manifests and data while preserving their relative layout. Create the virtual environment on the destination.

**In this Windows workspace**, activate the existing local CPU environment and select its model-weight cache in PowerShell.

```powershell
.\.venv\Scripts\Activate.ps1
$env:TORCH_HOME = (Join-Path (Get-Location) 'scratch/torch')
python -c "import sys, torch; print(sys.executable, torch.__version__, torch.cuda.is_available())"
```

If PowerShell prevents activation, run `.\.venv\Scripts\python.exe` in place of `python` in the commands below. The local environment uses this machine's existing PyTorch installation. Setting `TORCH_HOME` selects the project cache used for the smoke test.

**On a fresh GPU machine**, create an environment with Python 3.11 or 3.12. Examples for Windows and Linux follow.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install a matching `torch` and `torchvision` pair inside the environment. For an NVIDIA machine with a driver supporting CUDA 12.8, this command selects the CUDA builds of the same PyTorch and torchvision versions as the CPU pilot. For other driver or GPU combinations, use the official selector. [PyTorch versioned install commands](https://pytorch.org/get-started/previous-versions/) · [PyTorch installer](https://pytorch.org/get-started/locally/)

```text
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining packages and initialise the LPIPS network. It downloads the official AlexNet backbone on first use (about 233 MB). Do this once while the GPU machine has network access. [LPIPS reference implementation](https://github.com/richzhang/PerceptualSimilarity)

```text
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import lpips; lpips.LPIPS(net='alex', version='0.1').eval()"
python -m unittest discover -s tests -v
```

Expect `True` and the GPU name before starting a production run with `--device cuda`. Use a persistent disk for data and checkpoints on a remote machine. Record the software environment with `python -m pip freeze > environment.txt` alongside your final results.

## 2. Exercise the whole workflow on a small scene

This generates a textured sphere with known poses. The short 64 × 64 run exercises training, metrics, and video encoding in a few commands. It uses 10 dense views, eight sparse views, and three held-out views. Keep its outputs in a separate directory from the benchmark.

```text
python -m sparse_nerf.demo --output scratch/quick-scene
python -m sparse_nerf.prepare --format blender --source scratch/quick-scene --output scratch/quick-manifests/sphere.json --dense-count 10 --name sphere
python -m sparse_nerf.experiment --manifests scratch/quick-manifests/sphere.json --output scratch/quick-runs --device cpu --steps 20 --batch-size 32 --width 32 --depth 3 --coarse-samples 8 --fine-samples 8 --chunk 256 --frames 4 --fps 4
```

For faster small CPU jobs on a many-core host, set `$env:OMP_NUM_THREADS='2'` and `$env:MKL_NUM_THREADS='2'` before launching Python in PowerShell, or `export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` in Bash. Open `scratch/quick-runs/reports/seed-42/report.md`. A 20-step training run is a test of the software path. Use the full settings below for your scientific result.

## 3. Obtain Lego and Chair

The built-in downloader fetches the two scene archives (310 MB combined), verifies their published SHA-256 checksums, and extracts the training and test images and camera transforms. Run this once on a fresh machine. On an existing installation it checks the expected files and dimensions. [NerfBaselines scene archives](https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data/tree/main/blender)

```text
python -m sparse_nerf.download --output data/blender
```

The dataset credits the Lego model to Heinzelnisse (CC BY-NC) and the Chair model to 1DInc (CC0). Include these credits when sharing renders. [Dataset credits](https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data/raw/main/blender/README.txt)

### Alternative sources

Download the **NeRF Synthetic Blender** dataset. In the original authors' data folder, select `nerf_synthetic.zip`, extract it, and place the two scene directories at `data/blender/lego` and `data/blender/chair`. Preserve filenames and RGBA PNGs. The camera transforms for both scenes are already supplied. [Original NeRF data](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4) · [Original NeRF code and data description](https://github.com/bmild/nerf)

If Nerfstudio is installed, its downloader provides the same directory layout. Run it once from the project root into a fresh `data/blender` destination. The upstream downloader replaces an existing `blender` directory. [Nerfstudio dataset guide](https://docs.nerf.studio/quickstart/existing_dataset.html) · [Downloader implementation](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/scripts/downloads/download_data.py)

```text
ns-download-data blender --save-dir data
```

Each scene should have this structure. Only the training and test splits enter the benchmark.

```text
data/blender/lego/
  transforms_train.json
  transforms_test.json
  train/r_0.png ...
  test/r_0.png ...
data/blender/chair/
  transforms_train.json
  transforms_test.json
  train/ ...
  test/ ...
```

## 4. Capture and calibrate your own scene

1. Pick a stationary, textured object, for example a patterned shoe or toy. Use a fixed, textured background and diffuse, constant light. Move the **camera** around the object. A turntable with a stationary background gives COLMAP conflicting motion.
2. Aim for **150–180 sharp, overlapping photos** to allow for failed registration. The experiment needs **at least 120 registered frames** for 100 training views plus 20 held-out targets. Circle the object at two or three elevations, connecting passes smoothly. Include the subject and some of its surroundings in each frame. Keep one lens, zoom, focus distance, orientation, and resolution. Lock exposure and white balance if available.
3. Remove blurred or near-duplicate images before calibration. Give photos chronological, zero-padded names and place them in `data/raw/my_object/`. Use normal perspective RGB images. The loader supports a shared pinhole or standard OpenCV perspective calibration and corrects supported radial/tangential distortion. Fisheye and varying intrinsics require preprocessing to a common calibrated perspective camera.
4. In a **separate Nerfstudio environment**, install Nerfstudio, COLMAP and FFmpeg using the platform-specific guide. Check that `ns-process-data images --help`, `colmap -h`, and `ffmpeg -version` work there. From the project root, run the command below. Return to the PyTorch training environment afterward. [Nerfstudio installation](https://docs.nerf.studio/quickstart/installation.html) · [Custom capture and COLMAP instructions](https://docs.nerf.studio/quickstart/custom_dataset.html)

```text
ns-process-data images --data data/raw/my_object --output-dir data/processed/my_object
```

5. Check `data/processed/my_object/transforms.json` and its referenced images. Inspect the sparse reconstruction in COLMAP if possible. Camera centres should follow the path you walked and point toward the object. Check the registered frame count from the project root.

```text
python -c "import json; print(len(json.load(open('data/processed/my_object/transforms.json'))['frames']))"
```

All three runs use poses estimated in one common coordinate frame. Held-out **RGB targets** never enter the NeRF training loss. COLMAP uses all photographs to recover cameras, so describe the task as **eight-view photometric training with known poses**. The fixed capture normalization centres the scene on the approximate intersection of the dense training cameras' viewing axes and scales their median radius to four. [Nerfstudio camera and pixel conventions](https://docs.nerf.studio/quickstart/data_conventions.html)

If you have a Linux NVIDIA machine with GPU-enabled Docker, you can also process the photos with the Nerfstudio team's prebuilt environment. Run this command in the project root after placing the photos under `data/raw/my_object`. The mount preserves the processed data on the host. [Official Docker setup](https://docs.nerf.studio/quickstart/installation.html#use-docker-image)

```bash
docker run --rm --gpus all --shm-size=4g -u "$(id -u):$(id -g)" -v "$PWD/data:/workspace" ghcr.io/nerfstudio-project/nerfstudio:1.1.3 ns-process-data images --data /workspace/raw/my_object --output-dir /workspace/processed/my_object
```

## 5. Freeze the splits

Run these commands in the training environment to write the split files. The two synthetic manifests are already prepared in this workspace. After capturing your object, only the third command is needed here.

```text
python -m sparse_nerf.prepare --format blender --source data/blender/lego --output manifests/lego.json
python -m sparse_nerf.prepare --format blender --source data/blender/chair --output manifests/chair.json
python -m sparse_nerf.prepare --format nerfstudio --source data/processed/my_object --output manifests/my_object.json --name my_object
```

For the standard synthetic release expect **100 dense, eight sparse, and 200 test views** per scene. The full official test split is used. For the capture, the default is **100 / eight / 20**. The program picks the capture test views at spaced positions in capture order, then selects training cameras. For each scene it chooses an eight-view subset of the dense training pool by farthest-point sampling of camera centres. Both sparse runs get exactly those eight images. A seed controls model initialisation and ray sampling. It does not change the split.

Manifests store the frame filenames, poses, intrinsics, backgrounds, ray bounds and exact split indices. Save these files with the results and keep source images fixed. Blender transparency is composited onto white. Capture RGB retains the actual photographed background, and empty rays use black. The default camera-axis sampling bounds are [2, 6] for Blender and [0.2, 8] for a normalized capture. If the capture clips the subject, inspect the geometry, choose `--near` and `--far` during preparation, and use that same new manifest for all three runs.

Treat the images as part of the experiment record. If you edit or replace photos after preparation, use a new manifest and a fresh output root, then retrain all three protocols. The manifest fingerprint covers filenames, camera geometry and split membership. Evaluation separately fingerprints the held-out pixels used for scoring.

## 6. Train, evaluate, and render

Start the nine-run benchmark from the training environment with the three prepared manifests.

```text
python -m sparse_nerf.experiment --manifests manifests/lego.json manifests/chair.json manifests/my_object.json --output runs/benchmark --device cuda
```

This runs **nine training jobs sequentially**, scores each final checkpoint, renders **six five-second MP4s** (120 frames at 24 fps), and writes a report for seed 42. The synthetic videos use a common orbit around each object. The real-scene movies follow the same interpolated path through the held-out camera poses. Metrics compare renders to real test images at their exact poses. The interpolated movie frames provide qualitative comparisons.

You can start with the two synthetic manifests while photographing the real scene. Rerun the command with the third manifest when ready. After an interruption, rerun the **same command and settings**. It restores model, optimiser and random-number state from the most recent checkpoint, reuses completed work and rebuilds the report. Use `--refresh` to recompute metrics and videos. Changing training settings requires a fresh output root. Copy `data/`, `manifests/` and `runs/` with their relative layout when moving interrupted jobs between machines of the same device type.

| Setting | Benchmark default |
| --- | --- |
| Optimisation | Adam, 50,000 updates, 512 random rays/update |
| Learning rate | Exponential decay from 0.0005 to 0.00005 |
| Network | Separate coarse and fine fields, 8 position layers, width 256 |
| Encoding | 10 spatial and 4 viewing-direction frequency bands, plus raw coordinates |
| Depth samples | 64 stratified coarse, 128 additional hierarchical samples |
| Resolution | Integer downscale to fit the longest edge within 400 px, e.g. 800 × 800 → 400 × 400 |
| Objective | Coarse RGB MSE + fine RGB MSE, sampled from training images |
| Regulariser | Progressive xyz-triplet frequency visibility for positions and viewing directions, fully open by 50% |
| Saving | Every 1,000 steps and at the final step |

The training log prints step throughput and estimated remaining time. For budgeting, measure the steady-state rate on your GPU. With rate *r* steps/s, nine 50,000-step runs take approximately **450,000 / (3,600 × r) GPU-hours**, plus evaluation and rendering. If training runs out of memory, use a smaller shared `--batch-size` (e.g. 256) in a fresh output root for the whole comparison. If evaluation or video runs out of memory, rerun with `--chunk 256` to reduce rays per rendering batch. Use the same steps, batch size, learning-rate schedule and resolution for all protocols on a given scene.

For a quick GPU pilot, use a separate output directory and one scene. This also lets you check poses, framing, resource use, and the output files before launching the nine-run series.

```text
python -m sparse_nerf.experiment --manifests manifests/lego.json --output runs/pilot --device cuda --steps 500 --batch-size 128 --width 64 --depth 4 --coarse-samples 16 --fine-samples 32 --downscale 8 --frames 12
```

**Individual commands** let you train, evaluate or render a particular run. These commands use the regularised Lego configuration at the benchmark defaults.

```text
python -m sparse_nerf.train --manifest manifests/lego.json --protocol sparse-regularized --run runs/single/lego-reg --device cuda --resume
python -m sparse_nerf.evaluate --run runs/single/lego-reg --device cuda --chunk 1024
python -m sparse_nerf.video --run runs/single/lego-reg --output runs/single/lego-reg/novel.mp4 --device cuda --frames 120 --fps 24 --chunk 1024
```

For repeated trials, run the full benchmark command with `--seed 43` and `--seed 44` under the same benchmark output directory. The following command pools all complete scene/seed triplets and reports across-seed sample standard deviations.

```text
python -m sparse_nerf.report --runs runs/benchmark --output results/final
```

## 7. Read the results and finish the submission

| Output | Use |
| --- | --- |
| `runs/benchmark/reports/seed-42/report.md` | Scene-by-scene scores and recovered gap for the default run |
| `summary.csv` and `paired.csv` beside the report | Raw protocol means and signed per-seed improvements and recovery percentages |
| `runs/benchmark/<scene>/seed-42/<protocol>/metrics.json` | Per-test-frame scores and comparison metadata |
| `run.json` and `checkpoint.pt` in each run | Settings, manifest fingerprint, hardware, model, optimiser and training step |
| `novel_120f_24fps.mp4` in each sparse run | Matched-camera sparse vanilla or sparse regularised movie |

**PSNR** is computed for each full sRGB test image on a [0, 1] scale, then averaged in dB. Higher is better. **LPIPS** uses AlexNet v0.1, with RGB mapped to [−1, 1], and averages the distance over the same images. Lower is better. The paper's Blender protocol uses a smaller selection of test views and VGG LPIPS. Label the backbone in your report. [LPIPS evaluation code](https://github.com/richzhang/PerceptualSimilarity) · [FreeNeRF supplement](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Yang_FreeNeRF_Improving_Few-Shot_CVPR_2023_supplemental.pdf)

Let **D** be dense vanilla, **S** sparse vanilla, and **R** sparse regularised. Compute both quantities independently per scene and seed.

| Metric | Observed loss from 100 → 8 | Improvement from the prior | Gap recovered |
| --- | --- | --- | --- |
| PSNR | D − S | R − S | 100 × (R − S) / (D − S) |
| LPIPS | S − D | S − R | 100 × (S − R) / (S − D) |

For example, hypothetical PSNR values D = 30, S = 20 and R = 24 dB give 4 dB improvement and 40% recovery. A negative percentage means the regulariser worsened that metric. A value above 100% means the regularised run exceeded the dense reference. A dash means the observed dense advantage was reversed or no larger than 0.01 dB PSNR / 0.0001 LPIPS. Report the absolute differences in those cases. Treat the dense model as a measured reference, and let the numbers determine whether the eight-view performance dropped.

For delivery, include the three manifests, nine measured score rows per seed, the per-scene gap table, and six clearly labelled videos. In the discussion, compare corresponding timestamps for floating density, thin details such as chair legs, texture blur and stability under camera motion. Note cases where the metrics disagree, the single-capture limitation, and the seed count. The capture protocol and settings in `run.json` make it possible for someone else to reproduce your comparison.

## Implementation map

| Module | Responsibility |
| --- | --- |
| `sparse_nerf/prepare.py`, `data.py` | Scene manifests, splits, calibrated rays, distortion correction and paths |
| `sparse_nerf/model.py` | Encoding mask, coarse/fine radiance fields, stratified and importance sampling, compositing |
| `sparse_nerf/train.py`, `run.py` | Ray sampling, optimisation, resumable checkpoints and chunked rendering |
| `sparse_nerf/evaluate.py`, `video.py`, `report.py` | Held-out metrics, H.264 encoding, paired comparisons |
| `sparse_nerf/experiment.py` | Whole experiment from one or more prepared manifests |

The mask opens consecutive triplets in the positional encoding. It starts with the raw coordinates visible, lets the next triplet grow continuously in weight, and includes higher frequencies as training progresses. Both coarse and fine fields receive the masked spatial features and masked normalized view-direction features. At evaluation all bands are open, and both methods use deterministic samples. [Authors' frequency-mask implementation](https://github.com/Jiawei-Yang/FreeNeRF/blob/main/internal/math.py#L249-L278)

Here the vanilla NeRF encoder places each band's sine and cosine features next to each other, so the mask opens bands from low to high. The authors' public JAX mip-NeRF encoder groups all sine bands together and then all cosine bands. This ordering difference is part of adapting their mask to the vanilla renderer. [Authors' JAX encoder](https://github.com/Jiawei-Yang/FreeNeRF/blob/main/internal/mip.py#L145-L179)
