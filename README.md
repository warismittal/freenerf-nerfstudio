# Sparse-View NeRF with a Regularisation Prior (UCS532, Topic 16)

**Problem.** NeRF trained on 8 input views instead of 100 collapses: it reproduces its training images but novel views fill
with floaters. We implement one published regulariser, **FreeNeRF** (frequency annealing + occlusion penalty, Yang et al.,
CVPR 2023), inside **nerfstudio**, and measure how much of the 100-to-8-view gap it recovers on two NeRF-synthetic scenes
(Lego, Chair) and one self-captured scene, against vanilla NeRF. Headline number, the gap-recovery ratio:

```
rho = (PSNR_reg@8 - PSNR_vanilla@8) / (PSNR_vanilla@dense - PSNR_vanilla@8)        (0 = no recovery, 1 = gap closed)
```

Author: **Waris (1024170155)**, Thapar Institute of Engineering and Technology. Supervisor: Dr. Shailendra Tiwari.

## Reproduce the main table (one command)
```bash
pip install -r requirements.txt && pip install -e .
bash run_all.sh                    # NVIDIA GPU: data -> all runs of the submission plan -> results/tables/main_table.md
bash run_all.sh --tables-only      # any machine, no GPU: rebuild every table/figure/verdict from the committed run results
```
`run_all.sh` skips runs whose results already exist, so it can be resumed after an interrupted session.
Seeds are fixed per run (`--machine.seed`), printed as `SEED=` in the console and written into every log in `results/logs/`.

## Dataset (not in this repository)
- **NeRF-synthetic** (Mildenhall et al., ECCV 2020): download `nerf_synthetic.zip` from the NeRF project page
  (link in the README of https://github.com/bmild/nerf), or `ns-download-data blender`. `scripts/get_data.sh` does it and also
  finds the dataset when it is attached as a Kaggle input.
- **Subset used:** scenes Lego and Chair (reported), Drums (Gate-2 development check); images downscaled to 400x400 (area filter);
  **8 training views** `[26, 86, 2, 55, 75, 93, 16, 73]` and **25 test views** (every 8th of 200), the DietNeRF protocol used by
  FreeNeRF. The 8 indices are DietNeRF's `np.random.seed(0); np.random.choice(100, 8, replace=False)`, verified against
  `ajayjain/DietNeRF/dietnerf/run_nerf.py`. Dense reference: all 100 training views. Splits are fixed in `manifests/` and hashed
  (`manifests/HASHES.txt`); the trainer's folder physically contains only training images.
- **Self-captured scene:** about 70 phone photos of one object, poses by COLMAP via `ns-process-data`, converted with
  `scripts/real_scene.py`. Download link for the processed scene: see *Results* (photos contain no people or personal data).

## Requirements
`requirements.txt` (nerfstudio, PyTorch with CUDA, lpips, scikit-image, scipy, numpy, pandas, matplotlib). Exact versions used are
recorded in `requirements.cuda.lock`, written by the Kaggle session that produced the results.

## Hardware and runtime
<!-- filled from results/logs after the runs -->
| Item | Value |
|---|---|
| GPU | to be filled from `results/logs/*.log` (`gpu` field) |
| Time per 8-view run (tau) | to be filled (`TRAIN_SECONDS`) |
| Iterations per run | to be filled |
| Total GPU time | to be filled |

## Results
The main table is generated, never typed: `results/tables/main_table.md` (8 input views, mean ± sd over seeds, ρ with 95%
two-level bootstrap interval). Other outputs: `results/tables/{summary,gap_recovery,view_equivalence,stratified}.csv`,
figures `results/figures/R1..R7`, hypotheses H1-H6 and the reproduction verdict in `results/hypotheses.json`, controls in
`results/controls.json`, novel-view videos in `results/videos/`.

<!-- MAIN_TABLE -->
*(filled after the runs)*
<!-- /MAIN_TABLE -->

## What was implemented and what was reused
| Implemented here | Reused |
|---|---|
| FreeNeRF frequency mask on nerfstudio's encoding (`freenerf_ns/mask.py`, `encoding.py`), occlusion term, RegNeRF depth-smoothness term (`losses.py`), the model subclass and arms (`model.py`, `arms.py`) | nerfstudio's vanilla NeRF model, field, samplers, renderer, trainer (Apache-2.0) |
| Isolated sparse datasets, hashed manifests, real-scene converter (`datasets.py`) | COLMAP via `ns-process-data` |
| Evaluation: metrics, failure diagnostics (G, Ω_M, B, R_hi), bootstrap, ρ, N_eq, hypotheses, verdict (`evalkit/`) | LPIPS reference implementation (`lpips` package), scikit-image SSIM |
| Experiment orchestration, controls C1-C9, videos (`scripts/`) | FreeNeRF equations from the paper; DietNeRF 8-view protocol |

## Credits
- B. Mildenhall et al., *NeRF*, ECCV 2020; NeRF-synthetic dataset. https://github.com/bmild/nerf
- J. Yang, M. Pavone, Y. Wang, *FreeNeRF*, CVPR 2023. https://github.com/Jiawei-Yang/FreeNeRF
- M. Niemeyer et al., *RegNeRF*, CVPR 2022 (depth-smoothness term). https://github.com/google-research/google-research/tree/master/regnerf
- A. Jain, M. Tancik, P. Abbeel, *DietNeRF*, ICCV 2021 (8-view protocol). https://github.com/ajayjain/DietNeRF
- M. Tancik et al., *Nerfstudio*, SIGGRAPH 2023. https://github.com/nerfstudio-project/nerfstudio
- R. Zhang et al., *LPIPS*, CVPR 2018. https://github.com/richzhang/PerceptualSimilarity
- J. L. Schönberger, J.-M. Frahm, *COLMAP*, CVPR 2016. https://colmap.github.io

## Contributions and AI declaration
**Waris and Tanishq** designed the project: the problem framing and proposal, the detailed module-by-module mapping of
what to build, and the explanation flowcharts of the pipeline (data split, method, evaluation, hypotheses and controls)
that the implementation follows. They chose the regulariser, the evaluation protocol and the self-captured scene,
captured the photos, reviewed the code and results, and are responsible for every claim in the report.

**AI assistance (Claude, Anthropic)** was used for the writing work: it wrote the code and tests in this repository from
that design, ran and monitored the Kaggle training jobs on our account, and drafted the report and slide text from the
logged results, which we then reviewed. Every reported number is produced by the code in this repository and traces to
`results/logs/`.

## Repository map
```
run_all.sh          one command: reproduce the main table        scripts/kaggle_job.sh  non-interactive GPU session entry point
freenerf_ns/        method (mask, losses, model, encoding, config, export) + datasets, arms, experiment plans
evalkit/            metrics, diagnostics, statistics, controls, per-run evaluation, aggregation
scripts/            data, runs, tuning, controls, videos, smoke test      tests/   74 tests (7 need torch + nerfstudio)
manifests/          hashed splits      configs/frozen.json   hyper-parameters      results/  committed tables, figures, logs
docs/INTEGRATION_CHECKLIST.md   every nerfstudio API assumption and how it is checked
```
