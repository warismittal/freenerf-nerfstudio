# Held-out novel-view results

Each run averages full-image sRGB PSNR and LPIPS (AlexNet v0.1) over the complete test split. With multiple seeds, the table gives the mean ± sample standard deviation of those run means.

| Scene | Training protocol | Train views | Test views | Seeds | PSNR ↑ (dB) | LPIPS ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chair | dense-vanilla | 100 | 200 | 1 | 17.461 | 0.6490 |
| chair | sparse-regularized | 8 | 200 | 1 | 16.715 | 0.6201 |
| chair | sparse-vanilla | 8 | 200 | 1 | 15.853 | 0.6366 |
| lego | dense-vanilla | 100 | 200 | 1 | 12.228 | 0.7145 |
| lego | sparse-regularized | 8 | 200 | 1 | 14.857 | 0.7142 |
| lego | sparse-vanilla | 8 | 200 | 1 | 12.134 | 0.6875 |

## Experiment settings

| Scene | Evaluation resolution (width × height) | Training steps per run | Rays per update |
| --- | ---: | ---: | ---: |
| chair | 80 × 80 | 1,000 | 96 |
| lego | 80 × 80 | 1,000 | 96 |

## Paired gap recovery

A positive improvement means the regularised model scored better than sparse vanilla. Recovery is 100 × improvement / dense-reference gap, computed separately for each seed. A negative percentage indicates deterioration. Values above 100% exceed the dense reference. A dash marks a gap at or below 0.01 dB PSNR or 0.0001 LPIPS, including reversed gaps.

| Scene | Seed | PSNR gap | PSNR improvement | PSNR recovered | LPIPS gap | LPIPS improvement | LPIPS recovered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chair | 42 | 1.608 | +0.863 | +53.6% | -0.0124 | +0.0165 | — |
| lego | 42 | 0.094 | +2.723 | +2898.0% | -0.0269 | -0.0267 | — |
