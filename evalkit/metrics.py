"""Image-quality metrics on float images in [0,1]. PSNR/SSIM are float64 NumPy; LPIPS needs torch + lpips (lazy)."""
from __future__ import annotations

import numpy as np

PSNR_CAP = 100.0


def mse(pred, gt, mask=None) -> float:
    d = (np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64)) ** 2
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if not m.any():
            return float("nan")
        d = d[m]
    return float(d.mean())


def psnr(pred, gt, mask=None) -> float:
    """PSNR = 10 log10(1/MSE). `mask` ([H,W] bool) restricts the average to those pixels (foreground PSNR)."""
    e = mse(pred, gt, mask)
    if e != e:
        return float("nan")
    return float("inf") if e == 0 else -10.0 * np.log10(e)


def ssim(pred, gt) -> float:
    """SSIM with the NeRF-community settings: 11x11 Gaussian window, sigma 1.5, K1=0.01, K2=0.03 (Wang et al. 2004)."""
    from skimage.metrics import structural_similarity

    return float(structural_similarity(np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64),
                                       channel_axis=2, data_range=1.0, win_size=11, gaussian_weights=True,
                                       sigma=1.5, use_sample_covariance=False, K1=0.01, K2=0.03))


_LPIPS = {}


def lpips_distance(pred, gt, net: str = "alex", device: str = "cpu") -> float:
    import lpips
    import torch

    if (net, device) not in _LPIPS:
        _LPIPS[(net, device)] = lpips.LPIPS(net=net, verbose=False).to(device).eval()

    def t(a):
        return (torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float() * 2 - 1).to(device)

    with torch.no_grad():
        return float(_LPIPS[(net, device)](t(pred), t(gt)).item())


def average_metric(psnr_db: float, ssim_v: float, lpips_v: float) -> float:
    """RegNeRF/FreeNeRF 'Average': geometric mean of MSE = 10^(-PSNR/10), sqrt(1-SSIM) and LPIPS."""
    return float((10 ** (-psnr_db / 10.0) * np.sqrt(max(1.0 - ssim_v, 0.0)) * lpips_v) ** (1.0 / 3.0))


def view_metrics(pred, gt, alpha=None, lpips_nets=("alex", "vgg"), device="cpu") -> dict:
    """All per-view metrics for one rendered view against its ground truth."""
    out = {"psnr": min(psnr(pred, gt), PSNR_CAP), "ssim": ssim(pred, gt)}
    if alpha is not None and (alpha > 0).any() and (alpha <= 0).any():
        out["fg_psnr"] = min(psnr(pred, gt, mask=alpha > 0), PSNR_CAP)
    else:
        out["fg_psnr"] = out["psnr"]
    for net in lpips_nets:
        out[f"lpips_{net}"] = lpips_distance(pred, gt, net, device)
    if "lpips_alex" in out:
        out["average"] = average_metric(out["psnr"], out["ssim"], out["lpips_alex"])
    return out
