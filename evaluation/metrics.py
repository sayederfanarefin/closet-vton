from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MetricResult:
    values: Dict[str, float]


def _gaussian_kernel_2d(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size // 2)
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    kernel = kernel / kernel.sum()
    return kernel


def psnr(img1: torch.Tensor, img2: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    img1/img2: float tensors in [0,1], shape (3,H,W)
    mask (optional): float tensor in {0,1}, shape (1,H,W)
    """
    x = img1
    y = img2
    if mask is None:
        mse = F.mse_loss(x, y)
    else:
        m = mask.expand_as(x)
        denom = m.sum().clamp_min(1.0)
        mse = ((x - y) ** 2 * m).sum() / denom
    if mse.item() <= 0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(1.0) / mse))


def ssim(img1: torch.Tensor, img2: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Structural Similarity (SSIM) for RGB images.

    img1/img2: float tensors in [0,1], shape (3,H,W)
    mask (optional): float tensor in {0,1}, shape (1,H,W). If provided, SSIM map
    is averaged over masked pixels.
    """
    # Prepare tensors
    x = img1[None, ...]  # (1,3,H,W)
    y = img2[None, ...]
    device = x.device
    dtype = x.dtype
    c = x.shape[1]

    window_size = 11
    sigma = 1.5
    kernel = _gaussian_kernel_2d(window_size, sigma).to(device=device, dtype=dtype)
    kernel = kernel.expand(c, 1, window_size, window_size)

    pad = window_size // 2
    mu_x = F.conv2d(x, kernel, padding=pad, groups=c)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=pad, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=pad, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=c) - mu_xy

    # SSIM constants (for L=1 range)
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / den

    # Average over channels first
    ssim_map = ssim_map.mean(dim=1, keepdim=True)  # (1,1,H,W)

    if mask is None:
        return float(ssim_map.mean().item())

    m = mask[None, ...].to(device=device, dtype=dtype)  # (1,1,H,W)
    denom = m.sum().clamp_min(1.0)
    return float((ssim_map * m).sum().item() / denom.item())


def compute_metrics(
    *,
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> MetricResult:
    vals: Dict[str, float] = {
        "psnr": psnr(pred, gt, mask=None),
        "ssim": ssim(pred, gt, mask=None),
    }
    if mask is not None:
        vals["psnr_masked"] = psnr(pred, gt, mask=mask)
        vals["ssim_masked"] = ssim(pred, gt, mask=mask)
    return MetricResult(values=vals)

