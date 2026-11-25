from __future__ import annotations

from typing import Any, Optional

import torch


class LpipsScorer:
    """Reusable LPIPS scorer for RGB tensors in the repository's [0, 1] format."""

    def __init__(self, *, net: str = "alex", device: Optional[str] = None) -> None:
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS was requested but the optional 'lpips' package is not installed. "
                "Install the project requirements or run: pip install lpips"
            ) from exc

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = net
        self.model: Any = lpips.LPIPS(net=net).to(self.device).eval()

    @torch.inference_mode()
    def score(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Return perceptual distance (lower is better) for matching CHW RGB images."""
        if pred.ndim != 3 or gt.ndim != 3 or pred.shape[0] != 3 or gt.shape[0] != 3:
            raise ValueError("LPIPS expects pred and gt with shape (3, H, W).")
        if pred.shape != gt.shape:
            raise ValueError(
                f"LPIPS expects equal image shapes; got pred={tuple(pred.shape)}, "
                f"gt={tuple(gt.shape)}. Use --resize-to-gt."
            )

        # The reference implementation expects NCHW RGB values in [-1, 1].
        pred_batch = pred.unsqueeze(0).to(self.device, dtype=torch.float32).mul(2.0).sub(1.0)
        gt_batch = gt.unsqueeze(0).to(self.device, dtype=torch.float32).mul(2.0).sub(1.0)
        value = self.model(pred_batch, gt_batch)
        return float(value.reshape(-1).mean().item())
