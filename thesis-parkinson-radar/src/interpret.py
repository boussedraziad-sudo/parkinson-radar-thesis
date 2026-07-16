"""
Grad-CAM for CNN / ResNet interpretation.

Usage:
    cam = GradCAM(model, model.layer4[-1].conv2)  # last conv block of ResNet
    heatmap = cam(x, class_idx=1)                 # 2D array, same H,W as the feature map
    cam.close()
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """Single-layer Grad-CAM. Forward+backward hooks on a chosen module."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_act),
            target_layer.register_full_backward_hook(self._save_grad),
        ]

    def _save_act(self, module, inp, out) -> None:
        self.activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """Return a normalised 2D heatmap (max-abs scaled to 1)."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.requires_grad_(True)
        logits = self.model(x)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("hooks did not fire — check target_layer")

        # Channel-wise importance from gradient global-average-pool
        w = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (w * self.activations).sum(dim=1).squeeze(0)  # (H, W)
        cam = torch.relu(cam).cpu().numpy()
        m = cam.max()
        return cam / m if m > 0 else cam

    def close(self) -> None:
        for h in self._handles:
            h.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def upsample_heatmap_to(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear upsample a 2D heatmap to (H, W) — same routine as preprocessing."""
    t = torch.from_numpy(arr.astype(np.float32))[None, None]
    out = nn.functional.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return out.squeeze().numpy()
