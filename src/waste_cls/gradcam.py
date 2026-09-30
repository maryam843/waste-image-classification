"""Grad-CAM (Selvaraju et al., 2017) with forward/backward hooks. Depends on torch + numpy only,
so it can be copied into the Hugging Face Space."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """Heat map of where the target layer's activations pushed the chosen class score."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None

    def _forward_hook(self, _module, _inp, out):
        self._acts = out
        out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self._grads = grad

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int, np.ndarray]:
        """x: one normalised image [1,3,H,W]. Returns (cam in [0,1] of shape HxW, class, probabilities)."""
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError("GradCAM expects a single image with shape [1, 3, H, W]")
        was_training = self.model.training
        self.model.eval()
        handle = self.target_layer.register_forward_hook(self._forward_hook)
        try:
            # requires_grad on the input so gradients flow even through a frozen backbone
            x = x.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                idx = int(logits.argmax(1)) if class_idx is None else int(class_idx)
                self.model.zero_grad(set_to_none=True)
                logits[0, idx].backward()
        finally:
            handle.remove()
            self.model.train(was_training)
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        cam = cam - cam.min()
        peak = cam.max()
        cam = cam / peak if peak > 0 else np.zeros_like(cam)
        return cam.astype(np.float32), idx, probs


def colorize(cam: np.ndarray) -> np.ndarray:
    """Map a [0,1] heat map to RGB uint8 with a simple jet-like colour ramp."""
    c = np.clip(cam, 0, 1)[..., None]
    r = np.clip(1.5 - np.abs(4 * c - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * c - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * c - 1), 0, 1)
    return (np.concatenate([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend an RGB uint8 image (HxWx3) with its Grad-CAM heat map."""
    if image.shape[:2] != cam.shape:
        raise ValueError("image and cam must have the same height and width")
    heat = colorize(cam).astype(np.float32)
    out = (1 - alpha) * image.astype(np.float32) + alpha * heat
    return np.clip(out, 0, 255).astype(np.uint8)
