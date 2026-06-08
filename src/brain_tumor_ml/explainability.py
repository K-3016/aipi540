from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as functional


def grad_cam(model: nn.Module, tensor: torch.Tensor, class_index: int | None = None) -> np.ndarray:
    """Return a normalized Grad-CAM map for one image tensor."""
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("Grad-CAM expects a tensor with shape (1, channels, height, width).")

    target_layer = _last_convolution(model)
    captured: dict[str, torch.Tensor] = {}

    def save_activation(_module, _inputs, output):
        captured["activation"] = output

    def save_gradient(_module, _grad_input, grad_output):
        captured["gradient"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(save_activation)
    backward_handle = target_layer.register_full_backward_hook(save_gradient)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        target = int(logits.argmax(dim=1).item()) if class_index is None else class_index
        logits[0, target].backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    activation = captured["activation"]
    gradient = captured["gradient"]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    heatmap = functional.interpolate(
        heatmap,
        size=tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    maximum = heatmap.max()
    if maximum > 0:
        heatmap = heatmap / maximum
    return heatmap.detach().cpu().numpy()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay a red-yellow attention map on a grayscale image."""
    base = image.convert("L").resize((heatmap.shape[1], heatmap.shape[0]))
    base_rgb = np.repeat(np.asarray(base, dtype=np.float32)[..., None], 3, axis=2)
    intensity = np.clip(heatmap, 0, 1)[..., None]
    color = np.concatenate(
        [
            np.full_like(intensity, 255),
            intensity * 220,
            np.zeros_like(intensity),
        ],
        axis=2,
    )
    blended = base_rgb * (1 - alpha * intensity) + color * (alpha * intensity)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def _last_convolution(model: nn.Module) -> nn.Conv2d:
    layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not layers:
        raise ValueError("Grad-CAM requires a model with at least one convolutional layer.")
    return layers[-1]
