import numpy as np
import torch
from PIL import Image

from brain_tumor_ml.explainability import grad_cam, overlay_heatmap
from brain_tumor_ml.models import SmallCNN


def test_grad_cam_has_input_resolution_and_finite_values():
    model = SmallCNN()
    heatmap = grad_cam(model, torch.randn(1, 1, 64, 64))

    assert heatmap.shape == (64, 64)
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0
    assert heatmap.max() <= 1


def test_overlay_is_rgb():
    overlay = overlay_heatmap(Image.new("L", (32, 32), color=100), np.ones((16, 16)))
    assert overlay.mode == "RGB"
    assert overlay.size == (16, 16)
