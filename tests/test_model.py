import torch
from torch import nn

from brain_tumor_ml.models import SmallCNN


def test_cnn_output_shape():
    model = SmallCNN()
    output = model(torch.zeros(4, 1, 128, 128))
    assert output.shape == (4, 4)


def test_cnn_uses_batch_size_independent_normalization():
    model = SmallCNN()
    assert not any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())
