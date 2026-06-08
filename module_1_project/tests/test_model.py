import torch

from brain_tumor_ml.models import SmallCNN


def test_cnn_output_shape():
    model = SmallCNN()
    output = model(torch.zeros(4, 1, 128, 128))
    assert output.shape == (4, 4)
