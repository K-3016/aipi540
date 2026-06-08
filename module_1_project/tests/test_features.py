import numpy as np
from PIL import Image

from brain_tumor_ml.features import extract_features


def test_feature_vector_is_finite_and_stable_size():
    feature_vector = extract_features(Image.new("L", (32, 32), color=128))
    assert feature_vector.shape == (279,)
    assert np.isfinite(feature_vector).all()

