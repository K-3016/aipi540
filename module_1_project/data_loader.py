import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
except ImportError:
    tf = None


def load_image_dataset(
    data_dir: str,
    image_size: Tuple[int, int] = (128, 128),
    batch_size: int = 32,
    seed: int = 42,
):
    """Load train/val/test image datasets from a directory.

    The dataset is expected to contain one folder per class.
    """
    if tf is None:
        raise ImportError("TensorFlow is required to load image datasets.")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    data_subdirs = {
        split: os.path.join(data_dir, split)
        for split in ["train", "val", "test"]
    }
    if not all(os.path.isdir(path) for path in data_subdirs.values()):
        raise FileNotFoundError(
            "Dataset directory must contain `train`, `val`, and `test` subfolders."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_subdirs["train"],
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_subdirs["val"],
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_subdirs["test"],
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )

    class_names = train_ds.class_names
    return train_ds, val_ds, test_ds, class_names


def dataset_to_numpy(dataset) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    for batch_images, batch_labels in dataset:
        images.append(batch_images.numpy())
        labels.append(batch_labels.numpy())
    X = np.vstack(images)
    y = np.concatenate(labels)
    return X, y


def flatten_images(X: np.ndarray) -> np.ndarray:
    return X.reshape((X.shape[0], -1))


def extract_color_histograms(X: np.ndarray, bins: int = 16) -> np.ndarray:
    histograms = []
    for image in X:
        if image.ndim == 3 and image.shape[2] == 3:
            channels = [image[:, :, c] for c in range(3)]
        else:
            channels = [image]
        channel_hist = []
        for channel in channels:
            hist, _ = np.histogram(channel, bins=bins, range=(0, 255), density=True)
            channel_hist.extend(hist.tolist())
        histograms.append(channel_hist)
    return np.array(histograms)


def load_image_paths(data_dir: str) -> Tuple[List[str], List[int], Dict[int, str]]:
    image_paths = []
    labels = []
    label_map = {}
    classes = sorted(entry.name for entry in os.scandir(data_dir) if entry.is_dir())
    for idx, class_name in enumerate(classes):
        label_map[idx] = class_name
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(class_dir, filename))
                labels.append(idx)
    return image_paths, labels, label_map


def load_images_from_paths(
    image_paths: List[str],
    image_size: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    images = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image = image.resize(image_size)
        images.append(np.array(image, dtype=np.uint8))
    return np.stack(images, axis=0)


def load_classical_data(
    data_dir: str,
    image_size: Tuple[int, int] = (128, 128),
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    image_paths, labels, label_map = load_image_paths(data_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {data_dir}")

    X = load_images_from_paths(image_paths, image_size=image_size)
    y = np.array(labels)
    X = X.astype(np.float32) / 255.0
    features = extract_color_histograms((X * 255).astype(np.uint8))

    X_temp, X_test, y_temp, y_test = train_test_split(
        features, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, label_map
