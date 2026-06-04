import argparse
import json
import os

import joblib
import numpy as np

from data_loader import (
    dataset_to_numpy,
    extract_color_histograms,
    load_image_dataset,
)
from models import (
    MajorityBaseline,
    RandomBaseline,
    build_cnn_model,
    evaluate_baseline,
    evaluate_classification,
    train_classical_model,
)


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_metrics(metrics: dict, output_dir: str):
    out_path = os.path.join(output_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate brain tumor classification models.")
    parser.add_argument("--data-dir", type=str, required=True, help="Root dataset directory containing train/ val/ test/ subfolders.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save metrics and models.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for deep learning training.")
    parser.add_argument("--image-size", type=int, default=128, help="Image height and width for resizing.")
    parser.add_argument("--epochs", type=int, default=15, help="Epoch count for deep model training.")
    parser.add_argument("--classical", type=str, default="random_forest", choices=["random_forest", "logistic", "svm"], help="Classical model type.")
    args = parser.parse_args()

    ensure_output_dir(args.output_dir)

    train_ds, val_ds, test_ds, class_names = load_image_dataset(
        args.data_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )
    print(f"Classes found: {class_names}")

    X_train, y_train = dataset_to_numpy(train_ds)
    X_val, y_val = dataset_to_numpy(val_ds)
    X_test, y_test = dataset_to_numpy(test_ds)

    def to_uint8(images: np.ndarray) -> np.ndarray:
        if images.dtype == np.uint8:
            return images
        if images.max() <= 1.0:
            return (images * 255).astype("uint8")
        return images.astype("uint8")

    print("Evaluating naive baselines...")
    majority = MajorityBaseline()
    majority.fit(y_train)
    majority_metrics = evaluate_baseline(majority, X_test, y_test)

    random = RandomBaseline()
    random.fit(y_train)
    random_metrics = evaluate_baseline(random, X_test, y_test)

    print("Training classical ML model...")
    train_features = extract_color_histograms(to_uint8(X_train))
    val_features = extract_color_histograms(to_uint8(X_val))
    test_features = extract_color_histograms(to_uint8(X_test))
    classical_model = train_classical_model(train_features, y_train, model_type=args.classical)
    classical_train_metrics = evaluate_classification(classical_model, train_features, y_train)
    classical_val_metrics = evaluate_classification(classical_model, val_features, y_val)
    classical_test_metrics = evaluate_classification(classical_model, test_features, y_test)

    print("Training deep learning CNN model...")
    cnn_model = build_cnn_model(input_shape=X_train.shape[1:], num_classes=len(class_names))
    callbacks = [
        __import__("tensorflow").keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        )
    ]
    cnn_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    test_loss, test_acc = cnn_model.evaluate(test_ds)
    deep_metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }

    metrics = {
        "baseline_majority": majority_metrics,
        "baseline_random": random_metrics,
        "classical": {
            "model_type": args.classical,
            "train": classical_train_metrics,
            "val": classical_val_metrics,
            "test": classical_test_metrics,
        },
        "deep": deep_metrics,
        "class_names": class_names,
    }
    save_metrics(metrics, args.output_dir)

    classical_path = os.path.join(args.output_dir, "classical_model.joblib")
    joblib.dump(classical_model, classical_path)
    print(f"Saved classical model to {classical_path}")

    cnn_path = os.path.join(args.output_dir, "cnn_model")
    cnn_model.save(cnn_path)
    print(f"Saved CNN model to {cnn_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
