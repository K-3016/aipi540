"""Train all required models or run prediction on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from brain_tumor_ml.inference import PredictionService
from brain_tumor_ml.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    train.add_argument("--model-dir", type=Path, default=Path("models"))
    train.add_argument("--patient-id-regex")
    train.add_argument("--image-size", type=int, default=128)
    train.add_argument("--epochs", type=int, default=15)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--architecture", choices=("small_cnn", "resnet18"), default="small_cnn")
    train.add_argument("--pretrained", action="store_true")

    predict = subparsers.add_parser("predict")
    predict.add_argument("image", type=Path)
    predict.add_argument("--model-dir", type=Path, default=Path("models"))

    args = parser.parse_args()
    if args.command == "train":
        results = run_training(
            data_dir=args.data_dir,
            output_dir=args.model_dir,
            patient_id_regex=args.patient_id_regex,
            image_size=args.image_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            architecture=args.architecture,
            pretrained=args.pretrained,
        )
        print(json.dumps(results, indent=2))
    else:
        service = PredictionService(args.model_dir)
        with Image.open(args.image) as image:
            prediction = service.predict(image)
        print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
