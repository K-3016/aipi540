from __future__ import annotations

import argparse
import os
from pathlib import Path

from brain_tumor_ml.demo_data import generate_demo_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Brain MRI tumor classification project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo-data", help="Generate non-medical synthetic smoke-test data.")
    demo.add_argument("--output-dir", type=Path, default=Path("data/processed/demo"))
    demo.add_argument("--samples-per-class", type=int, default=30)
    demo.add_argument("--seed", type=int, default=42)

    kaggle = subparsers.add_parser(
        "prepare-kaggle",
        help="Download and prepare the Kaggle brain tumor MRI dataset.",
    )
    kaggle.add_argument("--output-dir", type=Path, default=Path("data/processed/kaggle"))
    kaggle.add_argument("--validation-fraction", type=float, default=0.15)
    kaggle.add_argument("--seed", type=int, default=42)
    kaggle.add_argument(
        "--source-dir",
        type=Path,
        help="Prepare an existing download instead of downloading with kagglehub.",
    )

    train = subparsers.add_parser("train", help="Train and evaluate all three required models.")
    _add_training_arguments(train)

    experiment = subparsers.add_parser(
        "experiment", help="Compare CNN training with and without data augmentation."
    )
    experiment.add_argument("--model-dir", type=Path, default=Path("models"))
    experiment.add_argument("--output-dir", type=Path, default=Path("data/outputs"))
    experiment.add_argument("--epochs", type=int, default=5)
    experiment.add_argument("--batch-size", type=int, default=16)
    experiment.add_argument("--seed", type=int, default=42)

    pipeline = subparsers.add_parser(
        "pipeline", help="Train all models and run the augmentation experiment."
    )
    _add_training_arguments(pipeline)
    pipeline.add_argument("--report-dir", type=Path, default=Path("data/outputs"))
    pipeline.add_argument(
        "--experiment-epochs",
        type=int,
        default=5,
        help="Epochs for each augmentation experiment CNN (default: 5).",
    )
    pipeline.add_argument(
        "--skip-experiment",
        action="store_true",
        help="Train the three required models now and run the experiment later.",
    )

    serve = subparsers.add_parser("serve", help="Launch the prediction and Grad-CAM web UI.")
    serve.add_argument("--model-dir", type=Path, default=Path("models"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    args = parser.parse_args()
    if args.command == "demo-data":
        generate_demo_data(args.output_dir, args.samples_per_class, args.seed)
        print(f"Generated synthetic data in {args.output_dir}.")
    elif args.command == "prepare-kaggle":
        from brain_tumor_ml.kaggle_data import download_and_prepare_kaggle_dataset

        counts = download_and_prepare_kaggle_dataset(
            output_dir=args.output_dir,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            source_dir=args.source_dir,
        )
        print(f"Prepared Kaggle dataset in {args.output_dir}: {counts}")
    elif args.command == "train":
        _train_from_args(args)
    elif args.command == "experiment":
        from brain_tumor_ml.experiment import run_experiment

        run_experiment(
            args.model_dir,
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        print(f"Wrote augmentation experiment results to {args.output_dir}.")
    elif args.command == "pipeline":
        _train_from_args(args)
        if args.skip_experiment:
            print(
                "Skipped augmentation experiment. Run "
                "'python main.py experiment --epochs 5' later.",
                flush=True,
            )
        else:
            from brain_tumor_ml.experiment import run_experiment

            run_experiment(
                args.model_dir,
                args.report_dir,
                epochs=args.experiment_epochs,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            print(f"Wrote augmentation experiment results to {args.report_dir}.", flush=True)
    elif args.command == "serve":
        import uvicorn

        os.environ["BRAIN_TUMOR_ARTIFACT_DIR"] = str(args.model_dir)
        uvicorn.run(
            "brain_tumor_ml.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--patient-id-regex")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)


def _train_from_args(args: argparse.Namespace) -> None:
    from brain_tumor_ml.train import run_training

    results = run_training(
        data_dir=args.data_dir,
        output_dir=args.model_dir,
        patient_id_regex=args.patient_id_regex,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        architecture="small_cnn",
        pretrained=False,
        report_dir=getattr(args, "report_dir", Path("data/outputs")),
    )
    for model_name, metrics in results.items():
        print(
            f"{model_name}: balanced_accuracy={metrics['balanced_accuracy']:.3f}, "
            f"macro_f1={metrics['macro_f1']:.3f}"
        )
    if (
        results["deep_cnn"]["balanced_accuracy"]
        <= results["naive_baseline"]["balanced_accuracy"]
    ):
        print(
            "WARNING: The CNN did not beat the naive baseline. "
            "Do not deploy this checkpoint; train for more epochs and inspect metrics.json."
        )


if __name__ == "__main__":
    main()
