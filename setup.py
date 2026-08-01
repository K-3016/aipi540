"""Environment bootstrap and optional package metadata for MedExplain."""

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
REQUIRED_DIRS = (
    "data/raw",
    "data/processed",
    "data/outputs",
    "models",
    "notebooks",
)


def initialize_project() -> None:
    """Create local directories and print setup instructions."""
    if sys.version_info < (3, 10):
        raise SystemExit("MedExplain requires Python 3.10 or newer.")
    for relative_dir in REQUIRED_DIRS:
        directory = ROOT / relative_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch(exist_ok=True)
    print(
        "MedExplain directories are ready.\n\n"
        "Next steps:\n"
        "  python -m venv .venv\n"
        "  source .venv/bin/activate  # Windows: .venv\\Scripts\\activate\n"
        "  pip install -r requirements.txt\n"
        "  python main.py prepare-data --num-examples 500\n"
        "  python main.py train        # GPU strongly recommended\n"
        "  python main.py evaluate\n"
        "  streamlit run main.py\n\n"
        "Setup does not download a model or medical dataset."
    )


if __name__ == "__main__" and len(sys.argv) == 1:
    initialize_project()
else:
    setup(
        name="medexplain",
        version="0.1.0",
        description="Patient-friendly medical language rewriting with LoRA",
        python_requires=">=3.10",
        packages=find_packages(),
    )

