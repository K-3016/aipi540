from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .constants import CLASS_NAMES


def generate_demo_data(output_dir: Path, samples_per_class: int = 30, seed: int = 42) -> None:
    """Generate non-medical synthetic images solely for pipeline smoke testing."""
    output_dir = Path(output_dir)
    rng = np.random.default_rng(seed)
    for class_name in CLASS_NAMES:
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(samples_per_class):
            base = rng.normal(55, 10, (128, 128)).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(base, mode="L")
            draw = ImageDraw.Draw(image)
            x, y = rng.integers(35, 93, size=2)
            if class_name == "glioma":
                radius = int(rng.integers(16, 24))
                points = []
                for angle in np.linspace(0, 2 * np.pi, 18, endpoint=False):
                    r = radius * rng.uniform(0.55, 1.25)
                    points.append((x + r * np.cos(angle), y + r * np.sin(angle)))
                draw.polygon(points, fill=205)
                image = image.filter(ImageFilter.GaussianBlur(0.8))
            elif class_name == "meningioma":
                radius = int(rng.integers(10, 18))
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=185)
                image = image.filter(ImageFilter.GaussianBlur(2.0))
            elif class_name == "pituitary":
                half_width = int(rng.integers(8, 14))
                draw.rounded_rectangle(
                    (64 - half_width, 54 - half_width, 64 + half_width, 54 + half_width),
                    radius=5,
                    fill=220,
                )
                image = image.filter(ImageFilter.GaussianBlur(1.2))
            else:
                image = image.filter(ImageFilter.GaussianBlur(1.5))
            image.save(class_dir / f"patient_{index:03d}_slice_0.png")
    (output_dir / "SYNTHETIC_DATA_ONLY.txt").write_text(
        "These images are artificial and are only for software smoke testing. "
        "They must not be used to claim medical model performance.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic pipeline smoke-test images.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--samples-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_demo_data(**vars(args))
    print(f"Generated synthetic demo images in {args.output_dir}.")


if __name__ == "__main__":
    main()
