# Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""Run the noisy-text robustness experiment.

Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campus_triage.experiment import run_robustness_experiment


def main() -> None:
    """Run robustness experiment and print the result table."""

    results = run_robustness_experiment()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
