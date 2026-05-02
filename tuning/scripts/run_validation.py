"""tuning/scripts/run_validation.py — canonical post-tune validation.

Calls ``tuning.validate.main()`` with the project's canonical defaults.
Extra argv is forwarded; argparse takes the last occurrence of each
flag, so flags after the script name override defaults.

Usage::

    # full validation (200 randomised scenarios, seed 42)
    python -m tuning.scripts.run_validation

    # quick check (5 scenarios)
    python -m tuning.scripts.run_validation --n-scenarios 5
"""
from __future__ import annotations

import sys

from tuning import validate as validate_cli


DEFAULT_ARGS = [
    "--params",      "configs/tuned_params_002.yaml",
    "--baseline",    "configs/baseline_002.yaml",
    "--n-scenarios", "200",
    "--seed",        "42",
    "--report",      "results/tuning/validation_report_002.html",
]


if __name__ == "__main__":
    sys.exit(validate_cli.main(DEFAULT_ARGS + sys.argv[1:]))
