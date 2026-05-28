"""tuning/scripts/save_baseline.py — write a baseline YAML for tuning.

Persists the ``MultiTSOConfig`` returned by ``make_base_config()`` of
``experiments/002_M_TSO_M_DSO_COMPARE.py`` to YAML, so that the tuning
CLIs (``tuning.tune`` / ``tuning.validate``) can consume it via their
``--baseline`` flag.

The 002 module name starts with a digit, so it is loaded via
``importlib.import_module`` (the same trick 002 itself uses to pull from
``experiments.000_M_TSO_M_DSO``).
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from tuning._io import save_config_yaml


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuning.scripts.save_baseline")
    p.add_argument(
        "--output", type=Path,
        default=Path("configs/baseline_002_ieee39.yaml"),
        help="Where to write the YAML.",
    )
    args = p.parse_args(argv)

    runner = importlib.import_module("experiments.002_M_TSO_M_DSO_COMPARE")
    cfg = runner.make_base_config()
    save_config_yaml(cfg, args.output)
    print(f"[save_baseline] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
