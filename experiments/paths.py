# Shared filesystem paths for experiment entry points.

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"


def results_path(*parts: str) -> str:
    return str(RESULTS_ROOT.joinpath(*parts))
