"""Standard, config-tracked result directories for experiment entry points."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
import json
import os
from pathlib import Path
import pickle
import platform
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np

from experiments.paths import RESULTS_ROOT


_RUN_RE = re.compile(r"^(?P<counter>\d{4})_")


class RunDir:
    """Paths belonging to one immutable experiment run."""

    def __init__(self, root: Path, subdirs: Iterable[str]) -> None:
        self.root = root
        for name in subdirs:
            if name not in {"figures", "csv"}:
                setattr(self, name, root / name)

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def csv(self) -> Path:
        return self.root / "csv"


def _jsonable(value: Any) -> Any:
    """Best-effort conversion that preserves type information where useful."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            "_type": f"{type(value).__module__}.{type(value).__qualname__}",
            **{field.name: _jsonable(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            "_type": f"{type(value).__module__}.{type(value).__qualname__}",
            **{
                key: _jsonable(item)
                for key, item in vars(value).items()
                if not key.startswith("_")
            },
        }
    return repr(value)


def _git_hash() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=RESULTS_ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip() or None


def new_run_dir(
    experiment: str,
    config: Any = None,
    *,
    subdirs: tuple[str, ...] = ("figures", "csv"),
) -> RunDir:
    """Create ``results/<experiment>/<NNNN>_<timestamp>/`` and save provenance."""
    experiment_root = RESULTS_ROOT / experiment
    experiment_root.mkdir(parents=True, exist_ok=True)

    counters = [
        int(match.group("counter"))
        for path in experiment_root.iterdir()
        if path.is_dir() and (match := _RUN_RE.match(path.name))
    ]
    counter = max(counters, default=0) + 1
    timestamp = datetime.now().astimezone()

    while True:
        root = experiment_root / f"{counter:04d}_{timestamp:%Y-%m-%d_%H%M%S}"
        try:
            root.mkdir()
            break
        except FileExistsError:
            counter += 1

    for name in subdirs:
        (root / name).mkdir(parents=True, exist_ok=True)

    if config is not None and hasattr(config, "result_dir"):
        config.result_dir = str(root)

    with (root / "config.pkl").open("wb") as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (root / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(config), handle, indent=2, ensure_ascii=False)

    meta = {
        "experiment": experiment,
        "counter": counter,
        "timestamp": timestamp.isoformat(),
        "git_hash": _git_hash(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    with (root / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    return RunDir(root, subdirs)


def latest_run_dir(experiment: str) -> RunDir:
    """Return the most recent numbered run directory for an experiment."""
    experiment_root = RESULTS_ROOT / experiment
    candidates = [
        path for path in experiment_root.iterdir()
        if path.is_dir() and _RUN_RE.match(path.name)
    ] if experiment_root.exists() else []
    if not candidates:
        raise FileNotFoundError(
            f"no numbered result run found for {experiment!r}"
        )
    root = max(candidates, key=lambda path: int(_RUN_RE.match(path.name).group("counter")))
    return RunDir(root, ("figures", "csv"))
