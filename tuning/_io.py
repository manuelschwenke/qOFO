"""
tuning/_io.py
=============
YAML and JSON helpers for the tuning module.

Kept private; callers should go through the :mod:`tuning.tune` and
:mod:`tuning.validate` CLIs (or :mod:`tuning.reports`).
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from configs.multi_tso_config import MultiTSOConfig


# ---------------------------------------------------------------------------
# YAML load / save for MultiTSOConfig
# ---------------------------------------------------------------------------

def save_config_yaml(cfg: MultiTSOConfig, path: Path) -> None:
    """Save a :class:`MultiTSOConfig` to YAML.

    Non-trivial fields (datetime, list-of-:class:`ContingencyEvent`) are
    serialised as ISO strings / dicts respectively.  Numpy scalars are
    coerced to Python types via :func:`jsonable`.
    """
    d = dataclasses.asdict(cfg)
    if "start_time" in d and isinstance(d["start_time"], datetime):
        d["start_time"] = d["start_time"].isoformat()
    d = jsonable(d)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(d, f, sort_keys=True, default_flow_style=False)


def load_config_yaml(path: Path) -> MultiTSOConfig:
    """Load a :class:`MultiTSOConfig` from YAML.

    Reverses :func:`save_config_yaml`.  Reconstructs ``datetime`` and
    (best-effort) :class:`ContingencyEvent` objects.
    """
    from experiments.helpers.records import ContingencyEvent

    with path.open("r") as f:
        d = yaml.safe_load(f)

    if "start_time" in d and isinstance(d["start_time"], str):
        d["start_time"] = datetime.fromisoformat(d["start_time"])
    if "contingencies" in d and isinstance(d["contingencies"], list):
        d["contingencies"] = [
            ContingencyEvent(**c) if isinstance(c, dict) else c
            for c in d["contingencies"]
        ]

    return MultiTSOConfig(**d)


# ---------------------------------------------------------------------------
# Tuned-params YAML (BO 8-dim subset only)
# ---------------------------------------------------------------------------

def save_tuned_params(
    params: dict[str, float],
    meta: dict[str, Any],
    path: Path,
) -> None:
    """Save the BO-tuned 8-dim params plus study metadata to YAML.

    Schema::

        params:
          g_v: 12345.0
          g_q: 200.0
          ...
        meta:
          study_name: ...
          n_trials: ...
          best_value: ...
          ceilings: {g_w_der: ..., ...}
          timestamp: ...
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": jsonable(dict(params)),
        "meta":   jsonable(dict(meta)),
    }
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=True, default_flow_style=False)


def load_tuned_params(
    path: Path,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Load tuned params; returns ``(params, meta)``."""
    with path.open("r") as f:
        payload = yaml.safe_load(f)
    return dict(payload["params"]), dict(payload.get("meta", {}))


# ---------------------------------------------------------------------------
# JSON helpers (for trial-level diagnostics)
# ---------------------------------------------------------------------------

def jsonable(obj: Any) -> Any:
    """Recursively convert numpy / dataclass / datetime objects into
    types that :mod:`json` and :mod:`yaml` can serialise."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj
