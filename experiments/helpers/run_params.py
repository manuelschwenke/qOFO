#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/helpers/run_params.py
=================================
Snapshot the parameters that produced a result, next to the result.

Why this exists
---------------
Experiment scripts build their config by calling a factory such as
``make_config()`` or ``make_cigre_config()``, which reads whatever the file
happens to contain at import time.  A result pickle therefore records WHAT
happened but not UNDER WHICH PARAMETERS, and the two drift apart the moment
anyone edits the factory -- which is easy to do while runs are in flight.  Two
arms of the same comparison, started an hour apart, can then be measuring
different controllers with nothing in the artefacts to show it.

Measured on 2026-08-13: ``g_w_pcc`` moved 80 -> 150 and ``g_q`` 200 -> 250
between two invocations of the same comparison script, silently.

So: every result gets a sidecar recording the resolved parameters, and the
whole config is captured rather than a curated subset -- a curated list only
protects the fields somebody thought of.

Usage
-----
    from experiments.helpers.run_params import dump_params

    dump_params(out_dir / f"{tag}.params.json", cfg,
                extra={"tag": tag, "arm": boundary})

Author: Manuel Schwenke / Claude Code
Date: 2026-08-13
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

__all__ = ["config_fingerprint", "dump_params", "sanitise"]

#: Fields excluded from the fingerprint: they change what is written or shown,
#: not what is computed, so including them would defeat caching for no gain.
_FINGERPRINT_IGNORE = frozenset({
    "verbose", "live_plot_controller", "live_plot_cascade", "live_plot_system",
    "live_plot_tracking", "live_plot_sbx", "live_plot_show_reserves",
    "live_plot_show_tie_flows", "run_stability_analysis", "results_dir",
    "run_name", "save_records",
})


def config_fingerprint(cfg: Any, length: int = 10) -> str:
    """Short hash of everything in *cfg* that affects the result.

    Cache keys built from the swept parameters alone are unsound: the rest of
    the config is read live from a factory, so a result computed under one
    baseline is silently reused under another.  Measured on 2026-08-13 --
    ``n_total_s`` moved 7200 -> 18000 between two invocations of the same
    comparison and the cached 2 h arm was served against a fresh 5 h arm,
    producing a 49 % difference that was entirely horizon.

    Put this in the cache key, so a changed config misses instead of lying.
    """
    payload = sanitise(cfg)
    if isinstance(payload, dict):
        payload = {k: v for k, v in payload.items()
                   if k not in _FINGERPRINT_IGNORE}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def sanitise(obj: Any) -> Any:
    """Best-effort conversion to something ``json.dump`` accepts.

    Falls back to ``repr`` rather than raising: a slightly lossy record of an
    exotic field is worth more than no record at all, and a snapshot that can
    fail is a snapshot people switch off.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: sanitise(getattr(obj, f.name, None))
                for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        # JSON keys must be strings, and several configs key by tuple
        # (e.g. tie_thevenin_k by (line_idx, far_bus)).
        return {str(k): sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [sanitise(v) for v in obj]
    try:
        return repr(obj)
    except Exception:
        return "<unrepresentable>"


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def dump_params(path: Path | str, cfg: Any,
                extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write the resolved parameters of one run to *path* as JSON.

    Parameters
    ----------
    path : the sidecar to write, e.g. ``out / f"{tag}.params.json"``.
    cfg  : the config object actually passed to the runner.
    extra: anything the config does not carry -- arm label, per-zone scales
           applied outside the config, the search point, and so on.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "written": datetime.now().isoformat(timespec="seconds"),
        "script": Path(sys.argv[0]).name if sys.argv else None,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "extra": sanitise(extra or {}),
        "config": sanitise(cfg),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
    return path
