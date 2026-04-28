"""
tuning/_sim_loader.py
=====================
Internal helper: load ``run_multi_tso_dso`` once and cache it.

The experiment script lives in ``experiments/000_M_TSO_M_DSO.py``; its
leading digit makes it unimportable via the regular ``import`` system.
We load it via :mod:`importlib.util` on first call and cache both the
module reference (for callers that need to monkey-patch internals) and
the bare ``run_multi_tso_dso`` callable (the common case).

This module is **private** — not part of the public ``tuning`` API.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

_RUNNER_MODULE: ModuleType | None = None


def _runner_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "experiments" / "000_M_TSO_M_DSO.py"
    )


def get_runner_module() -> ModuleType:
    """Return the cached runner module.

    First call loads ``experiments/000_M_TSO_M_DSO.py`` via
    :mod:`importlib.util`; subsequent calls return the cache.

    Callers that need to monkey-patch internals (e.g.
    :mod:`tuning.ceilings` patches ``_run_delayed_stability_analysis`` to
    capture its return value) should use this helper rather than
    :func:`get_run_multi_tso_dso`.

    Raises
    ------
    FileNotFoundError
        If the experiment script is not found at the expected path.
    ImportError
        If the import spec cannot be created.
    """
    global _RUNNER_MODULE
    if _RUNNER_MODULE is not None:
        return _RUNNER_MODULE

    script_path = _runner_path()
    if not script_path.exists():
        raise FileNotFoundError(f"Experiment script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("_qofo_runner", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {script_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _RUNNER_MODULE = mod
    return mod


def get_run_multi_tso_dso() -> Callable[..., Any]:
    """Return the cached ``run_multi_tso_dso`` callable.

    Raises
    ------
    AttributeError
        If the loaded module does not define ``run_multi_tso_dso``.
    """
    mod = get_runner_module()
    if not hasattr(mod, "run_multi_tso_dso"):
        raise AttributeError(
            "Loaded runner module has no `run_multi_tso_dso` attribute"
        )
    return mod.run_multi_tso_dso
