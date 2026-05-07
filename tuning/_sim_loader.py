"""
tuning/_sim_loader.py
=====================
Internal helper: load ``run_multi_tso_dso`` once and cache it.

The runner module ``experiments.runners.multi_tso_dso`` defines
``run_multi_tso_dso`` and imports its private helpers (notably
``_run_delayed_stability_analysis``) into its module namespace.  We import
it through the regular ``import`` system and cache both the module
reference (for callers that need to monkey-patch internals) and the bare
``run_multi_tso_dso`` callable (the common case).

This module is **private** — not part of the public ``tuning`` API.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Callable

_RUNNER_MODULE_NAME = "experiments.runners.multi_tso_dso"
_RUNNER_MODULE: ModuleType | None = None


def get_runner_module() -> ModuleType:
    """Return the cached runner module.

    First call imports ``experiments.runners.multi_tso_dso``; subsequent
    calls return the cache.

    Callers that need to monkey-patch internals (e.g.
    :mod:`tuning.ceilings` patches ``_run_delayed_stability_analysis`` to
    capture its return value) should use this helper rather than
    :func:`get_run_multi_tso_dso`.  The patch must target the runner
    module's namespace because ``run_multi_tso_dso`` resolves the helper
    via its own module globals at call time.

    Raises
    ------
    ImportError
        If the runner module cannot be imported.
    """
    global _RUNNER_MODULE
    if _RUNNER_MODULE is not None:
        return _RUNNER_MODULE

    _RUNNER_MODULE = importlib.import_module(_RUNNER_MODULE_NAME)
    return _RUNNER_MODULE


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
