#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/diag_v5_curvature.py
================================
Fast, read-only probe of the V5 (single centralized OFO) closed-loop
**curvature** spectrum, used to pick the global ``g_w`` cooling factor
``KAPPA_V5`` in ``experiments/005_CIGRE_MULTI.py``.

One OFO tick (unconstrained, slack/usage dropped) is
``sigma* = -G_w^{-1} H_V^T diag(g_v) (V - V*)``, so the per-tick
voltage-error map is ``e_{k+1} = (I - M) e_k`` with

    M = H_V G_w^{-1} H_V^T diag(g_v).

OFO is stable iff ``eig(M) ⊂ (0, 2)`` and well-damped for
``lambda_max(M) ≲ 1``.  The runner prints the eigenvalues of M once,
right after the central controller is initialised (guarded by
``MultiTSOConfig.debug_central_curvature``); this script enables that
flag and installs a ``pre_loop_hook`` that returns truthy, so the 600-step
simulation loop is skipped — the probe builds the plant + Jacobian +
central controller, dumps ``lambda_max`` / ``lambda_min`` / cond / suggested
``kappa``, and exits in setup time only.

Usage
-----
    python experiments/diag_v5_curvature.py

Then set ``KAPPA_V5`` in ``005_CIGRE_MULTI.py`` to the printed
"suggested kappa" (or a fraction of it for a more aggressive reference)
and run the full V5 sweep.

Author: Manuel Schwenke / Claude Code
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib

from experiments.runners import run_multi_tso_dso

# Reuse the exact CIGRE config + V5 overrides from the case-study driver.
_cigre = importlib.import_module("experiments.005_CIGRE_MULTI")


def main() -> None:
    cfg = _cigre.make_cigre_config()
    for k, v in _cigre.VARIANTS["V5"].items():
        setattr(cfg, k, v)
    cfg.debug_central_curvature = True
    cfg.verbose = 1
    # No live plots / figures — this is a setup-only probe.
    cfg.live_plot_controller = False
    cfg.live_plot_cascade = False
    cfg.live_plot_system = False

    print("=" * 72)
    print("  V5 curvature probe (no simulation loop)")
    print(f"  g_v={cfg.g_v:g}  central_dso_g_v={cfg.central_dso_g_v:g}  "
          f"central_period_s={cfg.central_period_s}")
    print(f"  g_w der/dso_der/gen/tso_oltc/dso_oltc = "
          f"{cfg.g_w_der:g}/{cfg.g_w_dso_der:g}/{cfg.g_w_gen:g}/"
          f"{cfg.g_w_tso_oltc:g}/{cfg.g_w_dso_oltc:g}")
    print("=" * 72)

    # Returning truthy from the hook makes run_multi_tso_dso skip the main
    # loop (see its docstring); the curvature dump has already fired during
    # central_controller.initialise(...).
    run_multi_tso_dso(cfg, pre_loop_hook=lambda _state: True)


if __name__ == "__main__":
    main()
