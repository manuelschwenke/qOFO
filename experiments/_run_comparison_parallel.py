#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_comparison.py
====================
Run the *identity* and *kalman* H-predictor modes side-by-side and produce
a comparison figure.  The two modes share the same ``exp`` module globals
(``H_PREDICTOR_MODE``, ``_kalman_h_predictor``, …), so they **must** execute
in separate OS processes; we use :mod:`concurrent.futures.ProcessPoolExecutor`
(backed by loky) for that isolation.

Output-file disambiguation
--------------------------
``exp.run()`` names both output files with the *same* ``datetime.now()``
timestamp:

    <YYYY-MM-DD--HH-MM-SS>_<mode>[_frozen][_biasedXXpct].pkl
    <YYYY-MM-DD--HH-MM-SS>_dso2_ctrl.npz

Because the two processes start at slightly different wall-clock times their
``now`` stems are naturally distinct — no collision.  Each worker returns its
``(pkl_path, sidecar_path)`` tuple directly, so the main process never needs
to reconstruct the names via glob-splitting.
"""

import os
import sys
import subprocess
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib as mpl

if not os.environ.get("MPLBACKEND"):
    os.environ["QT_API"] = "pyqt5"
    try:
        mpl.use("Qt5Agg", force=True)
    except (ImportError, ValueError):
        pass

import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)

import importlib
ana = importlib.import_module("experiments.003_analysis")
ana.plot_comparison = lambda *a, **k: None  # silence per-run pop-ups

RES = os.path.join(_root, "results", "003_cigre_2026")

MODES = {
    "identity": "Constant H",
    "kalman":   "Kalman",
    # "ann":    "ANN",
}


# ---------------------------------------------------------------------------
# Worker — runs in its own OS process; all exp globals are process-local
# ---------------------------------------------------------------------------

def _run_one_mode(mode: str) -> tuple:
    """Configure and run one H-predictor mode; return (pkl_path, sidecar_path).

    Executed in a child process.  Returns a 2-tuple:
        pkl_path     : absolute path to the written .pkl file
        sidecar_path : absolute path to the written _dso2_ctrl.npz, or None

    Raises on failure so the parent's ``future.result()`` re-raises it.
    """
    import os, sys, importlib
    from datetime import datetime

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(_root)
    sys.path.insert(0, _root)

    exp = importlib.import_module("experiments.003_S_DSO_CIGRE_2026")

    # Shared overrides (identical to the original sequential script)
    exp.H_INIT_BIAS_STD,  exp.H_INIT_BIAS_SEED  = 0.5, 64
    exp.H_PREDICTOR_ROWS, exp.FROZEN_OP_POINT   = "all", False
    exp.KALMAN_NIS_DETECT_ENABLED = True

    # Mode-specific global
    exp.H_PREDICTOR_MODE = mode

    # Patch make_config
    _orig_make_config = exp.make_config
    def _cfg(_o=_orig_make_config):
        cfg = _o()
        cfg.n_total_s, cfg.dso_period_s = 600 * 60.0, 20.0
        cfg.start_time = datetime(2016, 9, 7, 8, 0)
        cfg.contingencies = [
            exp.ContingencyEvent(
                minute=120, element_type="line", element_index=49, action="trip"
            ),
        ]
        return cfg
    exp.make_config = _cfg

    # Snapshot result dir before run to find new files afterwards
    out_dir = exp.make_config().result_dir
    os.makedirs(out_dir, exist_ok=True)
    before = set(os.listdir(out_dir))

    exp.run()   # <<< simulation + file I/O >>>

    after    = set(os.listdir(out_dir))
    new      = sorted(after - before,
                      key=lambda f: os.path.getmtime(os.path.join(out_dir, f)))

    pkl_files = [f for f in new if f.endswith(".pkl")]
    npz_files = [f for f in new if f.endswith("_dso2_ctrl.npz")]

    if not pkl_files:
        raise RuntimeError(
            f"[cmp/{mode}] run() completed but no new .pkl found in {out_dir!r}. "
            f"New files: {new}"
        )

    pkl_path     = os.path.join(out_dir, pkl_files[-1])
    sidecar_path = os.path.join(out_dir, npz_files[-1]) if npz_files else None
    return pkl_path, sidecar_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(RES, exist_ok=True)
    runs = []

    with ProcessPoolExecutor(max_workers=len(MODES)) as pool:
        future_to_mode = {
            pool.submit(_run_one_mode, mode): (mode, label)
            for mode, label in MODES.items()
        }

        for future in as_completed(future_to_mode):
            mode, label = future_to_mode[future]
            try:
                pkl_path, sidecar_path = future.result()
                print(f"[cmp] {mode!r} done -> {pkl_path}")
                runs.append((pkl_path, sidecar_path, label))
            except Exception:
                print(f"[cmp] mode={mode!r} FAILED (skipping):\n"
                      f"{traceback.format_exc()}")

    if not runs:
        print("[cmp] no successful runs — nothing to plot.")
        sys.exit(1)

    # Restore stable order (identity first, kalman second, …)
    _order = list(MODES.keys())

    def _mode_rank(entry):
        fname = os.path.basename(entry[0])
        for i, k in enumerate(_order):
            if f"_{k}" in fname:
                return i
        return len(_order)

    runs.sort(key=_mode_rank)

    ts  = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    png = os.path.join(RES, f"comparison_changing_{ts}.png")

    ana.plot_multi_comparison(runs, save_path=png)
    print("done ->", png)

    if mpl.get_backend().lower() in ("agg", "pdf", "ps", "svg", "template", "cairo"):
        try:
            os.startfile(png)
        except AttributeError:
            subprocess.run(
                ["open" if sys.platform == "darwin" else "xdg-open", png],
                check=False,
            )
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()