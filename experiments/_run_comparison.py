#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Turn-key comparison of the H-predictor modes (Constant H / Kalman / ANN),
each in its OWN subprocess (parallel, and isolated -- no in-process state
leakage between modes), then a timestamped multi-panel figure that pops up.

    python experiments/_run_comparison.py            # changing scenario (default)
    python experiments/_run_comparison.py frozen      # frozen-OP scenario

If TensorFlow is unavailable the `ann` leg just exits non-zero and is skipped;
drop it from MODES to silence it.  .venv312.
"""
from __future__ import annotations
import os, sys, glob, importlib, subprocess
from datetime import datetime

_here = os.path.dirname(os.path.abspath(__file__)); _root = os.path.dirname(_here)
os.chdir(_root); sys.path.insert(0, _root)
RES = os.path.join(_root, "results", "003_cigre_2026")
LEG = os.path.join(_here, "_run_cmp_leg.py")

SCEN  = sys.argv[1] if len(sys.argv) > 1 else "changing"
MODES = {"identity": "Constant H", "kalman": "Kalman", "ann": "ANN"}

print(f"launching {len(MODES)} modes in parallel ({SCEN}): {list(MODES)}")
env = dict(os.environ, PYTHONIOENCODING="utf-8")
procs = {m: subprocess.Popen([sys.executable, LEG, m, SCEN], env=env) for m in MODES}
for m, p in procs.items():
    p.wait()
    print(f"  {m:9s} exited ({p.returncode})")

ana = importlib.import_module("experiments.003_analysis")
ana.plot_comparison = lambda *a, **k: None

runs = []
for m, label in MODES.items():
    pkl  = os.path.join(RES, f"cmp_{SCEN}_{m}.pkl")
    side = os.path.join(RES, f"cmp_{SCEN}_{m}.npz")
    if os.path.exists(pkl):
        runs.append((pkl, side if os.path.exists(side) else None, label))
    else:
        print(f"  [cmp] {m} produced no pkl -- skipped")

if not runs:
    print("[cmp] no successful runs -- nothing to plot."); sys.exit(1)

import numpy as np
print(f"\n  === {SCEN} ===")
print(f"  {'mode':12s}{'H-err q_trafo fin20':>22s}{'one-step skill':>16s}")
for pkl, side, label in runs:
    if side is None:
        print(f"  {label:12s}{'(no sidecar)':>22s}"); continue
    d = np.load(side)
    he = ana.h_analytical_error(d); r = ana.one_step_qtrafo_error(d)
    hq = float(np.mean(he["relative_q_trafo"][-20:])) if he and "relative_q_trafo" in he else float("nan")
    sk = f"{r['skill']:+.3f}" if r else "n/a"
    print(f"  {label:12s}{hq:>22.3f}{sk:>16s}")

ts  = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
png = os.path.join(RES, f"comparison_{SCEN}_{ts}.png")
ana.plot_multi_comparison(runs, save_path=png)
print("done ->", png)

try:
    os.startfile(png)
except AttributeError:
    subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", png], check=False)
