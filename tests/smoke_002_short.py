"""Quick smoke test for the five scenarios in 002_M_TSO_M_DSO_COMPARE.

Runs every scenario with a shortened simulation horizon (10 minutes,
before the first contingency at minute 60) just to verify install +
warmup + first few TSO/DSO steps converge.  Use the full sweep
(``python experiments/002_M_TSO_M_DSO_COMPARE.py``) for the actual
comparison.
"""
from __future__ import annotations

import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mod = importlib.import_module("experiments.002_M_TSO_M_DSO_COMPARE")

ROOT = os.path.join("results", "_smoke_002_short")
os.makedirs(ROOT, exist_ok=True)


def run_one(name: str) -> tuple[int, bool, float]:
    cfg = mod.make_base_config()
    for k, v in mod.SCENARIOS[name].items():
        setattr(cfg, k, v)
    cfg.n_total_s = 60 * 10
    cfg.result_dir = os.path.join(ROOT, name)
    os.makedirs(cfg.result_dir, exist_ok=True)
    cfg.verbose = 0

    t0 = time.perf_counter()
    try:
        log = mod.run_multi_tso_dso(cfg)
        ok = True
        n = len(log)
    except Exception as exc:
        ok = False
        n = 0
        print(f"  [{name}] FAILED: {type(exc).__name__}: {exc}")
    dt = time.perf_counter() - t0
    print(f"  [{name}] n_steps={n}  converged={ok}  wall={dt:5.1f} s")
    return n, ok, dt


if __name__ == "__main__":
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(mod.SCENARIOS.keys())
    print(f"Smoke 002 short: scenarios={only}")
    print(f"  n_total_s = 600 s (10 min); contingencies start at min 60 so")
    print(f"  this only exercises baseline operation + initial warmup.")
    print()
    for name in only:
        run_one(name)
