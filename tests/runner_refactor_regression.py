"""Gate-E full-runner regression: the plant-argument refactor of
``experiments/runners/multi_tso_dso.py`` must not change a single bit of an
experiment run.

Two-phase tool (not a pytest -- the reference must be generated with the
PRE-refactor code, the check with the post-refactor code):

    python tests\\runner_refactor_regression.py make    # freeze reference
    python tests\\runner_refactor_regression.py check   # compare re-run

The scenario: default `MultiTSOConfig` with a short window (n_total_s = 360
-> 18 steps at the 20 s STS cadence, two 180 s TSO firings), fixed noise
seed, verbose 0.  The full iteration log (every `MultiTSOIterationRecord`)
is compared recursively: floats/ints exactly, numpy arrays with
`array_equal(equal_nan=True)` -- bit-for-bit, no tolerances.

Author: Manuel Schwenke / Claude Code (2026-07-20)
"""

from __future__ import annotations

import dataclasses
import pickle
import sys
from pathlib import Path
from typing import Any, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REFERENCE = (Path(__file__).resolve().parents[1] / "results"
             / "plant_refactor_reference" / "reference_log.pkl")


def _make_config():
    from configs.config import MultiTSOConfig
    return MultiTSOConfig(
        n_total_s=360.0,       # 18 steps @ 20 s STS; TSO fires at 180/360 s
        verbose=0,
    )


def _run() -> List[Any]:
    from experiments.runners import run_multi_tso_dso
    return run_multi_tso_dso(_make_config())


def _diff(path: str, a: Any, b: Any, out: List[str]) -> None:
    """Recursive exact comparison; appends human-readable diffs to ``out``."""
    if type(a) is not type(b) and not (
            isinstance(a, (int, float, np.floating, np.integer))
            and isinstance(b, (int, float, np.floating, np.integer))):
        out.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
        return
    if isinstance(a, np.ndarray):
        if a.shape != b.shape:
            out.append(f"{path}: shape {a.shape} != {b.shape}")
        elif a.dtype.kind == "f" or b.dtype.kind == "f":
            if not np.array_equal(a, b, equal_nan=True):
                idx = np.unravel_index(
                    int(np.argmax(~np.isclose(a, b, rtol=0, atol=0,
                                              equal_nan=True))), a.shape)
                out.append(f"{path}: arrays differ, first at {idx}: "
                           f"{a[idx]!r} != {b[idx]!r}")
        elif not np.array_equal(a, b):
            out.append(f"{path}: arrays differ")
        return
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        for f in dataclasses.fields(a):
            # Wall-clock diagnostics can never reproduce; everything else
            # (including simulated time) must be bit-identical.
            if "solve_s" in f.name or "solve_time" in f.name \
                    or f.name.endswith("_wall_s"):
                continue
            _diff(f"{path}.{f.name}", getattr(a, f.name), getattr(b, f.name),
                  out)
        return
    if isinstance(a, dict):
        if set(a) != set(b):
            out.append(f"{path}: keys {sorted(map(str, set(a) ^ set(b)))} "
                       f"differ")
            return
        for k in a:
            _diff(f"{path}[{k!r}]", a[k], b[k], out)
        return
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(f"{path}[{i}]", x, y, out)
        return
    if isinstance(a, (float, np.floating)):
        if not (a == b or (np.isnan(a) and np.isnan(b))):
            out.append(f"{path}: {a!r} != {b!r}")
        return
    if a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "make":
        log = _run()
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        with REFERENCE.open("wb") as fh:
            pickle.dump(log, fh)
        print(f"[make] reference frozen: {len(log)} records -> {REFERENCE}")
        return 0

    if not REFERENCE.exists():
        print(f"[check] no reference at {REFERENCE}; run 'make' first "
              f"(with the PRE-refactor code)")
        return 2
    with REFERENCE.open("rb") as fh:
        ref = pickle.load(fh)
    log = _run()
    diffs: List[str] = []
    if len(ref) != len(log):
        diffs.append(f"log length {len(ref)} != {len(log)}")
    else:
        for i, (a, b) in enumerate(zip(ref, log)):
            _diff(f"log[{i}]", a, b, diffs)
    if diffs:
        print(f"[check] FAIL -- {len(diffs)} difference(s), first 20:")
        for d in diffs[:20]:
            print(f"  {d}")
        return 1
    print(f"[check] PASS -- {len(log)} records bit-identical to the "
          f"pre-refactor reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
