#!/usr/bin/env python3
"""Does each reduced (Ward) network reproduce the combined solution?

The reduction exists to give each controller a cached linear model of its own
zone. It is only valid if the reduced network, solved on its own, lands on the
SAME operating point as the combined network at the buses it retains. If it
does not, every sensitivity derived from it is linearised about a point the
plant is not at.

"Does it converge" is NOT that test -- a reduced net with stranded boundary
flows can converge happily to a different operating point. This checks the
property directly: it captures ``res_bus`` immediately before the reduction
solves (those values are still the combined solution, carried through the
deepcopy) and compares them against the solved reduced net.

Runs the static plant only; no PowerFactory is involved, so it is safe to use
while a co-simulation is in flight.

Usage::

    python tools/check_reduction_fidelity.py
    python tools/check_reduction_fidelity.py --balance      # with the fix on
    python tools/check_reduction_fidelity.py --window "2016-05-01 16:00"

Author: Manuel Schwenke / Claude Code (2026-08-01)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sensitivity.network_reduction as NR  # noqa: E402
from experiments.helpers.rms_cosim_config import make_cosim_config  # noqa: E402
from experiments.runners import run_multi_tso_dso  # noqa: E402

STUDY_WINDOWS = [
    "2016-02-22 13:00", "2016-01-05 08:00",
    "2016-01-15 03:00", "2016-12-18 14:00",
]

_orig = NR.runpp_with_stored_jacobian
_findings: List[dict] = []


def _say(msg: str) -> None:
    sys.__stdout__.write(msg + "\n")
    sys.__stdout__.flush()


def _checking_solve(net, *a, **kw):
    """Capture the combined-solution voltages, solve, then compare."""
    ref = None
    if net.res_bus is not None and not net.res_bus.empty:
        ref = net.res_bus["vm_pu"].copy()
    n_ward = 0
    if len(net.load) and "name" in net.load:
        n_ward = int(net.load["name"].astype(str)
                     .str.startswith("WARD").sum())
    try:
        out = _orig(net, *a, **kw)
    except Exception as exc:                                   # noqa: BLE001
        _findings.append({"ok": False, "err": type(exc).__name__,
                          "buses": len(net.bus), "ward": n_ward})
        raise
    rec = {"ok": True, "buses": len(net.bus), "ward": n_ward,
           "dmax": float("nan"), "dmean": float("nan"), "bus": -1}
    if ref is not None:
        common = ref.index.intersection(net.res_bus.index)
        a_ = ref.loc[common].to_numpy(dtype=float)
        b_ = net.res_bus.loc[common, "vm_pu"].to_numpy(dtype=float)
        m = np.isfinite(a_) & np.isfinite(b_) & (a_ > 0.01) & (b_ > 0.01)
        if m.any():
            d = np.abs(a_[m] - b_[m])
            rec["dmax"] = float(d.max())
            rec["dmean"] = float(d.mean())
            rec["bus"] = int(np.asarray(common)[m][int(np.argmax(d))])
    _findings.append(rec)
    return out


def check(window: str, balance: bool) -> None:
    cfg = make_cosim_config(20.0, verbose=0)
    cfg.scenario = "rural_700"
    cfg.start_time = datetime.strptime(window, "%Y-%m-%d %H:%M")
    cfg.use_profiles = True
    cfg.use_zonal_gen_dispatch = False
    cfg.der_q_capability_override_pu = None
    cfg.g_w_dso_oltc = 200.0
    for k in ("tso_qv_deadband_pu", "dso_qv_deadband_pu",
              "der_qv_deadband_override_pu"):
        setattr(cfg, k, 0.005)
    cfg.dso_der_scale = {"DSO_3": 2.0}
    cfg.dso_load_p_scale = {"DSO_3": 2.0}
    cfg.reduction_balance_to_cached = balance

    _findings.clear()
    NR.runpp_with_stored_jacobian = _checking_solve
    buf = io.StringIO()
    verdict = "ok"
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            run_multi_tso_dso(cfg)
    except Exception as exc:                                   # noqa: BLE001
        verdict = type(exc).__name__
    finally:
        NR.runpp_with_stored_jacobian = _orig

    _say(f"\n{window}   balance={balance}   runner: {verdict}")
    _say(f"  {'net':>4} {'buses':>6} {'ward':>5} {'max dV [pu]':>12} "
         f"{'mean dV':>10}  worst bus")
    for i, f in enumerate(_findings):
        if not f["ok"]:
            _say(f"  {i:>4} {f['buses']:>6} {f['ward']:>5} "
                 f"{'DIVERGED':>12} {f['err']:>10}")
            continue
        flag = ""
        if np.isfinite(f["dmax"]):
            flag = "   <-- MISMATCH" if f["dmax"] > 1e-3 else "   ok"
        _say(f"  {i:>4} {f['buses']:>6} {f['ward']:>5} {f['dmax']:12.6f} "
             f"{f['dmean']:10.6f}  {f['bus']:>4}{flag}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", action="append", default=None)
    ap.add_argument("--balance", action="store_true",
                    help="enable reduction_balance_to_cached")
    ap.add_argument("--both", action="store_true",
                    help="run each window with the flag off and on")
    args = ap.parse_args(argv)

    windows = args.window or STUDY_WINDOWS
    _say("Reduced-network fidelity: does the reduced net reproduce the")
    _say("combined solution at the buses it keeps?  (tolerance 1e-3 pu)")
    for w in windows:
        for bal in ((False, True) if args.both else (args.balance,)):
            check(w, bal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
