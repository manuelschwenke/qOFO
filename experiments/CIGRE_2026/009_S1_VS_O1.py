"""
009_S1_VS_O1.py
===============

The controlled control-law comparison: classical pilot-node SVR (``S1``) against
TS-OFO on the continuous actuators only (``O1``).

Both variants command the same actuators, leave the TS taps to the same
rule-based ``DiscreteTapControl`` logic (``tso_oltc_mode='local'``), sit above
the same reference-anchored q(v) layer, run at the same dispatch epochs, and
regulate transmission voltages against the same references.  The ONLY thing
that differs is the control law.  That is what makes this the one place in the
study where the optimisation-based formulation is measured against the deployed
alternative on equal terms (thesis ``S1 -> O1``).

Metrics
-------
Primary, per TSO control area -- the time-mean spatial RMS voltage-tracking
error over the area's reference bus set ``R_a``:

    e_v,a = sqrt( mean_t [ zone_v_rms_err_pu[a](t)^2 ] )

Secondary -- the same quantity on the pilot bus alone:

    e_v,p,a = sqrt( mean_t [ (v_p,a(t) - v_ref)^2 ] )

The pair is the point of the comparison, not a formality.  ``e_v,p,a`` is S1's
OWN objective and S1 may well win it; ``e_v,a`` asks how the rest of the area
follows when it does.  The gap between them is the cost of the single-pilot-node
surrogate.  Report both even -- especially -- when S1 comes out ahead on the
pilot bus.

Run:
    python -m experiments.CIGRE_2026.009_S1_VS_O1              # run + report
    python -m experiments.CIGRE_2026.009_S1_VS_O1 --report     # report cached
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from experiments.paths import RESULTS_ROOT

from experiments.CIGRE_2026 import __name__ as _pkg  # noqa: F401  (package guard)

import importlib.util as _u

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = _u.spec_from_file_location("m005", os.path.join(_HERE, "005_CIGRE_MULTI.py"))
_m005 = _u.module_from_spec(_spec)
_spec.loader.exec_module(_m005)

PAIR = ("S1", "O1")


def resolve_out_root(explicit: Optional[str] = None) -> str:
    """Locate the directory holding the logs to report on.

    ``005_CIGRE_MULTI.main()`` reassigns its module-level ``OUT_ROOT`` at
    runtime to a FRESH timestamped run directory
    (``results/cigre_2026_005/<NNNN>_<timestamp>/``, see ``new_run_dir``).
    Importing 005 here therefore yields only the module-level *default*
    (``results/005_cigre``), which is where a direct ``run_one`` call writes but
    NOT where a full sweep does.  Trusting that constant makes ``--report`` look
    in the wrong place and report "no records" for a sweep that succeeded.

    Resolution order: explicit ``--run-dir``, then the newest
    ``results/cigre_2026_005/*`` that actually contains a log for the pair,
    then the module default.
    """
    if explicit:
        return explicit

    base = Path(RESULTS_ROOT) / "cigre_2026_005"
    if base.is_dir():
        runs = sorted((d for d in base.iterdir() if d.is_dir()), reverse=True)
        for d in runs:
            if any((d / n / "log.pkl").exists() for n in PAIR):
                return str(d)

    return _m005.OUT_ROOT


OUT_ROOT = resolve_out_root()


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------


def _time_rms(values: Sequence[float]) -> float:
    """sqrt of the time-mean square.  NaNs are dropped, not zero-filled."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr ** 2)))


def zone_tracking_error(log: List, zone: int) -> float:
    """e_v,a -- time-mean spatial RMS over the area's reference bus set."""
    return _time_rms([
        r.zone_v_rms_err_pu.get(zone, float("nan"))
        for r in log if getattr(r, "zone_v_rms_err_pu", None)
    ])


def pilot_tracking_error(log: List, pilot_bus: int, v_ref: float) -> float:
    """e_v,p,a -- time-mean RMS on the pilot bus alone."""
    errs = []
    for r in log:
        vm = getattr(r, "bus_vm_pu", None)
        if not vm or pilot_bus not in vm:
            continue
        errs.append(float(vm[pilot_bus]) - v_ref)
    return _time_rms(errs)


def tap_operations(log: List) -> int:
    """Total TS tap movements over the horizon (both variants: local logic)."""
    n = 0
    prev: Dict[int, np.ndarray] = {}
    for r in log:
        for z, taps in (getattr(r, "zone_oltc_taps", {}) or {}).items():
            t = np.asarray(taps, dtype=float)
            if z in prev and prev[z].shape == t.shape:
                n += int(np.sum(np.abs(t - prev[z]) > 0.5))
            prev[z] = t
    return n


def reserve_spread(log: List, zone: int) -> float:
    """Time-mean within-area spread of NORMALISED reactive reserve.

    Uses ``gen_q_reserve`` and ``tso_der_q_reserve``, which are
    ``min(q_max - q, q - q_min) / (q_max - q_min)`` at the current operating
    point, so devices of different rating are comparable.

    This is the metric the alignment law of \\eqref{eq:svr-alignment} speaks to:
    it equalises RELATIVE utilisation, so S1 is expected to show the smaller
    spread.  An earlier version of this function took the standard deviation of
    |Q| in absolute Mvar, which cannot test that claim at all -- under perfect
    relative alignment, differently rated machines carry different absolute Q.
    Do not go back to it.

    Returns NaN where the area has fewer than two devices with a finite reserve
    (zone 2 carries one machine and one DER), which is reported as "n/a" rather
    than as a number.
    """
    spreads = []
    for r in log:
        parts = []
        for fld in ("gen_q_reserve", "tso_der_q_reserve"):
            v = (getattr(r, fld, {}) or {}).get(zone)
            if v is not None:
                parts.append(np.asarray(v, dtype=float).ravel())
        if not parts:
            continue
        a = np.concatenate(parts)
        a = a[np.isfinite(a)]
        if a.size >= 2:
            spreads.append(float(np.std(a)))
    return float(np.mean(spreads)) if spreads else float("nan")


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------


#: Keys allowed to differ between S1 and O1.  ``tso_mode`` is the control law
#: itself; the ``svr_*`` keys are inert when ``tso_mode='ofo'``.  Anything else
#: differing means the pair is no longer controlled and the comparison has
#: stopped measuring the control law alone.
ALLOWED_DIFFS = {
    "tso_mode",
    "svr_pilot_buses", "svr_t_rvr_s", "svr_t_rpr_s",
    "svr_deadband_pu", "svr_v_ref_pu",
}


def assert_controlled_pair() -> None:
    """Fail loudly if S1 and O1 differ in anything but the control law.

    This is the guard behind the whole comparison.  Any override added to one
    variant and not the other silently turns a control-law measurement into a
    measurement of two unrelated configurations, and nothing else in the
    pipeline would notice.
    """
    a, b = _m005.VARIANTS["S1"], _m005.VARIANTS["O1"]
    bad = []
    for k in sorted(set(a) | set(b)):
        if k in ALLOWED_DIFFS:
            continue
        if a.get(k, "<default>") != b.get(k, "<default>"):
            bad.append(f"    {k}: S1={a.get(k, '<default>')!r}  O1={b.get(k, '<default>')!r}")
    if bad:
        raise SystemExit(
            "S1 and O1 are NOT a controlled pair -- they differ in:\n"
            + "\n".join(bad)
            + "\n\nMirror the change in both variants, or add the key to "
              "ALLOWED_DIFFS with a reason if the asymmetry is intended."
        )
    print("guard: S1/O1 differ only in the control law -- controlled pair OK")


def load(name: str, root: Optional[str] = None) -> List:
    path = os.path.join(root or OUT_ROOT, name, "log.pkl")
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def report(logs: Dict[str, List]) -> None:
    pilots = _m005.PILOT_BUSES
    cfg = _m005.make_cigre_config()
    v_refs = {z: float(cfg.zone_v_setpoints_pu.get(z, cfg.v_setpoint_pu))
              for z in pilots}

    zones = sorted(pilots)
    missing = [n for n in PAIR if not logs.get(n)]
    if missing:
        print(f"!! no records for {', '.join(missing)} -- run without --report first")
        return

    print()
    print("=" * 74)
    print("  S1 (classical pilot-node SVR)   vs   O1 (TS-OFO, continuous only)")
    print("  Controlled step: only the control law differs.")
    print("=" * 74)
    for n in PAIR:
        print(f"  {n}: {len(logs[n])} records")

    # -- primary --------------------------------------------------------
    print("\n-- PRIMARY: e_v,a  [pu]  time-mean spatial RMS over R_a ------------")
    print(f"{'Zone':>5}  {'pilot':>6}  {'S1':>10}  {'O1':>10}  {'O1 vs S1':>10}")
    print("-" * 50)
    tot = {n: [] for n in PAIR}
    for z in zones:
        vals = {n: zone_tracking_error(logs[n], z) for n in PAIR}
        for n in PAIR:
            tot[n].append(vals[n])
        rel = ((vals["O1"] / vals["S1"] - 1.0) * 100.0
               if vals["S1"] and math.isfinite(vals["S1"]) and vals["S1"] > 0
               else float("nan"))
        print(f"{z:>5}  {pilots[z]:>6}  {vals['S1']:>10.5f}  {vals['O1']:>10.5f}  "
              f"{rel:>9.1f}%")
    agg = {n: float(np.sqrt(np.nanmean(np.asarray(tot[n]) ** 2))) for n in PAIR}
    rel = (agg["O1"] / agg["S1"] - 1.0) * 100.0 if agg["S1"] > 0 else float("nan")
    print("-" * 50)
    print(f"{'all':>5}  {'':>6}  {agg['S1']:>10.5f}  {agg['O1']:>10.5f}  {rel:>9.1f}%")

    # -- secondary ------------------------------------------------------
    print("\n-- SECONDARY: e_v,p,a  [pu]  pilot bus only (S1's own objective) ----")
    print(f"{'Zone':>5}  {'pilot':>6}  {'S1':>10}  {'O1':>10}  {'O1 vs S1':>10}")
    print("-" * 50)
    for z in zones:
        vals = {n: pilot_tracking_error(logs[n], pilots[z], v_refs[z]) for n in PAIR}
        rel = ((vals["O1"] / vals["S1"] - 1.0) * 100.0
               if vals["S1"] and math.isfinite(vals["S1"]) and vals["S1"] > 0
               else float("nan"))
        print(f"{z:>5}  {pilots[z]:>6}  {vals['S1']:>10.5f}  {vals['O1']:>10.5f}  "
              f"{rel:>9.1f}%")

    # -- supporting -----------------------------------------------------
    print("\n-- SUPPORTING ------------------------------------------------------")
    print(f"{'':>30}  {'S1':>10}  {'O1':>10}")
    print(f"{'TS tap operations':>30}  "
          f"{tap_operations(logs['S1']):>10d}  {tap_operations(logs['O1']):>10d}")
    for z in zones:
        s = {n: reserve_spread(logs[n], z) for n in PAIR}
        cells = {n: ("       n/a" if not math.isfinite(s[n]) else f"{s[n]:>10.4f}")
                 for n in PAIR}
        print(f"{'reserve spread, zone ' + str(z):>30}  {cells['S1']}  {cells['O1']}")

    # Surrogate gap: how much worse the area is than the pilot bus alone.
    print(f"\n{'':>30}  {'S1':>10}  {'O1':>10}")
    for z in zones:
        r = {}
        for n in PAIR:
            ea = zone_tracking_error(logs[n], z)
            ep = pilot_tracking_error(logs[n], pilots[z], v_refs[z])
            r[n] = ea / ep if ep and math.isfinite(ep) and ep > 0 else float("nan")
        print(f"{'e_v,a / e_v,p,a, zone ' + str(z):>30}  "
              f"{r['S1']:>10.2f}  {r['O1']:>10.2f}")

    print("\nReading: e_v,p,a is what the classical scheme optimises; e_v,a is what")
    print("the area does. Their RATIO is the cost of the single-pilot-node")
    print("surrogate -- a value well above 1 means the pilot bus is held while")
    print("the rest of the area is not. TS tap operations must be 0 for BOTH:")
    print("the taps are held at their planning position in this pair, so any")
    print("non-zero count means the discrete treatment is not actually equal.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="report cached logs without re-running")
    ap.add_argument("--run-dir", default=None,
                    help="directory holding <variant>/log.pkl; default is the "
                         "newest results/cigre_2026_005/* containing the pair")
    args = ap.parse_args()

    assert_controlled_pair()

    root = resolve_out_root(args.run_dir)
    logs: Dict[str, List] = {}
    for name in PAIR:
        if args.report:
            logs[name] = load(name, root)
        else:
            logs[name] = _m005.run_one(name, _m005.VARIANTS[name])
    if args.report:
        print(f"reading logs from: {root}")
    report(logs)


if __name__ == "__main__":
    main()
