"""tuning/scripts/audit_design_set.py — excitation gate for a design set.

Why this exists
---------------
A tuning weight whose actuator never moves has no leverage on any objective, so
no amount of budget can identify it.  That is not hypothetical: across 1555
recorded scenario-runs the OLTC taps were frozen in 77 % of clean runs, and a
direct measurement of ``nominal_quiet`` produced **1 TSO tap and 0 DSO taps**.
Both OLTC weights were therefore unidentifiable *by construction*, and every
trial spent on them was wasted.

The remedy is to make excitation a **measurable admission criterion** rather
than an assumption.  Each candidate scenario is run once at the reference
weights and admitted only if it exercises the actuator classes being tuned.

Publishing the resulting table is itself worth doing: "we verified the design
set excites every tuned actuator class" is a methodological statement, and its
absence is exactly what produced the defect above.

Usage::

    python -m tuning.scripts.audit_design_set                 # tune_set_v2
    python -m tuning.scripts.audit_design_set --set design    # legacy set
    python -m tuning.scripts.audit_design_set --csv out.csv
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]

DEFAULT_BASELINE = _SCRIPT_DIR / "configs" / "baseline_ieee39.yaml"


@dataclass(frozen=True)
class ExcitationCriteria:
    """Thresholds a scenario must meet to be admitted.

    Each maps to one tuned actuator class, so a scenario that fails a criterion
    cannot inform the weight that governs that class.
    """

    min_tap_moves_tso: int = 1
    """Machine-transformer OLTCs must move at least once, or
    ``g_w_tso_oltc`` is unidentifiable from this scenario."""

    min_tap_moves_dso: int = 1
    """Same for the network-coupling OLTCs and ``g_w_dso_oltc``."""

    max_gen_q_reserve: float = 0.15
    """Generator reactive reserve must drop below this at some point.  Until
    the continuous actuators approach saturation, the tap changers are never
    the binding authority and the OLTC weights stay inert."""

    min_peak_v_dev_pu: float = 0.01
    """Peak |V - V_set| on the TS.  Below this the voltage objective is
    already satisfied and the tracking weights have nothing to trade off."""


@dataclass(frozen=True)
class ScenarioAudit:
    name: str
    network: str
    n_records: int
    tap_moves_tso: int
    tap_moves_dso: int
    min_gen_q_reserve: float
    peak_v_dev_pu: float
    band_touches: int
    feasible: bool
    failure: str = ""

    def verdict(self, c: ExcitationCriteria) -> tuple[bool, list[str]]:
        """Per-scenario admissibility.

        **The only per-scenario requirement is that the reference completes the
        run.**  A scenario the known-good weights cannot survive is not a
        discriminator: every candidate fails it identically, so it contributes
        noise and wall-clock but no information about the weights.

        Excitation is deliberately *not* judged here.  A weight is identifiable
        if **some** scenario exercises its actuator, not if every one does —
        this module said so from the start and the first version of this method
        then demanded it of each scenario anyway, rejecting a perfectly good set.
        Excitation is a **set-level** property; see :func:`set_level_verdict`.
        """
        if not self.feasible:
            return False, [
                "diverged at the REFERENCE weights — not a valid test case, "
                "since every candidate would fail it identically"
            ]
        return True, []


def set_level_verdict(
    audits: Sequence[ScenarioAudit],
    c: ExcitationCriteria,
) -> tuple[bool, list[str]]:
    """Whether the **set as a whole** can identify each tuned weight.

    This is the criterion that matters.  Each tuned actuator class needs at
    least one scenario that moves it; a quiescent scenario alongside is
    desirable, not a defect, since tuning that only ever sees stress drifts
    toward a controller that only behaves under stress.
    """
    ok = [a for a in audits if a.feasible]
    lines: list[str] = []
    passed = True

    n_tso = sum(1 for a in ok if a.tap_moves_tso >= c.min_tap_moves_tso)
    n_dso = sum(1 for a in ok if a.tap_moves_dso >= c.min_tap_moves_dso)
    n_sat = sum(1 for a in ok if a.min_gen_q_reserve < c.max_gen_q_reserve)
    n_dev = sum(1 for a in ok if a.peak_v_dev_pu >= c.min_peak_v_dev_pu)

    for label, count, weight in (
        ("moves machine-trafo OLTCs", n_tso, "g_w_tso_oltc"),
        ("moves coupling-trafo OLTCs", n_dso, "g_w_dso_oltc"),
        ("saturates generator reactive reserve", n_sat,
         "the OLTC weights (until continuous authority saturates the tap "
         "changers are never the binding actuator)"),
        ("produces a material voltage excursion", n_dev,
         "the voltage tracking weights"),
    ):
        status = "ok" if count else "NONE"
        lines.append(f"  {label:44s} {count}/{len(ok)} scenarios  [{status}]")
        if not count:
            passed = False
            lines.append(f"      -> {weight} is UNIDENTIFIABLE from this set")
    return passed, lines


def _audit_one(scenario, cfg, scales) -> ScenarioAudit:
    from tuning.objectives_v2 import _run_scenario

    res, records = _run_scenario(scenario, cfg, scales)
    if not records:
        return ScenarioAudit(
            scenario.name, scenario.scenario, 0, 0, 0,
            float("nan"), float("nan"), 0, False,
            res.failure_reason.splitlines()[0][:120]
            if res.failure_reason else "empty log",
        )

    m = res.metrics
    v_set = float(cfg.v_setpoint_pu)

    reserves: list[float] = []
    for r in records:
        for arr in r.gen_q_reserve.values():
            vals = np.atleast_1d(np.asarray(arr, dtype=float))
            finite = vals[np.isfinite(vals)]
            if finite.size:
                reserves.append(float(finite.min()))
    min_reserve = min(reserves) if reserves else float("nan")

    devs: list[float] = []
    for r in records:
        for lo, hi in zip(r.zone_v_min.values(), r.zone_v_max.values()):
            for v in (lo, hi):
                if v is not None and math.isfinite(float(v)):
                    devs.append(abs(float(v) - v_set))
    peak_dev = max(devs) if devs else float("nan")

    return ScenarioAudit(
        name=scenario.name,
        network=scenario.scenario,
        n_records=m.n_records,
        tap_moves_tso=int(m.n_tap_switches_tso),
        tap_moves_dso=int(m.n_tap_switches_dso),
        min_gen_q_reserve=min_reserve,
        peak_v_dev_pu=peak_dev,
        band_touches=int(m.n_viol_v_ts + m.n_viol_v_ds),
        feasible=bool(m.feasible),
        failure=res.failure_reason.splitlines()[0][:120]
        if res.failure_reason else "",
    )


def run_audit(
    scenarios: Sequence,
    baseline: Path,
    criteria: ExcitationCriteria | None = None,
) -> list[ScenarioAudit]:
    from tuning._io import load_config_yaml
    from tuning.metrics import MetricScales
    from tuning.parameters import FIXED_OVERRIDES

    criteria = criteria or ExcitationCriteria()
    cfg = load_config_yaml(baseline)
    cfg = dataclasses.replace(cfg, **FIXED_OVERRIDES)
    scales = MetricScales()

    audits: list[ScenarioAudit] = []
    for sc in scenarios:
        print(f"  running {sc.name} ...", flush=True)
        audits.append(_audit_one(sc, cfg, scales))
    return audits


def _print_table(audits: Sequence[ScenarioAudit],
                 criteria: ExcitationCriteria) -> bool:
    print()
    print(f"{'scenario':24s}{'network':11s}{'recs':>6s}{'tapTS':>7s}"
          f"{'tapDS':>7s}{'minQres':>9s}{'peak|dV|':>10s}{'verdict':>10s}")
    print("-" * 84)
    all_ok = True
    for a in audits:
        ok, reasons = a.verdict(criteria)
        all_ok &= ok
        print(f"{a.name:24s}{a.network:11s}{a.n_records:6d}"
              f"{a.tap_moves_tso:7d}{a.tap_moves_dso:7d}"
              f"{a.min_gen_q_reserve:9.3f}{a.peak_v_dev_pu:10.4f}"
              f"{'ADMIT' if ok else 'REJECT':>10s}")
        for reason in reasons:
            print(f"{'':24s}  -> {reason}")

    print()
    print("Set-level excitation (the criterion that decides identifiability):")
    set_ok, lines = set_level_verdict(audits, criteria)
    for line in lines:
        print(line)
    tot_tso = sum(a.tap_moves_tso for a in audits if a.feasible)
    tot_dso = sum(a.tap_moves_dso for a in audits if a.feasible)
    print(f"  total tap moves: TSO {tot_tso}, DSO {tot_dso}")
    return all_ok and set_ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuning.scripts.audit_design_set")
    p.add_argument("--set", dest="which", default="tune_v2",
                   choices=("tune_v2", "design", "holdout_v2"))
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--n", type=int, default=0,
                   help="Limit the number of scenarios (0 = all).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args(argv)

    from tuning.scenarios import design_set, holdout_set_v2, tune_set_v2

    scenarios = {
        "tune_v2": tune_set_v2,
        "design": design_set,
        "holdout_v2": lambda: holdout_set_v2(args.seed, 40),
    }[args.which]()
    if args.n:
        scenarios = scenarios[:args.n]

    print(f"[audit] {args.which}: {len(scenarios)} scenarios, "
          f"baseline={args.baseline}")
    criteria = ExcitationCriteria()
    audits = run_audit(scenarios, args.baseline, criteria)
    all_ok = _print_table(audits, criteria)

    if args.csv:
        import csv

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(
                fh, fieldnames=[f.name for f in dataclasses.fields(ScenarioAudit)])
            w.writeheader()
            for a in audits:
                w.writerow(dataclasses.asdict(a))
        print(f"[audit] wrote {args.csv}")

    print()
    print("[audit] GATE PASSED" if all_ok else
          "[audit] GATE FAILED -- do not start a tuning run on this set. "
          "A weight whose actuator never moves cannot be identified, however "
          "large the budget.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
