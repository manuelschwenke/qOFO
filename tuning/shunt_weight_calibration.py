"""Calibrate the hybrid bank movement weight from recorded bank gradients.

Hybrid dispatch (``shunt_dispatch='hybrid'``) adds one integer bank-level
variable per tertiary MSC/MSR bank to the TS MIQP.  The solver minimises

    wᵀ G_w w + ∇f ᵀ w,        G_w[bank, bank] = ``g_w_tso_shunt``,

over bank increments ``w_bank ∈ {-1, 0, +1}`` (see
:mod:`optimisation.miqp_solver`; there is no factor ½).  Holding every other
actuator fixed, a one-level step on bank ``j`` lowers the objective iff

    |∇f_j| > g_w_tso_shunt,     ∇f_j = (∂y/∂level_j)ᵀ ∇_y f.

``g_w_tso_shunt`` is therefore an admission threshold in objective units per
level, not an accumulation rate: the earlier ``shunt_int_g_w`` calibration
(protocol D2b, p90 gradient mapped to a 60-minute accumulation time) does not
transfer to it.  This module reads the per-bank gradient recorded as
``Shunt­Record.zone_tso_shunt_g_h`` — which both the hybrid and the integrator
site store *per Mvar* of nameplate coordinate — and reports the distribution of
``|g_h| · q_step_mvar`` so the weight can be set to a declared quantile of it.

Assumptions and limits
----------------------
* The admission identity above neglects the joint solve: the MIQP chooses bank
  steps together with the continuous actuators and under the output
  constraints (several banks of a zone may move in one instant since
  2026-10-22).  The bank with the largest gradient is therefore not
  necessarily the one selected, and an admitted bank-instant is not
  necessarily a commit.
* A step being admitted says nothing about it being beneficial.  The gradient
  is dominated by whichever output term carries the largest weight.
* ``--engage-only`` keeps samples whose demanded direction is engagement.  It
  presumes every bank sits at level 0 and is held, which holds for a baseline
  run with no commit and is verified here.  Release pressure on a bank at
  level 0 cannot be acted on and would otherwise inflate the tail.
* The resulting number is specific to the objective weights that produced the
  record.  If the common objective scale changes, recalibrate; the reusable
  artefact is the quantile rule, not the value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# Unpickling a record file imports the experiment record classes.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUANTILES: Tuple[float, ...] = (0.50, 0.75, 0.90, 0.95, 0.99, 1.00)
CANDIDATE_WEIGHTS: Tuple[float, ...] = (5., 10., 15., 20., 25., 30., 40., 60.)


def bank_step_mvar(run_dir: Path) -> float:
    """Return the common bank step of a run, refusing unequal MSC/MSR steps.

    ``zone_tso_shunt_g_h`` carries no device label per entry, so a run whose
    capacitor and reactor steps differ cannot be converted unambiguously here.
    """
    config = json.loads((run_dir/"config.json").read_text(encoding="utf-8"))
    msc, msr = (float(config["tso_shunt_msc_q_step_mvar"]),
                float(config["tso_shunt_msr_q_step_mvar"]))
    if msc != msr:
        raise ValueError(
            f"{run_dir.name}: MSC step {msc} differs from MSR step {msr}; the "
            "recorded gradients carry no device label, so pass --q-step")
    return msc


def collect(run_dir: Path, q_step_mvar: float) -> Tuple[NDArray[np.float64], int, int]:
    """Return per-level bank gradients, the TS instant count and commit count.

    Rows are ``(gradient, demanded_direction)`` with the direction following the
    OFO sign convention: a negative gradient demands engagement.
    """
    records = pickle.loads((run_dir/"records.pkl").read_bytes())
    rows: List[Tuple[float, int]] = []
    instants = commits = 0
    for record in records:
        commits += len(record.shunt_commits)
        if not record.zone_tso_shunt_g_h:
            continue
        instants += 1
        for gradient in record.zone_tso_shunt_g_h.values():
            for value in np.asarray(gradient, dtype=float)*q_step_mvar:
                if not np.isfinite(value):
                    raise ValueError(f"{run_dir.name}: non-finite bank gradient")
                rows.append((float(value), -1 if value > 0 else 1))
    if not rows:
        raise ValueError(f"{run_dir.name}: no bank gradients recorded")
    return np.asarray(rows, dtype=float), instants, commits


def summarise(samples: NDArray[np.float64], *, engage_only: bool) -> Dict[str, object]:
    """Quantiles of |gradient| and the bank-instants each candidate admits."""
    magnitude = np.abs(samples[:, 0])
    if engage_only:
        magnitude = magnitude[samples[:, 1] > 0]
    if magnitude.size == 0:
        raise ValueError("No samples left after filtering")
    return {
        "n_samples": int(magnitude.size),
        "quantiles": {f"p{int(q*100)}": float(v) for q, v in
                      zip(QUANTILES, np.quantile(magnitude, QUANTILES))},
        "admitted_bank_instants": {f"{c:g}": int((magnitude > c).sum())
                                   for c in CANDIDATE_WEIGHTS},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", type=Path, required=True,
                        help="Run directories holding records.pkl and config.json")
    parser.add_argument("--quantile", type=float, default=0.99,
                        help="Quantile of the admissible-direction magnitudes "
                             "proposed as g_w_tso_shunt (default 0.99)")
    parser.add_argument("--q-step", type=float, default=None,
                        help="Override the bank step [Mvar] read from config.json")
    parser.add_argument("--all-directions", action="store_true",
                        help="Keep release pressure on banks that cannot release")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if not 0. < args.quantile <= 1.:
        parser.error("--quantile must lie in (0, 1]")

    report: Dict[str, object] = {"quantile": args.quantile, "runs": {}}
    pooled: List[NDArray[np.float64]] = []
    for run in args.runs:
        q_step = args.q_step if args.q_step is not None else bank_step_mvar(run)
        samples, instants, commits = collect(run, q_step)
        entry = {"q_step_mvar": q_step, "ts_instants": instants,
                 "shunt_commits": commits,
                 "all": summarise(samples, engage_only=False)}
        if commits == 0:
            entry["engage_only"] = summarise(samples, engage_only=True)
        else:
            # Levels move during the run, so a held-bank direction filter would
            # misstate which samples were actionable.
            entry["engage_only"] = "not applicable: banks switched in this run"
        report["runs"][str(run)] = entry
        pooled.append(samples)
        print(f"{run}: {instants} TS instants, {samples.shape[0]} bank-instants, "
              f"{commits} commits, q_step {q_step:g} Mvar")
        for scope in ("all", "engage_only"):
            value = entry[scope]
            if isinstance(value, str):
                print(f"  {scope:12s} {value}")
                continue
            print(f"  {scope:12s} n={value['n_samples']:4d}  " + "  ".join(
                f"{k}={v:8.3f}" for k, v in value["quantiles"].items()))

    merged = np.vstack(pooled)
    engage_only = not args.all_directions and all(
        isinstance(v, dict) and v["shunt_commits"] == 0
        for v in report["runs"].values())
    pooled_summary = summarise(merged, engage_only=engage_only)
    proposed = float(np.quantile(
        np.abs(merged[:, 0])[merged[:, 1] > 0] if engage_only
        else np.abs(merged[:, 0]), args.quantile))
    report.update(pooled=pooled_summary, engage_only=engage_only,
                  proposed_g_w_tso_shunt=proposed)
    print(f"\npooled ({'engagement-available' if engage_only else 'all'} "
          f"directions, n={pooled_summary['n_samples']}):")
    print("  " + "  ".join(f"{k}={v:8.3f}" for k, v in
                           pooled_summary["quantiles"].items()))
    print("  admitted bank-instants: " + "  ".join(
        f"g_w={k}: {v}" for k, v in pooled_summary["admitted_bank_instants"].items()))
    print(f"  proposed g_w_tso_shunt at p{args.quantile*100:g} = {proposed:.3f}")
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
