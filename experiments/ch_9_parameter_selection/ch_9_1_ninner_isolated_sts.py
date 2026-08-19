r"""Measure ``N_inner`` on the isolated subordinate loop -- thesis eq. (9.2).

``N_inner`` is the number of subordinate (STS-OFO) iterations a
capability-band-traversing interface-Q setpoint step needs before the tracking
error enters the settling band **and stays**. The supervisory layer is silent
throughout, so what is measured is the subordinate loop alone.

**Parent-silent variant: frozen OFO parent (option (ii) of the handoff).**
``tso_mode`` stays ``'ofo'`` and ``tso_period_s`` is set beyond the horizon, so
the supervisory OFO solves once at ``t = 0`` to establish the operating point
and never revises it (``run_tso = (step == 1) or _is_period_hit(...)``). The TS
actuators then hold while an exogenous setpoint drives the subordinate layer.

The alternative -- ``tso_mode='local'``, which the existing injection path is
gated on -- is available as ``--variant local`` for cross-check, but it is not
the default and should not be read as the primary result: it swaps in local
Q(V) as the supervisory controller, so the TS plant *moves in response to the
STS*, which confounds a measurement meant to isolate the subordinate loop. That
difference is the thing being measured, not an implementation detail.

**The step.** To the CAIR band edge -- the hardest admissible traversal -- in
both directions, and from both an importing and an exporting initial operating
point. The loop is allowed to settle at setpoint A before the step to B, so the
iteration count is a property of the step and not of the initial condition;
this is what ``q_pcc_setpoint_schedule_per_dso`` exists for.

**The circularity, stated rather than hidden.** ``G_w`` was calibrated with
``N_inner = 9`` *assumed*, and this script measures ``N_inner`` with those
weights in place. That is a fixed-point argument evaluated at one iteration. It
is reported as a **check on the guess** -- the guess is used, the weights
follow, and this tests whether the guess survives its own consequence -- and
never as an independent measurement. If it returns ``N_inner > 9`` the decision
is the author's: raise ``T_TS``, or recalibrate the weights and repeat.

**What QSS does and does not say.** "Converged" here means the QSS iteration
converged, not that the plant settled; the closed-loop RMS chapter tests the
latter.

Usage::

    python experiments\ch_9_parameter_selection\ch_9_1_ninner_isolated_sts.py --self-test
    python experiments\ch_9_parameter_selection\ch_9_1_ninner_isolated_sts.py --probe
    python experiments\ch_9_parameter_selection\ch_9_1_ninner_isolated_sts.py --workers 6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# =====================================================================
#  Constants entering a number
# =====================================================================

#: Subordinate period [s]. Fixed for the same reason as in the T_TS sweep: it
#: is what licenses the quasi-steady-state model, and ``N_inner`` is a count of
#: THESE iterations, so changing it changes the unit of the answer.
STS_PERIOD_S = 20.0

#: Seconds the loop is given to settle at setpoint A before the step to B.
#: 600 s = 30 subordinate iterations, comfortably beyond any plausible
#: ``N_inner``, so the step starts from a settled loop rather than mid-transient.
SETTLE_S = 600.0

#: Seconds observed after the step. The measurement window, and therefore the
#: right-censoring point: an interval still outside the band at the last sample
#: is reported censored at ``OBSERVE_S / STS_PERIOD_S`` iterations.
OBSERVE_S = 900.0

#: Fraction of the reported capability band traversed by the step. 1.0 is the
#: band edge -- what the chapter promises and the hardest admissible traversal.
#: Backed off slightly by default because the band is reported from the
#: operating point *before* the step and a setpoint exactly on the edge is
#: rejected as infeasible by the subordinate MIQP as often as it is tracked,
#: which measures the constraint rather than the loop.
DEFAULT_BAND_FRACTION = 0.95

#: Settling band on the interface flow [Mvar]; see the sweep module for why
#: this is a decision rather than a detail.
DEFAULT_BAND_MVAR = 1.0

V_MIN_PU, V_MAX_PU = 0.9, 1.1

#: Design-bank strata in which the DER reactive capability is structurally
#: zero, so no capability-band traversal exists to measure.
#:
#: ``WINDOW_META[...]["stratum"]`` is exactly the DER reactive-capability tier,
#: as the T_TS sweep's own measured band widths confirm: ``full`` ~190 Mvar
#: (6 windows), ``partial`` ~55-85 Mvar (4), and ``none`` ~0.1-0.7 Mvar
#: (``d_quiet_summer``, ``d_ramp_up_winter``). The last is the VDE dead zone
#: that ``tuning_mc/stage_1_search.py`` already names as the reason ``tau``,
#: ``lambda_dso`` and ``dso_g_v_ratio`` are structurally inert there.
#:
#: These windows are excluded by default, and the exclusion is REPORTED: a
#: window with no admissible traversal is not a failed measurement, and
#: counting it as one would put a spurious censoring fraction into eq. (9.2).
DEAD_ZONE_STRATA = ("none",)


# =====================================================================
#  Provenance
# =====================================================================

def _git(*args: str) -> Optional[str]:
    try:
        proc = subprocess.run(("git", *args), cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def provenance(args: argparse.Namespace, design: Dict[str, Any]) -> Dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "script": "experiments/ch_9_parameter_selection/ch_9_1_ninner_isolated_sts.py",
        "argv": sys.argv,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in sorted(vars(args).items())},
        "timestamp": datetime.now().astimezone().isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "design": design,
        "constants": {
            "sts_period_s": STS_PERIOD_S, "settle_s": SETTLE_S,
            "observe_s": OBSERVE_S, "band_fraction": args.band_fraction,
            "band_mvar": args.band_mvar, "variant": args.variant,
        },
        "circularity": (
            "G_w was calibrated with N_inner = 9 ASSUMED; this measures "
            "N_inner with those weights in place. It is a check on the guess, "
            "not an independent measurement."),
    }


# =====================================================================
#  Measurement
# =====================================================================

def measure_n_inner(records: Sequence[Any], *, step_time_s: float,
                    sts_period_s: float, band_mvar: float,
                    ) -> List[Dict[str, Any]]:
    """``N_inner`` per interface transformer, counted from the step instant.

    The count is the first subordinate iteration at or after ``step_time_s`` at
    which ``|q_actual - q_set|`` enters the band and remains inside it for
    every later sample. "And stays" is the whole content of the definition: a
    trajectory that touches the band and leaves it again has not settled, and
    counting first entry would report the loop as faster than it is.

    Right-censored at the end of the observation window; censoring is returned
    as a flag, never as a dropped row.
    """
    out: List[Dict[str, Any]] = []
    trafos = sorted({t for r in records
                     for t in getattr(r, "dso_trafo_q_set_mvar", {}) or {}})
    groups: Dict[str, str] = {}
    for r in records:
        groups.update(getattr(r, "dso_trafo_group", {}) or {})

    post = [r for r in records
            if float(getattr(r, "time_s", -1.0)) >= step_time_s]
    if not post:
        return out
    cap = len(post)

    for t in trafos:
        err: List[float] = []
        for r in post:
            a = getattr(r, "dso_trafo_q_actual_mvar", {}).get(t)
            sv = getattr(r, "dso_trafo_q_set_mvar", {}).get(t)
            err.append(abs(float(a) - float(sv))
                       if a is not None and sv is not None else float("nan"))
        if not any(math.isfinite(x) for x in err):
            continue

        n_inner: Optional[int] = None
        for i in range(len(err)):
            tail = err[i:]
            if tail and all(math.isfinite(x) and x <= band_mvar for x in tail):
                n_inner = i
                break
        censored = n_inner is None

        sets = [getattr(r, "dso_trafo_q_set_mvar", {}).get(t) for r in post]
        sets = [float(v) for v in sets if v is not None]
        acts = [getattr(r, "dso_trafo_q_actual_mvar", {}).get(t) for r in post]
        acts = [float(v) for v in acts if v is not None]
        taps = [getattr(r, "dso_trafo_tap_pos", {}).get(t) for r in post]
        taps = [int(v) for v in taps if v is not None]

        grp = groups.get(t, "?")
        vmin = [getattr(r, "dso_group_v_min_pu", {}).get(grp) for r in post]
        vmax = [getattr(r, "dso_group_v_max_pu", {}).get(grp) for r in post]
        vmin = [float(v) for v in vmin
                if v is not None and math.isfinite(float(v))]
        vmax = [float(v) for v in vmax
                if v is not None and math.isfinite(float(v))]
        slack = [getattr(r, "dso_z_slack_max", {}).get(grp) for r in post]
        slack = [float(v) for v in slack
                 if v is not None and math.isfinite(float(v))]

        out.append({
            "trafo": t, "group": grp,
            "n_inner": cap if censored else n_inner,
            "censored": censored,
            "n_inner_cap": cap,
            "q_set_after_mvar": sets[-1] if sets else float("nan"),
            "q_actual_after_mvar": acts[-1] if acts else float("nan"),
            "residual_mvar": err[-1] if err else float("nan"),
            "tap_moves": sum(1 for a, b in zip(taps, taps[1:]) if a != b),
            "v_min_pu": min(vmin) if vmin else float("nan"),
            "v_max_pu": max(vmax) if vmax else float("nan"),
            "v_violation": bool((vmin and min(vmin) < V_MIN_PU)
                                or (vmax and max(vmax) > V_MAX_PU)),
            "z_slack_max": max(slack) if slack else float("nan"),
        })
    return out


def band_edge_targets(records: Sequence[Any], *, at_time_s: float,
                      fraction: float, min_width_mvar: float = 1.0,
                      ) -> Dict[str, Dict[str, float]]:
    """Per-transformer ``{q_now, cap_min, cap_max, target_up, target_down}``.

    Read from the capability the subordinate layer *reports*, which is what the
    supervisory layer would have to work with. Targets are placed a
    ``fraction`` of the way from the current flow to each band edge.

    **The reported band collapses to zero width at a non-trivial fraction of
    instants** -- ``cap_min == cap_max == q_now``, i.e. "nothing is available
    from here" -- at every interface, and with a live supervisory layer as well
    as a frozen one (measured 2026-08-19; the T_TS sweep's own per-interval
    widths have median 67-181 Mvar but minimum 0.00 for all twelve
    transformers). Reading a single instant therefore lands on a degenerate
    band often enough to matter, and a degenerate band silently produces
    ``target == q_now``: a step of zero, which then "settles" in zero
    iterations and reports ``N_inner = 0`` for a step that never happened.
    That is exactly the shape of failure this measurement must not have, and
    it is what the first probe returned.

    So the band is taken from the LAST sample at or before ``at_time_s`` whose
    reported width is at least ``min_width_mvar``, and a transformer with no
    such sample is omitted -- which the caller turns into an explicit skipped
    case rather than a zero-step measurement.
    """
    out: Dict[str, Dict[str, float]] = {}
    usable = [r for r in records
              if float(getattr(r, "time_s", -1.0)) <= at_time_s]
    if not usable:
        return out
    trafos = {t for r in usable
              for t in (getattr(r, "dso_trafo_q_actual_mvar", {}) or {})}
    for t in sorted(trafos):
        snap = None
        for r in reversed(usable):
            lo = (getattr(r, "dso_trafo_q_cap_min_mvar", {}) or {}).get(t)
            hi = (getattr(r, "dso_trafo_q_cap_max_mvar", {}) or {}).get(t)
            if lo is None or hi is None:
                continue
            if float(hi) - float(lo) >= min_width_mvar:
                snap = r
                break
        if snap is None:
            continue
        # Band from the last non-degenerate report; operating point from the
        # step instant, which is where the step actually starts.
        last = usable[-1]
        q_now = (getattr(last, "dso_trafo_q_actual_mvar", {}) or {}).get(t)
        if q_now is None:
            continue
        lo = float((getattr(snap, "dso_trafo_q_cap_min_mvar", {}) or {})[t])
        hi = float((getattr(snap, "dso_trafo_q_cap_max_mvar", {}) or {})[t])
        q_now = float(q_now)
        # A band read earlier may no longer bracket the current flow.
        lo, hi = min(lo, q_now), max(hi, q_now)
        out[t] = {
            "q_now": q_now, "cap_min": lo, "cap_max": hi,
            "width": hi - lo,
            "band_read_at_s": float(getattr(snap, "time_s", float("nan"))),
            "target_up": q_now + fraction * (hi - q_now),
            "target_down": q_now - fraction * (q_now - lo),
        }
    return out


# =====================================================================
#  Worker
# =====================================================================

def _retime_parent(cfg, tso_period_s: float, sbx_cycle: str = "iterations"):
    """Set ``tso_period_s`` on the config AND on the nested SBX-H config.

    The handoff said the sweep was ``dataclasses.replace(spec, tso_period_s=X)``
    "and nothing else". It is not: the selected design runs
    ``coordination_mode = 'sbx_h'``, and ``SBXConfig`` carries its own
    ``tso_period_s`` which the runner requires to match
    (``multi_tso_dso.py:1280``). Leaving it stale raises, which is the right
    behaviour -- the guard exists because ``k_sched`` is a cycle length counted
    in *TSO iterations*, so the two periods are not independent.

    **That coupling is a confound, and it is reported rather than hidden.**
    With ``k_sched = 2`` held fixed, sweeping ``T_TS`` also scales the SBX-H
    settlement cycle in wall-clock time: 2 min at ``T_TS = 60 s``, 10 min at
    300 s. Two readings are available:

    * ``iterations`` (default) -- hold ``k_sched``, i.e. hold the *controller's
      own configuration* fixed and let its wall-clock consequence follow the
      period. This is what changing ``T_TS`` in service would actually do.
    * ``wallclock`` -- hold the settlement cycle near its 360 s default by
      re-deriving ``k_sched = max(1, round(360 / T_TS))``. Cleaner on the SBX
      axis, but it re-tunes a second mechanism mid-sweep and cannot land
      exactly (240 s admits no integer giving 360 s).

    Neither is neutral. The default holds the controller fixed because that is
    the object the chapter is selecting a period for.
    """
    import dataclasses

    cfg = dataclasses.replace(cfg, tso_period_s=float(tso_period_s))
    sbx = getattr(cfg, "sbx_config", None)
    if sbx is not None and hasattr(sbx, "tso_period_s"):
        fields = {"tso_period_s": float(tso_period_s)}
        if sbx_cycle == "wallclock" and hasattr(sbx, "k_sched"):
            default_cycle_s = float(sbx.k_sched) * float(sbx.tso_period_s)
            fields["k_sched"] = max(1, round(default_cycle_s / float(tso_period_s)))
        cfg = dataclasses.replace(cfg, sbx_config=dataclasses.replace(sbx, **fields))
    sbxv = getattr(cfg, "sbxv_config", None)
    if sbxv is not None and hasattr(sbxv, "tso_period_s"):
        cfg = dataclasses.replace(
            cfg, sbxv_config=dataclasses.replace(
                sbxv, tso_period_s=float(tso_period_s)))
    return cfg


def _run_case(job: Dict[str, Any]) -> Dict[str, Any]:
    """One (window, DSO, direction) isolated-STS step.

    Two passes. The first runs the settle phase only, to read the capability
    band the subordinate layer reports *at the step instant* -- the band moves
    with the operating point, so a target computed from a t=0 snapshot would
    not be the band edge by the time the step lands. The second runs the full
    horizon with the schedule built from that reading.
    """
    import dataclasses
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "ch_9_parameter_selection"))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    from _ch9_selected_design import build_selected_config
    from tuning.metrics import MetricScales
    from tuning.objectives_v2 import _run_scenario
    from tuning_mc.scenarios_mc_v2 import tier1_design_set

    cfg, _prov = build_selected_config()
    spec = next(s for s in tier1_design_set() if s.name == job["window"])

    total_s = SETTLE_S + OBSERVE_S
    # Parent silent. Frozen OFO by default: one solve at t=0, then hold.
    if job["variant"] == "frozen_ofo":
        parent = {"tso_mode": "ofo", "tso_period_s": total_s * 10.0,
                  "q_pcc_injection_with_ofo_parent": True}
    else:
        parent = {"tso_mode": "local", "tso_period_s": total_s * 10.0,
                  "q_pcc_injection_with_ofo_parent": True}

    # tso_period_s also lives on the nested SBX-H config; see _retime_parent.
    base = _retime_parent(dataclasses.replace(cfg, **parent), total_s * 10.0)
    spec = dataclasses.replace(spec, duration_s=total_s,
                               dso_period_s=STS_PERIOD_S, dt_s=STS_PERIOD_S,
                               tso_period_s=total_s * 10.0,
                               contingencies=())

    # --- pass 1: settle only, to read the reported band at the step instant.
    probe_spec = dataclasses.replace(spec, duration_s=SETTLE_S + STS_PERIOD_S)
    res_p, rec_p = _run_scenario(probe_spec, base, MetricScales())
    if res_p.failure_reason:
        return {**job, "failure": res_p.failure_reason, "rows": [],
                "targets": {}, "wall_s": res_p.wall_time_s}

    targets = band_edge_targets(rec_p, at_time_s=SETTLE_S,
                                fraction=float(job["band_fraction"]))
    groups: Dict[str, str] = {}
    for r in rec_p:
        groups.update(getattr(r, "dso_trafo_group", {}) or {})
    mine = [t for t, g in groups.items() if g == job["dso"]]
    if not mine or not all(t in targets for t in mine):
        return {**job, "failure": f"no reported capability band for {job['dso']}",
                "rows": [], "targets": {}, "wall_s": res_p.wall_time_s}

    key = "target_up" if job["direction"] == "up" else "target_down"
    # Order must match HVNetworkInfo.coupling_trafo_indices, which is the order
    # the transformer keys sort in.
    mine_sorted = sorted(mine)
    # A step smaller than the settling band is not a measurement: the loop is
    # already inside the band before it starts, and N_inner would come back 0
    # for a step that never happened.
    degenerate = [t for t in mine_sorted
                  if abs(targets[t][key] - targets[t]["q_now"])
                  < float(job["band_mvar"])]
    if degenerate:
        return {**job, "rows": [], "targets": targets,
                "wall_s": res_p.wall_time_s,
                "failure": (f"reported capability gives a step below the "
                            f"settling band for {degenerate}: no admissible "
                            f"traversal at this operating point")}
    q_hold = [targets[t]["q_now"] for t in mine_sorted]
    q_step = [targets[t][key] for t in mine_sorted]

    cfg_run = dataclasses.replace(base, q_pcc_setpoint_schedule_per_dso={
        job["dso"]: [{"t_s": 0.0, "q_mvar": q_hold},
                     {"t_s": SETTLE_S, "q_mvar": q_step}]})

    res, records = _run_scenario(spec, cfg_run, MetricScales())
    if res.failure_reason:
        return {**job, "failure": res.failure_reason, "rows": [],
                "targets": targets, "wall_s": res.wall_time_s}

    rows = measure_n_inner(records, step_time_s=SETTLE_S,
                           sts_period_s=STS_PERIOD_S,
                           band_mvar=float(job["band_mvar"]))
    rows = [r for r in rows if r["group"] == job["dso"]]
    for r in rows:
        r.update({"window": job["window"], "dso": job["dso"],
                  "direction": job["direction"], "variant": job["variant"],
                  "step_mvar": (targets[r["trafo"]][key]
                                - targets[r["trafo"]]["q_now"]),
                  "band_width_mvar": targets[r["trafo"]]["width"]})
    return {**job, "failure": "", "rows": rows, "targets": targets,
            "wall_s": res.wall_time_s + res_p.wall_time_s}


# =====================================================================
#  Aggregation and outputs
# =====================================================================

def _quantile(xs: Sequence[float], q: float) -> float:
    v = sorted(x for x in xs if math.isfinite(x))
    if not v:
        return float("nan")
    if len(v) == 1:
        return v[0]
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``N_inner`` distributions by (DSO, direction) and pooled."""
    out: List[Dict[str, Any]] = []
    keys = sorted({(r["dso"], r["direction"]) for r in rows})
    for dso, direction in keys + [("__pooled__", "both")]:
        sub = (list(rows) if dso == "__pooled__"
               else [r for r in rows
                     if r["dso"] == dso and r["direction"] == direction])
        if not sub:
            continue
        n = [float(r["n_inner"]) for r in sub]
        out.append({
            "dso": dso, "direction": direction, "n_steps": len(sub),
            "n_inner_median": _quantile(n, 0.50),
            "n_inner_p95": _quantile(n, 0.95),
            "n_inner_max": max(n) if n else float("nan"),
            "censoring_fraction": (sum(1 for r in sub if r["censored"])
                                   / len(sub)),
            "step_median_mvar": _quantile(
                [abs(r["step_mvar"]) for r in sub], 0.50),
            "band_width_median_mvar": _quantile(
                [r["band_width_mvar"] for r in sub], 0.50),
            "residual_median_mvar": _quantile(
                [r["residual_mvar"] for r in sub], 0.50),
            "tap_moves_total": sum(r["tap_moves"] for r in sub),
            "v_violations": sum(1 for r in sub if r["v_violation"]),
        })
    return out


_ROW_COLS = ["window", "dso", "direction", "variant", "trafo", "group",
             "n_inner", "censored", "n_inner_cap", "step_mvar",
             "band_width_mvar", "q_set_after_mvar", "q_actual_after_mvar",
             "residual_mvar", "tap_moves", "v_min_pu", "v_max_pu",
             "v_violation", "z_slack_max"]


def write_outputs(out_dir: Path, rows: List[Dict[str, Any]],
                  agg: List[Dict[str, Any]], failures: List[Dict[str, Any]],
                  meta: Optional[Dict[str, Any]] = None) -> None:
    with (out_dir / "steps.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_ROW_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if agg:
        with (out_dir / "n_inner_summary.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
            w.writeheader()
            for r in agg:
                w.writerow(r)

    md = ["# N_inner on the isolated subordinate loop (thesis eq. 9.2)", "",
          "`N_inner` is the first subordinate iteration after a "
          "capability-band-traversing interface-Q step at which the tracking "
          "error enters the settling band **and stays**. Right-censored at the "
          "end of the observation window; censoring is reported, never dropped.",
          "",
          "## This is a check on the guess, not an independent measurement",
          "",
          "`G_w` was calibrated with `N_inner = 9` **assumed**, and this "
          "measures `N_inner` with those weights in place. That is a "
          "fixed-point argument evaluated at one iteration: the guess is used, "
          "the weights follow, and this tests whether the guess survives its "
          "own consequence. Quote it that way. If the answer exceeds 9 the "
          "choice -- raise `T_TS`, or recalibrate the weights and repeat -- is "
          "an author decision, not a script output.", "",
          "## Result", "",
          "| DSO | direction | steps | N_inner med | p95 | max | censored | "
          "step med [Mvar] | band width med [Mvar] | residual med [Mvar] | "
          "taps | V viol |",
          "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for a in agg:
        md.append(
            f"| {a['dso']} | {a['direction']} | {a['n_steps']} | "
            f"{a['n_inner_median']:.1f} | {a['n_inner_p95']:.1f} | "
            f"{a['n_inner_max']:.0f} | {a['censoring_fraction']:.2f} | "
            f"{a['step_median_mvar']:.1f} | "
            f"{a['band_width_median_mvar']:.1f} | "
            f"{a['residual_median_mvar']:.2f} | {a['tap_moves_total']} | "
            f"{a['v_violations']} |")

    if meta and meta.get("excluded_windows"):
        md += ["", "## Windows excluded (no admissible traversal)", "",
               "The DER reactive capability is structurally zero in these "
               "windows (VDE dead zone), so there is no capability band to "
               "traverse and no `N_inner` to measure. They are excluded, not "
               "failed: counting them would put a spurious censoring fraction "
               "into eq. (9.2).", ""]
        md += [f"- `{w}`" for w in meta["excluded_windows"]]

    if failures:
        md += ["", "## Failed cases", "",
               "**The result is not complete.**", ""]
        md += [f"- `{f['window']}` / `{f['dso']}` / {f['direction']}: "
               f"{(f.get('failure') or '').splitlines()[0]}" for f in failures]

    if meta:
        d = meta.get("design", {})
        c = meta.get("constants", {})
        md += ["", "## Provenance", "",
               f"- run: `{meta['timestamp']}`",
               f"- commit: `{meta['git_commit']}` on `{meta['git_branch']}`"
               + ("  **(working tree dirty -- not reproducible from the "
                  "commit alone)**" if meta.get("git_dirty") else ""),
               f"- parent-silent variant: **{c.get('variant')}** "
               f"(`frozen_ofo` = supervisory OFO solves once at t=0 and holds; "
               f"`local` = local Q(V) supervisory control, a DIFFERENT "
               f"baseline controller and a cross-check only)",
               f"- weights: campaign `{d.get('campaign')}` candidate "
               f"`{d.get('candidate_key')}`, {d.get('verified')}",
               f"- settle {c.get('settle_s')} s, observe {c.get('observe_s')} s, "
               f"band {c.get('band_mvar')} Mvar, step "
               f"{c.get('band_fraction')} of the reported band",
               f"- `T_STS` = {STS_PERIOD_S:.0f} s, bank `tier1_design_set` on "
               f"`rural_700`",
               f"- command: `{' '.join(meta['argv'])}`"]

    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


# =====================================================================
#  Offline self-test
# =====================================================================

class _R:
    def __init__(self, t, q_set, q_act, tap=0, cap=(-100.0, 100.0)):
        self.time_s = t
        self.tso_active = False
        self.dso_active = True
        self.dso_trafo_q_set_mvar = {"T1": q_set}
        self.dso_trafo_q_actual_mvar = {"T1": q_act}
        self.dso_trafo_tap_pos = {"T1": tap}
        self.dso_trafo_group = {"T1": "DSO_1"}
        self.dso_trafo_q_cap_min_mvar = {"T1": cap[0]}
        self.dso_trafo_q_cap_max_mvar = {"T1": cap[1]}
        self.dso_z_slack_max = {"DSO_1": 0.0}
        self.dso_group_v_min_pu = {"DSO_1": 0.99}
        self.dso_group_v_max_pu = {"DSO_1": 1.01}


def self_test() -> int:
    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
              + (f" -- {detail}" if detail and not cond else ""))
        ok = ok and cond

    print("[self-test] isolated-STS N_inner measurement")

    # Settle at 0, step to 50 at t = 100, error decays over 4 iterations.
    recs = ([_R(float(20 * i), 0.0, 0.0) for i in range(5)]
            + [_R(100.0, 50.0, 20.0), _R(120.0, 50.0, 40.0),
               _R(140.0, 50.0, 48.0), _R(160.0, 50.0, 49.5),
               _R(180.0, 50.0, 49.8), _R(200.0, 50.0, 50.0)])
    rows = measure_n_inner(recs, step_time_s=100.0, sts_period_s=20.0,
                           band_mvar=1.0)
    check("one row per interface transformer", len(rows) == 1)
    check("N_inner counted from the step instant (3 iterations)",
          rows[0]["n_inner"] == 3 and not rows[0]["censored"],
          f"got {rows[0]['n_inner']}")

    # Never settles -> censored at the window length, not dropped.
    recs_c = ([_R(float(20 * i), 0.0, 0.0) for i in range(5)]
              + [_R(float(100 + 20 * i), 50.0, 10.0) for i in range(6)])
    rc = measure_n_inner(recs_c, step_time_s=100.0, sts_period_s=20.0,
                         band_mvar=1.0)
    check("a step that never settles is censored, not dropped",
          rc[0]["censored"] and rc[0]["n_inner"] == rc[0]["n_inner_cap"],
          f"got {rc[0]}")

    # Touch-and-leave must not count as settled.
    recs_b = ([_R(0.0, 0.0, 0.0)]
              + [_R(100.0, 50.0, 49.5), _R(120.0, 50.0, 45.0),
                 _R(140.0, 50.0, 45.0)])
    rb = measure_n_inner(recs_b, step_time_s=100.0, sts_period_s=20.0,
                         band_mvar=1.0)
    check("entering the band then leaving it does not count as settled",
          rb[0]["censored"], f"got n_inner={rb[0]['n_inner']}")

    # Band-edge targets, read at the step instant.
    tg = band_edge_targets(recs, at_time_s=100.0, fraction=1.0)
    check("band edge target up == reported cap_max",
          abs(tg["T1"]["target_up"] - 100.0) < 1e-9, f"got {tg['T1']}")
    check("band edge target down == reported cap_min",
          abs(tg["T1"]["target_down"] + 100.0) < 1e-9, f"got {tg['T1']}")
    tg95 = band_edge_targets(recs, at_time_s=100.0, fraction=0.95)
    check("a fraction below 1 backs off from the edge",
          tg95["T1"]["target_up"] < tg["T1"]["target_up"])

    # A degenerate reported band must be REFUSED, not turned into a zero step
    # that "settles" in 0 iterations. This is what the first probe returned.
    degen = [_R(0.0, 0.0, 5.0, cap=(5.0, 5.0)),
             _R(20.0, 0.0, 5.0, cap=(5.0, 5.0))]
    check("a zero-width reported band yields no target at all",
          band_edge_targets(degen, at_time_s=20.0, fraction=0.95) == {})
    mixed = [_R(0.0, 0.0, 5.0, cap=(-100.0, 100.0)),
             _R(20.0, 0.0, 5.0, cap=(5.0, 5.0))]
    mt = band_edge_targets(mixed, at_time_s=20.0, fraction=1.0)
    check("the last NON-degenerate report is used when the latest is degenerate",
          "T1" in mt and mt["T1"]["width"] >= 100.0, f"got {mt}")

    # RunResult accessors, checked offline (see the sweep module).
    try:
        import dataclasses as _dc

        from tuning.runner import RunResult
        fields = {f.name for f in _dc.fields(RunResult)}
        check("the worker reads RunResult fields that exist",
              {"failure_reason", "wall_time_s"} <= fields,
              f"RunResult has {sorted(fields)}")
    except ImportError as exc:
        check("RunResult importable for the field check", False, str(exc))

    # The config fields the worker sets must exist on MultiTSOConfig.
    try:
        import dataclasses as _dc

        from configs.config import MultiTSOConfig
        cfields = {f.name for f in _dc.fields(MultiTSOConfig)}
        check("the parent-silent config fields exist",
              {"q_pcc_injection_with_ofo_parent",
               "q_pcc_setpoint_schedule_per_dso", "tso_mode",
               "tso_period_s"} <= cfields)
    except ImportError as exc:
        check("MultiTSOConfig importable", False, str(exc))

    for r in rows:
        r.update({"window": "w", "dso": "DSO_1", "direction": "up",
                  "variant": "frozen_ofo", "step_mvar": 50.0,
                  "band_width_mvar": 200.0})
    agg = aggregate(rows)
    check("aggregation reports a pooled row", any(a["dso"] == "__pooled__"
                                                  for a in agg))

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        write_outputs(out, rows, agg, failures=[], meta=None)
        for name in ("steps.csv", "n_inner_summary.csv", "summary.md"):
            check(f"{name} written", (out / name).exists())
        body = (out / "summary.md").read_text(encoding="utf-8")
        check("summary states the circularity",
              "check on the guess" in body)

    print(f"[self-test] {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


# =====================================================================
#  Entry point
# =====================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Isolated-STS measurement of N_inner (thesis eq. 9.2).")
    ap.add_argument("--label", default="ninner_isolated")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "ch9_ninner")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--band-mvar", type=float, default=DEFAULT_BAND_MVAR)
    ap.add_argument("--band-fraction", type=float,
                    default=DEFAULT_BAND_FRACTION,
                    help="fraction of the reported capability band traversed")
    ap.add_argument("--variant", choices=("frozen_ofo", "local"),
                    default="frozen_ofo",
                    help="parent-silent variant; 'local' swaps in a DIFFERENT "
                         "supervisory controller and is a cross-check only")
    ap.add_argument("--windows", default="")
    ap.add_argument("--dsos", default="",
                    help="comma-separated DSO ids; default every DSO present")
    ap.add_argument("--include-dead-zone", action="store_true",
                    help="also run windows whose stratum has structurally "
                         "zero DER reactive capability; they have no "
                         "admissible band traversal and will be refused")
    ap.add_argument("--probe", action="store_true",
                    help="one window, one DSO, both directions -- to confirm "
                         "the band is traversable before the full grid")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test()

    from _ch9_selected_design import build_selected_config
    from tuning_mc.scenarios_mc_v2 import WINDOW_META, tier1_design_set

    cfg, design = build_selected_config()
    print(f"[design] campaign {design['campaign']} candidate "
          f"{design['candidate_key']}: {design['verified']}")

    specs = tier1_design_set()
    names = [s.name for s in specs]
    if a.windows:
        want = [w.strip() for w in a.windows.split(",") if w.strip()]
        unknown = [w for w in want if w not in names]
        if unknown:
            print(f"[abort] unknown window(s): {unknown}")
            return 1
        names = want

    # Windows with no DER reactive capability carry no band to traverse.
    excluded = [w for w in names
                if WINDOW_META.get(w, {}).get("stratum") in DEAD_ZONE_STRATA]
    if excluded and not a.include_dead_zone:
        names = [w for w in names if w not in excluded]
        print(f"[ninner] excluded {len(excluded)} window(s) whose DER reactive "
              f"capability is structurally zero (VDE dead zone), so no "
              f"capability-band traversal exists: {excluded}")
        print("[ninner]   -- this is an exclusion, NOT a failed measurement; "
              "pass --include-dead-zone to run them anyway")
    if not names:
        print("[abort] every requested window is in the dead zone; nothing "
              "to measure")
        return 1

    dsos = ([d.strip() for d in a.dsos.split(",") if d.strip()]
            if a.dsos else [f"DSO_{i}" for i in range(1, 5)])
    if a.probe:
        names, dsos = names[:1], dsos[:1]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = a.out / a.label / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (a.out / a.label / "_latest.txt").write_text(stamp, encoding="utf-8")

    meta = provenance(a, design)
    meta["excluded_windows"] = excluded if not a.include_dead_zone else []
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")

    jobs = [{"window": w, "dso": d, "direction": dirn, "variant": a.variant,
             "band_mvar": a.band_mvar, "band_fraction": a.band_fraction}
            for w in names for d in dsos for dirn in ("up", "down")]
    with (out_dir / "cases.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["window", "dso", "direction", "variant"])
        for j in jobs:
            wr.writerow([j["window"], j["dso"], j["direction"], j["variant"]])

    print(f"[ninner] -> {out_dir}")
    print(f"[ninner] {len(jobs)} steps = {len(names)} windows x {len(dsos)} "
          f"DSOs x 2 directions, {a.workers} workers, variant {a.variant}")
    print(f"[ninner] commit {meta['git_commit']} on {meta['git_branch']}"
          + ("  [WORKING TREE DIRTY]" if meta.get("git_dirty") else ""))

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_run_case, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            done += 1
            try:
                res = fut.result()
            except Exception as exc:
                failures.append({**j, "failure": f"{type(exc).__name__}: {exc}"})
                print(f"  [{done}/{len(jobs)}] {j['window']}/{j['dso']}/"
                      f"{j['direction']}  WORKER DIED: {exc}")
                continue
            if res["failure"]:
                failures.append(res)
                print(f"  [{done}/{len(jobs)}] {res['window']}/{res['dso']}/"
                      f"{res['direction']}  FAILED: "
                      f"{res['failure'].splitlines()[0]}")
                continue
            rows.extend(res["rows"])
            ns = [r["n_inner"] for r in res["rows"]]
            print(f"  [{done}/{len(jobs)}] {res['window']}/{res['dso']}/"
                  f"{res['direction']}  N_inner={ns}, {res['wall_s']:.0f} s")

    if not rows:
        print("[abort] no step produced a measurement; nothing written")
        return 1

    agg = aggregate(rows)
    write_outputs(out_dir, rows, agg, failures, meta)

    pooled = next((x for x in agg if x["dso"] == "__pooled__"), None)
    if pooled:
        print(f"\n[ninner] pooled: median {pooled['n_inner_median']:.1f}, "
              f"p95 {pooled['n_inner_p95']:.1f}, "
              f"max {pooled['n_inner_max']:.0f}, "
              f"censored {pooled['censoring_fraction']:.2f} "
              f"over {pooled['n_steps']} steps")
        print("[ninner] this is a CHECK ON THE GUESS N_inner = 9, not an "
              "independent measurement: G_w was calibrated assuming it.")

    rc = 0
    if failures:
        print(f"\n[ninner] {len(failures)} case(s) FAILED -- result incomplete")
        rc = 2
    if pooled and pooled["censoring_fraction"] >= 0.5:
        print(f"\n[ninner] >= 50 % of steps censored: the band may be "
              f"unreachable at this setting; see summary.md")
        rc = 2
    print(f"\n[ninner] wrote {out_dir}  (exit {rc})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
