r"""Sweep the supervisory period ``T_TS`` -- the selection evidence for Sec. 9.1.

**What this answers.** Eq. (9.2) fixes ``T_TS = N_inner * T_STS`` with
``N_inner = 9`` an educated guess that has never been evaluated. This script
does not measure ``N_inner`` (that is ``ch_9_1_ninner_isolated_sts.py``, the
isolated-STS experiment). It answers the *other* question: across
``T_TS in {60, 120, 180, 240, 300} s`` at fixed ``T_STS = 20 s``, how much of
the correction the supervisory layer asks for does the subordinate layer
actually deliver before the next dispatch -- and is 180 s defensible.

**Why QSS is admissible here.** ``T_STS = 20 s`` exceeds the binding open-loop
settling of the Task-A battery, so the plant is settled at every STS sample by
construction and the quasi-steady-state model is not merely convenient. That
argument fails if ``T_STS`` is also swept downward, so ``T_STS`` is fixed:
``dso_period_s = dt_s = 20 s`` throughout, and only ``tso_period_s`` moves.

**What "converged" means here.** The QSS iteration converged -- not that the
plant settled. The intra-interval electromechanical transient is invisible to
this model by construction; the closed-loop RMS chapter is where that is tested.

**The metric, and the trap it avoids.** Results are reported per ``T_TS`` as a
*distribution over dispatch intervals*, never as one time-aggregated scalar. At
a fixed horizon ``T_TS = 60 s`` yields five times as many dispatches as 300 s,
so any RMS residual accumulated over time mixes "residual per dispatch" with
"dispatch frequency" and would show 60 s in a flattering light for a reason
that has nothing to do with control.

Weights are the Sec. 9.3 selection, rebuilt and asserted by
``_ch9_selected_design`` -- ``N_inner`` is a joint property of the period ratio
and of ``G_w``, so the file defaults of ``run_multi_system_ofo.py`` would
answer for a different controller.

Usage::

    python experiments\ch_9_parameter_selection\ch_9_1_ts_period_sweep.py --self-test
    python experiments\ch_9_parameter_selection\ch_9_1_ts_period_sweep.py --pilot --workers 6
    python experiments\ch_9_parameter_selection\ch_9_1_ts_period_sweep.py --workers 6
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

#: Supervisory periods swept [s]. Every one is an integer multiple of
#: ``T_STS = 20 s``, so the GCD requirement on ``dt_s`` (configs/config.py:188)
#: holds at every point and the runner's ``time_s % period_s < 1`` test fires
#: cleanly.
TS_PERIODS_S: Tuple[float, ...] = (60.0, 120.0, 180.0, 240.0, 300.0)

#: The pilot subset: the two ends and the incumbent. Its purpose is not
#: results -- it is to read the actual CAIR band widths and check the settling
#: band is reachable at all before the full grid is spent on it.
PILOT_PERIODS_S: Tuple[float, ...] = (60.0, 180.0, 300.0)
PILOT_N_WINDOWS = 3

#: Subordinate period, FIXED. See the module docstring: sweeping this would
#: remove the argument that licenses the quasi-steady-state model.
STS_PERIOD_S = 20.0

#: Default settling band on the interface flow [Mvar]. 1 Mvar absolute is what
#: the open-loop RMS battery used, so reusing it keeps Sec. 9.1 internally
#: consistent. **It is a CLI flag because it is a real decision**: at a PCC
#: with a ~200 Mvar CAIR width this is 0.5 %, which a MIQP with discrete taps
#: may never reach -- and then every interval returns censored and the sweep
#: says nothing. The pilot measures the actual band widths so the choice can be
#: made on evidence.
DEFAULT_BAND_MVAR = 1.0

#: Intervals in which the supervisory layer barely moved are dropped from the
#: ratio: ``rho_k = r_k / delta_k`` explodes as ``delta_k -> 0`` and would
#: report a control failure where there was no command.
DEFAULT_DELTA_FLOOR_MVAR = 1.0

#: Fallback voltage band [pu], used only if the config carries none. The
#: limits are READ FROM THE CONFIG (``v_min_pu`` / ``v_max_pu``), which at the
#: selected design are 0.9 / 1.1 -- not the 0.95 / 1.05 an earlier version of
#: this script hardcoded, which counted a fifth of all intervals as violating
#: a band the study does not impose.
V_MIN_PU, V_MAX_PU = 0.9, 1.1


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
    """Everything needed to say where a number in Sec. 9.1 came from.

    ``git_dirty`` is the one that matters: the handoff records that two of the
    three existing runs of record were made from an uncommitted tree and are
    therefore not reproducible from their commit hash alone.
    """
    status = _git("status", "--porcelain")
    return {
        "script": "experiments/ch_9_parameter_selection/ch_9_1_ts_period_sweep.py",
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
            "ts_periods_s": list(TS_PERIODS_S),
            "sts_period_s": STS_PERIOD_S,
            "band_mvar": args.band_mvar,
            "delta_floor_mvar": args.delta_floor_mvar,
            "v_min_pu": V_MIN_PU, "v_max_pu": V_MAX_PU,
        },
    }


# =====================================================================
#  Post-processing: one scenario's records -> per-interval rows
# =====================================================================

def _interval_bounds(records: Sequence[Any]) -> List[Tuple[int, int]]:
    """``[(i_start, i_end_exclusive)]`` for each supervisory dispatch interval.

    Segmented on ``tso_active``, which the runner sets when
    ``time_s % tso_period_s < 1``.
    """
    starts = [i for i, r in enumerate(records)
              if bool(getattr(r, "tso_active", False))]
    if not starts:
        return []
    bounds = []
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else len(records)
        if e > s:
            bounds.append((s, e))
    return bounds


def _tap_lockout_flags(records: Sequence[Any], trafo: str,
                       cooldown_s: float, int_cooldown_iters: int,
                       ) -> List[bool]:
    """Per-sample "this changer was unavailable" flags, INFERRED.

    The runner does not log the lockout state, so it is reconstructed from the
    observed tap moves and the two cooldown mechanisms that gate a changer:

    * ``oltc_cooldown_s`` -- wall-clock, and
    * ``int_cooldown`` -- an *iteration* count on the OFO MIQP integer block.

    The binding one is whichever is longer, which at the selected design is the
    iteration count (6 iterations x 20 s = 120 s against a 30 s wall clock).
    This is an inference and is labelled as such wherever it is reported: a tap
    that did not move because the controller chose not to move it is
    indistinguishable here from one that could not.
    """
    n = len(records)
    flags = [False] * n
    prev: Optional[int] = None
    for i, r in enumerate(records):
        pos = getattr(r, "dso_trafo_tap_pos", {}).get(trafo)
        if pos is None:
            continue
        pos = int(pos)
        if prev is not None and pos != prev:
            t_move = float(getattr(r, "time_s", 0.0))
            for j in range(i + 1, n):
                t_j = float(getattr(records[j], "time_s", 0.0))
                locked = ((t_j - t_move) < cooldown_s
                          or (j - i) <= int_cooldown_iters)
                if locked:
                    flags[j] = True
                else:
                    break
        prev = pos
    return flags


def analyse_records(records: Sequence[Any], *, tso_period_s: float,
                    sts_period_s: float, band_mvar: float,
                    delta_floor_mvar: float, cooldown_s: float,
                    cooldown_s_mt: float, int_cooldown_iters: int,
                    v_min_pu: float = V_MIN_PU, v_max_pu: float = V_MAX_PU,
                    ) -> List[Dict[str, Any]]:
    """One row per (interface transformer, supervisory dispatch interval).

    ``rho_k`` is the fraction of the requested correction still outstanding when
    the next dispatch lands -- the residual the supervisory controller inherits,
    normalised by what it asked for. ``n_k`` is the empirical ``N_inner``
    *in situ*: the first subordinate iteration at which the tracking error
    enters the band **and stays** inside it for the rest of the interval.
    Right-censored at ``T_TS / T_STS``; censoring is reported, never dropped.
    """
    rows: List[Dict[str, Any]] = []
    bounds = _interval_bounds(records)
    if not bounds:
        return rows

    trafos = sorted({t for r in records
                     for t in getattr(r, "dso_trafo_q_set_mvar", {}) or {}})
    groups: Dict[str, str] = {}
    for r in records:
        groups.update(getattr(r, "dso_trafo_group", {}) or {})

    n_inner_cap = int(round(tso_period_s / sts_period_s))

    lock = {t: _tap_lockout_flags(
                records, t,
                cooldown_s_mt if str(t).startswith("MT") else cooldown_s,
                int_cooldown_iters)
            for t in trafos}

    for t in trafos:
        prev_set: Optional[float] = None
        for k, (s, e) in enumerate(bounds):
            seg = records[s:e]
            sets = [getattr(r, "dso_trafo_q_set_mvar", {}).get(t) for r in seg]
            sets = [float(v) for v in sets
                    if v is not None and math.isfinite(float(v))]
            if not sets:
                continue
            q_set = sets[-1]

            # Error trace across the interval, against the plant truth.
            err: List[float] = []
            for r in seg:
                a = getattr(r, "dso_trafo_q_actual_mvar", {}).get(t)
                sv = getattr(r, "dso_trafo_q_set_mvar", {}).get(t)
                if a is None or sv is None:
                    err.append(float("nan"))
                else:
                    err.append(abs(float(a) - float(sv)))
            finite = [x for x in err if math.isfinite(x)]
            r_k = finite[-1] if finite else float("nan")

            delta = (abs(q_set - prev_set) if prev_set is not None
                     else float("nan"))
            prev_set = q_set

            # n_k: first index from which the error stays inside the band.
            n_k: Optional[int] = None
            for i in range(len(err)):
                tail = err[i:]
                if tail and all(math.isfinite(x) and x <= band_mvar
                                for x in tail):
                    n_k = i
                    break
            censored = n_k is None
            n_k_report = n_inner_cap if censored else n_k

            caps_min = [getattr(r, "dso_trafo_q_cap_min_mvar", {}).get(t)
                        for r in seg]
            caps_max = [getattr(r, "dso_trafo_q_cap_max_mvar", {}).get(t)
                        for r in seg]
            caps_min = [float(v) for v in caps_min
                        if v is not None and math.isfinite(float(v))]
            caps_max = [float(v) for v in caps_max
                        if v is not None and math.isfinite(float(v))]
            band_w = ((max(caps_max) - min(caps_min))
                      if caps_min and caps_max else float("nan"))

            taps = [getattr(r, "dso_trafo_tap_pos", {}).get(t) for r in seg]
            taps = [int(v) for v in taps if v is not None]
            tap_moves = sum(1 for a, b in zip(taps, taps[1:]) if a != b)

            grp = groups.get(t, "?")
            slack = [getattr(r, "dso_z_slack_max", {}).get(grp) for r in seg]
            slack = [float(v) for v in slack
                     if v is not None and math.isfinite(float(v))]
            sig = [getattr(r, "dso_sigma_norm", {}).get(grp) for r in seg]
            sig = [float(v) for v in sig
                   if v is not None and math.isfinite(float(v))]
            vmin = [getattr(r, "dso_group_v_min_pu", {}).get(grp) for r in seg]
            vmax = [getattr(r, "dso_group_v_max_pu", {}).get(grp) for r in seg]
            vmin = [float(v) for v in vmin
                    if v is not None and math.isfinite(float(v))]
            vmax = [float(v) for v in vmax
                    if v is not None and math.isfinite(float(v))]

            lock_seg = lock[t][s:e]
            rows.append({
                "trafo": t, "group": grp, "k": k,
                "tso_period_s": float(tso_period_s),
                "t_start_s": float(getattr(seg[0], "time_s", float("nan"))),
                "n_samples": len(seg),
                "q_set_mvar": q_set,
                "delta_mvar": delta,
                "r_k_mvar": r_k,
                "rho_k": (r_k / delta if (math.isfinite(delta)
                                          and delta >= delta_floor_mvar
                                          and math.isfinite(r_k))
                          else float("nan")),
                "delta_below_floor": bool(math.isfinite(delta)
                                          and delta < delta_floor_mvar),
                "n_k": n_k_report, "n_k_censored": censored,
                "n_inner_cap": n_inner_cap,
                "cair_width_mvar": band_w,
                "tap_moves": tap_moves,
                "lockout_occupancy": (sum(lock_seg) / len(lock_seg)
                                      if lock_seg else float("nan")),
                "z_slack_max": max(slack) if slack else float("nan"),
                "sigma_norm_max": max(sig) if sig else float("nan"),
                "v_min_pu": min(vmin) if vmin else float("nan"),
                "v_max_pu": max(vmax) if vmax else float("nan"),
                "v_violation": bool((vmin and min(vmin) < v_min_pu)
                                    or (vmax and max(vmax) > v_max_pu)),
            })
    return rows


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


def _run_point(job: Dict[str, Any]) -> Dict[str, Any]:
    """One (window, ``T_TS``) simulation, post-processed to interval rows.

    Runs in a worker process and returns only the compact rows, never the
    record list: a 90 min window at 20 s carries 270 records of nested dicts,
    and shipping every one of those back through IPC is pointless.
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
    # The supervisory period lives in TWO places; see _retime_parent.
    cfg = _retime_parent(cfg, float(job["tso_period_s"]),
                         job.get("sbx_cycle", "iterations"))
    spec = next(s for s in tier1_design_set() if s.name == job["window"])
    spec = dataclasses.replace(spec, tso_period_s=float(job["tso_period_s"]),
                               dso_period_s=STS_PERIOD_S, dt_s=STS_PERIOD_S)

    res, records = _run_scenario(spec, cfg, MetricScales())
    rows = analyse_records(
        records, tso_period_s=float(job["tso_period_s"]),
        sts_period_s=STS_PERIOD_S, band_mvar=float(job["band_mvar"]),
        delta_floor_mvar=float(job["delta_floor_mvar"]),
        cooldown_s=float(cfg.oltc_cooldown_s),
        cooldown_s_mt=float(cfg.oltc_cooldown_s_mt or cfg.oltc_cooldown_s),
        int_cooldown_iters=int(cfg.int_cooldown),
        v_min_pu=float(getattr(cfg, "v_min_pu", V_MIN_PU)),
        v_max_pu=float(getattr(cfg, "v_max_pu", V_MAX_PU)))
    for r in rows:
        r["window"] = job["window"]
    return {"window": job["window"], "tso_period_s": float(job["tso_period_s"]),
            "failure": res.failure_reason, "wall_s": res.wall_time_s,
            "n_records": len(records), "rows": rows}


# =====================================================================
#  Aggregation
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
    """Distributions per (``T_TS``, group) and per ``T_TS`` pooled.

    Never one scalar per ``T_TS``: see the module docstring on why a
    time-aggregated residual is not comparable across periods.
    """
    out: List[Dict[str, Any]] = []
    periods = sorted({r["tso_period_s"] for r in rows})
    for p in periods:
        sub_p = [r for r in rows if r["tso_period_s"] == p]
        groups = sorted({r["group"] for r in sub_p}) + ["__pooled__"]
        for g in groups:
            sub = (sub_p if g == "__pooled__"
                   else [r for r in sub_p if r["group"] == g])
            if not sub:
                continue
            rho = [r["rho_k"] for r in sub if math.isfinite(r["rho_k"])]
            nk = [float(r["n_k"]) for r in sub]
            # Censored intervals all report n_k = the cap, so once censoring
            # exceeds 5 % the p95 IS the cap and carries no information about
            # how long settling takes -- only that it did not happen. The
            # uncensored quantiles answer "when it settles, how fast", and the
            # censoring fraction answers "how often it does not". Reporting
            # only the pooled p95 conflates the two into a number that looks
            # like a measurement and is an artefact of the interval length.
            nk_unc = [float(r["n_k"]) for r in sub if not r["n_k_censored"]]
            n_cens = sum(1 for r in sub if r["n_k_censored"])
            occ = [r["lockout_occupancy"] for r in sub
                   if math.isfinite(r["lockout_occupancy"])]
            out.append({
                "tso_period_s": p,
                "n_inner_configured": int(round(p / STS_PERIOD_S)),
                "group": g,
                "n_intervals": len(sub),
                "n_intervals_scored": len(rho),
                "n_dropped_delta_floor": sum(1 for r in sub
                                             if r["delta_below_floor"]),
                "rho_median": _quantile(rho, 0.50),
                "rho_p95": _quantile(rho, 0.95),
                "rho_max": max(rho) if rho else float("nan"),
                "n_k_median": _quantile(nk, 0.50),
                "n_k_p95": _quantile(nk, 0.95),
                "n_k_max": max(nk) if nk else float("nan"),
                "n_k_uncensored_median": _quantile(nk_unc, 0.50),
                "n_k_uncensored_p95": _quantile(nk_unc, 0.95),
                "n_k_uncensored_max": max(nk_unc) if nk_unc else float("nan"),
                "n_uncensored": len(nk_unc),
                "censoring_fraction": n_cens / len(sub),
                "cair_width_median_mvar": _quantile(
                    [r["cair_width_mvar"] for r in sub], 0.50),
                "lockout_occupancy_mean": (sum(occ) / len(occ)
                                           if occ else float("nan")),
                "tap_moves_total": sum(r["tap_moves"] for r in sub),
                "tap_moves_per_interval": (sum(r["tap_moves"] for r in sub)
                                           / len(sub)),
                "v_violations": sum(1 for r in sub if r["v_violation"]),
                # Counts are NOT comparable across periods -- a short T_TS
                # yields proportionally more intervals, so a raw count of
                # violating intervals rises with dispatch frequency for the
                # same physical behaviour. This is the same trap the rho
                # distribution exists to avoid; the fraction is the comparable
                # quantity and is what the summary tabulates.
                "v_violation_fraction": (sum(1 for r in sub
                                             if r["v_violation"]) / len(sub)),
                "z_slack_max": max((r["z_slack_max"] for r in sub
                                    if math.isfinite(r["z_slack_max"])),
                                   default=float("nan")),
            })
    return out


# =====================================================================
#  Outputs
# =====================================================================

_ROW_COLS = ["window", "tso_period_s", "trafo", "group", "k", "t_start_s",
             "n_samples", "q_set_mvar", "delta_mvar", "r_k_mvar", "rho_k",
             "delta_below_floor", "n_k", "n_k_censored", "n_inner_cap",
             "cair_width_mvar", "tap_moves", "lockout_occupancy",
             "z_slack_max", "sigma_norm_max", "v_min_pu", "v_max_pu",
             "v_violation"]


def write_outputs(out_dir: Path, rows: List[Dict[str, Any]],
                  agg: List[Dict[str, Any]], failures: List[Dict[str, Any]],
                  meta: Optional[Dict[str, Any]] = None) -> None:
    """Machine-readable rows, the aggregate table, and the prose summary."""
    with (out_dir / "intervals.csv").open("w", newline="",
                                          encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_ROW_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    agg_cols = list(agg[0].keys()) if agg else []
    with (out_dir / "summary_by_period.csv").open("w", newline="",
                                                  encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=agg_cols)
        w.writeheader()
        for r in agg:
            w.writerow(r)

    md = ["# T_TS sweep (thesis Sec. 9.1, selection of the supervisory period)",
          "",
          "`rho_k = r_k / delta_k` is the fraction of the requested correction "
          "still outstanding when the next dispatch lands. `n_k` is the "
          "empirical `N_inner` *in situ*, right-censored at `T_TS / T_STS`.",
          "",
          "**Reported as a distribution per `T_TS`, never as one "
          "time-aggregated scalar**: at a fixed horizon a short period yields "
          "proportionally more dispatches, so a pooled RMS residual mixes "
          "residual-per-dispatch with dispatch frequency.",
          "",
          "**`N_inner` is not read off this sweep.** This is the closed-loop "
          "selection evidence for `T_TS`; the isolated measurement of eq. (9.2) "
          "is `ch_9_1_ninner_isolated_sts.py`.",
          "",
          "## Pooled over all interfaces", "",
          "| T_TS [s] | N_inner cfg | intervals | scored | rho med | rho p95 | "
          "rho max | n_k med | n_k p95 | censored | lockout occ | taps/ival | "
          "V viol frac |",
          "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for a in [x for x in agg if x["group"] == "__pooled__"]:
        md.append(
            f"| {a['tso_period_s']:.0f} | {a['n_inner_configured']} | "
            f"{a['n_intervals']} | {a['n_intervals_scored']} | "
            f"{a['rho_median']:.3f} | {a['rho_p95']:.3f} | {a['rho_max']:.3f} | "
            f"{a['n_k_median']:.1f} | {a['n_k_p95']:.1f} | "
            f"{a['censoring_fraction']:.2f} | "
            f"{a['lockout_occupancy_mean']:.2f} | "
            f"{a['tap_moves_per_interval']:.3f} | "
            f"{a['v_violation_fraction']:.3f} |")

    md += ["", "## Per interface (STS)", "",
           "| T_TS [s] | group | intervals | scored | rho med | rho p95 | "
           "rho max | n_k med | n_k p95 | censored | CAIR width med [Mvar] | "
           "lockout occ | taps/ival | V viol frac | max z_slack |",
           "|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for a in [x for x in agg if x["group"] != "__pooled__"]:
        md.append(
            f"| {a['tso_period_s']:.0f} | {a['group']} | {a['n_intervals']} | "
            f"{a['n_intervals_scored']} | {a['rho_median']:.3f} | "
            f"{a['rho_p95']:.3f} | {a['rho_max']:.3f} | "
            f"{a['n_k_median']:.1f} | {a['n_k_p95']:.1f} | "
            f"{a['censoring_fraction']:.2f} | "
            f"{a['cair_width_median_mvar']:.1f} | "
            f"{a['lockout_occupancy_mean']:.2f} | "
            f"{a['tap_moves_per_interval']:.3f} | "
            f"{a['v_violation_fraction']:.3f} | {a['z_slack_max']:.3g} |")

    md += ["", "## What rho can and cannot show", "",
           "`rho_k` measures how well the subordinate layer executed the "
           "correction it was **told** to make. It cannot show whether that "
           "correction was still the right one by the time it landed, so a "
           "stale-setpoint cost at long `T_TS` does **not** appear here and "
           "the absence of a U-shape in `rho` is not evidence against one. "
           "That cost lives in the supervisory tracking objective (`f_ts`, "
           "`f_q`), which this script does not compute.", "",
           "## Reading the lockout column", "",
           "`lockout occ` is the mean fraction of subordinate iterations in a "
           "dispatch interval during which the tap changer was unavailable. It "
           "is **inferred** from observed tap moves and the two cooldown "
           "mechanisms (`oltc_cooldown_s` wall-clock and `int_cooldown` "
           "iterations), not logged by the runner: a changer that did not move "
           "because the controller chose not to move it is indistinguishable "
           "here from one that could not. It matters most at `T_TS = 60 s`, "
           "where it separates *the subordinate layer cannot converge in three "
           "iterations* from *the changer was unavailable*."]

    if failures:
        md += ["", "## Failed points", "",
               "**The result is not complete.** Each of these is a scenario "
               "that raised; its intervals are absent from every distribution "
               "above.", ""]
        md += [f"- `{f['window']}` at `T_TS = {f['tso_period_s']:.0f} s`: "
               f"{(f['failure'] or '').splitlines()[0] if f['failure'] else 'unknown'}"
               for f in failures]

    cens = [a for a in agg
            if a["group"] == "__pooled__" and a["censoring_fraction"] >= 0.5]
    if cens:
        md += ["", "## Censoring warning", "",
               "At these periods at least half of all dispatch intervals never "
               "brought the tracking error inside the band and stayed there, so "
               "`n_k` is a lower bound set by the interval length rather than a "
               "measurement. Reconsider the band before quoting `n_k` here.", ""]
        md += [f"- `T_TS = {a['tso_period_s']:.0f} s`: "
               f"censoring fraction {a['censoring_fraction']:.2f}" for a in cens]

    if meta:
        d = meta.get("design", {})
        md += ["", "## Provenance", "",
               f"- run: `{meta['timestamp']}`",
               f"- commit: `{meta['git_commit']}` on `{meta['git_branch']}`"
               + ("  **(working tree dirty -- not reproducible from the "
                  "commit alone)**" if meta.get("git_dirty") else ""),
               f"- weights: campaign `{d.get('campaign')}` candidate "
               f"`{d.get('candidate_key')}`, {d.get('verified')}",
               f"- archived `rho_emp_p95` of that candidate: "
               f"`{d.get('archived_rho_emp_p95')}`",
               f"- settling band {meta['constants']['band_mvar']} Mvar, "
               f"delta floor {meta['constants']['delta_floor_mvar']} Mvar",
               f"- `T_STS` = {STS_PERIOD_S:.0f} s fixed "
               f"(`dso_period_s = dt_s`), bank `tier1_design_set` on "
               f"`rural_700`",
               f"- command: `{' '.join(meta['argv'])}`"]

    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


# =====================================================================
#  Re-aggregation of a completed run
# =====================================================================

def reaggregate(run_dir: Path, out_dir: Path, *, v_min_pu: float,
                v_max_pu: float) -> int:
    """Rebuild the aggregate and the summary from a finished run's rows.

    ``intervals.csv`` stores the raw per-interval quantities -- ``rho_k``,
    ``n_k``, the censoring flag, and the voltage extremes -- so a correction to
    a *derived* column can be applied exactly without re-simulating. Used after
    the voltage band was found to be hardcoded rather than read from the
    config: the 60 scenario runs behind the rows are unaffected by that defect,
    and re-running them would consume an hour of a shared machine to reproduce
    numbers that are already on disk.

    Writes to a NEW directory and records the source, so the corrected result
    never overwrites the run it was derived from.
    """
    src = run_dir / "intervals.csv"
    if not src.exists():
        print(f"[reaggregate] no intervals.csv under {run_dir}")
        return 1

    rows: List[Dict[str, Any]] = []
    with src.open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            r: Dict[str, Any] = dict(raw)
            for k in ("tso_period_s", "t_start_s", "q_set_mvar", "delta_mvar",
                      "r_k_mvar", "rho_k", "cair_width_mvar",
                      "lockout_occupancy", "z_slack_max", "sigma_norm_max",
                      "v_min_pu", "v_max_pu"):
                try:
                    r[k] = float(raw[k])
                except (KeyError, TypeError, ValueError):
                    r[k] = float("nan")
            for k in ("k", "n_samples", "n_k", "n_inner_cap", "tap_moves"):
                try:
                    r[k] = int(float(raw[k]))
                except (KeyError, TypeError, ValueError):
                    r[k] = 0
            r["n_k_censored"] = str(raw.get("n_k_censored")) == "True"
            r["delta_below_floor"] = str(raw.get("delta_below_floor")) == "True"
            # The one derived column being corrected.
            r["v_violation"] = bool(r["v_min_pu"] < v_min_pu
                                    or r["v_max_pu"] > v_max_pu)
            rows.append(r)

    agg = aggregate(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = None
    src_meta = run_dir / "run_meta.json"
    if src_meta.exists():
        meta = json.loads(src_meta.read_text(encoding="utf-8"))
        meta.setdefault("constants", {})
        meta["constants"]["v_min_pu"] = v_min_pu
        meta["constants"]["v_max_pu"] = v_max_pu
        meta["reaggregated_from"] = str(run_dir)
        meta["reaggregation_note"] = (
            "Derived columns recomputed from the stored per-interval rows; the "
            "scenario runs themselves are those of the source directory and "
            "were not repeated. Only the voltage-violation flag changed: it "
            "was evaluated against a hardcoded 0.95/1.05 band instead of the "
            "configured v_min_pu/v_max_pu.")
        (out_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8")

    write_outputs(out_dir, rows, agg, failures=[], meta=meta)
    print(f"[reaggregate] {len(rows)} rows from {run_dir}")
    print(f"[reaggregate] band {v_min_pu}-{v_max_pu} pu -> {out_dir}")
    for a in [x for x in agg if x["group"] == "__pooled__"]:
        print(f"  T_TS {a['tso_period_s']:5.0f}  rho med {a['rho_median']:.3f}"
              f"  cens {a['censoring_fraction']:.2f}"
              f"  n_k(unc) med {a['n_k_uncensored_median']:.1f}"
              f"  p95 {a['n_k_uncensored_p95']:.1f}"
              f"  V viol frac {a['v_violation_fraction']:.3f}")
    return 0


# =====================================================================
#  Offline self-test (no simulation)
# =====================================================================

class _R:
    """Minimal stand-in for ``MultiTSOIterationRecord``."""

    def __init__(self, t, tso, q_set, q_act, tap, cap=(-100.0, 100.0)):
        self.time_s = t
        self.tso_active = tso
        self.dso_active = True
        self.dso_trafo_q_set_mvar = {"T1": q_set}
        self.dso_trafo_q_actual_mvar = {"T1": q_act}
        self.dso_trafo_q_meas_mvar = {"T1": q_act}
        self.dso_trafo_tap_pos = {"T1": tap}
        self.dso_trafo_group = {"T1": "DSO_1"}
        self.dso_trafo_q_cap_min_mvar = {"T1": cap[0]}
        self.dso_trafo_q_cap_max_mvar = {"T1": cap[1]}
        self.dso_z_slack_max = {"DSO_1": 0.0}
        self.dso_sigma_norm = {"DSO_1": 0.1}
        self.dso_group_v_min_pu = {"DSO_1": 0.99}
        self.dso_group_v_max_pu = {"DSO_1": 1.01}


def self_test() -> int:
    """Check the post-processing offline: no runner, no plant, no cores."""
    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
              + (f" -- {detail}" if detail and not cond else ""))
        ok = ok and cond

    print("[self-test] T_TS sweep post-processing")

    # Two dispatch intervals of 3 subordinate iterations each (T_TS = 60 s).
    # Interval 0: setpoint 10, error decays 5 -> 2 -> 0.5 (enters band at i=2).
    # Interval 1: setpoint 20 (delta = 10), error stays at 4 -> censored.
    recs = [
        _R(0.0,  True,  10.0, 5.0,  0), _R(20.0, False, 10.0, 8.0, 0),
        _R(40.0, False, 10.0, 9.5,  0),
        _R(60.0, True,  20.0, 16.0, 0), _R(80.0, False, 20.0, 16.0, 0),
        _R(100.0, False, 20.0, 16.0, 0),
    ]
    rows = analyse_records(recs, tso_period_s=60.0, sts_period_s=20.0,
                           band_mvar=1.0, delta_floor_mvar=1.0,
                           cooldown_s=30.0, cooldown_s_mt=180.0,
                           int_cooldown_iters=6)
    check("one row per (trafo, interval)", len(rows) == 2, f"got {len(rows)}")

    i0, i1 = rows[0], rows[1]
    check("first interval has no delta (no predecessor)",
          math.isnan(i0["delta_mvar"]))
    check("delta_k = |q_set(k) - q_set(k-1)| = 10.0",
          abs(i1["delta_mvar"] - 10.0) < 1e-9, f"got {i1['delta_mvar']}")
    check("r_k is the LAST sample before the next dispatch (0.5)",
          abs(i0["r_k_mvar"] - 0.5) < 1e-9, f"got {i0['r_k_mvar']}")
    check("rho_k = r_k / delta_k = 4.0 / 10.0",
          abs(i1["rho_k"] - 0.4) < 1e-9, f"got {i1['rho_k']}")
    check("n_k is the first index from which the error STAYS in band",
          i0["n_k"] == 2 and not i0["n_k_censored"], f"got {i0['n_k']}")
    check("an interval that never enters the band is censored at T_TS/T_STS",
          i1["n_k_censored"] and i1["n_k"] == 3, f"got {i1['n_k']}")
    check("CAIR width is read from the reported band",
          abs(i0["cair_width_mvar"] - 200.0) < 1e-9,
          f"got {i0['cair_width_mvar']}")

    # A late excursion must NOT count as settled: entering the band and
    # leaving it again is the failure the "and stays" rule exists to catch.
    recs_bounce = [
        _R(0.0,  True,  10.0, 9.5, 0), _R(20.0, False, 10.0, 9.6, 0),
        _R(40.0, False, 10.0, 4.0, 0),
    ]
    rb = analyse_records(recs_bounce, tso_period_s=60.0, sts_period_s=20.0,
                         band_mvar=1.0, delta_floor_mvar=1.0, cooldown_s=30.0,
                         cooldown_s_mt=180.0, int_cooldown_iters=6)
    check("entering the band then leaving it is censored, not settled",
          rb[0]["n_k_censored"], f"got n_k={rb[0]['n_k']}")

    # Delta floor.
    recs_small = [
        _R(0.0,  True, 10.0, 10.0, 0), _R(20.0, False, 10.0, 10.0, 0),
        _R(40.0, True, 10.2, 10.2, 0), _R(60.0, False, 10.2, 10.2, 0),
    ]
    rs = analyse_records(recs_small, tso_period_s=40.0, sts_period_s=20.0,
                         band_mvar=1.0, delta_floor_mvar=1.0, cooldown_s=30.0,
                         cooldown_s_mt=180.0, int_cooldown_iters=6)
    check("an interval the TS barely moved is dropped from the ratio",
          rs[1]["delta_below_floor"] and math.isnan(rs[1]["rho_k"]))

    # Lockout: a tap move locks the changer for int_cooldown iterations.
    recs_tap = [_R(float(20 * i), i % 3 == 0, 10.0, 10.0, 0 if i < 1 else 1)
                for i in range(9)]
    rt = analyse_records(recs_tap, tso_period_s=60.0, sts_period_s=20.0,
                         band_mvar=1.0, delta_floor_mvar=1.0, cooldown_s=30.0,
                         cooldown_s_mt=180.0, int_cooldown_iters=6)
    check("a tap move marks the changer locked for the iteration cooldown",
          any(r["lockout_occupancy"] > 0 for r in rt),
          f"got {[r['lockout_occupancy'] for r in rt]}")
    check("the tap move itself is counted",
          sum(r["tap_moves"] for r in rt) == 1,
          f"got {sum(r['tap_moves'] for r in rt)}")

    agg = aggregate(rows)
    pooled = [a for a in agg if a["group"] == "__pooled__"]
    check("aggregation reports a pooled row and a per-group row",
          len(pooled) == 1 and len(agg) == 2, f"got {len(agg)}")
    check("censoring fraction is reported, not silently dropped",
          abs(pooled[0]["censoring_fraction"] - 0.5) < 1e-9,
          f"got {pooled[0]['censoring_fraction']}")
    check("rho distribution ignores the unscored first interval",
          pooled[0]["n_intervals_scored"] == 1,
          f"got {pooled[0]['n_intervals_scored']}")

    # No supervisory dispatch at all must not crash.
    check("a record list with no TS dispatch yields no rows",
          analyse_records([_R(0.0, False, 1.0, 1.0, 0)], tso_period_s=60.0,
                          sts_period_s=20.0, band_mvar=1.0,
                          delta_floor_mvar=1.0, cooldown_s=30.0,
                          cooldown_s_mt=180.0, int_cooldown_iters=6) == [])

    # The worker reads RunResult by attribute name, and a wrong name only
    # surfaces AFTER the simulation has run -- nine 90 min scenarios were
    # spent on ``res.failure`` before this check existed (2026-08-19).
    try:
        import dataclasses as _dc

        from tuning.runner import RunResult
        fields = {f.name for f in _dc.fields(RunResult)}
        check("the worker reads RunResult fields that exist",
              {"failure_reason", "wall_time_s"} <= fields,
              f"RunResult has {sorted(fields)}")
    except ImportError as exc:
        check("RunResult importable for the field check", False, str(exc))

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        write_outputs(out, rows, agg, failures=[], meta=None)
        for name in ("intervals.csv", "summary_by_period.csv", "summary.md"):
            check(f"{name} written", (out / name).exists())
        body = (out / "summary.md").read_text(encoding="utf-8")
        check("summary refuses to sell the sweep as N_inner",
              "is not read off this sweep" in body)
        check("summary states the lockout column is inferred",
              "inferred" in body)

    print(f"[self-test] {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


# =====================================================================
#  Entry point
# =====================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sweep T_TS at fixed T_STS = 20 s (thesis Sec. 9.1).")
    ap.add_argument("--label", default="ts_period_sweep")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "ch9_ts_period_sweep")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel scenarios; the server is shared, so this "
                         "is a budget, not a maximum to be filled")
    ap.add_argument("--band-mvar", type=float, default=DEFAULT_BAND_MVAR,
                    help="settling band on the interface flow [Mvar]")
    ap.add_argument("--delta-floor-mvar", type=float,
                    default=DEFAULT_DELTA_FLOOR_MVAR,
                    help="intervals with a smaller commanded change are "
                         "dropped from the rho_k distribution")
    ap.add_argument("--periods", default="",
                    help="comma-separated T_TS values [s]; default the full grid")
    ap.add_argument("--windows", default="",
                    help="comma-separated window names; default all 12")
    ap.add_argument("--sbx-cycle", choices=("iterations", "wallclock"),
                    default="iterations",
                    help="how the SBX-H settlement cycle follows T_TS: hold "
                         "k_sched (default, holds the controller fixed) or "
                         "hold the cycle near 360 s by re-deriving k_sched")
    ap.add_argument("--pilot", action="store_true",
                    help=f"{PILOT_PERIODS_S} on {PILOT_N_WINDOWS} windows, to "
                         f"read the CAIR widths before spending the full grid")
    ap.add_argument("--self-test", action="store_true",
                    help="check the post-processing offline; no simulation")
    ap.add_argument("--reaggregate", type=Path, default=None,
                    help="rebuild the aggregate and summary from a finished "
                         "run's intervals.csv, without re-simulating")
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test()

    if a.reaggregate is not None:
        from _ch9_selected_design import build_selected_config
        cfg, _d = build_selected_config()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return reaggregate(
            a.reaggregate, a.reaggregate.parent / f"{a.reaggregate.name}_reagg_{stamp}",
            v_min_pu=float(getattr(cfg, "v_min_pu", V_MIN_PU)),
            v_max_pu=float(getattr(cfg, "v_max_pu", V_MAX_PU)))

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
            print(f"[abort] unknown window(s): {unknown}; have {names}")
            return 1
        names = want
    periods = ([float(x) for x in a.periods.split(",") if x.strip()]
               if a.periods else list(TS_PERIODS_S))
    if a.pilot:
        periods = list(PILOT_PERIODS_S)
        names = names[:PILOT_N_WINDOWS]

    bad = [p for p in periods if abs(p / STS_PERIOD_S
                                     - round(p / STS_PERIOD_S)) > 1e-9]
    if bad:
        print(f"[abort] T_TS values {bad} are not integer multiples of "
              f"T_STS = {STS_PERIOD_S} s; the GCD requirement on dt_s "
              f"(configs/config.py:188) would be violated.")
        return 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = a.out / a.label / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (a.out / a.label / "_latest.txt").write_text(stamp, encoding="utf-8")

    meta = provenance(a, design)
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")

    jobs = [{"window": w, "tso_period_s": p, "band_mvar": a.band_mvar,
             "delta_floor_mvar": a.delta_floor_mvar,
             "sbx_cycle": a.sbx_cycle}
            for p in periods for w in names]
    # Written BEFORE the runs, so an aborted campaign still documents what it
    # meant to measure.
    with (out_dir / "cases.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["window", "role", "season", "stratum", "tso_period_s",
                    "n_inner_configured"])
        for j in jobs:
            m = WINDOW_META.get(j["window"], {})
            w.writerow([j["window"], m.get("role"), m.get("season"),
                        m.get("stratum"), j["tso_period_s"],
                        int(round(j["tso_period_s"] / STS_PERIOD_S))])

    print(f"[sweep] -> {out_dir}")
    print(f"[sweep] {len(jobs)} runs = {len(periods)} periods x "
          f"{len(names)} windows, {a.workers} workers")
    print(f"[sweep] commit {meta['git_commit']} on {meta['git_branch']}"
          + ("  [WORKING TREE DIRTY]" if meta.get("git_dirty") else ""))

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_run_point, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            done += 1
            try:
                res = fut.result()
            except Exception as exc:  # a worker died outright
                failures.append({"window": j["window"],
                                 "tso_period_s": j["tso_period_s"],
                                 "failure": f"{type(exc).__name__}: {exc}"})
                print(f"  [{done}/{len(jobs)}] {j['window']} @ "
                      f"{j['tso_period_s']:.0f}s  WORKER DIED: {exc}")
                continue
            if res["failure"]:
                failures.append(res)
                print(f"  [{done}/{len(jobs)}] {res['window']} @ "
                      f"{res['tso_period_s']:.0f}s  FAILED: "
                      f"{res['failure'].splitlines()[0]}")
                continue
            rows.extend(res["rows"])
            print(f"  [{done}/{len(jobs)}] {res['window']} @ "
                  f"{res['tso_period_s']:.0f}s  {res['n_records']} records, "
                  f"{len(res['rows'])} intervals, {res['wall_s']:.0f} s")

    if not rows:
        print("[abort] no scenario produced an interval; nothing written")
        return 1

    agg = aggregate(rows)
    write_outputs(out_dir, rows, agg, failures, meta)

    print("\n[sweep] pooled result:")
    print(f"  {'T_TS':>6} {'N_cfg':>6} {'rho med':>8} {'rho p95':>8} "
          f"{'n_k med':>8} {'cens':>6} {'lockout':>8} {'CAIR w':>8}")
    for a_ in [x for x in agg if x["group"] == "__pooled__"]:
        print(f"  {a_['tso_period_s']:6.0f} {a_['n_inner_configured']:6d} "
              f"{a_['rho_median']:8.3f} {a_['rho_p95']:8.3f} "
              f"{a_['n_k_median']:8.1f} {a_['censoring_fraction']:6.2f} "
              f"{a_['lockout_occupancy_mean']:8.2f} "
              f"{a_['cair_width_median_mvar']:8.1f}")

    censored_all = [x for x in agg
                    if x["group"] == "__pooled__"
                    and x["censoring_fraction"] >= 0.5]
    rc = 0
    if failures:
        print(f"\n[sweep] {len(failures)} point(s) FAILED -- result incomplete")
        rc = 2
    if censored_all:
        print(f"\n[sweep] {len(censored_all)} period(s) with >= 50 % censored "
              f"intervals: the band may be unreachable; see summary.md")
        rc = 2
    print(f"\n[sweep] wrote {out_dir}  (exit {rc})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
