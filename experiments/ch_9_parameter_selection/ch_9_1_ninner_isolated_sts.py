r"""Settling of the isolated subordinate loop: ``n_s`` against the ``N_inner``
budget.

Produces the data behind ``tab:param:timescales:ninner`` and
``fig:param:timescales:settling_cdf``: the settling counts that Experiment C
(``ch:param:experiment:e``) checks against the requirement
``n_s,95 <= 3*N_inner`` = 27 at the final weights.

**It does not measure ``N_inner``, and the chapter no longer asks it to.**
This file was written for the superseded chapter, where ``N_inner`` was
"measured rather than assumed" as eq. (9.2) of §9.1.  Since the 2026-09-29
restructure ``T_TS`` = 180 s is set by definition and ``N_inner`` = 9 follows
from ``T_TS / T_STS``; what this script contributes is ``n_s``, the settling
count, which is checked against a budget *expressed* in ``N_inner``.  The
equation is also no longer numbered 9.2, so cite the label rather than the
number.

The measurement definition below is unchanged and is the part that matters:
``n_s`` is the number of subordinate (STS-OFO) iterations a
capability-band-traversing interface-Q setpoint step needs before **the
iteration converges** -- that is, before the interface flow goes flat,
``|q_pcc(k) - q_pcc(k-1)| -> 0``. The supervisory layer is silent throughout,
so what is measured is the subordinate loop alone.

**Convergence is flatness, not arrival.** The subordinate controller is
multi-objective: it trades interface-Q tracking against voltage regulation, so
its optimum generally sits at a non-zero Q offset from the commanded setpoint.
A converged loop therefore need NOT reach ``q_set``, and requiring it to would
score a correctly-converged controller as a failure.

*Corrected 2026-08-20.* An earlier version defined ``N_inner`` as the first
iteration at which ``|q_pcc - q_set|`` entered a band and stayed -- a tracking
criterion. Against the final tuned weights it reported 91 % of steps as
"censored" while the areas sat with 20-31 Mvar of reported headroom, zero
voltage slack and, for DSO_3, zero tap moves: the loops had converged to their
multi-objective optima and the criterion could not see it. The steady-state
offset is now reported as its own quantity, which is where that trade-off
belongs.

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

**Relaxing the capability (``--der-q-override-pu``).** The first full run found
that 89 % of band-edge steps never settle, and that this is *reachability*, not
loop speed: the censored cases end a median 11 Mvar (23 % of the commanded
step) from a setpoint placed at 95 % of the capability the subordinate layer
itself reported. To measure settling *as such*, the DER reactive capability can
be forced to a symmetric ``[-x*S_n, +x*S_n]`` box via
``MultiTSOConfig.der_q_capability_override_pu``, which
``ActuatorBounds._compute_single_der_q_capability`` and the plant's own clip
honour identically -- they must, or the controller would optimise against a
plant it does not have.

**The step magnitudes stay those of the REAL capability.** Targets are read in
a probe pass run at the *baseline* config and only the measurement pass is
relaxed. Sizing the step from the relaxed band instead would command a Q the
scenario never declares and measure nothing useful; keeping it fixed makes the
two runs a clean A/B on the same physical ask.

**This is a diagnostic, not a physical model.** ``+-x`` pu at rated P implies
``S = sqrt(1 + x^2)`` pu, i.e. a converter oversized beyond what the scenario
declares. Every output says so, and a run made under it must never be quoted as
one made under real limits. Watch ``z_slack_max`` in the result: if settling is
still refused with capability relaxed, the binding constraint has moved to
voltage, which is the more interesting finding.

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
    python experiments\ch_9_parameter_selection\ch_9_1_ninner_isolated_sts.py --max-step-mvar 20
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
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

from _ch9_selected_design import (  # noqa: E402
    INITIAL_PARAMSET, SELECTED_PARAMSET,
)

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

#: Flatness tolerance on the per-iteration CHANGE in interface flow [Mvar].
#:
#: The convergence criterion: the loop has converged at iteration ``k`` when
#: ``|q(k) - q(k-1)|`` is at or below this and stays there. It is deliberately
#: a separate quantity from :data:`DEFAULT_BAND_MVAR`, which bounds a tracking
#: ERROR; conflating the two is what made a converged-but-offset loop look
#: censored. Reported across a sweep so the choice is visible rather than
#: decisive.
DEFAULT_FLAT_MVAR = 0.1

#: Flatness tolerances reported alongside the primary one [Mvar].
#:
#: Extended 2026-09-15 from ``(0.02, 0.05, 0.1, 0.25, 1.0)``.  The old ladder
#: jumped 4x between 0.25 and 1.0, and the interesting region turned out to be
#: inside that gap: on the Experiment-C bank the p95 of ``N_inner`` falls from
#: 14.0 to 1.0 across it, so every question of the form "at which tolerance
#: does the p95 reach N?" had to be answered by interpolating a 4x span.  The
#: median crossing (0.211 Mvar for N = 9) was inside a measured interval; the
#: p95 crossing (~0.41 Mvar) was not.
#:
#: The added points are geometric, so the ladder is uniform in log-tolerance --
#: which is the axis ``N_inner`` is roughly linear in, since the tail of the
#: iteration decays geometrically.
FLAT_SWEEP_MVAR: Tuple[float, ...] = (
    0.02, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0,
)

#: Band [Mvar] around the FINAL interface flow for the settling count
#: ``n_settle_final_1mvar`` (author decision 2026-09-28, handoff
#: ``HANDOFF_D_settling_rerun.md`` section 2).
SETTLE_FINAL_BAND_MVAR = 1.0

#: Iterations before the end of the observation over which the final value
#: must itself be steady; otherwise the step is censored.
SETTLE_FINAL_GUARD_ITER = 10

#: The subordinate iteration budget per supervisory period, ``N_inner``.  Used
#: only to report the share of steps settled within it.
N_INNER_BUDGET = 9

V_MIN_PU, V_MAX_PU = 0.9, 1.1

#: Design-bank strata in which the DER reactive capability is structurally
#: zero, so no capability-band traversal exists to measure.
#:
#: ``WINDOW_META[...]["stratum"]`` is exactly the DER reactive-capability tier,
#: as the T_TS sweep's own measured band widths confirm: ``full`` ~190 Mvar
#: (6 windows), ``partial`` ~55-85 Mvar (4), and ``none`` ~0.1-0.7 Mvar
#: (``d_quiet_summer``, ``d_ramp_up_winter``). The last is the VDE dead zone
#: that the retired ``tuning_mc.stage_1_search`` already named as the reason ``tau``,
#: ``lambda_dso`` and ``dso_g_v_ratio`` are structurally inert there.
#:
#: These windows are excluded by default, and the exclusion is REPORTED: a
#: window with no admissible traversal is not a failed measurement, and
#: counting it as one would put a spurious censoring fraction into eq. (9.2).
DEAD_ZONE_STRATA = ("none",)

#: Window banks this experiment can be measured on.
#:
#: ``conditions`` -- the six operating conditions W1-W6 of thesis table
#: ``tab:timescales:conditions``, declared in ``tuning/scenarios_mc_v2.py``.
#: This is the default from 2026-09-15: F1 and F2 are specified on W1-W6, and a
#: subordinate-response measurement taken on a different bank cannot be read
#: alongside them.  W5 has structurally zero DER reactive capability at both
#: levels and is excluded by :data:`DEAD_ZONE_STRATA`, so five windows are
#: measured -- which is what the chapter already says about Experiment B.
#:
#: ``design12`` -- the twelve ``tier1_design_set()`` search windows, the bank of
#: the 288-planned / 240-measured archive.  Kept so the new runs can be compared
#: against that archive's case manifest; it is a broader coverage check, not the
#: reported result.
BANKS = ("conditions", "design12")
DEFAULT_BANK = "conditions"


def bank_specs(bank: str):
    """Scenario specs of the named window bank.

    Importing inside the function keeps ``tuning`` off this module's import
    path, which matters because the worker processes import it too.
    """
    from tuning.scenarios_mc_v2 import condition_set, tier1_design_set
    if bank == "conditions":
        # ``hours`` is irrelevant here -- _run_case replaces duration_s with
        # SETTLE_S + OBSERVE_S -- but calling it is what registers W1-W6 in
        # WINDOW_META, which the stratum exclusion below reads.
        return condition_set()
    if bank == "design12":
        return tier1_design_set()
    raise SystemExit(f"unknown bank {bank!r}; known: {', '.join(BANKS)}")


# =====================================================================
#  Provenance
# =====================================================================

def _git(*args: str) -> Optional[str]:
    git_exe = shutil.which("git")
    if git_exe is None and os.name == "nt":
        candidates = (
            Path(os.environ.get("ProgramFiles", r"C:\\Program Files"))
            / "Git" / "cmd" / "git.exe",
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs" / "Git" / "cmd" / "git.exe",
        )
        git_exe = next((str(p) for p in candidates if p.is_file()), None)
    if git_exe is None:
        return None
    try:
        proc = subprocess.run((git_exe, *args), cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def provenance(args: argparse.Namespace, design: Dict[str, Any]) -> Dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "script": "experiments/ch_9_parameter_selection/ch_9_1_ninner_isolated_sts.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
            "der_q_override_pu": args.der_q_override_pu,
            "flat_mvar": args.flat_mvar,
            "max_step_mvar": args.max_step_mvar,
            "case_design": args.case_design,
        },
        "circularity": (
            "G_w was calibrated with N_inner = 9 ASSUMED; this measures "
            "N_inner with those weights in place. It is a check on the guess, "
            "not an independent measurement."),
    }


# =====================================================================
#  Measurement
# =====================================================================

def n_settle_final(q_seq: Sequence[float], *,
                   band_mvar: float = SETTLE_FINAL_BAND_MVAR,
                   guard_iter: int = SETTLE_FINAL_GUARD_ITER,
                   ) -> Tuple[int, bool]:
    """``(n_settle, censored)`` against the FINAL interface flow.

    ``q_seq[k]`` is the interface flow at subordinate iteration ``k`` after
    the step (index 0 = first dispatch after the step).  ``n_settle`` is the
    first ``k`` from which ``|q(j) - q_final| <= band_mvar`` for every
    ``j >= k``, with ``q_final = q_seq[-1]``.

    Settling to the final value rather than to the request: each STS
    controller trades interface tracking against its own voltages, so the
    flow settles at a compromise that may differ from ``q_set``.  The distance
    to the request is the residual, reported separately and never used here.

    Guard: if ``|q_final - q(end - guard_iter)| > band_mvar`` the final value
    is itself not settled, and the step is censored at ``len(q_seq)``.
    Reference implementation: handoff 2026-09-28, section 2.
    """
    q = [float(v) for v in q_seq]
    n_cap = len(q)
    if not q or not all(math.isfinite(v) for v in q):
        return n_cap, True
    if n_cap > guard_iter and abs(q[-1] - q[-1 - guard_iter]) > band_mvar:
        return n_cap, True
    bad = [k for k, v in enumerate(q) if abs(v - q[-1]) > band_mvar]
    return (0 if not bad else bad[-1] + 1), False


def measure_n_inner(records: Sequence[Any], *, step_time_s: float,
                    sts_period_s: float, band_mvar: float,
                    direction: str = "up", flat_mvar: float = DEFAULT_FLAT_MVAR,
                    ) -> List[Dict[str, Any]]:
    """``N_inner`` per interface transformer, counted from the step instant.

    The count is the first subordinate iteration at or after ``step_time_s`` at
    which ``|q_actual - q_set|`` enters the band and remains inside it for
    every later sample. "And stays" is the whole content of the definition: a
    trajectory that touches the band and leaves it again has not settled, and
    counting first entry would report the loop as faster than it is.

    Right-censored at the end of the observation window; censoring is returned
    as a flag, never as a dropped row.

    **Each step is also classified as capability-limited or not**, from the
    subordinate layer's OWN reported headroom in the commanded direction at the
    end of the window: ``cap_max - q`` going up, ``q - cap_min`` going down. If
    that is smaller than the settling band, the DSO is telling us it cannot
    move a further band-width that way, and a failure to settle says nothing
    about how fast the loop converges -- only that the setpoint was outside
    what the area can deliver.

    This replaces the earlier approach of forcing an artificial capability
    box: the plant keeps its real limits, and the two populations are
    separated afterwards. ``N_inner`` is then read over the unlimited steps --
    the genuinely converging series -- with the limited ones reported
    alongside rather than mixed in or silently dropped.
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

        def _first_stable(series: List[float], tol: float) -> Optional[int]:
            for i in range(len(series)):
                tail = series[i:]
                if tail and all(math.isfinite(x) and x <= tol for x in tail):
                    return i
            return None

        # THE convergence criterion: the per-iteration change in interface
        # flow goes flat and stays flat. Index 0 of `delta` is the change from
        # the first post-step sample to the second, so a converged-on-arrival
        # loop scores 0.
        q_seq = [(getattr(r, "dso_trafo_q_actual_mvar", {}) or {}).get(t)
                 for r in post]
        q_seq = [float(v) if v is not None else float("nan") for v in q_seq]
        delta = [abs(q_seq[i + 1] - q_seq[i]) for i in range(len(q_seq) - 1)]

        n_inner = _first_stable(delta, flat_mvar)
        censored = n_inner is None
        n_settle, settle_censored = n_settle_final(q_seq)
        by_flat = {}
        for tol in sorted(set(FLAT_SWEEP_MVAR) | {flat_mvar}):
            v = _first_stable(delta, tol)
            by_flat[f"n_inner_flat_{tol:g}"] = cap if v is None else v
            by_flat[f"unconverged_flat_{tol:g}"] = v is None

        # The TRACKING view, kept for comparison and clearly named. It is what
        # the earlier version reported as N_inner; it answers a different
        # question and must not be conflated with convergence.
        n_track = _first_stable(err, band_mvar)
        track_censored = n_track is None

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

        # Headroom the DSO itself reports, in the commanded direction, at the
        # end of the window.
        cmin = [(getattr(r, "dso_trafo_q_cap_min_mvar", {}) or {}).get(t)
                for r in post]
        cmax = [(getattr(r, "dso_trafo_q_cap_max_mvar", {}) or {}).get(t)
                for r in post]
        cmin = [float(v) for v in cmin if v is not None and math.isfinite(float(v))]
        cmax = [float(v) for v in cmax if v is not None and math.isfinite(float(v))]
        q_end = acts[-1] if acts else float("nan")
        if cmin and cmax and math.isfinite(q_end):
            headroom = ((cmax[-1] - q_end) if direction == "up"
                        else (q_end - cmin[-1]))
        else:
            headroom = float("nan")
        cap_limited = bool(math.isfinite(headroom) and headroom <= band_mvar)

        # The multi-objective trade-off, reported as its own quantity: how far
        # the converged optimum sits from the commanded setpoint. This is a
        # property of the controller's objective, NOT a convergence failure.
        q_set_end = sets[-1] if sets else float("nan")
        q_act_end = q_seq[-1] if q_seq else float("nan")
        offset = (abs(q_act_end - q_set_end)
                  if math.isfinite(q_act_end) and math.isfinite(q_set_end)
                  else float("nan"))
        last_delta = next((d for d in reversed(delta) if math.isfinite(d)),
                          float("nan"))

        out.append({
            **by_flat,
            "n_settle_final_1mvar": n_settle,
            "n_settle_censored": settle_censored,
            "n_inner_tracking": cap if track_censored else n_track,
            "tracking_censored": track_censored,
            "steady_state_offset_mvar": offset,
            "final_delta_mvar": last_delta,
            "headroom_end_mvar": headroom,
            "capability_limited": cap_limited,
            "direction_measured": direction,
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


def cap_target_step(q_now: float, q_target: float,
                    max_step_mvar: Optional[float]) -> float:
    """Return ``q_target`` with ``|q_target - q_now|`` optionally capped.

    The reported capability still defines the admissible direction and the raw
    target.  The cap changes only the exogenous supervisory-size excitation
    used by the measurement pass; it does not change plant or controller
    limits.  ``None`` preserves the historical capability-edge experiment.
    """
    if max_step_mvar is None:
        return float(q_target)
    cap = float(max_step_mvar)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("max_step_mvar must be finite and > 0")
    delta = float(q_target) - float(q_now)
    if abs(delta) <= cap:
        return float(q_target)
    return float(q_now) + math.copysign(cap, delta)


def extract_interface_trajectory(records: Sequence[Any], *, trafo: str,
                                 group: str, step_time_s: float,
                                 sts_period_s: float) -> List[Dict[str, Any]]:
    # Post-step signed trajectory for diagnosing drift versus oscillation.
    out: List[Dict[str, Any]] = []
    previous: Optional[float] = None
    for record in records:
        time_s = float(getattr(record, "time_s", float("nan")))
        if not math.isfinite(time_s) or time_s < step_time_s:
            continue
        q_set = (getattr(record, "dso_trafo_q_set_mvar", {}) or {}).get(trafo)
        q_actual = (
            getattr(record, "dso_trafo_q_actual_mvar", {}) or {}).get(trafo)
        if q_set is None or q_actual is None:
            continue
        q_set = float(q_set)
        q_actual = float(q_actual)
        cap_min = (
            getattr(record, "dso_trafo_q_cap_min_mvar", {}) or {}).get(trafo)
        cap_max = (
            getattr(record, "dso_trafo_q_cap_max_mvar", {}) or {}).get(trafo)
        tap = (getattr(record, "dso_trafo_tap_pos", {}) or {}).get(trafo)
        v_min = (
            getattr(record, "dso_group_v_min_pu", {}) or {}).get(group)
        v_max = (
            getattr(record, "dso_group_v_max_pu", {}) or {}).get(group)
        slack = (getattr(record, "dso_z_slack_max", {}) or {}).get(group)
        out.append({
            "time_s": time_s,
            "inner_iteration": int(round(
                (time_s - step_time_s) / sts_period_s)),
            "q_set_mvar": q_set,
            "q_actual_mvar": q_actual,
            "q_actual_delta_mvar": (
                float("nan") if previous is None else q_actual - previous),
            "abs_tracking_error_mvar": abs(q_actual - q_set),
            "cap_min_mvar": (
                float(cap_min) if cap_min is not None else float("nan")),
            "cap_max_mvar": (
                float(cap_max) if cap_max is not None else float("nan")),
            "tap_pos": int(tap) if tap is not None else None,
            "v_min_pu": float(v_min) if v_min is not None else float("nan"),
            "v_max_pu": float(v_max) if v_max is not None else float("nan"),
            "z_slack_max": (
                float(slack) if slack is not None else float("nan")),
        })
        previous = q_actual
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

    from _ch9_selected_design import SELECTED_PARAMSET, build_selected_config
    from tuning.metrics import MetricScales
    from tuning.runner import run_scenario_with_records as _run_scenario

    # The controller identity is carried in the JOB, not defaulted here: the
    # initial-weight (former Experiment C, dropped) and final-weight
    # (Experiment C since the 2026-09-29 letter remap, was E) legs run
    # the same request bank and must be distinguishable from the run record
    # alone.  Before 2026-09-15 this called build_selected_config() with no
    # arguments and the worker silently ran whatever the module constant was.
    cfg, _prov = build_selected_config(
        paramset=job.get("paramset", SELECTED_PARAMSET),
        apply_frozen_network=bool(job.get("frozen_network", True)))
    if job.get("weights"):
        # Complete weight vector on top of the paramset, exactly as the
        # Stage-II screen builds its candidates (apply_to_config raises on a
        # partial vector).  Used by the D1 settling screen.
        from tuning.parameters import apply_to_config
        cfg = apply_to_config(cfg, dict(job["weights"]))
    if job.get("cfg_overrides"):
        # Existing MultiTSOConfig fields only (e.g. dso_g_q_per_area for a
        # per-area DIAGNOSTIC); replace() raises on an unknown field.
        cfg = dataclasses.replace(cfg, **dict(job["cfg_overrides"]))
    if job.get("no_measurement_noise"):
        # dataclasses.replace on the nested config, not in-place: the cfg is
        # shared by the probe and measurement passes below.
        cfg = dataclasses.replace(
            cfg, measurement_noise=dataclasses.replace(
                cfg.measurement_noise, enabled=False))
    spec = next(s for s in bank_specs(job.get("bank", DEFAULT_BANK))
                if s.name == job["window"])

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
    # The probe pass ALWAYS runs at the real capability: the step magnitude
    # must be the one the plant actually declares, or the comparison against
    # the capped run is not a comparison of the same physical ask.
    override = job.get("der_q_override_pu")
    measure_cfg_extra = ({} if override is None
                         else {"der_q_capability_override_pu": float(override)})
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
    # Order must match HVNetworkInfo.coupling_trafo_indices. The record keys
    # contain the global integer transformer index, so sort by that integer.
    # Lexical ordering breaks DSO_4 as trafo_10, trafo_11, trafo_9 and silently
    # applies the step to a sibling interface.
    mine_sorted = _ordered_trafo_keys(mine)

    # ONE interface is stepped; its siblings are held at their current flow.
    #
    # Stepping every transformer of a DSO to its own band edge at once is NOT
    # an admissible traversal: the reported capability is per interface, but
    # the DER reactive power backing it is shared across the sub-network, so
    # the per-interface bands are individually feasible and jointly are not.
    # Measured 2026-08-19: driving all three interfaces of DSO_4 to their edges
    # left trafo_9 87-101 Mvar from a 37-44 Mvar setpoint while its siblings
    # tracked, i.e. the subordinate MIQP allocated the shared capability and
    # starved one interface. Every such case came back censored, and the
    # resulting "N_inner" measured the infeasibility, not the loop.
    #
    # This is the same defect class as the known voltage-blind over-reporting
    # of the capability message; here it shows up as non-additivity across the
    # interfaces of one DSO.
    focus = job.get("trafo")
    if focus is not None and focus not in targets:
        return {**job, "rows": [], "targets": targets,
                "wall_s": res_p.wall_time_s,
                "failure": f"no usable capability band for {focus}"}
    stepped = [focus] if focus is not None else mine_sorted

    applied_targets = {
        t: cap_target_step(targets[t]["q_now"], targets[t][key],
                           job.get("max_step_mvar"))
        for t in stepped
    }

    degenerate = [t for t in stepped
                  if abs(applied_targets[t] - targets[t]["q_now"])
                  < float(job["band_mvar"])]
    if degenerate:
        return {**job, "rows": [], "targets": targets,
                "wall_s": res_p.wall_time_s,
                "failure": (f"reported capability gives a step below the "
                            f"settling band for {degenerate}: no admissible "
                            f"traversal at this operating point")}
    q_hold = [targets[t]["q_now"] for t in mine_sorted]
    q_step = [applied_targets[t] if t in stepped else targets[t]["q_now"]
              for t in mine_sorted]

    cfg_run = dataclasses.replace(
        base,
        q_pcc_setpoint_schedule_per_dso={
            job["dso"]: [{"t_s": 0.0, "q_mvar": q_hold},
                         {"t_s": SETTLE_S, "q_mvar": q_step}]},
        **measure_cfg_extra)

    res, records = _run_scenario(spec, cfg_run, MetricScales())
    if res.failure_reason:
        return {**job, "failure": res.failure_reason, "rows": [],
                "targets": targets, "wall_s": res.wall_time_s}

    rows = measure_n_inner(records, step_time_s=SETTLE_S,
                           sts_period_s=STS_PERIOD_S,
                           band_mvar=float(job["band_mvar"]),
                           direction=str(job["direction"]),
                           flat_mvar=float(job.get("flat_mvar",
                                                   DEFAULT_FLAT_MVAR)))
    rows = [r for r in rows if r["group"] == job["dso"]]

    # Guard the experimental intervention itself. A valid solver run is not a
    # valid step-response experiment unless the recorded interface setpoints
    # equal the vector intended for delivery. This catches mapping regressions
    # before they enter aggregate dissertation output.
    expected_sets = {
        t: applied_targets[t] if t in stepped else targets[t]["q_now"]
        for t in mine_sorted
    }
    delivered = {r["trafo"]: r["q_set_after_mvar"] for r in rows}
    delivery_errors = {
        t: delivered.get(t, float("nan")) - expected
        for t, expected in expected_sets.items()
    }
    bad_delivery = {
        t: err for t, err in delivery_errors.items()
        if not math.isfinite(err) or abs(err) > 1e-6
    }
    if bad_delivery:
        return {**job, "rows": [], "targets": targets,
                "wall_s": res.wall_time_s + res_p.wall_time_s,
                "failure": ("setpoint delivery mismatch: intended transformer "
                            f"intervention was not realised: {bad_delivery}")}

    trajectory: List[Dict[str, Any]] = []
    if job.get("save_trajectories") and focus is not None:
        trajectory = extract_interface_trajectory(
            records, trafo=focus, group=groups.get(focus, job["dso"]),
            step_time_s=SETTLE_S, sts_period_s=STS_PERIOD_S)
        for point in trajectory:
            point.update({
                "window": job["window"], "dso": job["dso"],
                "trafo": focus, "direction": job["direction"],
            })

    rows = [r for r in rows if focus is None or r["trafo"] == focus]
    for r in rows:
        raw_step = targets[r["trafo"]][key] - targets[r["trafo"]]["q_now"]
        applied_step = (applied_targets[r["trafo"]]
                        - targets[r["trafo"]]["q_now"])
        r.update({"window": job["window"], "dso": job["dso"],
                  "direction": job["direction"], "variant": job["variant"],
                  "step_mvar": applied_step,
                  "raw_step_mvar": raw_step,
                  "step_cap_mvar": job.get("max_step_mvar"),
                  "step_capped": abs(applied_step) + 1e-12 < abs(raw_step),
                  "command_delivery_error_mvar": delivery_errors[r["trafo"]],
                  "band_width_mvar": targets[r["trafo"]]["width"]})
    return {**job, "failure": "", "rows": rows, "targets": targets,
            "trajectory": trajectory,
            "wall_s": res.wall_time_s + res_p.wall_time_s}


def _ordered_trafo_keys(keys: Sequence[str]) -> List[str]:
    # Return record transformer keys in numeric global-index order.
    def index(key: str) -> int:
        try:
            return int(str(key).rsplit("_", 1)[-1])
        except ValueError as exc:
            raise ValueError(
                f"transformer key has no numeric suffix: {key!r}") from exc

    return sorted(keys, key=lambda key: (index(key), str(key)))


def _half_fraction_direction(window_index: int, dso_index: int,
                             trafo_local_index: int) -> str:
    # Balanced direction for a resolution-one-half experiment grid.
    return ("up" if (window_index + dso_index + trafo_local_index) % 2 == 0
            else "down")


def _trafo_indices(dso: str, n_per_dso: int) -> List[int]:
    """Global interface-transformer indices belonging to ``dso``.

    The record keys are ``DSO_<n>|trafo_<i>`` with ``i`` running across ALL
    interfaces of the plant, not restarting per DSO: DSO_1 owns 0-2, DSO_2
    owns 3-5, DSO_3 6-8 and DSO_4 9-11 in the four-sub-network benchmark.
    A job naming ``DSO_4|trafo_0`` would simply never match, and the case
    would be reported as "no usable capability band" for a transformer that
    does not exist -- so the mapping is computed rather than assumed.
    """
    try:
        k = int(str(dso).split("_")[-1]) - 1
    except ValueError:
        return list(range(n_per_dso))
    return [k * n_per_dso + i for i in range(n_per_dso)]


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
        # The population that actually answers eq. (9.2): steps the area could
        # deliver. A capability-limited step that never settles measures the
        # limit, not the loop.
        free = [r for r in sub if not r.get("capability_limited")]
        nf = [float(r["n_inner"]) for r in free]
        free_unc = [float(r["n_inner"]) for r in free if not r["censored"]]
        out.append({
            "dso": dso, "direction": direction, "n_steps": len(sub),
            "n_capability_limited": sum(1 for r in sub
                                        if r.get("capability_limited")),
            "n_free": len(free),
            "free_n_inner_median": _quantile(nf, 0.50),
            "free_n_inner_p95": _quantile(nf, 0.95),
            "free_censoring_fraction": (
                sum(1 for r in free if r["censored"]) / len(free)
                if free else float("nan")),
            "free_uncensored_median": _quantile(free_unc, 0.50),
            "free_uncensored_p95": _quantile(free_unc, 0.95),
            "headroom_end_median_mvar": _quantile(
                [r.get("headroom_end_mvar", float("nan")) for r in sub], 0.50),
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


def aggregate_settle(rows: Sequence[Dict[str, Any]],
                     budget: int = N_INNER_BUDGET) -> List[Dict[str, Any]]:
    """``n_settle_final_1mvar`` distributions per DSO (both directions) and pooled.

    Also the residual ``|q_final - q_set|``, which is reported and never used
    as a settling limit (handoff 2026-09-28, section 2).
    """
    out: List[Dict[str, Any]] = []
    dsos = sorted({r["dso"] for r in rows})
    for dso in dsos + ["__pooled__"]:
        sub = (list(rows) if dso == "__pooled__"
               else [r for r in rows if r["dso"] == dso])
        n = [float(r["n_settle_final_1mvar"]) for r in sub
             if "n_settle_final_1mvar" in r]
        if not n:
            continue
        res = [float(r["residual_mvar"]) for r in sub]
        out.append({
            "dso": dso, "n_steps": len(n),
            "n_settle_median": _quantile(n, 0.50),
            "n_settle_p90": _quantile(n, 0.90),
            "n_settle_p95": _quantile(n, 0.95),
            "n_settle_max": max(n),
            f"share_le_{budget}": sum(1 for v in n if v <= budget) / len(n),
            "n_settle_censored": sum(1 for r in sub
                                     if r.get("n_settle_censored")),
            "residual_median_mvar": _quantile(res, 0.50),
            "residual_p95_mvar": _quantile(res, 0.95),
            "residual_max_mvar": max((x for x in res if math.isfinite(x)),
                                     default=float("nan")),
            "n_residual_gt_1mvar": sum(1 for x in res
                                       if math.isfinite(x) and x > 1.0),
            "v_violations": sum(1 for r in sub if r.get("v_violation")),
            "tap_moves_total": sum(int(r.get("tap_moves", 0)) for r in sub),
        })
    return out


_ROW_COLS = ["window", "dso", "direction", "variant", "trafo", "group",
             "n_settle_final_1mvar", "n_settle_censored",
             "n_inner", "censored", "n_inner_tracking", "tracking_censored",
             "steady_state_offset_mvar", "final_delta_mvar",
             "capability_limited", "headroom_end_mvar",
             "n_inner_cap", "step_mvar", "raw_step_mvar",
             "step_cap_mvar", "step_capped",
             "command_delivery_error_mvar", "case_wall_s",
             "band_width_mvar", "q_set_after_mvar", "q_actual_after_mvar",
             "residual_mvar", "tap_moves", "v_min_pu", "v_max_pu",
             "v_violation", "z_slack_max"]


_TRACE_COLS = [
    "window", "dso", "trafo", "direction", "time_s", "inner_iteration",
    "q_set_mvar", "q_actual_mvar", "q_actual_delta_mvar",
    "abs_tracking_error_mvar", "cap_min_mvar", "cap_max_mvar", "tap_pos",
    "v_min_pu", "v_max_pu", "z_slack_max", "case_wall_s",
]


def write_trajectories(out_dir: Path,
                       trajectories: Sequence[Dict[str, Any]]) -> None:
    with (out_dir / "trajectories.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_TRACE_COLS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trajectories)


def write_outputs(out_dir: Path, rows: List[Dict[str, Any]],
                  agg: List[Dict[str, Any]], failures: List[Dict[str, Any]],
                  meta: Optional[Dict[str, Any]] = None) -> None:
    extra = sorted({k for r in rows for k in r
                    if k.startswith(("n_inner_flat_", "unconverged_flat_"))})
    with (out_dir / "steps.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_ROW_COLS + extra,
                           extrasaction="ignore")
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

    settle = aggregate_settle(rows)
    if settle:
        with (out_dir / "n_settle_summary.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(settle[0].keys()))
            w.writeheader()
            for r in settle:
                w.writerow(r)

    ov = (meta or {}).get("constants", {}).get("der_q_override_pu")
    md = ["# N_inner on the isolated subordinate loop (thesis eq. 9.2)", ""]
    if ov is not None:
        md += [f"> **DIAGNOSTIC RUN — DER reactive capability forced to "
               f"`[-{ov:g}*S_n, +{ov:g}*S_n]`.** This is not a physical model: "
               f"`+-{ov:g}` pu at rated P implies an apparent-power rating of "
               f"`sqrt(1 + {ov:g}^2)` pu, beyond what the scenario declares. "
               f"Numbers from this run must never be quoted as measured under "
               f"real limits. Step magnitudes are still those of the REAL "
               f"capability — only the measurement pass is relaxed — so this "
               f"isolates settling speed from capability saturation.", ""]
    md += [
          "`N_inner` is the first subordinate iteration after an interface-Q "
          "setpoint step from which the per-iteration flow change "
          "`|q_pcc(k) - q_pcc(k-1)|` stays within the configured flatness "
          "tolerance. Right-censored at the end of the observation window; "
          "censoring is reported, never dropped.",
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
          f"### Settling to the final value (`n_settle_final_1mvar`)", "",
          f"First iteration `k` after the step (iteration 0 = first "
          f"subordinate dispatch after the step) from which the interface "
          f"flow stays within {SETTLE_FINAL_BAND_MVAR:g} Mvar of its FINAL "
          f"value (value at the end of the observation). If the flow moves "
          f"more than {SETTLE_FINAL_BAND_MVAR:g} Mvar within the last "
          f"{SETTLE_FINAL_GUARD_ITER} iterations the final value is not "
          f"settled and the step is censored at the observation length. The "
          f"residual `|q_final - q_set|` is reported, never used as a limit. "
          f"Not a flatness count.", "",
          f"| DSO | steps | median | p90 | p95 | max | share <= "
          f"{N_INNER_BUDGET} | censored | residual med [Mvar] | residual p95 | "
          f"residual max | residual > 1 Mvar |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for s in settle:
        md.append(
            f"| {s['dso']} | {s['n_steps']} | {s['n_settle_median']:.1f} | "
            f"{s['n_settle_p90']:.1f} | {s['n_settle_p95']:.1f} | "
            f"{s['n_settle_max']:.0f} | "
            f"{100 * s[f'share_le_{N_INNER_BUDGET}']:.0f} % | "
            f"{s['n_settle_censored']} | {s['residual_median_mvar']:.2f} | "
            f"{s['residual_p95_mvar']:.2f} | {s['residual_max_mvar']:.2f} | "
            f"{s['n_residual_gt_1mvar']} |")
    md += ["",
          "### Capability-limited steps are separated, not mixed in", "",
          "A step is **capability-limited** when the subordinate layer's own "
          "reported headroom in the commanded direction, at the end of the "
          "window, is below the settling band -- it is saying it cannot move "
          "a further band-width that way. Such a step failing to settle "
          "measures the limit, not the loop. `N_inner` is read over the "
          "**free** steps.", "",
          "| DSO | direction | steps | cap-limited | free | free N_inner med | "
          "free p95 | free censored | headroom med [Mvar] | step med [Mvar] |",
          "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for a in agg:
        md.append(
            f"| {a['dso']} | {a['direction']} | {a['n_steps']} | "
            f"{a['n_capability_limited']} | {a['n_free']} | "
            f"{a['free_n_inner_median']:.1f} | {a['free_n_inner_p95']:.1f} | "
            f"{a['free_censoring_fraction']:.2f} | "
            f"{a['headroom_end_median_mvar']:.2f} | "
            f"{a['step_median_mvar']:.1f} |")

    md += ["", "### All steps, including capability-limited ones", "",
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
               f"- script SHA-256: `{meta.get('script_sha256')}`",
               f"- commit: `{meta['git_commit']}` on `{meta['git_branch']}`"
               + ("  **(working tree dirty -- not reproducible from the "
                  "commit alone)**" if meta.get("git_dirty") else ""),
               f"- parent-silent variant: **{c.get('variant')}** "
               f"(`frozen_ofo` = supervisory OFO solves once at t=0 and holds; "
               f"`local` = local Q(V) supervisory control, a DIFFERENT "
               f"baseline controller and a cross-check only)",
               f"- weights: paramset `{d.get('paramset')}` candidate "
               f"`{d.get('candidate_key')}` from `{d.get('campaign')}`",
               f"- network: "
               + ("**doubled DSO_3** (`dso_der_scale`/`dso_load_p_scale` "
                  "x2), the declared Chapter-9 batch network. The Stage-I/II "
                  "weight campaign ran SYMMETRIC, so this run does not share "
                  "its network."
                  if d.get("frozen_network") else
                  "symmetric, as the Stage-I/II weight campaign used"),
               f"- settle {c.get('settle_s')} s, observe {c.get('observe_s')} s, "
               f"tracking band {c.get('band_mvar')} Mvar, flatness tolerance "
               f"{c.get('flat_mvar')} Mvar, raw target "
               f"{c.get('band_fraction')} of the reported band"
               + (f", applied step capped at {c.get('max_step_mvar')} Mvar"
                  if c.get("max_step_mvar") is not None else ""),
               f"- `T_STS` = {STS_PERIOD_S:.0f} s, bank "
               f"`{(meta.get('args') or {}).get('bank', DEFAULT_BANK)}` on "
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

    # CONVERGENCE = the interface flow goes flat, NOT that it reaches q_set.
    # q rises 10 -> 30 -> 45 -> 49 -> 50 -> 50 -> 50, so the per-iteration
    # change is 20, 15, 4, 1, 0, 0: flat from index 4.
    recs = ([_R(float(20 * i), 0.0, 0.0) for i in range(5)]
            + [_R(100.0, 50.0, 10.0), _R(120.0, 50.0, 30.0),
               _R(140.0, 50.0, 45.0), _R(160.0, 50.0, 49.0),
               _R(180.0, 50.0, 50.0), _R(200.0, 50.0, 50.0),
               _R(220.0, 50.0, 50.0)])
    rows = measure_n_inner(recs, step_time_s=100.0, sts_period_s=20.0,
                           band_mvar=1.0, flat_mvar=0.1)
    check("one row per interface transformer", len(rows) == 1)
    check("N_inner is the first iteration from which |dq| stays flat",
          rows[0]["n_inner"] == 4 and not rows[0]["censored"],
          f"got {rows[0]['n_inner']}")

    # THE case the tracking criterion got wrong: the loop converges to its
    # multi-objective optimum 25 Mvar short of the setpoint. It HAS converged.
    recs_off = ([_R(0.0, 0.0, 0.0)]
                + [_R(100.0, 50.0, 10.0), _R(120.0, 50.0, 20.0),
                   _R(140.0, 50.0, 25.0), _R(160.0, 50.0, 25.0),
                   _R(180.0, 50.0, 25.0)])
    ro = measure_n_inner(recs_off, step_time_s=100.0, sts_period_s=20.0,
                         band_mvar=1.0, flat_mvar=0.1)[0]
    check("a loop that goes flat short of q_set HAS converged",
          not ro["censored"] and ro["n_inner"] == 2,
          f"got n_inner={ro['n_inner']} censored={ro['censored']}")
    check("its distance from q_set is reported as a steady-state offset",
          abs(ro["steady_state_offset_mvar"] - 25.0) < 1e-9,
          f"got {ro['steady_state_offset_mvar']}")
    check("the TRACKING view still calls that unsettled, and is kept apart",
          ro["tracking_censored"], f"got {ro['n_inner_tracking']}")

    # SETTLING to the final value (handoff 2026-09-28): the slow creep that
    # flatness cannot see.  0.5 Mvar per iteration is "flat" at 1 Mvar but the
    # flow is still travelling towards a final value 10 Mvar away.
    creep = [0.5 * min(k, 20) for k in range(46)]
    n_s, c_s = n_settle_final(creep)
    check("n_settle counts from the final value, not from flatness",
          n_s == 18 and not c_s, f"got {n_s}, censored={c_s}")
    n_s, c_s = n_settle_final([5.0] * 46)
    check("a flow already at its final value settles at 0",
          n_s == 0 and not c_s, f"got {n_s}")
    drift = [0.2 * k for k in range(46)]          # 2 Mvar over the last 10
    n_s, c_s = n_settle_final(drift)
    check("a final value still moving > 1 Mvar in 10 iterations is censored",
          c_s and n_s == 46, f"got {n_s}, censored={c_s}")
    n_s, c_s = n_settle_final([0.0, 10.0, 3.0] + [3.5] * 43)
    check("an excursion outside the band delays settling past it",
          n_s == 2 and not c_s, f"got {n_s}")
    check("the row carries n_settle_final_1mvar",
          "n_settle_final_1mvar" in rows[0]
          and rows[0]["n_settle_final_1mvar"] == 3,
          f"got {rows[0].get('n_settle_final_1mvar')}")

    # A loop that keeps moving has NOT converged, however close it is.
    recs_osc = ([_R(0.0, 0.0, 0.0)]
                + [_R(float(100 + 20 * i), 50.0, 40.0 + (10.0 if i % 2 else 0.0))
                   for i in range(6)])
    rc = measure_n_inner(recs_osc, step_time_s=100.0, sts_period_s=20.0,
                         band_mvar=1.0, flat_mvar=0.1)
    check("a loop still oscillating is NOT converged",
          rc[0]["censored"] and rc[0]["n_inner"] == rc[0]["n_inner_cap"],
          f"got {rc[0]['n_inner']}")

    # Flatness tolerance is reported across a sweep, not silently decisive.
    trace = extract_interface_trajectory(
        recs, trafo="T1", group="DSO_1", step_time_s=100.0,
        sts_period_s=20.0)
    check("trajectory export retains signed per-iteration interface motion",
          len(trace) == 7
          and math.isnan(trace[0]["q_actual_delta_mvar"])
          and trace[1]["q_actual_delta_mvar"] == 20.0
          and trace[-1]["q_actual_delta_mvar"] == 0.0)

    check("a coarser flatness tolerance converges no later than a finer one",
          rc[0]["n_inner_flat_1"] <= rc[0]["n_inner_flat_0.02"],
          f"got {rc[0]['n_inner_flat_1']} vs {rc[0]['n_inner_flat_0.02']}")

    # Band-edge targets, read at the step instant.
    tg = band_edge_targets(recs, at_time_s=100.0, fraction=1.0)
    check("band edge target up == reported cap_max",
          abs(tg["T1"]["target_up"] - 100.0) < 1e-9, f"got {tg['T1']}")
    check("band edge target down == reported cap_min",
          abs(tg["T1"]["target_down"] + 100.0) < 1e-9, f"got {tg['T1']}")
    tg95 = band_edge_targets(recs, at_time_s=100.0, fraction=0.95)
    check("a fraction below 1 backs off from the edge",
          tg95["T1"]["target_up"] < tg["T1"]["target_up"])
    check("a supervisory-size cap limits a large upward step",
          abs(cap_target_step(10.0, 80.0, 20.0) - 30.0) < 1e-9)
    check("a supervisory-size cap limits a large downward step",
          abs(cap_target_step(10.0, -80.0, 20.0) + 10.0) < 1e-9)
    check("a target below the cap is unchanged",
          abs(cap_target_step(10.0, 18.0, 20.0) - 18.0) < 1e-9)
    try:
        cap_target_step(0.0, 1.0, 0.0)
    except ValueError:
        check("a non-positive supervisory-size cap is rejected", True)
    else:
        check("a non-positive supervisory-size cap is rejected", False)

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

    # Capability classification: same trajectory, different reported headroom.
    lim = ([_R(0.0, 0.0, 0.0)]
           + [_R(float(100 + 20 * i), 50.0, 40.0, cap=(-100.0, 40.5))
              for i in range(4)])
    lr = measure_n_inner(lim, step_time_s=100.0, sts_period_s=20.0,
                         band_mvar=1.0, direction="up")[0]
    check("a step with no reported headroom left is capability-limited",
          lr["capability_limited"], f"headroom={lr['headroom_end_mvar']}")
    free = ([_R(0.0, 0.0, 0.0)]
            + [_R(float(100 + 20 * i), 50.0, 40.0, cap=(-100.0, 200.0))
               for i in range(4)])
    fr = measure_n_inner(free, step_time_s=100.0, sts_period_s=20.0,
                         band_mvar=1.0, direction="up")[0]
    check("the same trajectory with headroom left is NOT capability-limited",
          not fr["capability_limited"], f"headroom={fr['headroom_end_mvar']}")
    check("the flag is independent of whether the loop converged",
          lr["censored"] == fr["censored"])

    check("interface indices do not restart per DSO",
          _trafo_indices("DSO_1", 3) == [0, 1, 2]
          and _trafo_indices("DSO_4", 3) == [9, 10, 11],
          f"got {_trafo_indices('DSO_4', 3)}")
    check("two-digit transformer keys retain numeric controller order",
          _ordered_trafo_keys(["DSO_4|trafo_10", "DSO_4|trafo_11",
                               "DSO_4|trafo_9"])
          == ["DSO_4|trafo_9", "DSO_4|trafo_10", "DSO_4|trafo_11"])
    half_dirs = [
        _half_fraction_direction(w, d, t)
        for w in range(10) for d in range(4) for t in range(3)
    ]
    check("balanced-half design has 120 cases split 60 up / 60 down",
          len(half_dirs) == 120
          and half_dirs.count("up") == half_dirs.count("down") == 60)
    check("each DSO-interface receives five steps in each direction",
          all([
              _half_fraction_direction(w, d, t)
              for w in range(10)
          ].count("up") == 5
              for d in range(4) for t in range(3)))

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
        steps_header = (out / "steps.csv").read_text(
            encoding="utf-8").splitlines()[0]
        check("steps.csv retains the delivery-error audit column",
              "command_delivery_error_mvar" in steps_header)
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
    ap.add_argument("--paramset", default=SELECTED_PARAMSET,
                    help=f"controller weights, from configs/paramsets/. "
                         f"{INITIAL_PARAMSET!r} is the Stage-I ANALYTIC design "
                         f"point (Experiment C, initial weights); "
                         f"{SELECTED_PARAMSET!r} is the Stage-II selection "
                         f"(Experiment C, final weights; letter was E "
                         f"before 2026-09-29). Recorded in "
                         f"run_meta.json and resolved_config.json either way.")
    ap.add_argument("--bank", choices=BANKS, default=DEFAULT_BANK,
                    help="window bank. 'conditions' = W1-W6 of the thesis "
                         "conditions table, which is also what F1/F2 use "
                         "(W5 excluded as zero-capability, so 5 windows "
                         "measured). 'design12' = the twelve tier1_design_set "
                         "search windows of the 240-measured archive.")
    ap.add_argument("--no-measurement-noise", action="store_true",
                    help="disable the measurement-chain error model. A "
                         "DIAGNOSTIC, not a run of record: it isolates how "
                         "much of the per-iteration flow change at "
                         "convergence is the loop and how much is the "
                         "instrument. The flatness tolerance that defines "
                         "N_inner cannot be set below that floor, so the "
                         "floor has to be measured rather than assumed.")
    ap.add_argument("--symmetric-network", action="store_true",
                    help="drop the doubled-DSO_3 multipliers. The Chapter-9 "
                         "batch is declared on the DOUBLED network and that is "
                         "the default; the Stage-II search that produced "
                         "ch9_selected ran symmetric, so this flag is what "
                         "makes a run comparable with the selection bank.")
    ap.add_argument("--weights-json", type=Path, default=None,
                    help="JSON object with a COMPLETE weight vector applied on "
                         "top of --paramset via tuning.parameters."
                         "apply_to_config (as the Stage-II screen does). The "
                         "fingerprint is recorded as the candidate key.")
    ap.add_argument("--label", default="ninner_isolated")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "THESIS_ch9_1_ninner")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--band-mvar", type=float, default=DEFAULT_BAND_MVAR)
    ap.add_argument("--band-fraction", type=float,
                    default=DEFAULT_BAND_FRACTION,
                    help="fraction of the reported capability band traversed")
    ap.add_argument("--max-step-mvar", type=float, default=None,
                    help="maximum absolute per-interface Q-setpoint step [Mvar] "
                         "after constructing the capability-band target; "
                         "default None preserves the capability-edge run")
    ap.add_argument("--case-design", choices=("full", "balanced_half"),
                    default="full",
                    help="full runs both directions at every window/DSO/"
                         "interface; balanced_half assigns one direction per "
                         "triple while balancing every interface across windows")
    ap.add_argument("--save-trajectories", action="store_true",
                    help="persist post-step signed interface trajectories for "
                         "diagnosing drift versus oscillation; default off")
    ap.add_argument("--variant", choices=("frozen_ofo", "local"),
                    default="frozen_ofo",
                    help="parent-silent variant; 'local' swaps in a DIFFERENT "
                         "supervisory controller and is a cross-check only")
    ap.add_argument("--windows", default="")
    ap.add_argument("--dsos", default="",
                    help="comma-separated DSO ids; default every DSO present")
    ap.add_argument("--focus-trafo", default="",
                    help="optional exact record key, e.g. DSO_4|trafo_10; "
                         "restricts the grid for targeted diagnostics")
    ap.add_argument("--direction", choices=("both", "up", "down"),
                    default="both",
                    help="optional directional restriction; default both")
    ap.add_argument("--flat-mvar", type=float, default=DEFAULT_FLAT_MVAR,
                    help="convergence tolerance on the per-iteration CHANGE "
                         "in interface flow [Mvar]; the loop has converged "
                         "when |q(k)-q(k-1)| stays at or below this")
    ap.add_argument("--der-q-override-pu", type=float, default=None,
                    help="force every DER Q capability to [-x*S_n, +x*S_n] "
                         "for the MEASUREMENT pass only, to observe settling "
                         "rather than capability saturation. Step magnitudes "
                         "still come from the real capability. DIAGNOSTIC "
                         "ONLY -- not a physical model.")
    ap.add_argument("--trafos-per-dso", type=int, default=3,
                    help="interface transformers per DSO in this benchmark")
    ap.add_argument("--include-dead-zone", action="store_true",
                    help="also run windows whose stratum has structurally "
                         "zero DER reactive capability; they have no "
                         "admissible band traversal and will be refused")
    ap.add_argument("--probe", action="store_true",
                    help="one window, one DSO, both directions -- to confirm "
                         "the band is traversable before the full grid")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if (a.max_step_mvar is not None
            and (not math.isfinite(a.max_step_mvar)
                 or a.max_step_mvar <= 0.0)):
        ap.error("--max-step-mvar must be finite and > 0")

    if a.self_test:
        return self_test()

    from _ch9_selected_design import build_selected_config, resolved_manifest
    from tuning.scenarios_mc_v2 import WINDOW_META

    cfg, design = build_selected_config(
        paramset=a.paramset,
        apply_frozen_network=not a.symmetric_network)
    weights_override: Optional[Dict[str, float]] = None
    if a.weights_json is not None:
        from tuning.parameters import apply_to_config, weight_fingerprint
        weights_override = {
            k: float(v) for k, v in
            json.loads(a.weights_json.read_text(encoding="utf-8")).items()}
        cfg = apply_to_config(cfg, weights_override)
        design["weights_override"] = weights_override
        design["candidate_key"] = weight_fingerprint(weights_override)
    print(f"[design] paramset {design['paramset']} candidate "
          f"{design['candidate_key']} from {design['campaign']}")
    print(f"[design] base {design['base_config']}; "
          f"shunts={cfg.install_tso_tertiary_shunts}/{cfg.shunt_dispatch} "
          f"boundary={cfg.tie_boundary_equivalent} "
          f"network={'DSO_3 x2' if design['frozen_network'] else 'symmetric'}")
    if design["frozen_network"]:
        print("[design]   -- the Stage-I/II weight campaign ran SYMMETRIC; "
              "this run does not share its network.")

    specs = bank_specs(a.bank)
    print(f"[ninner] bank {a.bank}: {len(specs)} window(s)")
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
        a.trafos_per_dso = 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = a.out / a.label / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (a.out / a.label / "_latest.txt").write_text(stamp, encoding="utf-8")

    meta = provenance(a, design)
    meta["excluded_windows"] = excluded if not a.include_dead_zone else []
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")
    # The COMPLETE resolved configuration, not the name of the helper that
    # built it: a run record that names a factory cannot be re-resolved once
    # the factory changes, which is how the 2026-08 batch lost the ability to
    # say what network and actuator set it had actually simulated.
    (out_dir / "resolved_config.json").write_text(
        json.dumps(resolved_manifest(cfg, design), indent=2, default=str),
        encoding="utf-8")
    shutil.copy2(Path(__file__).resolve(), out_dir / "script_snapshot.py")

    # One interface is stepped in each job. The full design pairs both
    # directions at every operating point. The balanced half-fraction preserves
    # every window, DSO and interface but assigns one direction per triple.
    n_trafo = int(a.trafos_per_dso)
    jobs: List[Dict[str, Any]] = []
    for w_idx, w in enumerate(names):
        for d_idx, d in enumerate(dsos):
            for local_idx, i in enumerate(_trafo_indices(d, n_trafo)):
                directions = (
                    ("up", "down") if a.case_design == "full"
                    else (_half_fraction_direction(w_idx, d_idx, local_idx),)
                )
                if a.direction != "both":
                    directions = tuple(
                        value for value in directions
                        if value == a.direction)
                for dirn in directions:
                    jobs.append({
                        "window": w, "dso": d, "direction": dirn,
                        "variant": a.variant, "band_mvar": a.band_mvar,
                        "band_fraction": a.band_fraction,
                        "max_step_mvar": a.max_step_mvar,
                        "der_q_override_pu": a.der_q_override_pu,
                        "flat_mvar": a.flat_mvar,
                        "save_trajectories": a.save_trajectories,
                        "trafo": f"{d}|trafo_{i}",
                        "paramset": a.paramset,
                        "weights": weights_override,
                        "frozen_network": not bool(a.symmetric_network),
                        "bank": a.bank,
                        "no_measurement_noise": bool(a.no_measurement_noise),
                    })

    if a.focus_trafo:
        jobs = [job for job in jobs if job["trafo"] == a.focus_trafo]
    if not jobs:
        print("[abort] case restrictions produced an empty experiment grid")
        return 1

    with (out_dir / "cases.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["window", "dso", "trafo", "direction", "variant",
                     "max_step_mvar"])
        for j in jobs:
            wr.writerow([j["window"], j["dso"], j["trafo"], j["direction"],
                         j["variant"], j["max_step_mvar"]])

    print(f"[ninner] -> {out_dir}")
    direction_note = (
        "both directions per triple" if a.case_design == "full"
        else "balanced half-fraction (one direction per triple)"
    )
    print(f"[ninner] {len(jobs)} steps = {len(names)} windows x {len(dsos)} "
          f"DSOs x {n_trafo} interfaces, {direction_note}, "
          f"{a.workers} workers, variant {a.variant}")
    if a.der_q_override_pu is not None:
        print(f"[ninner] *** DIAGNOSTIC: DER Q capability forced to "
              f"+-{a.der_q_override_pu:g}*S_n for the measurement pass. "
              f"Not a physical model; step sizes still from real capability.")
    if a.max_step_mvar is not None:
        print(f"[ninner] applied per-interface step capped at "
              f"{a.max_step_mvar:g} Mvar; raw capability-edge request retained "
              f"in steps.csv")
    print(f"[ninner] commit {meta['git_commit']} on {meta['git_branch']}"
          + ("  [WORKING TREE DIRTY]" if meta.get("git_dirty") else ""))

    rows: List[Dict[str, Any]] = []
    trajectories: List[Dict[str, Any]] = []
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
            for row in res["rows"]:
                row["case_wall_s"] = res["wall_s"]
            for point in res.get("trajectory", []):
                point["case_wall_s"] = res["wall_s"]
            rows.extend(res["rows"])
            trajectories.extend(res.get("trajectory", []))
            ns = [r["n_inner"] for r in res["rows"]]
            print(f"  [{done}/{len(jobs)}] {res['window']}/{res['dso']}/"
                  f"{res['direction']}  N_inner={ns}, {res['wall_s']:.0f} s")

    if not rows:
        print("[abort] no step produced a measurement; nothing written")
        return 1

    agg = aggregate(rows)
    write_outputs(out_dir, rows, agg, failures, meta)
    if trajectories:
        write_trajectories(out_dir, trajectories)

    pooled = next((x for x in agg if x["dso"] == "__pooled__"), None)
    if pooled:
        print(f"[ninner] capability-limited {pooled['n_capability_limited']}"
              f" of {pooled['n_steps']}; over the {pooled['n_free']} FREE "
              f"steps: median {pooled['free_n_inner_median']:.1f}, "
              f"p95 {pooled['free_n_inner_p95']:.1f}, censored "
              f"{pooled['free_censoring_fraction']:.2f}")
        print(f"\n[ninner] pooled (all steps): median {pooled['n_inner_median']:.1f}, "
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
