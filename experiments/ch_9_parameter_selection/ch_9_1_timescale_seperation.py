r"""
experiments/ch_9_parameter_selection/ch_9_1_timescale_seperation.py
==================================================================
The settling battery that fills **Table 9.1** of the dissertation
(``tab:param:timescales:settling``, \cref{ch:param:timescales}) and the four
bracketed values in the surrounding text. One command, one run directory,
nothing else.

Moved here from ``pf/timescale_study.py`` on 2026-08-06 so that every number
in Ch. 9 has an entry point named after the section it fills; ``pf/`` keeps
the PowerFactory driver infrastructure (``ScreeningContext``, the step
catalogues, the settling metric) that this and the dead-band study share.

What the section argues
-----------------------
The cascade needs ``tau_plant << T_DS << T_TS``. Two settling times decide it,
and they answer different questions:

* **Dispatches** -- the largest single command each actuator class issues.
  This is what bounds ``T_DS``: the period must outlast the transient the
  controller's own action excites.
* **Disturbances** -- a synchronous-machine outage or a load step. No
  controller commands these, so they bound *nothing*; they are measured to
  establish for how many dispatch periods after an event a controller still
  samples a transient rather than a settled plant. Keeping them out of the
  bound is deliberate: a contingency cannot be designed out of an interval.

From the dispatch rows the script derives

    T_DS >= max( T_s^cont , T_s^tap ) + margin

with ``T_s^cont`` the worst *continuous* dispatch row and ``T_s^tap`` the
settling of a **single** tap -- single, because the controller caps taps at
one step per subordinate iteration and then locks the changer out for its
cooldown, so a multi-step command cannot arise. The two-step case in the
catalogue is an *instrument* that splits the tap settling into its mechanical
and electrical parts; it is never a dispatch the controller would issue and
is therefore excluded from the bound (see ``derive``).

What this script does NOT produce
---------------------------------
``N_inner`` of \cref{eq:param:timescales:ninner} is a **closed-loop** property
of the isolated DSO-OFO (parent silent, capability-band-traversing setpoint
step) and is not measured by this open-loop battery. The thesis carries a
``\todo`` for it. It needs a separate entry point in this folder; do not read
``T_TS/T_DS = 9`` out of the summary below as a measurement -- it is printed
there as the *configured* ratio only.

Reproducibility
---------------
Every run writes, next to its results:

* ``run_meta.json``  -- argv, git commit + dirty flag, interpreter, platform,
                        PF project/study case, every constant that enters a
                        number (bands, horizons, RMS step, periods)
* ``cases.csv``      -- the resolved catalogue, written *before* the battery
                        runs, so an aborted run still records what it tried
* ``run.log``        -- the full console transcript
* ``_latest.txt``    -- written one level up; the stamp of the newest run, so
                        "which run is Table 9.1 from" has a single answer

and the exit code is non-zero if any row of Table 9.1 is missing, so an
incomplete battery cannot be mistaken for a complete one.

The post-processing (settling -> table -> derived quantities) is verifiable
without a PowerFactory seat:

    python experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py --self-test

Outputs, in ``results/timescale/<label>/<stamp>/``
-------------------------------------------------
* ``timescale_table.tex``     -- the tabular *body* of Table 9.1, ready to
                                 paste over the ``[TBD]`` rows
* ``timescale_summary.md``    -- the same numbers plus the derived design
                                 quantities and the run's provenance
* ``timescale_summary.csv``   -- one row per case, machine readable
* ``traj_<case>.csv``         -- per-signal time series (``--save-trajectories``)

Usage
-----
    python experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py --label full_t0_wecc --save-trajectories
    python experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py --dry-run     # no PF, no licence
    python experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py --self-test   # no PF, checks the post-processing

See ``experiments/ch_9_parameter_selection/README.md`` and
``docs/handover_timescale_study.md`` for the full run procedure.

Author: Manuel Schwenke / Claude Code (2026-08-04, relocated 2026-08-06)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pf.session import DEFAULT_PROJECT_PATH, PFSessionError  # noqa: E402
from pf.screening import (  # noqa: E402
    RMS_STEP_MS,
    RMS_STUDY_CASE,
    ScreeningContext,
    default_catalogue,
    disturbance_catalogue,
    monitored_outputs,
    resolve_outage_targets,
    settling_metrics,
)

#: Settling bands. Same values the dead-band study and the earlier battery
#: used, so numbers remain comparable across chapters. Quoted verbatim in the
#: caption of Table 9.1 -- change one and the caption is wrong.
BAND_VOLTAGE_PU = 1e-3
BAND_Q_MVAR = 1.0
BAND_DIAGNOSTIC = 1e-4

#: Result down-sampling used when reading a trajectory back (every n-th row of
#: the 10 ms RMS output). At stride 5 the settling time is resolved to 50 ms,
#: which is far below the band-crossing timescale and keeps the read fast.
#: It is recorded in ``run_meta.json`` because it *is* the quantisation of
#: every ``T_s`` reported.
READ_STRIDE = 5

#: Configured periods, from experiments/run_multi_system_ofo.py. Only used to
#: report the margin -- the script never assumes them.
T_DS_S = 20.0
T_TS_S = 180.0

#: OLTC rate limits actually enforced by the controller
#: (``experiments/run_multi_system_ofo.py``: ``local_oltc_max_step_per_dt=1``,
#: ``oltc_cooldown_s_nc=60``, ``oltc_cooldown_s_mt=180``).
#:
#: These matter for how the tap rows are read. **A multi-step tap command
#: cannot occur**: one step per subordinate iteration is a hard cap, and a
#: changer that moves is then locked out for its cooldown -- 3 dispatch
#: intervals for a coupler, 9 for a machine transformer. So the quantity that
#: must fit inside ``T_DS`` is the settling of a SINGLE tap, and the two-step
#: case in the catalogue is an *instrument* for splitting that settling into
#: its mechanical and electrical parts, never a dispatch the controller would
#: issue.
OLTC_MAX_STEP_PER_DT = 1
OLTC_COOLDOWN_NC_S = 60.0
OLTC_COOLDOWN_MT_S = 180.0

#: Case-name substring identifying the two-step tap. It is an instrument, not
#: a dispatch, and is therefore excluded from the bound of eq. (9.1) while
#: still being tabulated (the thesis table labels it "instrument only").
INSTRUMENT_ONLY: Tuple[str, ...] = ("tap_+2seq",)

#: Thesis Table 9.1 row order: (required substrings, caption, is_disturbance).
#:
#: Matched by LITERAL substring, deliberately not by regex: the StepDef names
#: contain ``+`` (``tap_+1_NC3W_DSO_1_t0``), which a regex reads as a
#: quantifier, so ``tap_+1`` silently matches nothing and the row is emitted
#: as "[not run]" for a case that did run. Caught in offline testing
#: 2026-08-04; a silently missing row is the worst failure mode here, because
#: it looks like a deliberate omission in the thesis table. ``--self-test``
#: now guards this offline, and ``main`` exits non-zero if any row is missing.
#:
#: The captions are the thesis wording verbatim. The battery measures MORE
#: cases than the table has rows (the machine-transformer tap, the further
#: load steps); those are reported under "measured but not tabulated" in the
#: summary rather than silently dropped.
#: *Reworked 2026-08-19.* The emitter and the thesis table had diverged: the
#: emitter carried one lumped AVR row and no machine-transformer row, while the
#: table split AVR by machine and did carry the machine transformer. The table
#: was therefore filled by hand, and a hand-transcription error came with it
#: (the coupler-tap row read 11.13 s against a measured 16.28 s, and the
#: location column read "STS 1 B00", an unfilled placeholder). The fix is
#: structural: the emitter now produces exactly the thesis rows, one row per
#: emitted line, and the location column is taken from the measured worst
#: signal -- never typed. If a row is wanted in the thesis, it is added HERE.
#:
#: The two-step tap cases stay OUT of the table (the thesis dropped that row)
#: but stay IN the battery: they are what separates ``T_mech`` from
#: ``T_elec``, per tap class, in ``derive``.
TABLE_ROWS: Tuple[Tuple[Tuple[str, ...], str, bool], ...] = (
    (("der_q_", "WP_TSO"),
     r"Reactive-power step, $+60$\,Mvar, \gls{TSO} park",            False),
    (("der_q_", "DER_"),
     r"Reactive-power step, $+20$\,Mvar, \gls{DSO} \gls{DER}",       False),
    (("avr_vref_+0.02", "G09"),
     r"\gls{AVR} voltage-reference step, $+0.02$\,pu, G\,09",        False),
    (("avr_vref_+0.02", "G10"),
     r"\gls{AVR} voltage-reference step, $+0.02$\,pu, G\,10",        False),
    (("avr_vref_+0.001", "G09"),
     r"\gls{AVR} voltage-reference step, $+0.001$\,pu, G\,09",       False),
    (("avr_vref_+0.001", "G10"),
     r"\gls{AVR} voltage-reference step, $+0.001$\,pu, G\,10",       False),
    (("tap_+1_NC3W",),
     r"\gls{OLTC} coupling transformer, one step",                    False),
    (("tap_+1_MT",),
     r"\gls{OLTC} machine transformer, one step",                     False),
    (("shunt_+1",),
     r"\gls{MSC} switch-in",                                          False),
    (("outage_",),
     r"Synchronous-machine outage",                                    True),
    (("load_",),
     r"Load step",                                                     True),
)


# =====================================================================
#  Provenance and logging
# =====================================================================

class _Tee:
    """Mirror stdout into a run log, so the transcript survives the shell."""

    def __init__(self, stream: TextIO, path: Path) -> None:
        self._stream = stream
        self._fh = path.open("w", encoding="utf-8")

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._fh.write(text)
        self._fh.flush()
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _git(*args: str) -> Optional[str]:
    try:
        proc = subprocess.run(("git", *args), cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def provenance(args: argparse.Namespace) -> Dict[str, Any]:
    """Everything needed to say where a number in Table 9.1 came from.

    ``git_dirty`` is the one that matters: a settling time measured from an
    uncommitted working tree cannot be reproduced from the commit hash alone,
    and the run is not refused for it -- it is recorded, and the summary says
    so in the line the thesis cites.
    """
    status = _git("status", "--porcelain")
    return {
        "script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "argv": sys.argv,
        "args": {k: v for k, v in sorted(vars(args).items())},
        "timestamp": datetime.now().astimezone().isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "pf_project": args.project,
        "pf_study_case": RMS_STUDY_CASE,
        "constants": {
            "band_voltage_pu": BAND_VOLTAGE_PU,
            "band_q_mvar": BAND_Q_MVAR,
            "band_diagnostic": BAND_DIAGNOSTIC,
            "rms_step_ms": RMS_STEP_MS,
            "adaptive_step": False,
            "read_stride": READ_STRIDE,
            "pre_settle_s": args.pre_settle_s,
            "t_ds_s": T_DS_S,
            "t_ts_s": T_TS_S,
            "oltc_max_step_per_dt": OLTC_MAX_STEP_PER_DT,
            "oltc_cooldown_nc_s": OLTC_COOLDOWN_NC_S,
            "oltc_cooldown_mt_s": OLTC_COOLDOWN_MT_S,
        },
    }


def write_case_manifest(out_dir: Path, cases: Sequence[Any]) -> None:
    """Record the resolved catalogue BEFORE the battery runs.

    An aborted or licence-starved run then still documents which cases were
    to be measured, which is what makes a partial result auditable instead of
    merely incomplete.
    """
    with (out_dir / "cases.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "kind", "disturbance", "target", "variable",
                    "delta", "unit", "note"])
        for sd in cases:
            w.writerow([sd.name, sd.kind, int(bool(sd.disturbance)),
                        getattr(sd.target, "loc_name", ""), sd.variable,
                        sd.delta, sd.unit, sd.note])


# =====================================================================
#  Measurement
# =====================================================================

def _band(var: str) -> float:
    if var.startswith("m:u"):
        return BAND_VOLTAGE_PU
    if var.startswith("m:Q"):
        return BAND_Q_MVAR
    return BAND_DIAGNOSTIC


def _is_controlled(var: str) -> bool:
    """Machine speed is recorded but never gates: it is not a controlled output."""
    return var.startswith(("m:u", "m:Q"))


def _read_scalar(obj: Any, var: str) -> float:
    for a in (var, f"s:{var}", f"c:{var}"):
        try:
            return float(obj.GetAttribute(a))
        except Exception:
            continue
    raise PFSessionError(f"cannot read {var!r} on {getattr(obj,'loc_name',obj)!r}")


def preflight(ctx: ScreeningContext, duration: float = 60.0,
              pre_settle_s: float = 0.0) -> float:
    """Refuse to measure settling on a model that is not at equilibrium.

    A flat run that drifts means every settling time measured afterwards would
    be that drift plus the response. This is the one check that must never be
    skipped. Returns the measured drift so it can be recorded.

    ``pre_settle_s`` advances the plant before the measurement window opens,
    and the drift is then evaluated over the LAST ``duration`` seconds only.

    *Added 2026-08-20.* The RMS plant does not hold the load-flow point: the
    anchored ZIP load model (``load_model = "zip"``, P = P_prof*(V/1.03),
    Q = Q_prof*(V/1.03)^2, see the 2026-07-17 log) makes every load
    voltage-following, and at the DSO buses Q is capacitive, so a rising
    voltage increases reactive injection quadratically. The plant therefore
    leaves the load-flow point and converges on the ZIP model's own
    equilibrium ~1.4e-2 pu away, reaching it within the first seconds and then
    creeping. Measuring from the load-flow instant charges that relaxation to
    every settling time.

    Settling first and measuring from the converged point is the honest
    reading: the battery is specified on a *fixed* operating point, not
    specifically on the load-flow one. **State in the caption which it is** --
    with a non-zero pre-settle the operating point is the RMS steady state of
    the ZIP plant, which is not identical to the load-flow solution.
    """
    from pf.screening import FLAT_DRIFT_TOL
    mons = [(o, v, l) for (o, v, l) in monitored_outputs(ctx.app) if v == "m:u"]
    ctx.purge_events()
    ctx.set_monitors(mons)
    ctx.initialise()
    total = float(pre_settle_s) + float(duration)
    ctx.simulate(total)
    worst, where = 0.0, None
    for obj, var, label in mons:
        t, y = ctx.read(obj, var, stride=READ_STRIDE)
        # Evaluate only the measurement window, so the settling transient the
        # pre-settle exists to absorb is not counted as drift.
        window = [v for tt, v in zip(t, y) if tt >= float(pre_settle_s)]
        if not window:
            window = y
        d = max(window) - min(window)
        if d > worst:
            worst, where = d, label
    if pre_settle_s > 0:
        print(f"[preflight] {pre_settle_s:.0f} s pre-settle, then "
              f"{duration:.0f} s flat run: max drift {worst:.2e} pu at {where}")
    else:
        print(f"[preflight] {duration:.0f} s flat run: max drift {worst:.2e} pu at {where}")
    if worst >= FLAT_DRIFT_TOL:
        raise PFSessionError(
            f"model is NOT at equilibrium (drift {worst:.2e} pu >= "
            f"{FLAT_DRIFT_TOL:.0e}); fix the initialisation before measuring "
            f"settling -- every T_s below would be contaminated"
            + ("" if pre_settle_s > 0 else
               "; a --pre-settle-s may be needed, see the docstring"))
    return worst


def run_case(ctx: ScreeningContext, sd: Any, mons: List[Tuple[Any, str, str]],
             t_event: float, horizon: float,
             out_dir: Path, save_traj: bool,
             pre_settle_s: float = 0.0) -> Dict[str, Any]:
    """Run one dispatch or disturbance and return its settling summary.

    ``pre_settle_s`` advances the plant to its own equilibrium BEFORE the step
    is armed. This is not optional cosmetics: ``ctx.initialise()`` runs
    ``ComInc``, which resets to the load-flow point every case, and the plant
    does not hold that point -- the anchored ZIP loads are voltage-following,
    so it relaxes ~1.4e-2 pu over the first seconds (see :func:`preflight`).
    With the default ``t_event = 5 s`` the step would land inside that
    relaxation and every settling time would be the relaxation plus the
    response. A clean preflight does NOT protect against this, because each
    case re-initialises.

    **Events are armed after the settle, not before.** ``add_param_event``
    writes an absolute time while ``add_tap_event`` / ``add_outage_event``
    fold into PowerFactory's 60 s event window using the current calculation
    clock (``EVENT_WINDOW_S``, established 2026-07-31). Arming after the
    settle is exactly the case that fold exists for; arming before it and
    scheduling far ahead would put the tap and outage cases on the untested
    side of that quirk, whose failure mode is an event that never fires and a
    settling time of 0.00 s -- a plausible-looking wrong number.
    """
    ctx.purge_events()
    ctx.set_monitors(mons)
    ctx.initialise()
    if pre_settle_s > 0:
        ctx.simulate(float(pre_settle_s))
    t_event = float(pre_settle_s) + float(t_event)
    if sd.kind == "param":
        cur = _read_scalar(sd.target, sd.variable)
        ctx.add_param_event(sd.target, sd.variable, cur + sd.delta, t_event)
    elif sd.kind == "tap":
        # A transformer with a TAPCTRL must be commanded through the DSL's
        # ``ntapcmd``: the block holds the tap wherever ntapcmd points, so an
        # EvtTap is overwritten on the next solver step and the tap never
        # moves.  Measured 2026-08-07: every tap case returned T_s = 0.00 s
        # because nothing actuated.  The 5 s mechanical travel is already in
        # the block (``Tmech``), so the command goes in at t_event and the
        # tap_times schedule becomes the sequence of COMMANDS.
        # Shunts have no TAPCTRL and keep the EvtTap path, which works.
        from pf.tap_ctrl import tapctrl_of

        dsl = tapctrl_of(ctx.app, sd.target)
        if dsl is None:
            for i, dt in enumerate(sd.tap_times):
                ctx.add_tap_event(sd.target, int(sd.delta), t_event + dt,
                                  seq=i)
        else:
            base = _read_scalar(dsl, "ntapcmd")
            for i, dt in enumerate(sd.tap_times):
                ctx.add_param_event(dsl, "ntapcmd",
                                    base + sd.delta * (i + 1), t_event + dt)
    elif sd.kind == "outage":
        ctx.add_outage_event(sd.target, t_event, seq=0)
        for i, extra in enumerate(sd.also_trip, start=1):
            ctx.add_outage_event(extra, t_event, seq=i)
    elif sd.kind == "load":
        ctx.add_load_event(sd.target, sd.load_pct[0], sd.load_pct[1], t_event)
    else:
        raise PFSessionError(f"unknown kind {sd.kind!r}")

    ctx.simulate(t_event + horizon)
    rows, traj = [], []
    for obj, var, label in mons:
        try:
            t, y = ctx.read(obj, var, stride=READ_STRIDE)
        except Exception:
            if sd.kind == "outage":
                continue        # tripped element's m: variables disappear
            raise
        m = settling_metrics(t, y, t_event, abs_floor=_band(var))
        m["controlled"] = _is_controlled(var)
        rows.append((label, m))
        if save_traj and _is_controlled(var):
            traj.extend((label, a, b) for a, b in zip(t, y))
    if save_traj and traj:
        with (out_dir / f"traj_{sd.name}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["signal", "t", "y"])
            for label, a, b in traj:
                w.writerow([label, f"{a:.4f}", f"{b:.9g}"])

    ctrl = [(l, m) for l, m in rows if m["controlled"]]
    if not ctrl:
        raise PFSessionError(f"{sd.name}: no controlled output recorded")
    ctrl.sort(key=lambda r: -r[1]["t_settle"])
    label, worst = ctrl[0]
    # A settling time equal to the horizon is NOT a settling time: the
    # trajectory left the band at the last sample the run holds, so the true
    # value is >= horizon and unknown. Flagged rather than silently reported,
    # because such a row would otherwise set the bound of eq. (9.1) at a
    # value that is an artefact of the run length.
    censored = worst["t_settle"] >= horizon - 1e-9
    return {"case": sd.name, "note": sd.note, "kind": sd.kind,
            "disturbance": bool(sd.disturbance), "worst_signal": label,
            "t_settle_s": worst["t_settle"], "overshoot": worst["overshoot"],
            "step": worst["step"], "horizon_s": horizon,
            "censored": bool(censored),
            "instrument_only": any(s in sd.name for s in INSTRUMENT_ONLY),
            "tabulated": any(all(s in sd.name for s in subs)
                             for subs, _c, _d in TABLE_ROWS)}


# =====================================================================
#  Derived design quantities
# =====================================================================

def _tap_split(disp: List[Dict[str, Any]], one_sub: str, two_sub: str,
               ) -> Dict[str, Any]:
    r"""Separate mechanical travel from electrical transient for ONE tap class.

    A completed tap is a step change in transformer ratio and excites the
    network like any other step, so a ``|dtau|``-step command settles in

        T_s(|dtau|) = T_mech * |dtau| + T_elec

    and two measured points separate the terms exactly, ``T_elec`` being
    common to both and cancelling in the difference::

        T_mech = T_s(2) - T_s(1)
        T_elec = T_s(1) - T_mech = 2*T_s(1) - T_s(2)

    Both substrings are class-qualified by the caller (``tap_+1_NC3W`` /
    ``tap_+2seq_NC3W``), because a coupler tap and a machine-transformer tap
    have different splits: the coupler is seen largely as an algebraic change
    in the interface flow, whereas the machine transformer acts against the
    excitation control and can leave a tail comparable to the travel itself.
    """
    one = next((r for r in disp if one_sub in r["case"]), None)
    two = next((r for r in disp if two_sub in r["case"]), None)
    if one and two:
        t_mech = two["t_settle_s"] - one["t_settle_s"]
        return {"t_tap": one["t_settle_s"], "t_mech": t_mech,
                "t_elec": one["t_settle_s"] - t_mech,
                "one_case": one["case"], "two_case": two["case"],
                "source": ("T_mech from the 2-step minus 1-step difference; "
                           "T_elec = T_s(1) - T_mech")}
    if one:
        # Cannot separate the two with one point. Attributing all of it to
        # the mechanism would understate the per-step cost and inflate any
        # step cap, so the conservative split is taken and flagged.
        return {"t_tap": one["t_settle_s"], "t_mech": one["t_settle_s"],
                "t_elec": 0.0, "one_case": one["case"], "two_case": None,
                "source": ("single-step case only -- T_mech and T_elec NOT "
                           "separable; all settling attributed to the "
                           "mechanism (conservative, but the split is "
                           "unmeasured)")}
    return {"t_tap": float("nan"), "t_mech": float("nan"),
            "t_elec": float("nan"), "one_case": None, "two_case": None,
            "source": "no tap case ran"}


def derive(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    r"""Turn the measured rows into the design quantities of eq. (9.1).

    Two things are computed and they are not the same.

    **The bound.** ``T_DS`` must outlast the transient excited by *any*
    dispatch the controller can actually issue, so the binding row is the
    worst over every realisable dispatch case -- continuous rows, tap rows and
    the MSC switch-in alike. Only the two-step tap is excluded, and only
    because the controller cannot issue it (``OLTC_MAX_STEP_PER_DT = 1`` plus
    the cooldown lockout); it is an instrument, not a dispatch.

    *Changed 2026-08-06.* The earlier version took ``max(T_s^cont, T_s^tap)``
    with ``T_s^cont`` over the ``param`` rows only and ``T_s^tap`` over the
    coupler tap only, which silently excluded the MSC switch-in and the
    machine-transformer tap from the bound. Both are dispatches the controller
    issues, so a slow one would have gone unnoticed. ``T_s^cont`` and
    ``T_s^tap`` are still reported -- the thesis text names them -- but the
    margin is now taken against the true worst realisable dispatch.

    **The tap split.** A tap case is NOT purely a mechanical delay. A
    completed tap is a step change in transformer ratio and excites the
    network like any other step, so the settling of a ``|dtau|``-step command
    is

        T_s(|dtau|) = T_mech * |dtau| + T_elec

    with ``T_mech`` the mechanical time per step and ``T_elec`` the electrical
    transient following the LAST step. Two measured points separate them
    exactly, because ``T_elec`` is common to both and cancels in the
    difference::

        T_mech = T_s(2) - T_s(1)
        T_elec = T_s(1) - T_mech = 2*T_s(1) - T_s(2)

    Whether ``T_elec`` is in fact small is a property of the transformer and
    of which output binds -- a network-coupler tap is seen largely as an
    algebraic change in the interface flow, whereas a machine-transformer tap
    acts against the excitation control and can leave a tail comparable to
    ``T_mech``. It is therefore measured, never assumed.

    The split treats ``T_elec`` as independent of ``|dtau|``; two points fix
    the line but cannot show it is one, which a 1..4 sweep would test.
    """
    disp = [r for r in results if not r["disturbance"]]
    cont = [r for r in disp if r["kind"] == "param"]
    t_cont = max((r["t_settle_s"] for r in cont), default=float("nan"))
    cont_worst = max(cont, key=lambda r: r["t_settle_s"])["case"] if cont else None

    # The split is per TAP CLASS and the classes do not share it. Matching
    # ``tap_+2seq`` alone was safe only while the coupler owned the single
    # two-step case; the machine transformer got one on 2026-08-19, and an
    # unqualified substring would then pair a coupler one-step with whichever
    # two-step case the catalogue happened to emit first -- silently, and
    # producing a T_mech that belongs to neither transformer.
    splits = {cls: _tap_split(disp, f"tap_+1_{sub}", f"tap_+2seq_{sub}")
              for cls, sub in (("coupler", "NC3W"), ("machine_trafo", "MT"))}

    # The thesis text names T_s^tap / T_mech / T_elec for the COUPLER; the
    # machine-transformer split is reported alongside it, not folded into it.
    cpl = splits["coupler"]
    t_tap, t_mech, t_elec = cpl["t_tap"], cpl["t_mech"], cpl["t_elec"]
    t_tap_src = cpl["source"]

    realisable = [r for r in disp if not r["instrument_only"]]
    binding_row = (max(realisable, key=lambda r: r["t_settle_s"])
                   if realisable else None)
    binding = binding_row["t_settle_s"] if binding_row else float("nan")
    if binding_row is None:
        binding_kind = "no realisable dispatch measured"
    elif binding_row["kind"] == "param":
        binding_kind = "continuous dispatch"
    elif "shunt_" in binding_row["case"]:
        binding_kind = "MSC switch-in"
    else:
        binding_kind = "single tap"
    margin = T_DS_S - binding if binding == binding else float("nan")

    dist = [r for r in results if r["disturbance"]]
    censored = [r["case"] for r in results if r["censored"]]
    return {"tap_splits": splits,
            "t_cont": t_cont, "t_cont_case": cont_worst, "t_tap": t_tap,
            "t_mech": t_mech, "t_elec": t_elec, "t_tap_source": t_tap_src,
            "binding": binding, "binding_kind": binding_kind,
            "binding_case": binding_row["case"] if binding_row else None,
            "margin": margin, "n_dispatch": len(disp),
            "n_disturbance": len(dist), "censored": censored,
            "worst_disturbance": (max(dist, key=lambda r: r["t_settle_s"])
                                  if dist else None)}


# =====================================================================
#  Outputs
# =====================================================================

def build_table(results: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Table 9.1 body plus the list of captions no measured case filled."""
    tex: List[str] = [r"\multicolumn{3}{l}{\emph{Dispatches}} \\"]
    missing: List[str] = []
    emitted: set = set()
    dist_header_written = False
    for subs, caption, is_dist in TABLE_ROWS:
        if is_dist and not dist_header_written:
            tex.append(r"\midrule")
            tex.append(r"\multicolumn{3}{l}{\emph{Disturbances}} \\")
            dist_header_written = True
        hit = next((r for r in results
                    if all(s in r["case"] for s in subs)
                    and r["case"] not in emitted), None)
        if hit is None:
            tex.append(f"{caption:<45}& [not run] & [--] \\\\")
            missing.append(caption)
            continue
        emitted.add(hit["case"])
        sig = hit["worst_signal"].replace("_", r"\_")
        mark = r"$>$" if hit["censored"] else ""
        tex.append(f"{caption:<45}& \\texttt{{{sig}}} & "
                   f"{mark}{hit['t_settle_s']:.2f} \\\\")
    return tex, missing


def write_outputs(out_dir: Path, results: List[Dict[str, Any]],
                  d: Dict[str, Any], meta: Optional[Dict[str, Any]] = None,
                  ) -> List[str]:
    """Write the three thesis-facing files; return the missing table rows."""
    cols = ["case", "kind", "disturbance", "instrument_only", "tabulated",
            "censored", "note", "worst_signal", "t_settle_s", "overshoot",
            "step", "horizon_s"]
    with (out_dir / "timescale_summary.csv").open("w", newline="",
                                                  encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in cols})

    tex, missing = build_table(results)
    (out_dir / "timescale_table.tex").write_text("\n".join(tex), encoding="utf-8")

    md = ["# Timescale study (thesis Table 9.1)", "",
          "Dispatch rows bound `T_DS`; disturbance rows do not and are "
          "reported only to establish for how many periods after an event a "
          "controller still samples a transient.", "",
          "| case | kind | worst controlled output | T_s [s] |",
          "|:--|:--|:--|--:|"]
    for r in results:
        tag = "disturbance" if r["disturbance"] else r["kind"]
        if r["instrument_only"]:
            tag += ", instrument"
        cens = " (censored: >= horizon)" if r["censored"] else ""
        md.append(f"| {r['case']} | {tag} | {r['worst_signal']} | "
                  f"{r['t_settle_s']:.2f}{cens} |")

    md += ["", "## Derived design quantities", "",
           f"- worst continuous dispatch `T_s^cont` = **{d['t_cont']:.2f} s** "
           f"(`{d['t_cont_case']}`)",
           f"- single-tap settling `T_s^tap` = **{d['t_tap']:.2f} s** "
           f"(coupling transformer) = `T_mech` {d['t_mech']:.2f} s "
           f"(mechanical travel) + `T_elec` {d['t_elec']:.2f} s (electrical "
           f"transient the completed tap excites -- a tap is a step change in "
           f"ratio like any other, so this ADDS rather than being covered by "
           f"the mechanical time)",
           f"  ({d['t_tap_source']})",
           f"- **binding row: {d['binding_kind']}** (`{d['binding_case']}`) at "
           f"{d['binding']:.2f} s -- the worst over EVERY realisable dispatch, "
           f"not only the continuous and coupler-tap rows; at "
           f"`T_DS` = {T_DS_S:.0f} s the margin is **{d['margin']:.2f} s**",
           f"- no `|dtau|_max` is derived: the controller caps taps at "
           f"{OLTC_MAX_STEP_PER_DT} step per iteration and then locks the "
           f"changer out for {OLTC_COOLDOWN_NC_S:.0f} s (coupler) / "
           f"{OLTC_COOLDOWN_MT_S:.0f} s (machine transformer), i.e. "
           f"{OLTC_COOLDOWN_NC_S/T_DS_S:.0f} and "
           f"{OLTC_COOLDOWN_MT_S/T_DS_S:.0f} dispatch intervals, so a "
           f"multi-step command cannot arise",
           f"- `T_TS` = {T_TS_S:.0f} s over `T_DS` = {T_DS_S:.0f} s is the "
           f"**configured** ratio {T_TS_S/T_DS_S:.0f}. This is NOT the "
           f"measured `N_inner` of eq. (9.2): that is a closed-loop property "
           f"of the isolated DSO-OFO and this open-loop battery does not "
           f"measure it. Do not quote this line as evidence for `N_inner`."]

    # Both tap classes, side by side. The thesis caption asserts the
    # mechanical block parameter (5 s) is what a tap costs; whether that
    # holds is a measurement, and it is reported per class because the two
    # classes do not share a split.
    md += ["", "## Tap split by class (T_mech / T_elec)", "",
           "`T_mech = T_s(2) - T_s(1)`, `T_elec = 2*T_s(1) - T_s(2)`, from "
           "the one-step and the sequential two-step case of the SAME "
           "transformer. The two-step case is an instrument: the controller "
           "caps taps at one step per iteration and then locks the changer "
           "out, so it is never a dispatch and never enters the bound.", "",
           "| class | T_s(1) [s] | T_s(2) [s] | T_mech [s] | T_elec [s] |",
           "|:--|--:|--:|--:|--:|"]
    for cls, sp in d.get("tap_splits", {}).items():
        two_t = next((r["t_settle_s"] for r in results
                      if sp["two_case"] and r["case"] == sp["two_case"]),
                     float("nan"))
        md.append(f"| {cls} | {sp['t_tap']:.2f} | {two_t:.2f} | "
                  f"{sp['t_mech']:.2f} | {sp['t_elec']:.2f} |")
    for cls, sp in d.get("tap_splits", {}).items():
        if sp["two_case"] is None:
            md.append("")
            md.append(f"- **{cls}: {sp['source']}**")

    if d["worst_disturbance"]:
        wd = d["worst_disturbance"]
        md += ["",
               f"- worst **disturbance** settling {wd['t_settle_s']:.2f} s "
               f"(`{wd['case']}`) = {wd['t_settle_s']/T_DS_S:.1f} dispatch "
               f"intervals. This does NOT enter the bound; where it exceeds "
               f"`T_DS` the quasi-steady-state premise is violated for that "
               f"window and \\cref{{ch:case_rms}} evaluates the consequence."]

    not_tabulated = [r for r in results if not r["tabulated"]]
    if not_tabulated:
        md += ["", "## Measured but not in Table 9.1", "",
               "These cases ran and enter the derived quantities where the "
               "docstring of `derive` says they do, but the thesis table has "
               "no row for them. Listed so the omission is visible rather "
               "than silent.", ""]
        md += [f"- `{r['case']}` -- {r['note']} -- T_s = {r['t_settle_s']:.2f} s"
               for r in not_tabulated]

    if missing:
        md += ["", "## Table rows NOT filled by this run", "",
               "The battery produced no case for these rows; the `.tex` body "
               "carries `[not run]` for them. **Do not paste the table into "
               "the thesis until this list is empty.**", ""]
        md += [f"- {c}" for c in missing]

    if d["censored"]:
        md += ["", "## Censored settling times", "",
               "The trajectory was still outside the band at the last sample "
               "of the run, so the reported `T_s` is a lower bound, not a "
               "measurement. Re-run these with a longer horizon before "
               "quoting them.", ""]
        md += [f"- `{c}`" for c in d["censored"]]

    if meta:
        md += ["", "## Provenance", "",
               f"- run: `{meta['timestamp']}`",
               f"- commit: `{meta['git_commit']}` on `{meta['git_branch']}`"
               + ("  **(working tree dirty -- this run is not reproducible "
                  "from the commit alone)**" if meta.get("git_dirty") else ""),
               f"- project `{meta['pf_project']}`, study case "
               f"`{meta['pf_study_case']}`",
               f"- bands: {BAND_VOLTAGE_PU:g} pu on voltages, "
               f"{BAND_Q_MVAR:g} Mvar on interface flows; RMS step "
               f"{RMS_STEP_MS:g} ms fixed, read stride {READ_STRIDE} "
               f"({RMS_STEP_MS*READ_STRIDE:g} ms resolution on every T_s)",
               f"- command: `{' '.join(meta['argv'])}`",
               "", "Full detail in `run_meta.json`; the attempted catalogue "
               "is in `cases.csv`."]

    md += ["", "Paste `timescale_table.tex` over the `[TBD]` rows of "
           "`tab:param:timescales:settling`."]
    (out_dir / "timescale_summary.md").write_text("\n".join(md), encoding="utf-8")
    return missing


# =====================================================================
#  Offline self-test (no PowerFactory)
# =====================================================================

def _synthetic_results() -> List[Dict[str, Any]]:
    """One row per Table 9.1 case, with the name shapes the catalogue emits."""
    def row(case, kind, dist, t, note="synthetic", signal="u_TN_bus16",
            horizon=60.0):
        return {"case": case, "note": note, "kind": kind, "disturbance": dist,
                "worst_signal": signal, "t_settle_s": t, "overshoot": 0.1,
                "step": 1.0, "horizon_s": horizon, "censored": False,
                "instrument_only": any(s in case for s in INSTRUMENT_ONLY),
                "tabulated": any(all(s in case for s in subs)
                                 for subs, _c, _d in TABLE_ROWS)}
    return [
        row("der_q_+60Mvar_WP_TSO_1", "param", False, 3.10),
        row("der_q_+15Mvar_DER_DSO_1_b3", "param", False, 2.40),
        row("avr_vref_+0.02_G09", "param", False, 6.80),
        row("avr_vref_+0.02_G10", "param", False, 5.40),
        # The realistic-magnitude rows settle faster than the 0.02 pu worst
        # case, which is the expected ordering; the self-test asserts the
        # bound does not come from them.
        row("avr_vref_+0.001_G09", "param", False, 1.90),
        row("avr_vref_+0.001_G10", "param", False, 1.60),
        row("tap_+1_NC3W_DSO_1_t0", "tap", False, 7.20),
        row("tap_+2seq_NC3W_DSO_1_t0", "tap", False, 11.90),
        row("tap_+1_MT_g0_t0", "tap", False, 8.30),
        # Deliberately a DIFFERENT split from the coupler (T_mech = 3.40 s
        # against the coupler's 4.70 s): if the two classes were ever paired
        # by an unqualified ``tap_+2seq`` match, this row would leak into the
        # coupler split and the test below would catch it.
        row("tap_+2seq_MT_g0_t0", "tap", False, 11.70),
        row("shunt_+1_MSC_TN_bus16_0", "tap", False, 9.10),
        row("outage_G03", "outage", True, 240.0, horizon=600.0),
        row("load_+25pct_TN_load15", "load", True, 95.0, horizon=600.0),
    ]


def self_test() -> int:
    """Check the post-processing offline: no PF, no licence, no plant.

    Guards the three failure modes that would put a wrong number in the
    thesis: a table row silently matching nothing, the bound missing a
    realisable dispatch, and the tap split being computed the wrong way round.
    """
    import tempfile
    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
              + (f" -- {detail}" if detail and not cond else ""))
        ok = ok and cond

    print("[self-test] post-processing of the timescale battery")
    res = _synthetic_results()
    tex, missing = build_table(res)
    check("every Table 9.1 row is filled", not missing, f"missing: {missing}")
    check("no '[not run]' in the emitted body",
          not any("[not run]" in line for line in tex))
    check("row count = one line per table row + 2 group headers + 1 midrule",
          len(tex) == len(TABLE_ROWS) + 3, f"got {len(tex)}")
    check("the '+' in tap names is matched literally",
          any("7.20" in line for line in tex))
    check("the two-step instruments are NOT tabulated",
          not any("11.90" in line or "11.70" in line for line in tex))
    check("AVR rows are split by machine AND magnitude (4 rows)",
          sum(1 for line in tex if "voltage-reference" in line) == 4,
          f"got {[l for l in tex if 'voltage-reference' in l]}")
    check("the machine-transformer tap has its own table row",
          any("machine transformer" in line and "8.30" in line
              for line in tex))
    check("the coupler row carries the MEASURED value, not a typed one",
          any("coupling transformer" in line and "7.20" in line
              for line in tex))

    d = derive(res)
    check("T_s^cont is the worst *param* row (AVR, 6.80 s)",
          abs(d["t_cont"] - 6.80) < 1e-9, f"got {d['t_cont']}")
    check("coupler T_mech = T_s(2) - T_s(1) = 4.70 s",
          abs(d["t_mech"] - 4.70) < 1e-9, f"got {d['t_mech']}")
    check("coupler T_elec = 2*T_s(1) - T_s(2) = 2.50 s",
          abs(d["t_elec"] - 2.50) < 1e-9, f"got {d['t_elec']}")
    mt = d["tap_splits"]["machine_trafo"]
    check("machine-trafo split uses ITS OWN two-step case (T_mech = 3.40 s)",
          abs(mt["t_mech"] - 3.40) < 1e-9, f"got {mt['t_mech']}")
    check("machine-trafo T_elec = 4.90 s",
          abs(mt["t_elec"] - 4.90) < 1e-9, f"got {mt['t_elec']}")
    check("the two tap classes are paired within their own class",
          d["tap_splits"]["coupler"]["two_case"] == "tap_+2seq_NC3W_DSO_1_t0"
          and mt["two_case"] == "tap_+2seq_MT_g0_t0")
    check("the bound sees the MSC row (9.10 s binds, not the 6.80 s AVR)",
          abs(d["binding"] - 9.10) < 1e-9, f"got {d['binding']}")
    check("the 2-step instrument is excluded from the bound",
          d["binding"] < 11.90)
    check("margin = T_DS - binding = 10.90 s",
          abs(d["margin"] - 10.90) < 1e-9, f"got {d['margin']}")
    check("disturbances do not enter the bound", d["binding"] < 95.0)
    check("worst disturbance is the outage",
          d["worst_disturbance"]["case"] == "outage_G03")
    check("the machine-transformer tap is now tabulated",
          any(r["case"] == "tap_+1_MT_g0_t0" and r["tabulated"]
              for r in res))
    check("the two-step instruments are the untabulated cases",
          {r["case"] for r in res if not r["tabulated"]}
          == {"tap_+2seq_NC3W_DSO_1_t0", "tap_+2seq_MT_g0_t0"},
          f"got {sorted(r['case'] for r in res if not r['tabulated'])}")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        write_outputs(out, res, d, meta=None)
        for name in ("timescale_summary.csv", "timescale_summary.md",
                     "timescale_table.tex"):
            check(f"{name} written", (out / name).exists())
        body = (out / "timescale_summary.md").read_text(encoding="utf-8")
        check("summary lists the untabulated two-step instruments",
              "tap_+2seq_MT_g0_t0" in body)
        check("summary reports the split for BOTH tap classes",
              "machine_trafo" in body and "coupler" in body)
        check("summary refuses to sell T_TS/T_DS as N_inner",
              "does not measure it" in body)

    # A run that lost a case must not look complete.
    partial = [r for r in res if "shunt_" not in r["case"]]
    _tex_p, missing_p = build_table(partial)
    check("a lost case is reported as a missing table row",
          missing_p == [r"\gls{MSC} switch-in"], f"got {missing_p}")

    print(f"[self-test] {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


# =====================================================================
#  Entry point
# =====================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Settling battery for the timescale selection "
                    "(thesis Sec. 9.1, Table 9.1).")
    ap.add_argument("--label", default="full_t0_wecc",
                    help="snapshot label for the results folder")
    ap.add_argument("--project", default=DEFAULT_PROJECT_PATH)
    ap.add_argument("--t-event", type=float, default=5.0)
    ap.add_argument("--dispatch-horizon", type=float, default=60.0,
                    help="run length [s] after a dispatch (default 60 = 3x T_DS)")
    ap.add_argument("--disturbance-horizon", type=float, default=600.0,
                    help="run length [s] after a disturbance (default 600). A "
                         "contingency recovery is far longer than a dispatch "
                         "transient; too short a horizon reports a settled "
                         "plant that is not settled.")
    ap.add_argument("--outage-gens", nargs="*", type=int, default=[1, 7],
                    metavar="IDX",
                    help="machines by PANDAPOWER gen index (the convention the "
                         "contingency configs use). Default 1 7 = G 03 (650 MW) "
                         "and G 09 (830 MW), matching the dead-band N-1 study. "
                         "NOTE the numberings are offset; each resolution is "
                         "printed. gen[8] is the slack equivalent and is "
                         "refused.")
    ap.add_argument("--outage-machines", nargs="*", default=None, metavar="NAME")
    ap.add_argument("--no-disturbances", action="store_true",
                    help="dispatch rows only")
    ap.add_argument("--save-trajectories", action="store_true",
                    help="also persist per-signal time series")
    ap.add_argument("--pre-settle-s", type=float, default=0.0,
                    help="advance the plant this long before the preflight "
                         "measurement window opens, and before the battery. "
                         "The RMS plant does not hold the load-flow point (ZIP "
                         "loads are voltage-following); settling first makes "
                         "the operating point the RMS steady state, which the "
                         "caption must then say.")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the flat-run equilibrium check (NOT advised)")
    ap.add_argument("--out-dir", default=None,
                    help="write here instead of results/timescale/<label>/<stamp>")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the cases that would run; no PowerFactory needed")
    ap.add_argument("--self-test", action="store_true",
                    help="check the post-processing offline; no PowerFactory "
                         "and no licence seat needed")
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test()

    if a.dry_run:
        print("[dry-run] would run, in order:")
        print("  preflight  : 60 s flat run, assert drift < tolerance")
        print(f"  dispatches : per actuator class, {a.dispatch_horizon:.0f} s each")
        if a.no_disturbances:
            print("  disturbances: skipped (--no-disturbances)")
        else:
            print(f"  disturbances: gen trips + load steps, "
                  f"{a.disturbance_horizon:.0f} s each")
        resolve_outage_targets(a.outage_machines, a.outage_gens)
        print("  outputs    : timescale_summary.{csv,md}, timescale_table.tex, "
              "run_meta.json, cases.csv, run.log")
        prov = provenance(a)
        print(f"  commit     : {prov['git_commit']} on {prov['git_branch']}"
              + ("  [WORKING TREE DIRTY]" if prov["git_dirty"] else ""))
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label_dir = REPO_ROOT / "results" / "timescale" / a.label
    out_dir = Path(a.out_dir) if a.out_dir else label_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    tee = _Tee(sys.stdout, out_dir / "run.log")
    sys.stdout = tee                      # type: ignore[assignment]
    try:
        meta = provenance(a)
        (out_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[timescale] -> {out_dir}")
        print(f"[timescale] commit {meta['git_commit']} on {meta['git_branch']}"
              + ("  [WORKING TREE DIRTY -- not reproducible from the commit "
                 "alone]" if meta["git_dirty"] else ""))

        from pf.session import connect
        app = connect(a.project, study_case=RMS_STUDY_CASE)
        ctx = ScreeningContext(app, rms_step_ms=RMS_STEP_MS,
                               adaptive_step=False)
        drift = None
        if not a.skip_preflight:
            drift = preflight(ctx, pre_settle_s=a.pre_settle_s)
        else:
            print("[timescale] WARNING: preflight skipped; every T_s below may "
                  "be contaminated by initialisation drift")

        mons = monitored_outputs(app)
        cases = list(default_catalogue(app))
        if not a.no_disturbances:
            targets = resolve_outage_targets(a.outage_machines, a.outage_gens)
            cases += disturbance_catalogue(app, targets or None)
        write_case_manifest(out_dir, cases)
        print(f"[timescale] {len(cases)} cases "
              f"({sum(1 for c in cases if not c.disturbance)} dispatches, "
              f"{sum(1 for c in cases if c.disturbance)} disturbances), "
              f"{len(mons)} monitored signals")

        results: List[Dict[str, Any]] = []
        failures: List[Tuple[str, str]] = []
        for i, sd in enumerate(cases, 1):
            horizon = a.disturbance_horizon if sd.disturbance else a.dispatch_horizon
            print(f"[{i}/{len(cases)}] {sd.name} ({horizon:.0f} s) ...")
            try:
                r = run_case(ctx, sd, mons, a.t_event, horizon, out_dir,
                             a.save_trajectories, pre_settle_s=a.pre_settle_s)
            except Exception as exc:                  # one bad case must not
                print(f"    FAILED: {exc}")           # lose the whole battery
                failures.append((sd.name, str(exc)))
                continue
            results.append(r)
            print(f"    worst {r['worst_signal']} T_s = {r['t_settle_s']:.2f} s"
                  + ("  [CENSORED: still outside the band at the end of the "
                     "run]" if r["censored"] else ""))

        meta["preflight_drift_pu"] = drift
        meta["failures"] = [{"case": c, "error": e} for c, e in failures]
        meta["n_cases_attempted"] = len(cases)
        meta["n_cases_succeeded"] = len(results)
        (out_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        if not results:
            print("[timescale] no case succeeded; nothing written",
                  file=sys.stderr)
            return 1

        d = derive(results)
        missing = write_outputs(out_dir, results, d, meta)
        if not a.out_dir:
            (label_dir / "_latest.txt").write_text(stamp + "\n",
                                                   encoding="utf-8")

        print(f"\n[timescale] T_s^cont = {d['t_cont']:.2f} s, "
              f"T_s^tap = {d['t_tap']:.2f} s "
              f"(= {d['t_mech']:.2f} mech + {d['t_elec']:.2f} elec)")
        print(f"[timescale] binding: {d['binding_kind']} "
              f"({d['binding_case']}) at {d['binding']:.2f} s "
              f"-> margin {d['margin']:.2f} s at T_DS = {T_DS_S:.0f} s")
        print(f"[timescale] summary -> {out_dir/'timescale_summary.md'}")

        rc = 0
        if failures:
            print(f"[timescale] {len(failures)} case(s) FAILED: "
                  f"{', '.join(c for c, _ in failures)}")
            rc = 2
        if missing:
            print(f"[timescale] Table 9.1 INCOMPLETE -- {len(missing)} row(s) "
                  f"unfilled: {'; '.join(missing)}")
            print("[timescale] do NOT paste timescale_table.tex into the "
                  "thesis until every row is filled")
            rc = 2
        if d["censored"]:
            print(f"[timescale] {len(d['censored'])} censored settling "
                  f"time(s): {', '.join(d['censored'])} -- re-run these with "
                  f"a longer horizon")
            rc = 2
        return rc
    except Exception:
        # The transcript is what gets reported back (handover Sec. 6), so a
        # fatal error belongs in run.log and not only on the terminal.
        import traceback
        print("[timescale] ABORTED:")
        traceback.print_exc(file=sys.stdout)
        raise
    finally:
        sys.stdout = tee._stream            # type: ignore[assignment]
        tee.close()


if __name__ == "__main__":
    sys.exit(main())
