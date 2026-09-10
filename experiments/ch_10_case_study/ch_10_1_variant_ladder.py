"""
ch_10_1_variant_ladder.py
=========================

The dissertation's control-variant ladder (thesis \\cref{ch:case3}), on the
**validated** parameter set.

Why this file exists and 005_CIGRE_MULTI does not serve
-------------------------------------------------------
``experiments/CIGRE_2026/005_CIGRE_MULTI.py`` carries a parameter set frozen for
a specific paper.  Measured 2026-09-02, it differs from the authoritative
``experiments/run_multi_system_ofo.make_config`` -- which declares itself "the
only config factory in the project" -- in **15 fields**, three of them
structural rather than tuning:

    g_w_der   14.4 vs 100      g_w_gen  1e9 vs 5e9    g_w_pcc  54.8 vs 200
    g_w_tso_oltc 3783 vs 1e4   g_q      250 vs 200    g_w_dso_oltc 183 vs 200
    install_tso_tertiary_shunts  True vs False        <- structural
    shunt_dispatch  'integrator' vs 'off'             <- structural
    tie_boundary_equivalent  'thevenin' vs 'pq'       <- structural

The paper's set is left untouched.  This driver derives from the authoritative
factory with ``dataclasses.replace`` and overrides ONLY the horizon, so the
weights can never drift again: a change in ``make_config`` propagates here for
free, and nothing is restated that could go stale.  **Do not add weight
literals below.**

Consequences of running on the validated set
--------------------------------------------
* Switched shunts (MSC/MSR) EXIST and are dispatched, which they are not under
  the paper's set.  They run on the separate ``ShuntIntegrator`` (hysteresis +
  dwell), NOT as MIQP integers -- ``zone_shunt_buses`` is left empty unless
  ``shunt_dispatch == 'miqp'``.
* They are NOT a common layer beneath every variant, and assuming so was wrong:
  the integrator skips any zone whose TSOController has no cached measurement,
  and that cache is filled only when the TSO **OFO** runs.  The banks are
  therefore inert wherever there is no TSO OFO (L1, L2, S1) and live wherever
  there is (O1..M).  Left alone that would confound S1->O1.  They are switched
  off explicitly through O1 -- see ``_NO_SHUNTS``.
* O1->O2 is consequently "TS discrete devices enter the loop": OLTCs join the
  MIQP integer block and the shunts start switching.  Two device classes, one
  mechanism.
* The transmission integer block is OLTCs only, so ``tso_oltc_mode='local'``
  empties it entirely.
* The boundary equivalent is Thevenin, not PQ.

Horizon
-------
360 min, so that all six of the authoritative contingencies fire (gen trip 30,
load 120, gen restore 180, line trip 210, line restore 300, load trip 360).
The contingency schedule itself is inherited, not restated.

Run:
    python -m experiments.ch_10_case_study.ch_10_1_variant_ladder
    python -m experiments.ch_10_case_study.ch_10_1_variant_ladder --only S1,O1
    python -m experiments.ch_10_case_study.ch_10_1_variant_ladder --report
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from configs.config import MultiTSOConfig
from experiments.paths import RESULTS_ROOT
from experiments.run_multi_system_ofo import make_config
from experiments.runners.multi_tso_dso import run_multi_tso_dso

EXPERIMENT = "ch10_ladder"

#: Default horizon [min].  360 fires all six inherited contingencies
#: (gen trip 30, load 120, gen restore 180, line trip 210, line restore
#: 300, load trip 360).  A SHORTER horizon truncates that schedule --
#: at 120 min only the gen trip at 30 and the load connect at 120 fire,
#: the machine is never restored, and the run ends in a degraded
#: topology.  Useful as a fast read, NOT interchangeable with the full
#: trajectory; never compare metrics across horizons.
HORIZON_MIN = 120

#: Deliberate, per-run override of the AVR move budget, applied to EVERY
#: variant.  None = inherit make_config (currently 1e9).
#:
#: This is NOT a restated weight -- it is a sweep axis, set from the command
#: line and echoed into the run header so a result can always be traced to the
#: gauge that produced it.  Do not give it a literal default; the ladder must
#: keep deriving its weights.
#:
#: Measured 2026-09-02 (120 min, validated config), AVR travel ratio to S1 and
#: contraction p95 against the ~1.5 budget:
#:     1e9  0.06/0.43/0.42   0.05/0.23/0.15   frozen in zone 1
#:     1e8  0.23/1.01/1.53   0.22/0.44/0.34   largest admissible
#:     1e7  0.63/1.52/1.91   2.09/2.62/2.25   EXCEEDS the budget
#:     1e6  railed (0.12000 = setpoint bound)
#: Note g_w_gen is inert for L1, L2 and S1 -- none of them runs a TSO OFO on
#: the machines -- so those three rungs are expected to come out identical
#: across gauges.  That doubles as a determinism check on the harness.
G_W_GEN: Optional[float] = None


# ---------------------------------------------------------------------------
#  Configuration -- derived, never restated
# ---------------------------------------------------------------------------


def make_thesis_config() -> MultiTSOConfig:
    """Authoritative config with the case-study horizon.

    ``dataclasses.replace`` on ``make_config()`` is the whole point: weights,
    shunt settings, boundary equivalent and contingency schedule are inherited.
    Overrides are the horizon and the DSO_3 scenario scaling only -- no weight
    is ever restated here.
    """
    return dataclasses.replace(
        make_config(),
        n_total_s=60.0 * HORIZON_MIN,
        verbose=1,
        # DSO_3 x2.  NOT a weight -- a scenario definition.  ``make_config``
        # leaves both multipliers empty, but the chapter-9 dead-band studies
        # (analysis/deadband_selection.py, deadband_2d.py, deadband_n1.py) all
        # ADMIT only runs carrying {"DSO_3": 2.0} on BOTH, and
        # helpers/rms_cosim_config.py declares them as its defaults.  The runner
        # warns that a scaled run is "NOT comparable with an unscaled run", so
        # chapter 10 has to match chapter 9 or the two chapters describe
        # different networks.  Adopted 2026-09-08.
        dso_der_scale={"DSO_3": 2.0},
        dso_load_p_scale={"DSO_3": 2.0},
    )


def fired_contingencies(cfg: MultiTSOConfig) -> tuple:
    """(fired, skipped) contingency descriptions for the chosen horizon."""
    end = cfg.n_total_s / 60.0
    f = [f"{e.minute}:{e.element_type}/{e.action}" for e in cfg.contingencies
         if e.minute <= end]
    s_ = [f"{e.minute}:{e.element_type}/{e.action}" for e in cfg.contingencies
          if e.minute > end]
    return f, s_


#: Pilot buses for S1, selected on the base case by
#: ``experiments/CIGRE_2026/008_PILOT_BUS_SELECT.py`` with CONFIG_SOURCE =
#: "thesis", and held fixed thereafter as classical SVR does.
#:
#: Regenerated against the validated config on 2026-09-02 and found IDENTICAL
#: to the paper-config selection, scores included to five decimals.  The two
#: structural differences do not perturb the base-case Jacobian: the MSC/MSR
#: banks sit at step 0 and contribute no admittance, and the Thevenin boundary
#: equivalent is a controller-side model, not a plant change.
#:
#: Zone 2 is a NEAR-TIE -- bus 7 scores 0.17914 against bus 6's 0.17832, a 0.5 %
#: margin, and the two swap between installed-capacity scenarios.  Read nothing
#: into which bus zone 2's pilot is.
PILOT_BUSES: Dict[int, int] = {1: 25, 2: 7, 3: 20}


#: The ladder.  Names are the thesis names; there is no V1..V5 here.
#: Switched shunts OFF for every rung up to and including O1.
#:
#: This is not a free choice -- it repairs a confound.  The ShuntIntegrator
#: skips any zone whose TSOController has no ``_last_measurement``
#: (multi_tso_dso.py: "if meas_i is None: continue"), and that attribute is set
#: only inside ``TSOController.step`` (tso_controller.py:2623), i.e. only when
#: the TSO **OFO** runs.  So on the validated config the banks are inert in L1,
#: L2 and S1 -- which have no TSO OFO -- but LIVE in O1, which does.  Left
#: alone, the S1->O1 step would change the control law AND switch on a discrete
#: actuator class, and it is the one step the chapter's argument rests on.
#:
#: Turning them off outright (rather than installing them and leaving them
#: unswitched) keeps the plant identical: banks sit at step 0 and contribute no
#: admittance, which is why the pilot-node selection came out bit-identical with
#: and without them.  Note ``shunt_dispatch='off'`` ALONE is not enough --
#: the runner promotes off+installed to 'miqp', which would put the banks in the
#: integer block.  Both fields must be set.
_NO_SHUNTS = dict(install_tso_tertiary_shunts=False, shunt_dispatch="off")

#: MACHINE 2W TAPS HELD ACROSS THE WHOLE LOWER LADDER (L1, L2, S1, O1).
#:
#: Added 2026-09-03.  Until then only S1/O1 held them, because a tap relay on
#: the HV side of a machine transformer fights the AVR on its LV side.  L1/L2
#: kept the relay and logged 57 and 54 tap operations against S1/O1's ZERO --
#: so L2->S1 was not "hierarchical coordination", it was "lose transformer tap
#: control", and the +6.9 % it showed was largely that.
#:
#: The COUPLER 3W taps are deliberately NOT frozen: they carry no AVR, so
#: nothing fights them, and they stay under DiscreteTapControl in every variant
#: whose DSO layer is local (installed in Phase 2 of the runner, dropped only
#: when ``not _local_dso or _central``).  Freezing those too would strip a
#: discrete layer that real systems do run.
#:
#: Net effect: L1, L2, S1 and O1 now share one discrete treatment -- 2W held,
#: 3W under the local relay -- so every step below O1 is one-mechanism.

VARIANTS: Dict[str, Dict[str, Any]] = {
    "L1": dict(
        tso_mode="local", tso_local_mode="qv",
        dso_mode="local", local_der_mode="cos_phi_1",
        tso_q_mode="qv", dso_q_mode="cosphi",
        tso_oltc_mode="local",
        **_NO_SHUNTS,
    ),
    "L2": dict(
        tso_mode="local", tso_local_mode="qv",
        dso_mode="local", local_der_mode="qv",
        tso_q_mode="qv", dso_q_mode="qv",
        tso_oltc_mode="local",
        **_NO_SHUNTS,
    ),
    "S1": dict(
        tso_mode="svr",
        dso_mode="local", local_der_mode="qv",
        tso_q_mode="qv", dso_q_mode="qv",
        tso_oltc_mode="local",
        svr_pilot_buses=PILOT_BUSES,
        svr_t_rvr_s=300.0,
        svr_t_rpr_s=60.0,
        # PI on BOTH loops (RVR outer, RPR inner).  k_p = 0.2 is INHERITED from
        # the 42-min screen, NOT tuned here -- S1 is the reference, not the
        # contribution.  Confirmed on the full horizon 2026-09-04/05: with the
        # guard on, PI improves every metric, and at the 210-min line trip the
        # guard cuts the peak 34 % (0.02492 -> 0.01638) against guard-off.
        # Adopted as the ladder's S1 on 2026-09-05; the stored S1 log is the
        # matching PI+guard run.  Changing either without the other re-creates
        # the code/results desync found on 2026-09-04.
        svr_k_p_rvr=0.2,
        svr_k_p_rpr=0.2,
        # Pilot-voltage priority prevents the inner Q regulator from
        # withdrawing voltage support between regional dispatches.
        # Explicit extension of classical SVR; disable to replay its law.
        svr_rpr_voltage_priority=True,
        # NO supervisory dead band.  The 0.01 pu dead band elsewhere in the
        # config belongs to the LOCAL DROOP layer, which sits beneath both
        # schemes; the OFO's supervisory tracking has no insensitive region, so
        # giving one to the RVR would freeze its integrator for most of the
        # horizon.  Zero is also generous to the reference: a deployed SVR
        # carries a dead band and would do worse.
        svr_deadband_pu=0.0,
        **_NO_SHUNTS,
    ),
    "O1": dict(
        tso_mode="ofo",
        dso_mode="local", local_der_mode="qv",
        tso_q_mode="qv", dso_q_mode="qv",
        tso_oltc_mode="local",     # <- with _NO_SHUNTS, the discrete difference from O2
        **_NO_SHUNTS,
    ),
    "O2": dict(
        tso_mode="ofo",
        dso_mode="local", local_der_mode="qv",
        tso_q_mode="qv", dso_q_mode="qv",
    ),
    "O3": dict(
        tso_mode="ofo",
        dso_mode="ofo",
        # coordination_mode is deliberately NOT set: O3 inherits ``sbx_h`` from
        # make_config, exactly as O2 does.  Until 2026-09-08 it forced "none",
        # so O2->O3 added the DSO OFO *and* removed horizontal boundary
        # exchange in one move -- two mechanisms of opposite sign in one step.
        # Measured over 116 MC scenarios under that setting the step was only
        # -9.7 % on e_v, against -23 % for O1->O2.  With coordination held the
        # step isolates the DSO layer, which is what the rung is meant to test.
    ),
    "M": dict(
        control_scope="central",
        tso_mode="ofo",
        dso_mode="local",
        tso_q_mode="qv", dso_q_mode="qv",
        local_sensitivities_tso=False,
        local_sensitivities_dso=False,
        # central_period_s=None -> the controller fires EVERY simulation step.
        #
        # This must not be set to tso_period_s.  M replaces the fast STS-OFO
        # layer as well as the slow TS one, so pinning it to 180 s while O3
        # keeps a 20 s subordinate layer makes the "upper bound" nine times
        # slower than the scheme it is supposed to bound.  Measured 2026-09-02
        # with central_period_s=180: O3 beat M by 7 % aggregate, which said
        # nothing about decomposition and everything about the cadence.
        # MultiTSOConfig.central_period_s calls None "the correct best-case
        # cadence" for exactly this reason.
        central_period_s=None,
    ),
}

LADDER = ["L1", "L2", "S1", "O1", "O2", "O3", "M"]

#: What each step is allowed to change.  Enforced, not documented -- an
#: override added to one rung and not its neighbour silently turns a
#: one-mechanism step into a comparison of two unrelated configurations, and
#: nothing else in the pipeline would notice.
STEP_ALLOWED: Dict[str, set] = {
    "L1->L2": {"local_der_mode", "dso_q_mode"},
    # tso_oltc_mode is NO LONGER listed: since 2026-09-03 L1/L2 also hold the
    # machine 2W taps, so the whole lower ladder shares one discrete treatment
    # and this step is genuinely one-mechanism.  Do not re-add it.
    "L2->S1": {"tso_mode", "tso_local_mode", "svr_pilot_buses",
               "svr_t_rvr_s", "svr_t_rpr_s", "svr_k_p_rvr", "svr_k_p_rpr",
               "svr_rvr_p_every_step",
               "svr_deadband_pu",
               "svr_rpr_voltage_priority"},
    "S1->O1": {"tso_mode", "svr_pilot_buses", "svr_t_rvr_s", "svr_t_rpr_s",
                "svr_k_p_rvr", "svr_k_p_rpr", "svr_rvr_p_every_step", "svr_deadband_pu",
               "svr_rpr_voltage_priority"},
    # O1->O2 is "TS discrete devices enter the loop": the OLTCs join the
    # MIQP integer block AND the switched shunts become dispatched by the
    # ShuntIntegrator.  Two device classes, ONE mechanism -- discrete
    # actuation moves from planning into the control loop.  Splitting it
    # would need an eighth rung for little argumentative return.
    "O1->O2": {"tso_oltc_mode", "install_tso_tertiary_shunts",
               "shunt_dispatch"},
    "O2->O3": {"dso_mode", "local_der_mode", "dso_q_mode", "coordination_mode"},
    "O3->M": None,      # not a controlled step; no constraint asserted
}


def _resolved(name: str) -> MultiTSOConfig:
    """The config the runner actually receives for a variant."""
    cfg = make_thesis_config()
    for k, v in VARIANTS[name].items():
        setattr(cfg, k, v)
    if G_W_GEN is not None:
        cfg.g_w_gen = float(G_W_GEN)
    return cfg


def assert_ladder_controlled() -> None:
    """Fail loudly if a step changes more than it is allowed to.

    Compares RESOLVED configs, not the override dicts.  Comparing the dicts
    reports a difference whenever one rung states a value explicitly and its
    neighbour inherits the identical default -- e.g. O2 sets tso_q_mode='qv'
    and O3 omits it, but the default IS 'qv', so nothing differs where it
    matters.  Only what reaches the runner counts.
    """
    problems: List[str] = []
    for a, b in zip(LADDER, LADDER[1:]):
        key = f"{a}->{b}"
        allowed = STEP_ALLOWED.get(key)
        if allowed is None:
            continue
        ca, cb = _resolved(a), _resolved(b)
        for f in dataclasses.fields(MultiTSOConfig):
            k = f.name
            if k in allowed or k in ("result_dir", "verbose"):
                continue
            va, vb = getattr(ca, k, None), getattr(cb, k, None)
            try:
                same = bool(va == vb)
            except Exception:
                same = repr(va) == repr(vb)
            if not same:
                problems.append(f"  {key}: {k}  {a}={va!r}  {b}={vb!r}")
    if problems:
        raise SystemExit(
            "Ladder is NOT controlled -- these steps change more than one "
            "mechanism:\n" + "\n".join(problems)
            + "\n\nMirror the change in both rungs, or widen STEP_ALLOWED with "
              "a reason if the extra difference is intended.")
    print("guard: every ladder step changes only what it is allowed to")


# ---------------------------------------------------------------------------
#  Run
# ---------------------------------------------------------------------------


def run_dir(create: bool = False) -> Path:
    base = Path(RESULTS_ROOT) / EXPERIMENT
    if create:
        base.mkdir(parents=True, exist_ok=True)
        n = 1 + max([int(p.name.split("_")[0]) for p in base.iterdir()
                     if p.is_dir() and p.name.split("_")[0].isdigit()] or [0])
        d = base / f"{n:04d}"
        d.mkdir(exist_ok=True)
        return d
    runs = sorted((p for p in base.iterdir() if p.is_dir()), reverse=True) \
        if base.is_dir() else []
    return runs[0] if runs else base


def run_one(name: str, out: Path) -> List:
    cfg = make_thesis_config()
    for k, v in VARIANTS[name].items():
        setattr(cfg, k, v)
    if G_W_GEN is not None:
        cfg.g_w_gen = float(G_W_GEN)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    cfg.result_dir = str(d)
    print("\n" + "=" * 72)
    print(f"  {name}   tso={cfg.tso_mode}  dso={cfg.dso_mode}  "
          f"oltc={getattr(cfg, 'tso_oltc_mode', 'ofo')}  "
          f"g_w_gen={cfg.g_w_gen:.3g}  shunts={cfg.shunt_dispatch}")
    print("=" * 72, flush=True)
    try:
        log = run_multi_tso_dso(cfg)
    except Exception as exc:  # keep the sweep alive, record the failure
        print(f"  [{name}] FAILED: {type(exc).__name__}: {exc}", flush=True)
        log = []
    with open(d / "log.pkl", "wb") as f:
        pickle.dump(log, f)
    print(f"  [{name}] wrote {len(log)} records", flush=True)
    return log


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------


def _rms(v) -> float:
    a = np.asarray([x for x in v if x is not None], float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")


def e_v_area(log, z) -> float:
    return _rms([r.zone_v_rms_err_pu.get(z, np.nan) for r in log
                 if getattr(r, "zone_v_rms_err_pu", None)])


def e_v_pilot(log, bus, ref) -> float:
    return _rms([float(r.bus_vm_pu[bus]) - ref for r in log
                 if getattr(r, "bus_vm_pu", None) and bus in r.bus_vm_pu])


def avr_travel(log, z) -> float:
    vg = np.array([r.zone_v_gen[z] for r in log
                   if z in getattr(r, "zone_v_gen", {}) and len(r.zone_v_gen[z])])
    return float(vg.max() - vg.min()) if vg.size else float("nan")


def tap_ops(log) -> int:
    n, prev = 0, {}
    for r in log:
        for z, t in (getattr(r, "zone_oltc_taps", {}) or {}).items():
            t = np.asarray(t, float)
            if z in prev and prev[z].shape == t.shape:
                n += int(np.sum(np.abs(t - prev[z]) > 0.5))
            prev[z] = t
    return n


def report(logs: Dict[str, List], cfg: MultiTSOConfig) -> None:
    zones = sorted(PILOT_BUSES)
    refs = {z: float(cfg.zone_v_setpoints_pu.get(z, cfg.v_setpoint_pu))
            if cfg.zone_v_setpoints_pu else float(cfg.v_setpoint_pu)
            for z in zones}
    names = [n for n in LADDER if logs.get(n)]
    if not names:
        print("!! no logs -- run without --report first")
        return

    print("\n" + "=" * 74)
    print(f"  Dissertation variant ladder -- validated weights, {HORIZON_MIN} min")
    print("=" * 74)

    for title, fn in (
        ("PRIMARY  e_v,a [pu]  time-mean spatial RMS over R_a",
         lambda n, z: e_v_area(logs[n], z)),
        ("SECONDARY  e_v,p,a [pu]  pilot bus only",
         lambda n, z: e_v_pilot(logs[n], PILOT_BUSES[z], refs[z])),
        ("AVR setpoint travel [pu]",
         lambda n, z: avr_travel(logs[n], z)),
    ):
        print(f"\n-- {title} " + "-" * max(0, 50 - len(title)))
        print(f"{'':>5}" + "".join(f"{'z'+str(z):>10}" for z in zones) + f"{'all':>10}")
        for n in names:
            v = [fn(n, z) for z in zones]
            agg = _rms(v) if "travel" not in title else float(np.nanmean(v))
            print(f"{n:>5}" + "".join(f"{x:>10.5f}" for x in v) + f"{agg:>10.5f}")

    print(f"\n-- surrogate ratio  e_v,a / e_v,p,a " + "-" * 22)
    print(f"{'':>5}" + "".join(f"{'z'+str(z):>10}" for z in zones))
    for n in names:
        row = []
        for z in zones:
            ep = e_v_pilot(logs[n], PILOT_BUSES[z], refs[z])
            row.append(e_v_area(logs[n], z) / ep if ep > 0 else float("nan"))
        print(f"{n:>5}" + "".join(f"{x:>10.2f}" for x in row))

    print(f"\n-- TS tap operations " + "-" * 37)
    for n in names:
        print(f"{n:>5}{tap_ops(logs[n]):>10d}")

    print("\nSteps: L1->L2 STS support | L2->S1 hierarchical coordination |")
    print("S1->O1 THE CONTROL LAW | O1->O2 discrete co-optimisation |")
    print("O2->O3 the cascade | O3->M decomposition loss (not controlled).")


def main() -> None:
    global HORIZON_MIN, G_W_GEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated variant names")
    ap.add_argument("--report", action="store_true", help="report newest run")
    ap.add_argument("--g-w-gen", type=float, default=None,
                    help="override g_w_gen for EVERY variant (sweep axis); "
                         "default inherits make_config")
    ap.add_argument("--run-dir", default=None,
                    help="reuse an existing run directory instead of "
                         "creating one (resume a partial sweep)")
    ap.add_argument("--minutes", type=int, default=HORIZON_MIN,
                    help=f"horizon in minutes (default {HORIZON_MIN}); "
                         "shorter truncates the contingency schedule")
    args = ap.parse_args()
    HORIZON_MIN = int(args.minutes)
    G_W_GEN = args.g_w_gen

    assert_ladder_controlled()
    cfg = make_thesis_config()
    # The banner must show what the RUNS use, not what make_config holds --
    # otherwise a sweep's log claims the inherited gauge while every variant
    # ran on the override.
    if G_W_GEN is not None:
        cfg.g_w_gen = float(G_W_GEN)
    fired, skipped = fired_contingencies(cfg)
    print(f"horizon: {HORIZON_MIN} min | contingencies fired: {len(fired)} "
          f"{fired}")
    if skipped:
        print(f"         NOT reached (horizon too short): {skipped}")
    print(f"config: g_w_gen={cfg.g_w_gen:.3g}  g_w_der={cfg.g_w_der:.4g}  "
          f"g_q={cfg.g_q:g}  shunts={cfg.shunt_dispatch}  "
          f"boundary={cfg.tie_boundary_equivalent}  scenario={cfg.scenario}")

    if args.report:
        # --run-dir must be honoured here too, not just when running.  Without
        # it a report silently reads the NEWEST run directory, which during a
        # chained sweep is the one still being written.
        d = Path(args.run_dir) if args.run_dir else run_dir()
        print(f"reading: {d}")
        logs = {}
        for n in LADDER:
            p = d / n / "log.pkl"
            if p.exists():
                with open(p, "rb") as f:
                    logs[n] = pickle.load(f)
        report(logs, cfg)
        return

    sel = [s.strip() for s in args.only.split(",")] if args.only else list(LADDER)
    bad = [s for s in sel if s not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad}; valid {LADDER}")

    out = Path(args.run_dir) if args.run_dir else run_dir(create=True)
    out.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {out}")
    logs = {n: run_one(n, out) for n in sel}
    report(logs, cfg)


if __name__ == "__main__":
    main()
