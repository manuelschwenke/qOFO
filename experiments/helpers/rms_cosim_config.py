"""Shared configuration and CLI for the PowerFactory RMS co-simulation.

Two entry points build on this:

* ``experiments/run_rms_cosim.py`` -- runs the RMS co-simulation alone.
* ``experiments/run_comparison_rms_cosim_qss.py`` -- runs the RMS
  co-simulation *and* the quasi-static (QSS) reference, and compares them.

Both drive :func:`experiments.runners.run_multi_tso_dso`; the comparison
script simply runs it twice, once per plant.  Note that the two runs are
independent closed loops -- each plant feeds its own controller stack -- so
the comparison measures closed-loop equivalence, not open-loop plant
equivalence.  The open-loop ``u -> y`` test is
``experiments/run_rms_openloop_uy.py``.

(The former name ``run_rms_phase6_replay`` was a misnomer: nothing is
replayed there.  A thin deprecation shim keeps the old path working.)

Author: Manuel Schwenke / Claude Code (2026-07-31)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional

from configs.config import MultiTSOConfig

#: Dispatch cadence [s]: STS/DSO period and the runner's integration step.
DT_S = 20.0
#: TS/TSO dispatch period [s].
TSO_PERIOD_S = 180.0

#: Per-DSO scenario multipliers applied unless ``--symmetric-dso`` is given.
#: DSO_3 is the reinforced stress case of
#: ``docs/daily_log/07_2026/2026-07-30_ieee39_dso_powerfactory_sync_handover.md``:
#: 2x installed DER (700 -> 1400 MW) and 2x active-load base
#: (261.80 -> 523.61 MW).  Reactive load is deliberately NOT scaled.
DEFAULT_DSO_DER_SCALE: Dict[str, float] = {"DSO_3": 2.0}
DEFAULT_DSO_LOAD_P_SCALE: Dict[str, float] = {"DSO_3": 2.0}


@dataclass
class CoSimSpecification:
    """Experiment-level provenance stored by ``new_run_dir``."""

    runner_static: Optional[MultiTSOConfig]
    runner_rms: MultiTSOConfig
    plant_static: str = "PandapowerStaticPlant"
    plant_rms: str = "PowerFactoryPlant"
    exogenous_input: str = "SimBench profiles via pre-ComInc ElmFile sources"
    comparison: str = "closed-loop; each plant feeds its own controller run"
    configuration_source: str = "experiments.run_multi_system_ofo.make_config"
    settling_band_relative: float = 0.02
    settling_q_floor_mvar: float = 1.0
    settling_voltage_floor_pu: float = 1e-3

    #: RMS solver settings, recorded because they change the numerics and are
    #: NOT recoverable from the trace alone.  Runs before 2026-08-06 all used
    #: a fixed 10 ms step; mixing them with adaptive-step runs in one analysis
    #: would compare different integrations of the same model.
    rms_step_ms: float = 10.0
    rms_step_max_ms: Optional[float] = None
    adaptive_step: bool = False

    #: Dead-band x droop study only: Q(V) dead band [pu] installed on the RMS
    #: parks at the contingency, the run-up keeping the configured one.  Lets
    #: two legs differ ONLY in what the local layer does after the
    #: disturbance.  A run with this set is NOT Gate-E certifiable: the static
    #: plant keeps its configured Q(V) throughout.
    qv_deadband_at_contingency: Optional[float] = None


def make_cosim_config(duration_s: float, *, verbose: int) -> MultiTSOConfig:
    """Reference multi-system config inside the co-simulation envelope.

    Controller weights, coordination, actuator installation and dispatch are
    inherited from :func:`experiments.run_multi_system_ofo.make_config`.
    Only the experimental horizon and the sources of exogenous evolution are
    overridden, so both plants face the same controller stack.
    """
    from experiments.run_multi_system_ofo import make_config

    cfg = make_config()
    cfg.n_total_s = float(duration_s)
    cfg.tso_period_s = TSO_PERIOD_S
    cfg.dso_period_s = DT_S
    cfg.dt_s = DT_S
    cfg.use_profiles = True
    # TEMPORARY (2026-07-21): P-independent +-1.0 pu DER capability.  Set to
    # None with --physical-capability for the real VDE / STATCOM diagrams.
    # NOT a physical model: +-1.0 pu at rated P implies S = 1.41 pu.  Any
    # published result must use --physical-capability.
    cfg.der_q_capability_override_pu = 1.0
    cfg.contingencies = []
    cfg.measurement_noise.enabled = False
    cfg.enable_reachability_guard = False
    cfg.live_plot_controller = False
    cfg.live_plot_cascade = False
    cfg.live_plot_system = False
    cfg.live_plot_tracking = False
    cfg.live_plot_sbx = False
    cfg.verbose = int(verbose)
    return cfg


def validate_duration(duration_s: float) -> None:
    if duration_s <= 0.0:
        raise ValueError("duration must be positive")
    steps = duration_s / DT_S
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError(f"duration must be a multiple of {DT_S:g} s")


def parse_dso_map(values, cast, what: str) -> Dict[str, float]:
    """Parse repeatable ``DSO_x=value`` flags into a dict."""
    out: Dict[str, float] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--{what} expects 'DSO_x=value', got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        if not key.startswith("DSO_"):
            raise SystemExit(
                f"--{what}: {key!r} is not a DSO id (expected DSO_1..4)")
        try:
            out[key] = cast(raw)
        except ValueError:
            raise SystemExit(f"--{what}: {raw!r} is not a number") from None
    return out


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """CLI shared by both entry points."""
    parser.add_argument("--duration", type=float, default=3600.0,
                        help="horizon [s]; must be a multiple of %g" % DT_S)
    parser.add_argument("--stride", type=int, default=10,
                        help="RMS result-row stride (10 = 0.1 s at the 10 ms "
                             "RMS step)")
    parser.add_argument("--project", default=None,
                        help="PowerFactory project path")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--show-gui", action=argparse.BooleanOptionalAction, default=False,
        help="raise the PowerFactory desktop at startup. Default off: the run "
             "starts hidden (App.Show() costs 20-30 s and the per-interval "
             "redraw dominates once results accumulate). show_gui.bat in the "
             "run directory raises the window mid-run either way.")
    parser.add_argument("--gui-refresh-every", type=int, default=1)
    parser.add_argument("--live-plot", action="store_true",
                        help="Python-side live plot of the TS bus voltages")
    parser.add_argument(
        "--profiles", action=argparse.BooleanOptionalAction, default=True,
        help="time-series load/DER profiles on both plants. Forces "
             "use_zonal_gen_dispatch=False (the machine-P schedule writes "
             "net.gen directly and is not wired to PF).")
    parser.add_argument("--profile-delivery", choices=("elmfile", "events"),
                        default="elmfile")
    parser.add_argument("--profile-settle", type=float, default=0.0,
                        help="seconds the RMS plant pre-settles after each "
                             "profile step so controllers read post-profile")
    parser.add_argument("--scenario", default="rural_700",
                        help="'base_410' or 'rural_700'. Results from "
                             "different scenarios are NOT comparable.")
    parser.add_argument("--start-time", default="2016-01-05 08:00",
                        help="profile start instant, 'YYYY-MM-DD HH:MM'")
    parser.add_argument("--der-deadband", type=float, default=0.005,
                        help="DER Q(V) deadband [pu] on BOTH plants and BOTH "
                             "voltage levels (nominal 0.01).  Sets the blanket "
                             "override as well, so it pins every park to one "
                             "value; use --tso-deadband/--dso-deadband to vary "
                             "the levels independently.")
    parser.add_argument("--tso-deadband", type=float, default=None,
                        help="TS-connected DER Q(V) deadband [pu].  Overrides "
                             "--der-deadband for the TS population and clears "
                             "the blanket override so the two levels can "
                             "differ.")
    parser.add_argument("--dso-deadband", type=float, default=None,
                        help="DS-connected DER Q(V) deadband [pu].  See "
                             "--tso-deadband.")
    parser.add_argument("--der-slope", type=float, default=None,
                        help="DER Q(V) DROOP [pu] on both plants and both "
                             "levels. Despite the name it is the droop, not a "
                             "gain: it divides the voltage error (static "
                             "R = S_n/slope, RMS Kdroop = 1/slope), so 0.06 "
                             "means 0.06 pu of deviation commands full rated "
                             "Q -- a 6 %% droop. The grid code permits 5-15 %%. "
                             "Smaller = steeper.")
    parser.add_argument("--tso-slope", type=float, default=None,
                        help="TS-connected DER droop [pu]; overrides "
                             "--der-slope for the TS population")
    parser.add_argument("--dso-slope", type=float, default=None,
                        help="DS-connected DER droop [pu]; see --tso-slope")
    parser.add_argument(
        "--physical-capability", action="store_true",
        help="revert the TEMPORARY +-1.0 pu DER capability override and use "
             "the real VDE-AR-N-4120 / STATCOM diagrams. REQUIRED for any "
             "physically meaningful result.")
    parser.add_argument("--dso-oltc-switch-cost", type=float, default=None,
                        help="override g_w_dso_oltc (make_config default 150)")
    parser.add_argument("--load-step-time", type=float, default=None,
                        help="simulation time [s] of an exogenous load step "
                             "applied to BOTH plants. Omitted = no step. Not a "
                             "contingency: it perturbs the profile frame, so it "
                             "reaches the RMS plant through the supported "
                             "EvtLod path")
    parser.add_argument("--load-step-factor", type=float, default=1.0,
                        help="multiplier for the stepped active-load profiles "
                             "(e.g. 1.1 = +10%%, 1.25 = +25%%)")
    parser.add_argument("--load-step-bus", type=int, default=None,
                        help="apply a LOCALISED additive step at this bus "
                             "instead of scaling every profile column. Far "
                             "more efficient per MW: 43 MW at bus 119 induces "
                             "the same 0.02 pu DER deviation that needs ~3900 "
                             "MW system-wide")
    parser.add_argument("--load-step-delta-mw", type=float, default=0.0,
                        help="additive step [MW] at --load-step-bus")
    parser.add_argument("--trip-gen", type=int, default=None,
                        help="pandapower net.gen index to TRIP -- a real N-1 "
                             "contingency, delivered to the static plant by "
                             "mutating the net and to the RMS plant as an "
                             "EvtOutage. Far stronger per unit of realism than "
                             "a load step: measured at 2016-01-05 08:00, "
                             "tripping gen 7 moves the worst park 0.0104 pu "
                             "and gen 2 moves it 0.1025 pu, against 0.0035 pu "
                             "for a +400 MW load step. The impact is driven by "
                             "the LOST AVR VOLTAGE SUPPORT, not by the "
                             "machine's MW: gen 7 is the largest unit (830 MW) "
                             "and the weakest disturbance")
    parser.add_argument("--trip-time", type=float, default=100.0,
                        help="simulation time [s] of the --trip-gen outage "
                             "or the --q-step-bus load step")
    parser.add_argument("--q-step-bus", type=int, default=None,
                        help="step the REACTIVE power of the (single, "
                             "in-service) load at this pandapower bus by "
                             "--q-step-mvar, at --trip-time. Delivered to the "
                             "static plant by mutating net.load.q_mvar and to "
                             "the RMS plant as an EvtLod. Unlike --trip-gen it "
                             "leaves the topology unchanged, so Gate E stays "
                             "meaningful. Sized 2026-08-06 at bus 7: +200 Mvar "
                             "moves the worst TS park 0.025 pu, 2.5x a 0.01 pu "
                             "dead band, with no bus below 0.96 pu.")
    parser.add_argument("--q-step-mvar", type=float, default=None,
                        help="reactive step [Mvar] for --q-step-bus")
    parser.add_argument("--no-qv-seed", action="store_true",
                        help="disable seed_qv_equilibrium on the STATIC plant")
    parser.add_argument("--seed-der-anchor", action="store_true",
                        help="initialise every DER Q(V) anchor to its local "
                             "res_bus.vm_pu at init (both plants)")
    # -- per-DSO scenario multipliers ------------------------------------
    parser.add_argument("--dso-der-scale", action="append", default=None,
                        metavar="DSO=F",
                        help=f"per-DSO installed-DER multiplier, repeatable. "
                             f"Default {DEFAULT_DSO_DER_SCALE}.")
    parser.add_argument("--dso-load-p-scale", action="append", default=None,
                        metavar="DSO=F",
                        help=f"per-DSO active-load multiplier, repeatable. "
                             f"Default {DEFAULT_DSO_LOAD_P_SCALE}.")
    parser.add_argument("--dso-load-q-base", action="append", default=None,
                        metavar="DSO=MVAR",
                        help="per-DSO aggregate reactive-load profile base")
    parser.add_argument(
        "--symmetric-dso", action="store_true",
        help="drop the default DSO_3 multipliers and run all four DSOs "
             "identically.")


def apply_cli_overrides(args, cfgs) -> None:
    """Apply every common CLI flag to each config in ``cfgs``."""
    from datetime import datetime

    # Deadband: one number for both levels by default, but a 2D
    # delta_TS x delta_DS study needs them independent.  The blanket override
    # is a single scalar applied to every park, so it must be CLEARED whenever
    # the levels are set separately -- otherwise it wins in pf.plant and both
    # levels silently collapse onto one value.
    _db_split = args.tso_deadband is not None or args.dso_deadband is not None
    _db_ts = float(args.tso_deadband if args.tso_deadband is not None
                   else args.der_deadband)
    _db_ds = float(args.dso_deadband if args.dso_deadband is not None
                   else args.der_deadband)

    for cfg in cfgs:
        cfg.use_profiles = bool(args.profiles)
        # Never dispatched: the zonal machine-P schedule writes net.gen
        # directly and the PF plant cannot follow it.
        cfg.use_zonal_gen_dispatch = False
        if args.profiles:
            cfg.rms_profile_settle_s = float(args.profile_settle)
        cfg.scenario = str(args.scenario)
        cfg.start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M")
        cfg.tso_qv_deadband_pu = _db_ts
        cfg.dso_qv_deadband_pu = _db_ds
        cfg.der_qv_deadband_override_pu = (
            None if _db_split else float(args.der_deadband))
        # Droop, resolved per level like the dead band.  Unlike the dead band
        # there is no blanket-override field to clear: the RMS plant reads the
        # per-sgen droop map published by the runner (see
        # core.actuator_bounds.set_der_qv_slope_by_sgen), which is why
        # --der-slope alone used to move only the static plant.
        if args.der_slope is not None:
            cfg.tso_qv_slope_pu = float(args.der_slope)
            cfg.dso_qv_slope_pu = float(args.der_slope)
        if args.tso_slope is not None:
            cfg.tso_qv_slope_pu = float(args.tso_slope)
        if args.dso_slope is not None:
            cfg.dso_qv_slope_pu = float(args.dso_slope)
        if args.physical_capability:
            cfg.der_q_capability_override_pu = None
        if args.dso_oltc_switch_cost is not None:
            cfg.g_w_dso_oltc = float(args.dso_oltc_switch_cost)
        if args.seed_der_anchor:
            cfg.seed_der_anchor_to_local_v = True
        if args.trip_gen is not None:
            from experiments.helpers.records import ContingencyEvent
            # time_s overrides `minute`, so the trip lands on the exact second
            # rather than a minute boundary.
            cfg.contingencies = [ContingencyEvent(
                minute=0, time_s=float(args.trip_time),
                element_type="gen", element_index=int(args.trip_gen),
                action="trip")]
        if args.q_step_bus is not None:
            if args.q_step_mvar is None:
                raise SystemExit("--q-step-bus needs --q-step-mvar")
            if args.trip_gen is not None:
                raise SystemExit(
                    "--q-step-bus and --trip-gen are both contingencies at "
                    "--trip-time; run them as separate experiments")
            from experiments.helpers.records import ContingencyEvent
            # element_index is resolved from the bus by
            # prepare_load_contingencies (the IEEE 39 loads carry no names).
            cfg.contingencies = [ContingencyEvent(
                minute=0, time_s=float(args.trip_time),
                element_type="load", action="q_step",
                bus=int(args.q_step_bus), p_mw=0.0,
                q_mvar=float(args.q_step_mvar))]
        if args.load_step_time is not None:
            cfg.load_step_time_s = float(args.load_step_time)
            cfg.load_step_factor = float(args.load_step_factor)
            if args.load_step_bus is not None:
                cfg.load_step_bus = int(args.load_step_bus)
                cfg.load_step_delta_mw = float(args.load_step_delta_mw)

    if args.symmetric_dso:
        der_scale: Dict[str, float] = {}
        load_p: Dict[str, float] = {}
    else:
        der_scale = (parse_dso_map(args.dso_der_scale, float, "dso-der-scale")
                     or dict(DEFAULT_DSO_DER_SCALE))
        load_p = (parse_dso_map(args.dso_load_p_scale, float,
                                "dso-load-p-scale")
                  or dict(DEFAULT_DSO_LOAD_P_SCALE))
    load_q = parse_dso_map(args.dso_load_q_base, float, "dso-load-q-base")
    for cfg in cfgs:
        cfg.dso_der_scale = dict(der_scale)
        cfg.dso_load_p_scale = dict(load_p)
        cfg.dso_load_q_profile_base_mvar = dict(load_q)

    print(f"  [scenario] {args.scenario} @ {args.start_time}; "
          f"profiles={'on' if args.profiles else 'OFF'}; "
          f"deadband: TS={_db_ts:g} DS={_db_ds:g} pu"
          + ("  (per-level, blanket override cleared)" if _db_split else ""))
    print(f"  [dso-override] der_scale={der_scale or '{}'} "
          f"load_p_scale={load_p or '{}'}"
          + ("  (symmetric)" if args.symmetric_dso else ""))
    if not args.physical_capability:
        print("  [!] DER capability OVERRIDDEN to +-1.0 pu -- diagnostic "
              "only, NOT a physical model. Use --physical-capability for "
              "publishable results.")


# ── backwards-compatible aliases (old names used by existing imports) ──
ReplaySpecification = CoSimSpecification
make_gate_e_config = make_cosim_config
_validate_duration = validate_duration
