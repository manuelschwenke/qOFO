#!/usr/bin/env python3
"""Open-loop u -> y plant-equivalence test (Gate E, 2026-07-22).

The standard Gate-E replay runs two *independent* closed loops -- each plant
feeds its own controller stack -- so a divergence conflates plant fidelity
with the controllers deciding differently.  This test removes the second
factor: it captures the exact actuator + profile timeline the STATIC run
produced, then replays it verbatim to the RMS plant and compares ``y``.

Identical ``u`` (and identical exogenous profile) into both plants:

* if the outputs agree, the plant/load-application is validated and every
  observed closed-loop divergence (e.g. the DSO_4 coupler runaway) is purely
  the controllers deciding differently under a slower plant;
* if they disagree, a genuine plant-model difference remains.

Both plants act on the same pandapower index namespace (the whole point of
``core.plant.Plant``), so a captured ``ActuatorWrites`` applies unchanged to
the RMS plant.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.plant import PandapowerStaticPlant  # noqa: E402
from experiments.helpers.rms_replay import (  # noqa: E402
    endpoint_comparison,
    static_controlled_trajectories,
    rms_controlled_trajectories,
    trajectories_long_frame,
)
from experiments.results_io import new_run_dir  # noqa: E402
from experiments.helpers.rms_cosim_config import (  # noqa: E402
    DT_S,
    CoSimSpecification as ReplaySpecification,
    make_cosim_config as make_gate_e_config,
)
from experiments.run_comparison_rms_cosim_qss import (  # noqa: E402
    _type_endpoint_summary,
)
from experiments.runners import run_multi_tso_dso  # noqa: E402
from pf.replay import PowerFactoryReplayFactory  # noqa: E402
from pf.session import DEFAULT_PROJECT_PATH  # noqa: E402


class CapturingStaticPlant(PandapowerStaticPlant):
    """Static plant that records every actuator / profile / advance call.

    Subclasses (not wraps) the static plant so the runner's
    ``isinstance(plant, PandapowerStaticPlant)`` branches still take the
    static path.  The recorded ``timeline`` is a flat, ordered list of the
    exogenous, ``apply_u`` and ``advance`` calls, which the RMS replay below
    re-issues one for one.
    """

    def __init__(self, net, **kw):
        super().__init__(net, **kw)
        self.timeline: List[Tuple] = []

    def apply_exogenous(self, profiles, t) -> None:
        self.timeline.append(("exogenous", profiles, t))
        super().apply_exogenous(profiles, t)

    def apply_u(self, writes) -> None:
        self.timeline.append(("apply_u", copy.deepcopy(writes)))
        super().apply_u(writes)

    def advance(self, duration_s: float) -> None:
        self.timeline.append(("advance", float(duration_s)))
        super().advance(duration_s)


class _CaptureFactory:
    """plant_factory that builds the capturing static plant and keeps the
    net/meta/zone_map so the RMS plant can be built from the same init."""

    def __init__(self):
        self.plant: CapturingStaticPlant | None = None
        self.net_init = None
        self.meta = None
        self.zone_map = None

    def __call__(self, net, *, meta, zone_map) -> CapturingStaticPlant:
        # Deep-copy the net at construction (post-init, pre-loop) so the RMS
        # plant starts from the identical operating point the static run did.
        self.net_init = copy.deepcopy(net)
        self.meta = meta
        self.zone_map = zone_map
        self.plant = CapturingStaticPlant(net)
        return self.plant


def _replay_to_rms(rms_plant, timeline, settle_s: float, dt_s: float) -> None:
    """Re-issue the captured timeline to the RMS plant, in order.

    The static plant emits TWO advance calls per interval when profiles are
    on -- the profile-branch advance (before control) and the end-of-step
    advance (after) -- both of which it treats as instant re-solves.  For the
    RMS plant each advance is real time, so they must be mapped to the same
    interval structure the closed-loop RMS run uses: a ``settle_s`` pre-settle
    that lets the profile events fire, then ``dt_s - settle_s`` after the
    control events, totalling ``dt_s`` per interval (clock unchanged).  Which
    of the two an advance is, is told by whether an ``apply_u`` has occurred
    since the last profile step.
    """
    n_u = n_adv = n_exo = 0
    seen_u = False
    for entry in timeline:
        kind = entry[0]
        if kind == "exogenous":
            rms_plant.apply_exogenous(entry[1], entry[2])
            seen_u = False
            n_exo += 1
        elif kind == "apply_u":
            rms_plant.apply_u(entry[1])
            seen_u = True
            n_u += 1
        elif kind == "advance":
            if not seen_u:
                rms_plant.advance(settle_s)          # profile pre-settle
            else:
                rms_plant.advance(dt_s - settle_s)   # rest of the interval
                seen_u = False
            rms_plant.read_y()
            n_adv += 1
    print(f"  [replay] issued {n_exo} profile, {n_u} apply_u, {n_adv} advance "
          f"calls to the RMS plant")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--profiles", action="store_true")
    ap.add_argument("--profile-settle", type=float, default=0.0)
    ap.add_argument("--project", default=DEFAULT_PROJECT_PATH)
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument(
        "--der-deadband", type=float, default=None,
        help="override the DER Q(V) deadband [pu] on BOTH plants (nominal 0.01). "
             "Lets the PLANT-LEVEL residual be measured as a function of the "
             "dead-zone width, so the closed-loop amplification factor can be "
             "separated from any plant-level deadband effect (2026-07-25).",
    )
    ap.add_argument(
        "--start-time", default=None,
        help="profile start instant, 'YYYY-MM-DD HH:MM' (selects the "
             "operating condition).",
    )
    ap.add_argument(
        "--scenario", default=None,
        help="network scenario: 'base_410' (410 MW installed DER per DSO) or "
             "'rural_700' (700 MW). Results from different scenarios are NOT "
             "comparable, so set this explicitly rather than relying on the "
             "config default.",
    )
    ap.add_argument(
        "--physical-capability", action="store_true",
        help="revert the TEMPORARY +-1.0 pu DER capability override and use the "
             "real VDE-AR-N-4120 / STATCOM operating diagrams.",
    )
    ap.add_argument(
        "--der-slope", type=float, default=None,
        help="override the DER Q(V) droop slope [pu] on BOTH plants (nominal "
             "0.06); smaller = steeper (gain 1/slope).",
    )
    args = ap.parse_args(argv)

    static_cfg = make_gate_e_config(args.duration, verbose=args.verbose)
    rms_cfg = make_gate_e_config(args.duration, verbose=args.verbose)
    if args.profiles:
        for c in (static_cfg, rms_cfg):
            c.use_profiles = True
            c.rms_profile_settle_s = float(args.profile_settle)
            c.use_zonal_gen_dispatch = False
    if args.scenario:
        for c in (static_cfg, rms_cfg):
            c.scenario = str(args.scenario)
        print(f"  [scenario] network scenario -> {args.scenario}")
    if args.start_time:
        from datetime import datetime as _dt
        _st = _dt.strptime(args.start_time, "%Y-%m-%d %H:%M")
        for c in (static_cfg, rms_cfg):
            c.start_time = _st
        print(f"  [start] profile start -> {_st:%Y-%m-%d %H:%M}")
    if args.physical_capability:
        for c in (static_cfg, rms_cfg):
            c.der_q_capability_override_pu = None
        print("  [capability] +-1.0 pu override REVERTED -- real VDE / STATCOM "
              "diagrams active (physical run)")
    if args.der_deadband is not None:
        for c in (static_cfg, rms_cfg):
            c.tso_qv_deadband_pu = float(args.der_deadband)
            c.dso_qv_deadband_pu = float(args.der_deadband)
            c.der_qv_deadband_override_pu = float(args.der_deadband)
        print(f"  [deadband] DER Q(V) deadband -> {args.der_deadband:g} pu "
              f"(both plants)")
    if args.der_slope is not None:
        for c in (static_cfg, rms_cfg):
            c.tso_qv_slope_pu = float(args.der_slope)
            c.dso_qv_slope_pu = float(args.der_slope)
        print(f"  [slope] DER Q(V) slope -> {args.der_slope:g} pu (both plants)")

    spec = ReplaySpecification(static_cfg, rms_cfg)
    spec.comparison = "open-loop u->y: static's recorded u replayed to RMS"
    run_dir = new_run_dir("rms_openloop_uy", spec,
                          subdirs=("figures", "csv", "snapshot"))
    print(f"[u->y] results -> {run_dir.root}")

    # 1. Static closed loop, capturing the actuator/profile timeline.
    print("\n[u->y] static closed-loop reference (capturing u)")
    cap = _CaptureFactory()
    static_log = run_multi_tso_dso(static_cfg, plant_factory=cap)
    with (run_dir.root / "static_records.pkl").open("wb") as fh:
        pickle.dump(static_log, fh, protocol=pickle.HIGHEST_PROTOCOL)
    timeline = cap.plant.timeline

    # 2. Build the RMS plant from the identical init, then replay u.
    print("\n[u->y] building RMS plant and replaying the captured u")
    factory = PowerFactoryReplayFactory(
        out_dir=run_dir.snapshot,
        project=args.project,
        on_missing_avr="skip",
        distributed_slack=rms_cfg.distributed_slack,
        enforce_q_lims=rms_cfg.enforce_q_lims_plant,
        event_pool_slots=1,
        preallocate_profiles=bool(args.profiles),
    )
    rms_plant = factory(cap.net_init, meta=cap.meta, zone_map=cap.zone_map)
    _replay_to_rms(rms_plant, timeline,
                   settle_s=float(rms_cfg.rms_profile_settle_s), dt_s=DT_S)

    if abs(rms_plant.t - args.duration) > 1e-6:
        raise RuntimeError(
            f"RMS plant ended at t={rms_plant.t}, expected {args.duration}")

    # 3. Harvest RMS y, compare to static y (same machinery as the closed run).
    raw = rms_plant.harvest_trajectories_bulk(
        run_dir.csv / "rms_comres_full.csv",
        since_s=0.0, stride=args.stride,
        labels=lambda l: l.startswith("qSTS_") or l.startswith("u_"))
    trajectories_long_frame(raw).to_csv(
        run_dir.csv / "rms_monitors_raw.csv", index=False)

    static_traj = static_controlled_trajectories(static_log)
    rms_traj = rms_controlled_trajectories(raw, factory.snapshot_doc)
    endpoint = endpoint_comparison(static_traj, rms_traj)
    endpoint.to_csv(run_dir.csv / "endpoint_comparison.csv", index=False)

    summary = {
        "test": "open_loop_u_to_y",
        "interpretation": (
            "identical u+profile into both plants; residual = plant/load "
            "difference only (no controller divergence)"),
        "duration_s": float(args.duration),
        "profiles": bool(args.profiles),
        "der_q_capability_override_pu": rms_cfg.der_q_capability_override_pu,
        "endpoint_error_by_quantity": _type_endpoint_summary(endpoint),
    }
    with (run_dir.root / "uy_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, allow_nan=False)

    print("\n[u->y] endpoint error under IDENTICAL u (plant-only residual):")
    for row in summary["endpoint_error_by_quantity"]:
        print("   ", {k: (round(v, 5) if isinstance(v, float) else v)
                      for k, v in row.items()})
    print(f"\n[u->y] done -> {run_dir.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
