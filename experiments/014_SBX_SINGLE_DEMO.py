#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/014_SBX_SINGLE_DEMO.py
==================================
Single-run SBX mechanism demonstration with the LIVE Figure 6
(``config.live_plot_sbx`` — toggleable like the other live plots).

Runs ONE simulation of one 013 scenario (``asym_z3`` / ``asym_z1`` /
``asym_z2`` / ``sym_z1z2`` / ``compl_z1z3`` — definitions, stress
calibration and timing are imported from ``experiments/013_SBX_LADDER``
so the two experiments can never drift apart) and shows, live per
corridor:

* measured flow q_meas vs the schedule staircase q_sched and the
  standard q_std,
* the tier-1 NO-REMUNERATION band q_sched ± q_band (shaded),
* requests (need-flag strips per corridor end) and deliveries
  (▼ unilateral paid deal, ◆ mutual unpaid deal, △ unwind, ✕ scarcity),
* the surplus staircases (running deal balance) and the cumulative
  settlement payments per area.

The tier-1 band defaults to the per-scenario value calibrated in the
013 campaign (2 × RMS of the inert arm's clean-cycle deviation,
STATUS_SBX.md §7.3); override with ``--band``.

At the end the figure is saved to
``results/014_SBX_DEMO/<scenario>/sbx_mechanism.png`` together with the
settlement ledger/summary and the corridor cycle table.

Run examples:
    python experiments/014_SBX_SINGLE_DEMO.py --scenario asym_z3
    python experiments/014_SBX_SINGLE_DEMO.py --scenario sym_z1z2 --minutes 240
    python experiments/014_SBX_SINGLE_DEMO.py --scenario asym_z1 --no-live

Author: Manuel Schwenke / Claude Code
Date: 2026-07-08 (SBX Phase 7 follow-up)
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sbx.fail import rep1  # noqa: E402

_013 = importlib.import_module("experiments.013_SBX_LADDER")

RESULT_DIR = REPO / "results" / "014_SBX_DEMO"

#: Per-scenario tier-1 band defaults from the 013 campaign calibration
#: (STATUS_SBX.md §7.3: 2 × RMS of the inert arm's clean-cycle
#: deviation, integer-ceiled). Override with --band.
CALIBRATED_BAND_MVAR = {
    "asym_z3": 34.0,
    "asym_z1": 92.0,
    "asym_z2": 26.0,
    "sym_z1z2": 26.0,
    "compl_z1z3": 83.0,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Single-run SBX demonstration with live Figure 6.")
    ap.add_argument("--scenario", type=str, default="asym_z3",
                    choices=sorted(_013.SCENARIOS.keys()))
    ap.add_argument("--minutes", type=float,
                    default=_013.DEFAULT_MINUTES,
                    help="horizon (default 360 = full protocol arc; "
                         "240 still shows stress + unwind)")
    ap.add_argument("--band", type=float, default=None,
                    help="tier-1 band [Mvar]; default = the 013 "
                         "campaign calibration for the scenario")
    ap.add_argument("--no-live", action="store_true",
                    help="disable the live figure (still saves the "
                         "final PNG via a headless redraw)")
    ap.add_argument("--arm", type=str, default="sbx",
                    choices=("sbx", "sbx_inert", "none"),
                    help="mechanism arm; Figure 6 and the settlement/"
                         "cycle outputs exist only for the sbx arms")
    ap.add_argument("--schedule", type=str, default=None,
                    help="path to a planning-anchored v_std schedule "
                         "JSON (SBX v3, from experiments/017_SBX_"
                         "PLANNING.py); planning then replaces the "
                         "settled-state snapshot")
    ap.add_argument("--local-sens", action="store_true",
                    help="run the zones (and DSOs) on their LOCAL "
                         "cached sensitivities (Ward-style reduced "
                         "nets) instead of the shared full-network "
                         "Jacobian — the configuration most consistent "
                         "with the SBX locality principle")
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    if args.no_live:
        import os
        os.environ.setdefault("MPLBACKEND", "Agg")

    scenario = args.scenario
    band = (float(args.band) if args.band is not None
            else CALIBRATED_BAND_MVAR[scenario])
    out_dir = RESULT_DIR / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = _013.SCENARIOS[scenario]
    print(f"=== 014 SBX demo: {scenario} ({spec['family']}) ===")
    print(f"    {spec['description']}")
    print(f"    horizon {args.minutes:.0f} min, tier-1 band "
          f"±{band:.0f} Mvar, live plot "
          f"{'OFF' if args.no_live else 'ON'}")

    cfg = _013.make_config(scenario, args.arm, args.minutes,
                           q_band_mvar=band)
    cfg.verbose = args.verbose
    # Figure 6 exists only when the SBX machinery runs.
    cfg.live_plot_sbx = (args.arm in ("sbx", "sbx_inert"))
    cfg.live_plot_controller = True
    cfg.live_plot_cascade = True
    if args.local_sens:
        # Zones control from their own reduced-net cached models
        # (runner validation relaxed 2026-07-08; 013 keeps the shared
        # path for campaign comparability).
        cfg.local_sensitivities_tso = True
        cfg.local_sensitivities_dso = True
    if args.schedule is not None:
        sched_path = Path(args.schedule)
        if not sched_path.exists():
            rep1("schedule JSON not found — run experiments/"
                 "017_SBX_PLANNING.py first", path=str(sched_path))
        cfg.sbx_v_std_schedule_path = str(sched_path)
        # v3: the warmup existed ONLY for the settled-state snapshot
        # (revised A7). With a planning-anchored schedule the contract
        # is known before the run starts — the mechanism (and Figure 6)
        # arms at the first TSO tick.
        cfg.sbx_warmup_s = 0.0
        print(f"    v3 planning schedule: {sched_path} "
              f"(contracts active from t = 0)")

    captured: dict = {}

    def hook(state):
        captured.update(state)
        return None

    from experiments.runners.multi_tso_dso import run_multi_tso_dso
    t0 = time.perf_counter()
    recs = run_multi_tso_dso(cfg, pre_loop_hook=hook)
    print(f"    {len(recs)} steps in {time.perf_counter() - t0:.0f} s")

    runtime = captured.get("sbx_runtime") or {}
    adapter = runtime.get("adapter")
    plotter = runtime.get("live_plotter")
    if args.arm == "none":
        # Autonomous baseline: no SBX machinery, nothing else to persist.
        print("    (arm 'none': no SBX outputs)")
        if not args.no_live:
            try:
                import matplotlib.pyplot as plt
                print("\nclose the figure window(s) to exit ...")
                plt.ioff()
                plt.show()
            except Exception:
                pass
        return 0
    if adapter is None:
        rep1("run produced no SBX adapter — did the horizon end before "
             "sbx_warmup_s?", minutes=args.minutes,
             warmup_s=cfg.sbx_warmup_s)

    # ── Persist: figure, settlement outputs, corridor cycle table ──────
    if plotter is not None:
        png = out_dir / "sbx_mechanism.png"
        plotter.save(png)
        print(f"    figure:  {png}")
    from sbx.settlement import write_settlement_outputs
    csv_path, md_path = write_settlement_outputs(
        adapter.scheduler.settlement_engines, out_dir, scenario)
    print(f"    ledger:  {csv_path}")
    print(f"    summary: {md_path}")

    sched = adapter.scheduler
    print("\n=== corridor cycle table ===")
    for key, rl in sched.records.items():
        print(f"corridor {key}:")
        for r in rl:
            mark = ("DEAL " + r.deal.kind if r.deal.dq_deal_mvar != 0.0
                    else "unwind" if r.unwound_mvar != 0.0
                    else "scarcity" if r.deal.kind == "scarcity" else "")
            print(f"  c{r.cycle:3d}: q_std={r.q_std_mvar:+8.2f} "
                  f"q_meas={r.q_meas_mvar:+8.2f} "
                  f"q_sched={r.q_sched_mvar:+8.2f} "
                  f"surplus={r.surplus_mvar:+7.2f} dv={r.dv_pu:+.5f} "
                  f"need_a={int(r.need_a)} need_b={int(r.need_b)} "
                  f"[{r.consistency}] {mark}")
    n_scarcity = len(sched.scarcity_events)
    if n_scarcity:
        print(f"scarcity events: {n_scarcity}")

    if not args.no_live:
        try:
            import matplotlib.pyplot as plt
            print("\nclose the figure window to exit ...")
            plt.ioff()
            plt.show()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    main()
