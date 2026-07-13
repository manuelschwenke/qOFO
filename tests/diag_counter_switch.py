# -*- coding: utf-8 -*-
"""
Closed-loop COUNTER-SWITCH scenario (Phase 5 deferral → Phase 6 item 6):
provoke simultaneous discrete pressure in two zones in the SAME round
and verify what §3.8.2 slotting buys.

Construction: two reactive-heavy load connects (200 MW / 150 Mvar) fire
at the same minute in zone 1 and zone 3 interiors. Run the bme rung
twice — slotting ON (default round robin) vs OFF — and compare:

  * rounds with ≥ 2 zones committing discrete moves (OFF exposes the
    conflict; ON must show 0 by construction);
  * realised ΔΦ of commits in the conflict window (counter-switching =
    accepted moves whose realised ΔΦ is positive because the other
    zone's simultaneous move changed the operating point);
  * ledger decision mix and total switch counts.

Run:  python tests/diag_counter_switch.py    (~8 min)
"""
from __future__ import annotations

import importlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.helpers.records import ContingencyEvent  # noqa: E402
from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402

ladder = importlib.import_module("experiments.011_BME_LADDER")

MINUTES = 40.0
EVENT_MIN = 10

# Zone-interior TN load buses (fixed 3-area partition; away from the
# boundary registry so each zone's own OLTC/DER react first).
_006 = importlib.import_module("experiments.006_CIGRE_MONTECARLO")
from network.zone_partition import fixed_zone_partition_ieee39  # noqa: E402


def _pick_buses():
    """One admissible load-connect TN bus in zone 1 and one in zone 3
    (same net-build pattern as 006's enumerate_elements, so indices
    line up with what the runner sees)."""
    from network.ieee39 import build_ieee39_net
    cfg = ladder.make_ladder_config("none", MINUTES)
    elements = _006.enumerate_elements(cfg)
    net, _ = build_ieee39_net(ext_grid_vm_pu=1.03, scenario=cfg.scenario,
                              verbose=False)
    zmap, _ = fixed_zone_partition_ieee39(net, verbose=False)
    lb = set(elements["load_buses"])
    b1 = sorted(lb & set(zmap[1]))
    b3 = sorted(lb & set(zmap[3]))
    if not b1 or not b3:
        raise RuntimeError(f"no interior load buses found (z1={b1}, z3={b3})")
    return b1[0], b3[0]


def _run(slotting: bool, bus1: int, bus3: int):
    cfg = ladder.make_ladder_config("bme", MINUTES)
    cfg.verbose = 0
    cfg.bme_slotting = slotting
    cfg.contingencies = [
        ContingencyEvent(minute=EVENT_MIN, element_type="load",
                         element_index=-1, bus=bus1, p_mw=200.0,
                         q_mvar=150.0, action="connect"),
        ContingencyEvent(minute=EVENT_MIN, element_type="load",
                         element_index=-1, bus=bus3, p_mw=200.0,
                         q_mvar=150.0, action="connect"),
    ]
    captured = {}
    recs = run_multi_tso_dso(cfg, pre_loop_hook=lambda s: captured.update(s))
    ledger = captured.get("bme_ledger")
    entries = ledger.to_records() if ledger is not None else []
    return recs, entries


def _analyse(tag: str, recs, entries):
    acc = [e for e in entries if e["reason"] == "accepted"]
    by_step = defaultdict(set)
    for e in acc:
        by_step[int(e["step"])].add(int(e["zone"]))
    multi = sorted(s for s, zs in by_step.items() if len(zs) >= 2)
    pos_real = [
        e for e in acc
        if e.get("realised_dphi") not in (None, "", "None")
        and np.isfinite(float(e["realised_dphi"]))
        and float(e["realised_dphi"]) > 0.0
    ]
    phi = np.array([np.nan if r.bme_phi_mw is None else r.bme_phi_mw
                    for r in recs])
    print(f"[{tag}] accepted={len(acc)}  "
          f"eps_reject={sum(1 for e in entries if e['reason'] == 'epsilon_reject')}  "
          f"slot_blocked={sum(1 for e in entries if e['reason'] == 'slot_blocked')}")
    print(f"[{tag}] rounds with >=2 zones committing: {len(multi)} {multi}")
    print(f"[{tag}] accepted commits with realised dPhi > 0 "
          f"(counter-productive): {len(pos_real)}")
    print(f"[{tag}] Phi max after event: "
          f"{np.nanmax(phi[3 * EVENT_MIN:]):.3f} MW, "
          f"Phi final: {phi[-1]:.3f} MW")
    return len(multi), len(pos_real)


if __name__ == "__main__":
    bus1, bus3 = _pick_buses()
    print(f"counter-switch scenario: simultaneous 200 MW/150 Mvar connects "
          f"at minute {EVENT_MIN}, buses {bus1} (z1) / {bus3} (z3)")
    recs_on, led_on = _run(True, bus1, bus3)
    recs_off, led_off = _run(False, bus1, bus3)
    m_on, p_on = _analyse("slotting ON ", recs_on, led_on)
    m_off, p_off = _analyse("slotting OFF", recs_off, led_off)
    assert m_on == 0, "slotting ON must never allow simultaneous commits"
    print("diag_counter_switch: DONE (ON prevents simultaneous commits; "
          "compare counter-productive counts above)")
