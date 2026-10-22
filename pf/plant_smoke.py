"""
pf/plant_smoke.py
=================
Phase-6b smoke test: drive ``PowerFactoryPlant`` through one miniature
OFO-style dispatch sequence and verify every interface leg against known
references (PF machine only; consumes the engine licence seat).

Checks
------
1. **read_y at t=0** matches the snapshot solution (the RMS init load
   flow is parity-aligned): max |dV| < 1e-3 pu over all buses.
2. **DER Q dispatch** (+20 Mvar on the first TSO park): after
   ``advance(20)`` the park's ``res_sgen.q_mvar`` moved by ~+20 Mvar.
3. **Coupler tap dispatch** (+1 on the first NC3W): the interface
   ``res_trafo3w.q_hv_mvar`` moves by several Mvar and the shadow tap
   state advances.
4. **Hold interval** (no writes): controlled outputs drift < 1e-3 pu
   between two consecutive ``read_y`` snapshots 10 s apart (the model is
   back near equilibrium after the ring decays).
5. **Trajectory harvest**: the monitored labels cover the dispatch window
   and the time axis is monotone.

Usage::

    python pf\\plant_smoke.py [--snapshot export/snapshots/full_t0_*.json]

Author: Manuel Schwenke / Claude Code (2026-07-20)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from pf.plant import PowerFactoryPlant  # noqa: E402
from core.plant import ActuatorWrites  # noqa: E402

DEFAULT_SNAPSHOT = next(
    (Path(__file__).resolve().parents[1] / "export" / "snapshots").glob(
        "full_t0_*.json"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PowerFactoryPlant smoke test")
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    args = parser.parse_args(argv)

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    print(f"[smoke] snapshot: {args.snapshot}")
    plant = PowerFactoryPlant(args.snapshot)
    doc = plant.doc

    # ── 1) t=0 measurement image vs snapshot solution ────────────────────
    net = plant.read_y()
    sol_bus = doc["solution"]["bus"]
    dv = max(abs(float(net.res_bus.at[int(k), "vm_pu"]) - float(rec["vm_pu"]))
             for k, rec in sol_bus.items())
    check("read_y@t0 vs snapshot", dv < 1e-3, f"max |dV| = {dv:.3e} pu")

    # ── 2) DER Q dispatch ────────────────────────────────────────────────
    park = min(i for i in plant._reec)          # deterministic first park
    q0 = float(net.res_sgen.at[park, "q_mvar"])
    plant.apply_u(ActuatorWrites(der_q_set_mvar={park: q0 + 20.0}))
    plant.advance(20.0)
    net = plant.read_y()
    q1 = float(net.res_sgen.at[park, "q_mvar"])
    check("DER +20 Mvar", abs((q1 - q0) - 20.0) < 2.0,
          f"sgen[{park}] Q {q0:.2f} -> {q1:.2f} Mvar (d = {q1 - q0:+.2f})")

    # ── 3) coupler tap dispatch ──────────────────────────────────────────
    t3w = min(plant._tr3)
    tap0 = plant._tap3w[t3w]
    qi0 = float(net.res_trafo3w.at[t3w, "q_hv_mvar"])
    plant.apply_u(ActuatorWrites(tap_3w={t3w: tap0 + 1}))
    plant.advance(20.0)
    net = plant.read_y()
    qi1 = float(net.res_trafo3w.at[t3w, "q_hv_mvar"])
    check("coupler tap +1", abs(qi1 - qi0) > 2.0 and plant._tap3w[t3w] == tap0 + 1,
          f"trafo3w[{t3w}] q_hv {qi0:.2f} -> {qi1:.2f} Mvar, "
          f"shadow tap {tap0} -> {plant._tap3w[t3w]}")

    # ── 4) hold interval: near-equilibrium drift ─────────────────────────
    vm_a = net.res_bus["vm_pu"].copy()
    plant.advance(10.0)
    vm_b = plant.read_y().res_bus["vm_pu"]
    drift = float((vm_b - vm_a).abs().max())
    check("hold-interval drift", drift < 1e-3, f"max |dV| = {drift:.3e} pu")

    # ── 5) trajectory harvest ────────────────────────────────────────────
    traj = plant.harvest_trajectories(since_s=plant.t - 10.0)
    label, (ta, ya) = next(iter(traj.items()))
    mono = bool(np.all(np.diff(ta) > 0)) and len(ta) > 10
    check("trajectory harvest", mono,
          f"{len(traj)} signals, e.g. {label}: {len(ta)} samples "
          f"t in [{ta[0]:.2f}, {ta[-1]:.2f}] s")

    print(f"[smoke] t_end = {plant.t:.1f} s; "
          f"{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
