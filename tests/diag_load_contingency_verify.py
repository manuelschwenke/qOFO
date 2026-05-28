"""
Diagnostic: verify that load contingency events actually take effect.

Four cases covered (each on the real wind_replace + add_hv_networks net):

  1. Dormant `connect` -> `trip` (legacy mode):
     create a dormant load, run `_apply_contingency` connect+PF, verify
     P_load > 0 at that bus AND res_bus.vm_pu drops; then trip+PF and
     verify P_load == 0 AND res_bus.vm_pu recovers.

  2. Trip an existing TN load by `element_index`:
     pick a constant-row TN load with non-zero P, trip+PF, verify
     in_service==False, res_load.p_mw==0, res_bus.vm_pu recovers.
     Then restore+PF, verify back to original.

  3. Trip an existing HV load by `name`:
     pick `DSO_1|HV_MV_Sub_6_const`, same trip/restore checks.

  4. **Persistence across apply_profiles**: trip a profile-tagged TN
     load, then call `apply_profiles` (the start-of-step hook), then PF.
     Verify the load is STILL out of service AND res_load contribution
     is zero — confirms `apply_profiles` does not silently reset
     `in_service` and that any p_mw rewrite is moot when in_service=False.

Exit code 0 iff every assertion passes.  Prints a compact per-case report.
"""

from __future__ import annotations

import io
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Force UTF-8 on Windows so Δ / Π etc. render in the console.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandapower as pp

from core.profiles import (
    DEFAULT_PROFILES_CSV,
    apply_profiles,
    load_profiles,
    snapshot_base_values,
)
from experiments.helpers.contingency import (
    _apply_contingency,
    prepare_load_contingencies,
)
from experiments.helpers.records import ContingencyEvent
from network.ieee39 import add_hv_networks, build_ieee39_net


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _build_net():
    net, meta = build_ieee39_net(scenario="wind_replace", verbose=False)
    meta = add_hv_networks(
        net, meta, install_tso_tertiary_shunts=False, verbose=False,
    )
    snapshot_base_values(net)
    return net, meta


def _pf(net):
    pp.runpp(
        net,
        run_control=False,
        calculate_voltage_angles=True,
        max_iteration=100,
        distributed_slack=False,
    )


def _check(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return False
    print(f"  ok:   {msg}")
    return True


# ---------------------------------------------------------------------------
#  Case 1: dormant connect -> trip
# ---------------------------------------------------------------------------


def case1_dormant_connect_trip() -> bool:
    print("\n=== Case 1: dormant connect -> trip (legacy mode) ===")
    net, meta = _build_net()
    _pf(net)
    target_bus = 5
    vm_before = float(net.res_bus.at[target_bus, "vm_pu"])
    p_load_before = float(net.res_load.loc[
        net.load["bus"] == target_bus, "p_mw"].sum())

    events = [
        ContingencyEvent(minute=10, element_type="load", bus=target_bus,
                         p_mw=400.0, q_mvar=200.0, action="connect"),
        ContingencyEvent(minute=20, element_type="load", bus=target_bus,
                         p_mw=400.0, q_mvar=200.0, action="trip"),
    ]
    prepare_load_contingencies(net, events, verbose=0)
    new_idx = events[0].element_index
    print(f"  bus {target_bus}: pre-connect res_bus.vm_pu = {vm_before:.4f}, "
          f"sum res_load.p_mw = {p_load_before:.2f} MW")

    # Apply connect
    _apply_contingency(net, events[0], verbose=0)
    _pf(net)
    vm_after_connect = float(net.res_bus.at[target_bus, "vm_pu"])
    p_load_after_connect = float(net.res_load.at[new_idx, "p_mw"])
    p_load_sum_after_connect = float(net.res_load.loc[
        net.load["bus"] == target_bus, "p_mw"].sum())
    print(f"  post-connect: vm_pu={vm_after_connect:.4f} (Δ={vm_after_connect - vm_before:+.4f}), "
          f"res_load at new idx={p_load_after_connect:.2f} MW, "
          f"bus total={p_load_sum_after_connect:.2f} MW")

    ok = True
    ok &= _check(net.load.at[new_idx, "in_service"], "connect set in_service=True")
    ok &= _check(abs(p_load_after_connect - 400.0) < 1.0,
                 f"new load contributes ~400 MW (got {p_load_after_connect:.2f})")
    ok &= _check(abs(p_load_sum_after_connect - p_load_before - 400.0) < 1.0,
                 "bus total P increased by ~400 MW")
    ok &= _check(vm_after_connect < vm_before - 1e-4,
                 f"voltage dropped (Δ={vm_after_connect - vm_before:+.4f})")

    # Apply trip
    _apply_contingency(net, events[1], verbose=0)
    _pf(net)
    vm_after_trip = float(net.res_bus.at[target_bus, "vm_pu"])
    p_load_after_trip = float(net.res_load.at[new_idx, "p_mw"])
    print(f"  post-trip:    vm_pu={vm_after_trip:.4f} (Δ={vm_after_trip - vm_after_connect:+.4f}), "
          f"res_load at idx={p_load_after_trip:.2f} MW")

    ok &= _check(not bool(net.load.at[new_idx, "in_service"]),
                 "trip set in_service=False")
    ok &= _check(abs(p_load_after_trip) < 1e-6,
                 f"out-of-service load contributes 0 MW (got {p_load_after_trip})")
    ok &= _check(vm_after_trip > vm_after_connect + 1e-4,
                 "voltage recovered after trip")
    return ok


# ---------------------------------------------------------------------------
#  Case 2: trip existing TN load by element_index
# ---------------------------------------------------------------------------


def case2_existing_tn_load_by_index() -> bool:
    print("\n=== Case 2: trip existing TN load by element_index ===")
    net, meta = _build_net()
    _pf(net)

    # Find a TN constant-row load with non-zero P
    tn_const = net.load[
        (net.load["subnet"].astype(str) == "TN")
        & (net.load["profile_p"].isna())
        & (net.load["p_mw"] > 1.0)
    ]
    target_idx = int(tn_const.index[0])
    target_bus = int(net.load.at[target_idx, "bus"])
    p_orig = float(net.load.at[target_idx, "p_mw"])
    in_service_orig = bool(net.load.at[target_idx, "in_service"])
    vm_before = float(net.res_bus.at[target_bus, "vm_pu"])
    p_load_at_bus_before = float(net.res_load.loc[
        net.load["bus"] == target_bus, "p_mw"].sum())
    print(f"  target load idx={target_idx} at bus {target_bus}: "
          f"p_mw={p_orig:.2f}, in_service={in_service_orig}")
    print(f"  pre-trip: vm_pu={vm_before:.4f}, bus total P={p_load_at_bus_before:.2f}")

    events = [
        ContingencyEvent(minute=10, element_type="load",
                         element_index=target_idx, action="trip"),
        ContingencyEvent(minute=20, element_type="load",
                         element_index=target_idx, action="restore"),
    ]
    prepare_load_contingencies(net, events, verbose=0)

    # Trip
    _apply_contingency(net, events[0], verbose=0)
    _pf(net)
    in_service_after_trip = bool(net.load.at[target_idx, "in_service"])
    p_load_after_trip = float(net.res_load.at[target_idx, "p_mw"])
    vm_after_trip = float(net.res_bus.at[target_bus, "vm_pu"])
    print(f"  post-trip: in_service={in_service_after_trip}, "
          f"res_load.p_mw={p_load_after_trip:.2f}, vm_pu={vm_after_trip:.4f}")

    ok = True
    ok &= _check(not in_service_after_trip, "in_service flipped to False")
    ok &= _check(abs(p_load_after_trip) < 1e-6,
                 f"res_load.p_mw == 0 (got {p_load_after_trip})")
    ok &= _check(vm_after_trip > vm_before + 1e-5,
                 f"voltage rose after load shed (Δ={vm_after_trip - vm_before:+.5f})")

    # Restore
    _apply_contingency(net, events[1], verbose=0)
    _pf(net)
    in_service_after_restore = bool(net.load.at[target_idx, "in_service"])
    p_load_after_restore = float(net.res_load.at[target_idx, "p_mw"])
    vm_after_restore = float(net.res_bus.at[target_bus, "vm_pu"])
    print(f"  post-restore: in_service={in_service_after_restore}, "
          f"res_load.p_mw={p_load_after_restore:.2f}, vm_pu={vm_after_restore:.4f}")

    ok &= _check(in_service_after_restore, "in_service flipped to True")
    ok &= _check(abs(p_load_after_restore - p_orig) < 1e-3,
                 f"res_load.p_mw restored to {p_orig:.2f}")
    ok &= _check(abs(vm_after_restore - vm_before) < 1e-4,
                 "voltage returned to original")
    return ok


# ---------------------------------------------------------------------------
#  Case 3: trip existing HV load by name
# ---------------------------------------------------------------------------


def case3_existing_hv_load_by_name() -> bool:
    print("\n=== Case 3: trip existing HV load by name ===")
    net, meta = _build_net()
    _pf(net)

    name = "DSO_1|HV_MV_Sub_6_const"
    matches = net.load.index[net.load["name"] == name].tolist()
    assert len(matches) == 1, f"expected unique match for {name!r}, got {matches}"
    target_idx = int(matches[0])
    target_bus = int(net.load.at[target_idx, "bus"])
    p_orig = float(net.load.at[target_idx, "p_mw"])
    q_orig = float(net.load.at[target_idx, "q_mvar"])
    vm_before = float(net.res_bus.at[target_bus, "vm_pu"])
    print(f"  target load idx={target_idx} ({name!r}) at HV bus {target_bus}: "
          f"p_mw={p_orig:.2f}, q_mvar={q_orig:.2f}")
    print(f"  pre-trip: vm_pu={vm_before:.4f}")

    events = [
        ContingencyEvent(minute=10, element_type="load",
                         name=name, action="trip"),
        ContingencyEvent(minute=20, element_type="load",
                         name=name, action="restore"),
    ]
    prepare_load_contingencies(net, events, verbose=0)
    assert events[0].element_index == target_idx
    assert events[1].element_index == target_idx

    # Trip
    _apply_contingency(net, events[0], verbose=0)
    _pf(net)
    p_load_after_trip = float(net.res_load.at[target_idx, "p_mw"])
    vm_after_trip = float(net.res_bus.at[target_bus, "vm_pu"])
    print(f"  post-trip: in_service={bool(net.load.at[target_idx, 'in_service'])}, "
          f"res_load.p_mw={p_load_after_trip:.2f}, vm_pu={vm_after_trip:.4f}")

    ok = True
    ok &= _check(not bool(net.load.at[target_idx, "in_service"]),
                 "in_service flipped to False")
    ok &= _check(abs(p_load_after_trip) < 1e-6,
                 f"res_load.p_mw == 0 (got {p_load_after_trip})")
    ok &= _check(vm_after_trip > vm_before + 1e-6,
                 f"voltage rose at HV bus after load shed (Δ={vm_after_trip - vm_before:+.6f})")

    # Restore
    _apply_contingency(net, events[1], verbose=0)
    _pf(net)
    p_load_after_restore = float(net.res_load.at[target_idx, "p_mw"])
    vm_after_restore = float(net.res_bus.at[target_bus, "vm_pu"])
    print(f"  post-restore: in_service={bool(net.load.at[target_idx, 'in_service'])}, "
          f"res_load.p_mw={p_load_after_restore:.2f}, vm_pu={vm_after_restore:.4f}")

    ok &= _check(bool(net.load.at[target_idx, "in_service"]),
                 "in_service flipped to True")
    ok &= _check(abs(p_load_after_restore - p_orig) < 1e-3,
                 f"res_load.p_mw restored to {p_orig:.2f}")
    ok &= _check(abs(vm_after_restore - vm_before) < 1e-4,
                 "voltage returned to original")
    return ok


# ---------------------------------------------------------------------------
#  Case 4: persistence across apply_profiles
# ---------------------------------------------------------------------------


def case4_persistence_across_apply_profiles() -> bool:
    print("\n=== Case 4: persistence across apply_profiles ===")
    net, meta = _build_net()
    profiles = load_profiles(DEFAULT_PROFILES_CSV, timestep_s=60.0)
    t0 = datetime(2016, 5, 5, 10, 0)
    profiles = profiles.loc[t0:t0 + timedelta(minutes=10)]
    apply_profiles(net, profiles, t0)
    _pf(net)

    # Pick a TN PROFILE load (so apply_profiles touches its p_mw every step)
    tn_prof = net.load[
        (net.load["subnet"].astype(str) == "TN")
        & (net.load["profile_p"].notna())
        & (net.load["p_mw"] > 1.0)
    ]
    target_idx = int(tn_prof.index[0])
    target_bus = int(net.load.at[target_idx, "bus"])
    vm_before = float(net.res_bus.at[target_bus, "vm_pu"])
    print(f"  target profile-load idx={target_idx} at bus {target_bus}: "
          f"profile_p={net.load.at[target_idx, 'profile_p']!r}, "
          f"p_mw={float(net.load.at[target_idx, 'p_mw']):.2f}")

    ev = ContingencyEvent(minute=10, element_type="load",
                          element_index=target_idx, action="trip")
    prepare_load_contingencies(net, [ev], verbose=0)
    _apply_contingency(net, ev, verbose=0)
    _pf(net)
    in_service_after_trip = bool(net.load.at[target_idx, "in_service"])
    p_load_after_trip = float(net.res_load.at[target_idx, "p_mw"])
    print(f"  post-trip: in_service={in_service_after_trip}, "
          f"res_load.p_mw={p_load_after_trip:.2f}")

    # Now call apply_profiles AGAIN (simulates the next step's start)
    t1 = t0 + timedelta(minutes=5)
    apply_profiles(net, profiles, t1)
    in_service_after_profiles = bool(net.load.at[target_idx, "in_service"])
    p_mw_after_profiles = float(net.load.at[target_idx, "p_mw"])
    print(f"  after apply_profiles: in_service={in_service_after_profiles}, "
          f"p_mw (net.load, NOT res)={p_mw_after_profiles:.2f}  "
          f"# note: apply_profiles writes p_mw from base*scale regardless")

    _pf(net)
    p_load_after_pf = float(net.res_load.at[target_idx, "p_mw"])
    print(f"  after subsequent PF: res_load.p_mw={p_load_after_pf:.2f}")

    ok = True
    ok &= _check(not in_service_after_profiles,
                 "in_service stayed False across apply_profiles (no silent reset)")
    ok &= _check(abs(p_load_after_pf) < 1e-6,
                 f"res_load.p_mw stayed 0 after subsequent PF (got {p_load_after_pf})")
    return ok


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> int:
    results = {
        "1. dormant connect+trip":      case1_dormant_connect_trip(),
        "2. existing TN trip+restore":  case2_existing_tn_load_by_index(),
        "3. existing HV trip+restore":  case3_existing_hv_load_by_name(),
        "4. persistence across apply_profiles": case4_persistence_across_apply_profiles(),
    }
    print("\n" + "=" * 60)
    print("Summary:")
    for name, ok in results.items():
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
    print("=" * 60)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
