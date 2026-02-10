#!/usr/bin/env python3
"""
Diagnostic script: How pandapower stores 3W transformer branches in _ppc['branch'].

Goal: understand the correct mapping from trafo3w elements to _ppc branch indices
for the TU Darmstadt benchmark network.
"""
import sys
import numpy as np

# ── 1. Build the TU Darmstadt network (runs power flow internally) ──────────
print("=" * 80)
print("1. Building TU Darmstadt network ...")
print("=" * 80)
from network.build_tuda_net import build_tuda_net
net, meta = build_tuda_net()
print(f"   Power flow converged: {net.converged}")
print(f"   Number of buses:  {len(net.bus)}")
print(f"   Number of lines:  {len(net.line)}")
print(f"   Number of trafos: {len(net.trafo)}")
print(f"   Number of trafo3w: {len(net.trafo3w)}")
print(f"   trafo3w indices: {list(net.trafo3w.index)}")
print(f"   meta.coupler_trafo3w_indices: {meta.coupler_trafo3w_indices}")
print()

# ── 2. Print branch lookup ranges ──────────────────────────────────────────
print("=" * 80)
print("2. net._pd2ppc_lookups['branch']  (branch type -> [start, end])")
print("=" * 80)
branch_lookup = net._pd2ppc_lookups.get('branch', {})
for key, val in sorted(branch_lookup.items(), key=lambda x: x[1][0]):
    print(f"   {key:12s} -> [{val[0]:3d}, {val[1]:3d})  "
          f"({val[1] - val[0]} branches)")
print()

# ── 3. Print trafo3w winding buses ─────────────────────────────────────────
print("=" * 80)
print("3. net.trafo3w[['hv_bus', 'mv_bus', 'lv_bus']]")
print("=" * 80)
print(net.trafo3w[['hv_bus', 'mv_bus', 'lv_bus', 'name']].to_string())
print()

# ── 4. Map winding buses to PPC bus indices ────────────────────────────────
print("=" * 80)
print("4. Winding buses -> PPC bus indices")
print("=" * 80)
bus_lookup = net._pd2ppc_lookups.get('bus')
if bus_lookup is not None:
    print(f"   bus_lookup type: {type(bus_lookup)}, shape: {getattr(bus_lookup, 'shape', 'N/A')}")
    print(f"   bus_lookup dtype: {getattr(bus_lookup, 'dtype', 'N/A')}")
    print()
    for t3w_idx in net.trafo3w.index:
        hv = int(net.trafo3w.at[t3w_idx, 'hv_bus'])
        mv = int(net.trafo3w.at[t3w_idx, 'mv_bus'])
        lv = int(net.trafo3w.at[t3w_idx, 'lv_bus'])
        hv_ppc = int(bus_lookup[hv])
        mv_ppc = int(bus_lookup[mv])
        lv_ppc = int(bus_lookup[lv])
        print(f"   trafo3w {t3w_idx}: "
              f"hv_bus {hv} -> ppc {hv_ppc},  "
              f"mv_bus {mv} -> ppc {mv_ppc},  "
              f"lv_bus {lv} -> ppc {lv_ppc}")
else:
    print("   bus lookup not available!")
print()

# ── 5. Print branch table around the trafo3w range ────────────────────────
print("=" * 80)
print("5. Branch table entries around the trafo3w range")
print("=" * 80)
branch = net._ppc['branch']
n_branches = len(branch)
print(f"   Total branches in _ppc: {n_branches}")

t3w_range = branch_lookup.get('trafo3w', None)
if t3w_range is not None:
    t3w_start, t3w_end = t3w_range
    show_start = max(0, t3w_start - 5)
    show_end = min(n_branches, t3w_end + 5)
    print(f"   trafo3w range: [{t3w_start}, {t3w_end})")
    print(f"   Showing branches [{show_start}, {show_end}):")
    print()
    print(f"   {'idx':>4s}  {'from':>6s}  {'to':>6s}  {'r':>12s}  {'x':>12s}  "
          f"{'b':>12s}  {'rateA':>10s}  {'tap':>8s}  {'shift':>8s}  {'status':>6s}")
    print(f"   {'----':>4s}  {'------':>6s}  {'------':>6s}  {'------------':>12s}  {'------------':>12s}  "
          f"{'------------':>12s}  {'----------':>10s}  {'--------':>8s}  {'--------':>8s}  {'------':>6s}")

    # MATPOWER branch columns:
    # 0: fbus, 1: tbus, 2: r, 3: x, 4: b, 5: rateA, 6: rateB, 7: rateC,
    # 8: ratio/tap, 9: angle/shift, 10: status
    for i in range(show_start, show_end):
        marker = " <-- T3W" if t3w_start <= i < t3w_end else ""
        # Identify what branch type this index belongs to
        br_type = "?"
        for key, (s, e) in branch_lookup.items():
            if s <= i < e:
                br_type = key
                break
        print(f"   {i:4d}  {int(branch[i, 0]):6d}  {int(branch[i, 1]):6d}  "
              f"{branch[i, 2]:12.8f}  {branch[i, 3]:12.8f}  "
              f"{branch[i, 4]:12.8f}  {branch[i, 5]:10.2f}  "
              f"{branch[i, 8]:8.5f}  {branch[i, 9]:8.3f}  "
              f"{int(branch[i, 10]):6d}  [{br_type}]{marker}")
print()

# ── 6. Identify which branches belong to each 3W transformer ──────────────
print("=" * 80)
print("6. Identify branches for each 3W transformer by scanning for known + aux buses")
print("=" * 80)

if t3w_range is not None and bus_lookup is not None:
    t3w_start, t3w_end = t3w_range

    # Collect ALL known winding buses across all trafo3w
    all_winding_buses = set()
    trafo3w_bus_map = {}
    for t3w_idx in net.trafo3w.index:
        hv = int(net.trafo3w.at[t3w_idx, 'hv_bus'])
        mv = int(net.trafo3w.at[t3w_idx, 'mv_bus'])
        lv = int(net.trafo3w.at[t3w_idx, 'lv_bus'])
        hv_ppc = int(bus_lookup[hv])
        mv_ppc = int(bus_lookup[mv])
        lv_ppc = int(bus_lookup[lv])
        trafo3w_bus_map[t3w_idx] = {
            'hv': hv_ppc, 'mv': mv_ppc, 'lv': lv_ppc,
            'hv_pp': hv, 'mv_pp': mv, 'lv_pp': lv,
        }
        all_winding_buses.update([hv_ppc, mv_ppc, lv_ppc])

    # Scan all branches in the trafo3w range
    print(f"   Scanning branches {t3w_start} to {t3w_end - 1}:")
    print()
    for i in range(t3w_start, t3w_end):
        from_bus = int(branch[i, 0])
        to_bus = int(branch[i, 1])
        from_in = from_bus in all_winding_buses
        to_in = to_bus in all_winding_buses
        # Try to identify which trafo3w this branch belongs to
        owner = "?"
        winding = "?"
        aux = "?"
        for t3w_idx, buses in trafo3w_bus_map.items():
            if from_bus in (buses['hv'], buses['mv'], buses['lv']):
                owner = f"trafo3w {t3w_idx}"
                if from_bus == buses['hv']: winding = "HV"
                elif from_bus == buses['mv']: winding = "MV"
                elif from_bus == buses['lv']: winding = "LV"
                aux = f"aux_bus={to_bus}"
                break
            elif to_bus in (buses['hv'], buses['mv'], buses['lv']):
                owner = f"trafo3w {t3w_idx}"
                if to_bus == buses['hv']: winding = "HV"
                elif to_bus == buses['mv']: winding = "MV"
                elif to_bus == buses['lv']: winding = "LV"
                aux = f"aux_bus={from_bus}"
                break
        print(f"   Branch {i:3d}: from={from_bus:3d} to={to_bus:3d}  "
              f"from_known={from_in}  to_known={to_in}  "
              f"-> {owner}, winding={winding}, {aux}")
    print()

    # Try the offset formula from the current code
    print("   --- Checking the formula: t3w_start + 3*pos + {0,1,2} ---")
    trafo3w_positions = list(net.trafo3w.index)
    for t3w_idx in net.trafo3w.index:
        pos = trafo3w_positions.index(t3w_idx)
        cands = [t3w_start + 3*pos, t3w_start + 3*pos + 1, t3w_start + 3*pos + 2]
        buses = trafo3w_bus_map[t3w_idx]
        known_set = {buses['hv'], buses['mv'], buses['lv']}
        print(f"\n   trafo3w {t3w_idx} (pos={pos}): candidate indices = {cands}")
        print(f"   Known PPC buses: hv={buses['hv']}, mv={buses['mv']}, lv={buses['lv']}")
        for ci in cands:
            if ci < n_branches:
                fb = int(branch[ci, 0])
                tb = int(branch[ci, 1])
                connects = fb in known_set or tb in known_set
                print(f"     branch[{ci}]: from={fb}, to={tb}, "
                      f"connects_to_known={connects}")
            else:
                print(f"     branch[{ci}]: OUT OF RANGE")

print()

# ── 7. Check for more direct ways to get branch indices ───────────────────
print("=" * 80)
print("7. Looking for direct mappings in net._ppc['internal'] and other places")
print("=" * 80)

# Check _ppc keys
print(f"\n   net._ppc keys: {list(net._ppc.keys())}")

if 'internal' in net._ppc:
    internal = net._ppc['internal']
    print(f"\n   net._ppc['internal'] keys: {list(internal.keys())}")
    for key in sorted(internal.keys()):
        val = internal[key]
        desc = f"type={type(val).__name__}"
        if hasattr(val, 'shape'):
            desc += f", shape={val.shape}"
        elif isinstance(val, (list, dict)):
            desc += f", len={len(val)}"
        elif isinstance(val, (int, float, bool, str, np.integer)):
            desc += f", value={val}"
        print(f"     {key:40s}  {desc}")

# Check specifically for trafo3w-related internal data
print()
print("   --- Searching for trafo3w-related keys in internal ---")
if 'internal' in net._ppc:
    for key in sorted(internal.keys()):
        if 'trafo' in key.lower() or '3w' in key.lower() or 'branch' in key.lower():
            val = internal[key]
            if hasattr(val, 'shape') and val.size < 100:
                print(f"     {key}: {val}")
            elif hasattr(val, 'shape'):
                print(f"     {key}: shape={val.shape}, dtype={val.dtype}")
            elif isinstance(val, dict) and len(val) < 20:
                print(f"     {key}: {val}")
            else:
                print(f"     {key}: type={type(val).__name__}, preview={repr(val)[:200]}")

# Check _pd2ppc_lookups thoroughly
print()
print("   --- All keys in net._pd2ppc_lookups ---")
for key, val in net._pd2ppc_lookups.items():
    desc = f"type={type(val).__name__}"
    if hasattr(val, 'shape'):
        desc += f", shape={val.shape}"
    elif isinstance(val, dict):
        desc += f", keys={list(val.keys())}"
    print(f"     {key:20s}  {desc}")

# Check bus table for auxiliary buses
print()
print("=" * 80)
print("   BONUS: PPC bus table - checking for auxiliary star-point buses")
print("=" * 80)
ppc_bus = net._ppc['bus']
n_ppc_buses = len(ppc_bus)
n_pp_buses = len(net.bus)
print(f"   PP buses: {n_pp_buses},  PPC buses: {n_ppc_buses}")
if n_ppc_buses > n_pp_buses:
    print(f"   -> {n_ppc_buses - n_pp_buses} auxiliary bus(es) added by pandapower")
    print(f"   Auxiliary bus indices (PPC): {list(range(n_pp_buses, n_ppc_buses))}")
    # Actually the aux bus indices in PPC might not be simply beyond n_pp_buses.
    # Let's look at which PPC bus indices appear in trafo3w branches but not in known buses
    if t3w_range is not None:
        aux_buses_found = set()
        for i in range(t3w_start, t3w_end):
            fb = int(branch[i, 0])
            tb = int(branch[i, 1])
            if fb not in all_winding_buses:
                aux_buses_found.add(fb)
            if tb not in all_winding_buses:
                aux_buses_found.add(tb)
        print(f"   Auxiliary (star-point) PPC bus indices found in trafo3w branches: "
              f"{sorted(aux_buses_found)}")

# Check the _pd2ppc_lookups['aux'] if it exists
if 'aux' in net._pd2ppc_lookups:
    print(f"\n   net._pd2ppc_lookups['aux']: {net._pd2ppc_lookups['aux']}")

# Also check net._pd2ppc_lookups for bus mapping details
print()
print("   --- Bus lookup mapping around the boundary ---")
if bus_lookup is not None:
    # Show how PP bus indices map to PPC bus indices
    for i in range(len(net.bus)):
        ppc_idx = int(bus_lookup[i])
        bus_name = net.bus.at[i, 'name'] if i in net.bus.index else '???'
        vn = net.bus.at[i, 'vn_kv'] if i in net.bus.index else '???'
        print(f"     PP bus {i:3d} ({bus_name:30s}, {vn:>7} kV) -> PPC bus {ppc_idx:3d}")

# ── 8. Check if pandapower provides a direct trafo3w -> branch map ─────────
print()
print("=" * 80)
print("8. Exploring pandapower's internal trafo3w handling")
print("=" * 80)

# Check for _pd2ppc_lookups with trafo3w info
try:
    import pandapower.topology as top
    print(f"   pandapower.topology available")
except ImportError:
    print(f"   pandapower.topology not available")

# Check if there's a _pd2ppc mapping for trafo3w
print()
print("   --- Checking net object for trafo3w internal attributes ---")
for attr_name in dir(net):
    if 'trafo3w' in attr_name.lower() or ('_ppc' in attr_name and 'trafo' in attr_name.lower()):
        print(f"     net.{attr_name}")

# Look inside _ppc['internal'] for branch ordering info
print()
if 'internal' in net._ppc:
    internal = net._ppc['internal']
    # Print key arrays that explain branch ordering
    for key in ['branch_is', 'J', 'Ybus', 'Yf', 'Yt']:
        if key in internal:
            val = internal[key]
            if hasattr(val, 'shape'):
                print(f"   internal['{key}']: shape={val.shape}, dtype={val.dtype}")

    # The important ones for understanding ordering
    for key in ['branch_is', 'branch_is_idx']:
        if key in internal:
            val = internal[key]
            print(f"   internal['{key}']: {val}")

# ── 9. Try to understand the ACTUAL ordering pandapower uses ───────────────
print()
print("=" * 80)
print("9. Determine actual branch ordering by matching impedances")
print("=" * 80)

# For each trafo3w branch, check if it connects to aux bus and which winding
if t3w_range is not None:
    t3w_start, t3w_end = t3w_range
    n_t3w_branches = t3w_end - t3w_start
    n_trafo3w = len(net.trafo3w)
    print(f"   trafo3w range has {n_t3w_branches} branches for {n_trafo3w} trafo3w elements")
    print(f"   Expected: {n_trafo3w * 3} branches (3 per trafo3w)")
    print()

    # Group branches by their auxiliary (star-point) bus
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(t3w_start, t3w_end):
        fb = int(branch[i, 0])
        tb = int(branch[i, 1])
        # One bus should be the star-point (aux), the other a winding bus
        if fb in all_winding_buses and tb not in all_winding_buses:
            groups[tb].append((i, fb, 'from_is_winding'))
        elif tb in all_winding_buses and fb not in all_winding_buses:
            groups[fb].append((i, tb, 'to_is_winding'))
        elif fb not in all_winding_buses and tb not in all_winding_buses:
            groups['neither'].append((i, fb, tb))
        else:
            groups['both_winding'].append((i, fb, tb))

    print("   Branches grouped by auxiliary star-point bus:")
    for aux_bus, entries in sorted(groups.items(), key=lambda x: str(x[0])):
        print(f"\n   Aux bus {aux_bus}:")
        for entry in entries:
            if len(entry) == 3 and isinstance(entry[2], str) and entry[2] in ('from_is_winding', 'to_is_winding'):
                br_i, winding_bus, direction = entry
                # Identify which trafo3w and winding
                for t3w_idx, buses in trafo3w_bus_map.items():
                    if winding_bus == buses['hv']:
                        w_name = f"HV (pp bus {buses['hv_pp']})"
                        t3w_name = f"trafo3w {t3w_idx}"
                        break
                    elif winding_bus == buses['mv']:
                        w_name = f"MV (pp bus {buses['mv_pp']})"
                        t3w_name = f"trafo3w {t3w_idx}"
                        break
                    elif winding_bus == buses['lv']:
                        w_name = f"LV (pp bus {buses['lv_pp']})"
                        t3w_name = f"trafo3w {t3w_idx}"
                        break
                else:
                    w_name = "UNKNOWN"
                    t3w_name = "UNKNOWN"
                print(f"     branch[{br_i}]: winding_bus={winding_bus} ({direction}) "
                      f"-> {t3w_name}, {w_name}")
            else:
                print(f"     branch entry: {entry}")

# ── 10. Summarize findings ────────────────────────────────────────────────
print()
print("=" * 80)
print("10. SUMMARY: Correct mapping strategy")
print("=" * 80)
if t3w_range is not None:
    t3w_start, t3w_end = t3w_range
    print(f"\n   The trafo3w branch range is [{t3w_start}, {t3w_end})")
    print(f"   Number of trafo3w elements: {len(net.trafo3w)}")
    print(f"   Expected branches: {len(net.trafo3w) * 3}")
    print(f"   Actual branches in range: {t3w_end - t3w_start}")
    print()
    print("   Branch details:")
    for i in range(t3w_start, t3w_end):
        fb = int(branch[i, 0])
        tb = int(branch[i, 1])
        # Identify
        for t3w_idx, buses in trafo3w_bus_map.items():
            winding_bus = None
            winding_name = None
            if fb in (buses['hv'], buses['mv'], buses['lv']):
                winding_bus = fb
                star_bus = tb
            elif tb in (buses['hv'], buses['mv'], buses['lv']):
                winding_bus = tb
                star_bus = fb
            if winding_bus is not None:
                if winding_bus == buses['hv']:
                    winding_name = 'HV'
                elif winding_bus == buses['mv']:
                    winding_name = 'MV'
                else:
                    winding_name = 'LV'
                offset = i - t3w_start
                print(f"     branch[{i}] (offset {offset}): "
                      f"trafo3w {t3w_idx}, {winding_name} winding "
                      f"(from={fb}, to={tb}, star={star_bus})")
                break
        else:
            print(f"     branch[{i}] (offset {i-t3w_start}): "
                  f"from={fb}, to={tb} -- NOT MATCHED TO ANY TRAFO3W!")

print("\n\nDiagnostic complete.")
