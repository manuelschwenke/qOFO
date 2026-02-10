"""
Diagnostic script: Why does tap_pos +/-1 on a pandapower 3W transformer
have NO effect on power-flow results?
"""
import pandapower as pp
import pandas as pd
import numpy as np
import copy

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print(f"pandapower version: {pp.__version__}")
print()

# 1. Reproduce the test fixture
def make_network():
    net = pp.create_empty_network()
    bus_ehv = pp.create_bus(net, vn_kv=380.0, name="EHV Bus")
    bus_hv  = pp.create_bus(net, vn_kv=110.0, name="HV Bus")
    bus_lv  = pp.create_bus(net, vn_kv=20.0,  name="LV Bus")
    pp.create_ext_grid(net, bus=bus_ehv, vm_pu=1.03, name="Grid")
    pp.create_transformer3w_from_parameters(
        net,
        hv_bus=bus_ehv, mv_bus=bus_hv, lv_bus=bus_lv,
        sn_hv_mva=300.0, sn_mv_mva=300.0, sn_lv_mva=75.0,
        vn_hv_kv=380.0, vn_mv_kv=110.0, vn_lv_kv=20.0,
        vk_hv_percent=9.0, vk_mv_percent=5.0, vk_lv_percent=7.5,
        vkr_hv_percent=0.26, vkr_mv_percent=0.06, vkr_lv_percent=0.06,
        pfe_kw=90.0, i0_percent=0.04,
        shift_mv_degree=0.0, shift_lv_degree=150.0,
        tap_side="hv", tap_neutral=0,
        tap_min=-9, tap_max=9,
        tap_pos=0, tap_step_percent=1.222,
        name="Trafo3W_Test",
    )
    pp.create_load(net, bus=bus_hv, p_mw=150.0, q_mvar=50.0, name="HV Load")
    pp.create_load(net, bus=bus_lv, p_mw=40.0,  q_mvar=10.0, name="LV Load")
    return net

net0 = make_network()
tap_cols = [c for c in net0.trafo3w.columns if "tap" in c.lower()]
print("=== Tap-related columns in trafo3w ===")
print(net0.trafo3w[tap_cols].to_string())
print()

# 2. Run power flow at tap_pos = 0, +1, -1
results = {}
for pos in [0, 1, -1]:
    net = copy.deepcopy(net0)
    net.trafo3w.at[0, "tap_pos"] = pos
    pp.runpp(net, verbose=False)
    vm = net.res_bus["vm_pu"].values.copy()
    va = net.res_bus["va_degree"].values.copy()
    results[pos] = (vm, va)
    print(f"tap_pos = {pos:+d}:")
    for i in range(len(vm)):
        print(f"  Bus {i} ({net.bus.at[i, chr(39) + 'name' if False else 'name']:>8s}):  Vm = {vm[i]:.8f} pu,  Va = {va[i]:.6f} deg")
    print()

vm0 = results[0][0]
vmp = results[1][0]
vmm = results[-1][0]
print("=== Voltage DIFFERENCES vs tap_pos=0 ===")
for i in range(len(vm0)):
    print(f"  Bus {i}: dV(+1) = {vmp[i]-vm0[i]:+.10e},  dV(-1) = {vmm[i]-vm0[i]:+.10e}")
print()

if np.allclose(vmp, vm0) and np.allclose(vmm, vm0):
    print("*** CONFIRMED: tap_pos has NO effect on power flow results! ***")
    print()
else:
    print("tap_pos DOES affect power flow results.")
    print()

# 3. Try with tap_at_star_point = True
print("=== Test with tap_at_star_point = True ===")
results_star = {}
for pos in [0, 1, -1]:
    net = copy.deepcopy(net0)
    net.trafo3w.at[0, "tap_at_star_point"] = True
    net.trafo3w.at[0, "tap_pos"] = pos
    pp.runpp(net, verbose=False)
    vm = net.res_bus["vm_pu"].values.copy()
    results_star[pos] = vm
    print(f"  tap_at_star_point=True, tap_pos={pos:+d}: Vm = {vm}")

vm0s = results_star[0]
vmps = results_star[1]
vmms = results_star[-1]
print(f"  dV(+1) = {vmps - vm0s}")
print(f"  dV(-1) = {vmms - vm0s}")
if np.allclose(vmps, vm0s) and np.allclose(vmms, vm0s):
    print("  *** Still NO effect with tap_at_star_point=True ***")
else:
    print("  tap_at_star_point=True DOES change behavior!")
print()

# 4. Try different tap_side values
print("=== Test different tap_side values ===")
for side in ["hv", "mv", "lv"]:
    results_side = {}
    for pos in [0, 5]:
        net = copy.deepcopy(net0)
        net.trafo3w.at[0, "tap_side"] = side
        net.trafo3w.at[0, "tap_pos"] = pos
        pp.runpp(net, verbose=False)
        vm = net.res_bus["vm_pu"].values.copy()
        results_side[pos] = vm
    diff = results_side[5] - results_side[0]
    has_effect = not np.allclose(diff, 0)
    print(f"  tap_side={side!r:5s}, dV(pos=5 vs pos=0) = {diff}  effect={has_effect}")
print()

# 5. Inspect the internal ppc branch data to see if tap ratio changes
print("=== Checking internal ppc branch tap ratios ===")
for pos in [0, 5]:
    net = copy.deepcopy(net0)
    net.trafo3w.at[0, "tap_pos"] = pos
    pp.runpp(net, verbose=False)
    if hasattr(net, "_ppc"):
        ppc = net._ppc
        branch = ppc["branch"]
        print(f"  tap_pos={pos}: branch tap ratios (col 8) = {branch[:, 8]}")
        print(f"  tap_pos={pos}: branch shift     (col 9) = {branch[:, 9]}")
    else:
        print(f"  No _ppc attribute found for tap_pos={pos}")
print()

# 6. Look at the trafo3w -> trafo conversion (pandapower internal)
print("=== Inspecting _calc_tap_from_dataframe source ===")
try:
    from pandapower.build_branch import _calc_tap_from_dataframe
    import inspect
    src = inspect.getsource(_calc_tap_from_dataframe)
    lines = src.split(chr(10))
    for line in lines[:50]:
        print("  ", line)
except Exception as e:
    print(f"  Could not inspect: {e}")
print()

# 7. Look for trafo3w-specific build functions
print("=== Looking for trafo3w build functions ===")
try:
    import pandapower.build_branch as bb
    funcs = [f for f in dir(bb) if "3w" in f.lower() or "trafo3w" in f.lower()]
    print(f"  Functions: {funcs}")
except Exception as e:
    print(f"  Error: {e}")
print()

# 8. Check if trafo3w creates internal auxiliary buses/trafos
print("=== After runpp: checking for internal auxiliary structures ===")
net = copy.deepcopy(net0)
net.trafo3w.at[0, "tap_pos"] = 3
pp.runpp(net, verbose=False)
print(f"  Number of buses in net.bus: {len(net.bus)}")
print(f"  Number of trafos in net.trafo: {len(net.trafo)}")
print(f"  Number of trafo3w in net.trafo3w: {len(net.trafo3w)}")
if len(net.trafo) > 0:
    print("  net.trafo tap columns:")
    t_tap_cols = [c for c in net.trafo.columns if "tap" in c.lower()]
    print(net.trafo[t_tap_cols].to_string())
print()

print("=== DIAGNOSIS COMPLETE ===")
