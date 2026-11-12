"""The 2026-11-10 fix (core.plant.runpp_warm_staged) on the captured pre-event plant of Case 1 SVR (t = 7280 s)."""
import copy
import os
import pickle

import pandapower as pp
from pandapower.auxiliary import ControllerNotConverged, LoadflowNotConverged

from core.plant import runpp_holding_taps, runpp_warm_staged

HERE = os.path.dirname(os.path.abspath(__file__))
n0 = pickle.load(open(os.path.join(HERE, "net_pre_event.pkl"), "rb"))
kw = dict(calculate_voltage_angles=True, distributed_slack=True, max_iteration=100, max_iter=300)
snap = (n0.res_bus.copy(), n0.sgen["q_mvar"].copy())
n = copy.deepcopy(n0)
n.load.loc[122, ["in_service", "p_mw", "q_mvar"]] = [True, 0.0, 250.0]
try:
    runpp_holding_taps(n, run_control=True, enforce_q_lims=True, init="results", **kw)
    print("normal end-of-step solve (loops + relays, Q limits): converged")
except (LoadflowNotConverged, ControllerNotConverged) as exc:
    print(f"normal end-of-step solve (loops + relays, Q limits): FAILED ({type(exc).__name__})")
try:
    runpp_holding_taps(copy.deepcopy(n), run_control=True, enforce_q_lims=False, init="flat", **kw)
    print("old fallback (flat start, no Q limits): converged")
except (LoadflowNotConverged, ControllerNotConverged) as exc:
    print(f"old fallback (flat start, no Q limits): FAILED ({type(exc).__name__})")
how = runpp_warm_staged(n, snap[0], snap[1], enforce_q_lims=True, **kw)
g = n.res_gen.join(n.gen[["name", "max_q_mvar"]])
print(f"new warm staged re-solve: converged ({how})")
print(f"  V min {n.res_bus.vm_pu.min():.4f} at bus {n.res_bus.vm_pu.idxmin()}; bus 11 {n.res_bus.vm_pu[11]:.4f}")
print("  gens:", {r['name']: (round(r.q_mvar, 1), r.max_q_mvar) for _, r in g[n.gen.in_service].iterrows()})
print(f"  STS IBG Q sum {n.res_sgen.q_mvar.sum():.1f} Mvar (pre-event {n0.res_sgen.q_mvar.sum():.1f})")
