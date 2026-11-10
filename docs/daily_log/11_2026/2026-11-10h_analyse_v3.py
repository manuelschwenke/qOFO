"""Step 366 of Case 1 SVR (fix v3): which solve of the state the failing PF tried converges?"""
import copy, os, pickle, warnings
import numpy as np, pandapower as pp
from pandapower.auxiliary import ControllerNotConverged, LoadflowNotConverged
from core.plant import runpp_relays_held
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
n0 = pickle.load(open(os.path.join(HERE, "v3_failed_net.pkl"), "rb"))
log = pickle.load(open(os.path.join(HERE, "v3_partial_log.pkl"), "rb"))
kw = dict(calculate_voltage_angles=True, distributed_slack=True, max_iteration=100)
print("res_bus NaN:", int(n0.res_bus.vm_pu.isna().sum()), " gen Q/limit:",
      {n0.gen.at[i, "name"]: (round(float(n0.res_gen.q_mvar.get(i, np.nan)), 1), n0.gen.at[i, "max_q_mvar"]) for i in n0.gen.index[n0.gen.in_service]})
r = log[-1]
print("last record t", r.time_s, "zone V min", {z: round(v, 3) for z, v in r.zone_v_min.items()})
cases = [("A warm, relays held, Q(V) on, Q lims", lambda n: runpp_relays_held(n, init="results", enforce_q_lims=True, **kw)),
         ("B warm, relays held, Q(V) on, no Q lims", lambda n: runpp_relays_held(n, init="results", enforce_q_lims=False, **kw)),
         ("C warm, no control, Q lims", lambda n: pp.runpp(n, init="results", enforce_q_lims=True, run_control=False, **kw)),
         ("D warm, no control, no Q lims", lambda n: pp.runpp(n, init="results", enforce_q_lims=False, run_control=False, **kw)),
         ("E dc init, no control, no Q lims", lambda n: pp.runpp(n, init="dc", enforce_q_lims=False, run_control=False, **kw))]
for tag, f in cases:
    n = copy.deepcopy(n0)
    try:
        f(n)
        g3 = float(n.res_gen.q_mvar[1])
        print(f"{tag}: converged; V min {n.res_bus.vm_pu.min():.4f} (bus {n.res_bus.vm_pu.idxmin()}), G3 Q {g3:.1f}, "
              f"IBG sum {n.res_sgen.q_mvar.sum():.1f}")
    except (LoadflowNotConverged, ControllerNotConverged) as exc:
        print(f"{tag}: FAILED ({type(exc).__name__})")
# loadability: scale every load Q/P from the converged pre-profile state? Use D's solution and scale loads up until failure
n = copy.deepcopy(n0)
try:
    pp.runpp(n, init="results", enforce_q_lims=True, run_control=False, **kw); base_ok = True
except Exception:
    base_ok = False
print("C repeated:", base_ok)
