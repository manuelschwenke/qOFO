"""Fix v4 (runpp_warm_staged, Q limits first) on the state the step-366 solve of Case 1 SVR failed on."""
import copy, os, pickle, warnings
from core.plant import runpp_warm_staged, snapshot_results
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
n0 = pickle.load(open(os.path.join(HERE, "v3_failed_net.pkl"), "rb"))
kw = dict(calculate_voltage_angles=True, distributed_slack=True, max_iteration=50, max_iter=300)
n = copy.deepcopy(n0)
how = runpp_warm_staged(n, snapshot_results(n0), enforce_q_lims=True, **kw)
print(f"v4 on the step-366 state: converged ({how}); V min {n.res_bus.vm_pu.min():.4f}, G3 Q {float(n.res_gen.q_mvar[1]):.1f}")
