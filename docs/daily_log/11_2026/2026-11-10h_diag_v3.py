"""Diagnostic copy of the Case 1 SVR rerun with fix v3 (04c8124), full horizon: keeps a reference to the plant net and
pickles it if the run fails.  Not a run of record."""
import copy, dataclasses, os, pickle
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
from experiments.ch_10_case1_svr import run_case1 as rc1
from experiments.runners import multi_tso_dso as R
HERE = os.path.dirname(os.path.abspath(__file__))
st = {}
cfg = dataclasses.replace(rc1.resolved("S1", 360, "A"), result_dir=os.path.join(HERE, "run_v3"))
os.makedirs(cfg.result_dir, exist_ok=True)
try:
    log = R.run_multi_tso_dso(cfg, pre_loop_hook=lambda s: st.update(net=s["net"]) or False)
    print(f"[diag] run ok, {len(log)} records", flush=True)
except Exception as exc:
    print(f"[diag] run FAILED: {type(exc).__name__}: {exc}; records {len(getattr(exc, 'partial_log', []) or [])}", flush=True)
    pickle.dump(copy.deepcopy(st.get("net")), open(os.path.join(HERE, "v3_failed_net.pkl"), "wb"))
    pickle.dump(list(getattr(exc, "partial_log", []) or []), open(os.path.join(HERE, "v3_partial_log.pkl"), "wb"))
