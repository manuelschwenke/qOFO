"""Case 1 SVR (code S1), scenario A, on the rerun base: why the plant power flow fails at the 250 Mvar connection
(t = 7300 s).  Runs SVR exactly as run_case1.run_one up to the record of step 364 (t = 7280 s), saves the plant net,
then replays the event offline with power-flow variants.  Outputs next to this file."""
import copy
import os
import pickle
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import dataclasses

import numpy as np
import pandapower as pp

HERE = os.path.dirname(os.path.abspath(__file__))
T_PRE = 7280.0


def capture():
    from experiments.ch_10_case1_svr import run_case1 as rc1
    from experiments.runners import multi_tso_dso as R
    cfg = dataclasses.replace(rc1.resolved("S1", 360, "A"), result_dir=os.path.join(HERE, "run"))
    os.makedirs(cfg.result_dir, exist_ok=True)
    st = {}

    def pre(state):
        st["net"] = state["net"]
        st["meta"] = state["meta"]
        return False

    def hook(step, log):
        if log and log[-1].time_s >= T_PRE - 1e-6:
            with open(os.path.join(HERE, "net_pre_event.pkl"), "wb") as fh:
                pickle.dump(copy.deepcopy(st["net"]), fh)
            with open(os.path.join(HERE, "log_pre_event.pkl"), "wb") as fh:
                pickle.dump(log, fh)
            raise R.EarlyStop()

    R.STEP_HOOK = hook
    t0 = time.time()
    try:
        R.run_multi_tso_dso(cfg, pre_loop_hook=pre)
    except R.EarlyStop:
        pass
    print(f"capture done in {time.time() - t0:.0f} s; distributed_slack={cfg.distributed_slack}, "
          f"enforce_q_lims_plant={cfg.enforce_q_lims_plant}", flush=True)
    return cfg


def summary(n, tag):
    g = n.res_gen.join(n.gen[["name", "max_q_mvar", "min_q_mvar", "vm_pu", "in_service", "slack"]], rsuffix="_set")
    on = g[g.in_service]
    vm = n.res_bus.vm_pu
    print(f"  [{tag}] V min {vm.min():.4f} at bus {vm.idxmin()}, bus 11 {vm.get(11, np.nan):.4f}, "
          f"bus 12 {vm.get(12, np.nan):.4f}")
    for i, r in on.iterrows():
        print(f"     gen {i} {r['name']}: Q {r.q_mvar:8.1f} [{r.min_q_mvar:.0f}, {r.max_q_mvar:.0f}] Mvar, "
              f"V_set {r.vm_pu_set:.4f}, V {r.vm_pu:.4f}{'  AT LIMIT' if r.q_mvar >= r.max_q_mvar - 0.5 else ''}")


def replay(cfg):
    with open(os.path.join(HERE, "net_pre_event.pkl"), "rb") as fh:
        n0 = pickle.load(fh)
    kw = dict(calculate_voltage_angles=True, distributed_slack=cfg.distributed_slack, max_iteration=100)
    print("load 122:", n0.load.loc[122, ["name", "bus", "p_mw", "q_mvar", "in_service"]].to_dict(), flush=True)
    print("taps trafo3w:", n0.trafo3w.tap_pos.astype(int).tolist(), " trafo (MT + network):",
          n0.trafo.tap_pos.astype(int).tolist())
    pp.runpp(n0, init="results", enforce_q_lims=True, run_control=False, **kw)
    summary(n0, "pre-event, converged")

    def ev(n):
        n.load.loc[122, "in_service"] = True
        n.load.loc[122, ["p_mw", "q_mvar"]] = [0.0, 250.0]
        return n

    cases = [
        ("A relays active, q_lims", dict(run_control=True, enforce_q_lims=True)),
        ("B relays held, q_lims", dict(run_control=False, enforce_q_lims=True)),
        ("C relays held, no q_lims", dict(run_control=False, enforce_q_lims=False)),
        ("D relays held, q_lims, iwamoto", dict(run_control=False, enforce_q_lims=True, algorithm="iwamoto_nr")),
    ]
    for tag, extra in cases:
        n = ev(copy.deepcopy(n0))
        try:
            pp.runpp(n, init="results", **kw, **extra)
            print(f"{tag}: converged; trafo3w taps {n.trafo3w.tap_pos.astype(int).tolist()}", flush=True)
            summary(n, tag)
        except Exception as exc:
            print(f"{tag}: FAILED ({type(exc).__name__})", flush=True)
    # continuation of the connected Q with the relays held and Q limits on: largest loadable fraction
    n = copy.deepcopy(n0)
    n.load.loc[122, "in_service"] = True
    n.load.loc[122, "p_mw"] = 0.0
    last = 0.0
    for q in np.arange(25.0, 250.1, 25.0):
        n.load.loc[122, "q_mvar"] = q
        try:
            pp.runpp(n, init="results", enforce_q_lims=True, run_control=False, **kw)
            last = q
        except Exception:
            print(f"continuation (relays held, q_lims): fails at {q:.0f} Mvar, last converged {last:.0f} Mvar",
                  flush=True)
            break
    else:
        print("continuation (relays held, q_lims): converged up to 250 Mvar", flush=True)
        summary(n, "continuation 250")
    if last > 0:
        n.load.loc[122, "q_mvar"] = last
        pp.runpp(n, init="results", enforce_q_lims=True, run_control=False, **kw)
        summary(n, f"continuation {last:.0f} Mvar")


if __name__ == "__main__":
    cfg = capture() if not os.path.exists(os.path.join(HERE, "net_pre_event.pkl")) or "--recapture" in sys.argv else None
    if cfg is None:
        from experiments.ch_10_case1_svr import run_case1 as rc1
        cfg = rc1.resolved("S1", 360, "A")
    replay(cfg)
