"""Mismatch loop gain of the TS-OFOs over E2's states (criterion of the E2 declaration).

    mu = max Re eig( G_w^-1 H_m^T D H_t )   over the continuous columns (der, pcc, gen)

H_m: the controller's own model, built at t = 0 (as cached in the run);
H_t: the same sensitivity code on the plant after the stage's events (frozen controls,
     IBG Q(V) loops acting), i.e. what the plant does with the controller's step;
D  : g_v on the voltage rows, tso_g_q_pcc on the Q_PCC rows (the TS layer's own weights).
At t = 0, H_t = H_m and mu equals the layer's continuous lambda_max(B).
Declared bound: mu < 2 in every state (TS-OFO of each area under O2 and O3).

Usage: python -I e2_mismatch.py <network> <paramset> [variants=O2,O3]
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import sys
sys.dont_write_bytecode = True
if os.environ.get("SCREEN_CPUS"):
    import psutil
    psutil.Process().cpu_affinity([int(c) for c in os.environ["SCREEN_CPUS"].split(",")])
sys.path.insert(0, r"F:\qOFO_ref")
os.chdir(r"F:\qOFO_ref")
import contextlib, copy, dataclasses, io, time, traceback, warnings
import numpy as np
import pandapower as pp
warnings.filterwarnings("ignore")

from experiments.ch_11_case2_multisystem import variants as v
from experiments.ch_11_case2_multisystem.ref import scenarios as sc
from experiments.runners.multi_tso_dso import run_multi_tso_dso
from configs.paramsets import load_paramset
from sensitivity.jacobian import JacobianSensitivities
from tuning.stage1_screen import TS_CLASSES, CLASS_TO_WEIGHT

NETWORK, PSET = sys.argv[1], sys.argv[2]
VARIANTS = (sys.argv[3] if len(sys.argv) > 3 else "O2,O3").split(",")
W = load_paramset(PSET)["fields"]
CONT = ("der", "pcc", "gen")
LINES = {"3-4": 4}


PF = {}


def solve(n):
    """The runner's plant settings (config.distributed_slack, config.enforce_q_lims_plant)."""
    pp.runpp(n, numba=False, calculate_voltage_angles=True, max_iteration=100, run_control=True,
             distributed_slack=PF["distributed_slack"], enforce_q_lims=PF["enforce_q_lims"])


def ev_gen(i, on=False):
    def f(n):
        n.gen.at[i, "in_service"] = on
        for t in n.trafo.index:
            if str(n.trafo.at[t, "name"]).startswith(f"MachineTrf|gen{i}_"):
                n.trafo.at[t, "in_service"] = on
    return f


def ev_line(i):
    return lambda n: n.line.__setitem__("in_service", n.line.in_service.where(n.line.index != i, False))


def ev_load(bus, q):
    return lambda n: pp.create_load(n, bus=bus, p_mw=0.0, q_mvar=q, name=f"EV|{bus}|{q}")


def capture(variant):
    cfg = v.resolved(variant, 360, sc.events("E2"), 20261027)
    cfg = dataclasses.replace(cfg, result_dir="", verbose=0, scenario=NETWORK)
    out = {}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        run_multi_tso_dso(cfg, pre_loop_hook=lambda s: out.update(s) or True)
    return out


def layout(ctrl):
    c = ctrl.config
    mp = getattr(c, "der_mapping", None)
    n_der = mp.n_der if mp is not None else len(c.der_indices)
    counts = (n_der, len(c.pcc_trafo_indices), len(c.gen_indices), len(c.oltc_trafo_indices),
              len(c.shunt_bus_indices))
    classes = [cls for cls, k in zip(TS_CLASSES, counts) for _ in range(k)]
    return len(c.voltage_bus_indices), len(c.pcc_trafo_indices), classes


def H_of(ctrl):
    return np.asarray(ctrl._expand_H_to_der_level(ctrl._build_sensitivity_matrix()), float)


def drop_tripped(ctrl, c2, Hm, net):
    """A machine that is out of service has a zero column in H_t; with a zero column j the
    spectrum of M is {0} and that of M without row/column j.  So drop the tripped machines
    from the copy's layout and the same columns from H_m."""
    cfg = ctrl.config
    keep = [k for k, g in enumerate(cfg.gen_indices) if bool(net.gen.at[g, "in_service"])]
    if len(keep) == len(cfg.gen_indices):
        return Hm, []
    n_v, n_q, classes = layout(ctrl)
    gen_cols = [i for i, c in enumerate(classes) if c == "gen"]
    drop = [gen_cols[k] for k in range(len(cfg.gen_indices)) if k not in keep]
    c2.config = dataclasses.replace(
        cfg, gen_indices=[cfg.gen_indices[k] for k in keep],
        gen_bus_indices=[cfg.gen_bus_indices[k] for k in keep] if cfg.gen_bus_indices else [],
        gen_oltc_map={j: cfg.gen_oltc_map[k] for j, k in enumerate(keep) if k in cfg.gen_oltc_map})
    c2._sat_mode = np.asarray(ctrl._sat_mode)[keep]
    return np.delete(Hm, drop, axis=1), [cfg.gen_indices[k] for k in range(len(cfg.gen_indices))
                                         if k not in keep]


def mu(ctrl, Hm, Ht):
    n_v, n_q, classes = layout(ctrl)
    keep = [i for i, c in enumerate(classes) if c in CONT]
    gw = np.array([float(W[CLASS_TO_WEIGHT[classes[i]]]) for i in keep])
    d = np.r_[np.full(n_v, float(W["g_v"])), np.full(n_q, float(W.get("tso_g_q_pcc", 0.0) or 0.0))]
    Am, At = Hm[:n_v + n_q][:, keep], Ht[:n_v + n_q][:, keep]
    M = (Am.T * d) @ At / gw[:, None]
    ev = np.linalg.eigvals(M)
    return float(ev.real.max()), {c: classes.count(c) for c in CONT}


if __name__ == "__main__":
    t0 = time.time()
    G3o, G3b, L = ev_gen(1), ev_gen(1, on=True), ev_load(3, 600.0)
    stages = [("0: t = 0", []), ("1: G3 out", [G3o]), ("2: G3 out + 600", [G3o, L]),
              ("3: G3 back + 600", [G3o, L, G3b]), ("4: + 3-4 open", [G3o, L, G3b, ev_line(4)])]
    print(f"network {NETWORK} | paramset {PSET} | variants {VARIANTS}", flush=True)
    for var in VARIANTS:
        st = capture(var)
        PF.update(distributed_slack=bool(st["config"].distributed_slack),
                  enforce_q_lims=bool(st["config"].enforce_q_lims_plant))
        tso = st["tso_controllers"]
        Hm = {z: H_of(tso[z]) for z in sorted(tso)}
        print(f"== {var}: zones {sorted(tso)} | columns {[mu(tso[z], Hm[z], Hm[z])[1] for z in sorted(tso)]}",
              flush=True)
        print(f"   plant PF settings {PF}", flush=True)
        for tag, evs in [("-: as captured", None)] + stages:
            n = copy.deepcopy(st["net"])
            for e in evs or []:
                e(n)
            try:
                if evs is not None:
                    solve(n)
                jac = JacobianSensitivities(n)
            except Exception as exc:  # noqa: BLE001
                print(f"   {tag:20s} | plant PF/J failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            cells = []
            for z in sorted(tso):
                c2 = copy.copy(tso[z])
                c2.sensitivities = jac
                c2._H_cache = None            # H is memoised; rebuild on the post-event plant
                try:
                    Hmz, gone = drop_tripped(tso[z], c2, Hm[z], n)
                    Ht = H_of(c2)
                    k = n_v_q = None
                    n_v, n_q, _ = layout(c2)
                    if Ht.shape[1] != Hmz.shape[1]:
                        cells.append(f"z{z} shape {Ht.shape} vs {Hmz.shape}")
                        continue
                    r = n_v + n_q
                    dH = float(np.abs(Ht[:r] - Hmz[:r]).max() / max(np.abs(Hmz[:r]).max(), 1e-12))
                    tag_g = f", gen {gone} out" if gone else ""
                    cells.append(f"z{z} {mu(c2, Hmz, Ht)[0]:.2f} (dH {dH:.2f}{tag_g})")
                except Exception as exc:  # noqa: BLE001
                    cells.append(f"z{z} H_t failed ({type(exc).__name__}: {str(exc)[:60]})")
            print(f"   {tag:20s} | " + " | ".join(cells), flush=True)
    print(f"[{time.time() - t0:.0f} s]")
