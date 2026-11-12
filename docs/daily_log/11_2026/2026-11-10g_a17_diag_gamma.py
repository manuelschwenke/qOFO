"""Init-only (no loop), E2, new base, 6-h commitment window.  For gamma in {1, 0.5, 0.25} (factor on CEN's STS
bus weights), from M's own frozen model:
  - continuous block lambdas at the selected g_w (raw) and after the block rule (TS -> 1.530, STS -> 1.40, common
    factor to joint 1.530);
  - C2 per MT tap / bank: h_j = sum_i g_v,i H_ij^2 over CEN's rows; owner rows = the owning zone's TS buses on the
    SAME full H ('B'); factor g_w,CEN / g_w,own = h_CEN / h_own_full.
"""
import contextlib, dataclasses, io
import numpy as np
from experiments.ch_11_case2_multisystem import variants as v
from experiments.ch_11_case2_multisystem.ref import scenarios as sc
from experiments.ch_11_case2_multisystem.ref.run_ref import NOISE_SEED
from experiments.runners.multi_tso_dso import run_multi_tso_dso
from configs.paramsets import load_paramset

prov = load_paramset(v.PARAMSET)["provenance"]
T_J, T_STS = float(prov["lambda_max_B"]), float(prov["design_lambda_max_B_sts"])


def lam(Hv, gv, gw, cols):
    A = np.sqrt(gv)[:, None] * Hv[:, cols] / np.sqrt(gw[cols])[None, :]
    return float(np.linalg.eigvalsh(A.T @ A)[-1])


def hook(st):
    cc, cfg, meta, zd, net = st["central_controller"], st["central_cfg"], st["meta"], st["zone_defs"], st["net"]
    H = cc._expand_H_to_der_level(cc._build_sensitivity_matrix())
    n_v = len(cfg.voltage_bus_indices)
    Hv = np.asarray(H[:n_v, :], float)
    gv1 = np.asarray(cc.g_v_per_bus, float)
    hv_bus = {int(b) for hv in meta.hv_networks for b in hv.bus_indices}
    sts_row = np.array([int(b) in hv_bus for b in cfg.voltage_bus_indices])
    gw_as_run, _ = cc._get_per_variable_weights()
    kappa = float(st["config"].central_lambda_target and 1.0)
    der = [int(d) for d in cfg.der_indices]
    ts_der = set(int(s) for s in meta.tso_der_indices)
    n_der, n_gen = len(der), len(cfg.gen_indices)
    ts = [k for k, d in enumerate(der) if d in ts_der] + list(range(n_der, n_der + n_gen))
    sts = [k for k, d in enumerate(der) if d not in ts_der]
    cont = list(range(n_der + n_gen))
    c = st["config"]
    gw0 = np.concatenate([[c.g_w_der if d in ts_der else c.g_w_dso_der for d in der], np.full(n_gen, c.g_w_gen),
                          np.full(len(cfg.oltc_trafo_indices), c.g_w_tso_oltc),
                          np.full(len(cfg.shunt_bus_indices), c.g_w_tso_shunt),
                          np.full(len(cfg.oltc_trafo3w_indices), c.g_w_dso_oltc)])
    print(f"as-run kappa (scope all) = {np.median(np.asarray(gw_as_run)[cont] / gw0[cont]):.4f}; "
          f"oos_oltc {dict(zip(cfg.oltc_trafo_indices, cc._oos_oltc_mask.tolist()))}")
    # owner rows: zone TS buses
    row_of = {int(b): i for i, b in enumerate(cfg.voltage_bus_indices)}
    own_rows = {z: [row_of[int(b)] for b in zd[z].v_bus_indices if int(b) in row_of] for z in zd}
    owner = {}
    for z in zd:
        for t in zd[z].oltc_trafo_indices:
            owner[("tap", int(t))] = z
        for b in zd[z].shunt_bus_indices:
            owner[("bank", int(b))] = z
    n0 = n_der + n_gen
    devs = [("tap", int(t)) for t in cfg.oltc_trafo_indices] + [("bank", int(b)) for b in cfg.shunt_bus_indices]
    for gam in (1.0, 0.5, 0.25):
        gv = np.where(sts_row, gam * gv1, gv1)
        l_ts, l_sts, l_j = lam(Hv, gv, gw0, ts), lam(Hv, gv, gw0, sts), lam(Hv, gv, gw0, cont)
        gw = gw0.copy()
        gw[ts] *= l_ts / T_J
        gw[sts] *= lam(Hv, gv, gw, sts) / T_STS
        s = max(lam(Hv, gv, gw, cont) / T_J, 1.0)
        gw[cont] *= s
        print(f"gamma {gam}: raw TS {l_ts:.3f} STS {l_sts:.3f} joint {l_j:.3f} (kappa_all {max(l_j / T_J, 1):.3f}); "
              f"block rule: g_w TS x{gw[ts[0]] / gw0[ts[0]]:.3f} STS x{gw[sts[0]] / gw0[sts[0]]:.3f} -> "
              f"TS {lam(Hv, gv, gw, ts):.3f} STS {lam(Hv, gv, gw, sts):.3f} joint {lam(Hv, gv, gw, cont):.3f}")
        line = []
        for k, d in enumerate(devs):
            j = n0 + k
            h_cen = float(np.sum(gv * Hv[:, j] ** 2))
            r = own_rows[owner[d]]
            h_own = float(np.sum(gv1[r] * Hv[r, j] ** 2))
            line.append(f"{d[0]}{d[1]}:z{owner[d]} x{h_cen / h_own:.2f}" if h_own > 0 else f"{d[0]}{d[1]}: h_own 0")
        print("   C2-B g_w factor vs owner: " + ", ".join(line))
    return True


cfg = dataclasses.replace(v.resolved("M", 10, sc.events("E2"), NOISE_SEED), verbose=0,
                          unit_commitment_window_s=21600)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    run_multi_tso_dso(cfg, pre_loop_hook=hook)
print("\n".join(ln for ln in buf.getvalue().splitlines()
                if ln.startswith(("as-run", "gamma", "   C2", "  [central]", "  [9]"))), flush=True)
