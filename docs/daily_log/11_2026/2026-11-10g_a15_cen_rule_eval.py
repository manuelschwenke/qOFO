"""CEN rule screen (gamma x block loop gain x owner per-switch g_w, R2 = E2, new base): S1 M_g* at the 1.530 cap
(3826639, e5wimi), S2 M145_g* at the gated 1.4524 cap (a45e22a, e5power, pulled into the same folder), against O3 on
the same code and the 27f0651 records (REF 0004 M/O3, Mblk screen): table metrics (run_ref.report, Table 10.5),
per-stage e_v, CEN's own objective split, period-2 check (sign reversals of successive increments) and switching
per device (moves, reversals, median gap to the next move of the same device).

    PYTHONPATH=F:/qOFO_cen python 2026-11-10g_a15_cen_rule_eval.py
"""
import json
import pickle
from pathlib import Path

import numpy as np

from experiments.ch_11_case2_multisystem.ref import run_ref, figures_ref as fr

SCR = Path(r"F:\qOFO_screens\2026-11-10_cen_rule_gamma\E2")
REC = Path(r"F:\qOFO_ref\results\ch_11_case2_multisystem\ref\0004_2026-11-09_185815_run_ref_v5_E2")
MBLK = Path(r"F:\qOFO_screens\2026-11-10_cen_block_kappa\E2")
ST = [(0, 30.67), (30.67, 91.67), (91.67, 152.33), (152.33, 241.0), (241.0, 360.1)]
TAPS = {1: [0, 10], 2: [2, 3, 4], 3: [6, 11]}          # zone_oltc_taps order (zone OLTC lists)
G_V, DSO_G_V = 4e4, 1e4                                 # objective split as in 2026-11-10d section 6
N_BUS = {"1": 8, "2": 12, "3": 10}                      # TS buses per area; 10 buses per STS


def row(m, log):
    a, sts = m["areas"], m["sts"]
    return {"e_v": 1e3 * m["e_v_rms_areas"], "e_vp": 1e3 * m["e_vp_rms_areas"],
            "Vmin_TS": min(min(r.zone_v_min.values()) for r in log),
            "t_out%": 100 * max(float(x["t_out"]) for x in a.values()),
            "U_Q_STS": np.mean([float(s["u_q_sts"]) for s in sts.values()]),
            "f_V_STS": 1e3 * np.sqrt(np.mean([float(s["f_v_sts"]) ** 2 for s in sts.values()])),
            "Vmin_STS": min(min(r.dso_group_v_min_pu.values()) for r in log),
            "avr_trav": sum(float(x["avr_trav"]) for x in a.values()),
            "taps": sum(float(x["taps"]) for x in a.values()),
            "banks": sum(float(x["shunt_ops"]) for x in a.values()),
            "cpl/dev": np.mean([float(s["coupler_taps"]) for s in sts.values()]) / 3.0}


def reversals(x):
    d = np.diff(np.asarray(x, float), axis=0)
    d[np.abs(d) < 1e-6] = 0
    rev = n = 0
    for j in range(d.shape[1]):
        nz = np.sign(d[:, j])[d[:, j] != 0]
        n += len(nz)
        rev += int(np.sum(nz[1:] != nz[:-1]))
    return rev, n


def device_moves(log, key, labels):
    t = np.array([r.time_s for r in log]) / 60.0
    x = np.array([np.concatenate([np.asarray(getattr(r, key)[z], float) for z in sorted(getattr(r, key))])
                  for r in log])
    out = {}
    for j, lab in enumerate(labels):
        d = np.diff(x[:, j])
        idx = np.nonzero(d)[0]
        s = np.sign(d[idx])
        gaps = np.diff(t[idx + 1])
        out[lab] = (len(idx), int(np.sum(s[1:] != s[:-1])), float(np.median(gaps)) if len(gaps) else float("nan"))
    return out


run_ref.report(SCR)
ms = json.loads((SCR / "metrics.json").read_text())
mr = json.loads((REC / "metrics.json").read_text())
mb = json.loads((MBLK / "metrics.json").read_text())
runs = {"O3 rec": (mr["O3"], REC / "O3"), "M rec": (mr["M"], REC / "M"), "Mblk": (mb["Mblk"], MBLK / "Mblk")}
for k in ("O3", "M_g100", "M_g050", "M_g025", "M145_g100", "M145_g050", "M145_g025"):
    if k in ms:
        runs[k] = (ms[k], SCR / k)
logs = {k: pickle.load(open(p / "log.pkl", "rb")) for k, (_, p) in runs.items()}

print("== Table 10.5 metrics (mpu unless noted)")
for k, (m, _) in runs.items():
    print(f"{k:7s}", {kk: round(float(vv), 3) for kk, vv in row(m, logs[k]).items()})

print("== stage means of e_v | per-area e_v | objective split (TS + STS, weighted squared deviation x1e0)")
for k, (m, _) in runs.items():
    log = logs[k]
    t, e = fr._t(log), fr._agg_err(log)
    a = m["areas"]
    sts = m["sts"]
    j_ts = G_V * sum(N_BUS[z] * float(a[z]["e_v"]) ** 2 for z in a)
    j_sts = DSO_G_V * 10 * sum(float(s["f_v_sts"]) ** 2 for s in sts.values())
    print(f"{k:7s} " + " ".join(f"{e[(t >= lo) & (t < hi)].mean():6.2f}" for lo, hi in ST)
          + " | " + " ".join(f"{1e3 * float(a[z]['e_v']):5.2f}" for z in ("1", "2", "3"))
          + f" | J ~ {j_ts:6.1f} + {j_sts:5.1f} = {j_ts + j_sts:6.1f}")

print("== period-2 check: sign reversals of successive increments")
for k in runs:
    log = logs[k]
    gens = sorted(log[0].zone_v_gen)
    vg = np.array([np.concatenate([np.atleast_1d(np.asarray(r.zone_v_gen[z], float)) for z in gens]) for r in log])
    qg = np.array([np.concatenate([np.atleast_1d(np.asarray(r.zone_q_gen[z], float)) for z in gens]) for r in log])
    ders = sorted(log[0].dso_der_q_mvar)
    ibg = np.hstack([np.array([np.atleast_1d(np.asarray(r.dso_der_q_mvar[d], float)) for r in log]) for d in ders])
    out = []
    for nm, x in (("V_gen", vg), ("Q_gen", qg), ("STS IBG Q", ibg)):
        rv, n = reversals(x)
        out.append(f"{nm} {rv}/{n} ({rv / max(n, 1):.0%})")
    out.append(f"mean |dQ| per IBG and 20 s {np.abs(np.diff(ibg, axis=0)).mean():.3f} Mvar")
    print(f"{k:7s} " + "; ".join(out))

print("== switching per device: moves / reversals / median gap to next move of the same device (min)")
tap_lab = [f"tap{t}" for z in sorted(TAPS) for t in TAPS[z]]
for k in runs:
    log = logs[k]
    n_b = sum(len(np.atleast_1d(log[0].zone_tso_shunt_states[z])) for z in log[0].zone_tso_shunt_states)
    dm = device_moves(log, "zone_oltc_taps", tap_lab)
    bm = device_moves(log, "zone_tso_shunt_states", [f"b{i}" for i in range(n_b)])
    tot = lambda d: (sum(v[0] for v in d.values()), sum(v[1] for v in d.values()),
                     float(np.nanmedian([v[2] for v in d.values()])) if d else float("nan"))
    print(f"{k:7s} taps {tot(dm)} banks(state changes) {tot(bm)} | "
          + " ".join(f"{lab} {v[0]}/{v[1]}" for lab, v in dm.items()))
