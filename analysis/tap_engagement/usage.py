"""Usage: python -X utf8 analysis/tap_engagement/usage.py <out_dir>  (after thresholds.py)

Task 2: measured tap and bank usage vs STS/TS voltage deviation, from the
margin_check pickles of run 0001 (no new simulation).

Selected fde2e81af6b1 and Stage-I point 3cb66225e1a7, W1-W4, W6, 12 h, 20 s records.
Reference: v_setpoint_pu = 1.03 on every STS and TS bus (configs/config.py:309,
runner passes np.full(n_v, v_set) to every DSO controller).
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = (ROOT / "results" / "ch_09_parameter_selection"
       / "0001_2026-10-25_215433_weight_selection_driver_band" / "margin_check")
# Event times [s] of tuning.scenarios_mc_v2._condition_events at 12 h: W1/W3
# gen-2 trip at 0.40 h-fraction and restore 1 h later; W2/W4 load ramp at
# bus 5 from 0.45 (4 x 20 min); W6 load shedding at bus 27 from 0.45 (3 x 20 min).
KEYS = {"sel": "fde2e81af6b1", "s1": "3cb66225e1a7"}
WINDOWS = ["W1", "W2", "W3", "W4", "W6"]
V_REF = 1.03
EVENTS = {"W1": [17280, 20880], "W3": [17280, 20880],
          "W2": [19440, 20640, 21840, 23040], "W4": [19440, 20640, 21840, 23040],
          "W6": [19440, 20640, 21840]}
OUT = Path(sys.argv[1])
thr = json.loads((OUT / "thresholds.json").read_text("utf-8"))
TH = {(r["window"], r["device"]): r for r in thr}

summary = {}
moves_all = []
for tag, key in KEYS.items():
    for w in WINDOWS:
        d = pickle.load(open(RUN / f"{key}__{w}.pkl", "rb"))
        s = d["series"]
        t = np.array([r["t"] for r in s])
        # --- STS taps
        devs = sorted(s[0]["dso_trafo_tap_pos"])
        tap = {k: np.array([r["dso_trafo_tap_pos"][k] for r in s], float) for k in devs}
        areas = sorted(s[0]["dso_group_v_mean_pu"])
        vmean = {a: np.array([r["dso_group_v_mean_pu"][a] for r in s]) for a in areas}
        vmax = {a: np.array([r["dso_group_v_max_pu"][a] for r in s]) for a in areas}
        vmin = {a: np.array([r["dso_group_v_min_pu"][a] for r in s]) for a in areas}
        qder = {a: np.array([r["dso_group_q_der_mvar"][a] for r in s]) for a in areas}
        qmin = {a: np.array([r["dso_group_q_der_min_mvar"][a] for r in s]) for a in areas}
        qmax = {a: np.array([r["dso_group_q_der_max_mvar"][a] for r in s]) for a in areas}
        # --- TS taps and banks
        zones = sorted(s[0]["zone_oltc_taps"])
        ztap = {f"z{z}[{k}]": np.array([r["zone_oltc_taps"][z][k] for r in s], float)
                for z in zones for k in range(len(s[0]["zone_oltc_taps"][z]))}
        bank = {f"z{z}b[{k}]": np.array([r["zone_tso_shunt_states"][z][k] for r in s], float)
                for z in sorted(s[0]["zone_tso_shunt_states"])
                for k in range(len(s[0]["zone_tso_shunt_states"][z]))}
        zvmean = {z: np.array([r["zone_v_mean"][z] for r in s]) for z in zones}
        zvmax = {z: np.array([r["zone_v_max"][z] for r in s]) for z in zones}
        zvmin = {z: np.array([r["zone_v_min"][z] for r in s]) for z in zones}
        res = {"sts": {}, "ts": {}, "bank": {}, "area": {}, "zone": {}}
        for k, x in {**tap, **ztap, **bank}.items():
            dx = np.diff(x)
            idx = np.nonzero(dx)[0]
            grp = "sts" if "|" in k else ("bank" if "b[" in k else "ts")
            res[grp][k] = {"ops": int(len(idx)), "steps": float(np.abs(dx).sum()),
                           "first": float(x[0]), "last": float(x[-1]),
                           "min": float(x.min()), "max": float(x.max())}
            for i in idx:
                tm = t[i + 1]
                ev = any(0 <= tm - e <= 1800 for e in EVENTS[w])
                rec = {"run": tag, "window": w, "device": k, "t_s": float(tm),
                       "dir": float(np.sign(dx[i])), "near_event_30min": ev}
                if grp == "sts":
                    a = k.split("|")[0]
                    rec.update(v_mean_dev=float(vmean[a][i] - V_REF), v_max=float(vmax[a][i]),
                               v_min=float(vmin[a][i]), q_der=float(qder[a][i]),
                               q_der_min=float(qmin[a][i]), q_der_max=float(qmax[a][i]),
                               thr_sys=TH[(w, k)][f"{tag}_sys"])
                elif grp == "ts":
                    z = int(k[1])
                    rec.update(v_mean_dev=float(zvmean[z][i] - V_REF), v_max=float(zvmax[z][i]),
                               v_min=float(zvmin[z][i]),
                               thr_sys=TH.get((w, k), {}).get(f"{tag}_sys"))
                else:
                    z = int(k[1])
                    rec.update(v_mean_dev=float(zvmean[z][i] - V_REF), v_max=float(zvmax[z][i]),
                               v_min=float(zvmin[z][i]))
                moves_all.append(rec)
        for a in areas:
            dm = vmean[a] - V_REF
            dmax = vmax[a] - V_REF
            # strongest coupler threshold of this area in this window
            ths = [TH[(w, k)][f"{tag}_sys"] for k in devs if k.startswith(a + "|")]
            res["area"][a] = {
                "mean_dev_p50": float(np.median(dm)), "mean_dev_p95": float(np.percentile(dm, 95)),
                "mean_dev_max": float(dm.max()), "mean_dev_min": float(dm.min()),
                "max_dev_p50": float(np.median(dmax)), "max_dev_max": float(dmax.max()),
                "v_max_max": float(vmax[a].max()), "v_min_min": float(vmin[a].min()),
                "min_thr": float(min(ths)),
                "frac_mean_gt_minthr": float(np.mean(np.abs(dm) > min(ths))),
                "frac_vmax_gt_minthr": float(np.mean(dmax > min(ths))),
                "min_thr_minus_mean_dev_max": float(min(ths) - np.abs(dm).max()),
                "minutes_above_1p08": float(np.sum(vmax[a] > 1.08) * 20 / 60),
                "der_util_max": float(np.max(np.where(qder[a] >= 0, qder[a] / qmax[a], qder[a] / qmin[a]))),
                "der_util_p50": float(np.median(np.where(qder[a] >= 0, qder[a] / qmax[a], qder[a] / qmin[a]))),
            }
        for z in zones:
            dm = zvmean[z] - V_REF
            res["zone"][z] = {"mean_dev_p50": float(np.median(dm)), "mean_dev_absmax": float(np.abs(dm).max()),
                              "max_dev_max": float((zvmax[z] - V_REF).max()),
                              "min_dev_min": float((zvmin[z] - V_REF).min())}
        summary[f"{tag}_{w}"] = res

(OUT / "usage.json").write_text(json.dumps({"summary": summary, "moves": moves_all}, indent=1), "utf-8")

# ---- print
for tag in ("s1", "sel"):
    print(f"\n===== {tag} ({KEYS[tag]}) =====")
    for w in WINDOWS:
        res = summary[f"{tag}_{w}"]
        sts = {k: v["ops"] for k, v in res["sts"].items() if v["ops"]}
        ts = {k: v["ops"] for k, v in res["ts"].items() if v["ops"]}
        bk = {k: v["ops"] for k, v in res["bank"].items() if v["ops"]}
        print(f"{w}: STS ops {sts or '-'} | TS ops {ts or '-'} | bank ops {bk or '-'}")
        for a, r in res["area"].items():
            print(f"    {a}: mean dev p50 {r['mean_dev_p50']:+.4f} p95 {r['mean_dev_p95']:+.4f} "
                  f"max {r['mean_dev_max']:+.4f} | vmax dev p50 {r['max_dev_p50']:+.4f} max {r['max_dev_max']:+.4f} "
                  f"| min thr {r['min_thr']:.4f} | frac(|mean dev|>thr) {r['frac_mean_gt_minthr']:.3f} "
                  f"frac(vmax dev>thr) {r['frac_vmax_gt_minthr']:.3f} | min>1.08 {r['minutes_above_1p08']:.0f} "
                  f"| DER util p50 {r['der_util_p50']:.2f} max {r['der_util_max']:.2f}")
        for z, r in res["zone"].items():
            print(f"    zone {z}: mean dev p50 {r['mean_dev_p50']:+.4f} |max| {r['mean_dev_absmax']:.4f} "
                  f"vmax dev {r['max_dev_max']:+.4f} vmin dev {r['min_dev_min']:+.4f}")
print("\n===== moves =====")
for m in moves_all:
    extra = ""
    if "v_mean_dev" in m:
        extra = f"mean dev {m['v_mean_dev']:+.4f} vmax {m['v_max']:.4f} vmin {m['v_min']:.4f}"
    if m.get("thr_sys") is not None:
        extra += f" thr {m['thr_sys']:.4f}"
    if "q_der" in m:
        extra += f" Qder {m['q_der']:.1f} [{m['q_der_min']:.1f},{m['q_der_max']:.1f}]"
    print(f"{m['run']:3s} {m['window']} {m['device']:15s} t={m['t_s']:7.0f} dir {m['dir']:+.0f} ev {int(m['near_event_30min'])} {extra}")
