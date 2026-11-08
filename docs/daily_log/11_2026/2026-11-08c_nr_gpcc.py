"""Effect of g_w_pcc on rural_700_north_res (other weights = v4).

sigma_V: STS share of the first unconstrained step's voltage effect under a uniform
deviation, effect_i = (g_v / g_w,i) * c_i^2, c_i = |sum_j H_V[j,i]| (continuous columns).
lambda: v4 screen (banks excluded) per TS layer, G3 on / G3 off.
"""
import sys, glob, os, pickle, json, numpy as np
sys.path.insert(0, r"F:\qOFO_net")
from tuning import stage1_screen as s1

OPS = r"C:\Users\MSCHWE~1\AppData\Local\Temp\259\claude\Z--Python-Projekte-qOFO\56eb5879-34eb-40a0-aeee-65e3879c4433\scratchpad\nr_ops"
W = {k: float(v) for k, v in json.load(open(r"F:\qOFO_net\configs\paramsets\ch9_selected_uc_2026_11_v4.json"))["fields"].items()
     if k != "sts_coupler_hold_after_bank"}
data = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(os.path.join(OPS, "*.pkl")))]
on = [d for d in data if all(d["extra"]["zones"][2]["gen_in_service"])]
off = [d for d in data if not all(d["extra"]["zones"][2]["gen_in_service"])]


def sigma_v(z, w):
    nv, npc, ng, nd = z["n_v"], z["n_pcc"], z["n_gen"], z["n_der"]
    c = np.abs(z["H"][:nv].sum(axis=0))
    e_pcc = sum(w["g_v"] / w["g_w_pcc"] * c[i] ** 2 for i in range(nd, nd + npc))
    e_der = sum(w["g_v"] / w["g_w_der"] * c[i] ** 2 for i in range(nd))
    e_gen = sum(w["g_v"] / w["g_w_gen"] * c[i] ** 2 for k, i in enumerate(range(nd + npc, nd + npc + ng))
                if z["gen_in_service"][k])
    return e_pcc / (e_pcc + e_der + e_gen)


print(f"G3 on: {len(on)} points, G3 off: {len(off)} points; ceiling {s1.select_ceiling():.3f}")
print("g_w_pcc | sigma_V z2 on | z2 off | z3 | lambda TS_1 / TS_2 / TS_3 worst (on | off)")
for g in (0.873, 0.38, 0.19, 0.15, 0.12):
    w = dict(W, g_w_pcc=g)
    s2on = np.median([sigma_v(d["extra"]["zones"][2], w) for d in on])
    s2off = np.median([sigma_v(d["extra"]["zones"][2], w) for d in off])
    s3 = np.median([sigma_v(d["extra"]["zones"][3], w) for d in data])
    lam = {}
    for lab, grp in (("on", on), ("off", off)):
        rows = s1.screen_weights([d["op"] for d in grp], w, exclude_classes=("shunt",))
        lam[lab] = {k: max(r.lambda_max for r in rows if r.key == k) for k in ("TS_1", "TS_2", "TS_3")}
    print(f"{g:7.3f} | {s2on:12.2f} | {s2off:6.2f} | {s3:4.2f} | "
          f"{lam['on']['TS_1']:.2f} / {lam['on']['TS_2']:.2f} / {lam['on']['TS_3']:.2f}  |  "
          f"{lam['off']['TS_1']:.2f} / {lam['off']['TS_2']:.2f} / {lam['off']['TS_3']:.2f}")
