"""Checks on top of thresholds/usage/candidates (decision note 2026-10-28, sec. 3-5).

    python -X utf8 analysis/tap_engagement/followups.py <out_dir>

Reads thresholds.json, usage.json, candidates.json from <out_dir>. Prints
(1) the necessary-condition check of every recorded tap move,
(2) lambda_max(B) of a tap-price-only adjustment of the selected set,
(3) the share of var(log theta_STS) carried by dso_g_v over the candidates,
(4) a re-selection over the 94 feasible candidates under bounded-threshold
    rules, with the procedure's lexicographic ranking (2 % band on e_v(TS),
    then e_v(DS)). Pure post-processing; no simulation.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tuning import stage1_screen as s1  # noqa: E402

OUT = Path(sys.argv[1])
V_REF = 1.03
u = json.loads((OUT / "usage.json").read_text("utf-8"))
cands = json.loads((OUT / "candidates.json").read_text("utf-8"))

print("(1) necessary condition max_j |V_j - V_ref| >= theta_i at each move (non-event moves)")
for run in ("s1", "sel"):
    for grp in ("sts", "ts"):
        mv = [m for m in u["moves"] if m["run"] == run and m.get("thr_sys") is not None
              and (("|" in m["device"]) == (grp == "sts"))]
        ne = [m for m in mv if not m["near_event_30min"]]
        ratio = [round(m["v_mean_dev"] / m["thr_sys"], 2) for m in ne]
        nec = [max(m["v_max"] - V_REF, V_REF - m["v_min"]) >= m["thr_sys"] for m in ne]
        print(f"  {run:3s} {grp}: moves {len(mv)}, event-coincident {len(mv) - len(ne)}, "
              f"other {len(ne)}; mean-dev/theta {ratio}; condition holds {sum(nec)}/{len(nec)}")

print("\n(2) tap-price-only adjustment of the selected set")
ops = [pickle.load(open(ROOT / f"results/stage1/operating_points/op_rural_700_net82afed5b_{w}.npz", "rb"))
       for w in ["W1", "W2", "W3", "W4", "W5", "W6"]]
W = json.loads((ROOT / "configs/paramsets/ch9_selected_band_2026_10.json").read_text("utf-8"))["fields"]
sts = [lc for op in ops for lc in op.layers if lc.layer == "STS"]
hmax = max(float(lc.colsum_V[i]) for lc in sts for i, c in enumerate(lc.classes) if c == "dso_oltc")
hmin = min(float(lc.colsum_V[i]) for lc in sts for i, c in enumerate(lc.classes) if c == "dso_oltc")


def lam(w, layer):
    rows = [r for r in s1.screen_weights(ops, w) if r.layer == layer]
    r = max(rows, key=lambda r: r.lambda_max)
    return r.lambda_max, r.key, r.condition


print(f"  selected: theta strongest {W['g_w_dso_oltc'] / (2 * W['dso_g_v'] * hmax):.4f}, "
      f"weakest {W['g_w_dso_oltc'] / (2 * W['dso_g_v'] * hmin):.4f}; STS lambda {lam(W, 'STS')}; "
      f"TS lambda {lam(W, 'TS')}")
for th in (0.015, 0.0191, 0.020, 0.025, 0.030):
    w = dict(W)
    w["g_w_dso_oltc"] = 2 * W["dso_g_v"] * hmax * th
    fl = max(s1.integer_floor(lc, *s1._layer_weights(lc, w), {"dso_oltc": w["g_w_dso_oltc"]}) for lc in sts)
    print(f"  theta {th:.4f} (weakest {w['g_w_dso_oltc'] / (2 * W['dso_g_v'] * hmin):.4f}): "
          f"g_w_dso_oltc {w['g_w_dso_oltc']:.4f} (x{w['g_w_dso_oltc'] / W['g_w_dso_oltc']:.3f}); "
          f"STS lambda {lam(w, 'STS')[0]:.4f}; STS integer floor {fl:.4f}; TS lambda {lam(w, 'TS')[0]:.4f}")

print("\n(3) var(log10 theta_STS) over the 242 candidates")
lt = np.log10([r["theta_sts"] for r in cands])
lg = np.log10([r["g_w_dso_oltc"] for r in cands])
lv = np.log10([r["dso_g_v"] for r in cands])
print(f"  var(log theta) {lt.var():.4f}; var(log g_w_dso_oltc) {lg.var():.4f}; var(log dso_g_v) {lv.var():.4f}; "
      f"cov {np.cov(lg, lv, bias=True)[0, 1]:+.4f}; share of dso_g_v {lv.var() / (lg.var() + lv.var()):.2f}")
low = [r for r in cands if r["dso_g_v"] < 1500]
print(f"  candidates with dso_g_v < 1500: {len(low)}, smallest theta_STS among them "
      f"{min(r['theta_sts'] for r in low):.4f}")

print("\n(4) re-selection over the feasible set under added rules (no simulation)")
feas = [r for r in cands if r["feasible"]]
s1row = next(r for r in cands if r["key"] == "3cb66225e1a7")


def order(pool):
    best = min(r["e_v_ts"] for r in pool)
    band = sorted([r for r in pool if r["e_v_ts"] <= 1.02 * best], key=lambda r: r["e_v_ds"])
    rest = sorted([r for r in pool if r not in band], key=lambda r: r["e_v_ts"])
    return band + rest


rules = {
    "as run": lambda r: True,
    "theta_STS <= 0.020": lambda r: r["theta_sts"] <= 0.020,
    "theta_STS <= 0.030": lambda r: r["theta_sts"] <= 0.030,
    "theta_STS <= 0.050": lambda r: r["theta_sts"] <= 0.050,
    "e_v_ds <= 1.5 x Stage-I": lambda r: r["e_v_ds"] <= 1.5 * s1row["e_v_ds"],
    "e_v_ds <= 1.25 x Stage-I": lambda r: r["e_v_ds"] <= 1.25 * s1row["e_v_ds"],
}
for name, f in rules.items():
    pool = [r for r in feas if f(r)]
    o = order(pool)[:3]
    print(f"  {name:26s} n {len(pool):3d}: " + " | ".join(
        f"{r['key']} theta {r['theta_sts']:.4f} e_v_ts {r['e_v_ts']:.5f} e_v_ds {r['e_v_ds']:.5f} "
        f"e_q {r['e_q']:.0f} STS ops {r['sts_ops_5w']}" for r in o))
