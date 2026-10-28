"""Usage: python -X utf8 analysis/tap_engagement/thresholds.py <out_dir>

Task 1: effective engagement thresholds per integer column, Stage-I vs selected.

Same machinery as Stage I: tuning.stage1_screen (engage_deviation, design_stage1)
on the cached operating points results/stage1/operating_points, key net82afed5b
(rural_700, DSO_3 x2, banks miqp with STS exclusion) -- the cache the run of
record loaded (stage2a.log).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tuning import stage1_screen as s1  # noqa: E402

OPS = ROOT / "results" / "stage1" / "operating_points"
WINDOWS = ["W1", "W2", "W3", "W4", "W5", "W6"]
ops = [pickle.load(open(OPS / f"op_rural_700_net82afed5b_{w}.npz", "rb")) for w in WINDOWS]


def fields(name):
    return json.loads((ROOT / "configs" / "paramsets" / f"{name}.json").read_text("utf-8"))["fields"]


W_S1 = fields("stage1_start_10_2026_banks")
W_SEL = fields("ch9_selected_band_2026_10")

# --- reproduction of the Stage-I design from the cache -----------------------
w_rep, rep = s1.design_stage1(ops, d_engage_ts=0.0075, d_engage_sts=0.015,
                              d_engage_shunt=0.010)
print("reproduction of stage1_start_10_2026_banks:")
for k in sorted(W_S1):
    print(f"  {k:15s} paramset {W_S1[k]:.10g}  reproduced {w_rep[k]:.10g}  rel {w_rep[k]/W_S1[k]-1:+.2e}")

# Device names: TS columns in zone oltc order (records: zone_oltc_taps[z][k]);
# STS columns in coupling-trafo order (records: DSO_n|trafo_m, ascending m).
STS_TRAFOS = {"DSO_1": [0, 1, 2], "DSO_2": [3, 4, 5], "DSO_3": [6, 7, 8], "DSO_4": [9, 10, 11]}

rows = []
for op in ops:
    for lc in op.layers:
        gv, gq = s1._layer_weights(lc, W_S1)
        gv2, gq2 = s1._layer_weights(lc, W_SEL)
        d1 = s1.engage_deviation(lc, gv, gq, lc.g_w_vector(W_S1))
        d2 = s1.engage_deviation(lc, gv2, gq2, lc.g_w_vector(W_SEL))
        b1 = s1.engage_deviation(lc, gv, gq, lc.g_w_vector(W_S1), reading="single_bus")
        b2 = s1.engage_deviation(lc, gv2, gq2, lc.g_w_vector(W_SEL), reading="single_bus")
        k_cls = {}
        for i, c in enumerate(lc.classes):
            if c not in s1.INTEGER_CLASSES:
                continue
            j = k_cls.get(c, 0)
            k_cls[c] = j + 1
            if lc.layer == "STS":
                dev = f"{lc.area}|trafo_{STS_TRAFOS[lc.area][j]}"
            elif c == "oltc":
                dev = f"z{lc.area}[{j}]"
            else:
                dev = f"z{lc.area}b[{j}]"
            col = lc.S_V.shape[0]
            # uniformity of the tap's voltage column (how well the area mean
            # stands in for the H-weighted mean deviation)
            rows.append(dict(window=op.condition, layer=lc.layer, area=lc.area, cls=c,
                             device=dev, colsum=float(lc.colsum_V[i]), maxabs=float(lc.maxabs_V[i]),
                             n_v=lc.n_v, s1_sys=float(d1[i]), sel_sys=float(d2[i]),
                             s1_bus=float(b1[i]), sel_bus=float(b2[i])))

out = Path(sys.argv[1]) / "thresholds.json"
out.write_text(json.dumps(rows, indent=1), "utf-8")

# factors
for cls, wname, gname in (("dso_oltc", "g_w_dso_oltc", "dso_g_v"),
                          ("oltc", "g_w_tso_oltc", "g_v"),
                          ("shunt", "g_w_tso_shunt", "g_v")):
    f = (W_SEL[wname] / W_S1[wname]) * (W_S1[gname] / W_SEL[gname])
    print(f"{cls:9s} factor sel/S1 = ({W_SEL[wname]:.4g}/{W_S1[wname]:.4g}) x ({W_S1[gname]:.4g}/{W_SEL[gname]:.4g}) = {f:.4f}")

# table: per device, min/max over W1-W6 and per window
import collections
by = collections.defaultdict(dict)
for r in rows:
    by[r["device"]][r["window"]] = r
print()
print(f"{'device':16s} {'cls':8s} " + " ".join(f"{w:>13s}" for w in WINDOWS) + "   (systematic: S1 -> selected, pu)")
for dev in sorted(by, key=lambda d: (by[d]["W1"]["layer"], d)):
    rr = by[dev]
    print(f"{dev:16s} {rr['W1']['cls']:8s} " + " ".join(
        f"{rr[w]['s1_sys']:.4f}->{rr[w]['sel_sys']:.4f}" for w in WINDOWS))
print()
print("single-bus reading (S1 -> selected), W1..W6 min/max over windows:")
for dev in sorted(by, key=lambda d: (by[d]["W1"]["layer"], d)):
    rr = by[dev]
    a = [rr[w]["s1_bus"] for w in WINDOWS]
    b = [rr[w]["sel_bus"] for w in WINDOWS]
    print(f"{dev:16s} S1 {min(a):.4f}-{max(a):.4f}   sel {min(b):.4f}-{max(b):.4f}   "
          f"colsum/(n_v*maxabs) {np.mean([rr[w]['colsum']/(rr[w]['n_v']*rr[w]['maxabs']) if rr[w]['maxabs']>0 else np.nan for w in WINDOWS]):.3f}  maxabs {rr['W1']['maxabs']:.3e}")
