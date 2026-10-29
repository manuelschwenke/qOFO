"""Usage: python -X utf8 analysis/tap_engagement/candidates.py <out_dir>

Task 3: effective engagement thresholds over the 242 Stage-II candidates
(union after round 4) and their fate in the filter.

theta_STS,min(cand) = min over STS coupler columns and W1..W6 of
g_w_dso_oltc / (2 dso_g_v colsum_i), computed with stage1_screen.engage_deviation
on the cached operating points (equivalently 0.015 * r / r_S1 with
r = g_w_dso_oltc / dso_g_v, since the Stage-I strongest column sits at 0.015).
Hunting is attributed to device classes from hunting_detail names:
'DSO_n|trafo_m[0]' = STS coupler, 'z[k]' = TS tap ('1[0]' etc.).
"""
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tuning import stage1_screen as s1  # noqa: E402

RUN = ROOT / r"results\ch_09_parameter_selection\0001_2026-10-25_215433_weight_selection_driver_band"
ops = [pickle.load(open(ROOT / f"results/stage1/operating_points/op_rural_700_net82afed5b_{w}.npz", "rb"))
       for w in ["W1", "W2", "W3", "W4", "W5", "W6"]]
sel = json.loads((RUN / "selection" / "selection_2b_round4.json").read_text("utf-8"))
rejected = {r["key"]: r["why"] for r in sel["rejected"]}
ranked = [r["key"] for r in sel["ranked"]]

# provenance of each candidate: stage2a or stage2b round r
origin = {}
for p in (RUN / "stage2a" / "candidates").glob("cand_*.json"):
    origin[p.stem[5:]] = "2a"
for r in range(1, 5):
    for p in (RUN / "stage2b" / f"round{r}" / "candidates").glob("cand_*.json"):
        origin.setdefault(p.stem[5:], f"2b{r}")


def thresholds(w):
    out = defaultdict(list)
    for op in ops:
        for lc in op.layers:
            gv, gq = s1._layer_weights(lc, w)
            dev = s1.engage_deviation(lc, gv, gq, lc.g_w_vector(w))
            for i, c in enumerate(lc.classes):
                if c in s1.INTEGER_CLASSES and np.isfinite(dev[i]):
                    out[c].append(dev[i])
    return {c: float(min(v)) for c, v in out.items()}


rows = []
for p in sorted((RUN / "selection" / "union_2b_round4").glob("cand_*.json")):
    c = json.loads(p.read_text("utf-8"))
    k = c["key"]
    w = c["weights"]
    th = thresholds(w)
    hunt = Counter()
    sts_ops = tso_ops = 0
    max_sts_ops_dev = 0
    for win, pw in c["per_window"].items():
        if win == "W5":
            continue
        for h in pw.get("hunting_detail") or []:
            name = h.get("transformer") if isinstance(h, dict) else str(h[0])
            hunt["sts" if "|" in name else "ts"] += 1
        dev_ops = Counter()
        for m in pw.get("moves") or []:
            if "|" in m[1]:
                sts_ops += 1
                dev_ops[m[1]] += 1
            else:
                tso_ops += 1
        if dev_ops:
            max_sts_ops_dev = max(max_sts_ops_dev, max(dev_ops.values()))
        hunt["bank"] += int(pw.get("n_shunt_hunting") or 0)
    why = rejected.get(k, [])
    agg = c.get("aggregate") or {}
    rows.append(dict(
        key=k, origin=origin.get(k, "?"), baseline=bool(c.get("is_baseline")),
        feasible=k in ranked, rank=(ranked.index(k) + 1) if k in ranked else None,
        r_sts=w["g_w_dso_oltc"] / w["dso_g_v"], theta_sts=th["dso_oltc"],
        theta_ts=th["oltc"], theta_bank=th.get("shunt"),
        dso_g_v=w["dso_g_v"], g_w_dso_oltc=w["g_w_dso_oltc"], g_q=w["g_q"],
        g_w_dso_der=w["g_w_dso_der"], lam=c.get("lambda_max_B"),
        e_v_ts=agg.get("e_v_ts"), e_v_ds=agg.get("e_v_ds"), e_q=agg.get("e_q"),
        sts_ops_5w=sts_ops, tso_ops_5w=tso_ops, max_sts_ops_dev_12h=max_sts_ops_dev,
        hunt_sts=hunt["sts"], hunt_ts=hunt["ts"], hunt_bank=hunt["bank"],
        why=why))

(Path(sys.argv[1]) / "candidates.json").write_text(json.dumps(rows, indent=1), "utf-8")

# ---------------------------------------------------------------- report
def fmt(x, f=".4f"):
    return "-" if x is None else format(x, f)


print(f"candidates {len(rows)}, feasible {sum(r['feasible'] for r in rows)}")
cats = Counter()
for r in rows:
    if r["feasible"]:
        continue
    tags = set()
    for s in r["why"]:
        if "bank hunting" in s:
            tags.add("bank_hunt")
        elif "hunting" in s:
            tags.add("tap_hunt")
        elif "interface-Q" in s:
            tags.add("eq")
        elif "tap ops" in s:
            tags.add("tap_ops")
        else:
            tags.add(s.split(" ")[0])
    for t in tags:
        cats[t] += 1
print("rejection reasons (a candidate may carry several):", dict(cats))

# threshold bins
bins = [0, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 1.0]
print("\ntheta_STS,min bins: n / feasible / tap-hunt rejected (STS-attributed) / median e_v_ts feasible / median e_v_ds feasible")
for lo, hi in zip(bins[:-1], bins[1:]):
    sub = [r for r in rows if lo <= r["theta_sts"] < hi]
    if not sub:
        continue
    feas = [r for r in sub if r["feasible"]]
    th_rej = [r for r in sub if not r["feasible"] and r["hunt_sts"] > 0]
    tap_rej = [r for r in sub if not r["feasible"] and any("hunting reversal" in s and "bank" not in s for s in r["why"])]
    print(f"  [{lo:.3f},{hi:.3f}): n {len(sub):3d}  feasible {len(feas):3d}  tap-hunt rej {len(tap_rej):3d} "
          f"(with STS hunting {len(th_rej):3d})  e_v_ts {fmt(np.median([r['e_v_ts'] for r in feas]) if feas else None)}  "
          f"e_v_ds {fmt(np.median([r['e_v_ds'] for r in feas]) if feas else None)}")

feas = [r for r in rows if r["feasible"]]
print("\nfeasible theta_STS,min: min %.4f p10 %.4f median %.4f p90 %.4f max %.4f" % tuple(
    np.percentile([r["theta_sts"] for r in feas], [0, 10, 50, 90, 100])))
allr = [r for r in rows]
print("all      theta_STS,min: min %.4f p10 %.4f median %.4f p90 %.4f max %.4f" % tuple(
    np.percentile([r["theta_sts"] for r in allr], [0, 10, 50, 90, 100])))

from scipy.stats import spearmanr, mannwhitneyu  # noqa: E402
x = np.array([r["theta_sts"] for r in rows])
f = np.array([r["feasible"] for r in rows])
hs = np.array([r["hunt_sts"] > 0 for r in rows])
print("Mann-Whitney theta(feasible) vs theta(infeasible): p = %.3g; medians %.4f vs %.4f" % (
    mannwhitneyu(x[f], x[~f]).pvalue, np.median(x[f]), np.median(x[~f])))
print("Mann-Whitney theta(STS hunting) vs theta(no STS hunting): p = %.3g; medians %.4f vs %.4f; n %d / %d" % (
    mannwhitneyu(x[hs], x[~hs]).pvalue, np.median(x[hs]), np.median(x[~hs]), hs.sum(), (~hs).sum()))
ev_ds = np.array([r["e_v_ds"] for r in feas]); th_f = np.array([r["theta_sts"] for r in feas])
ev_ts = np.array([r["e_v_ts"] for r in feas])
print("feasible: Spearman(theta_STS, e_v_ds) = %.3f (p %.2g); Spearman(theta_STS, e_v_ts) = %.3f (p %.2g)" % (
    *spearmanr(th_f, ev_ds), *spearmanr(th_f, ev_ts)))
print("feasible: Spearman(dso_g_v, e_v_ts) = %.3f; Spearman(g_q, e_v_ts) = %.3f; Spearman(theta, sts_ops) = %.3f" % (
    spearmanr([r['dso_g_v'] for r in feas], ev_ts)[0], spearmanr([r['g_q'] for r in feas], ev_ts)[0],
    spearmanr(th_f, [r['sts_ops_5w'] for r in feas])[0]))

print("\nhunting attribution over rejected with tap hunting:")
th = [r for r in rows if not r["feasible"] and any("hunting reversal" in s and "bank" not in s for s in r["why"])]
print("  n %d: with STS hunting %d, with TS hunting %d, both %d" % (
    len(th), sum(r["hunt_sts"] > 0 for r in th), sum(r["hunt_ts"] > 0 for r in th),
    sum(r["hunt_sts"] > 0 and r["hunt_ts"] > 0 for r in th)))
print("  theta_STS of STS-hunting rejects: min %.4f median %.4f max %.4f" % tuple(
    np.percentile([r["theta_sts"] for r in th if r["hunt_sts"] > 0], [0, 50, 100])))
print("  theta_STS of TS-only-hunting rejects: min %.4f median %.4f max %.4f" % tuple(
    np.percentile([r["theta_sts"] for r in th if r["hunt_sts"] == 0], [0, 50, 100])) if any(r["hunt_sts"] == 0 for r in th) else "")

print("\ntop 15 feasible by rank:")
for r in sorted(feas, key=lambda r: r["rank"])[:15]:
    print(f"  {r['rank']:3d} {r['key']} {r['origin']:4s} theta_STS {r['theta_sts']:.4f} theta_TS {r['theta_ts']:.4f} "
          f"bank {fmt(r['theta_bank'])} dso_g_v {r['dso_g_v']:9.1f} g_q {r['g_q']:.4f} e_v_ts {r['e_v_ts']:.5f} "
          f"e_v_ds {r['e_v_ds']:.5f} STS ops/5x12h {r['sts_ops_5w']}")
print("\nfeasible with lowest theta_STS:")
for r in sorted(feas, key=lambda r: r["theta_sts"])[:12]:
    print(f"  rank {r['rank']:3d} {r['key']} {r['origin']:4s} theta_STS {r['theta_sts']:.4f} dso_g_v {r['dso_g_v']:9.1f} "
          f"g_q {r['g_q']:.4f} e_v_ts {r['e_v_ts']:.5f} e_v_ds {r['e_v_ds']:.5f} STS ops {r['sts_ops_5w']} "
          f"max/dev/12h {r['max_sts_ops_dev_12h']}")
print("\nby origin: theta_STS median (all / feasible)")
for o in ["2a", "2b1", "2b2", "2b3", "2b4"]:
    sub = [r for r in rows if r["origin"] == o]
    fs = [r for r in sub if r["feasible"]]
    if sub:
        print(f"  {o}: n {len(sub)} feas {len(fs)}  {np.median([r['theta_sts'] for r in sub]):.4f} / "
              f"{np.median([r['theta_sts'] for r in fs]) if fs else float('nan'):.4f}  "
              f"dso_g_v median feas {np.median([r['dso_g_v'] for r in fs]) if fs else float('nan'):.1f}")
