"""Engagement check of Amendment 1: do the STS couplers engage when the area is
far from the reference, and only then?

    python -X utf8 analysis/tap_engagement/engagement_eval.py <amendment run dir> [out_dir]

Reads ``<run>/probe/<key>__<window>.pkl`` written by
``experiments/ch_09_parameter_selection/ch_9_3_amendment1.py``. The criteria
were fixed before the probe results were seen (2026-10-28):

* **Distance from the reference.** Per STS area, the deviation of the median
  (and mean) of the voltages the controller measures (noisy, controller view,
  as in the live plot) from V_ref = 1.03 pu. Reported as minutes with
  deviation >= 0.010 / 0.015 / 0.020 / 0.025 pu.
* **Far episodes.** Maximal intervals with |median dev| >= 0.020 pu lasting
  >= 10 min. An episode is *served* when at least one coupler of that area
  commits in the correcting direction inside it (tap +1 lowers the STS
  voltage: all coupler columns of H_V are negative). Reported: count, served,
  minutes, median time to the first correcting commit.
* **Unprompted commits.** Commits at |median dev| < 0.010 pu more than 30 min
  after any disturbance and after the 30 min start-up (cold-start anchor and
  initial moves; excluded as in the D1 reduction).
* **Commit ratio.** max_t |c_i + 2 g_u u_i| / (g_w,i + g_u) per coupler (> 1
  = a free column commits); fraction of iterations with any ratio >= 0.9.
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

V_REF = 1.03
DELTAS = (0.010, 0.015, 0.020, 0.025)
FAR, FAR_MIN_S, NEAR, EVENT_GRACE_S, STARTUP_S = 0.020, 600.0, 0.010, 1800.0, 1800.0
EVENTS = {"W1": [17280, 20880], "W3": [17280, 20880],
          "W2": [19440, 20640, 21840, 23040], "W4": [19440, 20640, 21840, 23040],
          "W6": [19440, 20640, 21840]}
LABEL = {"301c285ee52b": "amended #1", "c08b1d3d33ab": "amended #2",
         "fde2e81af6b1": "as frozen", "3cb66225e1a7": "Stage I"}


def episodes(t, mask, min_s):
    out, start = [], None
    for k in range(len(t)):
        if mask[k] and start is None:
            start = k
        if (not mask[k] or k == len(t) - 1) and start is not None:
            end = k if mask[k] else k - 1
            if t[end] - t[start] + (t[1] - t[0]) >= min_s:
                out.append((start, end))
            start = None
    return out


def evaluate(pkl: Path):
    d = pickle.load(pkl.open("rb"))
    w = d["window"]
    rec_t = np.array([r["t"] for r in d["series"]])
    res = {"key": d["key"], "window": w, "row": d["row"], "stored": d["stored_row"],
           "areas": {}, "ts": {}}
    for pkey, p in d["probe"].items():
        layer, area = pkey.split("|")
        t = p["t"]
        if np.isnan(t).any():                      # TS steps may lack sim_time_s
            t = np.where(np.isnan(t), p["it"] * 20.0, t)
        v = p["v"]
        ratio = np.abs(p["lin"]) / p["quad"]
        d_med = np.nanmedian(v, axis=1) - V_REF
        d_mean = np.nanmean(v, axis=1) - V_REF
        if layer == "TS":
            res["ts"][area] = {"names": p["names"],
                               "ratio_max": np.nanmax(ratio, axis=0).round(3).tolist(),
                               "abs_med_dev_max": float(np.nanmax(np.abs(d_med)))}
            continue
        dt = float(np.median(np.diff(t)))
        # commits from the 20 s records, mapped to the probe grid
        commits = []
        for name in p["names"]:
            pos = np.array([r["dso_trafo_tap_pos"].get(name, np.nan) for r in d["series"]])
            for k in np.nonzero(np.diff(pos))[0]:
                tc = rec_t[k + 1]
                # the decision iteration: the last STS step at or before the record
                # that first shows the new tap position
                j = int(np.clip(np.searchsorted(t, tc, side="right") - 1, 0, len(t) - 1))
                commits.append({"device": name, "t": float(tc), "dir": int(np.sign(pos[k + 1] - pos[k])),
                                "d_med": float(d_med[j]), "d_mean": float(d_mean[j]),
                                "v_max": float(np.nanmax(v[j])),
                                "ratio": float(ratio[j, p["names"].index(name)]),
                                "after_event": any(0 <= tc - e <= EVENT_GRACE_S for e in EVENTS[w]),
                                "startup": bool(tc < STARTUP_S)})
        a = {"minutes_med_ge": {f"{x:.3f}": float(np.sum(np.abs(d_med) >= x) * dt / 60) for x in DELTAS},
             "minutes_mean_ge": {f"{x:.3f}": float(np.sum(np.abs(d_mean) >= x) * dt / 60) for x in DELTAS},
             "d_med_p50": float(np.median(d_med)), "d_med_max": float(np.max(d_med)),
             "d_med_min": float(np.min(d_med)),
             "ratio_max": dict(zip(p["names"], np.max(ratio, axis=0).round(3).tolist())),
             "frac_ratio_ge_0p9": float(np.mean(np.max(ratio, axis=1) >= 0.9)),
             "commits": commits}
        eps = []
        for s, e in episodes(t, np.abs(d_med) >= FAR, FAR_MIN_S):
            sign = np.sign(np.mean(d_med[s:e + 1]))
            corr = [c for c in commits if t[s] <= c["t"] <= t[e] + dt and c["dir"] == sign]
            eps.append({"t0": float(t[s]), "t1": float(t[e]), "min": float((t[e] - t[s] + dt) / 60),
                        "sign": int(sign), "served": bool(corr),
                        "t_first_commit_min": (float((min(c["t"] for c in corr) - t[s]) / 60)
                                               if corr else None)})
        a["far_episodes"] = eps
        a["unprompted"] = [c for c in commits if abs(c["d_med"]) < NEAR
                           and not c["after_event"] and not c["startup"]]
        res["areas"][area] = a
    return res


def main():
    run = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else run
    rows = [evaluate(p) for p in sorted((run / "probe").glob("*__W*.pkl"))]
    (out / "engagement_eval.json").write_text(json.dumps(rows, indent=1), "utf-8")
    by = defaultdict(list)
    for r in rows:
        by[r["key"]].append(r)
    print("reproduction (e_v_ts re-simulated vs stored row):")
    for r in rows:
        st = (r["stored"] or {}).get("e_v_ts")
        print(f"  {r['key']} {r['window']}: {r['row']['e_v_ts']:.10g} vs {st if st is None else format(st, '.10g')}"
              f"  diff {r['row']['e_v_ts'] - st if st else float('nan'):+.1e} {r['row']['failure'] or ''}")
    print("\nper key, summed over W1-W4, W6 and the four STS areas:")
    print(f"{'key':13s} {'label':11s} {'STS commits':>11s} {'min |dmed|>=.010/.015/.020/.025':>34s} "
          f"{'far ep.':>7s} {'served':>6s} {'far min':>7s} {'t1st med':>8s} {'unprompted':>10s}")
    for k, rs in by.items():
        n_c = sum(len(a["commits"]) for r in rs for a in r["areas"].values())
        mins = [sum(a["minutes_med_ge"][f"{x:.3f}"] for r in rs for a in r["areas"].values()) for x in DELTAS]
        eps = [e for r in rs for a in r["areas"].values() for e in a["far_episodes"]]
        served = sum(e["served"] for e in eps)
        t1 = [e["t_first_commit_min"] for e in eps if e["t_first_commit_min"] is not None]
        unp = sum(len(a["unprompted"]) for r in rs for a in r["areas"].values())
        print(f"{k:13s} {LABEL.get(k, ''):11s} {n_c:11d} {' / '.join(f'{m:6.0f}' for m in mins):>34s} "
              f"{len(eps):7d} {served:6d} {sum(e['min'] for e in eps):7.0f} "
              f"{(np.median(t1) if t1 else float('nan')):8.1f} {unp:10d}")
    print("\nper key, window, area: median dev p50/max, minutes |dmed|>=0.02, commits, far episodes (served), ratio max")
    for k, rs in by.items():
        for r in sorted(rs, key=lambda r: r["window"]):
            for area, a in r["areas"].items():
                eps = a["far_episodes"]
                print(f"  {LABEL.get(k, k):11s} {r['window']} {area}: dmed p50 {a['d_med_p50']:+.4f} max {a['d_med_max']:+.4f} "
                      f"| >=0.02: {a['minutes_med_ge']['0.020']:5.0f} min | commits {len(a['commits']):2d} "
                      f"| far {len(eps)} ({sum(e['served'] for e in eps)}) | ratio max "
                      f"{' '.join(f'{x:.2f}' for x in a['ratio_max'].values())}")
    print("\ncommits (median deviation at the commit):")
    for k, rs in by.items():
        for r in rs:
            for area, a in r["areas"].items():
                for c in a["commits"]:
                    print(f"  {LABEL.get(k, k):11s} {r['window']} {c['device']:15s} t {c['t'] / 3600:5.2f} h dir {c['dir']:+d} "
                          f"dmed {c['d_med']:+.4f} dmean {c['d_mean']:+.4f} vmax {c['v_max']:.4f} ratio {c['ratio']:.3f}"
                          f"{' (event)' if c['after_event'] else ''}"
                          f"{' (start-up)' if c['startup'] else ''}")


if __name__ == "__main__":
    main()
