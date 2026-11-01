#!/usr/bin/env python3
"""
experiments/ch_11_case2_multisystem/mc/evaluate_mc.py
=====================================================

Evaluation of one ``run_mc`` campaign for the thesis (``ch:case3:montecarlo``,
``ch:case_sts:reserve``, ``ch:case3:adverse``, ``tab:case_svr:stats``).  Reads
``runs/*.json`` and the full logs ``runs/*.log.pkl`` and writes into
``<campaign>/evaluation/``:

``episodes.csv``      one row per (seed, variant)
``events.csv``        one row per (seed, variant, disturbance): growth until the
                      next dispatch (max over the areas), withdrawn support, phase
``failures.csv``      every run that is not ``ok``, classified
``paired.csv``        the declared family of paired tests
``summary.json``      everything quoted in the text
``tikz/*.dat``        tables of the pgfplots figures

Statistics, as declared (plan 2026-10-28 §5, Ch 7, Ch 10):

* all paired comparisons are over the comparison window: the first ``N_VALID``
  valid episodes in draw order on which all six variants completed, an episode
  on which any variant failed replaced by the next valid one (amendment of
  2026-11-01, ``run_mc.comparison_window``; ``--rule`` gives the earlier rules,
  ``--no-top-up`` the declared first ``N_VALID`` valid episodes); outcomes as
  ``run_mc.effective_outcome`` (a voltage outside the hard band at the horizon
  end is no failure since 2026-11-01);
* non-inferiority on the pilot error per area, margin DELTA_MPU: decided by the
  one-sided 95 % bootstrap upper bound of the median difference; a claim over
  the areas needs all three (intersection-union, no correction);
* superiority (delta = 0) on e_v, worst bus and time outside the range per area:
  one-sided Wilcoxon signed-rank, zero differences dropped, Holm over the WHOLE
  family of superiority tests (the driver's report corrects per comparison);
* failures: one-sided Clopper--Pearson bound.

Failure classes (``failures.csv``): ``at_event`` -- the run stops in the step of
a disturbance or its reversal, before any controller of that step acts;
``later`` -- it stops at another step.  ``guard`` is the reachability guard's
own flat-start power flow on a single-slack copy failing (ValueError), ``pf``
the plant's power flow.

    python -m experiments.ch_11_case2_multisystem.mc.evaluate_mc <campaign> --jobs 16

Author: Manuel Schwenke / Claude Code (2026-10-30)
"""
from __future__ import annotations

import argparse
import json
import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiments.ch_11_case2_multisystem import metrics_case2 as mc2
from experiments.ch_11_case2_multisystem.mc.run_mc import (
    DELTA_MPU, MASTER_SEED, N_VALID_MAX, WINDOW_RULE, WINDOW_RULES,
    _boot_upper_median, _cp_upper, comparison_window, effective_outcome,
)

V = ["L2", "S1", "S2", "O2", "O3", "M"]
AREAS = ("1", "2", "3")
DT, T_TS = 20.0, 180.0
#: candidate, comparator -- superiority family (NI on e_vp for all of them).
FAMILY = [("O3", "S1"), ("O3", "S2"), ("O3", "O2"), ("O2", "S1"), ("S2", "S1")]
SUP_METRICS = ("e_v", "worst", "t_out")


def _fleet_u_q(log: list) -> float:
    """eq:case3:uq: fleet and time mean over the machines of ALL areas."""
    out = []
    for r in log:
        parts = [mc2._fleet(r, z) for z in (1, 2, 3)]
        parts = [p for p in parts if p is not None and p[0].size]
        if not parts:
            continue
        q = np.concatenate([p[0] for p in parts])
        lo = np.concatenate([p[1] for p in parts])
        hi = np.concatenate([p[2] for p in parts])
        lim = np.where(q >= 0.0, hi, -lo)
        with np.errstate(divide="ignore", invalid="ignore"):
            out.append(float(np.nanmean(np.where(lim > 1e-9, np.abs(q) / lim, np.nan))))
    return float(np.nanmean(out)) if out else float("nan")


def _load(runs: Path, seed: int, v: str) -> Optional[list]:
    p = runs / f"{seed}_{v}.log.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def _log_task(job: Tuple[str, int]) -> Dict[str, Any]:
    """Quantities that need the log, for one episode: U_Q of every variant and
    the recruited flow per STS of the two access steps."""
    runs, seed = Path(job[0]), int(job[1])
    logs = {v: _load(runs, seed, v) for v in V}
    out: Dict[str, Any] = {"seed": seed, "u_q": {}, "recruited": {}}
    for v, log in logs.items():
        if log:
            out["u_q"][v] = _fleet_u_q(log)
    for key, (a, b) in {"O3-O2": ("O3", "O2"), "S2-S1": ("S2", "S1")}.items():
        la, lb = logs.get(a), logs.get(b)
        if not la or not lb or len(la) != len(lb):
            continue
        per = mc2.recruited_flow(la, lb)
        grp: Dict[str, float] = {}
        for k, g in la[0].dso_trafo_group.items():
            c = k.split("|")[-1]
            if c in per:
                grp[g] = grp.get(g, 0.0) + per[c]
        out["recruited"][key] = grp
    return out


def episodes(runs: Path, rule: str = WINDOW_RULE
             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ep, ev, fl = [], [], []
    for p in sorted(runs.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if rule == "2026-11-01" and r["outcome"] != "task_crash":
            r["outcome"] = effective_outcome(r["outcome"], r.get("diag"))
        seed, v = int(r["seed"]), r["variant"]
        row: Dict[str, Any] = {"seed": seed, "variant": v, "outcome": r["outcome"],
                               "cls": r.get("class"), "n_records": r.get("n_records")}
        m = r.get("metrics")
        if m:
            a = [m[z] for z in AREAS]
            row.update({f"e_v_{z}": m[z]["e_v"] * 1e3 for z in AREAS})
            row.update({f"e_vp_{z}": m[z]["e_vp"] * 1e3 for z in AREAS})
            row.update({f"worst_{z}": m[z]["worst"] * 1e3 for z in AREAS})
            row.update({f"t_out_{z}": m[z]["t_out"] * 100.0 for z in AREAS})
            row.update(
                e_v=float(np.sqrt(np.mean([x["e_v"] ** 2 for x in a]))) * 1e3,
                e_vp=float(np.sqrt(np.mean([x["e_vp"] ** 2 for x in a]))) * 1e3,
                worst=max(x["worst"] for x in a) * 1e3,
                t_out=max(x["t_out"] for x in a) * 100.0,
                avr=sum(x["avr_trav"] for x in a),
                ts_taps=sum(x["taps"] for x in a),
                banks=sum(x["shunt_ops"] for x in a),
                r_plus=sum(x["r_plus_mean"] for x in a),
                r_minus=sum(x["r_minus_mean"] for x in a),
                r_plus_min=sum(x["r_plus_min"] for x in a),
                r_minus_min=sum(x["r_minus_min"] for x in a),
                u_q_area=float(np.mean([x["u_q"] for x in a])))
            for z in AREAS:
                row[f"r_plus_{z}"] = m[z]["r_plus_mean"]
                row[f"r_minus_{z}"] = m[z]["r_minus_mean"]
            s = r.get("sts") or {}
            if s:
                row.update(
                    u_sts=float(np.mean([x["u_q_sts"] for x in s.values()])),
                    f_v=float(np.sqrt(np.mean([x["f_v_sts"] ** 2 for x in s.values()]))) * 1e3,
                    margin=min(x["margin_min"] for x in s.values()) * 1e3,
                    c_taps=sum(x["coupler_taps"] for x in s.values()))
            itf = r.get("interface") or {}
            if itf and v in ("O3", "S2"):
                row["e_sts"] = float(np.sqrt(np.mean([x["e_sts_rms"] ** 2 for x in itf.values()])))
        ep.append(row)
        by_t: Dict[float, list] = {}
        for e in r.get("events") or []:
            by_t.setdefault(float(e["t_e_s"]), []).append(e)
        for t_e, rows in by_t.items():
            first = rows[0]
            labels = str(first.get("events", ""))
            disturbance = any(
                (x.split(":")[1] in ("trip", "connect")) and not x.startswith("load:trip")
                for x in labels.split(";") if x.count(":") >= 1)
            ev.append({"seed": seed, "variant": v, "t_e_s": t_e, "phase_s": t_e % T_TS,
                       "events": labels, "disturbance": disturbance,
                       "kind": labels.split(":")[0],
                       "epost": max(float(x["epost"]) for x in rows) * 1e3,
                       "impact": max(float(x["impact"]) for x in rows) * 1e3,
                       "w": float(first.get("w_total_mvar_min", float("nan")))})
        if r["outcome"] != "ok":
            n = int(r.get("n_records") or 0)
            t_fail = (n + 1) * DT
            pre = json.loads((runs.parent / "prescreen" / f"{seed}.json").read_text(encoding="utf-8"))
            times = [float(e["time_s"]) for e in pre["events"]]
            exc = str((r.get("diag") or {}).get("exception", ""))
            short = r["outcome"] == "fail_abort"
            fl.append({
                "seed": seed, "variant": v, "outcome": r["outcome"], "n_records": n,
                "t_fail_min": t_fail / 60.0 if short else float("nan"),
                "where": ("" if not short else "at_event" if any(abs(t - t_fail) < DT / 2 for t in times)
                          else "later"),
                "at_dispatch": bool(short and abs(t_fail % T_TS) < 1e-6),
                "after_event_min": (min([(t_fail - t) / 60.0 for t in times if t <= t_fail], default=float("nan"))
                                    if short else float("nan")),
                "cause": "" if not short else "guard" if "single-slack" in exc else "pf",
                "exception": exc[:90]})
    return pd.DataFrame(ep), pd.DataFrame(ev), pd.DataFrame(fl)


def paired(df: pd.DataFrame, seeds: List[int]) -> pd.DataFrame:
    ok = df[(df.outcome == "ok") & df.seed.isin(seeds)]
    rows = []
    for cand, comp in FAMILY:
        A, B = ok[ok.variant == cand].set_index("seed"), ok[ok.variant == comp].set_index("seed")
        common = A.index.intersection(B.index)
        for z in AREAS:
            d = (A.loc[common, f"e_vp_{z}"] - B.loc[common, f"e_vp_{z}"]).to_numpy(float)
            ub = _boot_upper_median(d)
            rows.append(dict(cand=cand, comp=comp, area=int(z), metric="e_vp", test="NI", n=d.size,
                             median=float(np.median(d)), upper=ub, p=np.nan,
                             p_holm=np.nan, decision="non-inferior" if ub < DELTA_MPU else "not shown"))
            for c in SUP_METRICS:
                d = (A.loc[common, f"{c}_{z}"] - B.loc[common, f"{c}_{z}"]).to_numpy(float)
                try:
                    p = float(wilcoxon(d, alternative="less").pvalue)
                except ValueError:
                    p = 1.0
                rows.append(dict(cand=cand, comp=comp, area=int(z), metric=c, test="sup", n=d.size,
                                 median=float(np.median(d)), upper=_boot_upper_median(d), p=p,
                                 p_holm=np.nan, decision=""))
    out = pd.DataFrame(rows)
    sup = out.index[out.test == "sup"].tolist()
    order = sorted(sup, key=lambda i: out.at[i, "p"])
    m, run = len(order), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * float(out.at[i, "p"])))
        out.at[i, "p_holm"] = run
        out.at[i, "decision"] = "superior" if run < 0.05 else "not shown"
    return out


def q(x, p) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else float("nan")


def tables(df: pd.DataFrame, ev: pd.DataFrame, seeds: List[int], out: Path) -> Dict[str, Any]:
    tz = out / "tikz"
    tz.mkdir(parents=True, exist_ok=True)
    sub = df[df.seed.isin(seeds)]
    ok_all = sub[sub.outcome == "ok"].groupby("seed").variant.nunique()
    common = sorted(ok_all[ok_all == len(V)].index)
    c = sub[sub.seed.isin(common) & (sub.outcome == "ok")]
    res: Dict[str, Any] = {"n_common": len(common), "summary": {}, "classes": {}}
    cols = ["e_v", "e_vp", "worst", "t_out", "e_sts", "ts_taps", "banks", "c_taps", "avr", "u_q",
            "u_sts", "f_v", "margin", "r_plus", "r_minus", "r_plus_min", "r_minus_min"]
    for v in V:
        g = c[c.variant == v]
        res["summary"][v] = {k: {"med": q(g[k], 50), "p95": q(g[k], 95), "p5": q(g[k], 5)}
                             for k in cols if k in g}
        for k in ("c_taps", "ts_taps", "banks"):
            res["summary"][v][k]["mean"] = float(g[k].mean())
        res["summary"][v]["margin_min"] = float(g["margin"].min())
        res["summary"][v]["n_margin_neg"] = int((g["margin"] < 0).sum())
        res["classes"][v] = {cl: {"n": int((g.cls == cl).sum()),
                                  "e_v_med": q(g[g.cls == cl].e_v, 50),
                                  "e_v_p95": q(g[g.cls == cl].e_v, 95),
                                  "e_vp_med": q(g[g.cls == cl].e_vp, 50),
                                  "worst_med": q(g[g.cls == cl].worst, 50)}
                             for cl in ("undisturbed", "single", "multiple")}
    # (a) e_v per area; (c) U_Q: median, P25-P75, P5-P95
    with open(tz / "case3_mc_area.dat", "w") as f:
        f.write("area variant x med q25 q75 q05 q95\n")
        for i, z in enumerate(AREAS):
            for j, v in enumerate(V):
                x = c[c.variant == v][f"e_v_{z}"]
                f.write(f"{z} {v} {i + 1 + (j - 2.5) * 0.14:.3f} {q(x, 50):.3f} {q(x, 25):.3f} {q(x, 75):.3f} {q(x, 5):.3f} {q(x, 95):.3f}\n")
    with open(tz / "case3_mc_uq.dat", "w") as f:
        f.write("variant x med q25 q75 q05 q95\n")
        for j, v in enumerate(V):
            x = c[c.variant == v]["u_q"]
            f.write(f"{v} {j + 1} {q(x, 50):.4f} {q(x, 25):.4f} {q(x, 75):.4f} {q(x, 5):.4f} {q(x, 95):.4f}\n")
    # (b) paired differences of the pilot error against the classical variants
    with open(tz / "case3_mc_diff.dat", "w") as f:
        f.write("comp area x med q25 q75 q05 q95\n")
        for j, comp in enumerate(("S1", "S2")):
            A = c[c.variant == "O3"].set_index("seed")
            B = c[c.variant == comp].set_index("seed")
            for i, z in enumerate(AREAS):
                d = A[f"e_vp_{z}"] - B[f"e_vp_{z}"]
                f.write(f"{comp} {z} {i + 1 + (j - 0.5) * 0.3:.3f} {q(d, 50):.3f} {q(d, 25):.3f} {q(d, 75):.3f} {q(d, 5):.3f} {q(d, 95):.3f}\n")
    # withdrawal against the phase, all completed episodes of the first N
    e = ev[ev.seed.isin(seeds) & ev.disturbance]
    res["phase"] = {}
    with open(tz / "case3_mc_phase.dat", "w") as f:
        f.write("phase " + " ".join(f"{v}_med {v}_p95 {v}_n" for v in ("O2", "O3", "S2", "S1")) + "\n")
        for ph in sorted(e.phase_s.unique()):
            vals = []
            for v in ("O2", "O3", "S2", "S1"):
                x = e[(e.variant == v) & (e.phase_s == ph)].epost
                vals += [f"{q(x, 50):.3f}", f"{q(x, 95):.3f}", str(int(x.size))]
            f.write(f"{ph:.0f} " + " ".join(vals) + "\n")
    for v in ("L2", "S1", "S2", "O2", "O3", "M"):
        x = e[e.variant == v]
        res["phase"][v] = {"n": int(len(x)), "epost_med": q(x.epost, 50), "epost_p95": q(x.epost, 95),
                           "epost_max": float(x.epost.max()) if len(x) else float("nan"),
                           "w_med": q(x.w, 50), "w_p95": q(x.w, 95),
                           "by_kind": {k: {"n": int((x.kind == k).sum()),
                                           "epost_med": q(x[x.kind == k].epost, 50),
                                           "epost_p95": q(x[x.kind == k].epost, 95)}
                                       for k in ("gen", "line", "load")}}
    # error against effort
    with open(tz / "case3_mc_effort.dat", "w") as f:
        f.write("variant avr avr_q25 avr_q75 ev ev_q25 ev_q75 ops\n")
        for v in V:
            g = c[c.variant == v]
            f.write(f"{v} {q(g.avr, 50):.4f} {q(g.avr, 25):.4f} {q(g.avr, 75):.4f} "
                    f"{q(g.e_v, 50):.3f} {q(g.e_v, 25):.3f} {q(g.e_v, 75):.3f} "
                    f"{(g.ts_taps + g.banks).mean():.2f}\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("campaign")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--rule", choices=WINDOW_RULES, default=WINDOW_RULE)
    ap.add_argument("--no-top-up", action="store_true",
                    help="same as --rule 2026-10-07: the first --n valid episodes")
    ap.add_argument("--max-valid", type=int, default=N_VALID_MAX)
    a = ap.parse_args()
    if a.no_top_up:
        a.rule = "2026-10-07"
    root = Path(a.campaign)
    runs = root / "runs"
    out = root / "evaluation"
    out.mkdir(exist_ok=True)
    df, ev, fl = episodes(runs, a.rule)
    pre = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "prescreen").glob("*.json"))]
    valid = sorted(int(r["seed"]) for r in pre if r["valid"])
    outcome = {(int(s), v): o for s, v, o in zip(df.seed, df.variant, df.outcome)
               if o != "task_crash"}
    seeds, skipped = comparison_window(valid[:max(a.max_valid, a.n)], outcome, a.n, a.rule)
    if len(seeds) < a.n:
        print(f"WARNING: comparison window has {len(seeds)} of {a.n} episodes; "
              f"run the driver again to top it up")
    with Pool(a.jobs) as pool:
        extra = pool.map(_log_task, [(str(runs), s) for s in sorted(df.seed.unique())], chunksize=2)
    uq = {(x["seed"], v): u for x in extra for v, u in x["u_q"].items()}
    df["u_q"] = [uq.get((s, v), np.nan) for s, v in zip(df.seed, df.variant)]
    rec = pd.DataFrame([{"seed": x["seed"], "pair": k, "sts": g, "q": q_}
                        for x in extra for k, d in x["recruited"].items() for g, q_ in d.items()])
    df.to_csv(out / "episodes.csv", index=False)
    ev.to_csv(out / "events.csv", index=False)
    fl.to_csv(out / "failures.csv", index=False)
    rec.to_csv(out / "recruited.csv", index=False)
    pr = paired(df, seeds)
    pr.to_csv(out / "paired.csv", index=False)
    res = tables(df, ev, seeds, out)
    res["first_n_seeds"] = [seeds[0], seeds[-1]]
    res["window"] = {"rule": a.rule, "n": len(seeds), "first": seeds[0],
                     "last": seeds[-1], "replaced": skipped,
                     "incomplete": sorted(s for s in seeds
                                          if sum((s, v) in outcome for v in V) < len(V))}
    res["classes_n"] = {str(k): int(v) for k, v in pd.Series([r["class"] for r in pre if int(r["seed"]) in seeds]).value_counts().items()}
    res["draws"] = {"drawn": len(pre), "valid": len(valid)}
    r2 = rec[rec.seed.isin(seeds)]
    res["recruited"] = {k: {"per_sts_med": q(g.q, 50), "per_sts_p95": q(g.q, 95),
                            "by_sts_med": {s: q(h.q, 50) for s, h in g.groupby("sts")}}
                        for k, g in r2.groupby("pair")}
    ok = df[(df.outcome == "ok") & df.seed.isin(seeds)]
    res["steps"] = {}
    for key, (cand, comp) in {"O2->O3": ("O3", "O2"), "S1->S2": ("S2", "S1"), "S1->O2": ("O2", "S1"),
                              "S2->O3": ("O3", "S2"), "O3-M": ("O3", "M")}.items():
        A, B = ok[ok.variant == cand].set_index("seed"), ok[ok.variant == comp].set_index("seed")
        c = A.index.intersection(B.index)
        res["steps"][key] = {k: {"med": float(np.median((A.loc[c, k] - B.loc[c, k]).dropna())),
                                 "p05": q((A.loc[c, k] - B.loc[c, k]), 5),
                                 "p95": q((A.loc[c, k] - B.loc[c, k]), 95)}
                             for k in ("e_v", "e_vp", "worst", "u_q", "u_sts", "ts_taps", "banks", "c_taps", "avr", "r_plus", "r_minus", "f_v")}
        res["steps"][key]["n"] = len(c)
    # failures
    fails: Dict[str, Any] = {}
    n_all = df.groupby("variant").size().to_dict()
    infeasible = sorted(s for s, g in df[df.seed.isin(valid)].groupby("seed")
                        if len(g) == len(V) and (g.outcome != "ok").all())
    res["infeasible"] = infeasible
    for v in V:
        g = df[df.variant == v]
        k = int((g.outcome != "ok").sum())
        k_ex = int(((g.outcome != "ok") & ~g.seed.isin(infeasible)).sum())
        n_ex = int((~g.seed.isin(infeasible)).sum())
        fails[v] = {"n": int(n_all.get(v, 0)), "failed": k, "cp_all": _cp_upper(k, len(g)),
                    "failed_excl_infeasible": k_ex, "n_excl": n_ex, "cp_excl": _cp_upper(k_ex, n_ex),
                    "by_outcome": g.outcome.value_counts().to_dict()}
    res["failures"] = fails
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("window", "n_common", "classes_n", "draws", "infeasible", "failures", "recruited")},
                     indent=1, default=float))
    print(fl.drop(columns=["exception"]).to_string())


if __name__ == "__main__":
    main()
