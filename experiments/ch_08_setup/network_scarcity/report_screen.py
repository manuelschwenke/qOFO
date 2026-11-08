#!/usr/bin/env python3
"""
experiments/ch_08_setup/network_scarcity/report_screen.py
=========================================================

Tables of a ``screen_network`` run: one row per candidate, from the JSONs in
``<run>/ops/``.  Writes ``report.md`` and ``snapshots.csv`` into the run.

    python -m experiments.ch_08_setup.network_scarcity.report_screen <run>
        [--subset MC|X|W]

Definitions (all from ``screen_network``):

* ``scarce_v``: own-reserve ladder of the area; the added reactive demand at
  which the area's lowest 345 kV voltage leaves 0.95 pu with only the area's
  own transmission resources responding and every STS frozen.
* ``own_lim``: the added demand at which those own resources are at their
  Q limits (0 when the area has none committed).
* ``sts_up``: upward Q capability of the area's STS IBGs at the snapshot,
  under the 4120 v2 diagram (``4120``) or with the coupling parks reserved
  (``res``).

Author: Manuel Schwenke / Claude Code (2026-11-08)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

LADDER_MAX = 1500.0
AREA_STS = {2: ("DSO_1", "DSO_2", "DSO_3"), 3: ("DSO_4",)}


def load(run: Path, mode: str = "frozen") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for f in sorted((run / "ops").glob("*.json")):
        stem = f.stem.split("__")
        if (len(stem) == 3) != (mode != "frozen"):
            continue
        d = json.loads(f.read_text())
        r = {k: v for k, v in d.items()
             if not isinstance(v, (list, dict)) and k not in ("tb",)}
        r["uc_offline"] = ",".join(d.get("uc_offline") or [])
        r["g3_off"] = "G3_bus31" in (d.get("uc_offline") or [])
        n1 = d.get("n1") or []
        if n1:
            r["n1_n"] = len(n1)
            r["n1_fail"] = sum(not x["ok"] for x in n1)
            r["n1_noqlim"] = sum(x.get("how") == "noqlim" for x in n1)
            ok = [x for x in n1 if x["ok"]]
            r["n1_vmin"] = min((x["v_min"] for x in ok), default=np.nan)
            r["n1_vmax"] = max((x["v_max"] for x in ok), default=np.nan)
            r["n1_vmin_hv"] = min((x["v_min_hv"] for x in ok), default=np.nan)
            r["n1_ldmax"] = max((x["ld_max"] for x in ok), default=np.nan)
            worst = min(ok, key=lambda x: x["v_min"]) if ok else None
            r["n1_worst"] = worst["el"] if worst else ""
            r["n1_fail_el"] = ",".join(x["el"] for x in n1 if not x["ok"])
        for z, lad in (d.get("ladder") or {}).items():
            for k in ("own_lim", "v_low", "qlim_fail", "pf_fail"):
                v = lad.get(k)
                r[f"lad{z}_{k}"] = LADDER_MAX * 2 if v is None else float(v)
            st = lad.get("steps") or []
            if st:
                r[f"lad{z}_head0"] = st[0][2]
                r[f"lad{z}_tieq0"] = st[0][5]
                # share of an added 300 Mvar met over the ties
                s3 = [x for x in st if x[0] == 300.0]
                if s3:
                    r[f"lad{z}_tie_share300"] = (s3[0][5] - st[0][5]) / 300.0
                    r[f"lad{z}_own_share300"] = (st[0][2] - s3[0][2]) / 300.0
        for z, dsos in AREA_STS.items():
            r[f"sts_up4_{z}"] = sum(d.get(f"cap_up4_{k}", np.nan) for k in dsos)
            r[f"sts_upR_{z}"] = sum(d.get(f"cap_upR_{k}", np.nan) for k in dsos)
            r[f"sts_dn4_{z}"] = sum(d.get(f"cap_dn4_{k}", np.nan) for k in dsos)
            r[f"sts_dnR_{z}"] = sum(d.get(f"cap_dnR_{k}", np.nan) for k in dsos)
        rows.append(r)
    return pd.DataFrame(rows)


def _q(s: pd.Series, q: float) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.quantile(q)) if len(s) else float("nan")


def summary(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cand, g in df.groupby("cand", sort=False):
        ok = g[g.get("base_ok", False) == True]            # noqa: E712
        row: Dict[str, Any] = {"cand": cand, "n": len(g),
                               "init_fail": int((g.init_ok != True).sum()),   # noqa: E712
                               "base_fail": int(((g.init_ok == True)          # noqa: E712
                                                 & (g.get("base_ok") != True)).sum()),
                               "G3_off_%": 100 * float(ok.g3_off.mean()) if len(ok) else np.nan}
        if not len(ok):
            out.append(row)
            continue
        vmin = ok[["v_min_1", "v_min_2", "v_min_3"]].min(axis=1)
        vmax = ok[["v_max_1", "v_max_2", "v_max_3"]].max(axis=1)
        row.update({
            "V_TS min p5": _q(vmin, .05), "V_TS max p95": _q(vmax, .95),
            "V_HV [min,max]": f"[{ok.v_min_hv.min():.3f}, {ok.v_max_hv.max():.3f}]",
            "line max p50/p95/max [%]": f"{_q(ok.line_max, .5):.0f}/"
                                        f"{_q(ok.line_max, .95):.0f}/{ok.line_max.max():.0f}",
            ">100 % [% of OPs]": 100 * float((ok.line_max > 100).mean()),
        })
        if "n1_n" in ok:
            row.update({
                "N-1 fail [cases]": int(ok.n1_fail.sum()),
                "N-1 noqlim [cases]": int(ok.n1_noqlim.sum()),
                "N-1 Vmin p5 / min": f"{_q(ok.n1_vmin, .05):.3f} / {ok.n1_vmin.min():.3f}",
                "N-1 Vmin<0.90 [OPs]": int((ok.n1_vmin < 0.90).sum()),
                "N-1 load max p95 [%]": _q(ok.n1_ldmax, .95),
            })
        for z in (2, 3):
            if f"lad{z}_v_low" not in ok:
                continue
            vl = ok[f"lad{z}_v_low"]
            ol = ok[f"lad{z}_own_lim"]
            row.update({
                f"A{z} own up p50/p10 [Mvar]": f"{_q(ok[f'own_up_{z}'], .5):.0f}/"
                                               f"{_q(ok[f'own_up_{z}'], .1):.0f}",
                f"A{z} scarce_v p50/p10 [Mvar]": f"{_q(vl, .5):.0f}/{_q(vl, .1):.0f}",
                f"A{z} own_lim p50/p10 [Mvar]": f"{_q(ol, .5):.0f}/{_q(ol, .1):.0f}",
                f"A{z} scarce_v<=200 [%]": 100 * float((vl <= 200).mean()),
                f"A{z} tie share of +300": _q(ok.get(f"lad{z}_tie_share300", pd.Series()), .5),
                f"A{z} STS up 4120 p50/p10": f"{_q(ok[f'sts_up4_{z}'], .5):.0f}/"
                                             f"{_q(ok[f'sts_up4_{z}'], .1):.0f}",
                f"A{z} STS up res p50/p10": f"{_q(ok[f'sts_upR_{z}'], .5):.0f}/"
                                            f"{_q(ok[f'sts_upR_{z}'], .1):.0f}",
            })
            scarce = ok[vl <= 300]
            row[f"A{z} n scarce(<=300)"] = len(scarce)
            if len(scarce):
                row[f"A{z} STS up when scarce 4120/res p10"] = (
                    f"{_q(scarce[f'sts_up4_{z}'], .1):.0f}/"
                    f"{_q(scarce[f'sts_upR_{z}'], .1):.0f}")
        out.append(row)
    return pd.DataFrame(out).set_index("cand")


def _md(t: pd.DataFrame) -> str:
    """Markdown table without the optional ``tabulate`` dependency."""
    def f(v):
        return f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)
    head = "| | " + " | ".join(map(str, t.columns)) + " |"
    rule = "|---" * (len(t.columns) + 1) + "|"
    body = ["| " + str(i) + " | " + " | ".join(f(v) for v in r) + " |"
            for i, r in zip(t.index, t.values)]
    return "\n".join([head, rule] + body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--mode", default="frozen")
    ap.add_argument("--subset", default=None, help="op id prefix, e.g. MC or X")
    args = ap.parse_args()
    run = Path(args.run)
    df = load(run, args.mode)
    if args.subset:
        df = df[df.op.str.startswith(args.subset)]
    df.to_csv(run / f"snapshots{'' if args.mode == 'frozen' else '_' + args.mode}.csv",
              index=False)
    tab = summary(df)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    txt = _md(tab.T)
    fails = df[(df.init_ok != True) | (df.get("base_ok") != True)]   # noqa: E712
    lines = [f"# Network screen {run.name} ({args.mode}"
             f"{', ' + args.subset if args.subset else ''})", "", txt, ""]
    if len(fails):
        lines += ["## Initialisation / base failures", ""]
        for _, r in fails.iterrows():
            lines.append(f"- {r.op} {r.cand}: {r.get('err', '') or r.get('screen_error', '')}")
    (run / f"report{'' if args.mode == 'frozen' else '_' + args.mode}"
           f"{'_' + args.subset if args.subset else ''}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
