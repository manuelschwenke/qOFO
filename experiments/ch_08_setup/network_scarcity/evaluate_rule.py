#!/usr/bin/env python3
"""
experiments/ch_08_setup/network_scarcity/evaluate_rule.py
=========================================================

The selection rule declared on 2026-11-08 14:35 (daily log
``2026-11-08b_network_scarcity_screen.md`` section 6), evaluated on a
``screen_network`` run over the MC prescreen operating points:

* H1 initialisation rate >= base rate - 2 percentage points;
* H2 no N-1 outage element (frozen) that fails to converge in more than 5 % of
  the OPs, unless it does so in the base as well;
* H3 post-contingency 345 kV V_min, p5 over OPs of the worst case, >= 0.90 pu;
* H4 max 345 kV line loading p95 <= base p95 + 10 percentage points;
* H5 the G1 operability rule (separate run, ``g1_operability``);
* ranking: S1 share of OPs with area-2 ``scarce_v`` <= 400 Mvar, then S2 the
  median ``scarce_v``.

    python -m experiments.ch_08_setup.network_scarcity.evaluate_rule <run>

Author: Manuel Schwenke / Claude Code (2026-11-08)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

BASE = "B"


def main() -> None:
    run = Path(sys.argv[1])
    rows, fails = [], {}
    for f in sorted((run / "ops").glob("MC*.json")):
        if len(f.stem.split("__")) != 2:
            continue
        d = json.loads(f.read_text())
        c = d["cand"]
        r = {"cand": c, "op": d["op"], "init": bool(d.get("init_ok")) and bool(d.get("base_ok"))}
        if r["init"]:
            ok = [x for x in d["n1"] if x["ok"]]
            r["n1_vmin"] = min(x["v_min"] for x in ok) if ok else np.nan
            r["line_max"] = d["line_max"]
            r["scarce_v"] = d["ladder"]["2"]["v_low"] if d["ladder"]["2"]["v_low"] is not None else 3000.0
            r["scarce_v3"] = d["ladder"]["3"]["v_low"] if d["ladder"]["3"]["v_low"] is not None else 3000.0
            r["g3_off"] = "G3_bus31" in (d.get("uc_offline") or [])
            sts = sum(d.get(f"cap_up4_{k}", 0.0) for k in ("DSO_1", "DSO_2", "DSO_3"))
            stsR = sum(d.get(f"cap_upR_{k}", 0.0) for k in ("DSO_1", "DSO_2", "DSO_3"))
            r["sts_up4"], r["sts_upR"] = sts, stsR
            fails.setdefault(c, Counter()).update(x["el"] for x in d["n1"] if not x["ok"])
        rows.append(r)
    df = pd.DataFrame(rows)
    n_ops = df.groupby("cand").op.nunique()
    out = []
    base_init = df[df.cand == BASE].init.mean()
    ok_b = df[(df.cand == BASE) & df.init]
    base_ld95 = ok_b.line_max.quantile(0.95)
    base_bad = {e for e, k in fails.get(BASE, {}).items() if k > 0.05 * ok_b.op.nunique()}
    for c, g in df.groupby("cand"):
        ok = g[g.init]
        bad = {e: k for e, k in fails.get(c, {}).items() if k > 0.05 * len(ok)}
        new_bad = {e: k for e, k in bad.items() if e not in base_bad}
        r = {"cand": c, "n_ops": int(n_ops[c]), "init_rate": g.init.mean(),
             "H1": g.init.mean() >= base_init - 0.02,
             "N-1 >5% elements": ",".join(f"{e}:{k}" for e, k in sorted(bad.items())),
             "H2": not new_bad,
             "N-1 Vmin p5": ok.n1_vmin.quantile(0.05), "H3": ok.n1_vmin.quantile(0.05) >= 0.90,
             "line p95": ok.line_max.quantile(0.95),
             "H4": ok.line_max.quantile(0.95) <= base_ld95 + 10.0,
             "S1 share scarce_v<=400": (ok.scarce_v <= 400).mean(),
             "S2 median scarce_v": ok.scarce_v.median(),
             "share scarce_v<=300": (ok.scarce_v <= 300).mean(),
             "area3 share scarce_v<=400": (ok.scarce_v3 <= 400).mean(),
             "G3 off share": ok.g3_off.astype(bool).mean(),
             "S1 with G3 on": (ok[~ok.g3_off.astype(bool)].scarce_v <= 400).mean(),
             "STS up p10 when scarce, 4120": ok[ok.scarce_v <= 400].sts_up4.quantile(0.1),
             "STS up p10 when scarce, reserved": ok[ok.scarce_v <= 400].sts_upR.quantile(0.1)}
        out.append(r)
    t = pd.DataFrame(out).set_index("cand").T
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    txt = t.to_string()
    print(txt)
    (run / "rule_evaluation.txt").write_text(txt)
    df.to_csv(run / "rule_snapshots.csv", index=False)


if __name__ == "__main__":
    main()
