"""
tuning_mc/export_final.py
=========================
Collect everything the weight-selection evaluation rests on into one folder.

Why a script and not a hand-built folder: the export must be re-runnable, so
that a later correction to a result file propagates without anyone having to
remember which tables were copied where.  Every output below is derived from
``results/tuning_mc/campaign_0815/evals/*.json``; nothing is retyped.

Usage::

    python -m tuning_mc.export_final
    python -m tuning_mc.export_final --out results/tuning_mc/campaign_0815/FINAL
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CAMPAIGN = _REPO_ROOT / "results" / "tuning_mc" / "campaign_0815"

#: The screen: lambda_max(M_ii) < 2 for the isolated symmetric per-zone block,
#: with a 25 % reservation.  ``rho_emp_p95`` is that quantity, measured.
SCREEN = 1.5

KNOBS = ("engage_tso_pu", "lambda_tso", "lambda_dso", "tau",
         "engage_dso_pu", "dso_g_v_ratio")
WEIGHTS = ("g_w_der", "g_w_pcc", "g_w_dso_der", "g_w_tso_oltc", "g_w_dso_oltc")
COSTS = ("f_ts", "f_q", "f_ds")


def load(scenario_set: str) -> list[dict[str, Any]]:
    """Every candidate evaluated on one scenario set, newest file wins."""
    out: dict[str, dict] = {}
    for p in sorted((CAMPAIGN / "evals").glob(f"{scenario_set}_*.json")):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                   # noqa: BLE001
            continue
        if "lambda_tso_z1" in j.get("knobs", {}):
            continue                                        # per-zone probe
        out[j["key"]] = j
    return list(out.values())


def row(j: dict) -> dict[str, Any]:
    k = j["knobs"]
    r: dict[str, Any] = {"key": j["key"]}
    r.update({n: k.get(n) for n in KNOBS})
    r.update({n: j.get(n) for n in COSTS})
    r["rho_emp_p95"] = j.get("worst_rho_emp_p95")
    r["admissible"] = (isinstance(r["rho_emp_p95"], (int, float))
                       and r["rho_emp_p95"] <= SCREEN)
    r["worst_tap_ops_per_h"] = j.get("worst_tap_ops_per_h")
    r["worst_reversals_per_h"] = j.get("worst_reversals_per_h")
    r["taps_per_day"] = (24 * j["worst_tap_ops_per_h"]
                         if isinstance(j.get("worst_tap_ops_per_h"), (int, float))
                         else None)
    r.update({n: j.get("weights", {}).get(n) for n in WEIGHTS})
    r["dso_g_v"] = j.get("dso_g_v")
    r["wall_min"] = round(j.get("wall_s", 0) / 60, 1)
    r["limits_source"] = j.get("limits_source")
    return r


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def nondominated(rows: list[dict]) -> list[dict]:
    """Three-criterion non-dominated set over f_ts, f_q, f_ds."""
    ok = [r for r in rows if r["admissible"]
          and all(isinstance(r.get(c), (int, float)) for c in COSTS)]

    def dom(a, b):
        return (all(a[c] <= b[c] for c in COSTS)
                and any(a[c] < b[c] for c in COSTS))

    return [r for r in ok if not any(dom(o, r) for o in ok if o is not r)]


def git_state() -> dict[str, str]:
    def run(*a: str) -> str:
        try:
            return subprocess.run(a, cwd=_REPO_ROOT, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:                                   # noqa: BLE001
            return "unavailable"
    return {"commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": run("git", "status", "--porcelain")[:2000]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuning_mc.export_final")
    p.add_argument("--out", type=Path, default=CAMPAIGN / "FINAL")
    args = p.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "campaign": str(CAMPAIGN.relative_to(_REPO_ROOT)),
        "screen": {"quantity": "rho_emp_p95 = measured lambda_max(M_ii), "
                               "isolated symmetric per-zone block",
                   "bound": 2.0, "reservation": 0.25, "threshold": SCREEN},
        "git": git_state(),
        "files": {},
    }

    # --- 01 all candidates, and the screen applied to them ------------------
    tier1 = [row(j) for j in load("tier1")]
    tier1.sort(key=lambda r: (r["f_ts"] if isinstance(r["f_ts"], (int, float))
                              else 9e9))
    write_csv(out / "01_search" / "all_candidates_design_bank.csv", tier1)
    manifest["files"]["01_search/all_candidates_design_bank.csv"] = {
        "n": len(tier1), "what": "every candidate on the 12-window design bank; "
                                 "'admissible' applies the screen"}

    nd = nondominated(tier1)
    nd.sort(key=lambda r: r["f_ts"])
    write_csv(out / "02_selection" / "nondominated_3criteria.csv", nd)
    manifest["files"]["02_selection/nondominated_3criteria.csv"] = {
        "n": len(nd), "what": "admissible and non-dominated on (f_ts, f_q, f_ds)"}

    # --- 03 confirmation, 04 audit ------------------------------------------
    for tag, sset, what in (
            ("03_confirmation", "confirm",
             "9-window confirmation bank, even ISO weeks, never tuned on"),
            ("04_audit", "audit",
             "4 x 12 h profile-driven windows; wear judged here only")):
        rows = [row(j) for j in load(sset)]
        rows.sort(key=lambda r: (r["f_ts"] if isinstance(r["f_ts"], (int, float))
                                 else 9e9))
        write_csv(out / tag / f"{sset}.csv", rows)
        manifest["files"][f"{tag}/{sset}.csv"] = {"n": len(rows), "what": what}

    # per-transformer wear, long format
    wear_rows = []
    for j in load("audit"):
        k = j["knobs"]
        for scen, per in (j.get("per_transformer_wear") or {}).items():
            for t, v in per.items():
                wear_rows.append({
                    "key": j["key"],
                    **{n: k.get(n) for n in KNOBS},
                    "window": scen, "transformer": t,
                    "ops_per_h": v["ops_per_h"],
                    "ops_per_day": 24 * v["ops_per_h"],
                    "reversals_per_h": v["reversals_per_h"]})
    write_csv(out / "04_audit" / "per_transformer_wear.csv", wear_rows)
    manifest["files"]["04_audit/per_transformer_wear.csv"] = {
        "n": len(wear_rows),
        "what": "tap operations per individual transformer; the budget is "
                "per transformer, not per area or fleet"}

    # --- 00 method artefacts -------------------------------------------------
    meth = out / "00_method"
    meth.mkdir(parents=True, exist_ok=True)
    for src, dst in (("window_selection.json", "scenario_bank.json"),):
        s = CAMPAIGN / src
        if s.exists():
            (meth / dst).write_text(s.read_text(encoding="utf-8"),
                                    encoding="utf-8")
            manifest["files"][f"00_method/{dst}"] = {
                "what": "the design / confirmation / audit windows and how "
                        "they were selected from the capability screen"}
    cpl = CAMPAIGN / "coupled"
    if cpl.exists():
        agg = []
        for f in sorted(cpl.glob("*.json")):
            try:
                agg.append({"file": f.name, **json.loads(f.read_text())})
            except Exception:                               # noqa: BLE001
                pass
        (meth / "coupling_check.json").write_text(json.dumps(agg, indent=1),
                                                  encoding="utf-8")
        manifest["files"]["00_method/coupling_check.json"] = {
            "n": len(agg),
            "what": "isolated vs coupled contraction; supports the statement "
                    "that the screen is a screen"}

    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1),
                                       encoding="utf-8")
    print(f"[export] wrote {out}")
    for k, v in manifest["files"].items():
        print(f"   {k:<48} {v.get('n', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
