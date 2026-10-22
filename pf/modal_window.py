"""
pf/modal_window.py
==================
One operating window of the plant-time-constant modal analysis
(``ch:timescales:plant``; handoff 2026-09-22, steps M4/M5).

Run it AFTER the window has been synced and verified in separate processes::

    python -m pf.pf_sync <snap>.json --phase full --study-case 02_RMS_CoSim
    python -m pf.pf_sync <snap>.json --phase full --study-case 02_RMS_CoSim --dry-run

A ``--snapshot`` argument does not move the PF operating point: this script
linearises whatever the project holds and only uses the snapshot for the
per-park droop and capability data.

Inside ONE PF process (``GetApplication*`` is once per process):

1. record every QVPRE parameter vector (44) and TAPCTRL ``outserv`` (24) to
   ``originals.json`` -- the restore source, also usable after a crash via
   ``--restore``;
2. diagnostic at the synced operating point BEFORE any change: per park
   ``|V_LF - Vanchor|`` against ``db``, ``|qset S_n - Q_LF|``, limit pinning;
3. re-anchor every QVPRE at the synced point with the logic of
   ``PowerFactoryPlant._anchor_qv_precontrollers`` (``qset = Q_LF/S_n``,
   ``Vanchor = V_LF``, ``Kdroop = 1/slope``, ``qmin/qmax`` from the park's
   operating diagram at its snapshot P); ``db`` untouched here;
4. take the TAPCTRL blocks out of service (tap changers are not part of the
   linearisation; their 5 s delay enters as a device property);
5. per droop case, set ``db`` on all 44 blocks (``db_wide``: every droop
   inert; ``db_zero``: every droop on its slope -- at ``db = 0`` the DSL's
   ``qcorr = max(veff-db,0) + min(veff+db,0)`` is identically ``veff``, so
   there is no kink at the anchor), ComInc, ``cmd_modal(participation=True)``;
6. restore QVPRE parameters and TAPCTRL flags, read them back, verify;
7. acceptance check (droop modes must differ between the cases) and the
   per-window reduction (``analysis.modal_reduce``).

Usage::

    python -m pf.modal_window --window W1 --snapshot export/snapshots/<snap>.json \
        [--sync-log F] [--dryrun-log F] [--db-wide 0.1]
    python -m pf.modal_window --restore results/screening/modal_campaign/W1/<stamp>/originals.json

Author: Manuel Schwenke / Claude Code (2026-09-22)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.der_qv_local_loop import _qv_capability  # noqa: E402
from export.dynamic_snapshot import load_snapshot  # noqa: E402
from pf.naming import build_name_map  # noqa: E402
from pf.screening import (  # noqa: E402
    COMMOD_PARTICIPATION_OPTS, RMS_STUDY_CASE, ScreeningContext, cmd_modal,
)
from pf.session import PFSessionError, connect, get_all  # noqa: E402
from pf.tap_ctrl import DSL_NAME as TAPCTRL_TYPE  # noqa: E402
from pf.wecc_apply import (  # noqa: E402
    QVPRE_ELEMENT_NAME, QVPRE_PARAM_ORDER, QV_SLOPE_PU_DEFAULT, set_qv_params,
)

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "results" / "screening" / "modal_campaign"
DB_WIDE_PU = 0.1
#: Read-back tolerance: PF returns written doubles slightly off
#: (``pf_sync._REL_EPS``).
REL_EPS = 2e-6
N_QV = len(QVPRE_PARAM_ORDER)


# ── object handles ─────────────────────────────────────────────────────────

def _all_in_netdat(app, cls: str) -> List[Any]:
    """Every object of ``cls`` under Network Data, in service or not
    (``GetCalcRelevantObjects`` is not relied on for out-of-service ones)."""
    return list(app.GetProjectFolder("netdat").GetContents(f"*.{cls}", True) or [])


def resolve_parks(app, doc) -> List[Dict[str, Any]]:
    names = build_name_map(doc)
    gens = {o.loc_name: o for o in get_all(app, "ElmGenstat")}
    comps = {c.loc_name: c for c in get_all(app, "ElmComp")}
    parks = []
    for key, rec in doc["model"]["sgen"].items():
        idx = int(key)
        nm = names[("sgen", idx)]
        gen, comp = gens.get(nm), comps.get(f"WECC_{nm}")
        if gen is None or comp is None:
            raise PFSessionError(f"sgen[{idx}] {nm!r}: park or WECC composite missing")
        pre = next((d for d in comp.GetContents()
                    if d.GetClassName() == "ElmDsl"
                    and d.loc_name == QVPRE_ELEMENT_NAME), None)
        if pre is None:
            raise PFSessionError(f"sgen[{idx}] {nm!r}: no QVPRE block")
        cub = gen.GetAttribute("bus1")
        parks.append({"idx": idx, "name": nm, "gen": gen, "pre": pre,
                      "term": cub.cterm if cub is not None else None, "rec": rec})
    return parks


def resolve_taps(app) -> List[Any]:
    return [d for d in _all_in_netdat(app, "ElmDsl")
            if getattr(d.GetAttribute("typ_id"), "loc_name", "") == TAPCTRL_TYPE]


def read_params(pre) -> List[float]:
    return [float(pre.GetAttribute(f"params:{i}")) for i in range(N_QV)]


def write_params(pre, values: List[float]) -> None:
    for i, v in enumerate(values):
        pre.SetAttribute(f"params:{i}", float(v))


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(REL_EPS * max(abs(a), abs(b)), 1e-12)


# ── originals / restore ────────────────────────────────────────────────────

def record_originals(parks, taps) -> Dict[str, Any]:
    return {
        "recorded": datetime.now().isoformat(timespec="seconds"),
        "param_order": list(QVPRE_PARAM_ORDER),
        "qvpre": {p["pre"].GetFullName(): read_params(p["pre"]) for p in parks},
        "tapctrl_outserv": {t.GetFullName(): int(t.GetAttribute("outserv")) for t in taps},
    }


def restore(app, orig: Dict[str, Any]) -> Dict[str, Any]:
    """Write the recorded values back and verify them by read-back."""
    app.ResetCalculation()
    by_name = {o.GetFullName(): o for o in _all_in_netdat(app, "ElmDsl")}
    missing = [n for n in list(orig["qvpre"]) + list(orig["tapctrl_outserv"])
               if n not in by_name]
    if missing:
        raise PFSessionError(f"restore: {len(missing)} object(s) not found, e.g. {missing[:3]}")
    for name, vals in orig["qvpre"].items():
        write_params(by_name[name], vals)
    for name, flag in orig["tapctrl_outserv"].items():
        by_name[name].SetAttribute("outserv", int(flag))
    worst, bad = 0.0, []
    for name, vals in orig["qvpre"].items():
        back = read_params(by_name[name])
        for k, (a, b) in enumerate(zip(vals, back)):
            worst = max(worst, abs(a - b))
            if not _close(a, b):
                bad.append((name, QVPRE_PARAM_ORDER[k], a, b))
    tap_bad = [n for n, f in orig["tapctrl_outserv"].items()
               if int(by_name[n].GetAttribute("outserv")) != int(f)]
    db_before = sorted({round(v[3], 12) for v in orig["qvpre"].values()})
    db_after = sorted({round(read_params(by_name[n])[3], 12) for n in orig["qvpre"]})
    return {"qvpre_blocks": len(orig["qvpre"]), "max_abs_dev": worst,
            "mismatches": bad, "tap_mismatches": tap_bad,
            "db_before": db_before, "db_after": db_after,
            "ok": not bad and not tap_bad and db_before == db_after}


# ── diagnostics and anchoring ──────────────────────────────────────────────

def run_ldf(app) -> None:
    if app.GetFromStudyCase("ComLdf").Execute():
        raise PFSessionError("load flow at the synced operating point failed")


def harvest(parks) -> None:
    """Load-flow values per park; all harvested before any parameter write
    (writing a DSL parameter invalidates PF's results)."""
    for p in parks:
        g = p["gen"]
        p["sn"] = float(g.GetAttribute("sgn"))
        p["q_lf"] = float(g.GetAttribute("m:Q:bus1"))
        p["p_lf"] = float(g.GetAttribute("m:P:bus1"))
        p["v_lf"] = float(p["term"].GetAttribute("m:u"))
        p["stored"] = dict(zip(QVPRE_PARAM_ORDER, read_params(p["pre"])))


def diag_before(parks) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    for p in parks:
        s = p["stored"]
        dv = p["v_lf"] - s["Vanchor"]
        db_thesis = float(p["rec"].get("qv_deadband_pu") or 0.0)
        rows.append({
            "park": p["name"], "v_lf": p["v_lf"], "vanchor_stored": s["Vanchor"],
            "dv": dv, "db_stored": s["db"], "db_snapshot": db_thesis,
            "inside_stored_db": abs(dv) <= s["db"],
            "inside_snapshot_db": abs(dv) <= db_thesis,
            "q_lf_mvar": p["q_lf"], "qset_stored_mvar": s["qset"] * p["sn"],
            "dq_mvar": s["qset"] * p["sn"] - p["q_lf"],
            "kdroop_stored": s["Kdroop"],
            "qmin_stored": s["qmin"], "qmax_stored": s["qmax"],
        })
    agg = {
        "n": len(rows),
        "max_abs_dv": max(abs(r["dv"]) for r in rows),
        "n_dv_gt_1e-4": sum(abs(r["dv"]) > 1e-4 for r in rows),
        "inside_stored_db": sum(r["inside_stored_db"] for r in rows),
        "outside_stored_db": sum(not r["inside_stored_db"] for r in rows),
        "inside_snapshot_db": sum(r["inside_snapshot_db"] for r in rows),
        "outside_snapshot_db": sum(not r["inside_snapshot_db"] for r in rows),
        "db_stored_values": sorted({round(r["db_stored"], 9) for r in rows}),
        "max_abs_dq_mvar": max(abs(r["dq_mvar"]) for r in rows),
    }
    return rows, agg


def reanchor(parks) -> Dict[str, Any]:
    """plant.py's anchoring pass, standalone. ``db`` keeps its stored value."""
    pinned, outside_box, zero_box, pinned_to_lf, kd = [], [], [], [], set()
    for p in parks:
        rec = p["rec"]
        if "op_diagram" not in rec or "p_mw" not in rec:
            raise PFSessionError(f"sgen[{p['idx']}]: snapshot lacks op_diagram/p_mw")
        slope = float(rec.get("qv_slope_pu") or QV_SLOPE_PU_DEFAULT)
        q_min, q_max = _qv_capability(p["sn"], str(rec["op_diagram"]), float(rec["p_mw"]))
        qset = p["q_lf"] / p["sn"]
        lo, hi = q_min / p["sn"], q_max / p["sn"]
        if qset < lo - 1e-6 or qset > hi + 1e-6:
            # The snapshot's load-flow Q lies outside the park's own capability
            # box (W3-W5 TSO parks, seeded Q).  Clipping would leave the RMS
            # initial state off the load-flow point; pinning the box at the
            # load-flow Q keeps the linearisation AT the snapshot operating
            # point, with this park's droop inert (it is clipped either way).
            outside_box.append((p["name"], qset, lo, hi))
            pinned_to_lf.append(p["name"])
            lo = hi = qset
        set_qv_params(p["pre"], qset_pu=qset, v_anchor_pu=p["v_lf"],
                      slope_pu=slope, deadband_pu=p["stored"]["db"],
                      q_min_pu=lo, q_max_pu=hi)
        p["anchored"] = {"qset": qset, "Vanchor": p["v_lf"], "Kdroop": 1.0 / slope,
                         "qmin": lo, "qmax": hi}
        kd.add(round(1.0 / slope, 9))
        if p["name"] in pinned_to_lf:
            pass
        elif hi - lo <= 1e-9:
            zero_box.append(p["name"])
        elif qset <= lo + 1e-9 or qset >= hi - 1e-9:
            pinned.append(p["name"])
    return {"kdroop_values": sorted(kd), "zero_width_box": zero_box,
            "pinned_at_limit": pinned, "q_lf_outside_box": outside_box,
            "box_pinned_to_lf_q": pinned_to_lf}


def check_init(app, parks, db: Optional[float] = None) -> Dict[str, Any]:
    """After ComInc: filter state against anchor, output against qset and --
    the equilibrium test -- against the park's load-flow Q.  A park whose
    load-flow Q lies outside its capability box is clipped by QVPRE, so the
    RMS initial state is then NOT the load-flow operating point."""
    dx, dq, dqlf, inside = [], [], [], 0
    worst_lf = ("", 0.0)
    for p in parks:
        pre = p["pre"]
        vals = read_params(pre)
        x1 = float(pre.GetAttribute("s:x1"))
        qext = float(pre.GetAttribute("s:Qext"))
        dx.append(abs(x1 - vals[1]))
        dq.append(abs(qext - min(max(vals[0], vals[5]), vals[6])))
        e = abs(qext - p["q_lf"] / p["sn"])
        dqlf.append(e)
        if e > worst_lf[1]:
            worst_lf = (p["name"], e)
        if db is not None and abs(x1 - vals[1]) <= db:
            inside += 1
    content = app.GetOutputWindow().GetContent()
    lines = [ln for x in (content if isinstance(content, (list, tuple))
                          else [content]) for ln in str(x).splitlines()]
    flagged = [l.strip() for l in lines
               if re.search(r"not consistent|inconsisten|derivative|error", l, re.I)]
    return {"max_abs_x1_minus_vanchor": max(dx), "max_abs_qext_minus_qset": max(dq),
            "max_abs_qext_minus_qlf_pu": max(dqlf), "worst_qext_vs_qlf": worst_lf,
            "n_qext_off_qlf_gt_1e-4": sum(e > 1e-4 for e in dqlf),
            "inside_db": inside if db is not None else None,
            "init_warnings": flagged[:20], "n_init_warnings": len(flagged)}


# ── operating point for the residue analysis ───────────────────────────────

def _key(obj) -> List[str]:
    """``[parent loc_name, "<loc_name>.<Class>"]`` -- the last two components
    of a ``VariableToIdx_*.txt`` path."""
    par = obj.GetParent()
    return [par.loc_name if par is not None else "",
            f"{obj.loc_name}.{obj.GetClassName()}"]


def _get(obj, attr: str) -> Optional[float]:
    try:
        v = obj.GetAttribute(attr)
        return None if v is None else float(v)
    except Exception:
        return None


def record_op(app, parks) -> Dict[str, Any]:
    """Load-flow values the exported matrices do not carry: terminal ``ur/ui``
    (output rows and shunt input columns), branch flows (validation of the
    branch-Q output rows), shunt and park data.  Needs a converged load flow
    of the linearisation point (run it before calling)."""
    op: Dict[str, Any] = {"terminals": [], "tr3": [], "lines": [], "shunts": [],
                          "parks": []}
    for t in get_all(app, "ElmTerm"):
        op["terminals"].append({"key": _key(t), "name": t.loc_name,
                                "ur": _get(t, "m:ur"), "ui": _get(t, "m:ui"),
                                "u": _get(t, "m:u"), "uknom": _get(t, "uknom")})

    def _term(elm, attr):
        cub = elm.GetAttribute(attr)
        return None if cub is None or cub.cterm is None else cub.cterm.loc_name

    for tr in get_all(app, "ElmTr3"):
        op["tr3"].append({"name": tr.loc_name, "hv": _term(tr, "bushv"),
                          "mv": _term(tr, "busmv"), "lv": _term(tr, "buslv"),
                          **{f"{q}_{s}": _get(tr, f"m:{q}:bus{s}")
                             for q in ("P", "Q") for s in ("hv", "mv", "lv")},
                          "tap_h": _get(tr, "n3tap_h")})
    for ln in get_all(app, "ElmLne"):
        op["lines"].append({"name": ln.loc_name, "bus1": _term(ln, "bus1"),
                            "bus2": _term(ln, "bus2"),
                            **{f"{q}_{s}": _get(ln, f"m:{q}:bus{s}")
                               for q in ("P", "Q") for s in ("1", "2")}})
    for sh in get_all(app, "ElmShnt"):
        op["shunts"].append({"name": sh.loc_name, "key": _key(sh),
                             "bus": _term(sh, "bus1"),
                             **{a: _get(sh, a) for a in ("shtype", "ncapa", "ncapx",
                                                         "qcapn", "qrean", "ushnm",
                                                         "outserv")},
                             "Q_bus1": _get(sh, "m:Q:bus1")})
    for p in parks:
        op["parks"].append({"name": p["name"], "idx": p["idx"], "sn": p["sn"],
                            "q_lf": p["q_lf"], "p_lf": p["p_lf"], "v_lf": p["v_lf"],
                            "pre_key": _key(p["pre"]), **p.get("anchored", {})})
    return op


# ── acceptance check and report ────────────────────────────────────────────

def acceptance(run_dirs: Dict[str, Path]) -> Dict[str, Any]:
    import pandas as pd
    out: Dict[str, Any] = {}
    frames = {c: pd.read_csv(d / "modal_modes.csv") for c, d in run_dirs.items()}
    for c, df in frames.items():
        q = df[df["dominant_object"].astype(str).str.endswith("\\" + QVPRE_ELEMENT_NAME)]
        out[c] = {"n_conv": int((df["group"] == "conv").sum()),
                  "n_qvpre_dominated": len(q),
                  "qvpre_sigma_min": float(q["re"].min()) if len(q) else float("nan"),
                  "qvpre_sigma_max": float(q["re"].max()) if len(q) else float("nan"),
                  "qvpre_sigma_slower_than_50": int((q["re"] > -50.0 * (1 - 1e-6)).sum())}
    if {"db_wide", "db_zero"} <= set(frames):
        a = frames["db_wide"]; b = frames["db_zero"]
        key = lambda z: (z.real, z.imag)  # noqa: E731
        za = sorted((complex(r, i) for r, i in a.loc[a["group"] == "conv", ["re", "im"]].values), key=key)
        zb = sorted((complex(r, i) for r, i in b.loc[b["group"] == "conv", ["re", "im"]].values), key=key)
        same = len(za) == len(zb) and all(abs(x - y) <= 1e-6 * max(abs(x), 1.0)
                                          for x, y in zip(za, zb))
        out["conv_spectra_identical"] = same
        out["passed"] = not same
    return out


def parse_sync_log(path: Optional[Path]) -> Optional[Dict[str, int]]:
    if path is None or not Path(path).exists():
        return None
    counts = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^\s+(created|renamed|updated|deleted|unchanged): (\d+)", line)
        if m:
            counts[m.group(1)] = int(m.group(2))
    return counts


def _fmt(x: Any) -> str:
    return f"{x:.4g}" if isinstance(x, float) else str(x)


def write_report(folder: Path, state: Dict[str, Any]) -> None:
    L = [f"# Modal window {state['window']} ({state['stamp']})", "",
         f"- snapshot: `{state['snapshot']}` (sha256 `{state['sha256']}`)",
         f"- sync counts: {state.get('sync')}; dry-run counts: {state.get('dryrun')}",
         "", "## Diagnostic at the synced operating point, before any change", ""]
    L += [f"- {k}: {_fmt(v)}" for k, v in state.get("diag_before", {}).items()]
    L += ["", "## Re-anchoring", ""]
    L += [f"- {k}: {_fmt(v)}" for k, v in state.get("reanchor", {}).items()]
    L += [f"- after anchoring, ComInc: {state.get('init_anchored')}", "",
          f"## Tap controllers: {state.get('taps')}", "", "## Droop cases", ""]
    for c, info in state.get("cases", {}).items():
        L.append(f"- `{c}` (db = {info['db']}): run `{info['run_dir']}`; "
                 f"ComInc check {info['init']}")
    L += ["", "## Acceptance check (converter group must differ)", "",
          f"{json.dumps(state.get('acceptance'), indent=1)}", "",
          "## Restore", "", f"{json.dumps(state.get('restore'), indent=1, default=str)}"]
    if state.get("error"):
        L += ["", f"## ERROR", "", f"`{state['error']}`"]
    (folder / "window_report.md").write_text("\n".join(L), encoding="utf-8")


# ── driver ─────────────────────────────────────────────────────────────────

def run_window(window: str, snapshot: Path, *, db_wide: float = DB_WIDE_PU,
               sync_log: Optional[Path] = None, dryrun_log: Optional[Path] = None,
               disable_taps: bool = True, residue_runs: bool = True) -> int:
    import hashlib
    import pandas as pd
    from analysis import modal_reduce

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = CAMPAIGN / window / stamp
    folder.mkdir(parents=True, exist_ok=True)
    doc = load_snapshot(snapshot)
    state: Dict[str, Any] = {
        "window": window, "stamp": stamp, "snapshot": str(snapshot),
        "sha256": hashlib.sha256(Path(snapshot).read_bytes()).hexdigest(),
        "sync": parse_sync_log(sync_log), "dryrun": parse_sync_log(dryrun_log),
    }
    app = connect(study_case=RMS_STUDY_CASE)
    ctx = ScreeningContext(app)
    mod = app.GetFromStudyCase("ComMod")
    commod_keys = ["dirMatl"] + [k for k, _ in COMMOD_PARTICIPATION_OPTS]
    commod_before = {k: mod.GetAttribute(k) for k in commod_keys}
    parks = resolve_parks(app, doc)
    taps = resolve_taps(app)
    orig = record_originals(parks, taps)
    (folder / "originals.json").write_text(json.dumps(orig, indent=1), encoding="utf-8")
    print(f"[window {window}] {len(parks)} QVPRE, {len(taps)} TAPCTRL; originals -> "
          f"{folder / 'originals.json'}")
    run_dirs: Dict[str, Path] = {}
    rc = 0
    try:
        run_ldf(app)
        harvest(parks)
        rows, agg = diag_before(parks)
        pd.DataFrame(rows).to_csv(folder / "qvpre_diag_before.csv", index=False)
        state["diag_before"] = agg
        print(f"[window {window}] before any change: {agg}")

        state["reanchor"] = reanchor(parks)
        app.ClearOutputWindow()
        ctx.initialise()
        state["init_anchored"] = check_init(app, parks)
        pd.DataFrame([{"park": p["name"], **p["anchored"]} for p in parks]).to_csv(
            folder / "qvpre_anchored.csv", index=False)
        print(f"[window {window}] re-anchored: {state['reanchor']}; "
              f"ComInc {state['init_anchored']}")
        # Operating point of the linearisation (the load flow does not depend
        # on the DSL parameters just written).
        run_ldf(app)
        (folder / "op.json").write_text(json.dumps(record_op(app, parks), indent=1),
                                        encoding="utf-8")

        if disable_taps:
            app.ResetCalculation()
            for t in taps:
                t.SetAttribute("outserv", 1)
            state["taps"] = f"{len(taps)} TAPCTRL set out of service for the modal runs"
        else:
            state["taps"] = "left in service (classified 'other')"

        state["cases"] = {}
        for case, db in (("db_wide", db_wide), ("db_zero", 0.0)):
            app.ResetCalculation()
            for p in parks:
                p["pre"].SetAttribute("params:3", float(db))
            app.ClearOutputWindow()
            ctx.initialise()
            init = check_init(app, parks, db=db)
            out_dir = (REPO / "results" / "screening" / f"{window}_{case}"
                       / datetime.now().strftime("%Y%m%d-%H%M%S"))
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd_modal(ctx, out_dir, participation=True, window=window, droop_case=case)
            run_dirs[case] = out_dir
            state["cases"][case] = {"db": db, "run_dir": str(out_dir), "init": init}

        # Residue analysis support: the same two cases with the tap
        # controllers IN service, so the Jacobian carries ``nntapin`` and the
        # tap input columns can be read from it.  Not a modal result.
        if disable_taps and residue_runs:
            state["cases_tapsin"] = {}
            app.ResetCalculation()
            for t in taps:
                t.SetAttribute("outserv", orig["tapctrl_outserv"][t.GetFullName()])
            for case, db in (("db_wide", db_wide), ("db_zero", 0.0)):
                app.ResetCalculation()
                for p in parks:
                    p["pre"].SetAttribute("params:3", float(db))
                ctx.initialise()
                out_dir = (REPO / "results" / "screening" / f"{window}_{case}_tapsin"
                           / datetime.now().strftime("%Y%m%d-%H%M%S"))
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd_modal(ctx, out_dir, participation=True, window=window,
                          droop_case=f"{case}_tapsin")
                state["cases_tapsin"][case] = {"db": db, "run_dir": str(out_dir)}
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        rc = 2
        print(f"[window {window}] ERROR: {state['error']}")
    finally:
        state["restore"] = restore(app, orig)
        print(f"[window {window}] restore: ok={state['restore']['ok']}, "
              f"max dev {state['restore']['max_abs_dev']:.2e}, "
              f"db before {state['restore']['db_before']} after {state['restore']['db_after']}")
        commod_after = {k: mod.GetAttribute(k) for k in commod_keys}
        state["restore"]["commod_options_unchanged"] = commod_after == commod_before
        print(f"[window {window}] ComMod options unchanged: {commod_after == commod_before}")
        if not state["restore"]["ok"] or commod_after != commod_before:
            rc = 3

    if run_dirs:
        state["acceptance"] = acceptance(run_dirs)
        print(f"[window {window}] acceptance: {state['acceptance']}")
        modal_reduce.reduce([d / "modal_modes.csv" for d in run_dirs.values()],
                            folder / "reduction", label=window)
    (folder / "window_state.json").write_text(json.dumps(state, indent=1, default=str),
                                              encoding="utf-8")
    write_report(folder, state)
    print(f"[window {window}] report -> {folder / 'window_report.md'}")
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="One window of the modal analysis.")
    ap.add_argument("--window", help="window label, e.g. W1")
    ap.add_argument("--snapshot", type=Path)
    ap.add_argument("--db-wide", type=float, default=DB_WIDE_PU)
    ap.add_argument("--sync-log", type=Path)
    ap.add_argument("--dryrun-log", type=Path)
    ap.add_argument("--keep-taps", action="store_true",
                    help="do not take TAPCTRL out of service")
    ap.add_argument("--no-residue-runs", action="store_true",
                    help="skip the two taps-in-service runs and op.json use "
                         "by analysis/modal_observability.py")
    ap.add_argument("--restore", type=Path, metavar="ORIGINALS_JSON",
                    help="only restore QVPRE params and TAPCTRL flags from a file")
    args = ap.parse_args(argv)
    if args.restore:
        app = connect(study_case=RMS_STUDY_CASE)
        res = restore(app, json.loads(args.restore.read_text(encoding="utf-8")))
        print(json.dumps(res, indent=1, default=str))
        return 0 if res["ok"] else 3
    if not args.window or not args.snapshot:
        ap.error("--window and --snapshot are required (or --restore)")
    return run_window(args.window, args.snapshot, db_wide=args.db_wide,
                      sync_log=args.sync_log, dryrun_log=args.dryrun_log,
                      disable_taps=not args.keep_taps,
                      residue_runs=not args.no_residue_runs)


if __name__ == "__main__":
    sys.exit(main())
