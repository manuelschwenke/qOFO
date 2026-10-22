"""
experiments/case1_svr/run_case1.py
==================================

Case 1 of the thesis (``ch:case_svr``, Chapter 9 since 2026-10-06): OFO in
place of pilot-node secondary voltage regulation, on transmission resources
only.  Design record: dissertation repo ``00_daily_log/2026-10-06_part3_case_design.md``.

Variants (thesis ``tab:case_svr:variants``)
-------------------------------------------
``L2``   local q(v) only -- the floor.
``S1m``  "S1^-": classical RVR + RPR with the INTEGRAL laws only (K_p = 0) and
         no voltage-priority safeguard.  The exact equivalence partner of O0.
``S1``   classical RVR + RPR as deployed in this thesis: integral laws (K_p = 0)
         with the voltage-priority safeguard on.  Every performance comparison
         uses it.  (Until 2026-10-07 S1 was the PI form, K_p = 0.2 on both
         loops; author decision of 2026-10-07: the thesis uses the integral form
         only.  The PI runs stay resolvable as ``S1PI*`` for the one sentence of
         evidence that the proportional path adds little.)
``O0``   the S1m hierarchy written as one-input OFO steps
         (``svr_rvr_law = svr_rpr_law = 'ofo'``): weights from the match
         condition, the TS-OFO's own voltage limits as soft constraints.
``O1``   one TS-OFO per area acting directly on AVR set-points and TS-DER Q,
         machine taps held, no shunts.
``O2``   O1 with machine taps and TSO shunts in the loop.

Campaigns
---------
``K1-REF``  the reference run: 360 min, the six ``make_config`` contingencies
            (gen trip 30, load connect 120, gen restore 180, line trip 210,
            line restore 300, load trip 330 min).  Starts 2016-01-05 08:00,
            i.e. at design window W1 -- an equivalence check and an
            illustration, NOT validation.
``K1-PAR``  error--effort front.  O1/O2 with the AVR movement weight scaled by
            0.1 and 0.01; S1 with T_RVR = 112.5 s (b = T_TS/T_RVR = 1.6, the
            OFO's own budget ceiling), 137.4 s (b = 1.31, matched to the
            selected OFO weights) and 600 s.  300 s (b = 0.6) is the deployed
            S1 in K1-REF.

Configuration
-------------
``make_config()`` + DSO_3 x2 (the Chapter-8 batch network) + paramset
``ch9_selected`` (the Chapter-8 weights, lambda_max(B) = 1.31) + horizon.
Nothing else is restated.  ``assert_case1_controlled`` fails loudly if two
consecutive variants differ in more than the mechanism the step isolates.

Run
---
    python -m experiments.case1_svr.run_case1 --list
    python -m experiments.case1_svr.run_case1 --only S1m,O0 --run-dir <dir>
    python -m experiments.case1_svr.run_case1 --only O0 --minutes 30   # smoke

Author: Manuel Schwenke / Claude Code (2026-10-06)
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from configs.config import MultiTSOConfig
from configs.paramsets import apply_paramset
from experiments.paths import RESULTS_ROOT
from experiments.run_multi_system_ofo import make_config
from experiments.runners.multi_tso_dso import run_multi_tso_dso

EXPERIMENT = "THESIS_case1_svr"
PARAMSET = "ch9_selected"
HORIZON_MIN = 360

#: Pilot buses of the classical scheme, selected once on the base case
#: (``008_PILOT_BUS_SELECT.py``, archived to ``_local_archive/`` on
#: 2026-10-22) and held fixed; the
#: same as in ``experiments/ch_10_case_study/ch_10_1_variant_ladder.py`` and
#: thesis ``tab:svr:pilots``.
PILOT_BUSES: Dict[int, int] = {1: 25, 2: 7, 3: 20}

#: Switched shunts off below O2: a bank is only dispatched where the TSO MIQP
#: runs, so leaving them installed would switch a discrete device class on in
#: O1 but not in S1 (see the ladder script for the full record).
_NO_SHUNTS = dict(install_tso_tertiary_shunts=False, shunt_dispatch="off")

_LOCAL_STS = dict(dso_mode="local", local_der_mode="qv",
                  tso_q_mode="qv", dso_q_mode="qv")

_SVR = dict(
    tso_mode="svr", tso_oltc_mode="local",
    svr_pilot_buses=PILOT_BUSES, svr_t_rvr_s=300.0, svr_t_rpr_s=60.0,
    # No supervisory dead band: the OFO's tracking has none either.
    svr_deadband_pu=0.0,
    **_LOCAL_STS, **_NO_SHUNTS,
)

VARIANTS: Dict[str, Dict[str, Any]] = {
    "L2": dict(tso_mode="local", tso_local_mode="qv", tso_oltc_mode="local",
               **_LOCAL_STS, **_NO_SHUNTS),
    "S1m": dict(_SVR, svr_k_p_rvr=0.0, svr_k_p_rpr=0.0,
                svr_rpr_voltage_priority=False),
    # Integral laws WITH the safeguard (author 2026-10-07; was "S1g" in the
    # morning of that day).  S1m -> S1 isolates the safeguard.
    "S1": dict(_SVR, svr_k_p_rvr=0.0, svr_k_p_rpr=0.0,
               svr_rpr_voltage_priority=True),
    "O0": dict(_SVR, svr_k_p_rvr=0.0, svr_k_p_rpr=0.0,
               svr_rpr_voltage_priority=False,
               svr_rvr_law="ofo", svr_rpr_law="ofo"),
    "O1": dict(tso_mode="ofo", tso_oltc_mode="local",
               **_LOCAL_STS, **_NO_SHUNTS),
    "O2": dict(tso_mode="ofo", **_LOCAL_STS),
}

#: K1-PAR: (base variant, field overrides, AVR-weight scale).
PARETO: Dict[str, Tuple[str, Dict[str, Any], float]] = {
    "O1_gw0p1": ("O1", {}, 0.1),
    "O1_gw0p01": ("O1", {}, 0.01),
    "O2_gw0p1": ("O2", {}, 0.1),
    "O2_gw0p01": ("O2", {}, 0.01),
    # Integral S1: b = T_TS/T_RVR = 180/T_RVR.
    "S1_T112": ("S1", {"svr_t_rvr_s": 112.5}, 1.0),   # b = 1.6, the ceiling
    "S1_T137": ("S1", {"svr_t_rvr_s": 137.4}, 1.0),   # b = 1.31 = lambda_max(B)
    "S1_T600": ("S1", {"svr_t_rvr_s": 600.0}, 1.0),   # b = 0.3
}

#: LEGACY, evidence only (not part of the thesis variants since 2026-10-07):
#: the PI form of S1 (K_p = 0.2 on both loops, safeguard on) and its front.
#: Their runs of 2026-10-06 are kept in 20261006_K1 under these names.
LEGACY: Dict[str, Tuple[str, Dict[str, Any], float]] = {
    "S1PI": ("S1", {"svr_k_p_rvr": 0.2, "svr_k_p_rpr": 0.2}, 1.0),
    "S1PI_T150": ("S1", {"svr_k_p_rvr": 0.2, "svr_k_p_rpr": 0.2,
                         "svr_t_rvr_s": 150.0}, 1.0),
    "S1PI_T198": ("S1", {"svr_k_p_rvr": 0.2, "svr_k_p_rpr": 0.2,
                         "svr_t_rvr_s": 198.0}, 1.0),
    "S1PI_T600": ("S1", {"svr_k_p_rvr": 0.2, "svr_k_p_rpr": 0.2,
                         "svr_t_rvr_s": 600.0}, 1.0),
}

K1_REF = ["L2", "S1m", "S1", "O0", "O1", "O2"]
K1_PAR = list(PARETO)
K1_EXTRA = list(LEGACY)

#: What each controlled step may change.  Enforced on RESOLVED configs.
_SVR_FIELDS = {"tso_mode", "tso_local_mode", "svr_pilot_buses", "svr_t_rvr_s",
               "svr_t_rpr_s", "svr_k_p_rvr", "svr_k_p_rpr",
               "svr_rvr_p_every_step", "svr_deadband_pu",
               "svr_rpr_voltage_priority", "svr_rvr_law", "svr_rpr_law"}
STEP_ALLOWED: Dict[Tuple[str, str], set] = {
    ("L2", "S1m"): _SVR_FIELDS,
    ("S1m", "O0"): {"svr_rvr_law", "svr_rpr_law"},
    ("S1m", "S1"): {"svr_rpr_voltage_priority"},
    ("O0", "O1"): _SVR_FIELDS,
    ("O1", "O2"): {"tso_oltc_mode", "install_tso_tertiary_shunts",
                   "shunt_dispatch"},
}


# ---------------------------------------------------------------------------
#  Configuration -- derived, never restated
# ---------------------------------------------------------------------------


def base_config(minutes: int = HORIZON_MIN) -> MultiTSOConfig:
    """make_config + Chapter-8 batch network + Chapter-8 weights + horizon."""
    cfg = dataclasses.replace(
        make_config(),
        n_total_s=60.0 * minutes,
        verbose=1,
        dso_der_scale={"DSO_3": 2.0},
        dso_load_p_scale={"DSO_3": 2.0},
        # Batch runs: no live figures.  Pure visualisation, no effect on the
        # control path, identical for every variant.
        live_plot_controller=False,
        live_plot_cascade=False,
        live_plot_system=False,
        live_plot_tracking=False,
        live_plot_sbx=False,
        live_plot_sbxv=False,
    )
    return apply_paramset(cfg, PARAMSET, verbose=False)


def resolved(name: str, minutes: int = HORIZON_MIN) -> MultiTSOConfig:
    """The config the runner receives for a variant or a Pareto point."""
    cfg = base_config(minutes)
    if name in VARIANTS:
        return dataclasses.replace(cfg, **VARIANTS[name])
    if name in PARETO or name in LEGACY:
        base, over, gw_scale = PARETO.get(name) or LEGACY[name]
        cfg = dataclasses.replace(cfg, **dict(VARIANTS[base], **over))
        if gw_scale != 1.0:
            cfg = dataclasses.replace(cfg, g_w_gen=float(cfg.g_w_gen) * gw_scale)
        return cfg
    raise SystemExit(f"unknown variant {name!r}; valid: {K1_REF + K1_PAR + K1_EXTRA}")


def assert_case1_controlled() -> None:
    """Fail loudly if a step changes more than its one mechanism."""
    problems: List[str] = []
    for (a, b), allowed in STEP_ALLOWED.items():
        ca, cb = resolved(a), resolved(b)
        for f in dataclasses.fields(MultiTSOConfig):
            k = f.name
            if k in allowed or k in ("result_dir", "verbose"):
                continue
            va, vb = getattr(ca, k, None), getattr(cb, k, None)
            try:
                same = bool(va == vb)
            except Exception:
                same = repr(va) == repr(vb)
            if not same:
                problems.append(f"  {a}->{b}: {k}  {a}={va!r}  {b}={vb!r}")
    if problems:
        raise SystemExit("Case 1 is NOT controlled:\n" + "\n".join(problems))
    print("guard: every Case-1 step changes only what it is allowed to")


# ---------------------------------------------------------------------------
#  Run
# ---------------------------------------------------------------------------


def _provenance(out: Path) -> None:
    """Record the code state the campaign ran on (once per run directory)."""
    p = out / "provenance.json"
    if p.exists():
        return
    root = Path(__file__).resolve().parents[2]

    def _git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True,
                                  text=True, timeout=60).stdout.strip()
        except Exception as exc:  # provenance must never kill a run
            return f"unavailable: {exc}"

    p.write_text(json.dumps({
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "code_root": str(root),
        "git_head": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--porcelain"),
        "paramset": PARAMSET,
        "horizon_min_default": HORIZON_MIN,
    }, indent=2), encoding="utf-8")
    (out / "code_diff.patch").write_text(_git("diff"), encoding="utf-8")


def run_one(name: str, out: Path, minutes: int) -> int:
    cfg = resolved(name, minutes)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    cfg = dataclasses.replace(cfg, result_dir=str(d))
    (d / "config_repr.txt").write_text(repr(cfg), encoding="utf-8")
    print("\n" + "=" * 72)
    print(f"  {name}  tso={cfg.tso_mode}  oltc={cfg.tso_oltc_mode}  "
          f"shunts={cfg.shunt_dispatch}  g_w_gen={cfg.g_w_gen:.4g}  "
          f"svr: law={cfg.svr_rvr_law}/{cfg.svr_rpr_law} "
          f"K_p={cfg.svr_k_p_rvr}/{cfg.svr_k_p_rpr} "
          f"guard={cfg.svr_rpr_voltage_priority} T_RVR={cfg.svr_t_rvr_s}")
    print("=" * 72, flush=True)
    t0 = _dt.datetime.now()
    try:
        log = run_multi_tso_dso(cfg)
        status = "ok"
    except Exception as exc:  # record the failure, keep the batch alive
        print(f"  [{name}] FAILED: {type(exc).__name__}: {exc}", flush=True)
        log, status = [], f"failed: {type(exc).__name__}: {exc}"
    with open(d / "log.pkl", "wb") as f:
        pickle.dump(log, f)
    (d / "status.json").write_text(json.dumps({
        "variant": name, "status": status, "n_records": len(log),
        "minutes": minutes,
        "wall_s": (_dt.datetime.now() - t0).total_seconds(),
    }, indent=2), encoding="utf-8")
    print(f"  [{name}] {status}, {len(log)} records", flush=True)
    return len(log)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated names, or K1-REF / K1-PAR")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--minutes", type=int, default=HORIZON_MIN)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("K1-REF:", ", ".join(K1_REF))
        print("K1-PAR:", ", ".join(K1_PAR))
        return

    assert_case1_controlled()
    if args.only in (None, "K1-REF"):
        sel = list(K1_REF)
    elif args.only == "K1-PAR":
        sel = list(K1_PAR)
    else:
        sel = [s.strip() for s in args.only.split(",") if s.strip()]
    for s in sel:
        resolved(s)                      # validates the name early

    if args.run_dir:
        out = Path(args.run_dir)
    else:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = Path(RESULTS_ROOT) / EXPERIMENT / stamp
    out.mkdir(parents=True, exist_ok=True)
    _provenance(out)
    print(f"run dir: {out}  |  horizon {args.minutes} min  |  {sel}")
    for s in sel:
        run_one(s, out, args.minutes)


if __name__ == "__main__":
    main()
