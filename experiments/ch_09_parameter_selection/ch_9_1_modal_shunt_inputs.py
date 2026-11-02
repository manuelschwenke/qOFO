#!/usr/bin/env python3
r"""Experiment A, shunt-bank inputs: settling of the measured outputs after one
MSC/MSR step, from the linearisations of the modal campaign of 2026-09-23.

Fills the shunt row of ``tab:param:timescales:actuators``
(``ch:timescales:plant``).  The other rows come from the same linearisations
(``analysis.modal_observability.run``, bundle
``graphics/Ch8/data/2026-09-23_modal/`` of the dissertation); this run adds the
eight bank inputs (MSC and MSR of each sub-transmission system) on the
taps-out system of each window and droop case, with the same outputs (70
measured voltages, 12 interface Q) and bands (1e-3 pu, 1 Mvar).

The PowerFactory model of that campaign predates the placement of 2026-10-15
(both banks on the first tertiary).  Every bank is at step 0 in every window,
so the operating point does not depend on the placement, and the step is
applied at the thesis terminal (MSR: tertiary of the first coupler, MSC: of
the second).  ``analysis.modal_observability.shunt_inputs`` checks the PF
admittance scale per terminal and the bank data PF vs thesis, and fails on a
mismatch.

Usage::

    python -m experiments.ch_09_parameter_selection.ch_9_1_modal_shunt_inputs [--workers 3]

Author: Manuel Schwenke / Claude Code (2026-11-02)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_runs import add_run_arguments, start_thesis_run  # noqa: E402

#: Window folders of the modal campaign (the ones the 2026-09-23 bundle used).
CAMPAIGN = Path(r"\\e5server\homefolders$\mschwenke\Python_Projekte\qOFO_2027"
                r"\results\screening\modal_campaign")
WINDOWS = {"W1": "20260923-144506", "W2": "20260923-155125", "W3": "20260923-160109",
           "W4": "20260923-161128", "W5": "20260923-162032", "W6": "20260923-180102"}
T_D_SHUNT = 1.0          # command-to-action delay, as for every non-tap actuator
T_STS = 20.0


def _one(window: str, out: str) -> str:
    from analysis import modal_observability as mo
    mo.run_shunts(CAMPAIGN / window / WINDOWS[window], Path(out), ode_check=True)
    return window


def _mode_of(run_dir: Path, re_: float, im_: float) -> dict:
    """The participation-table row of the eigenvalue nearest to (re, im)."""
    modes = pd.read_csv(Path(run_dir) / "modal_modes.csv")
    z = modes["re"].to_numpy() + 1j * modes["im"].to_numpy()
    r = modes.iloc[int(np.argmin(np.abs(z - complex(re_, abs(im_)))))]
    return {"f_hz": float(r["f_hz"]), "zeta": float(r["zeta"]), "group": r["group"],
            "dominant_state": r["dominant_state"],
            "dominant_object": str(r["dominant_object"]).split("\\")[-1]}


def aggregate(root: Path) -> pd.DataFrame:
    rows = []
    for w, stamp in WINDOWS.items():
        state = json.loads((CAMPAIGN / w / stamp / "window_state.json").read_text(encoding="utf-8"))
        df = pd.read_csv(root / w / "pairs.csv")
        for (case, cls), x in df.groupby(["droop_case", "input_class"]):
            r = x.loc[x["Ts_abs_s"].idxmax()]
            m = _mode_of(state["cases"][case]["run_dir"], r["dom20_re"], r["dom20_im"])
            rows.append({"window": w, "droop_case": case, "input_class": cls,
                         "Ts_abs_s": r["Ts_abs_s"], "pair": f"{r['input']} -> {r['output']}",
                         "dy_inf": r["dy_inf"], "D_instant": r["D_instant"],
                         "dom20_re": r["dom20_re"], "dom20_im": r["dom20_im"],
                         "dom20_tau_s": r["dom20_tau_s"], **m})
    return pd.DataFrame(rows)


def checks(root: Path) -> pd.DataFrame:
    rows = []
    for w in WINDOWS:
        s = json.loads((root / w / "summary.json").read_text(encoding="utf-8"))
        for case, v in s.items():
            rows.append({"window": w, "droop_case": case, **{
                k: v["checks"][k] for k in v["checks"] if k != "measured_voltage_sets"},
                "unstable": v["info"]["unstable"],
                "max_abs_ramp_from_zero_modes": v["info"]["max_abs_ramp_from_zero_modes"]})
    return pd.DataFrame(rows)


def write_summary(root: Path, agg: pd.DataFrame, chk: pd.DataFrame) -> None:
    L = ["# Experiment A, shunt-bank inputs", "",
         f"Bands 1e-3 pu / 1 Mvar; one bank step (25 Mvar rated); T_d = {T_D_SHUNT:g} s; "
         f"threshold T_STS = {T_STS:g} s.", "",
         "## Per class, max over W1-W6 and both droop cases", "",
         "| input class | max T_s [s] | T_d + T_s [s] | at | binding pair | "
         "dominant mode @20 s |", "|:--|--:|--:|:--|:--|:--|"]
    for cls, x in agg.groupby("input_class"):
        r = x.loc[x["Ts_abs_s"].idxmax()]
        L.append(f"| {cls} | {r.Ts_abs_s:.2f} | {T_D_SHUNT + r.Ts_abs_s:.2f} | {r.window} "
                 f"{r.droop_case} | {r.pair} | {r.dom20_re:.4f}{r.dom20_im:+.4f}j "
                 f"(tau {r.dom20_tau_s:.2f} s, {r.f_hz:.3f} Hz, {r.dominant_state} @ "
                 f"{r.dominant_object}) |")
    L += ["", "## Per window and droop case", "",
          "| W | case | class | T_s [s] | pair | dominant mode @20 s |",
          "|:--|:--|:--|--:|:--|:--|"]
    for r in agg.sort_values(["input_class", "window", "droop_case"]).itertuples():
        L.append(f"| {r.window} | {r.droop_case} | {r.input_class} | {r.Ts_abs_s:.2f} | "
                 f"{r.pair} | tau {r.dom20_tau_s:.2f} s, {r.dominant_state} @ "
                 f"{r.dominant_object} |")
    try:
        table = chk.to_markdown(index=False)
    except ImportError:                         # tabulate not installed
        table = "```\n" + chk.to_string(index=False) + "\n```"
    L += ["", "## Checks", "", table, ""]
    (root / "summary.md").write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=3)
    add_run_arguments(ap)
    args = ap.parse_args(argv)
    # each worker's eigen-decomposition stays on a few BLAS threads
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "4")
    with start_thesis_run(
            __file__, target="ch:timescales:plant",
            feeds=["tab:param:timescales:actuators"],
            parameters={"windows": list(WINDOWS), "droop_cases": ["db_wide", "db_zero"],
                        "inputs": "MSC and MSR of DSO_1..DSO_4, one step of 25 Mvar",
                        "placement": "MSR tertiary of coupler 0, MSC of coupler 1 (thesis)",
                        "bands": "1e-3 pu, 1 Mvar", "T_d_shunt_s": T_D_SHUNT,
                        "linearisations": "PowerFactory modal campaign 2026-09-23 "
                                          "(qOFO_2027 commit 781a819)"},
            time_series="operating windows W1-W6 (snapshots export/snapshots/*_dso3x2.json)",
            note=args.note, tag=args.tag) as run:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_one, w, str(run.path(w))) for w in WINDOWS]
            for f in futs:
                print(f"[shunt] {f.result()} done", flush=True)
        agg = aggregate(run.root)
        agg.to_csv(run.path("residue_settling_shunt.csv"), index=False)
        chk = checks(run.root)
        chk.to_csv(run.path("checks.csv"), index=False)
        write_summary(run.root, agg, chk)
        run.describe_output("W1..W6/", "per window: pairs.csv, summary.md, shunt parity, "
                            "time-domain and load-flow checks (analysis.modal_observability.run_shunts)")
        run.describe_output("residue_settling_shunt.csv",
                            "per window, droop case and bank class: longest settling, binding "
                            "pair, dominant remaining mode")
        run.describe_output("checks.csv", "admittance scale, A vs Amat, ODE and load-flow checks")
        run.describe_output("summary.md", "the row of tab:param:timescales:actuators")
        print((run.root / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
