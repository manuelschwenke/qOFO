#!/usr/bin/env python3
"""Short headless sweep for the TSO-TSO tie-voltage coordinator.

The diagnostic focuses on L14 of the IEEE 39-bus case (bus 9 -> bus 39),
because this line often carries a large reactive exchange between zone 2
and zone 1.  It imports ``experiments/000_M_TSO_M_DSO.py`` and changes only
runtime configuration fields; controller code and experiment defaults are
left untouched.

Usage
-----
    python experiments/diag_tie_coord_sweep.py --horizon-min 12
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP000 = ROOT / "experiments" / "000_M_TSO_M_DSO.py"
L14 = 14


@dataclass(frozen=True)
class Case:
    name: str
    beta: float
    lambda_max: float
    alpha_lambda: float = 1e7
    g_z_q_tie: float = 0.0
    tso_g_q_tie: float = 0.0


def _load_exp000():
    spec = importlib.util.spec_from_file_location("exp000_diag", EXP000)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {EXP000}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def default_cases() -> list[Case]:
    cases: list[Case] = []
    for beta in (1.0, 0.1, 0.02, 0.005):
        for lambda_max in (2e5, 5e5, 1e6):
            cases.append(
                Case(
                    name=f"b{beta:g}_l{lambda_max:g}",
                    beta=beta,
                    lambda_max=lambda_max,
                )
            )

    # Direct tie-flow guardrail checks around the more promising low-beta
    # corridor settings.  These distinguish "voltage-only coordination works"
    # from "Q_tie needs to be an explicit controlled output/soft cap".
    for beta in (0.02, 0.005):
        cases.extend(
            [
                Case(
                    name=f"b{beta:g}_l1e6_gz1e2",
                    beta=beta,
                    lambda_max=1e6,
                    g_z_q_tie=1e2,
                ),
                Case(
                    name=f"b{beta:g}_l1e6_gq10",
                    beta=beta,
                    lambda_max=1e6,
                    tso_g_q_tie=10.0,
                ),
            ]
        )
    return cases


def run_case(mod, case: Case, horizon_min: float) -> dict[str, float | str]:
    cfg = mod.make_config()
    cfg.n_total_s = horizon_min * 60.0
    cfg.verbose = 0
    cfg.contingencies = []
    cfg.live_plot_controller = False
    cfg.live_plot_cascade = False
    cfg.live_plot_system = False
    cfg.live_plot_tracking = False
    cfg.live_plot_tie_coordination = False
    cfg.run_stability_analysis = False

    cfg.tie_ff_smoothing = case.beta
    cfg.tie_alpha_lambda = case.alpha_lambda
    cfg.tie_lambda_max = case.lambda_max
    cfg.g_z_q_tie = case.g_z_q_tie
    cfg.tso_g_q_tie = case.tso_g_q_tie

    # Suppress the large per-run banner; keep this script's CSV clean.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        log = mod.run_multi_tso_dso(cfg)

    rows = [r for r in log if L14 in getattr(r, "tie_q_mvar", {})]
    if not rows:
        raise RuntimeError("no L14 rows recorded")

    last = rows[-1]
    q14 = np.array([r.tie_q_mvar[L14] for r in rows], dtype=float)
    z2_rms = np.array(
        [r.zone_v_rms_err_pu.get(2, np.nan) for r in rows], dtype=float
    )
    all_tie_abs: list[float] = []
    for rec in rows:
        for q in rec.tie_q_mvar.values():
            qf = float(q)
            if math.isfinite(qf):
                all_tie_abs.append(abs(qf))

    v39 = float(last.tie_v_i.get(L14, np.nan))
    v9 = float(last.tie_v_j.get(L14, np.nan))
    return {
        "name": case.name,
        "beta": case.beta,
        "lambda_max": case.lambda_max,
        "alpha_lambda": case.alpha_lambda,
        "g_z_q_tie": case.g_z_q_tie,
        "tso_g_q_tie": case.tso_g_q_tie,
        "final_q14_z1_mvar": float(last.tie_q_mvar[L14]),
        "final_q14_z2_to_z1_mvar": float(-last.tie_q_mvar[L14]),
        "mean_abs_q14_mvar": float(np.mean(np.abs(q14))),
        "final_v39_pu": v39,
        "final_v9_pu": v9,
        "final_v39_minus_v9_pu": v39 - v9,
        "final_lambda": float(last.tie_lambda.get(L14, np.nan)),
        "final_z2_vmin_pu": float(last.zone_v_min.get(2, np.nan)),
        "final_z2_vmax_pu": float(last.zone_v_max.get(2, np.nan)),
        "final_z2_rms_pu": float(last.zone_v_rms_err_pu.get(2, np.nan)),
        "mean_z2_rms_pu": float(np.nanmean(z2_rms)),
        "mean_abs_all_tie_q_mvar": (
            float(np.mean(all_tie_abs)) if all_tie_abs else np.nan
        ),
    }


def _format_csv_value(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.8g}"


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-min", type=float, default=12.0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    warnings.filterwarnings("ignore")
    mod = _load_exp000()
    cases = default_cases()
    header = list(run_case(mod, cases[0], args.horizon_min).keys())
    print(",".join(header))
    print(",".join(_format_csv_value(v) for v in run_case(mod, cases[0], args.horizon_min).values()))
    for case in cases[1:]:
        try:
            row = run_case(mod, case, args.horizon_min)
            print(",".join(_format_csv_value(row[k]) for k in header), flush=True)
        except Exception as exc:
            fail = {k: "nan" for k in header}
            fail["name"] = case.name
            fail["beta"] = case.beta
            fail["lambda_max"] = case.lambda_max
            fail["alpha_lambda"] = case.alpha_lambda
            fail["g_z_q_tie"] = case.g_z_q_tie
            fail["tso_g_q_tie"] = case.tso_g_q_tie
            fail["final_q14_z1_mvar"] = f"FAILED:{type(exc).__name__}:{exc}"
            print(",".join(_format_csv_value(fail[k]) for k in header), flush=True)


if __name__ == "__main__":
    main()
