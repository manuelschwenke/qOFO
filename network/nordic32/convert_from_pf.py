#!/usr/bin/env python3
"""
network/nordic32/convert_from_pf.py
====================================
**Phase-A probe** — convert the Nordic SM PowerFactory project to
pandapower and dump diagnostics.

Run this ONCE on a machine that has DIgSILENT PowerFactory 2024+
installed and the Nordic_SM project already imported (File -> Import
Project from File -> Nordic_SM.pfd).  It then:

1. Connects to PF via :class:`core.pf_adapter.PFSession`.
2. Calls :func:`pandapower.converter.powerfactory.from_pfd`.
3. Runs :func:`pp.runpp` on the converted net.
4. Runs :func:`pandapower.converter.powerfactory.validate_pf_conversion`
   to compare PF vs. pandapower load-flow results bus-by-bus.
5. Writes two artefacts next to this file:

   * ``nordic_sm_converted.json`` — the converted pandapower net
     (``pp.to_json``), the canonical hand-off to the downstream
     build step.
   * ``nordic_sm_probe_report.json`` — element counts, parameter
     completeness, convergence stats, validation deltas.  Small
     enough to paste into a chat / commit.

Usage
-----
From a shell with the ``qOFO_clean`` env active and PF installed::

    cd <repo-root>
    python -m network.nordic32.convert_from_pf \\
        --project Nordic_SM \\
        --study-case "Base Case"

Options
-------
    --project        PF project name (default: Nordic_SM).
    --study-case     Study-case name to activate before conversion
                     (default: whatever is currently active in PF).
    --pv-as-slack    Pass True to the converter to import PV nodes as
                     slacks (default: False; usual choice for OFO).
    --no-validate    Skip the validate_pf_conversion step (if PF
                     results are stale or missing).

Exit codes
----------
    0   converter + LDF + validation all succeeded
    1   conversion raised
    2   converter ran but ``pp.runpp`` did not converge
    3   validation failed (diffs above threshold)

Author: Manuel Schwenke
Date: 2026-04-24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Project-root on sys.path (so ``core.pf_adapter`` imports work when
# this script is launched via ``python -m network.nordic32.convert_from_pf``).
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from core.pf_adapter import PFSession, LoadFlowDidNotConverge  # noqa: E402

_ARTIFACTS_DIR = _HERE.parent
_NET_OUT = _ARTIFACTS_DIR / "nordic_sm_converted.json"
_REPORT_OUT = _ARTIFACTS_DIR / "nordic_sm_probe_report.json"


# ---------------------------------------------------------------------------
#  Diagnostics collection
# ---------------------------------------------------------------------------


def _element_counts(net) -> Dict[str, int]:
    return {
        "bus": len(net.bus),
        "line": len(net.line),
        "trafo": len(net.trafo),
        "trafo3w": len(net.trafo3w),
        "gen": len(net.gen),
        "sgen": len(net.sgen),
        "load": len(net.load),
        "ext_grid": len(net.ext_grid),
        "shunt": len(net.shunt),
        "switch": len(net.switch) if hasattr(net, "switch") else 0,
    }


def _parameter_completeness(net) -> Dict[str, Dict[str, Any]]:
    """Count missing / default-valued parameters that we'll need to patch."""
    import numpy as np

    out: Dict[str, Dict[str, Any]] = {}

    # Generator Q-limits
    if len(net.gen) > 0:
        q_min = net.gen.get("min_q_mvar")
        q_max = net.gen.get("max_q_mvar")
        out["gen_q_limits"] = {
            "total": len(net.gen),
            "finite_q_min": int(np.isfinite(q_min).sum()) if q_min is not None else 0,
            "finite_q_max": int(np.isfinite(q_max).sum()) if q_max is not None else 0,
        }
        p_min = net.gen.get("min_p_mw")
        p_max = net.gen.get("max_p_mw")
        out["gen_p_limits"] = {
            "total": len(net.gen),
            "finite_p_min": int(np.isfinite(p_min).sum()) if p_min is not None else 0,
            "finite_p_max": int(np.isfinite(p_max).sum()) if p_max is not None else 0,
        }

    # Sgen Q-limits
    if len(net.sgen) > 0:
        q_min = net.sgen.get("min_q_mvar")
        q_max = net.sgen.get("max_q_mvar")
        out["sgen_q_limits"] = {
            "total": len(net.sgen),
            "finite_q_min": int(np.isfinite(q_min).sum()) if q_min is not None else 0,
            "finite_q_max": int(np.isfinite(q_max).sum()) if q_max is not None else 0,
        }

    # Trafo tap parameters
    if len(net.trafo) > 0:
        tap_step = net.trafo.get("tap_step_percent")
        tap_pos = net.trafo.get("tap_pos")
        out["trafo_tap"] = {
            "total": len(net.trafo),
            "has_tap_step_percent": int(tap_step.notna().sum()) if tap_step is not None else 0,
            "has_tap_pos": int(tap_pos.notna().sum()) if tap_pos is not None else 0,
        }

    # Line thermal ratings
    if len(net.line) > 0:
        max_i = net.line.get("max_i_ka")
        out["line_thermal"] = {
            "total": len(net.line),
            "finite_max_i_ka": int(np.isfinite(max_i).sum()) if max_i is not None else 0,
        }

    return out


def _convergence_snapshot(net) -> Dict[str, Any]:
    import pandapower as pp
    import numpy as np

    try:
        pp.runpp(net, calculate_voltage_angles=True, init="auto")
    except Exception as exc:
        return {"converged": False, "error": repr(exc)}

    if not net.converged:
        return {"converged": False, "error": "net.converged is False"}

    vm = net.res_bus["vm_pu"].to_numpy()
    return {
        "converged": True,
        "n_bus": int(vm.size),
        "vm_min": float(np.min(vm)),
        "vm_max": float(np.max(vm)),
        "vm_median": float(np.median(vm)),
        "n_below_0p9": int((vm < 0.9).sum()),
        "n_above_1p1": int((vm > 1.1).sum()),
        "p_load_mw": float(net.res_load["p_mw"].sum()) if len(net.res_load) else 0.0,
        "q_load_mvar": float(net.res_load["q_mvar"].sum()) if len(net.res_load) else 0.0,
        "p_gen_mw": float(net.res_gen["p_mw"].sum()) if len(net.res_gen) else 0.0,
        "q_gen_mvar": float(net.res_gen["q_mvar"].sum()) if len(net.res_gen) else 0.0,
    }


def _validate_against_pf(net) -> Dict[str, Any]:
    """Run pandapower's built-in PF-vs-pp comparison."""
    from pandapower.converter.powerfactory import validate_pf_conversion

    try:
        diffs = validate_pf_conversion(net)
    except Exception as exc:
        return {"ran": False, "error": repr(exc)}

    # ``diffs`` is typically a list of floats or a dict; handle both.
    if isinstance(diffs, dict):
        summary = {k: float(v) for k, v in diffs.items() if isinstance(v, (int, float))}
        max_abs = max((abs(v) for v in summary.values()), default=0.0)
    else:
        try:
            import numpy as np
            arr = np.asarray(diffs, dtype=float)
            summary = {
                "n": int(arr.size),
                "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
                "rms": float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0,
            }
            max_abs = summary["max_abs"]
        except Exception:
            summary = {"raw": repr(diffs)[:500]}
            max_abs = float("nan")

    return {"ran": True, "summary": summary, "max_abs_diff": max_abs}


def _name_samples(net, n: int = 5) -> Dict[str, list[str]]:
    """First few element names — lets us check profile-mapping feasibility."""
    def _take(df) -> list[str]:
        if "name" not in df.columns or len(df) == 0:
            return []
        return [str(x) for x in df["name"].head(n).tolist()]

    return {
        "bus_first_n": _take(net.bus),
        "gen_first_n": _take(net.gen),
        "sgen_first_n": _take(net.sgen),
        "load_first_n": _take(net.load),
        "trafo_first_n": _take(net.trafo),
    }


# ---------------------------------------------------------------------------
#  Main probe orchestration
# ---------------------------------------------------------------------------


def run_probe(
    project: str,
    *,
    study_case: str | None,
    pv_as_slack: bool,
    do_validate: bool,
) -> int:
    import pandapower as pp
    from pandapower.converter.powerfactory import from_pfd

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pandapower_version": pp.__version__,
        "project": project,
        "study_case": study_case,
        "pv_as_slack": pv_as_slack,
    }

    print(f"[probe] pandapower {pp.__version__}")
    print(f"[probe] opening PFSession(project='{project}', study_case={study_case!r}) ...")

    # -- Phase 1: open PF, convert, save ---------------------------------
    try:
        with PFSession(project, study_case=study_case) as session:
            print(f"[probe] PF app: {session.app}")
            print(f"[probe] active project: {session.project.loc_name}")
            print("[probe] calling from_pfd(...) — this can take 30-120s on a ~30-bus model")

            # ``from_pfd`` returns ``(net, controller)`` in current pandapower.
            result = from_pfd(
                session.app,
                project,
                pv_as_slack=pv_as_slack,
                tap_opt="nntap",
                export_controller=True,
                handle_us="Deactivate",
                is_unbalanced=False,
                create_sections=True,
                export_pf_ZoneArea=True,   # capture PF zone/area metadata if present
            )
            if isinstance(result, tuple):
                net = result[0]
                controller = result[1] if len(result) > 1 else None
            else:
                net = result
                controller = None
            report["controller_objects"] = (
                len(controller) if controller is not None and hasattr(controller, "__len__") else None
            )
    except Exception as exc:
        print(f"[probe] FAILED at conversion: {exc}")
        traceback.print_exc()
        report["stage_failed"] = "from_pfd"
        report["error"] = repr(exc)
        _write_report(report)
        return 1

    report["element_counts"] = _element_counts(net)
    report["parameter_completeness"] = _parameter_completeness(net)
    report["name_samples"] = _name_samples(net)

    print(f"[probe] element counts: {report['element_counts']}")

    # -- Phase 2: save converted net -------------------------------------
    try:
        pp.to_json(net, str(_NET_OUT))
        print(f"[probe] wrote {_NET_OUT}")
    except Exception as exc:
        print(f"[probe] WARNING: could not save net to JSON: {exc}")
        report["save_json_error"] = repr(exc)

    # -- Phase 3: convergence on the converted net -----------------------
    conv = _convergence_snapshot(net)
    report["convergence"] = conv
    print(f"[probe] pp.runpp convergence: {conv}")
    if not conv.get("converged", False):
        _write_report(report)
        return 2

    # -- Phase 4: PF vs pp validation ------------------------------------
    if do_validate:
        val = _validate_against_pf(net)
        report["pf_vs_pp_validation"] = val
        print(f"[probe] validate_pf_conversion: {val}")
        max_abs = val.get("max_abs_diff", 0.0)
        if isinstance(max_abs, float) and max_abs == max_abs and max_abs > 1e-2:
            # > 1% — worth flagging, not fatal.
            report["validation_warning"] = (
                f"max |PF - pp| = {max_abs:.4f}; investigate per-bus deltas before trusting "
                f"sensitivities."
            )
    else:
        report["pf_vs_pp_validation"] = "skipped"

    _write_report(report)

    # If validation showed diffs above 1e-1, flag as failure (clearly bad).
    if do_validate:
        max_abs = report.get("pf_vs_pp_validation", {}).get("max_abs_diff")
        if isinstance(max_abs, float) and max_abs == max_abs and max_abs > 1e-1:
            return 3

    print()
    print("[probe] DONE")
    print(f"[probe]   net:    {_NET_OUT}")
    print(f"[probe]   report: {_REPORT_OUT}")
    return 0


def _write_report(report: Dict[str, Any]) -> None:
    try:
        with open(_REPORT_OUT, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"[probe] wrote {_REPORT_OUT}")
    except Exception as exc:
        print(f"[probe] WARNING: could not save report: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Nordic SM (PF) to pandapower.")
    parser.add_argument("--project", default="Nordic_SM",
                        help="PowerFactory project name (default: Nordic_SM).")
    parser.add_argument("--study-case", default=None,
                        help="Study case to activate before conversion.")
    parser.add_argument("--pv-as-slack", action="store_true",
                        help="Import PV nodes as slacks (default: False).")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validate_pf_conversion step.")
    args = parser.parse_args()

    return run_probe(
        args.project,
        study_case=args.study_case,
        pv_as_slack=args.pv_as_slack,
        do_validate=not args.no_validate,
    )


if __name__ == "__main__":
    sys.exit(main())
