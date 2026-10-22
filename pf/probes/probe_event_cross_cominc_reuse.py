r"""Probe reuse of fired PowerFactory events after a fresh ComInc.

An event is one-shot inside one active calculation.  The persistent-pool
architecture instead retains the object, moves it to an inert time after it
fires, resets the calculation, and reuses the exact object after the next
``ComInc``.  This probe verifies that lifecycle for EvtParam and EvtLod.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pf.probes.probe_event_rearm import (  # noqa: E402
    FIRST_EVENT_S,
    INERT_TIME_S,
    LOAD_STEP_PERCENT,
    RMS_STUDY_CASE,
    TIME_TOL_S,
    _create_load_event,
    _create_param_event,
    _full_name,
    _largest_jump_time,
    _qset_commands,
    _sample_at,
    _select_load,
    _select_qvpre,
)
from pf.screening import ScreeningContext  # noqa: E402
from pf.session import DEFAULT_PROJECT_PATH, connect  # noqa: E402


FINAL_STOP_S = 2.0


def _arm_param(ev: Any, value: float) -> None:
    ev.SetAttribute("value", repr(float(value)))
    ev.SetAttribute("time", FIRST_EVENT_S)


def _arm_load(ev: Any) -> None:
    ev.SetAttribute("dP", LOAD_STEP_PERCENT)
    ev.SetAttribute("dQ", 0.0)
    ev.SetAttribute("time", FIRST_EVENT_S)


def _run_once(
    ctx: ScreeningContext, pre: Any, load: Any, param_ev: Any, load_ev: Any,
    q_command: float,
) -> Tuple[float, float, float]:
    _arm_param(param_ev, q_command)
    _arm_load(load_ev)
    ctx.simulate(FINAL_STOP_S)
    tq, qext = ctx.read(pre, "s:Qext", stride=1)
    tp, load_p = ctx.read(load, "m:P:bus1", stride=1)
    dq = _sample_at(tq, qext, FIRST_EVENT_S + 0.15) - _sample_at(
        tq, qext, FIRST_EVENT_S - 0.05
    )
    load_t, load_jump = _largest_jump_time(tp, load_p, FIRST_EVENT_S)
    return dq, load_t, load_jump


def _print_checks(checks: Sequence[Tuple[str, bool, str]]) -> bool:
    passed = True
    for label, ok, evidence in checks:
        passed &= bool(ok)
        print(f"  {'PASS' if ok else 'FAIL'} {label}: {evidence}")
    return passed


def main() -> int:
    ctx = None
    try:
        app = connect(DEFAULT_PROJECT_PATH, study_case=RMS_STUDY_CASE)
        ctx = ScreeningContext(app, verbose=False)
        ctx.purge_events()

        pre, q0, qmin, qmax = _select_qvpre(app)
        load = _select_load(app)
        q_first, q_second = _qset_commands(q0, qmin, qmax)
        param_ev = _create_param_event(ctx, pre, q0)
        load_ev = _create_load_event(ctx, load)
        identities = (_full_name(param_ev), _full_name(load_ev))

        ctx.set_monitors([
            (pre, "s:Qext", "qext_pu"),
            (load, "m:P:bus1", "load_p_mw"),
        ])
        ctx.initialise()

        first = _run_once(
            ctx, pre, load, param_ev, load_ev, q_first
        )

        # Persist a safe inert schedule while the first calculation remains
        # active.  It must not re-fire there, but the attribute value must be
        # available to the next ComInc.
        param_ev.SetAttribute("time", INERT_TIME_S)
        load_ev.SetAttribute("time", INERT_TIME_S)
        app.ResetCalculation()
        ctx.initialise()

        second = _run_once(
            ctx, pre, load, param_ev, load_ev, q_second
        )
        final_identities = (_full_name(param_ev), _full_name(load_ev))
        folder_count = len(list(ctx.evt_folder.GetContents()))

        checks = [
            (
                "same two objects survive ResetCalculation + ComInc",
                folder_count == 2 and identities == final_identities,
                f"count={folder_count}; identities unchanged="
                f"{identities == final_identities}",
            ),
            (
                "EvtParam fires in first calculation",
                abs(first[0]) > 1.0e-4,
                f"dQext={first[0]:+.6f} pu",
            ),
            (
                "EvtLod fires in first calculation",
                abs(first[1] - FIRST_EVENT_S) <= TIME_TOL_S
                and abs(first[2]) > 1.0e-3,
                f"jump={first[2]:+.6f} MW at t={first[1]:.3f}s",
            ),
            (
                "same EvtParam fires after new ComInc",
                abs(second[0]) > 1.0e-4,
                f"dQext={second[0]:+.6f} pu",
            ),
            (
                "same EvtLod fires after new ComInc",
                abs(second[1] - FIRST_EVENT_S) <= TIME_TOL_S
                and abs(second[2]) > 1.0e-3,
                f"jump={second[2]:+.6f} MW at t={second[1]:.3f}s",
            ),
        ]
        passed = _print_checks(checks)
        print(f"PROBE_RESULT={'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1
    except Exception as exc:  # noqa: BLE001 - live diagnostic entry point
        print(f"PROBE_RESULT=ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 2
    finally:
        if ctx is not None:
            try:
                ctx.purge_events()
                print("cleanup: calculation reset; probe events purged")
            except Exception as exc:  # noqa: BLE001 - do not mask result
                print(f"cleanup failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
