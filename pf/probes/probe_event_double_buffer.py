r"""Probe cyclic reuse of two pre-created PowerFactory event slots.

The one-object re-arm probe established that immediately moving a fired event
to a later time does not make it fire again.  This probe tests the proposed
constant-size fallback: two pre-created slots A/B are alternated, and each is
re-armed only after the other slot has fired.

Both ``EvtParam`` and ``EvtLod`` are exercised at t = 1, 3, 5, and 7 s.  A is
used at 1 and 5 s; B is used at 3 and 7 s.  Passing therefore requires the
same two objects to produce four distinct on-grid responses without creating
any event after ``ComInc``.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Any, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pf.probes.probe_event_rearm import (  # noqa: E402
    INERT_TIME_S,
    LOAD_STEP_PERCENT,
    RMS_STUDY_CASE,
    TIME_TOL_S,
    _full_name,
    _largest_jump_time,
    _qset_commands,
    _sample_at,
    _select_load,
    _select_qvpre,
)
from pf.replay import show_desktop  # noqa: E402
from pf.screening import ScreeningContext  # noqa: E402
from pf.session import DEFAULT_PROJECT_PATH, PFSessionError, connect  # noqa: E402


EVENT_TIMES_S = (1.0, 3.0, 5.0, 7.0)
STOP_TIMES_S = (2.0, 4.0, 6.0, 8.0)


def _new_param_slot(
    ctx: ScreeningContext, target: Any, initial_value: float, label: str
) -> Any:
    ev = ctx.evt_folder.CreateObject("EvtParam", f"probe_ab_qset_{label}")
    if ev is None:
        raise PFSessionError(f"EvtParam slot {label} creation failed")
    ev.SetAttribute("p_target", target)
    ev.SetAttribute("variable", "qset")
    ev.SetAttribute("value", repr(float(initial_value)))
    ev.SetAttribute("time", INERT_TIME_S)
    return ev


def _new_load_slot(ctx: ScreeningContext, target: Any, label: str) -> Any:
    ev = ctx.evt_folder.CreateObject("EvtLod", f"probe_ab_load_{label}")
    if ev is None:
        raise PFSessionError(f"EvtLod slot {label} creation failed")
    ev.SetAttribute("p_target", target)
    ev.SetAttribute("iopt_type", 0)
    ev.SetAttribute("dP", 0.0)
    ev.SetAttribute("dQ", 0.0)
    ev.SetAttribute("time", INERT_TIME_S)
    return ev


def _arm_param(ev: Any, value: float, when: float) -> None:
    ev.SetAttribute("value", repr(float(value)))
    ev.SetAttribute("time", float(when))


def _arm_load(ev: Any, when: float) -> None:
    ev.SetAttribute("dP", LOAD_STEP_PERCENT)
    ev.SetAttribute("dQ", 0.0)
    ev.SetAttribute("time", float(when))


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
        try:
            show_desktop(app)
        except Exception as exc:  # noqa: BLE001 - GUI is diagnostic only
            print(f"  [warn] PowerFactory GUI could not be shown: {exc}")

        ctx = ScreeningContext(app, verbose=False)
        ctx.purge_events()

        pre, q0, qmin, qmax = _select_qvpre(app)
        load = _select_load(app)
        q_a, q_b = _qset_commands(q0, qmin, qmax)

        param_slots = (
            _new_param_slot(ctx, pre, q0, "A"),
            _new_param_slot(ctx, pre, q0, "B"),
        )
        load_slots = (
            _new_load_slot(ctx, load, "A"),
            _new_load_slot(ctx, load, "B"),
        )
        identities = tuple(_full_name(ev) for ev in param_slots + load_slots)

        print("PowerFactory cyclic two-slot event re-arm probe")
        print(f"  EvtParam target: {_full_name(pre)}")
        print(f"  EvtLod target: {_full_name(load)}")
        print("  sequence: A@1 s, B@3 s, A(reuse)@5 s, B(reuse)@7 s")

        ctx.set_monitors([
            (pre, "s:Qext", "qext_pu"),
            (load, "m:P:bus1", "load_p_mw"),
        ])
        ctx.initialise()

        for index, (when, stop) in enumerate(zip(EVENT_TIMES_S, STOP_TIMES_S)):
            slot_index = index % 2
            q_command = q_a if slot_index == 0 else q_b
            _arm_param(param_slots[slot_index], q_command, when)
            _arm_load(load_slots[slot_index], when)
            print(
                f"  arm {'AB'[slot_index]}: qset={q_command:.6f}, "
                f"event t={when:.1f}s; simulate to {stop:.1f}s"
            )
            ctx.simulate(stop)

        tq, qext = ctx.read(pre, "s:Qext", stride=1)
        tp, load_p = ctx.read(load, "m:P:bus1", stride=1)
        checks: List[Tuple[str, bool, str]] = []

        folder_count = len(list(ctx.evt_folder.GetContents()))
        final_identities = tuple(
            _full_name(ev) for ev in param_slots + load_slots
        )
        checks.append((
            "constant four-object folder",
            folder_count == 4 and final_identities == identities,
            f"count={folder_count}; identities unchanged={final_identities == identities}",
        ))

        for index, when in enumerate(EVENT_TIMES_S):
            slot = "AB"[index % 2]
            generation = "first use" if index < 2 else "cyclic reuse"
            q_before = _sample_at(tq, qext, when - 0.05)
            q_after = _sample_at(tq, qext, when + 0.15)
            dq = q_after - q_before
            load_t, load_jump = _largest_jump_time(tp, load_p, when)
            load_ref = abs(_sample_at(tp, load_p, when - 0.05))
            q_ok = abs(dq) > 1.0e-4
            load_ok = (
                abs(load_t - when) <= TIME_TOL_S
                and abs(load_jump) > 0.02 * load_ref
            )
            checks.append((
                f"slot {slot} {generation} EvtParam fires at {when:g}s",
                q_ok,
                f"dQext={dq:+.6f} pu",
            ))
            checks.append((
                f"slot {slot} {generation} EvtLod fires at {when:g}s",
                load_ok,
                f"jump={load_jump:+.6f} MW at t={load_t:.3f}s",
            ))

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
