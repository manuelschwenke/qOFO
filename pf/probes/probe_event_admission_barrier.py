r"""Probe extra ``ComSim.Execute`` calls as an event-admission barrier.

PowerFactory admits only a bounded number of event objects created during an
active RMS calculation on each ``ComSim.Execute`` call.  This probe creates a
200-object batch *after* ``ComInc``, schedules it for t=1 s, and advances to
two earlier barrier stops before crossing the event time.  Passing means the
third Execute sees the entire batch before it is due, avoiding both a
horizon-sized preallocated pool and late execution.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pf.probes.probe_event_preallocation_volume import (  # noqa: E402
    DP_PERCENT,
    _largest_ratio_step,
    _select_medium_load,
)
from pf.probes.probe_event_rearm import (  # noqa: E402
    RMS_STUDY_CASE,
    _full_name,
    _qset_commands,
    _sample_at,
    _select_qvpre,
)
from pf.replay import show_desktop  # noqa: E402
from pf.screening import ScreeningContext  # noqa: E402
from pf.session import DEFAULT_PROJECT_PATH, PFSessionError, connect  # noqa: E402


LOAD_EVENT_COUNT = 199
EVENT_TIME_S = 1.0
BARRIER_STOPS_S = (0.10, 0.20)
FINAL_STOP_S = 2.0


def _new_dynamic_load_event(
    ctx: ScreeningContext, target: Any, seq: int
) -> Any:
    ev = ctx.evt_folder.CreateObject("EvtLod", f"probe_barrier_lod_{seq:03d}")
    if ev is None:
        raise PFSessionError(f"EvtLod {seq} creation failed")
    ev.SetAttribute("p_target", target)
    ev.SetAttribute("iopt_type", 0)
    ev.SetAttribute("dP", DP_PERCENT)
    ev.SetAttribute("dQ", 0.0)
    ev.SetAttribute("time", EVENT_TIME_S)
    return ev


def _new_dynamic_tail_param(
    ctx: ScreeningContext, target: Any, value: float
) -> Any:
    ev = ctx.evt_folder.CreateObject("EvtParam", "probe_barrier_tail_qset")
    if ev is None:
        raise PFSessionError("tail EvtParam creation failed")
    ev.SetAttribute("p_target", target)
    ev.SetAttribute("variable", "qset")
    ev.SetAttribute("value", repr(float(value)))
    ev.SetAttribute("time", EVENT_TIME_S)
    return ev


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
        load = _select_medium_load(app)
        pre, q0, qmin, qmax = _select_qvpre(app)
        q_command, _ = _qset_commands(q0, qmin, qmax)

        ctx.set_monitors([
            (load, "m:P:bus1", "load_p_mw"),
            (pre, "s:Qext", "qext_pu"),
        ])
        ctx.initialise()

        # Deliberately create every event after ComInc.  The tail EvtParam is
        # last in creation order, so its on-time response demonstrates that
        # the full 200-object queue was admitted before t=1 s.
        load_events = [
            _new_dynamic_load_event(ctx, load, seq)
            for seq in range(LOAD_EVENT_COUNT)
        ]
        tail_param = _new_dynamic_tail_param(ctx, pre, q_command)

        print("PowerFactory dynamic-event admission-barrier probe")
        print(
            f"  created after ComInc: {len(load_events)} EvtLod + "
            "1 tail EvtParam"
        )
        print(f"  load target: {_full_name(load)}")
        print(f"  tail target: {_full_name(pre)}")
        print(
            f"  barriers={BARRIER_STOPS_S}; all events scheduled "
            f"for t={EVENT_TIME_S:.1f}s"
        )

        for stop in BARRIER_STOPS_S:
            ctx.simulate(stop)
            print(f"  admission Execute completed at tstop={stop:.2f}s")
        ctx.simulate(FINAL_STOP_S)

        tp, load_p = ctx.read(load, "m:P:bus1", stride=1)
        tq, qext = ctx.read(pre, "s:Qext", stride=1)
        load_t, ratio = _largest_ratio_step(tp, load_p, EVENT_TIME_S)
        expected_ratio = (1.0 + DP_PERCENT / 100.0) ** LOAD_EVENT_COUNT
        dq = _sample_at(tq, qext, EVENT_TIME_S + 0.15) - _sample_at(
            tq, qext, EVENT_TIME_S - 0.05
        )
        event_count = len(list(ctx.evt_folder.GetContents()))

        checks = [
            (
                "dynamic folder contains full batch",
                event_count == LOAD_EVENT_COUNT + 1,
                f"count={event_count}",
            ),
            (
                "tail EvtParam is admitted and fires on-grid",
                abs(dq) > 1.0e-4,
                f"dQext={dq:+.6f} pu",
            ),
            (
                "load batch fires on-grid",
                abs(load_t - EVENT_TIME_S) <= 0.03
                and math.isclose(ratio, expected_ratio, rel_tol=0.0, abs_tol=0.005),
                f"ratio={ratio:.6f}, expected={expected_ratio:.6f}, "
                f"t={load_t:.3f}s",
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
