r"""Probe PowerFactory's persistent simulation-event re-arm semantics.

The RMS plant creates many ``EvtParam`` and ``EvtLod`` objects between
successive ``ComSim.Execute`` calls.  PF 2025 SP4 admits newly created events
to an active calculation too slowly for the profiles-on closed loop.  The
proposed fix keeps one persistent event per target/variable and rewrites its
payload and time instead.

This live probe verifies both semantics needed by that fix:

1. an event created before ``ComInc`` at an inert future time can be moved
   into the active simulation horizon and fires there; and
2. after firing, the same object can be re-armed by rewriting its payload and
   future time while the calculation remains active.

Both an ``EvtParam`` on a Q(V) pre-controller's ``qset`` and an ``EvtLod`` on
an in-service load are tested in one six-second RMS calculation.  The probe
is non-persistent with respect to plant state: ``finally`` resets the active
calculation and purges the two probe events.

Usage (PowerFactory machine)::

    python pf\probes\probe_event_rearm.py
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pf.screening import ScreeningContext  # noqa: E402
from pf.session import (  # noqa: E402
    DEFAULT_PROJECT_PATH,
    PFSessionError,
    connect,
    get_all,
)
from pf.wecc_apply import QVPRE_ELEMENT_NAME  # noqa: E402


RMS_STUDY_CASE = "02_RMS_CoSim"
INERT_TIME_S = 1.0e9
FIRST_EVENT_S = 1.0
SECOND_EVENT_S = 4.0
FIRST_STOP_S = 3.0
FINAL_STOP_S = 6.0
LOAD_STEP_PERCENT = 5.0
TIME_TOL_S = 0.03


def _full_name(obj: Any) -> str:
    try:
        return str(obj.GetFullName())
    except Exception:
        return str(getattr(obj, "loc_name", obj))


def _select_qvpre(app) -> Tuple[Any, float, float, float]:
    """Return a live QVPRE with usable qset headroom."""
    candidates = []
    for pre in get_all(app, "ElmDsl"):
        if pre.loc_name != QVPRE_ELEMENT_NAME:
            continue
        q0 = float(pre.GetAttribute("params:0"))
        qmin = float(pre.GetAttribute("params:5"))
        qmax = float(pre.GetAttribute("params:6"))
        if all(math.isfinite(x) for x in (q0, qmin, qmax)) and qmax > qmin:
            candidates.append((qmax - qmin, pre, q0, qmin, qmax))
    if not candidates:
        raise PFSessionError(
            f"no calculation-relevant {QVPRE_ELEMENT_NAME!r} ElmDsl with "
            "non-zero qset range"
        )
    _span, pre, q0, qmin, qmax = max(candidates, key=lambda row: row[0])
    return pre, q0, qmin, qmax


def _qset_commands(q0: float, qmin: float, qmax: float) -> Tuple[float, float]:
    """Two small, distinct, in-range commands in the roomiest direction."""
    room_up = qmax - q0
    room_down = q0 - qmin
    direction = 1.0 if room_up >= room_down else -1.0
    room = max(room_up, room_down)
    delta = min(0.02, 0.20 * room)
    if delta < 1.0e-4:
        raise PFSessionError(
            f"selected QVPRE has insufficient qset headroom: "
            f"q0={q0:.6g}, range=[{qmin:.6g}, {qmax:.6g}]"
        )
    return q0 + direction * delta, q0 + direction * 2.0 * delta


def _select_load(app) -> Any:
    """Largest in-service positive-P load, for an observable 5% step."""
    candidates = []
    for load in get_all(app, "ElmLod"):
        try:
            p = float(load.GetAttribute("plini"))
            cub = load.GetAttribute("bus1")
            in_service = int(load.GetAttribute("outserv")) == 0
        except Exception:
            continue
        if in_service and cub is not None and p > 1.0:
            candidates.append((p, load))
    if not candidates:
        raise PFSessionError("no in-service ElmLod with plini > 1 MW")
    return max(candidates, key=lambda row: row[0])[1]


def _sample_at(times: Sequence[float], values: Sequence[float], when: float) -> float:
    idx = min(range(len(times)), key=lambda i: abs(times[i] - when))
    return float(values[idx])


def _largest_jump_time(
    times: Sequence[float],
    values: Sequence[float],
    centre: float,
    half_width: float = 0.10,
) -> Tuple[float, float]:
    """Time and signed size of the largest adjacent-sample jump in a window."""
    pairs = [
        (float(times[i]), float(values[i]) - float(values[i - 1]))
        for i in range(1, len(times))
        if centre - half_width <= float(times[i]) <= centre + half_width
    ]
    if not pairs:
        raise PFSessionError(f"no result samples around t={centre:g}s")
    return max(pairs, key=lambda pair: abs(pair[1]))


def _create_param_event(ctx: ScreeningContext, target: Any, value: float):
    ev = ctx.evt_folder.CreateObject("EvtParam", "probe_rearm_qset")
    if ev is None:
        raise PFSessionError("EvtParam creation failed")
    ev.SetAttribute("p_target", target)
    ev.SetAttribute("variable", "qset")
    ev.SetAttribute("value", repr(float(value)))
    ev.SetAttribute("time", INERT_TIME_S)
    return ev


def _create_load_event(ctx: ScreeningContext, target: Any):
    ev = ctx.evt_folder.CreateObject("EvtLod", "probe_rearm_load")
    if ev is None:
        raise PFSessionError("EvtLod creation failed")
    ev.SetAttribute("p_target", target)
    ev.SetAttribute("iopt_type", 0)
    ev.SetAttribute("dP", 0.0)
    ev.SetAttribute("dQ", 0.0)
    ev.SetAttribute("time", INERT_TIME_S)
    return ev


def _arm_param(ev: Any, value: float, when: float) -> None:
    # Payload first, time last: the time rewrite is the explicit re-arm.
    ev.SetAttribute("value", repr(float(value)))
    ev.SetAttribute("time", float(when))


def _arm_load(ev: Any, d_p_percent: float, when: float) -> None:
    ev.SetAttribute("dP", float(d_p_percent))
    ev.SetAttribute("dQ", 0.0)
    ev.SetAttribute("time", float(when))


def _event_count(ctx: ScreeningContext) -> int:
    return len(list(ctx.evt_folder.GetContents()))


def _print_checks(checks: Iterable[Tuple[str, bool, str]]) -> bool:
    ok = True
    for label, passed, evidence in checks:
        ok &= bool(passed)
        print(f"  {'PASS' if passed else 'FAIL'} {label}: {evidence}")
    return ok


def main() -> int:
    ctx = None
    try:
        app = connect(DEFAULT_PROJECT_PATH, study_case=RMS_STUDY_CASE)
        ctx = ScreeningContext(app, verbose=False)
        ctx.purge_events()

        pre, q0, qmin, qmax = _select_qvpre(app)
        load = _select_load(app)
        q_first, q_second = _qset_commands(q0, qmin, qmax)

        print("PowerFactory persistent-event re-arm probe")
        print(f"  EvtParam target: {_full_name(pre)}")
        print(
            f"  qset baseline/range/commands: {q0:.6f} / "
            f"[{qmin:.6f}, {qmax:.6f}] / {q_first:.6f}, {q_second:.6f}"
        )
        print(
            f"  EvtLod target: {_full_name(load)} "
            f"(plini={float(load.GetAttribute('plini')):.6f} MW)"
        )

        param_ev = _create_param_event(ctx, pre, q0)
        load_ev = _create_load_event(ctx, load)
        param_name = _full_name(param_ev)
        load_name = _full_name(load_ev)
        preinit_count = _event_count(ctx)

        ctx.set_monitors([
            (pre, "s:Qext", "qext_pu"),
            (load, "m:P:bus1", "load_p_mw"),
        ])
        ctx.initialise()

        # Move both pre-registered far-future objects into the active horizon.
        _arm_param(param_ev, q_first, FIRST_EVENT_S)
        _arm_load(load_ev, LOAD_STEP_PERCENT, FIRST_EVENT_S)
        ctx.simulate(FIRST_STOP_S)
        q_after_first = float(pre.GetAttribute("qset"))
        p_after_first = float(load.GetAttribute("m:P:bus1"))

        # Re-arm those exact fired objects without resetting/reinitialising.
        _arm_param(param_ev, q_second, SECOND_EVENT_S)
        _arm_load(load_ev, LOAD_STEP_PERCENT, SECOND_EVENT_S)
        ctx.simulate(FINAL_STOP_S)
        q_after_second = float(pre.GetAttribute("qset"))
        p_after_second = float(load.GetAttribute("m:P:bus1"))

        tq, qext = ctx.read(pre, "s:Qext", stride=1)
        tp, load_p = ctx.read(load, "m:P:bus1", stride=1)
        load_t1, load_jump1 = _largest_jump_time(
            tp, load_p, FIRST_EVENT_S
        )
        load_t2, load_jump2 = _largest_jump_time(
            tp, load_p, SECOND_EVENT_S
        )
        qext_before_1 = _sample_at(tq, qext, FIRST_EVENT_S - 0.05)
        qext_after_1 = _sample_at(tq, qext, FIRST_EVENT_S + 0.15)
        qext_before_2 = _sample_at(tq, qext, SECOND_EVENT_S - 0.05)
        qext_after_2 = _sample_at(tq, qext, SECOND_EVENT_S + 0.15)
        final_count = _event_count(ctx)

        load_ref1 = abs(_sample_at(tp, load_p, FIRST_EVENT_S - 0.05))
        load_ref2 = abs(_sample_at(tp, load_p, SECOND_EVENT_S - 0.05))
        checks = [
            (
                "pre-ComInc objects retained",
                preinit_count == 2 and final_count == 2,
                f"event count {preinit_count} -> {final_count}",
            ),
            (
                "same EvtParam object retained",
                _full_name(param_ev) == param_name,
                _full_name(param_ev),
            ),
            (
                "same EvtLod object retained",
                _full_name(load_ev) == load_name,
                _full_name(load_ev),
            ),
            (
                "EvtParam first arm fired",
                math.isclose(q_after_first, q_first, rel_tol=0.0, abs_tol=1e-6),
                f"qset={q_after_first:.6f}, expected={q_first:.6f}",
            ),
            (
                "EvtParam fired-object re-arm fired",
                math.isclose(q_after_second, q_second, rel_tol=0.0, abs_tol=1e-6),
                f"qset={q_after_second:.6f}, expected={q_second:.6f}",
            ),
            (
                "EvtLod first arm fired on time",
                abs(load_t1 - FIRST_EVENT_S) <= TIME_TOL_S
                and load_jump1 > 0.02 * load_ref1,
                f"largest jump={load_jump1:+.6f} MW at t={load_t1:.3f}s",
            ),
            (
                "EvtLod fired-object re-arm fired on time",
                abs(load_t2 - SECOND_EVENT_S) <= TIME_TOL_S
                and load_jump2 > 0.02 * load_ref2,
                f"largest jump={load_jump2:+.6f} MW at t={load_t2:.3f}s",
            ),
            (
                "QVPRE output responds to both qset arms",
                abs(qext_after_1 - qext_before_1) > 1e-4
                and abs(qext_after_2 - qext_before_2) > 1e-4,
                f"dQext={qext_after_1 - qext_before_1:+.6f}, "
                f"{qext_after_2 - qext_before_2:+.6f} pu",
            ),
        ]
        passed = _print_checks(checks)
        print(
            f"  load final samples after arms: "
            f"{p_after_first:.6f}, {p_after_second:.6f} MW"
        )
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
            except Exception as exc:  # noqa: BLE001 - do not mask probe result
                print(f"cleanup failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
