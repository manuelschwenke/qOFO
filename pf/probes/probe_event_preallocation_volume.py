r"""Stress-probe pre-created one-shot PowerFactory event slots.

``probe_event_rearm.py`` establishes that a fired event cannot be re-armed,
but that an unused event created before ``ComInc`` can be moved from an inert
future time into the active horizon.  This follow-up tests that operation at
the profiles-on dispatch volume: two batches of 180 fresh ``EvtLod`` slots,
plus a fresh ``EvtParam`` sentinel at the tail of each batch.

The calculation is reset and all probe events are purged in ``finally``.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pf.probes.probe_event_rearm import (  # noqa: E402
    INERT_TIME_S,
    RMS_STUDY_CASE,
    _full_name,
    _qset_commands,
    _sample_at,
    _select_qvpre,
)
from pf.screening import ScreeningContext  # noqa: E402
from pf.session import DEFAULT_PROJECT_PATH, PFSessionError, connect, get_all  # noqa: E402


BATCH_SIZE = 180
DP_PERCENT = 0.02
FIRST_EVENT_S = 1.0
SECOND_EVENT_S = 4.0
FIRST_STOP_S = 3.0
FINAL_STOP_S = 6.0


def _select_medium_load(app) -> Any:
    candidates = []
    for load in get_all(app, "ElmLod"):
        try:
            p = float(load.GetAttribute("plini"))
            cub = load.GetAttribute("bus1")
            in_service = int(load.GetAttribute("outserv")) == 0
        except Exception:
            continue
        if in_service and cub is not None and p > 5.0:
            candidates.append((abs(p - 50.0), load))
    if not candidates:
        raise PFSessionError("no in-service ElmLod above 5 MW")
    return min(candidates, key=lambda row: row[0])[1]


def _new_load_slot(ctx: ScreeningContext, load: Any, seq: int) -> Any:
    ev = ctx.evt_folder.CreateObject("EvtLod", f"probe_slot_lod_{seq:03d}")
    if ev is None:
        raise PFSessionError(f"EvtLod slot {seq} creation failed")
    ev.SetAttribute("p_target", load)
    ev.SetAttribute("iopt_type", 0)
    ev.SetAttribute("dP", 0.0)
    ev.SetAttribute("dQ", 0.0)
    ev.SetAttribute("time", INERT_TIME_S)
    return ev


def _new_param_slot(
    ctx: ScreeningContext, target: Any, initial_value: float, seq: int
) -> Any:
    ev = ctx.evt_folder.CreateObject("EvtParam", f"probe_slot_qset_{seq}")
    if ev is None:
        raise PFSessionError(f"EvtParam sentinel {seq} creation failed")
    ev.SetAttribute("p_target", target)
    ev.SetAttribute("variable", "qset")
    ev.SetAttribute("value", repr(float(initial_value)))
    ev.SetAttribute("time", INERT_TIME_S)
    return ev


def _arm_load_batch(events: Sequence[Any], when: float) -> None:
    for ev in events:
        ev.SetAttribute("dP", DP_PERCENT)
        ev.SetAttribute("dQ", 0.0)
        ev.SetAttribute("time", float(when))


def _arm_param(ev: Any, value: float, when: float) -> None:
    ev.SetAttribute("value", repr(float(value)))
    ev.SetAttribute("time", float(when))


def _largest_ratio_step(
    times: Sequence[float], values: Sequence[float], centre: float
) -> Tuple[float, float]:
    candidates = []
    for i in range(1, len(times)):
        if centre - 0.10 <= float(times[i]) <= centre + 0.10:
            before = float(values[i - 1])
            after = float(values[i])
            if abs(before) > 1e-9:
                candidates.append((abs(after - before), float(times[i]), after / before))
    if not candidates:
        raise PFSessionError(f"no load samples around t={centre:g}s")
    _jump, time_s, ratio = max(candidates, key=lambda row: row[0])
    return time_s, ratio


def main() -> int:
    ctx = None
    try:
        app = connect(DEFAULT_PROJECT_PATH, study_case=RMS_STUDY_CASE)
        ctx = ScreeningContext(app, verbose=False)
        ctx.purge_events()

        load = _select_medium_load(app)
        pre, q0, qmin, qmax = _select_qvpre(app)
        q_first, q_second = _qset_commands(q0, qmin, qmax)

        # Creation order deliberately places each EvtParam sentinel after its
        # 180-object load batch, exercising objects beyond the inferred ~90
        # new-event admission boundary.
        batch1 = [_new_load_slot(ctx, load, i) for i in range(BATCH_SIZE)]
        param1 = _new_param_slot(ctx, pre, q0, 1)
        batch2 = [
            _new_load_slot(ctx, load, BATCH_SIZE + i)
            for i in range(BATCH_SIZE)
        ]
        param2 = _new_param_slot(ctx, pre, q0, 2)
        expected_count = 2 * BATCH_SIZE + 2

        print("PowerFactory pre-created event-slot volume probe")
        print(
            f"  load={_full_name(load)}; plini="
            f"{float(load.GetAttribute('plini')):.6f} MW"
        )
        print(f"  qset target={_full_name(pre)}")
        print(
            f"  pre-created={expected_count} events; "
            f"batch={BATCH_SIZE} EvtLod + 1 tail EvtParam"
        )

        ctx.set_monitors([
            (load, "m:P:bus1", "load_p_mw"),
            (pre, "s:Qext", "qext_pu"),
        ])
        ctx.initialise()

        _arm_load_batch(batch1, FIRST_EVENT_S)
        _arm_param(param1, q_first, FIRST_EVENT_S)
        ctx.simulate(FIRST_STOP_S)

        _arm_load_batch(batch2, SECOND_EVENT_S)
        _arm_param(param2, q_second, SECOND_EVENT_S)
        ctx.simulate(FINAL_STOP_S)

        tp, p = ctx.read(load, "m:P:bus1", stride=1)
        tq, qext = ctx.read(pre, "s:Qext", stride=1)
        t1, ratio1 = _largest_ratio_step(tp, p, FIRST_EVENT_S)
        t2, ratio2 = _largest_ratio_step(tp, p, SECOND_EVENT_S)
        expected_ratio = (1.0 + DP_PERCENT / 100.0) ** BATCH_SIZE
        dq1 = _sample_at(tq, qext, FIRST_EVENT_S + 0.15) - _sample_at(
            tq, qext, FIRST_EVENT_S - 0.05
        )
        dq2 = _sample_at(tq, qext, SECOND_EVENT_S + 0.15) - _sample_at(
            tq, qext, SECOND_EVENT_S - 0.05
        )
        event_count = len(list(ctx.evt_folder.GetContents()))

        checks = [
            (
                "folder count remains constant",
                event_count == expected_count,
                f"{event_count} == {expected_count}",
            ),
            (
                "first 180-event batch fires",
                abs(ratio1 - expected_ratio) <= 0.005
                and abs(t1 - FIRST_EVENT_S) <= 0.03,
                f"ratio={ratio1:.6f}, expected={expected_ratio:.6f}, t={t1:.3f}s",
            ),
            (
                "second fresh 180-event batch fires",
                abs(ratio2 - expected_ratio) <= 0.005
                and abs(t2 - SECOND_EVENT_S) <= 0.03,
                f"ratio={ratio2:.6f}, expected={expected_ratio:.6f}, t={t2:.3f}s",
            ),
            (
                "tail EvtParam sentinels fire in both batches",
                abs(dq1) > 1e-4 and abs(dq2) > 1e-4,
                f"dQext={dq1:+.6f}, {dq2:+.6f} pu",
            ),
        ]
        passed = True
        for label, ok, evidence in checks:
            passed &= bool(ok)
            print(f"  {'PASS' if ok else 'FAIL'} {label}: {evidence}")
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
