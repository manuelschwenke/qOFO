r"""Probe direct DSL-parameter writes inside one active RMS calculation.

The proposed event-free controller path writes a QVPRE command while
``ComSim`` is paused, then continues the same calculation without rerunning
``ComInc``.  This probe answers the prerequisite semantic question before a
mailbox/dispatcher model is built.

The probe is isolated from the replay infrastructure:

* a temporary empty ``IntEvt`` prevents registration of the persistent
  9,511-object replay pool;
* a temporary ``ElmRes`` prevents changes to the normal monitor catalogue;
* the original ``ComInc`` pointers and QVPRE parameter are restored in
  ``finally``.

Usage::

    python pf\probes\probe_live_dsl_parameter.py
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import traceback
from typing import Any, Sequence, Tuple

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
FIRST_WRITE_S = 0.5
SECOND_WRITE_S = 1.5
FINAL_STOP_S = 2.5
SAMPLE_OFFSET_S = 0.10

TEMP_EVENT_FOLDER = "qOFO Live DSL Parameter Probe Events"
TEMP_RESULT = "qOFO Live DSL Parameter Probe Results"


def _select_qvpre(app) -> Tuple[Any, Any, float, float, float]:
    """Return a QVPRE, its park, and a finite qset operating interval."""

    candidates = []
    for pre in get_all(app, "ElmDsl"):
        if pre.loc_name != QVPRE_ELEMENT_NAME:
            continue
        q0 = float(pre.GetAttribute("params:0"))
        qmin = float(pre.GetAttribute("params:5"))
        qmax = float(pre.GetAttribute("params:6"))
        comp = pre.GetParent()
        park = next(
            (
                obj for obj in comp.GetAttribute("pelm")
                if obj is not None and obj.GetClassName() == "ElmGenstat"
            ),
            None,
        )
        if (
            park is not None
            and int(park.GetAttribute("outserv")) == 0
            and all(math.isfinite(value) for value in (q0, qmin, qmax))
            and qmax > qmin
        ):
            candidates.append((qmax - qmin, pre, park, q0, qmin, qmax))
    if not candidates:
        raise PFSessionError("no usable calculation-relevant QVPRE found")
    _span, pre, park, q0, qmin, qmax = max(
        candidates, key=lambda row: row[0]
    )
    return pre, park, q0, qmin, qmax


def _commands(q0: float, qmin: float, qmax: float) -> Tuple[float, float]:
    room_up = qmax - q0
    room_down = q0 - qmin
    direction = 1.0 if room_up >= room_down else -1.0
    room = max(room_up, room_down)
    delta = min(0.03, 0.15 * room)
    if delta < 1.0e-4:
        raise PFSessionError(
            f"QVPRE has insufficient command headroom: "
            f"q0={q0}, [{qmin}, {qmax}]"
        )
    return q0 + direction * delta, q0 + direction * 2.0 * delta


def _sample_at(
    time: Sequence[float], values: Sequence[float], when: float
) -> float:
    index = min(range(len(time)), key=lambda i: abs(float(time[i]) - when))
    return float(values[index])

def _max_prefix_delta(before: Sequence[float], after: Sequence[float]) -> float:
    if len(after) < len(before):
        return math.inf
    return max(
        (abs(float(a) - float(b)) for a, b in zip(before, after)),
        default=0.0,
    )


def _delete_named(parent, class_name: str, name: str) -> None:
    for obj in list(parent.GetContents(f"{name}.{class_name}")):
        obj.Delete()


def main() -> int:
    app = None
    inc = None
    original_events = None
    original_result = None
    temp_events = None
    temp_result = None
    pre = None
    q0 = None
    try:
        app = connect(DEFAULT_PROJECT_PATH, study_case=RMS_STUDY_CASE)
        study = app.GetActiveStudyCase()
        inc = app.GetFromStudyCase("ComInc")
        original_events = inc.GetAttribute("p_event")
        original_result = inc.GetAttribute("p_resvar")

        app.ResetCalculation()
        _delete_named(study, "IntEvt", TEMP_EVENT_FOLDER)
        _delete_named(study, "ElmRes", TEMP_RESULT)
        temp_events = study.CreateObject("IntEvt", TEMP_EVENT_FOLDER)
        temp_result = study.CreateObject("ElmRes", TEMP_RESULT)
        if temp_events is None or temp_result is None:
            raise PFSessionError("failed to create isolated probe folders")
        inc.SetAttribute("p_event", temp_events)
        inc.SetAttribute("p_resvar", temp_result)

        ctx = ScreeningContext(app, verbose=False)
        ctx.res = temp_result
        pre, park, q0, qmin, qmax = _select_qvpre(app)
        q1, q2 = _commands(q0, qmin, qmax)
        ctx.set_monitors([
            (pre, "s:qset", "qset"),
            (pre, "s:Qext", "qext"),
            (park, "m:Q:bus1", "park_q"),
        ])

        print("PowerFactory active-RMS direct DSL-parameter probe")
        print(f"  QVPRE: {pre.GetFullName()}")
        print(f"  park:  {park.GetFullName()}")
        print(
            f"  qset baseline/range/commands: {q0:.6f} / "
            f"[{qmin:.6f}, {qmax:.6f}] / {q1:.6f}, {q2:.6f}"
        )
        print(
            f"  isolated event folder count: "
            f"{len(list(temp_events.GetContents()))}"
        )

        ctx.initialise()
        ctx.simulate(FIRST_WRITE_S)
        _t0e, qext_hist0 = ctx.read(pre, "s:Qext", stride=1)
        live0 = (
            float(pre.GetAttribute("qset")),
            float(pre.GetAttribute("s:Qext")),
            float(park.GetAttribute("m:Q:bus1")),
        )
        pre.SetAttribute("params:0", float(q1))
        first_database_value = float(pre.GetAttribute("params:0"))
        ctx.simulate(SECOND_WRITE_S)
        _t1e, qext_hist1 = ctx.read(pre, "s:Qext", stride=1)
        live1 = (
            float(pre.GetAttribute("qset")),
            float(pre.GetAttribute("s:Qext")),
            float(park.GetAttribute("m:Q:bus1")),
        )
        pre.SetAttribute("params:0", float(q2))
        second_database_value = float(pre.GetAttribute("params:0"))
        ctx.simulate(FINAL_STOP_S)
        _t2e, qext_hist2 = ctx.read(pre, "s:Qext", stride=1)
        live2 = (
            float(pre.GetAttribute("qset")),
            float(pre.GetAttribute("s:Qext")),
            float(park.GetAttribute("m:Q:bus1")),
        )
        prefix_change_after_first = _max_prefix_delta(qext_hist0, qext_hist1)
        prefix_change_after_second = _max_prefix_delta(qext_hist1, qext_hist2)

        tq, qset = ctx.read(pre, "s:qset", stride=1)
        te, qext = ctx.read(pre, "s:Qext", stride=1)
        tp, park_q = ctx.read(park, "m:Q:bus1", stride=1)
        qset0 = _sample_at(tq, qset, FIRST_WRITE_S - SAMPLE_OFFSET_S)
        qset1 = _sample_at(tq, qset, FIRST_WRITE_S + SAMPLE_OFFSET_S)
        qset2 = _sample_at(tq, qset, SECOND_WRITE_S + SAMPLE_OFFSET_S)
        qext0 = _sample_at(te, qext, FIRST_WRITE_S - SAMPLE_OFFSET_S)
        qext1 = _sample_at(te, qext, FIRST_WRITE_S + SAMPLE_OFFSET_S)
        qext2 = _sample_at(te, qext, SECOND_WRITE_S + SAMPLE_OFFSET_S)
        park0 = _sample_at(tp, park_q, FIRST_WRITE_S - SAMPLE_OFFSET_S)
        park1 = _sample_at(tp, park_q, FIRST_WRITE_S + SAMPLE_OFFSET_S)
        park2 = _sample_at(tp, park_q, SECOND_WRITE_S + SAMPLE_OFFSET_S)

        time_continuous = (
            len(tq) > 0
            and abs(float(tq[0])) <= 0.02
            and abs(float(tq[-1]) - FINAL_STOP_S) <= 0.02
            and all(float(b) >= float(a) for a, b in zip(tq, tq[1:]))
        )
        qset_visible = (
            math.isclose(qset0, q0, rel_tol=0.0, abs_tol=1.0e-5)
            and math.isclose(qset1, q1, rel_tol=0.0, abs_tol=1.0e-5)
            and math.isclose(qset2, q2, rel_tol=0.0, abs_tol=1.0e-5)
        )
        direction = math.copysign(1.0, q1 - q0)
        qext_responded = (
            direction * (qext1 - qext0) > 0.25 * abs(q1 - q0)
            and direction * (qext2 - qext1) > 0.25 * abs(q2 - q1)
        )
        park_responded = (
            direction * (park1 - park0) > 0.0
            and direction * (park2 - park1) > 0.0
        )
        checks = [
            (
                "database writes accepted",
                math.isclose(first_database_value, q1, abs_tol=1.0e-12)
                and math.isclose(second_database_value, q2, abs_tol=1.0e-12),
                f"params:0 -> {first_database_value:.6f}, "
                f"{second_database_value:.6f}",
            ),
            (
                "same active RMS timeline continued",
                time_continuous,
                f"rows={len(tq)}, time=[{tq[0]:.3f}, {tq[-1]:.3f}]",
            ),
            (
                "runtime qset consumed both writes",
                qset_visible,
                f"{qset0:.6f} -> {qset1:.6f} -> {qset2:.6f}",
            ),
            (
                "QVPRE output responded twice",
                qext_responded,
                f"{qext0:.6f} -> {qext1:.6f} -> {qext2:.6f} pu",
            ),
            (
                "physical park Q responded twice",
                park_responded,
                f"{park0:.6f} -> {park1:.6f} -> {park2:.6f} Mvar",
            ),
            (
                "no simulation events used",
                len(list(temp_events.GetContents())) == 0,
                f"event count={len(list(temp_events.GetContents()))}",
            ),
        ]
        prefix_stable = (
            prefix_change_after_first <= 1.0e-9
            and prefix_change_after_second <= 1.0e-9
        )
        checks.insert(
            2,
            (
                "previous result prefix remained unchanged",
                prefix_stable,
                f"max qext prefix deltas={prefix_change_after_first:.6g}, "
                f"{prefix_change_after_second:.6g}; paused qset="
                f"{live0[0]:.6f}, {live1[0]:.6f}, {live2[0]:.6f}",
            ),
        )
        passed = True
        for label, ok, evidence in checks:
            passed &= bool(ok)
            print(f"  {'PASS' if ok else 'FAIL'} {label}: {evidence}")
        print(
            "DIRECT_DSL_PARAMETER_PROBE="
            + ("PASS" if passed else "FAIL")
        )
        return 0 if passed else 2
    except Exception as exc:  # noqa: BLE001
        print(
            f"DIRECT_DSL_PARAMETER_PROBE=ERROR "
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return 3
    finally:
        if app is not None:
            try:
                app.ResetCalculation()
            except Exception:
                pass
        if pre is not None and q0 is not None:
            try:
                pre.SetAttribute("params:0", float(q0))
            except Exception:
                pass
        if inc is not None and original_events is not None:
            try:
                inc.SetAttribute("p_event", original_events)
            except Exception:
                pass
        if inc is not None and original_result is not None:
            try:
                inc.SetAttribute("p_resvar", original_result)
            except Exception:
                pass
        for obj in (temp_result, temp_events):
            if obj is not None:
                try:
                    obj.Delete()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
