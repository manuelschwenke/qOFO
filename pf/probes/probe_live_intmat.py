r"""Probe live ``IntMat`` updates inside one active RMS calculation.

The selected WECC REEC_D already reads its child ``vdlp.IntMat`` through the
DSL lookup-table mechanism, so it is a faithful test of the proposed mailbox
semantics without creating or wiring a new composite frame.

At two pauses between ``ComSim.Execute`` calls, the probe changes the
voltage-dependent active-current limit from its baseline to 0.40 pu and then
0.30 pu.  A valid online mailbox must satisfy all three conditions:

1. the REEC lookup output and physical park P consume both new tables;
2. the calculation continues without ``ComInc``; and
3. result samples recorded before each write remain unchanged.

Temporary event/result folders isolate the probe from the replay pool and
normal monitor catalogue.  The original matrix and command pointers are
restored in ``finally``.

Usage::

    python pf\probes\probe_live_intmat.py
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


RMS_STUDY_CASE = "02_RMS_CoSim"
FIRST_WRITE_S = 0.5
SECOND_WRITE_S = 1.5
FINAL_STOP_S = 2.5
FIRST_LIMIT_PU = 0.40
SECOND_LIMIT_PU = 0.30

TEMP_EVENT_FOLDER = "qOFO Live IntMat Probe Events"
TEMP_RESULT = "qOFO Live IntMat Probe Results"


def _copy_matrix(matrix) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _set_matrix(matrix_obj, values: Sequence[Sequence[float]]) -> None:
    rows = len(values)
    columns = len(values[0]) if rows else 0
    if rows < 1 or columns < 1:
        raise ValueError("IntMat test matrix must be non-empty")
    if any(len(row) != columns for row in values):
        raise ValueError("IntMat test matrix must be rectangular")
    if (
        int(matrix_obj.NRow()) != rows
        or int(matrix_obj.NCol()) != columns
    ):
        matrix_obj.Resize(rows, columns)
    for row, values_row in enumerate(values, start=1):
        for column, value in enumerate(values_row, start=1):
            matrix_obj.Set(row, column, float(value))


def _limited_table(original, limit_pu: float) -> list[list[float]]:
    if not 0.0 < limit_pu < 1.0:
        raise ValueError("probe limit must be in (0, 1) pu")
    return [[float(row[0]), float(limit_pu)] for row in original]


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


def _select_target(app) -> Tuple[Any, Any, Any]:
    """Choose the highest-loaded in-service REEC park with a vdlp IntMat."""

    candidates = []
    for comp in get_all(app, "ElmComp"):
        reec = next(
            (
                obj for obj in comp.GetContents()
                if obj.GetClassName() == "ElmDsl"
                and "REEC" in obj.loc_name
            ),
            None,
        )
        if reec is None:
            continue
        park = next(
            (
                obj for obj in comp.GetAttribute("pelm")
                if obj is not None and obj.GetClassName() == "ElmGenstat"
            ),
            None,
        )
        table = next(
            (
                obj for obj in reec.GetContents("*.IntMat")
                if obj.loc_name == "vdlp"
            ),
            None,
        )
        if park is None or table is None:
            continue
        if int(park.GetAttribute("outserv")) != 0:
            continue
        rating = float(park.GetAttribute("sgn"))
        p_set = float(park.GetAttribute("pgini"))
        if rating > 0.0 and p_set > 0.0:
            candidates.append((p_set / rating, reec, park, table))
    if not candidates:
        raise PFSessionError("no in-service REEC park with vdlp.IntMat found")
    _loading, reec, park, table = max(candidates, key=lambda row: row[0])
    return reec, park, table


def main() -> int:
    app = None
    inc = None
    original_events = None
    original_result = None
    temp_events = None
    temp_result = None
    table = None
    original_matrix = None
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
        reec, park, table = _select_target(app)
        original_matrix = _copy_matrix(table.GetAttribute("M"))
        first_matrix = _limited_table(original_matrix, FIRST_LIMIT_PU)
        second_matrix = _limited_table(original_matrix, SECOND_LIMIT_PU)
        ctx.set_monitors([
            (reec, "s:vdlp", "vdlp"),
            (reec, "s:Pord", "pord"),
            (park, "m:P:bus1", "park_p"),
        ])

        print("PowerFactory active-RMS IntMat mailbox probe")
        print(f"  REEC:  {reec.GetFullName()}")
        print(f"  park:  {park.GetFullName()}")
        print(f"  table: {table.GetFullName()}")
        print(f"  baseline matrix: {original_matrix}")
        print(
            f"  replacement limits: {FIRST_LIMIT_PU:.3f}, "
            f"{SECOND_LIMIT_PU:.3f} pu"
        )
        print(
            f"  isolated event folder count: "
            f"{len(list(temp_events.GetContents()))}"
        )

        ctx.initialise()
        ctx.simulate(FIRST_WRITE_S)
        t0_v, vdlp_hist0 = ctx.read(reec, "s:vdlp", stride=1)
        _t0_o, pord_hist0 = ctx.read(reec, "s:Pord", stride=1)
        _t0_p, park_hist0 = ctx.read(park, "m:P:bus1", stride=1)
        live0 = (
            float(reec.GetAttribute("s:vdlp")),
            float(reec.GetAttribute("s:Pord")),
            float(park.GetAttribute("m:P:bus1")),
        )

        _set_matrix(table, first_matrix)
        first_database_matrix = _copy_matrix(table.GetAttribute("M"))
        ctx.simulate(SECOND_WRITE_S)
        t1_v, vdlp_hist1 = ctx.read(reec, "s:vdlp", stride=1)
        _t1_o, pord_hist1 = ctx.read(reec, "s:Pord", stride=1)
        _t1_p, park_hist1 = ctx.read(park, "m:P:bus1", stride=1)
        live1 = (
            float(reec.GetAttribute("s:vdlp")),
            float(reec.GetAttribute("s:Pord")),
            float(park.GetAttribute("m:P:bus1")),
        )

        _set_matrix(table, second_matrix)
        second_database_matrix = _copy_matrix(table.GetAttribute("M"))
        ctx.simulate(FINAL_STOP_S)
        tv, vdlp_hist2 = ctx.read(reec, "s:vdlp", stride=1)
        _to, pord_hist2 = ctx.read(reec, "s:Pord", stride=1)
        _tp, park_hist2 = ctx.read(park, "m:P:bus1", stride=1)
        live2 = (
            float(reec.GetAttribute("s:vdlp")),
            float(reec.GetAttribute("s:Pord")),
            float(park.GetAttribute("m:P:bus1")),
        )

        prefix_change_after_first = max(
            _max_prefix_delta(vdlp_hist0, vdlp_hist1),
            _max_prefix_delta(pord_hist0, pord_hist1),
            _max_prefix_delta(park_hist0, park_hist1),
        )
        prefix_change_after_second = max(
            _max_prefix_delta(vdlp_hist1, vdlp_hist2),
            _max_prefix_delta(pord_hist1, pord_hist2),
            _max_prefix_delta(park_hist1, park_hist2),
        )
        time_continuous = (
            len(tv) > 0
            and abs(float(tv[0])) <= 0.02
            and abs(float(tv[-1]) - FINAL_STOP_S) <= 0.02
            and all(float(b) >= float(a) for a, b in zip(tv, tv[1:]))
        )
        prefix_stable = (
            prefix_change_after_first <= 1.0e-9
            and prefix_change_after_second <= 1.0e-9
        )
        database_writes = (
            first_database_matrix == first_matrix
            and second_database_matrix == second_matrix
        )
        lookup_consumed = (
            math.isclose(live1[0], FIRST_LIMIT_PU, abs_tol=2.0e-3)
            and math.isclose(live2[0], SECOND_LIMIT_PU, abs_tol=2.0e-3)
        )
        pord_responded = (
            live1[1] < live0[1] - 0.02
            and live2[1] < live1[1] - 0.02
        )
        park_responded = (
            live1[2] < live0[2] - 1.0
            and live2[2] < live1[2] - 1.0
        )
        checks = [
            (
                "matrix writes accepted",
                database_writes,
                f"M -> {first_database_matrix} -> {second_database_matrix}",
            ),
            (
                "same active RMS timeline continued",
                time_continuous,
                f"rows={len(tv)}, time=[{tv[0]:.3f}, {tv[-1]:.3f}]",
            ),
            (
                "previous result prefix remained unchanged",
                prefix_stable,
                f"max prefix deltas={prefix_change_after_first:.6g}, "
                f"{prefix_change_after_second:.6g}",
            ),
            (
                "DSL lookup consumed both matrices",
                lookup_consumed,
                f"vdlp={live0[0]:.6f} -> {live1[0]:.6f} -> "
                f"{live2[0]:.6f}",
            ),
            (
                "REEC Pord responded twice",
                pord_responded,
                f"Pord={live0[1]:.6f} -> {live1[1]:.6f} -> "
                f"{live2[1]:.6f} pu",
            ),
            (
                "physical park P responded twice",
                park_responded,
                f"P={live0[2]:.6f} -> {live1[2]:.6f} -> "
                f"{live2[2]:.6f} MW",
            ),
            (
                "no simulation events used",
                len(list(temp_events.GetContents())) == 0,
                f"event count={len(list(temp_events.GetContents()))}",
            ),
        ]
        passed = True
        for label, ok, evidence in checks:
            passed &= bool(ok)
            print(f"  {'PASS' if ok else 'FAIL'} {label}: {evidence}")
        print("LIVE_INTMAT_PROBE=" + ("PASS" if passed else "FAIL"))
        return 0 if passed else 2
    except Exception as exc:  # noqa: BLE001
        print(f"LIVE_INTMAT_PROBE=ERROR {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 3
    finally:
        if app is not None:
            try:
                app.ResetCalculation()
            except Exception:
                pass
        if table is not None and original_matrix is not None:
            try:
                _set_matrix(table, original_matrix)
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
