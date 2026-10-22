r"""Probe event-free RMS load-profile playback through ``ElmFile``.

The probe builds the smallest model documented in the installed PowerFactory
2025 SP4 technical references:

``ElmFile.y1/y2 -> ElmLod.Pext/Qext``.

It runs one existing three-phase load for 2.5 s against a normalized,
piecewise-constant two-channel file.  A temporary composite frame, composite
model, measurement-file object, event folder, and result file isolate the
test.  All temporary PowerFactory objects and study-case pointer changes are
removed in ``finally``.

Run with the project's PowerFactory-enabled environment::

    python pf\probes\probe_rms_elmfile_profile.py
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import traceback
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pf.screening import ScreeningContext  # noqa: E402
from pf.session import (  # noqa: E402
    DEFAULT_PROJECT_PATH,
    PFSessionError,
    connect,
    get_all,
)


RMS_STUDY_CASE = "02_RMS_CoSim"
FINAL_STOP_S = 2.5
PROFILE_FILE = (
    Path(__file__).resolve().parent / "probe_data" / "live_elmfile_profile.txt"
)

TEMP_EVENT_FOLDER = "qOFO ElmFile Profile Probe Events"
TEMP_RESULT = "qOFO ElmFile Profile Probe Results"
TEMP_FRAME = "qOFO ElmFile Profile Probe Frame"
TEMP_COMPOSITE = "qOFO ElmFile Profile Probe Composite"
TEMP_SOURCE = "qOFO ElmFile Profile Probe Source"


def _delete_named(parent, class_name: str, name: str) -> None:
    for obj in list(parent.GetContents(f"{name}.{class_name}")):
        obj.Delete()


def _sample(
    time_s: Sequence[float],
    values: Sequence[float],
    at_s: float,
) -> float:
    if not time_s or len(time_s) != len(values):
        raise PFSessionError("invalid probe result series")
    index = min(
        range(len(time_s)),
        key=lambda i: abs(float(time_s[i]) - float(at_s)),
    )
    if abs(float(time_s[index]) - float(at_s)) > 0.02:
        raise PFSessionError(
            f"no result row near t={at_s}: nearest={time_s[index]}"
        )
    return float(values[index])


def _select_load(app) -> Any:
    """Choose a substantial in-service load not owned by a composite model."""

    composite_elements = {
        element.GetFullName()
        for composite in get_all(app, "ElmComp")
        for element in composite.GetAttribute("pelm")
        if element is not None
    }
    candidates = []
    for load in get_all(app, "ElmLod"):
        if int(load.GetAttribute("outserv")):
            continue
        if load.GetFullName() in composite_elements:
            continue
        p_mw = float(load.GetAttribute("plini"))
        q_mvar = float(load.GetAttribute("qlini"))
        if p_mw > 1.0 and abs(q_mvar) > 0.5:
            candidates.append((p_mw + abs(q_mvar), load))
    if not candidates:
        raise PFSessionError(
            "no substantial in-service ElmLod outside a composite model found"
        )
    return max(candidates, key=lambda item: item[0])[1]


def _create_frame(model_folder):
    frame = model_folder.CreateObject("BlkDef", TEMP_FRAME)
    if frame is None:
        raise PFSessionError("failed to create temporary composite frame")

    load_slot = frame.CreateObject("BlkSlot", "Load")
    source_slot = frame.CreateObject("BlkSlot", "File")
    if load_slot is None or source_slot is None:
        raise PFSessionError("failed to create temporary frame slots")

    # Slot input/output vectors refer to the frame's signal names by position.
    load_slot.SetAttribute("sInput", ["Pext,Qext"])
    source_slot.SetAttribute("sOutput", ["y1,y2"])
    for index, name in enumerate(("Pext", "Qext")):
        signal = frame.CreateObject("BlkSig", name)
        if signal is None:
            raise PFSessionError(f"failed to create frame signal {name}")
        # A BlkSig is a routed line, not merely a name.  Its endpoint object,
        # variable index, and connection type must all be defined.
        signal.SetAttribute("pnodfrom", source_slot)
        signal.SetAttribute("pnodto", load_slot)
        signal.SetAttribute("inodfrom", index)
        signal.SetAttribute("inodto", index)
        signal.SetAttribute("iconfrom", 2)  # Output
        signal.SetAttribute("iconto", 1)  # Input
    return frame, load_slot, source_slot


def _configure_source(source, p0_mw: float, q0_mvar: float) -> None:
    if not PROFILE_FILE.is_file():
        raise PFSessionError(f"probe profile file missing: {PROFILE_FILE}")
    # The default iopt_imp=1 is the plain Measurement File mode in PF 2025 SP4.
    source.SetAttribute("iopt_imp", 1)
    source.SetAttribute("f_name", str(PROFILE_FILE))
    source.SetAttribute("icol", list(range(1, 25)))
    source.SetAttribute(
        "afac",
        [float(p0_mw), float(q0_mvar)] + [1.0] * 22,
    )
    source.SetAttribute("bfac", [0.0] * 24)
    source.SetAttribute("tini", 0.0)
    source.SetAttribute("approx", 0)  # piecewise constant, no interpolation


def _close(actual: float, expected: float) -> bool:
    return math.isclose(
        float(actual),
        float(expected),
        rel_tol=2.0e-4,
        abs_tol=2.0e-3,
    )


def main() -> int:
    app = None
    inc = None
    original_events = None
    original_result = None
    temp_events = None
    temp_result = None
    frame = None
    composite = None
    source = None
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

        load = _select_load(app)
        network = load.GetParent()
        model_folder = next(
            definition.GetParent()
            for definition in get_all(app, "BlkDef")
            if definition.loc_name == "Frame WECC PV qOFO"
        )

        # Remove only stale objects owned by this probe, in dependency order.
        _delete_named(network, "ElmComp", TEMP_COMPOSITE)
        _delete_named(network, "ElmFile", TEMP_SOURCE)
        _delete_named(model_folder, "BlkDef", TEMP_FRAME)

        p0_mw = float(load.GetAttribute("plini"))
        q0_mvar = float(load.GetAttribute("qlini"))
        frame, load_slot, source_slot = _create_frame(model_folder)
        source = network.CreateObject("ElmFile", TEMP_SOURCE)
        composite = network.CreateObject("ElmComp", TEMP_COMPOSITE)
        if source is None or composite is None:
            raise PFSessionError("failed to create temporary profile objects")
        _configure_source(source, p0_mw, q0_mvar)
        composite.SetAttribute("typ_id", frame)
        composite.SetAttribute("pblk", [load_slot, source_slot])
        composite.SetAttribute("pelm", [load, source])

        ctx = ScreeningContext(app, verbose=False)
        ctx.res = temp_result
        ctx.set_monitors([
            (source, "s:y1", "file_p"),
            (source, "s:y2", "file_q"),
            (load, "m:P:bus1", "load_p"),
            (load, "m:Q:bus1", "load_q"),
        ])

        print("PowerFactory RMS ElmFile profile probe")
        print(f"  load:   {load.GetFullName()}")
        print(f"  P0/Q0:  {p0_mw:.6f} MW / {q0_mvar:.6f} Mvar")
        print(f"  file:   {PROFILE_FILE}")
        print(f"  frame:  {frame.GetFullName()}")
        print(
            "  mapping: ElmFile.y1/y2 -> ElmLod.Pext/Qext, "
            "piecewise constant"
        )
        print(
            f"  isolated event folder count: "
            f"{len(list(temp_events.GetContents()))}"
        )

        ctx.initialise()
        ctx.simulate(FINAL_STOP_S)
        time_s, file_p = ctx.read(source, "s:y1", stride=1)
        _time_q, file_q = ctx.read(source, "s:y2", stride=1)
        _time_lp, load_p = ctx.read(load, "m:P:bus1", stride=1)
        _time_lq, load_q = ctx.read(load, "m:Q:bus1", stride=1)

        factors = ((0.25, 1.0, 1.0), (1.0, 1.2, 0.8), (2.0, 0.8, 1.2))
        observed = []
        source_ok = True
        load_factors_ok = True
        baseline_load_p = None
        baseline_load_q = None
        for at_s, p_factor, q_factor in factors:
            values = (
                _sample(time_s, file_p, at_s),
                _sample(time_s, file_q, at_s),
                _sample(time_s, load_p, at_s),
                _sample(time_s, load_q, at_s),
            )
            expected = (p0_mw * p_factor, q0_mvar * q_factor)
            observed.append((at_s, values, expected))
            source_ok &= _close(values[0], expected[0])
            source_ok &= _close(values[1], expected[1])
            if baseline_load_p is None:
                baseline_load_p = values[2]
                baseline_load_q = values[3]
            else:
                actual_p_factor = values[2] / baseline_load_p
                actual_q_factor = values[3] / baseline_load_q
                load_factors_ok &= math.isclose(
                    actual_p_factor, p_factor, abs_tol=3.0e-3
                )
                load_factors_ok &= math.isclose(
                    actual_q_factor, q_factor, abs_tol=3.0e-3
                )

        time_continuous = (
            len(time_s) > 0
            and abs(float(time_s[0])) <= 0.02
            and abs(float(time_s[-1]) - FINAL_STOP_S) <= 0.02
            and all(
                float(after) >= float(before)
                for before, after in zip(time_s, time_s[1:])
            )
        )
        events_empty = len(list(temp_events.GetContents())) == 0

        evidence = "; ".join(
            f"t={at_s:g}: file=({values[0]:.4f},{values[1]:.4f}), "
            f"load=({values[2]:.4f},{values[3]:.4f}), "
            f"file-expected=({expected[0]:.4f},{expected[1]:.4f}), "
            f"load-factor=("
            f"{values[2] / baseline_load_p:.4f},"
            f"{values[3] / baseline_load_q:.4f})"
            for at_s, values, expected in observed
        )
        checks = [
            (
                "same active RMS timeline completed",
                time_continuous,
                f"rows={len(time_s)}, time=[{time_s[0]:.3f}, "
                f"{time_s[-1]:.3f}]",
            ),
            ("ElmFile emitted all three profile levels", source_ok, evidence),
            (
                "physical load followed profile factors",
                load_factors_ok,
                evidence,
            ),
            (
                "no simulation events used",
                events_empty,
                f"event count={len(list(temp_events.GetContents()))}",
            ),
        ]
        passed = True
        for label, ok, detail in checks:
            passed &= bool(ok)
            print(f"  {'PASS' if ok else 'FAIL'} {label}: {detail}")
        print("RMS_ELMFILE_PROFILE_PROBE=" + ("PASS" if passed else "FAIL"))
        return 0 if passed else 2
    except Exception as exc:  # noqa: BLE001
        print(
            f"RMS_ELMFILE_PROFILE_PROBE=ERROR "
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
        # Delete references before referenced objects.
        for obj in (
            composite,
            source,
            frame,
            temp_result,
            temp_events,
        ):
            if obj is not None:
                try:
                    obj.Delete()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
