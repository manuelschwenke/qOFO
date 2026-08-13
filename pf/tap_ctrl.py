"""
pf/tap_ctrl.py
==============
DSL-driven OLTC tap control for RMS simulation.

Why this exists
---------------
``EvtTap`` proved unreliable for tap dispatch as we were using it.  Measured
2026-07-30: the first tap of a calculation lands on time, every later one lands
~60 s late or is lost, independent of event offset, pooling, pre-ComInc
existence, admission barriers and event-folder contents.

**Correction (2026-07-31):** the cause was later found and is NOT specific to
``EvtTap`` -- PowerFactory applies simulation-event times **modulo a 60 s
window** once a calculation is running, so an absolute time ``te`` fires at
``te mod 60``.  Scheduling ``te - 60*floor(t_clock/60)`` makes ``EvtTap`` land
on time indefinitely (validated over four windows).  See
``docs/daily_log/07_2026/2026-07-31_rms_tap_control_gate_e_result.md``.

This layer is kept anyway, and remains the preferred mechanism: it is the
vendor-documented path, it provides a real mechanical time constant rather than
an instantaneous jump, and it does not depend on an undocumented quirk.
DIgSILENT
``TechRef_2-W-Transformer_3Phase.pdf`` S5 (RMS-Simulation):

    "The model used by the RMS simulation is identical to the load flow model.
    However, tap controller definitions are not considered.  For the simulation
    of tap controllers, a separate dynamic model must be defined that can be
    interfaced with the transformer using the input variable ``nntapin``
    (tap-input)."

So the tap becomes a DSL output.  The commanded position is then just another
DSL parameter, dispatched with the *pooled* ``EvtParam`` machinery that is
already validated for ``REEC_D.Qext`` and the AVR ``usetp`` -- and which,
unlike ``EvtTap``, lands on time every interval.

Verified behaviour (MT_g0_t0, Tmech = 5 s, pooled EvtParam):

    commanded -1 @ 25 s  ->  nntapin -0.28 @27, -0.95 @40, -1.000 @64
    commanded -2 @ 65 s  ->  nntapin -1.28 @67, -1.95 @80, -2.000 @104
    commanded  0 @ 105 s ->  nntapin -1.45 @107, -0.11 @120, -0.002 @139

i.e. on time, both directions, repeatedly.

Manual step (once per project)
------------------------------
The frame's *connection topology* lives in its ``IntGrfnet`` graphic, which
cannot be authored through the API -- the same reason
``wecc_apply._project_blockdef`` refuses to create frames.  ``ensure_frame``
therefore creates the slots and the signal but the **wire must be drawn once in
the PowerFactory GUI**:

    1. open ``Frame Tap Control qOFO`` (User Defined Models),
    2. connect the ``Tap Control`` slot output ``nntapin`` to the
       ``Transformer`` slot input ``nntapin``,
    3. save.

Until that wire exists the DSL runs and its output moves, but the transformer
never sees it (symptom: ``s:nntapin`` follows the command while the bus voltage
and the ratio do not move at all).

Author: Manuel Schwenke / Claude Code (2026-07-30)
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from pf.session import PFSessionError, get_all

DSL_NAME = "TAPCTRL_qOFO"
#: Two frames, one per transformer class -- they differ ONLY in the signal
#: name the transformer slot declares as its input.
FRAME_NAME_2W = "Frame Tap Control qOFO"
FRAME_NAME_3W = "Frame Tap Control 3W qOFO"
FRAME_NAME = FRAME_NAME_2W          # backwards-compatible alias

#: Tap-input signal per class.  ``ElmTr2`` exposes a single ``nntapin``;
#: ``ElmTr3`` exposes three -- ``nntapin_h`` / ``nntapin_m`` / ``nntapin_l``,
#: one per winding (TechRef_3-W-Transformer, Table 5.1; the underscores are
#: legible in the 1-Phase edition where the 3-Phase table loses them).  Our
#: couplers carry the tap changer on the HV winding, so ``_h`` is the one that
#: acts -- ``_m`` and ``_l`` are accepted and silently do nothing.
TAP_INPUT_SIGNAL = {"ElmTr2": "nntapin", "ElmTr3": "nntapin_h"}
COMP_PREFIX = "TAPC_"
DSL_ELEMENT_NAME = "TAPCTRL"
SLOT_TRAFO = "Transformer"
SLOT_CTRL = "Tap Control"

#: Default mechanical time constant [s].  The tap slides to the commanded
#: position with this lag rather than stepping, which is both more physical
#: than an instantaneous jump and what gives PF a differentiable state to
#: initialise.  5 s matches the ``TAP_MECH_DELAY_S`` the event path used.
TMECH_DEFAULT_S = 5.0

DSL_OUTPUT = ["nntapin"]
DSL_PARAMS = ["ntapcmd,Tmech"]          # one comma-joined string, as PF wants
DSL_STATES = ["x1"]
DSL_EQUATIONS = [
    "inc(x1)=ntapcmd",
    # The OUTPUT needs its own inc() too.  Without this PF fails ComInc with
    # "Output 'nntapin' not initialised" even though x1 is initialised and
    # nntapin=x1 (verified 2026-07-30).
    "inc(nntapin)=ntapcmd",
    "x1.=(ntapcmd-x1)/Tmech",
    "nntapin=x1",
]
#: Index order of the ``params`` vector.
DSL_PARAM_ORDER = ("ntapcmd", "Tmech")


def tapctrl_of(app, trf):
    """The ``TAPCTRL`` DSL driving ``trf``, or ``None`` if it has no controller.

    Needed by anything that wants to MOVE a tap in RMS.  ``EvtTap`` is not the
    supported path on this model -- the DSL drives ``nntapin`` and simply
    holds the tap wherever ``ntapcmd`` points, so an ``EvtTap`` is overwritten
    on the next solver step and the tap never moves (2026-07-30, and again in
    the timescale battery of 2026-08-07 where every tap case measured
    T_s = 0.00 s because nothing actuated).  Command ``ntapcmd`` instead; the
    5 s mechanical travel is already in the block as ``Tmech``.
    """
    from pf.session import get_all

    want = getattr(trf, "loc_name", None)
    for comp in get_all(app, "ElmComp"):
        if not comp.loc_name.startswith(COMP_PREFIX):
            continue
        pelm = comp.GetAttribute("pelm") or []
        dsl = next((o for o in pelm if o is not None
                    and o.GetClassName() == "ElmDsl"), None)
        target = next((o for o in pelm if o is not None
                       and o.GetClassName() in ("ElmTr2", "ElmTr3")), None)
        if dsl is not None and target is not None \
                and target.loc_name == want:
            return dsl
    return None


def ensure_tapctrl_blockdef(app):
    """Create or update ``TAPCTRL_qOFO.BlkDef``.

    Rewriting the type invalidates existing ``ElmDsl`` instances of it (an
    element that predates its final BlkDef keeps a dead parameter table whose
    values read zero while writes appear to succeed), so ``apply_to_trafo``
    always recreates the element -- same rule as the QVPRE rollout.
    """
    blk = app.GetProjectFolder("blk")
    hits = list(blk.GetContents(f"{DSL_NAME}.BlkDef"))
    if len(hits) > 1:
        raise PFSessionError(
            f"{DSL_NAME!r}: {len(hits)} BlkDefs in 'User Defined Models'; "
            f"remove the duplicates before rolling out.")
    bd = hits[0] if hits else blk.CreateObject("BlkDef", DSL_NAME)
    bd.SetAttribute("sOutput", DSL_OUTPUT)
    bd.SetAttribute("sParams", DSL_PARAMS)
    bd.SetAttribute("sStates", DSL_STATES)
    bd.SetAttribute("sAddEquat", DSL_EQUATIONS)
    return bd


def ensure_frame(app, *, create_slots: bool = True):
    """Return the tap-control frame, creating its slots/signal if absent.

    Does NOT create the wiring -- see the module docstring.  Raises if the
    frame carries no ``IntGrfnet``, because that is the state in which the
    whole layer silently does nothing.
    """
    blk = app.GetProjectFolder("blk")
    hits = list(blk.GetContents(f"{FRAME_NAME}.BlkDef"))
    fr = hits[0] if hits else None
    if fr is None:
        if not create_slots:
            raise PFSessionError(
                f"{FRAME_NAME!r} not found in 'User Defined Models'.")
        fr = blk.CreateObject("BlkDef", FRAME_NAME)
    have = {c.GetClassName() for c in fr.GetContents()}
    if create_slots and "BlkSlot" not in have:
        s_tr = fr.CreateObject("BlkSlot", SLOT_TRAFO)
        s_tr.SetAttribute("sInput", ["nntapin"])
        s_tr.SetAttribute("sOutput", [])
        s_ct = fr.CreateObject("BlkSlot", SLOT_CTRL)
        s_ct.SetAttribute("sInput", [])
        s_ct.SetAttribute("sOutput", ["nntapin"])
        fr.CreateObject("BlkSig", "nntapin")
        # A frame must declare its internal signals or ComInc rejects it.
        fr.SetAttribute("sIntern", ["nntapin"])
    return fr


def frame_is_wired(fr) -> bool:
    """Whether the frame carries a graphic, i.e. the slots are connected.

    The connection topology lives in ``IntGrfnet``; slots and ``BlkSig``
    objects alone leave the signal floating and the transformer inert.
    """
    return any(c.GetClassName() == "IntGrfnet" for c in fr.GetContents())


def frame_for(app, trafo):
    """The frame matching this transformer's class."""
    cls = trafo.GetClassName()
    name = FRAME_NAME_3W if cls == "ElmTr3" else FRAME_NAME_2W
    blk = app.GetProjectFolder("blk")
    hits = list(blk.GetContents(f"{name}.BlkDef"))
    if len(hits) != 1:
        raise PFSessionError(
            f"{name!r} ({cls} tap-control frame) not found in 'User Defined "
            f"Models' ({len(hits)} matches). It is hand-authored -- the API "
            f"cannot create the IntGrfnet that carries its wiring.")
    fr = hits[0]
    if not frame_is_wired(fr):
        raise PFSessionError(
            f"{name!r} has no IntGrfnet: its slots are not connected, so the "
            f"DSL output never reaches the transformer. This fails SILENTLY "
            f"-- ComInc passes and the tap simply never moves.")
    return fr


def set_tap_params(ctrl, *, ntapcmd: float,
                   tmech_s: float = TMECH_DEFAULT_S) -> None:
    """Write the DSL parameter vector element-wise (a whole-vector write fails)."""
    n = len(DSL_PARAM_ORDER)
    try:
        ctrl.SetAttributeLength("params", n)
    except Exception:                                  # noqa: BLE001
        pass
    for i, value in enumerate((float(ntapcmd), float(tmech_s))):
        ctrl.SetAttribute(f"params:{i}", float(value))


def apply_to_trafo(app, trafo, *, frame, blockdef,
                   tmech_s: float = TMECH_DEFAULT_S):
    """Build (or rebuild) the tap-control composite for one transformer.

    ``ntapcmd`` is seeded with the transformer's ACTUAL load-flow tap so that
    ``inc(x1)=ntapcmd`` initialises the block where the plant already sits,
    instead of yanking the tap to zero at t=0.
    """
    name = f"{COMP_PREFIX}{trafo.loc_name}"
    for c in get_all(app, "ElmComp"):
        if c.loc_name == name:
            c.Delete()
    grid = trafo.GetParent()
    comp = grid.CreateObject("ElmComp", name)
    comp.SetAttribute("typ_id", frame)
    ctrl = comp.CreateObject("ElmDsl", DSL_ELEMENT_NAME)
    ctrl.SetAttribute("typ_id", blockdef)

    tap_attr = "nntap" if trafo.GetClassName() == "ElmTr2" else "n3tap_h"
    set_tap_params(ctrl, ntapcmd=float(trafo.GetAttribute(tap_attr)),
                   tmech_s=tmech_s)
    # pelm is ordered by the frame's slot order (pblk), NOT the visual order.
    pblk = comp.GetAttribute("pblk") or []
    order = [getattr(b, "loc_name", "") for b in pblk]
    if order and order[0] != SLOT_TRAFO:
        comp.SetAttribute("pelm", [ctrl, trafo])
    else:
        comp.SetAttribute("pelm", [trafo, ctrl])
    return comp, ctrl


def controllable_transformers(app, *, prefixes: Iterable[str] = ("MT_", "NT_"),
                              include_couplers: bool = True) -> List:
    """Every transformer the OFO stack can tap."""
    out = [t for t in get_all(app, "ElmTr2")
           if any(t.loc_name.startswith(p) for p in prefixes)]
    if include_couplers:
        out += [t for t in get_all(app, "ElmTr3")
                if t.loc_name.startswith("NC3W_")]
    return sorted(out, key=lambda o: o.loc_name)
