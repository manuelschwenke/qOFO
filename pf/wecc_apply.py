"""
pf/wecc_apply.py
===============
Attach a WECC RMS converter model to every DER static generator so its
reactive power becomes controllable during an RMS simulation (the OFO write
handle ``REEC_D.Qext``), replacing the load-flow-only ``qsetp``.

Recipe (verified 2026-07-20 on WP_TSO_s0_b18, ComInc green, Qext step drove
Q by -110 Mvar):

1. Copy the fully-parameterised library template composite
   ``Lib/Templ/TemplPv/WECC Large-scale PV Plant 110MVA 60Hz`` ->
   ``PV Plant/WECC Large-scale PV Plant`` (REGC_C + REEC_D + Protection +
   Weak-Grid + StaPqmea/StaVmea/StaImea) into the park's grid.
2. Re-point the Generator slot (pelm[0]) from the template's ElmPvsys to our
   ElmGenstat; re-point the three measurement devices to the park's terminal
   / cubicle.
3. Put REEC_D in **reactive-power control** (``PfFlag=0``, ``QFlag=0``;
   ``VFlag=0`` is immaterial in this mode) so
   ``Qext`` (pu of the park's rated S) is the reactive reference the OFO
   writes.

The composite is created in the park's own ElmNet so it is calculation-
relevant whenever that grid is active.  Idempotent: an existing
``WECC_<park>`` composite is deleted and rebuilt.

Run on the PF machine::

    python pf\\wecc_apply.py                 # all DER
    python pf\\wecc_apply.py --only WP_TSO   # name-prefix filter
    python pf\\wecc_apply.py --verify        # ComInc after applying

Author: Manuel Schwenke / Claude Code (2026-07-20)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.der_qv_local_loop import _qv_capability  # noqa: E402
from pf.pf_parity import PARITY_LDF_SETTINGS  # noqa: E402
from pf.session import (  # noqa: E402
    DEFAULT_PROJECT_PATH,
    PFSessionError,
    connect,
    deactivate_variations_except,
    get_all,
    set_variation_active,
)

RMS_STUDY_CASE = "02_RMS_CoSim"

#: DER static-generator name prefixes (naming.py convention).
DER_PREFIXES = ("WP_TSO_", "DER_", "WPC_")

#: Project-local frame carrying the hand-drawn Voltage-Measurement ``u`` ->
#: Plant-Control ``u`` signal.  Wires are graphic-authoritative in PF and
#: cannot be created headlessly, so this object is *found*, never recreated.
LOCAL_FRAME_NAME = "Frame WECC PV qOFO"

#: DSL block implementing the re-anchored Q(V) law (see
#: docs/daily_log/07_2026/2026-07-20_rms_phase6_qvpre_dsl_layer.md).
QVPRE_BLOCKDEF_NAME = "QVPRE_qOFO"
QVPRE_ELEMENT_NAME = "QVPRE"

#: Measurement-filter time constant [s] breaking the algebraic V->Q loop.
#: Three orders below the 20 s dispatch window, so the dispatch-interval
#: equilibrium is the exact static law.
QV_FILTER_TF_S = 0.02

#: Fallbacks when no snapshot is supplied (all 44 parks are uniform today).
QV_SLOPE_PU_DEFAULT = 0.06
QV_DEADBAND_PU_DEFAULT = 0.01


def _template_composite(app):
    gl = app.GetGlobalLibrary()
    def child(f, name):
        for o in f.GetContents():
            if o.loc_name == name:
                return o
        return None
    tpl = child(child(child(gl, "Templ"), "TemplPv"),
                "WECC Large-scale PV Plant 110MVA 60Hz")
    if tpl is None:
        raise PFSessionError("WECC PV-plant template not found in the library")
    return child(child(tpl, "PV Plant"), "WECC Large-scale PV Plant")


def _activate_full(app) -> None:
    deactivate_variations_except(app, keep=None)
    set_variation_active(app, "wind_replace", True)
    set_variation_active(app, "full", True)
    for g in app.GetProjectFolder("netdat").GetContents("*.ElmNet"):
        if g.loc_name.startswith("DSO_") and not g.IsCalcRelevant():
            g.Activate()


# QVPRE block definition, versioned here so the DSL law is reproducible.
# Mirrors controller/der_qv_local_loop.py::QVLocalLoop exactly:
#   q_target = q_set - R*db(V - V_anchor);  Q = clip(q_target, q_min, q_max)
# in pu of the park rating (R -> Kdroop = 1/slope_pu).  ``x1`` is the
# Tf-filtered terminal voltage.  Deadband and both limits are expressed
# arithmetically because DSL has no min/max operator:
#   max(a,b) = (a+b+|a-b|)/2      min(a,b) = (a+b-|a-b|)/2
QVPRE_INPUT = ["u"]
QVPRE_OUTPUT = ["Qext"]
# ONE comma-joined string per list attribute -- PF rejects separate names.
QVPRE_PARAMS = ["qset,Vanchor,Kdroop,db,Tf,qmin,qmax"]
QVPRE_STATES = ["x1"]
QVPRE_INTERN = ["veff,qcorr,qraw,qlo"]
QVPRE_EQUATIONS = "\n".join((
    "inc(x1)=u",
    "x1.=(u-x1)/Tf",
    "veff = x1-Vanchor",
    "qcorr = (veff-db+abs(veff-db))/2+(veff+db-abs(veff+db))/2",
    "qraw = qset-Kdroop*qcorr",
    "qlo = (qraw+qmin+abs(qraw-qmin))/2",
    "Qext = (qlo+qmax-abs(qlo-qmax))/2",
))
#: Parameter order in the ``params`` vector; index == position.
QVPRE_PARAM_ORDER = ("qset", "Vanchor", "Kdroop", "db", "Tf", "qmin", "qmax")


def ensure_qvpre_blockdef(app):
    """Create or update ``QVPRE_qOFO.BlkDef`` to the definition above.

    Rewriting the type invalidates every existing ``ElmDsl`` of that type:
    a block that predates its final BlkDef keeps a dead parameter table
    whose runtime values read zero while ``params`` writes still appear to
    succeed.  ``apply_to_park`` therefore always recreates the element.
    """
    blk = app.GetProjectFolder("blk")
    hits = list(blk.GetContents(f"{QVPRE_BLOCKDEF_NAME}.BlkDef"))
    if len(hits) > 1:
        raise PFSessionError(
            f"{QVPRE_BLOCKDEF_NAME!r}: {len(hits)} BlkDefs in 'User Defined "
            f"Models'; remove the duplicates before rolling out.")
    bd = hits[0] if hits else blk.CreateObject("BlkDef", QVPRE_BLOCKDEF_NAME)
    bd.SetAttribute("sInput", QVPRE_INPUT)
    bd.SetAttribute("sOutput", QVPRE_OUTPUT)
    bd.SetAttribute("sParams", QVPRE_PARAMS)
    bd.SetAttribute("sStates", QVPRE_STATES)
    bd.SetAttribute("sIntern", QVPRE_INTERN)
    bd.SetAttribute("sAddEquat", QVPRE_EQUATIONS.split("\n"))
    return bd


def _project_blockdef(app, name: str, purpose: str):
    """Find a project-local BlkDef; never create it (see LOCAL_FRAME_NAME)."""
    blk = app.GetProjectFolder("blk")
    hits = [o for o in blk.GetContents(f"{name}.BlkDef")]
    if len(hits) != 1:
        raise PFSessionError(
            f"{name!r} ({purpose}) not found in the project's 'User Defined "
            f"Models' folder ({len(hits)} matches). It is hand-authored and "
            f"must not be recreated by script -- restore it before rolling out."
        )
    return hits[0]


def _fill_slots(comp, park, pre) -> Dict[str, Optional[str]]:
    """Bind every frame slot by NAME via ``ElmComp.pblk``.

    ``pelm`` is ordered by ``pblk``, which is *not* the frame's
    ``GetContents`` slot order (Plant Control is pblk index 8, not 6).
    Index-based writes silently mis-slot the blocks and can evict the
    generator, which makes the park run as a plain constant-Q source.
    """
    kids = list(comp.GetContents())

    def by(cls: str, frag: Optional[str] = None):
        for o in kids:
            if o.GetClassName() == cls and (frag is None or frag in o.loc_name):
                return o
        return None

    want = {
        "Generator": park,
        "Electrical Control": by("ElmDsl", "REEC"),
        "Gen-Con Model": by("ElmDsl", "REGC"),
        "Power Measurement": by("StaPqmea"),
        "Voltage Measurement": by("StaVmea"),
        "Protection": by("ElmDsl", "Protection"),
        "Weak Grid Option": by("ElmDsl", "WTGWGO"),
        "Current Measurement": by("StaImea"),
        "Plant Control": pre,
    }
    pblk = comp.GetAttribute("pblk")
    pelm = []
    for slot in pblk:
        nm = slot.loc_name if slot is not None else None
        if nm not in want:
            raise PFSessionError(
                f"{comp.loc_name}: frame slot {nm!r} has no mapping")
        pelm.append(want[nm])
    comp.SetAttribute("pelm", pelm)
    return {s.loc_name: (f.loc_name if f is not None else None)
            for s, f in zip(pblk, comp.GetAttribute("pelm"))}


def set_qv_params(pre, *, qset_pu: float, v_anchor_pu: float,
                  slope_pu: float, deadband_pu: float,
                  q_min_pu: float, q_max_pu: float,
                  tf_s: float = QV_FILTER_TF_S) -> None:
    """Write the QVPRE parameter vector, ordered by ``QVPRE_PARAM_ORDER``.

    ``Kdroop = 1/slope_pu`` because the block works in pu of the park's
    rating: ``Q = S_n*(qset - Kdroop*db(u - Vanchor))`` reproduces the static
    ``Q = q_set - (S_n/slope)*db(V - V_anchor)``.

    ``q_min_pu``/``q_max_pu`` are the operating-diagram limits in pu of S_n,
    mirroring the static plant's ``np.clip(q_target, q_min, q_max)``.  Without
    them the RMS converter is bounded only by ``REEC_D.Imax`` (1.3 pu), which
    is a different plant and makes the Gate-E endpoint comparison invalid.
    """
    n = len(QVPRE_PARAM_ORDER)
    if pre.GetAttributeLength("params") != n:
        pre.SetAttributeLength("params", n)
    values = (qset_pu, v_anchor_pu, 1.0 / float(slope_pu), deadband_pu, tf_s,
              q_min_pu, q_max_pu)
    # Element-wise: writing the whole vector in one SetAttribute call fails.
    for i, value in enumerate(values):
        pre.SetAttribute(f"params:{i}", float(value))


def apply_to_park(app, tcomp, park, *, local_frame=None, qvpre_bd=None) -> str:
    """Build the WECC composite for one ElmGenstat; returns the comp name.

    With ``local_frame``/``qvpre_bd`` given, the composite is retargeted to
    the project-local frame and a fresh ``QVPRE`` block is bound into its
    Plant Control slot (the plant-side re-anchored Q(V) layer).  The block is
    always recreated: an ``ElmDsl`` that predates its final BlkDef keeps a
    dead parameter table whose runtime values read zero.
    """
    name = f"WECC_{park.loc_name}"
    grid = park.fold_id                       # the park's own ElmNet
    for c in get_all(app, "ElmComp"):
        if c.loc_name == name:
            c.Delete()
    comp = grid.AddCopy(tcomp)
    comp.loc_name = name

    cub = park.GetAttribute("bus1")
    term = cub.cterm

    for meas in comp.GetContents():
        cn = meas.GetClassName()
        if cn == "StaVmea":
            meas.SetAttribute("pbusbar", term)
        elif cn in ("StaPqmea", "StaImea"):
            meas.SetAttribute("pcubic", cub)
        elif cn == "ElmDsl" and "REEC" in meas.loc_name:
            meas.SetAttribute("PfFlag", 0.0)  # reactive-power control
            meas.SetAttribute("QFlag", 0.0)   # Qext is the Q reference
            meas.SetAttribute("VFlag", 0.0)   # immaterial while QFlag == 0

    pre = None
    if local_frame is not None and qvpre_bd is not None:
        comp.SetAttribute("typ_id", local_frame)
        for o in comp.GetContents(f"{QVPRE_ELEMENT_NAME}.ElmDsl"):
            o.Delete()
        pre = comp.CreateObject("ElmDsl", QVPRE_ELEMENT_NAME)
        pre.SetAttribute("typ_id", qvpre_bd)

    _fill_slots(comp, park, pre)
    return name


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Attach WECC RMS models to DER.")
    parser.add_argument("--only", default=None,
                        help="apply only to parks whose name starts with this")
    parser.add_argument("--verify", action="store_true",
                        help="run ComInc after applying")
    parser.add_argument("--no-qv-layer", action="store_true",
                        help="attach the bare WECC converter without the "
                             "plant-side re-anchored Q(V) pre-controller")
    parser.add_argument("--slope", type=float, default=QV_SLOPE_PU_DEFAULT,
                        help="Q(V) droop slope [pu] (qv_slope_pu)")
    parser.add_argument("--deadband", type=float,
                        default=QV_DEADBAND_PU_DEFAULT,
                        help="Q(V) deadband [pu] (qv_deadband_pu)")
    parser.add_argument("--op-diagram", default="VDE-AR-N-4120-v2",
                        help="operating diagram for the placeholder Q limits; "
                             "PowerFactoryPlant overrides these per park from "
                             "the snapshot at init")
    parser.add_argument("--project", default=DEFAULT_PROJECT_PATH)
    args = parser.parse_args(argv)

    app = connect(args.project, study_case=RMS_STUDY_CASE)
    _activate_full(app)
    tcomp = _template_composite(app)

    qv = not args.no_qv_layer
    local_frame = _project_blockdef(app, LOCAL_FRAME_NAME,
                                    "Q(V) frame with the wired u input") if qv else None
    qvpre_bd = _project_blockdef(app, QVPRE_BLOCKDEF_NAME,
                                 "re-anchored Q(V) DSL law") if qv else None

    prefixes = (args.only,) if args.only else DER_PREFIXES
    parks = [g for g in get_all(app, "ElmGenstat")
             if g.loc_name.startswith(prefixes)]
    print(f"applying WECC model{' + Q(V) layer' if qv else ''} to "
          f"{len(parks)} DER parks ...")
    built = []
    for p in parks:
        built.append(apply_to_park(app, tcomp, p,
                                   local_frame=local_frame, qvpre_bd=qvpre_bd))
    print(f"built {len(built)} WECC composites")

    if qv:
        # One load flow anchors every park: qset = Q_LF/S_n, Vanchor = V_LF,
        # so each QVPRE initialises holding its load-flow reactive power.
        ldf = app.GetFromStudyCase("ComLdf")
        for k, v in PARITY_LDF_SETTINGS.items():
            ldf.SetAttribute(k, v)
        if ldf.Execute():
            raise PFSessionError("anchor load flow failed")
        # Re-fetch by name: the rebuild deleted/created 44 composites, which
        # can stale the handles captured before it (result attributes then
        # raise as if no calculation existed).
        park_names = {b[len("WECC_"):] for b in built}
        parks_now = {g.loc_name: g for g in get_all(app, "ElmGenstat")
                     if g.loc_name in park_names}
        comps_now = {c.loc_name: c for c in get_all(app, "ElmComp")}
        # Two passes: writing a DSL parameter modifies the model and
        # invalidates PF's calculation results, so every load-flow value must
        # be harvested BEFORE the first parameter is written.
        harvest = {}
        for nm in sorted(park_names):
            p = parks_now[nm]
            harvest[nm] = (
                float(p.GetAttribute("sgn")),
                float(p.GetAttribute("m:Q:bus1")),
                float(p.GetAttribute("bus1").cterm.GetAttribute("m:u")),
                float(p.GetAttribute("m:P:bus1")),
            )
        anchored = 0
        for nm, (sn, q_lf, v_lf, p_lf) in harvest.items():
            comp = comps_now[f"WECC_{nm}"]
            pre = next(o for o in comp.GetContents(f"{QVPRE_ELEMENT_NAME}.ElmDsl"))
            # Placeholder limits from the load-flow P; PowerFactoryPlant
            # re-anchors from the snapshot's per-park op_diagram at init.
            q_min, q_max = _qv_capability(sn, args.op_diagram, p_lf)
            set_qv_params(pre, qset_pu=q_lf / sn, v_anchor_pu=v_lf,
                          slope_pu=args.slope, deadband_pu=args.deadband,
                          q_min_pu=q_min / sn, q_max_pu=q_max / sn)
            anchored += 1
        print(f"anchored {anchored} Q(V) pre-controllers at the load-flow point "
              f"({args.op_diagram} limits)")

    if args.verify:
        ldf = app.GetFromStudyCase("ComLdf")
        for k, v in PARITY_LDF_SETTINGS.items():
            ldf.SetAttribute(k, v)
        inc = app.GetFromStudyCase("ComInc")
        inc.SetAttribute("iopt_sim", "rms")
        inc.SetAttribute("iopt_net", "sym")
        inc.SetAttribute("dtgrd", 10.0)
        inc.SetAttribute("tstart", 0.0)
        ierr = inc.Execute()
        print(f"ComInc after rollout -> {ierr} "
              f"({'OK' if ierr == 0 else 'FAILED — check output window'})")
        return 0 if ierr == 0 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
