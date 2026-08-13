"""Build PowerFactory graphics pages for watching a co-simulation live.

Engine mode gives no automatic live plotting (``GetApplicationExt`` runs PF
as a calculation server), but an explicit ``page.SetResults(ElmRes)`` plus a
``DoAutoScale()`` after each ``ComSim`` chunk *does* drive a repaint --
verified 2026-07-21.  ``PowerFactoryPlant.advance`` issues the refresh.

**Use ``GetOrInsertCurvePlot``, not a copy of an existing page.**  The first
attempt cloned "Voltage Magnitudes", which is a *bar* plot of instantaneous
magnitudes, so the RMS traces rendered as bars.

## What can and cannot carry a setpoint

PF can only plot signals that exist in its result file, and the OFO's
setpoints are computed in Python:

* **DER Q** -- the setpoint *is* in PF: ``apply_u`` writes it to the QVPRE
  block's ``qset`` parameter, so ``s:qset`` is the commanded value.  It is in
  pu of the park rating while ``m:Q:bus1`` is in Mvar, hence the split view:
  actuals and setpoints share a page but not an axis.
* **Generator Q** -- there is no Q setpoint to plot.  The OFO dispatches the
  AVR *voltage* reference and the machine's Q follows from the network; Q_gen
  enters the MIQP as a soft-constrained output, not a command.  The page
  therefore shows machine Q against the commanded V-ref.
* **Interface Q_DS** -- the TSO's Q setpoint for each coupler exists only in
  Python.  Plotting it inside PF needs a "setpoint carrier" written by
  ``EvtParam`` each dispatch; see ``TODO`` in the interface page builder.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from pf.session import get_all


#: Every page this module builds.  ``find_pages`` resolves whichever exist so
#: the plant can refresh them all after an advance -- PowerFactory repaints a
#: page only when something asks it to, and asking only the fronted page means
#: switching tabs mid-run shows stale curves.
QOFO_PAGE_NAMES = (
    "TS Bus Voltages (qOFO)",
    "TS Zone Voltages (qOFO)",
    "DSO Voltages (qOFO)",
    "DER Q (qOFO)",
    "Generator Q (qOFO)",
    "Generator V and AVR V-ref (qOFO)",
    "Interface Q_DS (qOFO)",
    "Machine OLTC Taps (qOFO)",
    "Coupler OLTC Taps (qOFO)",
    "Shunt Steps (qOFO)",
)


def find_pages(app, names: Sequence[str] = QOFO_PAGE_NAMES) -> List[Any]:
    """Graphics-board pages matching ``names`` that actually exist."""
    board = app.GetFromStudyCase("SetDesktop")
    by_name = {o.loc_name: o for o in board.GetContents()}
    return [by_name[n] for n in names if n in by_name]


def _fresh_page(app, name: str):
    """Delete and recreate a graphics-board page (idempotent rebuild)."""
    board = app.GetFromStudyCase("SetDesktop")
    for obj in list(board.GetContents()):
        if obj.loc_name == name:
            obj.Delete()
    return board.CreateObject("GrpPage", name)


def _curve_plot(page, title: str, curves: Sequence[tuple]):
    """One curve plot on ``page`` carrying ``[(object, variable), ...]``."""
    plot = page.GetOrInsertCurvePlot(title)
    series = plot.GetDataSeries()
    series.ClearCurves()
    for obj, var in curves:
        series.AddCurve(obj, var)
    return plot


def _bind_results(app, page) -> None:
    try:
        page.SetResults(app.GetFromStudyCase("ElmRes"))
    except Exception as exc:                        # noqa: BLE001
        print(f"  [warn] SetResults failed on {page.loc_name!r}: {exc}")


# ---------------------------------------------------------------- pages
def build_ts_bus_voltage_page(app, name: str = "TS Bus Voltages (qOFO)"):
    page = _fresh_page(app, name)
    terms = sorted((t for t in get_all(app, "ElmTerm")
                    if t.loc_name.startswith("TN_bus")),
                   key=lambda x: int(x.loc_name[6:]))
    _curve_plot(page, "TS bus voltages", [(t, "m:u") for t in terms])
    _bind_results(app, page)
    return page, len(terms)


def build_zone_voltage_page(app, zone_map, name: str = "TS Zone Voltages (qOFO)"):
    """Split view: one curve plot per TSO control zone.

    ``zone_map`` is ``{zone_id: [pandapower bus index, ...]}`` as carried in
    the snapshot.  TN terminals are named ``TN_bus<pandapower index>``, which
    is what ties the PF objects back to the zone partition.

    Separate plots rather than one: each zone is regulated to its own
    setpoint by its own MIQP, so a zone's spread is the quantity of interest
    and a shared axis buries it under the inter-zone offset.
    """
    page = _fresh_page(app, name)
    terms = {}
    for t in get_all(app, "ElmTerm"):
        if t.loc_name.startswith("TN_bus"):
            try:
                terms[int(t.loc_name[len("TN_bus"):])] = t
            except ValueError:
                continue
    total = 0
    for zone in sorted(zone_map, key=lambda z: int(z)):
        buses = sorted(int(b) for b in zone_map[zone])
        curves = [(terms[b], "m:u") for b in buses if b in terms]
        if not curves:
            continue
        _curve_plot(page, f"TS zone {zone} bus voltages [pu]", curves)
        total += len(curves)
    _bind_results(app, page)
    return page, total


def build_dso_voltage_page(app, name: str = "DSO Voltages (qOFO)"):
    """Split view: one curve plot per DSO area, all its feeder-bus voltages.

    Added 2026-07-22 to see why a coupler OLTC tapped excessively -- the DSO
    voltage spread it is regulating is the driver, and the aggregate TS view
    hides it.  DSO feeder buses are named ``DSO_<n>_...``; they are recorded
    in the result file only when the plant is built with
    ``monitored_outputs(include_der=True)`` (the Gate-E path), so on a run
    without that these plots are empty.
    """
    page = _fresh_page(app, name)
    by_area: dict = {}
    for t in get_all(app, "ElmTerm"):
        nm = t.loc_name
        if nm.startswith("DSO_") and "_" in nm[4:]:
            area = nm[:nm.index("_", 4)]          # "DSO_1" from "DSO_1_bus46"
            by_area.setdefault(area, []).append(t)
    total = 0
    for area in sorted(by_area):
        terms = sorted(by_area[area], key=lambda x: x.loc_name)
        _curve_plot(page, f"{area} feeder-bus voltages [pu]",
                    [(t, "m:u") for t in terms])
        total += len(terms)
    _bind_results(app, page)
    return page, total, len(by_area)


def build_der_q_page(app, name: str = "DER Q (qOFO)"):
    """Four views: TSO-DER and DSO-DER, each Q actual [Mvar] and qset [pu].

    Split TSO vs DSO because the two DER classes play different roles -- the
    4 transmission wind parks (``WECC_WP_TSO_*``) are the TSO's continuous Q
    actuators, the ~40 distribution parks the DSO's -- and mixing 44 curves
    on one axis hides both.  Actual and setpoint stay on separate plots
    because the units differ (Mvar vs pu of S_n).

    The setpoint plots read ``QVPRE.s:qset``; that signal must be recorded by
    ``monitored_outputs(include_der=True)`` or the curves are empty (which is
    why the old single setpoint plot was blank).
    """
    page = _fresh_page(app, name)
    tso_act: List[tuple] = []
    tso_set: List[tuple] = []
    dso_act: List[tuple] = []
    dso_set: List[tuple] = []
    for comp in sorted((c for c in get_all(app, "ElmComp")
                        if c.loc_name.startswith("WECC_")),
                       key=lambda c: c.loc_name):
        is_tso = comp.loc_name.startswith("WECC_WP_TSO_")
        gen = next((e for b, e in zip(comp.GetAttribute("pblk"),
                                      comp.GetAttribute("pelm"))
                    if b is not None and b.loc_name == "Generator"), None)
        pre = next((d for d in comp.GetContents()
                    if d.GetClassName() == "ElmDsl" and d.loc_name == "QVPRE"),
                   None)
        if gen is not None:
            (tso_act if is_tso else dso_act).append((gen, "m:Q:bus1"))
        if pre is not None:
            (tso_set if is_tso else dso_set).append((pre, "s:qset"))
    _curve_plot(page, "TSO-DER reactive infeed [Mvar]", tso_act)
    _curve_plot(page, "TSO-DER Q setpoint qset [pu] (OFO command)", tso_set)
    _curve_plot(page, "DSO-DER reactive infeed [Mvar]", dso_act)
    _curve_plot(page, "DSO-DER Q setpoint qset [pu] (OFO command)", dso_set)
    _bind_results(app, page)
    return page, len(tso_act) + len(dso_act), len(tso_set) + len(dso_set)


def build_gen_q_page(app, name: str = "Generator Q (qOFO)"):
    """Split view: machine Q [Mvar] over commanded AVR V-ref [pu].

    NOTE the asymmetry with the DER page: the OFO does not command generator
    reactive power.  It dispatches the AVR voltage setpoint (written to the
    AVR DSL's ``usetp``), and Q is whatever the network then draws.  So the
    lower plot is the actual command; there is no Q setpoint to dash in.
    """
    page = _fresh_page(app, name)
    actual: List[tuple] = []
    vref: List[tuple] = []
    for mach in sorted(get_all(app, "ElmSym"), key=lambda m: m.loc_name):
        if mach.GetAttribute("outserv") != 0:
            continue
        actual.append((mach, "m:Q:bus1"))
        comp = mach.GetAttribute("c_pmod")
        if comp is None:
            continue                      # e.g. G 01, the network equivalent
        avr = next((d for d in comp.GetContents()
                    if d.GetClassName() == "ElmDsl"
                    and "avr" in d.loc_name.lower()), None)
        if avr is not None:
            vref.append((avr, "s:usetp"))
    _curve_plot(page, "Generator reactive infeed [Mvar]", actual)
    _curve_plot(page, "AVR voltage setpoint usetp [pu] (OFO command)", vref)
    _bind_results(app, page)
    return page, len(actual), len(vref)


def build_gen_v_page(app, name: str = "Generator V and AVR V-ref (qOFO)"):
    """Machine terminal voltage against the OFO's AVR voltage reference.

    This is the meaningful command/response pair for synchronous machines --
    unlike Q, which the OFO never commands (see build_gen_q_page).  Both
    signals are pu voltages, so they share one plot: the AVR regulates the
    terminal to ``usetp``, and any visible gap between a machine's V and its
    own V-ref means the AVR is not holding -- typically Q saturation.

    G 01 contributes a terminal voltage but no V-ref: the 10 GVA network
    equivalent has no AVR block, which is why its setpoint is withheld from
    the OFO's actuator set entirely.
    """
    page = _fresh_page(app, name)
    curves: List[tuple] = []
    for mach in sorted(get_all(app, "ElmSym"), key=lambda m: m.loc_name):
        if mach.GetAttribute("outserv") != 0:
            continue
        # Terminal voltage from the machine's ElmTerm, NOT the machine object:
        # this is the (object, variable) pair monitored_outputs records, and
        # a plot curve is empty unless it references the recorded object.
        cub = mach.GetAttribute("bus1")
        term = cub.cterm if cub is not None else None
        if term is not None:
            curves.append((term, "m:u"))
        comp = mach.GetAttribute("c_pmod")
        if comp is None:
            continue
        avr = next((d for d in comp.GetContents()
                    if d.GetClassName() == "ElmDsl"
                    and "avr" in d.loc_name.lower()), None)
        if avr is not None:
            curves.append((avr, "s:usetp"))
    _curve_plot(page, "Generator terminal V and AVR V-ref [pu]", curves)
    _bind_results(app, page)
    return page, len(curves)


def build_interface_q_page(app, name: str = "Interface Q_DS (qOFO)"):
    """Coupler-3W reactive flow at the EHV-HV interface.

    Flows only, by decision (2026-07-21).  The TSO's per-coupler Q_PCC
    setpoint is computed in Python and never written to PowerFactory, so
    plotting it here would require a "setpoint carrier" DSL block per coupler
    driven by ``EvtParam`` each dispatch.  That was judged not worth the
    machinery: the setpoint-vs-actual comparison is already available on the
    Python side, in ``csv/endpoint_comparison.csv`` and the Gate-E overlays.
    """
    page = _fresh_page(app, name)
    trafos = sorted((t for t in get_all(app, "ElmTr3")
                     if t.loc_name.startswith("NC3W_")),
                    key=lambda t: t.loc_name)
    _curve_plot(page, "Interface Q at coupler HV side [Mvar]",
                [(t, "m:Q:bushv") for t in trafos])
    _bind_results(app, page)
    return page, len(trafos)


def build_tap_pages(app, tr2_var: str, tr3_var: str, shunt_var: str):
    """Three pages: machine-2W taps, coupler-3W taps, shunt steps.

    The variable names are passed in rather than hard-coded because the
    *input* attributes (``nntap`` / ``n3tap_h`` / ``ncapa``) are not updated
    by simulation events -- that is exactly why the plant keeps a shadow of
    the discrete state.  The RMS-readable equivalents must be discovered
    against a live result file; see scratchpad probe_tapvars.py.

    These are the OFO's discrete actuators, so the pages show *commanded*
    state as realised in the plant -- the quantity that diverged between the
    static and RMS runs (2026-07-21: 7 of 27 actuators ended in different
    positions).
    """
    out = []

    page = _fresh_page(app, "Machine OLTC Taps (qOFO)")
    tr2 = sorted((t for t in get_all(app, "ElmTr2")
                  if t.loc_name.startswith("MT_")),
                 key=lambda t: t.loc_name)
    _curve_plot(page, "Machine transformer tap position (2W)",
                [(t, tr2_var) for t in tr2])
    _bind_results(app, page)
    out.append((page.loc_name, len(tr2)))

    page = _fresh_page(app, "Coupler OLTC Taps (qOFO)")
    tr3 = sorted((t for t in get_all(app, "ElmTr3")
                  if t.loc_name.startswith("NC3W_")),
                 key=lambda t: t.loc_name)
    _curve_plot(page, "EHV-HV coupler transformer tap position (3W)",
                [(t, tr3_var) for t in tr3])
    _bind_results(app, page)
    out.append((page.loc_name, len(tr3)))

    page = _fresh_page(app, "Shunt Steps (qOFO)")
    shunts = sorted(get_all(app, "ElmShnt"), key=lambda s: s.loc_name)
    _curve_plot(page, "MSC/MSR shunt step position",
                [(s, shunt_var) for s in shunts])
    _bind_results(app, page)
    out.append((page.loc_name, len(shunts)))
    return out


def build_all(app, zone_map=None, tap_vars=None) -> List[str]:
    """(Re)build every qOFO live page; returns the page names.

    ``zone_map`` ({zone: [bus index, ...]}, from the snapshot) enables the
    per-zone split page; ``tap_vars`` is ``(tr2_var, tr3_var, shunt_var)``
    for the discrete-actuator pages.  Either omitted -> those pages skipped.
    """
    names: List[str] = []
    pg, n = build_ts_bus_voltage_page(app)
    print(f"  {pg.loc_name!r}: {n} bus voltage curves")
    names.append(pg.loc_name)
    if zone_map:
        pg, n = build_zone_voltage_page(app, zone_map)
        print(f"  {pg.loc_name!r}: {n} curves across {len(zone_map)} zones")
        names.append(pg.loc_name)
    pg, n, na = build_dso_voltage_page(app)
    print(f"  {pg.loc_name!r}: {n} curves across {na} DSO areas")
    names.append(pg.loc_name)
    pg, na, ns = build_der_q_page(app)
    print(f"  {pg.loc_name!r}: {na} actual + {ns} setpoint curves")
    names.append(pg.loc_name)
    pg, na, nv = build_gen_q_page(app)
    print(f"  {pg.loc_name!r}: {na} machine Q + {nv} AVR V-ref curves")
    names.append(pg.loc_name)
    pg, n = build_gen_v_page(app)
    print(f"  {pg.loc_name!r}: {n} curves (terminal V + V-ref, overlaid)")
    names.append(pg.loc_name)
    pg, n = build_interface_q_page(app)
    print(f"  {pg.loc_name!r}: {n} interface flows (actuals only, by design)")
    names.append(pg.loc_name)
    if tap_vars:
        for nm, n in build_tap_pages(app, *tap_vars):
            print(f"  {nm!r}: {n} curves")
            names.append(nm)
    return names
