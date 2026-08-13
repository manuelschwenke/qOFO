"""
pf/plant.py
===========
``PowerFactoryPlant`` -- the DIgSILENT RMS plant behind the Phase-6
:class:`core.plant.Plant` interface (RMS build plan, Phase 6 step 2).

The OFO cascade keeps acting on the *pandapower index namespace*
(:class:`core.plant.ActuatorWrites`) and keeps reading a *pandapower
measurement image* (``read_y()`` returns a mirror net with refreshed
``res_*`` tables), so controllers and ``core.measurement`` are byte-for-byte
unaware which plant they face.  Internally:

* ``apply_u`` schedules RMS events at the current (paused) simulation time
  -- ``EvtParam`` on ``REEC_D.Qext`` for DER Q (pu of park S_n), ``EvtParam``
  on the machine AVR's ``usetp`` signal for V-refs, ``EvtParam`` on the
  ``TAPCTRL`` DSL's ``ntapcmd`` for OLTC taps (absolute position; see
  pf/tap_ctrl.py -- ``EvtTap`` cannot move a tap reliably mid-run), and
  ``EvtTap`` for MSC/MSR shunt steps (immediate, verified 2026-07-20).
* ``advance(T)`` continues ``ComSim`` by ``T`` seconds.
* ``read_y`` harvests paused-state attributes into the mirror net using the
  exact quantity mapping the parity gates validated (pf/pf_parity.py):
  ``m:u``/``m:phiu`` -> ``res_bus``, ``m:I:bus1`` -> ``res_line.i_from_ka``,
  ``m:P/Q:bushv`` -> ``res_trafo3w``, ``m:P/Q:bus1`` -> ``res_sgen`` /
  ``res_gen`` (machine results are plant totals, no ngnum scaling).

Discrete actuator state (taps, shunt steps) is tracked in a shadow store:
PF input attributes are *not* updated by simulation events, so the plant
remembers what it dispatched (initialised from the PF input data, which the
sync gates proved equal to the snapshot).

Known modelling divergences from :class:`core.plant.PandapowerStaticPlant`
(documented, deliberate):

* DER between dispatches: REEC_D in Q-control mode holds constant Q, while
  the static plant's ``QVLocalLoop`` re-droops around the reanchored V_ref.
  The RMS side therefore has *no* autonomous Q(V) response between OFO
  dispatches.
* ``G 01`` (10 GVA 'Rest of U.S.A. / Canada' equivalent) has **no AVR** in
  the template -- V-ref commands for it cannot be actuated in RMS
  (``on_missing_avr`` selects raise vs record-and-skip).

Author: Manuel Schwenke / Claude Code (2026-07-20)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.plant import ActuatorWrites, write_der_q_set  # noqa: E402
from export.dynamic_snapshot import (  # noqa: E402
    load_snapshot,
    load_snapshot_to_pandapower,
)
from pf.naming import build_name_map, machine_template_name  # noqa: E402
from controller.der_qv_local_loop import _qv_capability  # noqa: E402
from pf.wecc_apply import (  # noqa: E402
    QVPRE_ELEMENT_NAME,
    QV_DEADBAND_PU_DEFAULT,
    QV_SLOPE_PU_DEFAULT,
    set_qv_params,
)
from pf.pf_parity import PARITY_LDF_SETTINGS  # noqa: E402
from pf.screening import (  # noqa: E402
    RMS_STEP_MS,
    RMS_STUDY_CASE,
    SHUNT_STEP_VAR,
    TAP_MECH_DELAY_S,
    ScreeningContext,
    monitored_outputs,
)
from pf.profile_playback import (  # noqa: E402
    install_profile_playback,
    remove_profile_playback_models,
)
from pf.result_export import (  # noqa: E402
    export_comres_csv,
    load_comres_trajectories,
)
from pf.session import (  # noqa: E402
    DEFAULT_PROJECT_PATH,
    PFSessionError,
    connect,
    get_all,
)

logger = logging.getLogger("qofo.pf.plant")

#: Q(V) dead-band half-width [pu] that disables the droop outright.  0.5 pu is
#: what the dead-band study already uses to encode its "no droop" leg: no
#: credible voltage deviation reaches it, so the droop term is identically
#: zero.  The natural argument for ``qv_deadband_at_contingency`` when the
#: counterfactual wanted is "the local layer does nothing from here on".
QV_DISABLED_DEADBAND_PU = 0.5


class PowerFactoryPlant:
    """RMS co-simulation plant (:class:`core.plant.Plant` implementation).

    Parameters
    ----------
    snapshot : path or parsed snapshot document
        The reference snapshot the PF project is synced to (typically
        ``export/snapshots/full_t0_*.json``).  Supplies the name map, the
        mirror network and the actuator inventory.
    project, study_case :
        PF database path / study case (defaults: the qOFO project and
        ``02_RMS_CoSim``).
    on_missing_avr : "raise" | "skip"
        A ``gen_v_pu`` write targeting a machine without an AVR block
        (G 01) either raises (default, Fail-Fast) or is recorded in
        :attr:`skipped_writes` and skipped.
    app :
        Existing PF application handle (optional; a fresh engine session
        is opened otherwise).
    mirror_net :
        Externally owned pandapower net to use as the measurement image
        (e.g. the net built by ``run_multi_tso_dso`` when this plant is
        substituted via its ``plant_factory``).  Must correspond to the
        snapshot topology and carry converged ``res_*`` tables.  Default:
        rebuilt from the snapshot and converged once.
    event_pool_slots : int
        Minimum persistent slots per target available before ComInc. Pools
        discovered from earlier runs are retained; missing slots grow on
        demand and become reusable after the next ComInc.
    preallocate_profiles : bool
        Also reserve load and DER-active-power profile-event slots.
    """

    #: Gate-E comparisons must not claim plant equivalence until the RMS
    #: model implements the static plant's re-anchored Q(V) characteristic.
    #: Class default is the conservative "not equivalent"; ``_resolve_handles``
    #: raises it once every park carries a ``QVPRE`` pre-controller.
    der_qv_local_control_equivalent = False

    #: Event scheduling offset after the paused time [s].  Events are admitted
    #: into the running RMS calculation by *advancing* ComSim barriers that
    #: stop strictly before this offset (see ScreeningContext.admit_new_events,
    #: validated by probe_event_admission_barrier); the offset must therefore
    #: leave room for a few barrier stops (>~ a few RMS grid steps).  0.5 s is
    #: ~a 2.5 % slice of the 20 s dispatch interval -- negligible for tracking.
    _EVENT_EPS_S = 0.5

    def __init__(self, snapshot: Union[str, Path, Mapping[str, Any]], *,
                 project: str = DEFAULT_PROJECT_PATH,
                 study_case: str = RMS_STUDY_CASE,
                 on_missing_avr: str = "raise",
                 app=None,
                 mirror_net=None,
                 gui_pages=None,
                 gui_off_flag=None,
                 gui_refresh_every: int = 1,
                 live_view=None,
                 live_view_stride: int = 50,
                 event_pool_slots: int = 1,
                 preallocate_profiles: bool = False,
                 profile_playback_config=None,
                 qv_deadband_at_contingency: Optional[float] = None,
                 rms_step_ms: float = RMS_STEP_MS,
                 rms_step_max_ms: Optional[float] = None,
                 adaptive_step: bool = False):
        if on_missing_avr not in ("raise", "skip"):
            raise ValueError(f"on_missing_avr {on_missing_avr!r}")
        #: Diagnostic for the dead-band x droop study ONLY.  When set, every
        #: park runs its CONFIGURED Q(V) dead band through the run-up and this
        #: value is installed instead at the instant of the contingency.
        #:
        #: The point is a clean counterfactual: two legs that share config,
        #: controllers, static plant and RMS run-up exactly, and differ only
        #: in what the local layer does AFTER the disturbance.  Setting the
        #: leg apart through ``--tso-deadband`` instead does not achieve this
        #: -- that value also feeds the controllers and the static plant, so
        #: the closed loops diverge from t = 0 (measured 2026-08-06: 1.33e-2
        #: pu of run-up divergence, against 6.7e-7 pu when the configured dead
        #: band matches).
        #:
        #: RMS plant only; the static plant keeps its configured Q(V)
        #: throughout, so Gate E is NOT meaningful for such a run.
        self._qv_db_at_contingency = (
            None if qv_deadband_at_contingency is None
            else float(qv_deadband_at_contingency))
        #: {sgen index: dead band [pu] to install at the contingency}.
        self._qv_target_db: Dict[int, float] = {}
        #: RMS integration step [ms]; the minimum step when adaptive.
        self._rms_step_ms = float(rms_step_ms)
        self._rms_step_max_ms = (RMS_STEP_MS if rms_step_max_ms is None
                                 else float(rms_step_max_ms))
        self._adaptive_step = bool(adaptive_step)
        if int(event_pool_slots) < 1:
            raise ValueError("event_pool_slots must be >= 1")
        self._event_pool_slots = int(event_pool_slots)
        self._preallocate_profiles = bool(preallocate_profiles)
        self._profile_playback_config = profile_playback_config
        #: Graphics pages redrawn after each advance (empty = headless).  All
        #: of them, so any tab is current when switched to; throttled by
        #: gui_refresh_every.  See advance().
        self._gui_pages = list(gui_pages) if gui_pages else []
        #: Sentinel file: if it appears mid-run, the live refresh is disabled
        #: on the next interval (the only off-switch engine mode allows).
        self._gui_off_flag = str(gui_off_flag) if gui_off_flag else None
        #: Refresh the plot every Nth interval (1 = every interval).  Higher
        #: values trade live-plot smoothness for run speed on long horizons.
        self._gui_refresh_every = max(1, int(gui_refresh_every))
        self._gui_interval = 0
        self._gui_was_off = False
        #: Folder watched for the live-control sentinels (DISABLE_GUI /
        #: HIDE_GUI), and the current desktop-hidden state.  Hiding the
        #: window is the strongest off-switch: it removes the desktop AND
        #: skips the redraw.  Hide()/Show() are safe to call mid-run
        #: (verified: the simulation continues); PF may segfault at process
        #: exit afterward, which is harmless -- results are written first.
        self._gui_folder = (os.path.dirname(self._gui_off_flag)
                            if self._gui_off_flag else None)
        #: Seeded from the sentinel rather than hardcoded False: a run that
        #: starts with the desktop hidden pre-creates HIDE_GUI, and the toggle
        #: in advance() only acts when the wanted state DIFFERS from this.
        #: Assuming "shown" at construction would therefore make the first
        #: show_gui.bat a no-op -- the window would never appear.
        self._gui_hidden = bool(
            self._gui_folder
            and os.path.exists(os.path.join(self._gui_folder, "HIDE_GUI"))
        )
        #: Python-side live plot (pf.live_view.LiveTrajectoryView) or None.
        #: Engine mode gives no live PF plot, so this is the only way to
        #: watch an RMS run in progress -- see pf/live_view.py.
        self._live_view = live_view
        #: Result-row stride for the live harvest.  Coarser than the export
        #: stride on purpose: ``ctx.read`` walks the whole result each call,
        #: so the per-interval cost grows with simulated time.  50 = 0.5 s.
        self._live_view_stride = int(live_view_stride)
        self.on_missing_avr = on_missing_avr
        self.skipped_writes: List[Tuple[str, int, float]] = []

        doc = snapshot if isinstance(snapshot, Mapping) \
            else load_snapshot(snapshot)
        self.doc = doc
        self.names = build_name_map(doc)

        # Mirror net = the measurement image read_y() refreshes.  Either
        # externally owned (runner integration) or rebuilt from the
        # snapshot and converged once with its own solver options so
        # every res_* table exists with the right shape.
        if mirror_net is not None:
            self.net = mirror_net
        else:
            import pandapower as pp
            self.net, _ = load_snapshot_to_pandapower(doc)
            pp.runpp(self.net, **doc["solver_options"])

        self.app = app if app is not None \
            else connect(project, study_case=study_case)
        self.ctx = ScreeningContext(
            self.app, verbose=False, persistent_event_pool=True,
            rms_step_ms=self._rms_step_ms,
            rms_step_max_ms=self._rms_step_max_ms,
            adaptive_step=self._adaptive_step)
        print(f"  [rms] integration step {self._rms_step_ms} ms"
              + (f", adaptive up to {self._rms_step_max_ms} ms"
                 if self._adaptive_step else " (fixed)"))

        self._profile_playback = None
        self._profile_playback_cleanup = {}
        if self._profile_playback_config is None:
            self._profile_playback_cleanup = (
                remove_profile_playback_models(self.app))
        self._resolve_handles()
        self._init_shadow_state()
        if self._profile_playback_config is not None:
            config = dict(self._profile_playback_config)
            self._profile_playback = install_profile_playback(
                self.app,
                self.net,
                loads=self._loads,
                sgens=self._sgens,
                wgo=self._wgo,
                wecc_composites=self._wecc_comp,
                sgen_sn=self._sgen_sn,
                profiles=config["profiles"],
                start_time=config["start_time"],
                dt_s=config["dt_s"],
                duration_s=config["duration_s"],
                file_path=config["file_path"],
                transition_delay_s=(
                    self._EVENT_EPS_S + float(config["rms_step_s"])),
            )


        # Clean the event folder before ComInc.  The slot-reuse pool re-armed
        # fired events within one active calculation, which PF does NOT honour
        # (probe_event_rearm 2026-07-23: a fired EvtParam/EvtLod never re-fires)
        # -- it froze every DER at ~t=41 s in run 0043.  We instead create a
        # fresh event per dispatch and admit the batch with zero-time
        # ComSim.Execute barriers (proven by probe_event_admission_barrier).
        # The pool path is kept behind the flag for a future memory-bounded
        # variant (pre-created one-shot horizon pool, no reuse).
        if self.ctx.persistent_event_pool:
            discard_loads = (
                self._profile_playback.load_targets
                if self._profile_playback is not None else ())
            discard_pref = (
                tuple((target, "Pref_in")
                      for target in self._profile_playback.pref_targets)
                if self._profile_playback is not None else ())

            self._event_pool_prepare_stats = (
                self.ctx.prepare_persistent_event_pool(
                    discard_param_targets=discard_pref,
                    discard_load_targets=discard_loads)
            )
        else:
            self.ctx.purge_events()
            self._event_pool_prepare_stats = {}
        # Trajectory monitors (controlled outputs; see pf/screening.py).
        # include_der: Gate E needs per-park Q/V to attribute an endpoint gap
        # (capability clip vs. droop amplification); see monitored_outputs.
        # Keep the list: harvest_trajectories must iterate exactly what was
        # registered, not re-derive it with different arguments.
        self._monitors = monitored_outputs(self.app, include_der=True)
        self.ctx.set_monitors(self._monitors)
        self._anchor_qv_precontrollers()
        if self.ctx.persistent_event_pool:
            self._preallocate_event_slots()
        self.ctx.initialise()
        self.t = 0.0
        self._n_events = 0
        #: EvtTap shunt-step count; >1 makes the trajectory unreliable.
        self._shunt_events = 0
        # Outage tolerance in read_y: PF drops an element's ``m:`` result
        # variables when it leaves service. ``_seen_in_service`` records
        # elements that have read successfully at least once, so a failure
        # before that is treated as a defect rather than as an outage.
        self._out_of_service: set = set()
        self._seen_in_service: set = set()

    # ------------------------------------------------------------------
    #  Handle resolution (Fail-Fast: every mapped element must exist)
    # ------------------------------------------------------------------
    def _by_name(self, class_name: str) -> Dict[str, Any]:
        return {o.loc_name: o for o in get_all(self.app, class_name)}

    def _resolve_handles(self) -> None:
        model = self.doc["model"]
        names = self.names

        def _table(table: str, pf_objs: Dict[str, Any]) -> Dict[int, Any]:
            out: Dict[int, Any] = {}
            for key in model.get(table, {}):
                idx = int(key)
                nm = names[(table, idx)]
                obj = pf_objs.get(nm)
                if obj is None:
                    raise PFSessionError(
                        f"{table}[{idx}] -> {nm!r}: not found in the active "
                        f"PF model (wrong variation set or sync missing?)")
                out[idx] = obj
            return out

        self._terms = _table("bus", self._by_name("ElmTerm"))
        self._lines = _table("line", self._by_name("ElmLne"))
        self._tr2 = _table("trafo", self._by_name("ElmTr2"))
        self._tr3 = _table("trafo3w", self._by_name("ElmTr3"))
        self._loads = _table("load", self._by_name("ElmLod"))
        self._shunts = _table("shunt", self._by_name("ElmShnt"))
        self._sgens = _table("sgen", self._by_name("ElmGenstat"))

        # DER Q handles.  Preferred: the plant-side re-anchored Q(V)
        # pre-controller (``QVPRE``) in the composite's Plant Control slot,
        # which reproduces the static plant's ``QVLocalLoop`` law exactly.
        # Fallback: the REEC block's ``Qext`` (constant Q between dispatches
        # -- the documented actuator-law mismatch, kept so a bare-converter
        # model can still be driven and compared).
        comps = self._by_name("ElmComp")
        self._reec: Dict[int, Any] = {}
        self._qvpre: Dict[int, Any] = {}
        self._wgo: Dict[int, Any] = {}
        self._wecc_comp: Dict[int, Any] = {}
        self._park_term: Dict[int, Any] = {}
        self._sgen_sn: Dict[int, float] = {}
        for idx, gen in self._sgens.items():
            comp = comps.get(f"WECC_{gen.loc_name}")
            if comp is None:
                raise PFSessionError(
                    f"sgen[{idx}] {gen.loc_name!r}: WECC composite missing "
                    f"(run pf/wecc_apply.py)")
            reec = next((d for d in comp.GetContents()
                         if d.GetClassName() == "ElmDsl"
                         and "REEC" in d.loc_name), None)
            if reec is None:
                raise PFSessionError(
                    f"sgen[{idx}]: no REEC block in WECC_{gen.loc_name}")
            self._wecc_comp[idx] = comp
            self._reec[idx] = reec
            pre = next((d for d in comp.GetContents()
                        if d.GetClassName() == "ElmDsl"
                        and d.loc_name == QVPRE_ELEMENT_NAME), None)
            if pre is not None:
                self._qvpre[idx] = pre
            cub = gen.GetAttribute("bus1")
            self._park_term[idx] = cub.cterm if cub is not None else None
            self._sgen_sn[idx] = float(gen.GetAttribute("sgn"))
            # Active-power reference for profile-driven P.  The WECC chain is
            # WTGWGO_A(Pref_in) -> Pref_out -> REEC_D(Pref); Pref_in is the
            # free end and takes an EvtParam in pu of the park rating
            # (verified 2026-07-21: 0.5 -> 254.088 MW on the 508 MVA park).
            self._wgo[idx] = next(
                (d for d in comp.GetContents()
                 if d.GetClassName() == "ElmDsl" and "WTGWGO" in d.loc_name),
                None)

        n_qv = len(self._qvpre)
        if n_qv and n_qv != len(self._sgens):
            raise PFSessionError(
                f"Q(V) pre-controller present on {n_qv}/{len(self._sgens)} "
                f"parks -- mixed actuator laws would make the comparison "
                f"uninterpretable; re-run pf/wecc_apply.py")
        self.der_q_mode = "qv_precontroller" if n_qv else "constant_q"
        # Equivalence is an instance fact, not a class fact: it holds exactly
        # when every park runs the re-anchored Q(V) law the static plant uses.
        # Gate E reads this attribute to decide whether the comparison is a
        # validation or merely diagnostic.
        self.der_qv_local_control_equivalent = self.der_q_mode == "qv_precontroller"
        logger.info("DER actuator law: %s (%d parks)", self.der_q_mode,
                    len(self._sgens))

        # Machines + their AVR blocks (template-owned names).
        syms = self._by_name("ElmSym")
        self._machines: Dict[int, Any] = {}
        self._avr: Dict[int, Optional[Any]] = {}
        avr_by_sym: Dict[str, Any] = {}
        for comp in comps.values():
            pelm = comp.GetAttribute("pelm") or []
            sym = next((e for e in pelm if e is not None
                        and e.GetClassName() == "ElmSym"), None)
            if sym is None:
                continue
            avr_by_sym[sym.loc_name] = next(
                (d for d in comp.GetContents()
                 if d.GetClassName() == "ElmDsl"
                 and "avr" in d.loc_name.lower()), None)
        for key, rec in model["gen"].items():
            idx = int(key)
            tpl = machine_template_name(rec)
            mach = syms.get(tpl)
            if mach is None:
                raise PFSessionError(f"gen[{idx}] -> ElmSym {tpl!r} missing")
            self._machines[idx] = mach
            self._avr[idx] = avr_by_sym.get(tpl)

        # ── DSL tap controllers (pf/tap_ctrl.py) ─────────────────────────
        # RMS ignores tap-controller definitions, so a tap can only be moved
        # by a dynamic model driving the transformer's ``nntapin`` input
        # (TechRef_*-W-Transformer S5).  Each controllable transformer gets an
        # ``ElmComp`` named ``TAPC_<transformer>`` holding one ``TAPCTRL``
        # ``ElmDsl``; its ``ntapcmd`` parameter is the commanded ABSOLUTE tap
        # position.  Missing composites are not an error here -- only a tap
        # write to that transformer is (see _dispatch_taps), so a model
        # without the layer still runs everything else.
        from pf.tap_ctrl import COMP_PREFIX, DSL_ELEMENT_NAME  # noqa: E402
        self._tapctrl: Dict[Tuple[str, int], Any] = {}
        for table, objs in (("trafo", self._tr2), ("trafo3w", self._tr3)):
            for idx, obj in objs.items():
                comp = comps.get(f"{COMP_PREFIX}{obj.loc_name}")
                if comp is None:
                    continue
                dsl = next(
                    (d for d in comp.GetContents(f"{DSL_ELEMENT_NAME}.ElmDsl")),
                    None)
                if dsl is not None:
                    self._tapctrl[(table, idx)] = dsl
        logger.info("resolved %d DSL tap controllers", len(self._tapctrl))

    def _init_shadow_state(self) -> None:
        """Discrete state from PF *input* data (= snapshot, per Gate C)."""
        self._tap2w = {i: int(o.GetAttribute("nntap"))
                       for i, o in self._tr2.items()}
        self._tap3w = {i: int(o.GetAttribute("n3tap_h"))
                       for i, o in self._tr3.items()}
        self._shunt_step = {i: int(o.GetAttribute("ncapa"))
                            for i, o in self._shunts.items()}

    # ------------------------------------------------------------------
    #  Plant protocol
    # ------------------------------------------------------------------
    def _park_anchor_voltage(self, idx: int) -> float:
        """Park-bus voltage measured in the paused RMS state [pu]."""
        term = self._park_term.get(idx)
        if term is None:
            raise PFSessionError(f"sgen[{idx}]: no terminal for Q(V) anchor")
        return float(term.GetAttribute("m:u"))

    def _anchor_qv_precontrollers(self) -> None:
        """Re-anchor every Q(V) pre-controller to the synced operating point.

        ``pf/wecc_apply.py`` anchors at rollout time, but a replay re-syncs
        the model to its own snapshot, which moves the operating point.
        Without this pass the DERs would start off their own droop curve and
        the RMS run would begin with a spurious reactive transient.
        Values are harvested for all parks *before* any is written: writing a
        DSL parameter modifies the model and invalidates PF's results.
        """
        if not self._qvpre:
            return
        ldf = self.app.GetFromStudyCase("ComLdf")
        for key, value in PARITY_LDF_SETTINGS.items():
            ldf.SetAttribute(key, value)
        if ldf.Execute():
            raise PFSessionError("Q(V) anchor load flow failed")
        harvest = {}
        for idx in self._qvpre:
            gen = self._sgens[idx]
            harvest[idx] = (float(gen.GetAttribute("m:Q:bus1")),
                            self._park_anchor_voltage(idx))
        sgen_model = self.doc["model"].get("sgen", {})
        _applied_db: Dict[int, float] = {}
        _applied_sl: Dict[int, float] = {}
        for idx, (q_lf, v_lf) in harvest.items():
            # Droop parameters are structural: take each park's own values
            # from the snapshot so PF and pandapower share one source.
            rec = sgen_model.get(str(idx), {})
            slope = rec.get("qv_slope_pu") or QV_SLOPE_PU_DEFAULT
            # Same stale-snapshot problem as the deadband below: the exported
            # record carries MultiTSOConfig() defaults, not this run's droop.
            from core.actuator_bounds import (  # noqa: E402
                DER_QV_SLOPE_BY_SGEN_PU as _sl_map,
            )
            if int(idx) in _sl_map:
                slope = _sl_map[int(idx)]
            deadband = rec.get("qv_deadband_pu")
            if deadband is None:
                deadband = QV_DEADBAND_PU_DEFAULT
            # The snapshot record above is exported with MultiTSOConfig()
            # DEFAULTS, not with the deadband of the run being executed, so it
            # must not be the last word.  The runner publishes the per-park
            # values it has already written into net.sgen.qv_deadband_pu, which
            # makes both plants read ONE column; the blanket scalar stays on
            # top as an explicit diagnostic.  See
            # core.actuator_bounds.set_der_qv_deadband_override.
            from core.actuator_bounds import (  # noqa: E402
                DER_QV_DEADBAND_BY_SGEN_PU as _db_map,
                DER_QV_DEADBAND_OVERRIDE_PU as _db_ovr,
            )
            if int(idx) in _db_map:
                deadband = _db_map[int(idx)]
            if _db_ovr is not None:
                deadband = _db_ovr
            if self._qv_db_at_contingency is not None:
                # The run-up keeps the configured dead band -- that is what
                # makes it identical across legs.  Only the value installed at
                # the contingency differs.
                self._qv_target_db[int(idx)] = self._qv_db_at_contingency
            sn = self._sgen_sn[idx]
            # Capability limits from the same function the static plant
            # clips with, so both plants share one Q-capability model.
            # REEC_D declares no Qmax/Qmin -- without this the RMS park is
            # bounded only by Imax=1.3 pu and the comparison is invalid.
            # Never default the inputs: both diagrams return a ZERO-width box
            # for a missing/zero P (VDE below P/S_n = 0.1, the circle at
            # P = S_n), which would silently pin the park to constant Q.
            if "op_diagram" not in rec or "p_mw" not in rec:
                raise PFSessionError(
                    f"sgen[{idx}]: snapshot lacks 'op_diagram'/'p_mw'; the "
                    f"Q-capability box cannot be mirrored and the RMS park "
                    f"would run unbounded (REEC_D has no Qmax/Qmin)")
            q_min, q_max = _qv_capability(
                sn, str(rec["op_diagram"]), float(rec["p_mw"]),
            )
            if q_max - q_min <= 0.0:
                logger.warning(
                    "sgen[%d] (%s, P=%.1f MW, S_n=%.1f MVA): zero Q "
                    "capability -- the park cannot act as a Q actuator",
                    idx, rec["op_diagram"], float(rec["p_mw"]), sn)
            set_qv_params(self._qvpre[idx],
                          qset_pu=q_lf / sn,
                          v_anchor_pu=v_lf,
                          slope_pu=float(slope),
                          deadband_pu=float(deadband),
                          q_min_pu=q_min / sn,
                          q_max_pu=q_max / sn)
            _applied_db[int(idx)] = float(deadband)
            _applied_sl[int(idx)] = float(slope)
        logger.info("anchored %d Q(V) pre-controllers at the synced "
                    "operating point", len(harvest))
        # Record what the RMS plant actually applied.  The dead band reaches
        # this pass through three channels of differing precedence and the
        # snapshot one is stale by construction (exported with MultiTSOConfig()
        # defaults), so a run must not have to be taken on trust that the two
        # plants agree.  print, not logger: no logging handler is configured in
        # this pipeline, so logger.info output is discarded.
        print(f"  [qvpre] anchored {len(harvest)} Q(V) pre-controllers; "
              f"deadbands applied: "
              f"{sorted({round(v, 6) for v in _applied_db.values()})} pu; "
              f"droops applied: "
              f"{sorted({round(v, 6) for v in _applied_sl.values()})} pu")
    @staticmethod
    def _row_has_profile(table, idx: int, columns: Tuple[str, ...]) -> bool:
        """Whether a mirror-net row can be changed by apply_profiles."""
        if idx not in table.index:
            return False
        for column in columns:
            if column not in table.columns:
                continue
            value = table.at[idx, column]
            if value is None:
                continue
            try:
                if value != value:  # NaN without importing pandas here
                    continue
            except Exception:
                pass
            if str(value).strip():
                return True
        return False

    @staticmethod
    def _event_initial_value(obj, *attributes: str) -> float:
        """Best-effort inert payload; the slot is not due until first use."""
        for attribute in attributes:
            try:
                return float(obj.GetAttribute(attribute))
            except Exception:
                continue
        return 0.0

    def _preallocate_event_slots(self) -> None:
        """Ensure a minimum persistent capacity for every configured key.

        Existing slots discovered from earlier ComInc runs are retained.
        Missing capacity grows here before ComInc; later spillover grows
        on demand and is admitted with zero-time ComSim calls.
        """
        slots = self._event_pool_slots

        for idx, reec in self._reec.items():
            pre = self._qvpre.get(idx)
            if pre is None:
                self.ctx.ensure_param_event_capacity(
                    reec, "Qext", slots,
                    initial_value=self._event_initial_value(reec, "Qext"))
                continue
            self.ctx.ensure_param_event_capacity(
                pre, "qset", slots,
                initial_value=self._event_initial_value(pre, "params:0"))
            self.ctx.ensure_param_event_capacity(
                pre, "Vanchor", slots,
                initial_value=self._event_initial_value(pre, "params:1"))
            if self._qv_db_at_contingency is not None:
                # Fires exactly once, at the contingency -- but it still needs
                # a PRE-CREATED slot: PF admits only a couple of events
                # created after ComInc, and 44 of them at one instant would
                # hit that limit (see event_pool_slots in the driver).
                # ``db`` is index 3 of QVPRE_PARAM_ORDER.
                self.ctx.ensure_param_event_capacity(
                    pre, "db", 1,
                    initial_value=self._event_initial_value(pre, "params:3"))

        for avr in self._avr.values():
            if avr is not None:
                self.ctx.ensure_param_event_capacity(
                    avr, "usetp", slots,
                    initial_value=self._event_initial_value(avr, "usetp"))

        # Tap commands ride the same pooled EvtParam path.  Seed each slot
        # with the transformer's CURRENT tap so a slot that is never used
        # cannot yank the tap to zero, and so inc(x1)=ntapcmd initialises the
        # block where the plant already sits.
        for (table, idx), ctrl in self._tapctrl.items():
            shadow = self._tap2w if table == "trafo" else self._tap3w
            self.ctx.ensure_param_event_capacity(
                ctrl, "ntapcmd", slots,
                initial_value=float(shadow.get(idx, 0)))

        profiled_loads = 0
        if self._preallocate_profiles:
            for idx, load in self._loads.items():
                if not self._row_has_profile(
                        self.net.load, idx, ("profile_p", "profile_q")):
                    continue
                self.ctx.ensure_load_event_capacity(load, slots)
                profiled_loads += 1
            for idx, wgo in self._wgo.items():
                if wgo is None or idx not in self.net.sgen.index:
                    continue
                self.ctx.ensure_param_event_capacity(
                    wgo, "Pref_in", slots,
                    initial_value=self._event_initial_value(wgo, "Pref_in"))

        stats = self.ctx.event_pool_stats()
        logger.info(
            "persistent RMS event pool ready before ComInc: "
            "%d parameter + %d load slots (%d slots/key, %d profiled loads)",
            stats["param_total"], stats["load_total"], slots, profiled_loads)


    def apply_u(self, writes: ActuatorWrites) -> None:
        """Schedule this dispatch's events at the paused simulation time.

        Continuous references land one RMS step after ``t`` (parameter
        events); tap moves land ``TAP_MECH_DELAY_S`` later, sequential
        steps spaced by the same delay.  The mirror net's *input* columns
        are updated in the same call so discrete states read back
        consistently (``net.trafo*.tap_pos``, ``net.shunt.step``,
        ``net.gen.vm_pu``).
        """
        t_evt = self.t + self._EVENT_EPS_S
        net = self.net

        for idx, q in writes.der_q_set_mvar.items():
            idx = int(idx)
            if idx not in self._reec:
                raise PFSessionError(f"DER write: unknown sgen index {idx}")
            q_pu = float(q) / self._sgen_sn[idx]
            pre = self._qvpre.get(idx)
            if pre is None:
                # Bare converter: constant Q between dispatches.
                self._add_param(self._reec[idx], "Qext", q_pu, t_evt)
                write_der_q_set(net, idx, float(q))
                continue
            # Re-anchored Q(V): mirror the static plant's write order --
            # V_ref is re-anchored to the voltage measured *now* (the paused
            # RMS state is the plant's own measurement), then the OFO command
            # is applied.  Both plants therefore droop about the same anchor.
            v_anchor = self._park_anchor_voltage(idx)
            self._add_param(pre, "qset", q_pu, t_evt)
            self._add_param(pre, "Vanchor", v_anchor, t_evt)
            write_der_q_set(net, idx, float(q))
            # Force the mirror onto the plant's own anchor so the two never
            # drift apart through differing measurement sources.
            net.sgen.at[idx, "qv_vref_anchor_pu"] = v_anchor

        for idx, v in writes.gen_v_pu.items():
            idx = int(idx)
            avr = self._avr.get(idx)
            if idx not in self._machines:
                raise PFSessionError(f"V-ref write: unknown gen index {idx}")
            if avr is None:
                msg = (f"gen[{idx}] ({self._machines[idx].loc_name}): no AVR "
                       f"block -- V-ref not actuatable in RMS")
                if self.on_missing_avr == "raise":
                    raise PFSessionError(msg)
                logger.warning("%s (skipped)", msg)
                self.skipped_writes.append(("gen_v_pu", idx, float(v)))
                continue
            self._add_param(avr, "usetp", float(v), t_evt)
            net.gen.at[idx, "vm_pu"] = float(v)

        for idx, tap in writes.tap_2w.items():
            self._dispatch_taps("trafo", int(idx), int(round(tap)))
        for idx, tap in writes.tap_3w.items():
            self._dispatch_taps("trafo3w", int(idx), int(round(tap)))

        for idx, step in writes.shunt_step.items():
            idx = int(idx)
            sh = self._shunts.get(idx)
            if sh is None:
                raise PFSessionError(f"shunt write: unknown index {idx}")
            delta = int(round(step)) - self._shunt_step[idx]
            if delta:
                # MSC/MSR steps are the last actuator still on ``EvtTap``,
                # and EvtTap is UNRELIABLE after the first event of a
                # calculation: measured 2026-07-31 on SH_MSC_DSO_1_s0, four
                # commands (+1,+1,-1,-1) at 20.5/60.5/100.5/140.5 s landed at
                # 20.53 / 120.50 / 160.51 / never -- i.e. exactly +60 s late
                # from the second onward, one lost, final step +1 instead of 0.
                # Same signature the OLTC taps had before they moved to the
                # DSL path (pf/tap_ctrl.py); the shunt has NO documented RMS
                # input signal (TechRef_Shunt S6 lists EMT state variables
                # only), so the same escape is not available.
                #
                # Loud, once per run, and recorded in provenance: a run whose
                # integrator moves a bank more than once is not trustworthy on
                # its shunt trajectory.
                # MSC/MSR are the only actuator still dispatched by
                # ``EvtTap``.  That was unreliable until 2026-07-31, when the
                # cause was found: PF applies event times modulo a 60 s window
                # once the calculation runs.  ``add_tap_event`` now folds the
                # time into the current window (EVENT_WINDOW_S), and
                # ``_verify_shunt_steps`` asserts after every advance that the
                # plant actually holds what was commanded -- an EvtTap that
                # does not land is otherwise SILENT, because the shadow store
                # and the mirror net both keep the commanded value.
                self._shunt_events += 1
                self.ctx.add_tap_event(sh, delta, t_evt, seq=self._n_events)
                self._n_events += 1
                self._shunt_step[idx] += delta
            net.shunt.at[idx, "step"] = int(round(step))

    #: Contingencies this plant can translate into PowerFactory events.
    #: Anything else must raise rather than run a co-simulation in which the
    #: two plants see different topologies.
    _SUPPORTED_CONTINGENCIES = (("gen", "trip"), ("load", "q_step"))

    @classmethod
    def supports_contingency(cls, event) -> bool:
        """Whether ``event`` can be delivered to the RMS plant."""
        return (str(getattr(event, "element_type", "")),
                str(getattr(event, "action", ""))) in cls._SUPPORTED_CONTINGENCIES

    def _switch_qv_deadband(self, t_evt: float) -> None:
        """Install ``qv_deadband_at_contingency`` on every park at ``t_evt``.

        Called from :meth:`apply_contingency` for EVERY contingency type, so
        the dead-band x droop comparison behaves the same whether the
        disturbance is a machine trip or a load step.
        """
        if self._qv_db_at_contingency is None or not self._qv_target_db:
            return
        for i, pre in self._qvpre.items():
            target = self._qv_target_db.get(int(i))
            if target is None:
                continue
            self._add_param(pre, "db", float(target), t_evt)
        print(f"  [qvpre] Q(V) dead band re-installed at the disturbance "
              f"(t={t_evt:.1f}s) for {len(self._qv_target_db)} parks: "
              f"{self._qv_db_at_contingency} pu (the run-up kept the "
              f"configured dead band)")

    def apply_contingency(self, event, gen_trafo_map=None) -> None:
        """Translate a scheduled contingency into a PowerFactory event.

        The caller has already applied the event to the mirror net (which is
        this plant's ``self.net``), so only the PF side is done here.

        Two contingencies are supported: a synchronous-machine **trip**
        (``EvtOutage``) and a reactive-power **q_step** on an existing load
        (``EvtLod``).  A line trip, a load connect/shed or any ``restore``
        raises: the mirror would show the new topology while PF kept the old
        one, and every measurement afterwards would be a comparison between
        two different networks.  ``EvtOutage`` could express those too, but
        none of them has been verified against this adapter and an unverified
        topology change is worse than an explicit refusal.

        ``q_step`` deliberately steps an EXISTING load rather than connecting
        a dormant one: this plant is built from the snapshot, so a row created
        in the mirror by ``prepare_load_contingencies`` would have no
        PowerFactory counterpart.  It also leaves the topology untouched, so
        unlike a machine trip it does not by itself invalidate Gate E.

        The machine transformer is tripped as well when ``gen_trafo_map``
        provides it, because ``experiments.helpers._apply_contingency`` trips
        it in the mirror -- leaving it energised in PF would leave the two
        plants differing by one transformer's magnetising branch.

        ``outserv`` is deliberately NOT written: ``EvtOutage`` does not update
        it either, and writing the input attribute during an active
        calculation does not take effect (the same reason taps go through
        ``EvtTap``).  The outage is observable only through the disappearance
        of the element's ``m:`` result variables, which ``_read_m`` handles.
        """
        if not self.supports_contingency(event):
            raise NotImplementedError(
                f"RMS plant cannot apply contingency "
                f"{getattr(event, 'element_type', '?')}/"
                f"{getattr(event, 'action', '?')}; supported: "
                + ", ".join(f"{t}/{a}"
                            for t, a in self._SUPPORTED_CONTINGENCIES))

        idx = int(event.element_index)
        t_evt = self.t + self._EVENT_EPS_S

        # Switch the local Q(V) layer on at the SAME instant as the
        # disturbance, whichever kind it is.  This MUST come before the
        # per-type branches: the load branch returns early, and while this
        # block sat after it a load contingency silently ran with the droop
        # never switched -- measured 2026-08-06, the delta = 0.01 leg came
        # out bit-identical to the no-droop leg.
        self._switch_qv_deadband(t_evt)

        # ── reactive-power step on an existing load ───────────────────────
        if str(getattr(event, "element_type", "")) == "load":
            load = self._loads.get(idx)
            if load is None:
                raise PFSessionError(
                    f"contingency targets load[{idx}] but no ElmLod is mapped "
                    f"for it; the mirror would step a load PF does not know")
            d_q = float(event.q_mvar)
            # EvtLod percentages are additive on the load's ORIGINAL value,
            # not its present one (see ScreeningContext.add_load_event), so
            # the base must come from base_q_mvar -- the mirror's live
            # q_mvar already carries this very step, applied by the caller.
            base = None
            if "base_q_mvar" in self.net.load.columns:
                b = float(self.net.load.at[idx, "base_q_mvar"])
                base = b if abs(b) > 1e-9 else None
            if base is None:
                base = float(self.net.load.at[idx, "q_mvar"]) - d_q
            if abs(base) < 1e-9:
                raise PFSessionError(
                    f"load[{idx}] has a zero reactive base, so an EvtLod "
                    f"percentage cannot express a {d_q:+.1f} Mvar step; give "
                    f"the load a non-zero q_mvar or use a different mechanism")
            d_q_pct = 100.0 * d_q / base
            # add_load_step_event, NOT add_load_event: the latter re-arms a
            # pooled slot and writes an absolute time, and measured
            # 2026-08-06 that combination never fires for a single mid-run
            # step -- the RMS plant stayed bit-identical to its twin.
            self.ctx.add_load_step_event(load, 0.0, d_q_pct, t_evt,
                                         seq=self._n_events)
            self._n_events += 1
            bus = int(self.net.load.at[idx, "bus"])
            print(f"  [rms-contingency] EvtLod on {load.loc_name} "
                  f"(load[{idx}] @ bus {bus}) armed at RMS t={t_evt:.1f}s: "
                  f"dQ {d_q:+.1f} Mvar = {d_q_pct:+.1f} % of the "
                  f"{base:.1f} Mvar base, dP 0 %")
            logger.info("RMS contingency: Q step %+.1f Mvar (%.1f %%) on "
                        "load[%d] at t=%.1f s", d_q, d_q_pct, idx, t_evt)
            return

        mach = self._machines.get(idx)
        if mach is None:
            raise PFSessionError(
                f"contingency targets gen[{idx}] but no ElmSym is mapped for "
                f"it; the mirror would trip a machine PF does not know")

        self.ctx.add_outage_event(mach, t_evt, seq=self._n_events)
        self._n_events += 1
        targets = [mach.loc_name]

        if gen_trafo_map and idx in gen_trafo_map:
            t_idx = int(gen_trafo_map[idx])
            tr = self._tr2.get(t_idx)
            if tr is None:
                raise PFSessionError(
                    f"machine trafo[{t_idx}] of gen[{idx}] has no ElmTr2 "
                    f"handle; the mirror trips it but PF would keep it in "
                    f"service")
            self.ctx.add_outage_event(tr, t_evt, seq=self._n_events)
            self._n_events += 1
            targets.append(tr.loc_name)

        # The RMS event time is NOT the nominal event time, and the difference
        # is not an error.  At the dispatch step labelled t = T the plant has
        # simulated only to T - dt and advances to T at the end of the step, so
        # an event armed at ``self.t + EPS`` fires INSIDE the interval whose
        # measurements are reported for step T -- which is exactly when the
        # static plant's mutate-and-resolve makes the outage visible.  This is
        # the same convention apply_exogenous uses for profiles; scheduling at
        # the nominal absolute time instead would push the outage into the
        # NEXT step's measurements and put the two legs one interval apart.
        nominal = getattr(event, "effective_time_s", float("nan"))
        print(f"  [rms-contingency] EvtOutage on {', '.join(targets)} "
              f"(gen[{idx}]) armed at RMS t={t_evt:.1f}s, i.e. within the "
              f"interval reported for the nominal trip at t={nominal:g}s")
        logger.info("RMS contingency: outage of %s armed at t=%.1f s "
                    "(nominal %s s)", targets, t_evt, nominal)

    def apply_exogenous(self, profiles, t) -> None:
        """Push the profile operating point for wall-clock ``t`` into PF.

        With pre-ComInc playback installed, Python updates only the
        pandapower measurement mirror: ElmFile sources already drive the
        physical load P/Q and DER P on the RMS time axis.  The selectable
        legacy path uses two event mechanisms:

        * **loads** -- ``EvtLod`` incremental step, in *percent of the
          present value*.  The absolute profile factor is therefore tracked
          in ``_load_factor_p/q`` and converted to a delta.
        * **DER active power** -- ``EvtParam`` on the WECC weak-grid block's
          ``Pref_in``, which is absolute in pu of the park rating, so no
          shadow is needed.
        """
        from core.profiles import apply_profiles as _apply_static

        net = self.net
        if self._profile_playback is not None:
            _apply_static(net, profiles, t)
            logger.info(
                "profiles at t=%.1fs: ElmFile drives PF; mirror updated only",
                self.t)
            return

        t_evt = self.t + self._EVENT_EPS_S

        # Mirror net first: this is what the controller reads, and it gives
        # the post-profile p_mw/q_mvar the PF deltas are computed against.
        prev_p = {i: float(net.load.at[i, "p_mw"]) for i in net.load.index}
        prev_q = {i: float(net.load.at[i, "q_mvar"]) for i in net.load.index}
        _apply_static(net, profiles, t)

        n_lod = 0
        for idx, obj in self._loads.items():
            if idx not in net.load.index:
                continue
            new_p = float(net.load.at[idx, "p_mw"])
            new_q = float(net.load.at[idx, "q_mvar"])
            dp = self._percent_delta(prev_p.get(idx, new_p), new_p)
            dq = self._percent_delta(prev_q.get(idx, new_q), new_q)
            if dp == 0.0 and dq == 0.0:
                continue
            self.ctx.add_load_event(obj, dp, dq, t_evt)
            self._n_events += 1
            n_lod += 1

        n_der = 0
        for idx, wgo in self._wgo.items():
            if wgo is None or idx not in net.sgen.index:
                continue
            sn = self._sgen_sn[idx]
            if sn <= 0.0:
                continue
            p_ref_pu = float(net.sgen.at[idx, "p_mw"]) / sn
            self._add_param(wgo, "Pref_in", p_ref_pu, t_evt)
            n_der += 1
        logger.info("profiles at t=%.1fs: %d load events, %d DER Pref writes",
                    self.t, n_lod, n_der)

    def _live_refresh_every(self) -> int:
        """Refresh cadence, overridable live by a ``GUI_EVERY`` file.

        The plot cadence is fixed at construction, but the per-interval cost
        of a profile run is dominated by the simulation, not the redraw, so
        watching usually wants a *smaller* cadence than a batch run.  A file
        named ``GUI_EVERY`` next to the off-flag, containing an integer, lets
        that be tuned mid-run without a restart -- the same live-control
        pattern as the pause/resume sentinel.
        """
        if self._gui_off_flag is None:
            return self._gui_refresh_every
        path = os.path.join(os.path.dirname(self._gui_off_flag), "GUI_EVERY")
        try:
            with open(path, encoding="ascii") as fh:
                return max(1, int(fh.read().strip()))
        except (OSError, ValueError):
            return self._gui_refresh_every

    @staticmethod
    def _percent_delta(old: float, new: float) -> float:
        """Percent change ``old -> new`` for an EvtLod incremental step.

        Returns 0.0 when ``old`` is ~0: a percentage cannot express a step
        away from zero, and PF would apply a meaningless multiplier.  Loads
        that start at zero are therefore left alone rather than silently
        mis-scaled -- rare here (60 of 113 loads carry no profile at all).
        """
        if abs(old) < 1e-9:
            return 0.0
        return (new / old - 1.0) * 100.0

    def _dispatch_taps(self, table: str, idx: int, target: int) -> None:
        """Command an absolute tap position through the DSL tap controller.

        ``EvtTap`` is NOT used here.  As dispatched (absolute event times) it
        drops or defers every tap after the first -- root cause found
        2026-07-31: PF applies event times modulo a 60 s window once the
        calculation runs, so ``te`` fires at ``te mod 60``.  The DSL path
        avoids the question entirely and adds a real mechanical delay.  DIgSILENT ``TechRef_*-W-Transformer`` S5 is explicit
        that RMS ignores tap-controller definitions and that a dynamic model
        driving the ``nntapin`` input is the supported mechanism, which is
        what ``pf/tap_ctrl.py`` builds.  The commanded position is then just
        another DSL parameter, dispatched on the pooled ``EvtParam`` path
        that lands on time every interval (2026-07-30/31).

        ``ntapcmd`` is the ABSOLUTE position, so one event per dispatch
        replaces the previous one-event-per-step loop.
        """
        objs, shadow = ((self._tr2, self._tap2w) if table == "trafo"
                        else (self._tr3, self._tap3w))
        obj = objs.get(idx)
        if obj is None:
            raise PFSessionError(f"tap write: unknown {table} index {idx}")
        if target != shadow[idx]:
            ctrl = self._tapctrl.get((table, idx))
            if ctrl is None:
                raise PFSessionError(
                    f"tap write: {table}[{idx}] ({obj.loc_name}) has no "
                    f"TAPCTRL composite -- run pf.tap_ctrl rollout first. "
                    f"Without it the tap command never reaches the plant.")
            self.ctx.add_param_event(ctrl, "ntapcmd", float(target),
                                     self.t + self._EVENT_EPS_S)
            self._n_events += 1
        shadow[idx] = target
        col = "trafo" if table == "trafo" else "trafo3w"
        getattr(self.net, col).at[idx, "tap_pos"] = target

    def _verify_shunt_steps(self) -> None:
        """Assert the plant executed the MSC/MSR steps we commanded.

        MSC/MSR are the only actuator still dispatched by ``EvtTap``, and an
        EvtTap that does not land is SILENT: the shadow store and the mirror
        net both keep the commanded value, so a wrong trajectory looks
        perfectly consistent from the Python side.  That is exactly how the
        tap defect hid for as long as it did.

        ``c:ncapa`` is the position PF actually holds, so comparing it against
        the shadow turns a silent divergence into an immediate stop.  Steps
        are instantaneous (no mechanical lag in the shunt model), so after a
        completed advance the two must agree exactly.
        """
        if not self._shunt_events:
            return                              # nothing dispatched yet
        for idx, sh in self._shunts.items():
            want = int(self._shunt_step[idx])
            try:
                got = int(round(float(sh.GetAttribute(SHUNT_STEP_VAR))))
            except Exception:                   # noqa: BLE001
                continue                        # not calc-relevant; skip
            if got != want:
                raise PFSessionError(
                    f"shunt {sh.loc_name}: commanded step {want} but the "
                    f"plant holds {got} at t={self.t:.1f}s. An EvtTap did not "
                    f"land -- see EVENT_WINDOW_S in pf/screening.py and "
                    f"docs/daily_log/07_2026/2026-07-31_rms_tap_control_gate_e_result.md. "
                    f"Results from this run are invalid.")

    def advance(self, duration_s: float) -> None:
        """Continue the RMS simulation by ``duration_s`` seconds.

        Fired objects cannot be reused inside the active calculation. They
        are made inert after firing and become reusable after the next
        ComInc. Newly grown slots are admitted through zero-time ComSim calls
        before the real advance, without changing the simulated clock.

        ``duration_s <= 0`` is a no-op (the runner's static plant re-solves
        there; the RMS state simply persists until the next real advance).
        """
        if duration_s <= 0:
            return
        admission_calls = self.ctx.admit_new_events(self.t, self._EVENT_EPS_S)
        if admission_calls:
            logger.debug("admitted new RMS events with %d advancing barriers",
                         admission_calls)
        self.t += float(duration_s)
        self.ctx.simulate(self.t)
        self._verify_shunt_steps()
        # Live plot: an API-driven ComSim writes into ElmRes but never asks
        # the desktop to repaint, so a visible plot page stays empty until
        # the run ends.  DoAutoScale() forces a redraw with the data written
        # so far -- the curves then advance one dispatch interval at a time.
        #
        # Refresh ALL qOFO pages, so switching tabs mid-run always shows
        # current curves (not just the fronted one).  A PF redraw costs more
        # as result rows accumulate, and this multiplies that by the page
        # count, so it is gated by ``gui_refresh_every`` (refresh every Nth
        # interval) and by the ``gui_off_flag`` sentinel -- either lets a
        # long run trade live-plot frequency for speed without stopping.
        self._gui_interval += 1
        if self._gui_pages:
            # Strongest off-switch: hide the desktop window entirely (and skip
            # the redraw with it).  Toggled by a HIDE_GUI sentinel, both ways.
            if self._gui_folder is not None:
                want_hidden = os.path.exists(
                    os.path.join(self._gui_folder, "HIDE_GUI"))
                if want_hidden != self._gui_hidden:
                    try:
                        (self.app.Hide if want_hidden else self.app.Show)()
                        self._gui_hidden = want_hidden
                        logger.info("HIDE_GUI %s -> desktop %s",
                                    "present" if want_hidden else "removed",
                                    "HIDDEN" if want_hidden else "SHOWN")
                    except Exception as exc:       # noqa: BLE001
                        logger.warning("Hide/Show failed: %s", exc)
            if self._gui_hidden:
                # Window gone -> no point redrawing it; this is the maximum
                # speed-up.  Everything resumes when HIDE_GUI is removed.
                return
            # Live off-switch: engine mode gives no way to disable the plot on
            # a running process from outside (a second PF session would kill
            # the run), so the plant watches a sentinel file each interval.
            # Create the file to pause the refresh, delete it to resume --
            # the page list is kept either way, so it toggles both directions
            # without stopping the simulation.
            gui_off = bool(self._gui_off_flag
                           and os.path.exists(self._gui_off_flag))
            if gui_off != self._gui_was_off:
                logger.info("GUI-off flag %s -> live refresh %s",
                            self._gui_off_flag,
                            "PAUSED" if gui_off else "RESUMED")
                self._gui_was_off = gui_off
            every = self._live_refresh_every()
            if not gui_off and self._gui_interval % every == 0:
                for _page in list(self._gui_pages):
                    try:
                        _page.DoAutoScale()
                    except Exception as exc:       # noqa: BLE001
                        logger.warning("GUI refresh failed on %r, dropping "
                                       "it: %s",
                                       getattr(_page, "loc_name", _page), exc)
                        self._gui_pages.remove(_page)
        # Python-side live view: harvest what has been simulated so far and
        # redraw.  Disabled on first failure so a plotting problem can never
        # take down a run.
        if self._live_view is not None:
            try:
                traj = self.harvest_trajectories(
                    since_s=0.0,
                    stride=self._live_view_stride,
                    labels=self._live_view.matches,
                )
                self._live_view.update(traj)
            except Exception as exc:               # noqa: BLE001
                logger.warning("live view failed, disabling: %s", exc)
                self._live_view = None

    def read_y(self):
        """Refresh the mirror net's result tables from the paused state."""
        net = self.net

        bus_idx = list(self._terms)
        net.res_bus.loc[bus_idx, "vm_pu"] = [
            float(self._terms[i].GetAttribute("m:u")) for i in bus_idx]
        net.res_bus.loc[bus_idx, "va_degree"] = [
            float(self._terms[i].GetAttribute("m:phiu")) for i in bus_idx]

        # Controller measurements include line current and inter-zone P/Q.
        # Refresh both ends from the RMS state; retaining any pandapower
        # values here would mix two plants inside one controller sample.
        for i, line in self._lines.items():
            i_from = float(line.GetAttribute("m:I:bus1"))
            i_to = float(line.GetAttribute("m:I:bus2"))
            p_from = float(line.GetAttribute("m:P:bus1"))
            p_to = float(line.GetAttribute("m:P:bus2"))
            q_from = float(line.GetAttribute("m:Q:bus1"))
            q_to = float(line.GetAttribute("m:Q:bus2"))
            net.res_line.at[i, "i_from_ka"] = i_from
            net.res_line.at[i, "i_to_ka"] = i_to
            net.res_line.at[i, "p_from_mw"] = p_from
            net.res_line.at[i, "p_to_mw"] = p_to
            net.res_line.at[i, "q_from_mvar"] = q_from
            net.res_line.at[i, "q_to_mvar"] = q_to
            net.res_line.at[i, "pl_mw"] = p_from + p_to
            net.res_line.at[i, "ql_mvar"] = q_from + q_to
            rating = (
                float(net.line.at[i, "max_i_ka"])
                * float(net.line.at[i, "df"])
                * float(net.line.at[i, "parallel"])
            )
            net.res_line.at[i, "loading_percent"] = (
                100.0 * max(i_from, i_to) / rating
                if rating > 0.0 else float("nan")
            )

        for i, tr in self._tr3.items():
            net.res_trafo3w.at[i, "p_hv_mw"] = float(
                tr.GetAttribute("m:P:bushv"))
            net.res_trafo3w.at[i, "q_hv_mvar"] = float(
                tr.GetAttribute("m:Q:bushv"))

        for i, g in self._sgens.items():
            net.res_sgen.at[i, "p_mw"] = self._read_m(g, "m:P:bus1", ("sgen", i))
            net.res_sgen.at[i, "q_mvar"] = self._read_m(g, "m:Q:bus1", ("sgen", i))

        for i, m in self._machines.items():
            net.res_gen.at[i, "p_mw"] = self._read_m(m, "m:P:bus1", ("gen", i))
            net.res_gen.at[i, "q_mvar"] = self._read_m(m, "m:Q:bus1", ("gen", i))

        # Loads: without this refresh res_load stays frozen at the init
        # power-flow value while the profile moves the actual load, so every
        # load-based diagnostic (dso_group_load_*, residual_load, and any
        # RMS-vs-static load comparison) reads stale numbers.  Sign: a PF
        # load draws power, so its m:P is positive into the load, matching
        # pandapower's res_load convention.
        for i, ld in self._loads.items():
            if i in net.res_load.index:
                net.res_load.at[i, "p_mw"] = self._read_m(
                    ld, "m:P:bus1", ("load", i))
                net.res_load.at[i, "q_mvar"] = self._read_m(
                    ld, "m:Q:bus1", ("load", i))

        return net

    def _read_m(self, obj, attr: str, key, default: float = 0.0) -> float:
        """Read a PF result variable, tolerating an element that has left service.

        PowerFactory removes an element's ``m:`` result variables once it is
        taken out of service, so after an ``EvtOutage`` every read of the
        tripped element raises ``AttributeError`` (verified 2026-07-28: an
        outage of ``G 03`` crashed ``read_y`` at the machine loop). Any N-1
        study therefore needs this path to survive a missing result variable.

        The tolerance is deliberately narrow, because a blanket ``except``
        would turn a mistyped attribute name into a silent zero:

        * before the first successful read of a given element nothing can yet
          have been switched out, so a failure there is a genuine defect and is
          re-raised;
        * afterwards the element is recorded as out of service, reported once
          at WARNING level, and read as ``default`` until it returns.

        Elements that come back into service are dropped from the set again on
        the next successful read, so a restoration event is handled too.
        """
        try:
            value = float(obj.GetAttribute(attr))
        except Exception as exc:
            if key not in self._seen_in_service:
                raise PFSessionError(
                    f"{key}: {attr} unavailable on the first read, before any "
                    f"switching could have occurred -- this is a handle or "
                    f"attribute-name defect, not an outage ({exc})") from exc
            if key not in self._out_of_service:
                self._out_of_service.add(key)
                logger.warning(
                    "%s left service (%s unavailable); reading %s as %.1f "
                    "until it returns", key, attr, attr, default)
            return float(default)
        self._seen_in_service.add(key)
        self._out_of_service.discard(key)
        return value

    @property
    def out_of_service(self) -> set:
        """Elements whose result variables have disappeared (tripped)."""
        return set(self._out_of_service)

    # ------------------------------------------------------------------
    #  Trajectory harvest (per dispatch chunk)
    # ------------------------------------------------------------------
    def harvest_trajectories(self, since_s: float = 0.0, stride: int = 5,
                             labels=None,
                             ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """{label: (t[], y[])} for monitored outputs, t >= since_s.

        The shared ``ElmRes`` grows across chunked ``advance`` calls;
        ``since_s`` restricts the returned window.  ``labels`` (set of
        monitor labels, or a predicate) limits the per-cell reads -- the
        dominant cost on long runs.
        """
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for obj, var, label in self._monitors:
            if labels is not None:
                keep_label = (labels(label) if callable(labels)
                              else label in labels)
                if not keep_label:
                    continue
            t, y = self.ctx.read(obj, var, stride=stride)
            ta, ya = np.asarray(t), np.asarray(y)
            keep = ta >= since_s
            out[label] = (ta[keep], ya[keep])
        return out

    def harvest_trajectories_bulk(
        self,
        csv_path: Union[str, Path],
        since_s: float = 0.0,
        stride: int = 5,
        labels=None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Export ``ElmRes`` once and sample monitored columns offline.

        This is the end-of-run path for long RMS traces.  It avoids the
        per-cell ``ElmRes.GetValue`` calls used by
        :meth:`harvest_trajectories`, which remains available for the small
        rolling windows used by the live plot.
        """

        path = export_comres_csv(self.app, self.ctx.res, csv_path)
        return load_comres_trajectories(
            path,
            self._monitors,
            since_s=since_s,
            stride=stride,
            labels=labels,
        )

    # ------------------------------------------------------------------
    def _add_param(self, target, variable: str, value: float,
                   t_evt: float) -> None:
        self.ctx.add_param_event(target, variable, value, t_evt)
        self._n_events += 1
