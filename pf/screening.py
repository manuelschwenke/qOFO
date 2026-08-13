"""
pf/screening.py
===============
Phase-5 RMS screening battery for the IEEE39_qOFO PowerFactory model
(docs/RMS_IEEE39_PowerFactory_Build_Plan.md, Gate D).

Three subcommands, all run in the ``02_RMS_CoSim`` study case with the
layered ``wind_replace`` + ``full`` model active and the load flow aligned
to the validated ZIP parity operating point:

* ``flat``  -- 60 s no-event RMS run; asserts max bus-voltage drift
               < 1e-4 pu (the model is at a genuine equilibrium).
* ``modal`` -- ``ComMod`` eigenvalue analysis; markdown table of every mode
               with damping ratio, frequency, 2 %-band settling time
               ``T_s = 4/|Re λ|``, sorted by settling time descending.
               **No participation factors** -- an earlier version of this
               docstring claimed them and the code never computed them
               (corrected 2026-08-04). Consequently the table supports NO
               statement about what KIND of mode any row is: calling a
               0.78--1.34 Hz row "electromechanical" is an inference from
               frequency alone, and this model carries WECC converter
               control loops and AVRs that populate the same band. Add
               ``iPart``/eigenvector extraction before any modal row is
               given a physical label.
* ``steps`` -- per-actuator worst-case single-dispatch steps: DER Q via
               ``EvtParam`` on ``REEC_D.Qext``, machine AVR V-ref via
               ``EvtParam`` on the ``avr_IEEET1`` ``usetp`` signal, and
               OLTC taps / MSC steps via ``EvtTap`` (relative ``ntap``;
               taps carry the 5 s mechanical delay, incl. the sequential
               2-tap case).  Event dispatched at t = 5 s, 45 s run;
               settling of every monitored controlled output is measured
               from the dispatch instant.

Monitored controlled outputs: the 12 coupler-3W interface Q flows (q_STS),
all TN and DSO-PCC bus voltages, and machine speeds.

Every run writes into ``results/screening/{snapshot}/{timestamp}/``.

Author: Manuel Schwenke / Claude Code (2026-07-20)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pf.pf_parity import PARITY_LDF_SETTINGS  # noqa: E402
from pf.wecc_apply import QVPRE_ELEMENT_NAME  # noqa: E402
from pf.session import (  # noqa: E402
    DEFAULT_PROJECT_PATH,
    PFSessionError,
    connect,
    deactivate_variations_except,
    get_all,
    set_variation_active,
)

RMS_STUDY_CASE = "02_RMS_CoSim"
WIND_REPLACE_VARIATION = "wind_replace"
FULL_VARIATION = "full"

#: RMS integration + output step [ms] (PF unit).  10 ms = the standard
#: electromechanical step; 302 rows / 3 s, fast.
RMS_STEP_MS = 10.0

#: RMS-readable discrete actuator positions.  NOT the input attributes
#: (``nntap`` / ``n3tap_h`` / ``ncapa``): those are read at initialisation and
#: are never updated by ``EvtTap``, which is exactly why PowerFactoryPlant
#: keeps a shadow of the discrete state.  Confirmed against a live result
#: file 2026-07-21 (scratchpad probe_tapvars.py).
TAP_VAR_2W = "c:nntap"
TAP_VAR_3W = "c:n3tap_h"
SHUNT_STEP_VAR = "c:ncapa"

#: Far-future timestamp for pre-ComInc one-shot event slots.  Live-probed
#: 2026-07-22: unused slots can be moved into the active horizon.
EVENT_INERT_TIME_S = 1.0e9

#: Names owned by the persistent qOFO event pool.
PERSISTENT_EVENT_PREFIXES = {
    "EvtParam": "qofo_pool_p_",
    "EvtLod": "qofo_pool_l_",
    "EvtTap": "qofo_pool_t_",
}

#: Conservative lower bound below the live-probed ~90 newly admitted event
#: objects per ComSim.Execute. Zero-time admission calls use this batch size.
EVENT_ADMISSION_BATCH = 64

#: Flat-run drift gate [pu].
FLAT_DRIFT_TOL = 1e-4


# =====================================================================
#  Screening context
# =====================================================================

class ScreeningContext:
    """RMS study-case session: full model active, parity-aligned init."""

    def __init__(self, app, *, verbose: bool = True,
                 require_preallocated_events: bool = False,
                 persistent_event_pool: bool = False,
                 rms_step_ms: float = RMS_STEP_MS,
                 rms_step_max_ms: Optional[float] = None,
                 adaptive_step: bool = False):
        self.app = app
        self.verbose = verbose
        #: Integration step [ms].  With ``adaptive_step`` this is the SMALLEST
        #: step PF may take; otherwise it is the fixed step.
        self.rms_step_ms = float(rms_step_ms)
        #: Largest step [ms] PF may take when ``adaptive_step`` is on.
        self.rms_step_max_ms = (RMS_STEP_MS if rms_step_max_ms is None
                                else float(rms_step_max_ms))
        #: PF automatic step-size adaptation (ComInc ``iopt_adapt``).  Off by
        #: default, which is what every run before 2026-08-06 used.
        self.adaptive_step = bool(adaptive_step)
        if self.adaptive_step and self.rms_step_max_ms < self.rms_step_ms:
            raise ValueError(
                f"rms_step_max_ms ({self.rms_step_max_ms}) must be >= "
                f"rms_step_ms ({self.rms_step_ms})")
        self.require_preallocated_events = bool(require_preallocated_events)
        self.persistent_event_pool = bool(persistent_event_pool)
        self._param_event_slots: Dict[Tuple[str, str], List[Any]] = {}
        self._param_event_next: Dict[Tuple[str, str], int] = {}
        self._load_event_slots: Dict[str, List[Any]] = {}
        self._load_event_next: Dict[str, int] = {}
        self._tap_event_slots: Dict[str, List[Any]] = {}
        self._tap_event_next: Dict[str, int] = {}
        self._pool_serial = {
            class_name: 0 for class_name in PERSISTENT_EVENT_PREFIXES
        }
        self._calculation_active = False
        self._new_events_pending_admission = 0
        self._persistent_events_created = 0
        self._persistent_events_discovered = 0
        self._admission_executes = 0
        self._retired_events = 0
        self._armed_persistent_events: List[Tuple[float, Any]] = []

        deactivate_variations_except(app, keep=None)
        set_variation_active(app, WIND_REPLACE_VARIATION, True)
        set_variation_active(app, FULL_VARIATION, True)

        # Grid activation is stored per study case: the four DSO ElmNet
        # folders were activated only in 01_LDF_Parity at sync time, so in
        # the RMS study case they must be activated too, otherwise the full
        # DSO underlay is silently absent and screening runs the wind_replace
        # model instead (discovered 2026-07-20).
        netdat = app.GetProjectFolder("netdat")
        for grid in netdat.GetContents("*.ElmNet"):
            if grid.loc_name.startswith("DSO_") and not grid.IsCalcRelevant():
                if grid.Activate():
                    raise PFSessionError(f"Activate DSO grid {grid.loc_name} failed")

        self.inc = app.GetFromStudyCase("ComInc")
        self.sim = app.GetFromStudyCase("ComSim")
        self.res = app.GetFromStudyCase("ElmRes")
        self.evt_folder = self.inc.GetAttribute("p_event")
        if self.evt_folder is None:
            raise PFSessionError("ComInc has no simulation-event folder")

        # Align the RMS init load flow with the ZIP parity operating point.
        ldf = app.GetFromStudyCase("ComLdf")
        for k, v in PARITY_LDF_SETTINGS.items():
            ldf.SetAttribute(k, v)

        # Standard RMS (balanced electromechanical) configuration.
        self.inc.SetAttribute("iopt_sim", "rms")
        self.inc.SetAttribute("iopt_net", "sym")
        self.inc.SetAttribute("dtgrd", self.rms_step_ms)
        # Automatic step-size adaptation.  The converter models carry dynamics
        # far faster than the 10 ms default (REGC_C Te = 0.1 ms, PLL
        # Kipll = 1400), so a fixed 10 ms step under-integrates them and the
        # post-switching trajectory shows numerical zig-zag.  With adaptation
        # PF shortens the step only where its error tolerance demands it,
        # instead of paying for the small step over the whole horizon.
        self.inc.SetAttribute("iopt_adapt", 1 if self.adaptive_step else 0)
        if self.adaptive_step:
            self.inc.SetAttribute("dtgrd_max", self.rms_step_max_ms)
        self.inc.SetAttribute("tstart", 0.0)
        if self.verbose:
            print(f"  [rms] step {self.rms_step_ms} ms"
                  + (f", adaptive up to {self.rms_step_max_ms} ms"
                     if self.adaptive_step else ", fixed"))

    # ── event hygiene ────────────────────────────────────────────────────
    def purge_events(self) -> None:
        """Delete every simulation event (stale events corrupt runs).

        ⚠ ``Delete()`` silently no-ops while a simulation calculation is
        active (discovered 2026-07-20): without the ``ResetCalculation()``
        the events of every earlier step run in the same PF session stay
        in ``p_event`` and *replay* on the next ComInc, contaminating every
        battery entry after the first.  The folder is therefore verified
        empty afterwards -- Fail-Fast beats a silently corrupted battery.
        """
        self.app.ResetCalculation()
        self._calculation_active = False
        for ev in list(self.evt_folder.GetContents()):
            ev.Delete()
        left = list(self.evt_folder.GetContents())
        if left:
            raise PFSessionError(
                f"event purge failed; {len(left)} event(s) remain: "
                f"{[e.loc_name for e in left]}")
        self._param_event_slots.clear()
        self._param_event_next.clear()
        self._load_event_slots.clear()
        self._load_event_next.clear()
        self._tap_event_slots.clear()
        self._tap_event_next.clear()
        self._armed_persistent_events.clear()
        self._new_events_pending_admission = 0
        self._persistent_events_created = 0
        self._persistent_events_discovered = 0
        self._admission_executes = 0
        self._retired_events = 0
        self._pool_serial = {
            class_name: 0 for class_name in PERSISTENT_EVENT_PREFIXES
        }

    @staticmethod
    def _event_key(target) -> str:
        """Stable PF database identity used by the one-shot pools."""
        try:
            return str(target.GetFullName())
        except Exception:
            return str(getattr(target, "loc_name", target))

    def prepare_persistent_event_pool(
            self, *, discard_param_targets=(),
            discard_load_targets=()) -> Dict[str, int]:
        """Reset, discover and safely park qOFO-owned event slots.

        Fired objects are reusable only after a fresh ComInc (live-probed
        2026-07-22). Objects stay in the project between runs, but every slot
        is made inert before the next ComInc and every run cursor is reset.
        Unmanaged events are deleted after ResetCalculation.
        """
        discard_param_keys = {
            (self._event_key(target), str(variable))
            for target, variable in discard_param_targets
        }
        discard_load_keys = {
            self._event_key(target) for target in discard_load_targets
        }
        self.app.ResetCalculation()
        self._calculation_active = False
        self._param_event_slots.clear()
        self._param_event_next.clear()
        self._load_event_slots.clear()
        self._load_event_next.clear()
        self._tap_event_slots.clear()
        self._tap_event_next.clear()
        self._armed_persistent_events.clear()
        self._new_events_pending_admission = 0
        self._persistent_events_created = 0
        self._persistent_events_discovered = 0
        self._admission_executes = 0
        self._retired_events = 0
        self._pool_serial = {
            class_name: 0 for class_name in PERSISTENT_EVENT_PREFIXES
        }

        removed = 0
        owned_discarded = 0
        orphaned_removed = 0
        for ev in list(self.evt_folder.GetContents()):
            class_name = str(ev.GetClassName())
            prefix = PERSISTENT_EVENT_PREFIXES.get(class_name)
            name = str(ev.loc_name)
            if prefix is None or not name.startswith(prefix):
                ev.Delete()
                removed += 1
                continue
            try:
                serial = int(name[len(prefix):])
            except ValueError as exc:
                raise PFSessionError(
                    f"malformed persistent event name {name!r}"
                ) from exc
            self._pool_serial[class_name] = max(
                self._pool_serial[class_name], serial + 1
            )
            target = ev.GetAttribute("p_target")
            if target is None:
                # Orphaned slot: the object exists but carries no target, so it
                # can never be armed.  Reachable without any defect in this
                # module, because _new_pool_event creates the object and
                # _append_* assigns p_target as a separate step -- a run that
                # dies between the two (or whose target element is later
                # removed) leaves exactly this state, and the pool persists in
                # the project across runs.  Observed 2026-07-30 on
                # 'qofo_pool_p_0001100' after runs 0076-0079 aborted, which
                # then blocked every subsequent run.
                #
                # Deleted rather than raised: an unarmable slot is in the same
                # category as the unmanaged events removed above, and the count
                # is reported separately so genuine pool corruption stays
                # visible instead of being silently absorbed.
                ev.Delete()
                orphaned_removed += 1
                continue
            if class_name == "EvtParam":
                variable = str(ev.GetAttribute("variable"))
                key = (self._event_key(target), variable)
                if key in discard_param_keys:
                    ev.Delete()
                    owned_discarded += 1
                    continue
                ev.SetAttribute("time", EVENT_INERT_TIME_S)
                self._param_event_slots.setdefault(key, []).append(ev)
            elif class_name == "EvtLod":
                key = self._event_key(target)
                if key in discard_load_keys:
                    ev.Delete()
                    owned_discarded += 1
                    continue
                ev.SetAttribute("time", EVENT_INERT_TIME_S)
                self._load_event_slots.setdefault(key, []).append(ev)
            else:
                key = self._event_key(target)
                ev.SetAttribute("time", EVENT_INERT_TIME_S)
                self._tap_event_slots.setdefault(key, []).append(ev)
            self._persistent_events_discovered += 1

        for pool in (
            self._param_event_slots,
            self._load_event_slots,
            self._tap_event_slots,
        ):
            for events in pool.values():
                events.sort(key=lambda event: str(event.loc_name))
        self._param_event_next.update(
            (key, 0) for key in self._param_event_slots
        )
        self._load_event_next.update(
            (key, 0) for key in self._load_event_slots
        )
        self._tap_event_next.update(
            (key, 0) for key in self._tap_event_slots
        )
        stats = self.event_pool_stats()
        stats["unmanaged_removed"] = removed
        stats["owned_discarded"] = owned_discarded
        stats["orphaned_removed"] = orphaned_removed
        if orphaned_removed:
            print(f"[event_pool] removed {orphaned_removed} orphaned pool "
                  f"slot(s) with no p_target (leftovers of an aborted run)")
        return stats

    def _new_pool_event(self, class_name: str):
        prefix = PERSISTENT_EVENT_PREFIXES[class_name]
        serial = self._pool_serial[class_name]
        self._pool_serial[class_name] = serial + 1
        ev = self.evt_folder.CreateObject(
            class_name, f"{prefix}{serial:07d}"
        )
        if ev is None:
            raise PFSessionError(
                f"{class_name} persistent-slot creation failed"
            )
        self._persistent_events_created += 1
        if self._calculation_active:
            self._new_events_pending_admission += 1
        return ev

    def _append_param_slot(self, target, variable: str,
                           initial_value: float = 0.0):
        key = (self._event_key(target), str(variable))
        ev = self._new_pool_event("EvtParam")
        ev.SetAttribute("p_target", target)
        ev.SetAttribute("variable", str(variable))
        ev.SetAttribute("value", repr(float(initial_value)))
        ev.SetAttribute("time", EVENT_INERT_TIME_S)
        self._param_event_slots.setdefault(key, []).append(ev)
        self._param_event_next.setdefault(key, 0)
        return ev

    def _append_load_slot(self, target):
        key = self._event_key(target)
        ev = self._new_pool_event("EvtLod")
        ev.SetAttribute("p_target", target)
        ev.SetAttribute("iopt_type", 0)
        ev.SetAttribute("dP", 0.0)
        ev.SetAttribute("dQ", 0.0)
        ev.SetAttribute("time", EVENT_INERT_TIME_S)
        self._load_event_slots.setdefault(key, []).append(ev)
        self._load_event_next.setdefault(key, 0)
        return ev

    def _append_tap_slot(self, target):
        key = self._event_key(target)
        ev = self._new_pool_event("EvtTap")
        ev.SetAttribute("p_target", target)
        ev.SetAttribute("ntap", 0)
        ev.SetAttribute("time", EVENT_INERT_TIME_S)
        self._tap_event_slots.setdefault(key, []).append(ev)
        self._tap_event_next.setdefault(key, 0)
        return ev

    def preallocate_param_events(self, target, variable: str, slots: int,
                                 *, initial_value: float = 0.0) -> None:
        """Create ``slots`` one-shot EvtParams before ComInc.

        Live-probed 2026-07-22: an unused event registered before ComInc can
        be moved from ``EVENT_INERT_TIME_S`` into the active horizon, while
        a fired event cannot be re-armed.  Every slot is therefore consumed
        at most once.
        """
        if slots < 1:
            raise ValueError("parameter-event slots must be >= 1")
        key = (self._event_key(target), str(variable))
        if key in self._param_event_slots:
            raise PFSessionError(f"parameter-event pool already exists: {key}")
        self._param_event_slots[key] = []
        self._param_event_next[key] = 0
        self.ensure_param_event_capacity(
            target, variable, slots, initial_value=initial_value
        )

    def ensure_param_event_capacity(self, target, variable: str, slots: int,
                                    *, initial_value: float = 0.0) -> None:
        """Grow a parameter pool to at least ``slots``, never shrink it."""
        key = (self._event_key(target), str(variable))
        self._param_event_slots.setdefault(key, [])
        self._param_event_next.setdefault(key, 0)
        while len(self._param_event_slots[key]) < int(slots):
            self._append_param_slot(target, variable, initial_value)

    def preallocate_load_events(self, target, slots: int) -> None:
        """Create ``slots`` one-shot, initially inert EvtLod objects."""
        if slots < 1:
            raise ValueError("load-event slots must be >= 1")
        key = self._event_key(target)
        if key in self._load_event_slots:
            raise PFSessionError(f"load-event pool already exists: {key}")
        self._load_event_slots[key] = []
        self._load_event_next[key] = 0
        self.ensure_load_event_capacity(target, slots)

    def ensure_load_event_capacity(self, target, slots: int) -> None:
        """Grow a load-event pool to at least ``slots``, never shrink it."""
        key = self._event_key(target)
        self._load_event_slots.setdefault(key, [])
        self._load_event_next.setdefault(key, 0)
        while len(self._load_event_slots[key]) < int(slots):
            self._append_load_slot(target)

    @staticmethod
    def _take_slot(pool, next_by_key, key, kind: str):
        events = pool.get(key)
        if events is None:
            return None
        slot = next_by_key[key]
        if slot >= len(events):
            raise PFSessionError(
                f"preallocated {kind} event slots exhausted for {key}: "
                f"used {slot}, capacity {len(events)}")
        next_by_key[key] = slot + 1
        return events[slot]

    def event_pool_stats(self) -> Dict[str, int]:
        """Allocated/consumed slot counts for provenance and validation."""
        return {
            "param_total": sum(map(len, self._param_event_slots.values())),
            "param_used": sum(self._param_event_next.values()),
            "load_total": sum(map(len, self._load_event_slots.values())),
            "load_used": sum(self._load_event_next.values()),
            "tap_total": sum(map(len, self._tap_event_slots.values())),
            "tap_used": sum(self._tap_event_next.values()),
            "discovered": self._persistent_events_discovered,
            "created": self._persistent_events_created,
            "pending_admission": self._new_events_pending_admission,
            "admission_executes": self._admission_executes,
            "retired": self._retired_events,
        }

    def _track_persistent_arm(self, ev, t_event: float) -> None:
        if self.persistent_event_pool:
            self._armed_persistent_events.append((float(t_event), ev))

    def _persistent_param_slot(self, target, variable: str):
        key = (self._event_key(target), str(variable))
        events = self._param_event_slots.get(key)
        next_slot = self._param_event_next.get(key, 0)
        if events is None or next_slot >= len(events):
            self._append_param_slot(target, variable)
        return self._take_slot(
            self._param_event_slots, self._param_event_next, key, "parameter"
        )

    def _persistent_load_slot(self, target):
        key = self._event_key(target)
        events = self._load_event_slots.get(key)
        next_slot = self._load_event_next.get(key, 0)
        if events is None or next_slot >= len(events):
            self._append_load_slot(target)
        return self._take_slot(
            self._load_event_slots, self._load_event_next, key, "load"
        )

    def add_param_event(self, target, variable: str, new_value: float,
                        t_event: float) -> None:
        key = (self._event_key(target), str(variable))
        if self.persistent_event_pool:
            ev = self._persistent_param_slot(target, variable)
        else:
            ev = self._take_slot(
                self._param_event_slots, self._param_event_next, key, "parameter")
        if ev is None:
            if self.require_preallocated_events:
                raise PFSessionError(
                    f"no preallocated parameter-event pool for {key}; "
                    "refusing mid-run CreateObject (event starvation risk)")
            ev = self.evt_folder.CreateObject("EvtParam", f"step_{variable}")
            if ev is None:
                raise PFSessionError(f"EvtParam creation failed for {variable}")
            ev.SetAttribute("p_target", target)
            ev.SetAttribute("variable", variable)
            # Fresh event created inside the active RMS calculation: PF admits
            # such events only via extra ComSim.Execute barriers (probe
            # 2026-07-23), so mark it pending for admit_new_events(). Re-arm of
            # a fired object does NOT work, so this create-fresh + barrier path
            # is the correct alternative to the (broken) slot-reuse pool.
            if self._calculation_active:
                self._new_events_pending_admission += 1
        # Payload first, time last: moving the unused slot is the arm action.
        ev.SetAttribute("value", repr(float(new_value)))
        ev.SetAttribute("time", float(t_event))
        self._track_persistent_arm(ev, t_event)

    def add_load_event(self, target, d_p_percent: float, d_q_percent: float,
                       t_event: float) -> None:
        """Load step (``EvtLod``): scale P and Q by a *percentage*.

        ``iopt_type = 0`` is PF's "Incremental Change". **The percentages are
        additive on the load's ORIGINAL value, not relative to its present
        value** (measured 2026-07-28: a +10/-9.09, +25/-20, +50/-33.3,
        +100/-50 ladder left the load at 1.016, 1.069, 1.240 and 1.745x base,
        matching ``1 + sum(dP_i)`` to ~1 % at every rung, not the ~1.0x that
        present-value semantics would give).

        Consequently **an event of ``+X`` % is undone by exactly ``-X`` %**,
        not by ``-X/(1+X)`` %. Using the latter leaves a residue that compounds:
        over the four rungs above it drifted the base load by +75 %, which
        destroyed the independence of a disturbance ladder.

        A caller tracking an absolute profile factor must therefore accumulate
        against the original value, not the previous one.

        Verified 2026-07-21: a +50 % event on ``TN_load0_const_b0`` moved
        ``m:P:bus1`` 38.129 -> 57.181 MW at the event instant, with PF
        logging ``Load Event: 'Incremental Change' - Active Power changed by
        50,000 %``.  An earlier probe concluded EvtLod "never fires"; it had
        left ``p_target`` unset, so the event had no object to act on and PF
        silently did nothing.  **Always set p_target.**
        """
        key = self._event_key(target)
        if self.persistent_event_pool:
            ev = self._persistent_load_slot(target)
        else:
            ev = self._take_slot(
                self._load_event_slots, self._load_event_next, key, "load")
        if ev is None:
            if self.require_preallocated_events:
                raise PFSessionError(
                    f"no preallocated load-event pool for {key}; refusing "
                    "mid-run CreateObject (event starvation risk)")
            ev = self.evt_folder.CreateObject("EvtLod", "prof_lod")
            if ev is None:
                raise PFSessionError("EvtLod creation failed")
            ev.SetAttribute("p_target", target)
            ev.SetAttribute("iopt_type", 0)      # incremental, percent
            if self._calculation_active:         # create-fresh + barrier admit
                self._new_events_pending_admission += 1
        ev.SetAttribute("dP", float(d_p_percent))
        ev.SetAttribute("dQ", float(d_q_percent))
        ev.SetAttribute("time", float(t_event))
        self._track_persistent_arm(ev, t_event)

    def add_load_step_event(self, target, d_p_percent: float,
                            d_q_percent: float, t_event: float,
                            seq: int = 0) -> None:
        """One-shot load step for a CONTINGENCY: fresh ``EvtLod``, never pooled.

        :meth:`add_load_event` re-arms a pooled slot and writes an ABSOLUTE
        time.  Neither is safe for a single step armed mid-run:

        * a pooled slot re-armed after ComInc is not honoured -- the defect
          measured for ``EvtTap`` on 2026-07-30 (pooled slot: dead; fresh
          object + admission barrier: works);
        * PF reads ``time`` inside its current 60 s event window, so an
          absolute 180.5 s armed at a calculation clock of 180 s lands at
          360.5 s -- past the end of a 240 s run, i.e. never.

        Measured 2026-08-06 with the pooled/absolute combination: the RMS
        plant stayed BIT-IDENTICAL to its twin (total machine Q difference
        0.0 Mvar at every instant) while the mirror had already recorded the
        step as applied.  A disturbance that silently does not happen is the
        worst failure mode this adapter has, because twin-differencing makes
        the result look plausible.
        """
        ev = self.evt_folder.CreateObject(
            "EvtLod", f"cont{seq}_{target.loc_name}")
        if ev is None:
            raise PFSessionError(
                f"EvtLod creation failed for {target.loc_name}")
        ev.SetAttribute("p_target", target)
        ev.SetAttribute("iopt_type", 0)          # incremental, percent
        ev.SetAttribute("dP", float(d_p_percent))
        ev.SetAttribute("dQ", float(d_q_percent))
        if self._calculation_active:             # create-fresh + barrier admit
            self._new_events_pending_admission += 1
        window = EVENT_WINDOW_S * math.floor(
            getattr(self, "_sim_time", 0.0) / EVENT_WINDOW_S)
        ev.SetAttribute("time", float(t_event) - window)
        # Deliberately NOT _track_persistent_arm: this object is single-use.

    def add_tap_event(self, target, ntap: int, t_event: float,
                      seq: int = 0) -> None:
        """Discrete tap/step event (``EvtTap``): trafo OLTC or shunt step.

        ``ntap`` is the *relative* (signed) number of tap steps applied at
        ``t_event``.  Verified 2026-07-20 (pf/probes/probe_tap_avr.py): EvtTap
        carries only ``time``/``p_target``/``ntap``; on an ``ElmTr3`` it
        moves the winding that owns the tap changer (HV on the NC3W
        couplers), on an ``ElmShnt`` it switches ``ncapa`` by ``ntap``.
        ``EvtParam`` on the tap-position attribute does NOT work in RMS
        (taps are read at init only -- the zero-response finding of run
        083416).
        """
        # EvtTap NEVER uses the persistent pool -- see the 2026-07-30 finding.
        #
        # A pooled slot (a pre-created EvtTap re-armed from EVENT_INERT_TIME_S
        # to a future time) does not fire once the calculation is running: the
        # tap simply does not move.  Measured on MT_g0_t0 (HV terminal TN_bus1,
        # where one step is ~6 mpu):
        #
        #   pooled slot, mid-run, no barrier          0.033 mpu  (dead)
        #   pooled slot, mid-run, 2 forced barriers   0.037 mpu  (dead)
        #   fresh EvtTap + admit_new_events           +6.14 mpu  (works)
        #
        # It is not element-specific -- ElmTr3 couplers and ElmShnt steps were
        # verified to work mid-run -- and it is not the admission barrier: the
        # pool path also never incremented ``_new_events_pending_admission``,
        # so no barrier ran, and forcing one did not help either.  Re-arming
        # the object is simply not honoured for EvtTap the way it is for
        # EvtParam, which is what the pool was validated on.
        #
        # Cost of not pooling: one EvtTap object per tap dispatch instead of a
        # reused slot.  Taps are rare (3 in a 54-interval run), so the object
        # count is negligible against the ~8900 parameter slots, and
        # ``prepare_persistent_event_pool`` clears them between runs.
        #
        # Consequence of the old behaviour: every tap the RMS plant was
        # commanded after ComInc was silently ignored, while the mirror net
        # recorded it as applied.  Do not reintroduce pooling here without
        # re-measuring the step at the transformer's own terminal.
        ev = self.evt_folder.CreateObject(
            "EvtTap", f"tap{seq}_{target.loc_name}")
        if ev is None:
            raise PFSessionError(
                f"EvtTap creation failed for {target.loc_name}"
            )
        ev.SetAttribute("p_target", target)
        if self._calculation_active:             # create-fresh + barrier admit
            self._new_events_pending_admission += 1
        # ``i_tap`` is the Tap Action; ``ntap`` alone is IGNORED.  User Manual
        # 13.9.15 and the DPL example in the scripting chapter
        # ("...create=EvtTap ... i_tap=1 ... will decrease the tap position"):
        # 0 = increase by one, 1 = decrease by one.  The default is 0, so
        # every tap this adapter ever dispatched was an INCREASE regardless of
        # the commanded direction -- which is the whole of the apparent
        # "inverted sign" (2026-07-30).  Verified on MT_g0_t0 at its HV
        # terminal: i_tap=1 gives -6.26 mpu (pandapower tap_pos=-1: -8.02),
        # i_tap=0 gives +6.02 mpu.
        #
        # ``_dispatch_taps`` emits one event per single step, so |ntap| == 1
        # and the mapping is unambiguous.  ``ntap`` is still written: it
        # carries the value for the "set to value" action and is harmless
        # otherwise, and it keeps the object self-describing in the GUI.
        ev.SetAttribute("i_tap", 0 if int(ntap) > 0 else 1)
        ev.SetAttribute("ntap", int(ntap))
        # Fold the absolute time into PF's current 60 s event window (see
        # EVENT_WINDOW_S).  Without this every event armed at a calculation
        # clock >= 60 s fires 60*floor(clock/60) seconds late, or is lost when
        # the run ends first -- measured on the MSC banks: four commands
        # (+1,+1,-1,-1) landed 20.53 / 120.50 / 160.51 / never, leaving the
        # bank one step from where it was commanded.
        window = EVENT_WINDOW_S * math.floor(
            getattr(self, "_sim_time", 0.0) / EVENT_WINDOW_S)
        ev.SetAttribute("time", float(t_event) - window)
        # Deliberately NOT _track_persistent_arm: retirement writes ``time``
        # back onto an EvtTap that PF has already fired, and that breaks every
        # SUBSEQUENT tap event in the calculation.  Measured on MT_g0_t0 with
        # four mid-run taps: with retirement only the first fired (+6.20 mpu,
        # then 0.03 / -0.00 / 0.02 -- including one on a different
        # transformer); without it, consecutive taps all fire.  These are
        # one-shot objects that are never reused, so there is nothing to
        # retire; ``prepare_persistent_event_pool`` deletes them at the start
        # of the next run.

    def add_outage_event(self, target, t_event: float, seq: int = 0) -> None:
        """Element outage (``EvtOutage``): take ``target`` out at ``t_event``.

        Used for N-1 disturbances (e.g. a synchronous machine trip). Verified
        2026-07-26 (scratchpad/outage_probe.py) on ``G 10``: P collapses from
        131.9 MW, a witness machine picks up +19.4 MW through governor action
        and a park terminal sags -0.0055 pu, and PF logs
        ``evt - (t=05:000 s) Grid\\G 10.ElmSym:``.

        Two landmines:

        * ``p_target`` must be set or PF silently does nothing (the same trap
          that made ``EvtLod`` look permanently broken).
        * the element's **input** attribute ``outserv`` is NOT updated by the
          event -- it still reads 0 afterwards, exactly as tap positions are
          not updated by ``EvtTap``. Detect the outage from the disappearance
          of the ``m:`` result variables (``GetAttribute`` raises), never from
          ``outserv``.

        There is no persistent pool for outages: they are rare, one-shot
        disturbances rather than per-dispatch commands, so the create-fresh
        plus admission-barrier path is used.
        """
        ev = self.evt_folder.CreateObject(
            "EvtOutage", f"outage{seq}_{target.loc_name}")
        if ev is None:
            raise PFSessionError(
                f"EvtOutage creation failed for {target.loc_name}")
        ev.SetAttribute("p_target", target)
        if self._calculation_active:            # create-fresh + barrier admit
            self._new_events_pending_admission += 1
        # Fold the absolute time into PF's current 60 s event window, exactly
        # as add_tap_event does -- see EVENT_WINDOW_S.  The 2026-07-26
        # verification armed the outage at t = 5 s, i.e. inside the first
        # window, where the fold is a no-op and absolute time happens to be
        # right; that is why this was not caught then.  An outage armed at a
        # calculation clock >= 60 s without the fold fires
        # 60*floor(clock/60) s late, or never if the run ends first -- which
        # for a one-shot N-1 disturbance means the run silently contains no
        # contingency at all.
        window = EVENT_WINDOW_S * math.floor(
            getattr(self, "_sim_time", 0.0) / EVENT_WINDOW_S)
        ev.SetAttribute("time", float(t_event) - window)

    # ── monitored variables ──────────────────────────────────────────────
    def set_monitors(self, monitors: List[Tuple[Any, str, str]]) -> None:
        """Register (object, variable, label) monitors, clearing any prior."""
        # Clear existing monitor rows so runs never accumulate stale columns.
        for m in list(self.res.GetContents("*.IntMon")):
            m.Delete()
        for obj, var, _label in monitors:
            self.res.AddVariable(obj, var)

    # ── run ──────────────────────────────────────────────────────────────
    def initialise(self) -> None:
        if self.inc.Execute():
            raise PFSessionError("ComInc (RMS init) failed")
        self._calculation_active = True
        self._sim_time = 0.0

    def admit_new_events(self, current_time: float, horizon_s: float,
                         batch_size: int = EVENT_ADMISSION_BATCH) -> int:
        """Admit events created since the last advance, via *advancing* barriers.

        PF admits events created inside an active RMS calculation only across a
        ``ComSim.Execute`` that actually integrates forward -- a zero-advance
        Execute (``tstop == current``) admits nothing (validated 2026-07-23:
        run 0044 froze at t=41 s with zero-advance barriers).  The working
        pattern (``probe_event_admission_barrier``: events at t=1 s, barriers
        at 0.1/0.2 s, then cross) is a few short forward Executes that stop
        strictly *before* the event time, then the caller's main advance
        crosses it and the whole batch fires on grid.

        The events for this interval are scheduled at ``current_time +
        horizon_s``; the barriers land in the open interval
        ``(current_time, current_time + horizon_s)`` so they admit without
        prematurely crossing (and losing) the events.  Barrier count scales
        with the pending batch (~``batch_size`` admitted per Execute).
        """
        pending = self._new_events_pending_admission
        if pending <= 0:
            return 0
        if batch_size < 1:
            raise ValueError("event-admission batch_size must be >= 1")
        if horizon_s <= 0.0:
            raise ValueError("event-admission horizon_s must be > 0")
        calls = max(2, (pending + int(batch_size) - 1) // int(batch_size))
        for i in range(1, calls + 1):
            stop = current_time + horizon_s * i / (calls + 1)
            self.sim.SetAttribute("tstop", float(stop))
            self._sim_time = float(stop)
            if self.sim.Execute():
                raise PFSessionError(
                    f"event-admission ComSim failed at barrier t={stop}s")
        self._new_events_pending_admission = 0
        self._admission_executes += calls
        return calls

    def _retire_fired_events(self, tstop: float) -> None:
        """Move safely-fired persistent slots back to the inert timestamp."""
        if not self.persistent_event_pool:
            return
        cutoff = float(tstop) - RMS_STEP_MS / 1000.0
        keep = []
        for event_time, ev in self._armed_persistent_events:
            if event_time <= cutoff:
                ev.SetAttribute("time", EVENT_INERT_TIME_S)
                self._retired_events += 1
            else:
                keep.append((event_time, ev))
        self._armed_persistent_events = keep

    def simulate(self, tstop: float) -> None:
        self.sim.SetAttribute("tstop", float(tstop))
        if self.sim.Execute():
            raise PFSessionError(f"ComSim to t={tstop}s failed")
        self._sim_time = float(tstop)
        self._retire_fired_events(tstop)

    def read(self, obj, var: str, stride: int = 5) -> Tuple[List[float], List[float]]:
        """Return (time[], value[]) for one monitored variable.

        ``stride`` down-samples the result rows (every ``stride``-th step; at
        the 10 ms RMS step, stride 5 = 50 ms resolution — ample for
        settling-time analysis and ~5x fewer per-cell reads, which is what
        made the full-model reads time out).
        """
        self.res.Load()
        n = self.res.GetNumberOfRows()
        col = self.res.FindColumn(obj, var)
        if col < 0:
            raise PFSessionError(f"variable {var!r} not monitored on "
                                 f"{getattr(obj, 'loc_name', obj)!r}")
        rows = list(range(0, n, max(1, stride)))
        if rows[-1] != n - 1:
            rows.append(n - 1)                 # always keep the final point
        t = [self.res.GetValue(i, -1)[1] for i in rows]
        y = [self.res.GetValue(i, col)[1] for i in rows]
        return t, y


# =====================================================================
#  Monitored-output catalogue
# =====================================================================

def monitored_outputs(app, include_der: bool = False,
                      ) -> List[Tuple[Any, str, str]]:
    """(object, variable, label) list of the controlled outputs to record.

    Restricted to the OFO-relevant controlled outputs so the trajectory read
    stays fast: the 12 coupler-3W interface Q flows (q_STS) and their MV-side
    PCC voltages, the TN bus voltages, and the machine speeds.  The DSO
    internal (feeder) buses are omitted -- they are not controlled outputs.

    ``include_der`` additionally records each DER park's terminal Q and
    voltage.  These are *not* controlled outputs; they are the diagnostic
    needed to attribute a Gate-E endpoint gap, because an interface-Q error
    alone cannot distinguish a capability clip (park Q pinned at a limit the
    static plant does not share) from droop amplification of a voltage
    difference (park Q free, but responding to a different ``u``).
    """
    out: List[Tuple[Any, str, str]] = []
    pcc_terms = set()
    # Coupler-3W interface Q (q_STS, HV side) + the PCC (MV-side) voltage.
    for tr in get_all(app, "ElmTr3"):
        if tr.loc_name.startswith("NC3W_"):
            out.append((tr, "m:Q:bushv", f"qSTS_{tr.loc_name}"))
            mv = tr.GetAttribute("busmv")
            if mv is not None and mv.cterm is not None:
                pcc_terms.add(mv.cterm.loc_name)
    # TN bus voltages + PCC (coupler MV) voltages.  With ``include_der`` the
    # DSO feeder buses come too: they are not controlled outputs, but without
    # them a static-vs-RMS voltage comparison inside a DSO area is impossible
    # (the static side records every bus).
    for t in get_all(app, "ElmTerm"):
        nm = t.loc_name
        if nm.startswith("TN_bus") or nm in pcc_terms:
            out.append((t, "m:u", f"u_{nm}"))
        elif include_der and nm.startswith("DSO_"):
            out.append((t, "m:u", f"u_{nm}"))
    # Machine speeds.
    for m in get_all(app, "ElmSym"):
        if m.GetAttribute("outserv") == 0:
            out.append((m, "s:xspeed", f"spd_{m.loc_name}"))
    # Machine reactive power, terminal voltage, and AVR V-ref -- for the
    # Generator Q and Generator V plot pages.  A PF plot curve is empty
    # unless its variable is in the result file, and these are NOT among the
    # controlled outputs above, so they must be added explicitly.  Terminal
    # voltage is taken from the machine's ElmTerm (``m:u`` is reliable there;
    # on some element classes it is not), and the V-ref from the AVR DSL's
    # ``usetp`` signal inside the machine's composite (the OFO write handle,
    # 2026-07-20).  Under ``include_der`` so plain screening stays lean.
    if include_der:
        for m in get_all(app, "ElmSym"):
            if m.GetAttribute("outserv") != 0:
                continue
            out.append((m, "m:Q:bus1", f"qGEN_{m.loc_name}"))
            cub = m.GetAttribute("bus1")
            term = cub.cterm if cub is not None else None
            if term is not None:
                out.append((term, "m:u", f"uGEN_{m.loc_name}"))
            comp = m.GetAttribute("c_pmod")
            if comp is not None:
                avr = next((d for d in comp.GetContents()
                            if d.GetClassName() == "ElmDsl"
                            and "avr" in d.loc_name.lower()), None)
                if avr is not None:
                    out.append((avr, "s:usetp", f"vref_{m.loc_name}"))
    # Discrete actuator state: machine 2W taps, coupler 3W taps, shunt steps.
    # Recorded because the OFO's integer decisions are what diverged between
    # the static and RMS runs (2026-07-21), and because a plot page can only
    # show what the result file holds.  The variable names are the RMS
    # *calculated* ones -- the input attributes (nntap / n3tap_h / ncapa) are
    # not updated by simulation events, which is why the plant shadows them.
    if include_der:
        for t in get_all(app, "ElmTr2"):
            if t.loc_name.startswith("MT_"):
                out.append((t, TAP_VAR_2W, f"tap_{t.loc_name}"))
        for t in get_all(app, "ElmTr3"):
            if t.loc_name.startswith("NC3W_"):
                out.append((t, TAP_VAR_3W, f"tap_{t.loc_name}"))
        for s in get_all(app, "ElmShnt"):
            out.append((s, SHUNT_STEP_VAR, f"step_{s.loc_name}"))

    # DER park Q and terminal voltage (diagnostic; see docstring).
    # The voltage must be read from the park's ElmTerm: ``m:u`` does not
    # exist on an ElmGenstat in RMS (FindColumn returns -1), whereas
    # ``m:Q:bus1`` does.
    if include_der:
        comps = {c.loc_name: c for c in get_all(app, "ElmComp")}
        for g in get_all(app, "ElmGenstat"):
            if g.GetAttribute("outserv") != 0:
                continue
            out.append((g, "m:Q:bus1", f"qDER_{g.loc_name}"))
            cub = g.GetAttribute("bus1")
            term = cub.cterm if cub is not None else None
            if term is not None:
                out.append((term, "m:u", f"uDER_{g.loc_name}"))
            # QVPRE Q setpoint (the OFO command, pu of S_n) -- for the DER Q
            # page's setpoint plot, which was blank because s:qset was never
            # recorded.  The block lives in the park's WECC composite.
            comp = comps.get(f"WECC_{g.loc_name}")
            if comp is not None:
                pre = next((d for d in comp.GetContents()
                            if d.GetClassName() == "ElmDsl"
                            and d.loc_name == QVPRE_ELEMENT_NAME), None)
                if pre is not None:
                    out.append((pre, "s:qset", f"qsetDER_{g.loc_name}"))
    return out


# =====================================================================
#  Settling-time metric
# =====================================================================

def settling_metrics(t: List[float], y: List[float], t_event: float,
                     band: float = 0.02, abs_floor: float = 1e-5) -> Dict[str, float]:
    """2 %-band settling time / overshoot / final value after a step.

    ``y_initial`` is the value just before ``t_event``; ``y_final`` the mean
    of the last 0.5 s.  Settling time is measured from ``t_event`` to the
    last instant the trajectory leaves the envelope
    ``max(band*|step|, abs_floor)`` around ``y_final``.  ``abs_floor`` is the
    physically meaningful "settled" band for the quantity (e.g. 1e-3 pu for a
    voltage, 1 Mvar for an interface flow); without it, a near-zero step lets
    low-amplitude electromechanical ringing register a spurious settling time.
    """
    import numpy as np
    ta = np.asarray(t)
    ya = np.asarray(y)
    i_ev = int(np.searchsorted(ta, t_event))
    y_init = float(ya[max(i_ev - 1, 0)])
    tail = ya[ta >= ta[-1] - 0.5]
    y_final = float(tail.mean())
    step = abs(y_final - y_init)
    tol = max(band * step, abs_floor)
    after = ta >= t_event
    dev = np.abs(ya - y_final)
    outside = after & (dev > tol)
    t_settle = (float(ta[outside][-1]) - t_event) if outside.any() else 0.0
    # Overshoot beyond the final value, as a fraction of the step.
    if step > 1e-9:
        post = ya[after]
        over = float((np.max((post - y_final) * np.sign(y_final - y_init))) / step)
        overshoot = max(over, 0.0)
    else:
        overshoot = 0.0
    return {"y_init": y_init, "y_final": y_final, "step": step,
            "t_settle": t_settle, "overshoot": overshoot}


# =====================================================================
#  Step catalogue
# =====================================================================

#: OLTC mechanical delay [s]: a tap dispatched at t_event physically moves
#: at t_event + this; sequential multi-tap moves are spaced by it (plan
#: Phase 5 step 3).
TAP_MECH_DELAY_S = 5.0

#: PowerFactory applies simulation-event times MODULO this window once an RMS
#: calculation is running: an event scheduled at absolute ``te`` actually fires
#: at ``te mod EVENT_WINDOW_S``, which presents as a deferral of
#: ``60*floor(t_clock/60)``.  Undocumented by DIgSILENT; established 2026-07-31
#: over 30+ measurements on ElmTr2 / ElmTr3 / ElmShnt via EvtTap and on
#: EvtSwitch, and validated with 12 consecutive on-time events across four
#: windows to t = 220 s.  ``ComInc``/``ComSim`` expose no attribute equal to 60.
#: See docs/daily_log/07_2026/2026-07-31_rms_tap_control_gate_e_result.md.
EVENT_WINDOW_S = 60.0


@dataclass
class StepDef:
    name: str
    target: Any          # resolved PF object (REEC_D ElmDsl, ElmTr3, ...)
    variable: str
    delta: float         # param: increment on the current value; tap: ntap
    unit: str
    note: str = ""
    #: "param" (EvtParam) | "tap" (EvtTap) | "outage" (EvtOutage)
    #: | "load" (EvtLod).
    #:
    #: The first two are **dispatches** -- an actuator command the controller
    #: itself issues -- and they are what bounds ``T_DS``: premise P3 asks
    #: whether the transient excited by one dispatch has decayed by the next
    #: sample.  The last two are **disturbances**, which no controller
    #: commands.  They answer a different question (how long the plant is not
    #: a settled algebraic map after a credible event, and therefore for how
    #: long the OFO integrates against a moving target) and they must not be
    #: mixed into the Gate-D dispatch verdict.  ``cmd_steps`` gates them
    #: separately for that reason.
    kind: str = "param"
    #: tap kind only: physical tap instants as offsets from t_event [s]
    #: (each applies ``ntap = delta``); settling is still measured from
    #: t_event, the dispatch instant -- the OFO window starts there.
    tap_times: Tuple[float, ...] = ()
    #: outage kind only: additional elements tripped with the target (the
    #: unit transformer of a machine), mirroring
    #: ``PowerFactoryPlant.apply_contingency`` so that a screening trip and a
    #: closed-loop contingency remove the same plant.
    also_trip: Tuple[Any, ...] = ()
    #: load kind only: (dP, dQ) in percent of the load's ORIGINAL value.
    #: EvtLod increments are additive on the base, not on the present value
    #: (see ``ScreeningContext.add_load_event``).
    load_pct: Tuple[float, float] = (0.0, 0.0)
    #: True for a disturbance case: excluded from the Gate-D dispatch
    #: verdict, reported in its own table.
    disturbance: bool = False


def _machine_avrs(app) -> Dict[Any, Any]:
    """{ElmSym: AVR ElmDsl} for every in-service machine with an AVR block.

    The V-ref lives *inside* the ``avr_IEEET1`` DSL (input signal ``usetp``,
    initialised to the LF setpoint by ComInc); ``ElmSym.usetp`` is LF-only
    and gives a zero RMS response (probe 2026-07-20).  G 01 (the 10 GVA
    'Rest of U.S.A. / Canada' equivalent) has no controller blocks at all
    and is therefore absent here.
    """
    out: Dict[Any, Any] = {}
    for comp in get_all(app, "ElmComp"):
        pelm = comp.GetAttribute("pelm") or []
        sym = next((e for e in pelm if e is not None
                    and e.GetClassName() == "ElmSym"
                    and e.GetAttribute("outserv") == 0), None)
        if sym is None:
            continue
        avr = next((d for d in comp.GetContents()
                    if d.GetClassName() == "ElmDsl"
                    and "avr" in d.loc_name.lower()), None)
        if avr is not None:
            out[sym] = avr
    return out


def _reec_of(app, park_name: str):
    """(REEC_D block, ElmGenstat) of a DER park's WECC composite, or (None, None)."""
    comp = next((c for c in get_all(app, "ElmComp")
                 if c.loc_name == f"WECC_{park_name}"), None)
    gen = next((g for g in get_all(app, "ElmGenstat")
                if g.loc_name == park_name), None)
    if comp is None or gen is None:
        return None, None
    reec = next((d for d in comp.GetContents()
                 if d.GetClassName() == "ElmDsl" and "REEC" in d.loc_name), None)
    return reec, gen


def _qvpre_of(app, park_name: str):
    """The ``QVPRE`` block of a park's WECC composite, or ``None``.

    This is the CURRENT Q write handle: since 2026-07-21 it occupies the
    Plant Control slot and drives ``REEC_D.Qext`` every solver step, so it
    overrides anything written to ``Qext`` directly.
    """
    comp = next((c for c in get_all(app, "ElmComp")
                 if c.loc_name == f"WECC_{park_name}"), None)
    if comp is None:
        return None
    return next((d for d in comp.GetContents()
                 if d.GetClassName() == "ElmDsl"
                 and d.loc_name == QVPRE_ELEMENT_NAME), None)


def default_catalogue(app) -> List[StepDef]:
    """Worst-case single-dispatch steps, one per OFO actuator class.

    * **DER Q** via ``QVPRE.qset`` (pu of the park's rated S).  NOT
      ``REEC_D.Qext``: the Q(V) layer overwrites that every step.  Step
      magnitude = min(60 Mvar, 0.5*S_n) so
      a small-rated DSO park does not saturate; 60 Mvar is the largest TSO
      wind-park command observed in the controller run (0024).
      Representative parks: one large TSO wind park and one DSO DER.
    * **Machine AVR V-ref** via ``EvtParam`` on the AVR DSL's ``usetp``
      signal (largest AVR-equipped plant; G 01 has no AVR).
    * **OLTC taps** via ``EvtTap`` with the 5 s mechanical delay: coupler-3W
      single tap, the sequential 2-tap case (Gate D), and one machine
      trafo.
    * **MSC shunt** switch-in via ``EvtTap`` (no mechanical delay).
    """
    steps: List[StepDef] = []
    gens = [g.loc_name for g in get_all(app, "ElmGenstat")]

    targets: List[str] = []
    tso = [n for n in gens if n.startswith("WP_TSO_")]
    if tso:
        targets.append(sorted(tso)[0])
    dso = [n for n in gens if n.startswith("DER_")]
    if dso:
        targets.append(sorted(dso)[0])

    for park_name in targets:
        reec, gen = _reec_of(app, park_name)
        if reec is None:
            continue
        sn = float(gen.GetAttribute("sgn"))
        q_mvar = min(60.0, 0.5 * sn)
        # Step QVPRE.qset, NOT REEC_D.Qext.  Since the Q(V) rollout of
        # 2026-07-21 the QVPRE block sits in the Plant Control slot and
        # writes Qext every solver step, so an EvtParam on Qext is
        # overwritten and the park never moves -- measured 2026-08-07, when
        # every DER case in the timescale battery returned T_s = 0.00 s while
        # the AVR and shunt cases (whose handles are still current) did not.
        # qset carries the same unit, pu of the park's rated S.
        pre = _qvpre_of(app, park_name)
        target, var = ((pre, "qset") if pre is not None else (reec, "Qext"))
        if pre is None:
            print(f"  [catalogue] WARNING: {park_name} has no QVPRE; falling "
                  f"back to REEC_D.Qext, which the Q(V) layer overwrites")
        steps.append(StepDef(
            name=f"der_q_+{q_mvar:.0f}Mvar_{park_name}",
            target=target, variable=var, delta=q_mvar / sn, unit="pu",
            note=f"+{q_mvar:.0f} Mvar Q dispatch on {park_name} (S_n={sn:.0f} MVA)"))

    # Machine AVR V-ref: +0.02 pu on the largest AVR-equipped plant.
    # Rated MVA of an ElmSym lives on its type (TypSym.sgn, per parallel
    # unit); plant total = sgn * ngnum.
    def _plant_mva(s) -> float:
        return (float(s.GetAttribute("typ_id").GetAttribute("sgn"))
                * int(s.GetAttribute("ngnum")))

    # NOT the largest machine: that is G 01, the 10 GVA "Rest of
    # U.S.A./Canada" equivalent, which carries the angle reference and is an
    # aggregation of a whole interconnection rather than a dispatchable
    # plant.  A V-ref step on it is not a realisable dispatch, and it
    # dominated the binding row of the 2026-08-07 run at 34.98 s.  Take the
    # SECOND and THIRD largest instead -- real units, and two of them so the
    # row does not rest on a single machine (user decision 2026-08-07).
    avrs = _machine_avrs(app)
    if avrs:
        ranked = sorted(avrs, key=_plant_mva, reverse=True)
        for sym in ranked[1:3]:
            mva = _plant_mva(sym)
            steps.append(StepDef(
                name=f"avr_vref_+0.02_{sym.loc_name.replace(' ', '')}",
                target=avrs[sym], variable="usetp", delta=0.02, unit="pu",
                note=f"+0.02 pu AVR V-ref on {sym.loc_name} "
                     f"(plant {mva:.0f} MVA; rank "
                     f"{ranked.index(sym) + 1} of {len(ranked)} by rating, "
                     f"the largest being excluded as a network equivalent)"))

    # Coupler-3W OLTC: single tap and the sequential 2-tap case (plan
    # Phase 5 step 3 / Gate D) -- physical moves at t_event + 5 s (+ 10 s).
    tr3 = next(iter(sorted((t for t in get_all(app, "ElmTr3")
                            if t.loc_name.startswith("NC3W_")),
                           key=lambda o: o.loc_name)), None)
    if tr3 is not None:
        steps.append(StepDef(
            name=f"tap_+1_{tr3.loc_name}", target=tr3, variable="ntap",
            delta=1.0, unit="tap", kind="tap",
            tap_times=(TAP_MECH_DELAY_S,),
            note=f"+1 coupler tap on {tr3.loc_name} "
                 f"({TAP_MECH_DELAY_S:.0f} s mech delay)"))
        steps.append(StepDef(
            name=f"tap_+2seq_{tr3.loc_name}", target=tr3, variable="ntap",
            delta=1.0, unit="tap", kind="tap",
            tap_times=(TAP_MECH_DELAY_S, 2 * TAP_MECH_DELAY_S),
            note=f"+2 sequential coupler taps on {tr3.loc_name} "
                 f"({TAP_MECH_DELAY_S:.0f} s apart)"))

    # Machine-trafo OLTC (TSO tap actuator class).
    mt = next(iter(sorted((t for t in get_all(app, "ElmTr2")
                           if t.loc_name.startswith("MT_")),
                          key=lambda o: o.loc_name)), None)
    if mt is not None:
        steps.append(StepDef(
            name=f"tap_+1_{mt.loc_name}", target=mt, variable="ntap",
            delta=1.0, unit="tap", kind="tap",
            tap_times=(TAP_MECH_DELAY_S,),
            note=f"+1 machine-trafo tap on {mt.loc_name} "
                 f"({TAP_MECH_DELAY_S:.0f} s mech delay)"))

    # MSC switch-in (breaker action, no mechanical tap delay).
    msc = next(iter(sorted((s for s in get_all(app, "ElmShnt")
                            if "MSC" in s.loc_name
                            and s.GetAttribute("outserv") == 0),
                           key=lambda o: o.loc_name)), None)
    if msc is not None:
        steps.append(StepDef(
            name=f"shunt_+1_{msc.loc_name}", target=msc, variable="ntap",
            delta=1.0, unit="step", kind="tap", tap_times=(0.0,),
            note=f"MSC switch-in (+1 step) on {msc.loc_name}"))
    return steps


#: Machines that must never be offered as an outage target.
#:
#: ``G 01`` is the 10 GVA "Rest of U.S.A./Canada" equivalent
#: (``GEN_NAMEPLATE[38] = ("G1", 10000.0, "Equivalent")``) and, since the
#: Phase-2 sync moved the reference off Bus 31, it also carries the angle
#: reference.  Tripping it does not model a contingency: it removes the
#: interconnection the benchmark is embedded in, along with the reference the
#: RMS solution is expressed against.  A "G 01 outage" is a modelling error,
#: not a severe N-1, and it is refused here rather than left to produce a
#: frequency collapse that reads like a result.
OUTAGE_FORBIDDEN: Tuple[str, ...] = ("G 01",)

#: pandapower ``net.gen`` index -> PowerFactory ``ElmSym`` name.
#:
#: **The two numbering systems are offset and collide.** The contingency
#: configs, ``experiments/helpers/contingency.py`` and the dead-band N-1 study
#: all address machines by *pandapower gen index*; PowerFactory addresses them
#: by name. ``gen[1]`` is ``G 03`` and ``gen[7]`` is ``G 09``, while the PF
#: machine literally named ``G 01`` is ``gen[8]``. Saying "gen 1" and meaning
#: ``G 01`` therefore selects the wrong machine -- or, worse, the slack.
#: Established 2026-08-03 from ``pandapower.networks.case39()`` bus order
#: against ``network/ieee39/constants.GEN_NAMEPLATE``.
#:
#: Only G 01/03/04/07/09/10 survive the ``wind_replace`` variation; the rest
#: are replaced by wind parks and have no ElmSym in the RMS model.
GEN_INDEX_TO_PF: Dict[int, str] = {
    0: "G 10",   # bus 29, 1000 MVA Hydro
    1: "G 03",   # bus 31,  800 MVA Nuclear  -- dead-band study's 650 MW trip
    2: "G 04",   # bus 32,  800 MVA Coal
    3: "G 05",   # bus 33,  600 MVA Coal      (removed by wind_replace)
    4: "G 06",   # bus 34,  800 MVA Nuclear   (removed by wind_replace)
    5: "G 07",   # bus 35,  700 MVA Coal
    6: "G 08",   # bus 36,  700 MVA Nuclear   (removed by wind_replace)
    7: "G 09",   # bus 37, 1000 MVA Nuclear  -- dead-band study's 830 MW trip
    8: "G 01",   # bus 38, 10 GVA equivalent + angle reference -- FORBIDDEN
}


def resolve_outage_targets(names: Optional[Sequence[str]] = None,
                           indices: Optional[Sequence[int]] = None,
                           ) -> List[str]:
    """PF machine names from ``--outage-machines`` and/or ``--outage-gens``.

    Prints the resolution of every index so the numbering collision described
    at ``GEN_INDEX_TO_PF`` can never bite silently.
    """
    out: List[str] = list(names or [])
    for i in (indices or []):
        nm = GEN_INDEX_TO_PF.get(int(i))
        if nm is None:
            print(f"[steps] gen index {i} is not an IEEE39 machine; "
                  f"valid: {sorted(GEN_INDEX_TO_PF)}")
            continue
        print(f"[steps] gen[{i}] resolves to PowerFactory machine {nm!r} "
              f"(NOT 'G {int(i):02d}' -- the two numberings are offset)")
        out.append(nm)
    return out

#: Load steps for the disturbance battery, as percent of the load's ORIGINAL
#: P (Q follows at constant power factor).  Both signs: reactive response is
#: asymmetric once converters approach their capability box, and the
#: ZIP exponent ``kqu = 2`` makes the voltage sensitivity itself
#: level-dependent, so a +X % result does not transfer to -X %.
LOAD_STEP_PERCENTS: Tuple[float, ...] = (+10.0, -10.0, +25.0)


def disturbance_catalogue(app, machines: Optional[Sequence[str]] = None,
                          ) -> List[StepDef]:
    """Credible disturbances: synchronous-machine trips and load steps.

    These are **not** dispatches and do not enter the Gate-D verdict.  What
    they measure is for how long after a credible event the plant is not the
    settled algebraic map ``y_qss(u)`` that the OFO's sensitivity model
    assumes -- i.e. for how many dispatch intervals the supervisory layer is
    integrating against a moving plant.  The dispatch battery bounds
    ``T_DS``; this battery bounds the *validity* of the quasi-steady-state
    substitution during recovery, and it supplies the operating points at
    which the dispatch battery is worth re-running.

    Machine trips include the unit transformer, matching
    ``PowerFactoryPlant.apply_contingency`` and the N-1 convention of
    ``ch:setup:scenarios:n1`` used by the dead-band study, so that a
    screening trip and a closed-loop contingency remove the same plant.

    ``machines`` selects targets by ``loc_name``; the default is every
    in-service machine except those in ``OUTAGE_FORBIDDEN``, largest first,
    capped at three so a battery stays affordable.
    """
    out: List[StepDef] = []

    def _plant_mva(s) -> float:
        try:
            return (float(s.GetAttribute("typ_id").GetAttribute("sgn"))
                    * int(s.GetAttribute("ngnum")))
        except Exception:
            return 0.0

    syms = [m for m in get_all(app, "ElmSym")
            if m.GetAttribute("outserv") == 0]
    by_name = {m.loc_name: m for m in syms}

    if machines:
        chosen = []
        for nm in machines:
            if nm in OUTAGE_FORBIDDEN:
                print(f"[steps] REFUSED outage target {nm!r}: it is the "
                      f"slack / interconnection equivalent, not a "
                      f"contingency (see OUTAGE_FORBIDDEN).")
                continue
            if nm not in by_name:
                print(f"[steps] outage target {nm!r} not an in-service "
                      f"ElmSym; available: {', '.join(sorted(by_name))}")
                continue
            chosen.append(by_name[nm])
    else:
        chosen = sorted((m for m in syms if m.loc_name not in OUTAGE_FORBIDDEN),
                        key=_plant_mva, reverse=True)[:3]

    # Machine trafos are resolved by TOPOLOGY, not by name: the sync script
    # names them ``MT_g<i>_t0`` after a build index, not after the machine
    # (``MT_g0_t0`` sits under ``G 01``), so any name-matching heuristic
    # silently finds nothing and leaves the unit transformer energised --
    # which would differ from the closed-loop contingency by one magnetising
    # branch, exactly the mismatch apply_contingency exists to prevent.
    def _terminal(obj, attr: str):
        try:
            cub = obj.GetAttribute(attr)
        except Exception:
            return None
        return getattr(cub, "cterm", None) if cub is not None else None

    tr2_all = [t for t in get_all(app, "ElmTr2")
               if t.loc_name.startswith("MT_")]
    tr2_by_lv = {}
    for t in tr2_all:
        term = _terminal(t, "buslv")
        if term is not None:
            tr2_by_lv.setdefault(getattr(term, "loc_name", id(term)), t)

    for sym in chosen:
        mva = _plant_mva(sym)
        mterm = _terminal(sym, "bus1")
        mt = (tr2_by_lv.get(getattr(mterm, "loc_name", None))
              if mterm is not None else None)
        if mt is None:
            print(f"[steps] WARNING: no ElmTr2 found on the LV terminal of "
                  f"{sym.loc_name!r}; its unit transformer will stay "
                  f"energised, unlike the closed-loop contingency. Check the "
                  f"MT_* topology before quoting this run.")
        also = (mt,) if mt is not None else ()
        tag = sym.loc_name.replace(" ", "")
        out.append(StepDef(
            name=f"outage_{tag}", target=sym, variable="", delta=0.0,
            unit="trip", kind="outage", also_trip=also, disturbance=True,
            note=f"outage of {sym.loc_name} ({mva:.0f} MVA)"
                 + (f" + unit trafo {mt.loc_name}" if mt is not None else
                    " (no unit trafo found -- trafo left energised)")))

    # Load step on the largest TN load: a bulk demand change, the other
    # disturbance class the dispatch battery cannot produce.
    loads = [l for l in get_all(app, "ElmLod")
             if l.GetAttribute("outserv") == 0]
    if loads:
        def _p(l) -> float:
            try:
                return abs(float(l.GetAttribute("plini")))
            except Exception:
                return 0.0
        big = max(loads, key=_p)
        for pct in LOAD_STEP_PERCENTS:
            sign = "+" if pct > 0 else "-"
            out.append(StepDef(
                name=f"load_{sign}{abs(pct):.0f}pct_{big.loc_name}",
                target=big, variable="", delta=0.0, unit="%", kind="load",
                load_pct=(pct, pct), disturbance=True,
                note=f"{pct:+.0f} % P and Q step on {big.loc_name} "
                     f"({_p(big):.0f} MW base)"))
    return out


# =====================================================================
#  Subcommands
# =====================================================================

def cmd_flat(ctx: ScreeningContext, out_dir: Path, duration: float = 60.0) -> int:
    mons = monitored_outputs(ctx.app)
    volt = [(o, v, l) for (o, v, l) in mons if v == "m:u"]
    # purge first: it resets the calculation, without which the stale-object
    # deletions in set_monitors would silently no-op (2026-07-20 finding).
    ctx.purge_events()
    ctx.set_monitors(volt)
    ctx.initialise()
    ctx.simulate(duration)
    worst_name, worst_drift = None, 0.0
    rows = []
    for obj, var, label in volt:
        t, y = ctx.read(obj, var)
        drift = max(y) - min(y)
        rows.append((label, drift))
        if drift > worst_drift:
            worst_drift, worst_name = drift, label
    ok = worst_drift < FLAT_DRIFT_TOL
    with (out_dir / "flat.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["signal", "drift_pu"])
        w.writerows(sorted(rows, key=lambda r: -r[1]))
    print(f"[flat] {duration:.0f} s run: max drift {worst_drift:.2e} pu "
          f"at {worst_name} -> {'PASS' if ok else 'FAIL'} (tol {FLAT_DRIFT_TOL:.0e})")
    return 0 if ok else 1


#: ComMod attributes that, where they exist, make the linearised system
#: matrices available on disk.  Spellings vary across PF versions and none is
#: in the scripting reference, so a superset is attempted and whatever sticks
#: is reported.  ``pf/probes/probe_modal_residue.py`` establishes which of
#: these the installed version actually has.
COMMOD_EXPORT_ATTRS: Tuple[Tuple[str, Any], ...] = (
    ("iExpMat", 1), ("iMatExp", 1), ("iExport", 1), ("iWriteMat", 1),
)


def cmd_modal(ctx: ScreeningContext, out_dir: Path,
              export_matrices: bool = False) -> int:
    """Eigenvalue screen at the RMS operating point.

    ``export_matrices`` additionally asks ComMod to write the linearised
    system matrices next to the table.  With an output matrix ``C`` for the
    monitored variables the per-mode settling below can be replaced by the
    residue-weighted **output** settling that the timescale premise actually
    concerns (Path A of ``pf/probes/probe_modal_residue.py``); without it the
    empirical route of ``analysis/modal_residue.py`` is required, which needs
    the step battery re-run with ``--save-trajectories``.
    """
    import math
    ctx.initialise()
    mod = ctx.app.GetFromStudyCase("ComMod")
    if export_matrices:
        applied = []
        for attr, val in COMMOD_EXPORT_ATTRS:
            try:
                mod.SetAttribute(attr, val)
                applied.append(attr)
            except Exception:
                continue
        for attr in ("dirMat", "dirMatrix", "sExpPath"):
            try:
                mod.SetAttribute(attr, str(out_dir))
                applied.append(attr)
            except Exception:
                continue
        if applied:
            print(f"[modal] matrix export: set {', '.join(applied)}")
        else:
            print("[modal] matrix export: NO usable export attribute on this "
                  "ComMod -- the exact residue path is unavailable; use "
                  "analysis/modal_residue.py on a battery run with "
                  "--save-trajectories")
    if mod.Execute():
        raise PFSessionError("ComMod failed")
    res = [r for r in get_all(ctx.app, "ElmRes") if "Modal" in r.loc_name][0]
    res.Load()
    n = res.GetNumberOfRows()
    modes = []
    for i in range(n):
        re = res.GetValue(i, 0)[1]
        im = res.GetValue(i, 1)[1]
        if im < -1e-9:
            continue
        wn = math.hypot(re, im)
        zeta = (-re / wn) if wn > 1e-9 else 1.0
        f = abs(im) / (2 * math.pi)
        ts = 4.0 / abs(re) if abs(re) > 1e-9 else float("inf")
        modes.append((ts, re, im, f, zeta))
    modes.sort(reverse=True)
    lines = ["# Modal analysis (full model, RMS operating point)", "",
             f"- eigenvalues: {n}; distinct modes: {len(modes)}",
             f"- unstable (Re>0): {sum(1 for m in modes if m[1] > 1e-6)}",
             f"- modes with T_s > 10 s: {sum(1 for m in modes if m[0] > 10)}",
             "",
             "**Interpretation limit.** `T_s = 4/|Re lambda|` is the settling "
             "of a *unit-amplitude* excitation of one mode. It is NOT the "
             "settling of the controlled outputs to a dispatch, which is what "
             "bounds `T_DS` and what the step battery measures. A mode is "
             "only relevant to that bound in proportion to the amplitude it "
             "actually reaches in the interface Q flows and the constrained "
             "bus voltages -- both of which are ALGEBRAIC variables and so "
             "have no row in this state-space result. Quoting the slowest "
             "row below as a lower bound on `T_DS` is therefore a category "
             "error; use `analysis/modal_residue.py` for the "
             "amplitude-weighted output settling.",
             "", "| T_s [s] | Re | Im | f [Hz] | zeta | kind |",
             "|--:|--:|--:|--:|--:|:--|"]
    for ts, re, im, f, z in modes:
        kind = "non-osc" if abs(im) < 1e-6 else "osc"
        ts_s = "inf" if ts == float("inf") else f"{ts:.2f}"
        lines.append(f"| {ts_s} | {re:.4f} | {im:.4f} | {f:.3f} | {z:.3f} | {kind} |")
    (out_dir / "modal.md").write_text("\n".join(lines), encoding="utf-8")
    slow = [m for m in modes if m[0] > 10 and m[1] < 0]
    print(f"[modal] {len(modes)} modes; {len(slow)} slower than 10 s; "
          f"min damping {min(m[4] for m in modes if abs(m[2])>0.1):.3f}; "
          f"table -> {out_dir/'modal.md'}")
    return 0


def cmd_steps(ctx: ScreeningContext, out_dir: Path,
              t_event: float = 5.0, horizon: float = 45.0,
              window: float = 20.0,
              disturbances: bool = False,
              outage_machines: Optional[Sequence[str]] = None,
              disturbance_horizon: float = 120.0,
              save_trajectories: bool = False) -> int:
    """Dispatch battery (Gate D) and, optionally, the disturbance battery.

    ``disturbances`` appends machine trips and load steps.  They are run with
    their own, longer horizon and are reported in a separate table: they are
    not dispatches, so they cannot bound ``T_DS`` and must not move the
    Gate-D verdict.

    ``save_trajectories`` additionally writes ``traj_<name>.csv`` per case
    (long format ``signal,t,y``).  Without it only summary statistics
    survive, and the modal content of the run -- what
    ``analysis/modal_residue.py`` needs to turn the modal screen into an
    observability statement about the controlled outputs -- is unrecoverable.
    The 2026-07-20 Gate-D run had no such option and is lost in that sense.
    """
    mons = monitored_outputs(ctx.app)
    catalogue = default_catalogue(ctx.app)
    if disturbances:
        catalogue = catalogue + disturbance_catalogue(ctx.app, outage_machines)
    summary = [f"# Step-response battery (event at t={t_event}s, {horizon}s run)",
               "",
               "Settling = time to stay within an absolute band of the final "
               "value (voltage 1e-3 pu, interface Q 1 Mvar). Gate D uses the "
               "controlled outputs; machine-speed ring is diagnostic.",
               "",
               "## Dispatches (Gate D)",
               "",
               "| step | actuator | worst ctrl output | T_s ctrl [s] | "
               "spd ring [s] | overshoot | step |",
               "|:--|:--|:--|--:|--:|--:|--:|"]
    dist_rows: List[str] = []
    def _read_scalar(obj, var):
        # DSL references (Qext) are signals: readable as 's:var' post-init,
        # not as a plain attribute.
        for a in (var, f"s:{var}", f"c:{var}"):
            try:
                return float(obj.GetAttribute(a))
            except Exception:
                continue
        raise PFSessionError(f"cannot read {var!r} on "
                             f"{getattr(obj, 'loc_name', obj)!r}")

    # Physically meaningful "settled" band per quantity, and whether the
    # signal is a controlled output (drives the Gate-D verdict) or a
    # diagnostic state (machine speed -- reported, not gated).
    def _floor(var: str) -> float:
        if var.startswith("m:u"):
            return 1e-3          # 0.1 % pu voltage band
        if var.startswith("m:Q"):
            return 1.0           # 1 Mvar interface-flow band
        return 1e-4              # machine speed (diagnostic)

    def _is_controlled(var: str) -> bool:
        return var.startswith(("m:u", "m:Q"))

    verdict_ok = True
    for sd in catalogue:
        target = sd.target
        # purge first: it resets the calculation, without which the
        # stale-monitor deletions in set_monitors silently no-op.
        ctx.purge_events()
        ctx.set_monitors(mons)
        ctx.initialise()
        if sd.kind == "param":
            cur = _read_scalar(target, sd.variable)
            ctx.add_param_event(target, sd.variable, cur + sd.delta, t_event)
        elif sd.kind == "tap":
            for i, dt in enumerate(sd.tap_times):
                ctx.add_tap_event(target, int(sd.delta), t_event + dt, seq=i)
        elif sd.kind == "outage":
            # The unit transformer goes with the machine, as in
            # PowerFactoryPlant.apply_contingency -- leaving it energised
            # would differ from the closed-loop contingency by one
            # magnetising branch.
            ctx.add_outage_event(target, t_event, seq=0)
            for i, extra in enumerate(sd.also_trip, start=1):
                ctx.add_outage_event(extra, t_event, seq=i)
        elif sd.kind == "load":
            d_p, d_q = sd.load_pct
            ctx.add_load_event(target, d_p, d_q, t_event)
        else:
            raise PFSessionError(f"unknown StepDef.kind {sd.kind!r}")
        run_horizon = disturbance_horizon if sd.disturbance else horizon
        ctx.simulate(t_event + run_horizon)
        rows = []
        traj: List[Tuple[str, float, float]] = []
        for obj, var, label in mons:
            try:
                t, y = ctx.read(obj, var)
            except Exception:
                # An outage removes the element's m: variables entirely
                # (add_outage_event docstring): a tripped machine's speed is
                # simply gone, which is the documented detection path and not
                # an error.
                if sd.kind == "outage":
                    continue
                raise
            m = settling_metrics(t, y, t_event, abs_floor=_floor(var))
            m["controlled"] = _is_controlled(var)
            rows.append((label, m))
            if save_trajectories and _is_controlled(var):
                traj.extend((label, tt, yy) for tt, yy in zip(t, y))
        if save_trajectories and traj:
            with (out_dir / f"traj_{sd.name}.csv").open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["signal", "t", "y"])
                for label, tt, yy in traj:
                    w.writerow([label, f"{tt:.4f}", f"{yy:.9g}"])
        rows.sort(key=lambda r: -r[1]["t_settle"])
        ctrl = [(l, m) for l, m in rows if m["controlled"]]
        spd = [(l, m) for l, m in rows if not m["controlled"]]
        worst_label, worst = ctrl[0]          # worst controlled output
        spd_worst = spd[0][1]["t_settle"] if spd else 0.0
        with (out_dir / f"step_{sd.name}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["signal", "controlled", "y_init", "y_final", "step",
                        "t_settle_s", "overshoot"])
            for label, m in rows:
                w.writerow([label, int(m["controlled"]),
                            f"{m['y_init']:.6f}", f"{m['y_final']:.6f}",
                            f"{m['step']:.6f}", f"{m['t_settle']:.3f}",
                            f"{m['overshoot']:.4f}"])
        within = worst["t_settle"] <= window
        row = (f"| {sd.name} | {sd.note} | "
               f"{worst_label} | {worst['t_settle']:.2f} | {spd_worst:.1f} | "
               f"{worst['overshoot']:.3f} | {worst['step']:.4f} |")
        if sd.disturbance:
            # Reported, never gated: a disturbance is not a dispatch, so its
            # settling time does not bound T_DS.  What it bounds is how long
            # the OFO integrates against a plant that is not yet the
            # algebraic map its sensitivities assume.
            dist_rows.append(
                row + f" {'' if within else ' **>window**'}".rstrip())
            print(f"[steps] {sd.name} (disturbance, not gated): worst "
                  f"controlled-output settling {worst['t_settle']:.2f}s at "
                  f"{worst_label}; {'within' if within else 'BEYOND'} "
                  f"{window:.0f}s; machine-speed ring {spd_worst:.1f}s")
            continue
        verdict_ok = verdict_ok and within
        summary.append(row)
        flag = "OK" if within else f">{window:.0f}s"
        print(f"[steps] {sd.name}: worst controlled-output settling "
              f"{worst['t_settle']:.2f}s at {worst_label} [{flag}]; "
              f"machine-speed ring {spd_worst:.1f}s")
    summary.append("")
    summary.append(f"**Gate D (open-loop {window:.0f} s, controlled outputs): "
                   f"{'PASS' if verdict_ok else 'FAIL — output(s) settle > window'}**")
    summary.append("")
    summary.append("Machine-speed ring (diagnostic, not gated) reflects the "
                   "poorly-damped electromechanical modes; see modal.md.")
    if dist_rows:
        summary += [
            "",
            f"## Disturbances (not dispatches -- reported, NOT gated; "
            f"{disturbance_horizon:.0f}s run)",
            "",
            "A machine trip or a load step is not an actuator command, so "
            "its settling time does not bound `T_DS`. What it measures is "
            "how long after a credible event the plant is not the settled "
            "algebraic map the OFO's sensitivity model assumes -- i.e. over "
            "how many dispatch intervals the supervisory layer integrates "
            "against a moving plant. `G 01` is refused as an outage target: "
            "it is the 10 GVA interconnection equivalent and the angle "
            "reference, not a contingency.",
            "",
            "| case | disturbance | worst ctrl output | T_s ctrl [s] | "
            "spd ring [s] | overshoot | step |",
            "|:--|:--|:--|--:|--:|--:|--:|"]
        summary += dist_rows
    (out_dir / "steps.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"[steps] summary -> {out_dir/'steps.md'} "
          f"({'PASS' if verdict_ok else 'FAIL'})")
    return 0 if verdict_ok else 1


# =====================================================================
#  CLI
# =====================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RMS screening battery (Gate D).")
    parser.add_argument("command", choices=("flat", "modal", "steps"))
    parser.add_argument("--label", default="full_t0",
                        help="snapshot label for the results folder")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--window", type=float, default=20.0,
                        help="dispatch window [s] for the Gate-D verdict")
    parser.add_argument("--project", default=DEFAULT_PROJECT_PATH)
    parser.add_argument("--disturbances", action="store_true",
                        help="steps: also run machine trips and load steps. "
                             "Reported separately; they are not dispatches "
                             "and do not move the Gate-D verdict.")
    parser.add_argument("--outage-machines", nargs="*", default=None,
                        metavar="NAME",
                        help="steps: ElmSym loc_names to trip, e.g. "
                             "'G 07' 'G 09'. Default: the three largest "
                             "in-service machines. 'G 01' is refused -- it "
                             "is the 10 GVA interconnection equivalent and "
                             "the angle reference, not a contingency.")
    parser.add_argument("--outage-gens", nargs="*", type=int, default=None,
                        metavar="IDX",
                        help="steps: machines by PANDAPOWER gen index, the "
                             "convention the contingency configs and the "
                             "dead-band N-1 study use. NOTE the numberings "
                             "are offset: gen[1] is 'G 03' and gen[7] is "
                             "'G 09'. Each resolution is printed.")
    parser.add_argument("--disturbance-horizon", type=float, default=600.0,
                        help="steps: run length [s] after a disturbance. "
                             "Default 600 (30 dispatch intervals): measured "
                             "2026-08-03 on the dead-band N-1 corpus, a "
                             "credible trip leaves the controlled outputs "
                             "outside their bands for ~170 s, so a 120 s "
                             "horizon would truncate the recovery.")
    parser.add_argument("--save-trajectories", action="store_true",
                        help="steps: persist per-signal time series for the "
                             "residue analysis (analysis/modal_residue.py). "
                             "Without this only summary rows survive.")
    parser.add_argument("--export-matrices", action="store_true",
                        help="modal: also ask ComMod to write the linearised "
                             "system matrices (enables exact residues)")
    args = parser.parse_args(argv)

    app = connect(args.project, study_case=RMS_STUDY_CASE)
    ctx = ScreeningContext(app)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (Path(__file__).resolve().parents[1] / "results" / "screening"
               / args.label / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[screening] {args.command} -> {out_dir}")

    if args.command == "flat":
        return cmd_flat(ctx, out_dir, args.duration)
    if args.command == "modal":
        return cmd_modal(ctx, out_dir, export_matrices=args.export_matrices)
    targets = resolve_outage_targets(args.outage_machines, args.outage_gens)
    return cmd_steps(ctx, out_dir, window=args.window,
                     disturbances=args.disturbances,
                     outage_machines=targets or None,
                     disturbance_horizon=args.disturbance_horizon,
                     save_trajectories=args.save_trajectories)


if __name__ == "__main__":
    sys.exit(main())
