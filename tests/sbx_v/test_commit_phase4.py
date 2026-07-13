"""
SBX-V Phase 4 tests — commit-instant integration (plan §9 Phase 4).

Covers:

* **R3**: the priced segment structure changes ONLY at scheduled
  instants — identity of the spec tuple within a window stretch, the
  deterministic expiry-ramp schedule over the final ``ramp_steps``
  iterations, no ramp when a follow-up grant is confirmed, and the
  neutral bypass (no grants ⇒ ``None`` ⇒ R1 pass-through);
* the scheduled-envelope feedforward (MSR/MSC pattern): steps exactly
  at commit instants and during ramps, zero elsewhere;
* end-to-end need flag → pipeline → activation at the next boundary
  through ``CommitScheduler.step``;
* the Incapability path: declaration → Reserve-Observer log →
  ``IncapabilityRecord`` → settlement Tabelle 8.1 case 3a (and the
  DSO-delivers variant staying a logged event, STATUS §2.2);
* the flag-gated Notfall-Abruf (``emergency.py``), including R3
  preservation (an emergency call never rebuilds specs).
"""

from __future__ import annotations

import pytest

from sbx_h.fail import SBXError
from sbx_v.band import NormalBand
from sbx_v.commit import AreaIterationInput, CommitScheduler
from sbx_v.config import SBXVConfig
from sbx_v.directions import Direction
from sbx_v.emergency import EmergencyCall, EmergencyHandler
from sbx_v.grants_ledger import GrantsLedger
from sbx_v.messages import (FeasibilityReply, GrantConfirmation,
                           IncapabilityDeclaration, PotentialMessage,
                           Window)
from sbx_v.settlement import (CASE_81_3A, DSO_DELIVERS, TSO_DELIVERS,
                             GrantRecord, SettlementEngine,
                             WindowObservation)

CFG = SBXVConfig()                     # k_window = 5, ramp_steps = 3
BAND = NormalBand("a1", 50.0, 50.0)


def _scheduler(config: SBXVConfig = CFG) -> CommitScheduler:
    return CommitScheduler(config, {"a1": BAND}, {"a1": (0, 1, 2)})


def _confirm(sched: CommitScheduler, window_index: int,
             n_quanta: int = 1,
             direction: Direction = Direction.LOWERING) -> None:
    """Write one confirmed grant for ``window_index`` into the ledger
    (bypassing the pipeline — ledger-level test setup)."""
    w = sched.window_obj(window_index)
    reply = FeasibilityReply(
        request_id=f"req:a1:{direction.value}:w{window_index}",
        verdict="ACCEPT", n_quanta_offered=n_quanta)
    sched.ledger.note_feasibility("a1", direction, window_index, reply)
    sched.ledger.confirm(GrantConfirmation(
        order_id=f"ord:req:a1:{direction.value}:w{window_index}",
        aggregation_area_id="a1", direction=direction,
        n_quanta=n_quanta, window=w))


def _forecast(area_id: str, direction: Direction,
              window: Window) -> PotentialMessage:
    return PotentialMessage(
        aggregation_area_id=area_id, direction=direction,
        q_pot_mvar=150.0, q_vh_flagged_mvar=0.0, window=window,
        is_forecast=True)


# ======================================================================
#  R3 — spec schedule
# ======================================================================

class TestR3SpecSchedule:
    def test_neutral_bypass_only_when_pricing_disabled(self):
        # R1 arm: explicit neutral configuration → bypass everywhere.
        sched = _scheduler(SBXVConfig(miqp_pricing_enabled=False))
        _confirm(sched, window_index=1)
        for k in range(10):
            assert sched.specs_for(k) is None

    def test_band_priced_from_window_zero_without_grant(self):
        # V-D9: '0 in band, Grenzpreis beyond, when no grant is active'
        # — the band prices act from the first window onward.
        sched = _scheduler()
        specs = sched.specs_for(0)
        assert specs is not None
        (spec,) = specs
        assert spec.lowering.anchor_mvar == 50.0
        assert len(spec.lowering.segments) == 1      # open Grenzpreis tail
        assert spec.raising.anchor_mvar == 50.0

    def test_identity_within_the_main_stretch(self):
        sched = _scheduler()
        _confirm(sched, window_index=1)
        # Window 1 = iterations 5..9; ramp start = 10 − 3 = 7.
        s5 = sched.specs_for(5)
        s6 = sched.specs_for(6)
        assert s5 is s6                          # R3 identity
        assert s5 is not None

    def test_grant_activates_only_at_the_boundary(self):
        sched = _scheduler()
        _confirm(sched, window_index=1)
        # Window 0: the grant is NOT yet active — no-grant band spec.
        (spec0,) = sched.specs_for(4)
        assert len(spec0.lowering.segments) == 1
        assert spec0.lowering.anchor_mvar == 50.0
        specs = sched.specs_for(5)               # window 1: active
        (spec,) = specs
        # Leitfaden model with an active 30 Mvar LOWERING grant:
        # opposite-edge anchor, Durchschnitt span 50+50+30 = 130 Mvar.
        assert spec.lowering.anchor_mvar == -50.0
        assert spec.lowering.segments[0].width_mvar == pytest.approx(130.0)

    def test_expiry_ramp_schedule(self):
        sched = _scheduler()
        _confirm(sched, window_index=1)          # no follow-up → ramp
        # Effective grant over the final ramp_steps=3 iterations of
        # window 1 (k = 7, 8, 9): 30·(9−k)/3 → 20, 10, 0.
        for k, eff in ((5, 30.0), (6, 30.0), (7, 20.0), (8, 10.0),
                       (9, 0.0)):
            assert sched._effective_grant_mvar(
                "a1", Direction.LOWERING, k) == pytest.approx(eff), k
        # Spec widths follow: 100 + eff during the ramp.
        for k, width in ((7, 120.0), (8, 110.0)):
            (spec,) = sched.specs_for(k)
            assert spec.lowering.segments[0].width_mvar == \
                pytest.approx(width)
        # At k = 9 the effective grant is zero → no-grant side spec
        # (own-edge anchor, single open Grenzpreis segment).
        (spec9,) = sched.specs_for(9)
        assert spec9.lowering.anchor_mvar == 50.0
        assert len(spec9.lowering.segments) == 1

    def test_no_ramp_when_follow_up_confirmed(self):
        sched = _scheduler()
        _confirm(sched, window_index=1)
        _confirm(sched, window_index=2)          # re-issue → no ramp
        for k in range(5, 10):
            assert sched._effective_grant_mvar(
                "a1", Direction.LOWERING, k) == pytest.approx(30.0)
        # Whole window is ONE stretch (identity 5..9).
        assert sched.specs_for(5) is sched.specs_for(9)

    def test_ramp_decision_is_frozen(self):
        # A follow-up confirmed AFTER the ramp start does not cancel
        # the already-scheduled ramp (determinism, R3).
        sched = _scheduler()
        _confirm(sched, window_index=1)
        assert sched._effective_grant_mvar(
            "a1", Direction.LOWERING, 7) == pytest.approx(20.0)
        _confirm(sched, window_index=2)          # too late for the ramp
        assert sched._effective_grant_mvar(
            "a1", Direction.LOWERING, 8) == pytest.approx(10.0)

    def test_ramp_steps_bounded_by_window(self):
        with pytest.raises(SBXError):
            _scheduler(SBXVConfig(ramp_steps=6))  # > k_window = 5


# ======================================================================
#  Scheduled-envelope feedforward
# ======================================================================

class TestEnvelopeFeedforward:
    def test_steps_only_at_scheduled_instants(self):
        sched = _scheduler()
        _confirm(sched, window_index=1)
        steps = {k: sched.envelope_step_mvar("a1", k)
                 for k in range(0, 10)}
        # Activation at the commit instant k = 5: +30 (LOWERING, +1).
        assert steps[5] == pytest.approx(+30.0)
        # Ramp decrements at k = 7, 8, 9: −10 each.
        for k in (7, 8, 9):
            assert steps[k] == pytest.approx(-10.0)
        # Zero everywhere else (piecewise-constant envelope).
        for k in (0, 1, 2, 3, 4, 6):
            assert steps[k] == 0.0

    def test_neutral_configuration_is_zero_change(self):
        sched = _scheduler()
        for k in range(0, 10):
            assert sched.envelope_step_mvar("a1", k) == 0.0


# ======================================================================
#  End-to-end: need flag → pipeline → activation at the boundary
# ======================================================================

class TestStepToActivation:
    def test_flag_requests_next_window_and_activates_at_boundary(self):
        sched = _scheduler()
        # Both conditions set from iteration 2 (n_persist = 1): netted
        # dispatch saturated at the band edge AND overvoltage beyond
        # the threshold → LOWERING need.
        stressed = AreaIterationInput(
            q_pcc_netted_mvar=49.5,      # within sat_tol of edge 50
            v_dev_raising_pu=0.0,
            v_dev_lowering_pu=0.006)     # > 0.005
        calm = AreaIterationInput(
            q_pcc_netted_mvar=10.0,
            v_dev_raising_pu=0.0, v_dev_lowering_pu=0.0)
        sched.step(0, {"a1": calm}, _forecast)
        sched.step(1, {"a1": calm}, _forecast)
        decisions = sched.step(2, {"a1": stressed}, _forecast)
        assert decisions[("a1", Direction.LOWERING)].flag
        # The pipeline requested and confirmed for window 1 …
        assert sched.ledger.granted_mvar(
            "a1", Direction.LOWERING, 1) == pytest.approx(30.0)
        # … but window 0 keeps the no-grant spec: activation ONLY at
        # the boundary (the granted Durchschnitt segment appears at
        # k = 5, not before).
        for k in (2, 3):
            (spec,) = sched.specs_for(k)
            assert len(spec.lowering.segments) == 1
        (spec5,) = sched.specs_for(5)
        assert len(spec5.lowering.segments) == 2

    def test_one_request_per_window_reissue_later(self):
        sched = _scheduler()
        stressed = AreaIterationInput(
            q_pcc_netted_mvar=49.5, v_dev_raising_pu=0.0,
            v_dev_lowering_pu=0.006)
        for k in range(0, 5):
            sched.step(k, {"a1": stressed}, _forecast)
        # Window 0's flag issued exactly ONE request for window 1.
        assert sched.ledger.granted_mvar(
            "a1", Direction.LOWERING, 1) == pytest.approx(30.0)
        # Continuing stress in window 1 re-issues for window 2.
        for k in range(5, 10):
            sched.step(k, {"a1": stressed}, _forecast)
        assert sched.ledger.granted_mvar(
            "a1", Direction.LOWERING, 2) == pytest.approx(30.0)


# ======================================================================
#  Incapability path → settlement case 3a
# ======================================================================

class TestIncapabilityPath:
    def _sched_with_grant(self):
        sched = _scheduler()
        _confirm(sched, window_index=1)
        return sched

    def test_declaration_without_grant_fails_fast(self):
        sched = _scheduler()
        with pytest.raises(SBXError):
            sched.declare_incapability(IncapabilityDeclaration(
                aggregation_area_id="a1",
                direction=Direction.LOWERING,
                q_shortfall_mvar=10.0,
                window=sched.window_obj(1)))

    def test_tso_delivers_propagates_to_case_3a(self):
        sched = self._sched_with_grant()
        sched.declare_incapability(IncapabilityDeclaration(
            aggregation_area_id="a1", direction=Direction.LOWERING,
            q_shortfall_mvar=10.0, window=sched.window_obj(1)))
        grant_records = (GrantRecord(
            area_id="a1", direction=Direction.LOWERING,
            q_grant_mvar=30.0, delivering_party=TSO_DELIVERS,
            window_first=1, window_end=2),)
        (record,) = sched.to_incapability_records(grant_records)
        assert record.q_vh_provided_mvar == pytest.approx(20.0)
        # End-to-end through the Phase-2 engine: Tabelle 8.1 case 3a.
        engine = SettlementEngine(CFG, {"a1": BAND})
        result = engine.settle(
            [WindowObservation(area_id="a1", window_index=1,
                               t_start_s=900.0, q_meas_mvar=70.0,
                               q_set_mvar=None)],
            grant_records, [record])
        rows = [r for r in result.window_rows if r.case == CASE_81_3A]
        assert len(rows) == 1
        assert rows[0].cap_frac == pytest.approx(20.0 / 30.0)

    def test_dso_delivers_stays_a_logged_event(self):
        sched = self._sched_with_grant()
        sched.declare_incapability(IncapabilityDeclaration(
            aggregation_area_id="a1", direction=Direction.LOWERING,
            q_shortfall_mvar=10.0, window=sched.window_obj(1)))
        records = sched.to_incapability_records(
            sched.ledger.to_grant_records(
                delivering_party=DSO_DELIVERS))
        assert records == ()                    # Tabelle 8.2 case 2 via
        #                                         metering instead
        assert any(e[0] == "incapability" for e in sched.log)


# ======================================================================
#  Emergency (flag-gated)
# ======================================================================

class TestEmergency:
    CALL = EmergencyCall(aggregation_area_id="a1",
                         direction=Direction.LOWERING,
                         q_req_mvar=25.0)

    def test_disabled_by_default_fails_fast(self):
        handler = EmergencyHandler(SBXVConfig())
        with pytest.raises(SBXError):
            handler.call(self.CALL, k_now=3, k_window_end=5)

    def test_enabled_activation_window(self):
        handler = EmergencyHandler(
            SBXVConfig(emergency_call_enabled=True))
        handler.call(self.CALL, k_now=3, k_window_end=5)
        f = handler.active_extra_mvar
        assert f("a1", Direction.LOWERING, 2) == 0.0
        assert f("a1", Direction.LOWERING, 3) == 25.0
        assert f("a1", Direction.LOWERING, 4) == 25.0
        assert f("a1", Direction.LOWERING, 5) == 0.0   # window end
        assert f("a1", Direction.RAISING, 3) == 0.0
        assert handler.log == [("emergency", "a1", "lowering", 25.0,
                                3, 5)]

    def test_emergency_never_rebuilds_specs(self):
        # Open-tail design: everything beyond band + grant is already
        # Grenzpreis-priced, so the Notfall-Abruf must not change the
        # spec (R3 preserved by construction).
        cfg = SBXVConfig(emergency_call_enabled=True)
        handler = EmergencyHandler(cfg)
        sched = CommitScheduler(cfg, {"a1": BAND}, {"a1": (0,)},
                                emergency=handler)
        _confirm(sched, window_index=1)
        before = sched.specs_for(5)
        handler.call(self.CALL, k_now=5, k_window_end=10)
        assert sched.specs_for(6) is before      # identity, unchanged


# ======================================================================
#  Config validators added for the Phase-4 knobs
# ======================================================================

class TestPhase4ConfigValidation:
    def test_n_clear_positive(self):
        with pytest.raises(SBXError):
            SBXVConfig(n_clear=0)

    def test_sat_tol_non_negative(self):
        with pytest.raises(SBXError):
            SBXVConfig(sat_tol_mvar=-1.0)

    def test_v_dev_threshold_positive(self):
        with pytest.raises(SBXError):
            SBXVConfig(v_dev_threshold_pu=0.0)

    def test_reserve_margin_non_negative(self):
        with pytest.raises(SBXError):
            SBXVConfig(reserve_margin_mvar=-5.0)
