"""
SBX-V Phase 2 tests — metering and settlement (plan §7, V-D8).

Encodes the Leitfaden VERBATIM where the plan demands it:

* the worked example [LF §8.2]: band ±50 Mvar, grant 100 Mvar beyond
  the upper edge ⇒ Vorhalteleistung = 200 Mvar;
* the Saldierung numeric example [LF §8.3, Abb. 8.3] (4 Zählpunkte ×
  6 quarter-hours ⇒ netted means 16, −16, −20, 4, −8, −4);
* one test per Fall-Kategorie of Tabelle 8.1 (1, 2, 3, 3a, 4, 4a) and
  Tabelle 8.2 (1, 2, 3, 4), plus the ad-hoc interpretation case;
* tolerance [LF §4.7] and Leistungspreis suspension [LF §7.5.4].
"""

from __future__ import annotations

import csv

import pytest

from sbx_h.fail import SBXError
from sbx_v.band import NormalBand
from sbx_v.config import SBXVConfig
from sbx_v.directions import Direction
from sbx_v.metering import (AreaMeter, QuadrantRegisters,
                           aggregate_quadrant_registers)
from sbx_v.settlement import (CASE_81_2, CASE_81_3, CASE_81_3A, CASE_81_4,
                             CASE_81_4A, CASE_82_1, CASE_82_2, CASE_82_3,
                             CASE_82_4, CASE_82_ADHOC, CASE_FREE,
                             DSO_DELIVERS, TSO_DELIVERS, GrantRecord,
                             IncapabilityRecord, SettlementEngine,
                             WindowObservation, write_settlement_csv)

CFG = SBXVConfig()          # window 900 s (h = 0.25), tol 10 %,
#                             arb 5/10 €/Mvarh, LP 25/50 €/(Mvar·day)
BAND = NormalBand("a1", 50.0, 50.0)
H = 0.25


def _engine() -> SettlementEngine:
    return SettlementEngine(CFG, {"a1": BAND})


def _obs(q_meas, q_set=None, window=0):
    return WindowObservation(area_id="a1", window_index=window,
                             t_start_s=window * 900.0,
                             q_meas_mvar=q_meas, q_set_mvar=q_set)


def _grant(mvar, party, direction=Direction.LOWERING, first=0, end=1):
    return GrantRecord(area_id="a1", direction=direction,
                       q_grant_mvar=mvar, delivering_party=party,
                       window_first=first, window_end=end)


def _row(result, case):
    rows = [r for r in result.window_rows if r.case == case]
    assert len(rows) == 1, [r.case for r in result.window_rows]
    return rows[0]


# ======================================================================
#  Metering
# ======================================================================

class TestAreaMeter:
    def test_constant_signal(self):
        m = AreaMeter("a1", n_nvp=1, window_s=900.0)
        for k in range(15):
            m.record_step(60.0 * k, 60.0, [20.0])
        (reg,) = m.finalise()
        assert reg.q_mean_mvar == pytest.approx(20.0)
        assert reg.e_q_pos_mvarh == pytest.approx(20.0 * H)
        assert reg.e_q_neg_mvarh == 0.0

    def test_sign_change_splits_registers(self):
        # Four-quadrant metering: 7 min at +30, 8 min at −30.
        m = AreaMeter("a1", n_nvp=1, window_s=900.0)
        for k in range(7):
            m.record_step(60.0 * k, 60.0, [+30.0])
        for k in range(7, 15):
            m.record_step(60.0 * k, 60.0, [-30.0])
        (reg,) = m.finalise()
        assert reg.e_q_pos_mvarh == pytest.approx(30.0 * 7 / 60.0)
        assert reg.e_q_neg_mvarh == pytest.approx(30.0 * 8 / 60.0)
        assert reg.q_mean_mvar == pytest.approx(
            (reg.e_q_pos_mvarh - reg.e_q_neg_mvarh) / H)

    def test_multi_nvp_netting(self):
        m = AreaMeter("a1", n_nvp=3, window_s=900.0)
        for k in range(15):
            m.record_step(60.0 * k, 60.0, [+30.0, -10.0, -5.0])
        (reg,) = m.finalise()
        assert reg.q_mean_mvar == pytest.approx(15.0)

    def test_gap_rejected(self):
        m = AreaMeter("a1", n_nvp=1, window_s=900.0)
        m.record_step(0.0, 60.0, [1.0])
        with pytest.raises(SBXError):
            m.record_step(120.0, 60.0, [1.0])   # 60 s gap

    def test_straddling_interval_rejected(self):
        m = AreaMeter("a1", n_nvp=1, window_s=900.0)
        m.record_step(0.0, 700.0, [1.0])
        with pytest.raises(SBXError):
            m.record_step(700.0, 400.0, [1.0])  # crosses t = 900 s

    def test_incomplete_tail_not_settled(self):
        m = AreaMeter("a1", n_nvp=1, window_s=900.0)
        for k in range(10):
            m.record_step(60.0 * k, 60.0, [1.0])
        assert m.finalise() == []
        assert m.incomplete_tail_s == pytest.approx(600.0)


class TestLeitfadenSaldierungExample:
    """[LF §8.3, Abb. 8.3] verbatim: per-Zählpunkt work registers of the
    four NVPs, six quarter-hours; expected netted means (work → power:
    factor 4 at 15 min) are 16, −16, −20, 4, −8, −4."""

    # Per window: [(Q1+Q2, Q3+Q4) per Zählpunkt 1, 2, 3, n].
    LF_REGISTERS = [
        [(6, 3), (4, 3), (0, 2), (3, 1)],    # 1. 1/4 h
        [(0, 1), (0, 3), (3, 2), (0, 1)],    # 2. 1/4 h
        [(0, 3), (0, 1), (0, 3), (3, 1)],    # 3. 1/4 h
        [(2, 1), (0, 3), (0, 1), (7, 3)],    # 4. 1/4 h
        [(0, 2), (5, 1), (0, 3), (0, 1)],    # 5. 1/4 h
        [(5, 3), (0, 2), (3, 1), (0, 3)],    # 6. 1/4 h
    ]
    LF_Q_MEANS = [16.0, -16.0, -20.0, 4.0, -8.0, -4.0]

    def test_verbatim(self):
        for w, (regs, expected) in enumerate(
                zip(self.LF_REGISTERS, self.LF_Q_MEANS)):
            area = aggregate_quadrant_registers(
                area_id="agg", window_index=w, t_start_s=w * 900.0,
                window_s=900.0,
                per_nvp=[QuadrantRegisters(p, n) for p, n in regs])
            assert area.q_mean_mvar == pytest.approx(expected), f"window {w}"


# ======================================================================
#  Worked example [LF §8.2]
# ======================================================================

class TestWorkedExample:
    def test_vorhalteleistung_200_mvar(self):
        # Band ±50 Mvar, grant 100 Mvar beyond the upper edge, DSO
        # delivers ⇒ Vorhalteleistung = 200 Mvar (opposite edge to
        # grant maximum).
        res = _engine().settle(
            [_obs(q_meas=100.0, q_set=100.0)],
            [_grant(100.0, DSO_DELIVERS)])
        (day,) = res.day_rows
        assert day.vh_mvar == pytest.approx(200.0)
        assert day.pay_cap_avg_eur == pytest.approx(
            200.0 * CFG.price_lp_avg_eur_per_mvar_day)


# ======================================================================
#  Tabelle 8.2 — DSO delivers (primary direction)
# ======================================================================

class Test82:
    """Grant 100 Mvar (LOWERING) ⇒ VH = 200, grant max s = 150,
    tolerance = 20 Mvar."""

    GRANT = 100.0

    def _settle(self, q_meas, q_set, incap=()):
        return _engine().settle([_obs(q_meas, q_set)],
                                [_grant(self.GRANT, DSO_DELIVERS)],
                                incap)

    def test_case_1_correct_delivery(self):
        res = self._settle(q_meas=95.0, q_set=100.0)   # dev 5 < 20
        r = _row(res, CASE_82_1)
        # Energy from the OPPOSITE edge (−50) to the operating point.
        assert r.e_avg_mvarh == pytest.approx((95.0 + 50.0) * H)
        assert r.pay_energy_avg_eur == pytest.approx(145.0 * H * 5.0)
        assert r.cap_frac == 1.0
        (day,) = res.day_rows
        assert day.pay_cap_avg_eur == pytest.approx(200.0 * 25.0)

    def test_case_1_energy_reference_even_in_band(self):
        # With an active grant the opposite-edge reference applies even
        # for an in-band operating point (Tabelle 8.2 case 1 carries no
        # in-band deduction — [LF §7.1] deducts only when the UPSTREAM
        # delivers).
        res = self._settle(q_meas=0.0, q_set=0.0)
        r = _row(res, CASE_82_1)
        assert r.e_avg_mvarh == pytest.approx(50.0 * H)

    def test_case_2_under_delivery_pro_rata(self):
        res = self._settle(q_meas=100.0, q_set=140.0)  # dev 40 > 20
        r = _row(res, CASE_82_2)
        assert r.e_avg_mvarh == pytest.approx(150.0 * H)
        assert r.cap_frac == pytest.approx(150.0 / 200.0)
        (day,) = res.day_rows
        assert day.day_frac == pytest.approx(0.75)
        assert day.pay_cap_avg_eur == pytest.approx(200.0 * 0.75 * 25.0)

    def test_case_2_full_suspension_when_nothing_provided(self):
        # Operating point beyond the OPPOSITE edge: nothing provided on
        # the granted LOWERING side (full suspension) — AND the RAISING
        # band edge is violated on the DSO's own initiative, which
        # correctly yields an additional Tabelle 8.1 case-4 row.
        res = self._settle(q_meas=-60.0, q_set=100.0)
        r = _row(res, CASE_82_2)
        assert r.cap_frac == 0.0
        assert r.e_avg_mvarh == 0.0
        _row(res, CASE_81_4)
        (day_grant,) = [d for d in res.day_rows
                        if d.world == DSO_DELIVERS]
        assert day_grant.pay_cap_avg_eur == 0.0

    def test_case_3_over_delivery_capped_at_sollwert(self):
        res = self._settle(q_meas=120.0, q_set=80.0)   # over by 40 > 20
        r = _row(res, CASE_82_3)
        # Energy only up to the Sollwert [LF §7.5.5 / Tabelle 8.2-3].
        assert r.e_avg_mvarh == pytest.approx((80.0 + 50.0) * H)
        assert r.cap_frac == 1.0

    def test_case_4_call_beyond_grant(self):
        res = self._settle(q_meas=170.0, q_set=170.0)  # beyond 150
        r = _row(res, CASE_82_4)
        assert r.e_avg_mvarh == pytest.approx((150.0 + 50.0) * H)
        assert r.e_grenz_mvarh == pytest.approx(20.0 * H)
        assert r.pay_energy_grenz_eur == pytest.approx(20.0 * H * 10.0)
        assert r.cap_exceed_mvar == pytest.approx(20.0)
        (day,) = res.day_rows
        assert day.pay_cap_avg_eur == pytest.approx(200.0 * 25.0)
        assert day.pay_cap_grenz_eur == pytest.approx(20.0 * 50.0)

    def test_tolerance_boundary(self):
        # ±10 % of VH = 20 Mvar [LF §4.7]: 19 under → still case 1;
        # 21 under → case 2.
        res = self._settle(q_meas=81.0, q_set=100.0)
        _row(res, CASE_82_1)
        res = self._settle(q_meas=79.0, q_set=100.0)
        _row(res, CASE_82_2)

    def test_missing_abruf_fails_fast(self):
        with pytest.raises(SBXError):
            self._settle(q_meas=95.0, q_set=None)

    def test_raising_mirror(self):
        res = _engine().settle(
            [_obs(q_meas=-95.0, q_set=-100.0)],
            [_grant(self.GRANT, DSO_DELIVERS,
                    direction=Direction.RAISING)])
        r = _row(res, CASE_82_1)
        assert r.direction is Direction.RAISING
        assert r.e_avg_mvarh == pytest.approx((95.0 + 50.0) * H)


# ======================================================================
#  Tabelle 8.1 — TSO delivers (reverse direction)
# ======================================================================

class Test81:
    def test_case_1_free_band_no_grant(self):
        res = _engine().settle([_obs(q_meas=20.0)])
        r = _row(res, CASE_FREE)
        assert r.pay_energy_avg_eur == 0.0
        assert r.pay_energy_grenz_eur == 0.0
        assert res.day_rows == ()
        assert res.totals == ()

    def test_case_2_grant_in_band_capacity_only(self):
        res = _engine().settle([_obs(q_meas=30.0)],
                               [_grant(30.0, TSO_DELIVERS)])
        r = _row(res, CASE_81_2)
        assert r.e_avg_mvarh == 0.0
        (day,) = res.day_rows
        assert day.vh_mvar == pytest.approx(30.0)   # same-side reference
        assert day.pay_cap_avg_eur == pytest.approx(30.0 * 25.0)

    def test_case_3_call_within_grant(self):
        res = _engine().settle([_obs(q_meas=70.0)],
                               [_grant(30.0, TSO_DELIVERS)])
        r = _row(res, CASE_81_3)
        # Energy beyond the OWN edge only ([LF §7.1] in-band deduction).
        assert r.e_avg_mvarh == pytest.approx(20.0 * H)
        assert r.pay_energy_avg_eur == pytest.approx(20.0 * H * 5.0)

    def test_case_3a_incapability_suspends_capacity(self):
        incap = IncapabilityRecord(area_id="a1",
                                   direction=Direction.LOWERING,
                                   window_index=0,
                                   q_vh_provided_mvar=15.0)
        res = _engine().settle([_obs(q_meas=70.0)],
                               [_grant(30.0, TSO_DELIVERS)], [incap])
        r = _row(res, CASE_81_3A)
        assert r.cap_frac == pytest.approx(0.5)
        assert r.e_avg_mvarh == pytest.approx(20.0 * H)  # still avg price
        (day,) = res.day_rows
        assert day.pay_cap_avg_eur == pytest.approx(30.0 * 0.5 * 25.0)

    def test_case_3a_nothing_provided(self):
        incap = IncapabilityRecord(area_id="a1",
                                   direction=Direction.LOWERING,
                                   window_index=0,
                                   q_vh_provided_mvar=0.0)
        res = _engine().settle([_obs(q_meas=70.0)],
                               [_grant(30.0, TSO_DELIVERS)], [incap])
        (day,) = res.day_rows
        assert day.pay_cap_avg_eur == 0.0

    def test_case_4_exceedance_without_grant(self):
        # Measured beyond the band, no grant, no TSO call → the DSO
        # exceeded on its own; the upstream delivered — Grenzpreise.
        res = _engine().settle([_obs(q_meas=70.0)])
        r = _row(res, CASE_81_4)
        assert r.world == TSO_DELIVERS
        assert r.e_grenz_mvarh == pytest.approx(20.0 * H)
        assert r.pay_energy_grenz_eur == pytest.approx(20.0 * H * 10.0)
        (day,) = res.day_rows
        assert day.pay_cap_avg_eur == 0.0
        assert day.exceed_mvar == pytest.approx(20.0)
        assert day.pay_cap_grenz_eur == pytest.approx(20.0 * 50.0)

    def test_case_4a_call_beyond_grant(self):
        res = _engine().settle([_obs(q_meas=100.0)],
                               [_grant(30.0, TSO_DELIVERS)])
        r = _row(res, CASE_81_4A)
        assert r.e_avg_mvarh == pytest.approx(30.0 * H)   # 50 → 80
        assert r.e_grenz_mvarh == pytest.approx(20.0 * H)  # 80 → 100
        (day,) = res.day_rows
        assert day.pay_cap_avg_eur == pytest.approx(30.0 * 25.0)
        assert day.pay_cap_grenz_eur == pytest.approx(20.0 * 50.0)

    def test_incapability_needs_matching_grant(self):
        incap = IncapabilityRecord(area_id="a1",
                                   direction=Direction.LOWERING,
                                   window_index=0,
                                   q_vh_provided_mvar=0.0)
        with pytest.raises(SBXError):
            _engine().settle([_obs(q_meas=70.0)], [], [incap])


# ======================================================================
#  Ad-hoc downstream delivery (documented interpretation)
# ======================================================================

class TestAdhoc:
    def test_tso_call_beyond_band_without_grant(self):
        res = _engine().settle([_obs(q_meas=70.0, q_set=75.0)])
        r = _row(res, CASE_82_ADHOC)
        assert r.world == DSO_DELIVERS
        # Own-edge reference (no Vorhalteleistung exists).
        assert r.e_grenz_mvarh == pytest.approx(20.0 * H)
        assert r.cap_exceed_mvar == pytest.approx(25.0)   # highest call
        (day,) = res.day_rows
        assert day.pay_cap_grenz_eur == pytest.approx(25.0 * 50.0)

    def test_in_band_call_stays_free(self):
        res = _engine().settle([_obs(q_meas=20.0, q_set=25.0)])
        _row(res, CASE_FREE)


# ======================================================================
#  Daily accrual details
# ======================================================================

class TestDailyAccrual:
    def test_worst_window_governs_the_day(self):
        # Four granted windows, one under-delivered → the day's
        # Leistungspreis follows the worst window [LF §7.4, §7.5.4].
        obs = [
            _obs(q_meas=100.0, q_set=100.0, window=0),
            _obs(q_meas=100.0, q_set=140.0, window=1),   # under (0.75)
            _obs(q_meas=100.0, q_set=100.0, window=2),
            _obs(q_meas=100.0, q_set=100.0, window=3),
        ]
        res = _engine().settle(obs, [_grant(100.0, DSO_DELIVERS, end=4)])
        (day,) = res.day_rows
        assert day.day_frac == pytest.approx(0.75)

    def test_reissued_grants_accrue_one_day(self):
        # Two sequential grants on the same day → ONE day of Vorhaltung
        # (plan §6 re-issue pattern; [LF §7.4]).
        obs = [_obs(q_meas=100.0, q_set=100.0, window=w)
               for w in range(2)]
        res = _engine().settle(obs, [
            _grant(100.0, DSO_DELIVERS, first=0, end=1),
            _grant(100.0, DSO_DELIVERS, first=1, end=2),
        ])
        (day,) = res.day_rows
        assert day.pay_cap_avg_eur == pytest.approx(200.0 * 25.0)

    def test_overlapping_grants_rejected(self):
        obs = [_obs(q_meas=100.0, q_set=100.0, window=w)
               for w in range(2)]
        with pytest.raises(SBXError):
            _engine().settle(obs, [
                _grant(100.0, DSO_DELIVERS, first=0, end=2),
                _grant(50.0, DSO_DELIVERS, first=1, end=2),
            ])

    def test_granted_window_without_observation_fails_fast(self):
        with pytest.raises(SBXError):
            _engine().settle([_obs(q_meas=100.0, q_set=100.0, window=0)],
                             [_grant(100.0, DSO_DELIVERS, end=2)])


# ======================================================================
#  Totals and CSV output
# ======================================================================

class TestOutputs:
    def test_totals(self):
        res = _engine().settle([_obs(q_meas=170.0, q_set=170.0)],
                               [_grant(100.0, DSO_DELIVERS)])
        (tot,) = res.totals
        assert tot.world == DSO_DELIVERS
        assert tot.pay_total_eur == pytest.approx(
            (150.0 + 50.0) * H * 5.0    # energy avg
            + 20.0 * H * 10.0           # energy grenz
            + 200.0 * 25.0              # capacity avg
            + 20.0 * 50.0)              # capacity grenz

    def test_csv_schema(self, tmp_path):
        res = _engine().settle([_obs(q_meas=170.0, q_set=170.0)],
                               [_grant(100.0, DSO_DELIVERS)])
        prefix = str(tmp_path / "settle")
        write_settlement_csv(res, prefix)
        with open(prefix + "_windows.csv", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0][:6] == ["area_id", "window_index", "t_start_s",
                               "direction", "world", "case"]
        assert len(rows) == 1 + len(res.window_rows)
        with open(prefix + "_days.csv", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1 + len(res.day_rows)
        with open(prefix + "_totals.csv", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[1][3] == "tso"   # payer: requester pays deliverer
        assert float(rows[1][-1]) == pytest.approx(
            res.totals[0].pay_total_eur)
