"""
SBX Phase 6 — settlement acceptance tests (plan v2 §2.5 / §4 Phase 6).

Engine-level tests drive deterministic synthetic ``CycleObservation``s
on the REAL IEEE 39 corridors/contracts (tier 1 free band + netting,
tier 2 paid-vs-unpaid billing with the importing payer, tier 3
attribution to side A / side B, ΔP neutrality, UNATTRIBUTED residual
flag, payment conservation, the ``n_settle_cycles`` window, output
files).  Scheduler-level tests reuse the ``test_scheduler`` harness to
verify the elapsed-cycle wiring: the unilateral (paid) accumulator is
billed, and the paid-first unwind is visible in the settlement series.
"""
from __future__ import annotations

import numpy as np
import pandapower as pp
import pytest

from network.ieee39.build import build_ieee39_net
from network.zone_partition import fixed_zone_partition_ieee39
from sbx_h.config import SBXConfig
from sbx_h.contract import build_default_contract, q_std_mvar
from sbx_h.corridor import build_corridor_registry, corridor_sensitivities
from sbx_h.fail import SBXError
from sbx_h.settlement import (
    ATTRIB_DP_NEUTRAL, ATTRIB_NONE, ATTRIB_SIDE_A, ATTRIB_SIDE_B,
    ATTRIB_UNATTRIBUTED, CycleObservation, SettlementEngine,
    write_settlement_outputs,
)


@pytest.fixture(scope="module")
def world():
    net, _ = build_ieee39_net(scenario="wind_replace")
    zone_map, _ = fixed_zone_partition_ieee39(net)
    registry = build_corridor_registry(net, zone_map)
    pp.runpp(net)
    # Reference timing pinned EXPLICITLY (plan-v2 §5 values): the tests
    # assert absolute Mvar·h / EUR numbers, so they must not float with
    # the SBXConfig defaults (which Manuel tunes experimentally).
    cfg = SBXConfig(k_sched=5, dq_quant_rate_mvar_per_15min=10.0,
                    n_need=5)
    contracts = {
        key: build_default_contract(corr, net, cfg)
        for key, corr in registry.items()
    }
    base_p = {}
    for key, corr in registry.items():
        p = []
        for ln in corr.lines:
            side = ("from" if int(net.line.at[ln.line_idx, "from_bus"])
                    == ln.bus_a else "to")
            p.append(float(net.res_line.at[ln.line_idx, f"p_{side}_mw"]))
        base_p[key] = tuple(p)
    return registry, contracts, cfg, base_p


def make_engine(world, key=(1, 3), cfg=None):
    registry, contracts, default_cfg, base_p = world
    cfg = cfg or default_cfg
    return (SettlementEngine(registry[key], contracts[key], cfg),
            registry[key], contracts[key], base_p[key], cfg)


def make_obs(corr, contract, p_sched, cfg, *, cycle=1, surplus=0.0,
             paid=0.0, dq_meas=0.0, dv_a=None, dv_b=None, dp=None,
             q_meas=None):
    """Observation at the schedule point plus controlled deviations."""
    q_std = q_std_mvar(contract, corr, p_sched,
                       delta_max_rad=cfg.delta_max_rad)
    acting = "a" if surplus > 0 else ("b" if surplus < 0 else None)
    v_sched_a = tuple(contract.v_std_a_pu)
    v_sched_b = tuple(contract.v_std_b_pu)
    n = corr.n_lines
    v_meas_a = tuple(v + (dv_a[k] if dv_a else 0.0)
                     for k, v in enumerate(v_sched_a))
    v_meas_b = tuple(v + (dv_b[k] if dv_b else 0.0)
                     for k, v in enumerate(v_sched_b))
    p_meas = tuple(p + (dp[k] if dp else 0.0)
                   for k, p in enumerate(p_sched))
    return CycleObservation(
        cycle=cycle,
        q_meas_mvar=(q_std + surplus + dq_meas if q_meas is None
                     else q_meas),
        q_std_mvar=q_std,
        surplus_mvar=surplus,
        surplus_paid_mvar=paid,
        surplus_unpaid_mvar=surplus - paid,
        acting_end=acting,
        v_meas_a_pu=v_meas_a,
        v_meas_b_pu=v_meas_b,
        v_sched_a_pu=v_sched_a,
        v_sched_b_pu=v_sched_b,
        p_meas_mw=p_meas,
        p_sched_mw=tuple(p_sched),
        q_band_mvar=contract.q_band_mvar,
    ), q_std


T_H = 0.25  # 15-min cycle


def test_tier1_in_band_free_and_netted(world):
    eng, corr, contract, p, cfg = make_engine(world)
    s = eng.observe(make_obs(corr, contract, p, cfg, dq_meas=3.0)[0])
    assert s.attribution == ATTRIB_NONE
    assert s.tier2_eur == 0.0 and s.tier3_eur == 0.0
    assert s.band_dev_mvar == pytest.approx(3.0)
    assert s.netting_mvarh == pytest.approx(3.0 * T_H)
    assert all(x == 0.0 for x in s.payments_eur.values())
    s2 = eng.observe(make_obs(corr, contract, p, cfg, cycle=2,
                              dq_meas=-4.0)[0])
    assert eng.ledger.netting_mvarh == pytest.approx((3.0 - 4.0) * T_H)
    assert s2.tier3_eur == 0.0


def test_tier2_paid_surplus_billed_importer_pays(world):
    eng, corr, contract, p, cfg = make_engine(world)
    # surplus < 0: B exports the surplus (acting), A imports and pays.
    s = eng.observe(make_obs(corr, contract, p, cfg,
                             surplus=-20.0, paid=-20.0)[0])
    expected = cfg.p_surplus_eur_per_mvarh * 20.0 * T_H
    assert s.tier2_eur == pytest.approx(expected)
    assert s.tier2_payer == 1
    assert s.payments_eur[1] == pytest.approx(-expected)
    assert s.payments_eur[3] == pytest.approx(+expected)
    # surplus > 0: A exports (acting), B imports and pays.
    s = eng.observe(make_obs(corr, contract, p, cfg, cycle=2,
                             surplus=+20.0, paid=+20.0)[0])
    assert s.tier2_payer == 3
    assert s.payments_eur[3] == pytest.approx(-expected)


def test_tier2_mutual_unpaid_not_billed(world):
    eng, corr, contract, p, cfg = make_engine(world)
    s = eng.observe(make_obs(corr, contract, p, cfg,
                             surplus=-20.0, paid=0.0)[0])
    assert s.tier2_eur == 0.0 and s.tier2_payer is None
    assert all(x == 0.0 for x in s.payments_eur.values())


def test_tier2_paid_unpaid_split_partial(world):
    eng, corr, contract, p, cfg = make_engine(world)
    s = eng.observe(make_obs(corr, contract, p, cfg,
                             surplus=-30.0, paid=-10.0)[0])
    assert s.paid_mvarh == pytest.approx(10.0 * T_H)
    assert s.tier2_eur == pytest.approx(
        cfg.p_surplus_eur_per_mvarh * 10.0 * T_H)


@pytest.mark.parametrize("side", ["a", "b"])
def test_tier3_attribution_and_charge(world, side):
    eng, corr, contract, p, cfg = make_engine(world)
    per_line, _, _ = corridor_sensitivities(
        corr, list(contract.v_std_a_pu), list(contract.v_std_b_pu),
        list(p), delta_max_rad=cfg.delta_max_rad)
    s_v = per_line[0][0] if side == "a" else per_line[0][1]
    dv = np.sign(s_v) * 2.0e-3          # ≈ |s_v|·2e-3 Mvar beyond band
    dev = s_v * dv
    assert abs(dev) > contract.q_band_mvar + 1.0
    kwargs = {"dv_a": [dv]} if side == "a" else {"dv_b": [dv]}
    s = eng.observe(make_obs(corr, contract, p, cfg,
                             dq_meas=float(dev), **kwargs)[0])
    assert s.attribution == (ATTRIB_SIDE_A if side == "a"
                             else ATTRIB_SIDE_B)
    excess = abs(dev) - contract.q_band_mvar
    expected = (cfg.kappa_penalty * cfg.p_surplus_eur_per_mvarh
                * excess * T_H)
    assert s.tier3_eur == pytest.approx(expected, rel=0.05)
    payer = 1 if side == "a" else 3
    assert s.tier3_payer == payer
    assert s.payments_eur[payer] == pytest.approx(-s.tier3_eur)
    assert sum(s.payments_eur.values()) == pytest.approx(0.0, abs=1e-9)


def test_tier3_dp_neutral(world):
    eng, corr, contract, p, cfg = make_engine(world)
    per_line, _, _ = corridor_sensitivities(
        corr, list(contract.v_std_a_pu), list(contract.v_std_b_pu),
        list(p), delta_max_rad=cfg.delta_max_rad)
    s_p = per_line[0][2]
    dp = 200.0 * np.sign(s_p) if s_p != 0.0 else 200.0
    dev = s_p * dp
    if abs(dev) <= contract.q_band_mvar + 1.0:
        dp *= (contract.q_band_mvar + 2.0) / max(abs(dev), 1e-9)
        dev = s_p * dp
    s = eng.observe(make_obs(corr, contract, p, cfg,
                             dq_meas=float(dev), dp=[float(dp)])[0])
    assert s.attribution == ATTRIB_DP_NEUTRAL
    assert s.tier3_eur == 0.0 and s.tier3_payer is None
    assert all(x == 0.0 for x in s.payments_eur.values())


def test_tier3_unattributed_no_charge(world):
    eng, corr, contract, p, cfg = make_engine(world)
    # 20 Mvar deviation with NO measured cause: residual > tolerance.
    s = eng.observe(make_obs(corr, contract, p, cfg, dq_meas=20.0)[0])
    assert s.attribution == ATTRIB_UNATTRIBUTED
    assert s.tier3_eur == 0.0
    assert all(x == 0.0 for x in s.payments_eur.values())
    assert eng.ledger.n_unattributed == 1


def test_settle_window_rolling_mean(world):
    cfg2 = SBXConfig(n_settle_cycles=2)
    eng, corr, contract, p, _ = make_engine(world, cfg=cfg2)
    eng.observe(make_obs(corr, contract, p, cfg2, cycle=1,
                         dq_meas=+4.0)[0])
    s = eng.observe(make_obs(corr, contract, p, cfg2, cycle=2,
                             dq_meas=-4.0)[0])
    # Window mean of ±4 → zero deviation this settlement.
    assert s.band_dev_mvar == pytest.approx(0.0, abs=1e-9)


def test_rolling_window_bills_after_unwind(world):
    """Regression (016 short-cycle ablation, 2026-07-08): with
    n_settle_cycles > 1 the LATEST cycle can be fully unwound
    (acting_end None) while the window still carries paid surplus.
    The payer direction must come from the WINDOWED surplus — the same
    window the billed quantity is averaged over — not from the
    instantaneous acting side (the old code raised here)."""
    cfg2 = SBXConfig(k_sched=5, dq_quant_rate_mvar_per_15min=10.0,
                     n_settle_cycles=3)
    eng, corr, contract, p, _ = make_engine(world, cfg=cfg2)
    eng.observe(make_obs(corr, contract, p, cfg2, cycle=1,
                         surplus=+10.0, paid=+10.0)[0])
    eng.observe(make_obs(corr, contract, p, cfg2, cycle=2,
                         surplus=+10.0, paid=+10.0)[0])
    # Cycle 3: surplus fully unwound; window mean paid = 20/3 > 0.
    s = eng.observe(make_obs(corr, contract, p, cfg2, cycle=3,
                             surplus=0.0, paid=0.0)[0])
    expected = cfg2.p_surplus_eur_per_mvarh * (20.0 / 3.0) * T_H
    assert s.tier2_eur == pytest.approx(expected)
    # Windowed surplus > 0 → end A (zone 1) acts/exports, zone 3 pays.
    assert s.tier2_payer == 3
    assert s.payments_eur[3] == pytest.approx(-expected)
    assert s.payments_eur[1] == pytest.approx(+expected)


def test_outputs_written(world, tmp_path):
    eng, corr, contract, p, cfg = make_engine(world)
    eng.observe(make_obs(corr, contract, p, cfg,
                         surplus=-20.0, paid=-20.0)[0])
    csv_path, md_path = write_settlement_outputs(
        {eng.key: eng}, tmp_path, "unit")
    csv_text = open(csv_path, encoding="utf-8").read()
    assert "tier2_eur" in csv_text and "1-3" in csv_text
    md_text = open(md_path, encoding="utf-8").read()
    assert "| (1,3) |" in md_text


# ----------------------------------------------------------------------
#  Scheduler wiring (elapsed-cycle semantics, paid-first unwind visible)
# ----------------------------------------------------------------------

def test_scheduler_settles_deal_and_unwind(plant):
    from tests.sbx_h.test_scheduler import BOUND_OK, BOUND_UNDER, Harness
    # delivery_gate off: the synthetic harness feeds tie_q = 0, which
    # the v4 gate correctly reads as non-delivery; this test exercises
    # the v2/v3 deal/unwind/settlement mechanics.
    h = Harness(plant, __import__('tests.sbx_h.test_scheduler', fromlist=['ref_config']).ref_config(delivery_gate=False))
    h.bounds[1] = BOUND_UNDER
    h.run_cycles(3)                     # builds paid surplus on (1,2)
    key = (1, 2)
    setts = h.sched.settlements[key]
    assert setts, "no settlements emitted"
    billed = [s for s in setts if s.tier2_eur > 0.0]
    assert billed, "paid surplus never billed"
    # Import need of area 1 → surplus < 0 → acting side B → area 1
    # (importing) pays; conservation holds.
    for s in billed:
        assert s.tier2_payer == 1
        assert sum(s.payments_eur.values()) == pytest.approx(0.0,
                                                             abs=1e-9)
    # Remove the stress; after dwell the unwind reduces the PAID
    # accumulator first — the billed Mvar·h must fall back to zero.
    h.bounds[1] = BOUND_OK
    h.run_cycles(10)
    assert h.sched.settlements[key][-1].paid_mvarh == pytest.approx(0.0)
    paid_series = [s.paid_mvarh for s in h.sched.settlements[key]]
    peak = max(paid_series)
    assert peak > 0.0
    assert paid_series[-1] == 0.0
    # Monotone non-increase after the peak (paid-first unwind, no
    # re-request in this trajectory).
    tail = paid_series[paid_series.index(peak):]
    assert all(x1 >= x2 - 1e-9 for x1, x2 in zip(tail, tail[1:]))
