"""
Tests for the minimal SBX-H v6 hold/sag support-energy settlement.

The real IEEE 39 corridor (1, 3) supplies the pi-line parameters.
Synthetic measurements isolate role classification, Q direction,
measured-P baseline correction, payment conservation, capping, rolling
windows, and the optional strength diagnostic.
"""
from __future__ import annotations

import dataclasses

import pandapower as pp
import pytest

from network.ieee39.build import build_ieee39_net
from network.zone_partition import fixed_zone_partition_ieee39
from sbx_h.config import SBXConfig
from sbx_h.contract import build_default_contract, q_std_mvar
from sbx_h.corridor import build_corridor_registry
from sbx_h.fail import SBXError
from sbx_h.settlement import (
    DIRECTION_A_TO_B,
    DIRECTION_B_TO_A,
    SUPPORT_A_SAGS_B_HOLDS,
    SUPPORT_A_SAGS_B_NOT_HOLDING,
    SUPPORT_BOTH_SAG,
    SUPPORT_B_SAGS_A_HOLDS,
    SUPPORT_NONE,
    CycleObservation,
    SettlementEngine,
    write_settlement_outputs,
)

T_H = 6.0 / 60.0


@pytest.fixture(scope="module")
def world():
    net, _ = build_ieee39_net(scenario="wind_replace")
    zone_map, _ = fixed_zone_partition_ieee39(net)
    registry = build_corridor_registry(net, zone_map)
    pp.runpp(net)
    cfg = SBXConfig(k_sched=2, q_band_mvar=5.0)
    corr = registry[(1, 3)]
    contract = build_default_contract(corr, net, cfg)
    p_meas = []
    for line in corr.lines:
        side = (
            "from"
            if int(net.line.at[line.line_idx, "from_bus"]) == line.bus_a
            else "to"
        )
        p_meas.append(
            float(net.res_line.at[line.line_idx, f"p_{side}_mw"])
        )
    return cfg, corr, contract, tuple(p_meas)


def make_obs(
    corr,
    contract,
    p_meas,
    cfg,
    *,
    cycle=1,
    dq_meas=0.0,
    dv_a=0.0,
    dv_b=0.0,
    dp=0.0,
):
    """Create an observation relative to the measured-P baseline."""
    p_actual = tuple(p + dp for p in p_meas)
    q_baseline = q_std_mvar(contract, corr, p_actual)
    return CycleObservation(
        cycle=cycle,
        q_meas_mvar=q_baseline + dq_meas,
        v_meas_a_pu=tuple(v + dv_a for v in contract.v_std_a_pu),
        v_meas_b_pu=tuple(v + dv_b for v in contract.v_std_b_pu),
        v_sched_a_pu=contract.v_std_a_pu,
        v_sched_b_pu=contract.v_std_b_pu,
        p_meas_mw=p_actual,
        q_band_mvar=cfg.q_band_mvar,
    )


def engine(world):
    cfg, corr, contract, p_meas = world
    return (
        SettlementEngine(corr, contract, cfg),
        cfg,
        corr,
        contract,
        p_meas,
    )


def test_flow_deviation_without_sag_is_not_paid(world):
    eng, cfg, corr, contract, p_meas = engine(world)
    result = eng.observe(
        make_obs(corr, contract, p_meas, cfg, dq_meas=20.0)
    )
    assert result.support_state == SUPPORT_NONE
    assert result.support_mvar == 0.0
    assert result.support_eur == 0.0
    assert result.support_payer is None
    assert abs(sum(result.payments_eur.values())) < 1e-12


def test_a_sags_b_holds_and_b_to_a_flow_pays_b(world):
    eng, cfg, corr, contract, p_meas = engine(world)
    result = eng.observe(
        make_obs(
            corr,
            contract,
            p_meas,
            cfg,
            dq_meas=-12.0,
            dv_a=-0.006,
        )
    )
    assert result.support_state == SUPPORT_A_SAGS_B_HOLDS
    assert result.support_direction == DIRECTION_B_TO_A
    assert result.support_mvar == pytest.approx(7.0)
    assert result.support_payer == corr.area_a
    assert result.support_payee == corr.area_b
    expected = 7.0 * T_H * cfg.p_support_eur_per_mvarh
    assert result.support_eur == pytest.approx(expected)
    assert result.payments_eur[corr.area_a] == pytest.approx(-expected)
    assert result.payments_eur[corr.area_b] == pytest.approx(expected)
    assert abs(sum(result.payments_eur.values())) < 1e-12


def test_b_sags_a_holds_and_a_to_b_flow_pays_a(world):
    eng, cfg, corr, contract, p_meas = engine(world)
    result = eng.observe(
        make_obs(
            corr,
            contract,
            p_meas,
            cfg,
            dq_meas=10.0,
            dv_b=-0.006,
        )
    )
    assert result.support_state == SUPPORT_B_SAGS_A_HOLDS
    assert result.support_direction == DIRECTION_A_TO_B
    assert result.support_mvar == pytest.approx(5.0)
    assert result.support_payer == corr.area_b
    assert result.support_payee == corr.area_a


def test_better_side_must_still_hold_absolute_schedule(world):
    eng, cfg, corr, contract, p_meas = engine(world)
    result = eng.observe(
        make_obs(
            corr,
            contract,
            p_meas,
            cfg,
            dq_meas=-12.0,
            dv_a=-0.008,
            dv_b=-0.003,
        )
    )
    assert result.support_state == SUPPORT_A_SAGS_B_NOT_HOLDING
    assert not result.b_holds
    assert result.support_eur == 0.0


def test_both_sag_is_not_paid(world):
    eng, cfg, corr, contract, p_meas = engine(world)
    result = eng.observe(
        make_obs(
            corr,
            contract,
            p_meas,
            cfg,
            dq_meas=-12.0,
            dv_a=-0.007,
            dv_b=-0.007,
        )
    )
    assert result.support_state == SUPPORT_BOTH_SAG
    assert result.support_mvar == 0.0
    assert result.support_eur == 0.0


def test_wrong_flow_direction_is_not_paid(world):
    eng, cfg, corr, contract, p_meas = engine(world)
    result = eng.observe(
        make_obs(
            corr,
            contract,
            p_meas,
            cfg,
            dq_meas=12.0,
            dv_a=-0.006,
        )
    )
    assert result.support_state == SUPPORT_A_SAGS_B_HOLDS
    assert result.support_direction == DIRECTION_B_TO_A
    assert result.support_mvar == 0.0
    assert result.support_payer is None


def test_measured_p_recomputes_baseline_without_false_support(world):
    eng, cfg, corr, contract, p_meas = engine(world)
    result = eng.observe(
        make_obs(
            corr,
            contract,
            p_meas,
            cfg,
            dp=80.0,
            dv_a=-0.006,
        )
    )
    assert result.deviation_mvar == pytest.approx(0.0, abs=1e-9)
    assert result.support_mvar == 0.0
    assert result.support_eur == 0.0


def test_optional_support_cap_limits_exposure(world):
    cfg0, corr, contract0, p_meas = world
    cfg = dataclasses.replace(cfg0, q_support_cap_mvar=2.0)
    contract = dataclasses.replace(contract0, q_support_cap_mvar=2.0)
    eng = SettlementEngine(corr, contract, cfg)
    result = eng.observe(
        make_obs(
            corr,
            contract,
            p_meas,
            cfg,
            dq_meas=-20.0,
            dv_a=-0.006,
        )
    )
    assert result.uncapped_support_mvar == pytest.approx(15.0)
    assert result.support_mvar == pytest.approx(2.0)


def test_strength_is_diagnostic_only(world):
    eng, cfg, corr, contract, p_meas = engine(world)
    result = eng.observe(
        make_obs(
            corr,
            contract,
            p_meas,
            cfg,
            dq_meas=-12.0,
            dv_a=-0.006,
            dv_b=-0.001,
        )
    )
    assert result.support_mvar == pytest.approx(7.0)
    assert result.observed_strength_mvar_per_mpu == pytest.approx(7.0)
    expected = 7.0 * T_H * cfg.p_support_eur_per_mvarh
    assert result.support_eur == pytest.approx(expected)


def test_rolling_window_uses_window_means(world):
    cfg0, corr, contract0, p_meas = world
    cfg = dataclasses.replace(cfg0, n_settle_cycles=2)
    contract = dataclasses.replace(
        contract0,
        k_sched=cfg.k_sched,
        t_cycle_min=cfg.t_cycle_min,
    )
    eng = SettlementEngine(corr, contract, cfg)
    first = eng.observe(
        make_obs(
            corr,
            contract,
            p_meas,
            cfg,
            cycle=1,
            dq_meas=-9.0,
            dv_a=-0.006,
        )
    )
    second = eng.observe(
        make_obs(
            corr,
            contract,
            p_meas,
            cfg,
            cycle=2,
            dq_meas=-1.0,
            dv_a=-0.006,
        )
    )
    assert first.support_mvar == pytest.approx(4.0)
    assert second.deviation_mvar == pytest.approx(-5.0)
    assert second.support_mvar == pytest.approx(0.0)


def test_observation_validation(world):
    eng, cfg, corr, contract, p_meas = engine(world)
    good = make_obs(corr, contract, p_meas, cfg)
    with pytest.raises(SBXError, match="arity"):
        eng.observe(
            dataclasses.replace(good, v_meas_a_pu=(1.0,) * 99)
        )
    with pytest.raises(SBXError, match="q_band"):
        eng.observe(dataclasses.replace(good, q_band_mvar=0.0))
    with pytest.raises(SBXError, match="q_meas"):
        eng.observe(dataclasses.replace(good, q_meas_mvar=float("nan")))


def test_output_files(world, tmp_path):
    eng, cfg, corr, contract, p_meas = engine(world)
    eng.observe(
        make_obs(
            corr,
            contract,
            p_meas,
            cfg,
            dq_meas=-12.0,
            dv_a=-0.006,
        )
    )
    csv_path, md_path = write_settlement_outputs(
        {eng.key: eng},
        tmp_path,
        "v6_test",
    )
    csv_text = open(csv_path, encoding="utf-8").read()
    md_text = open(md_path, encoding="utf-8").read()
    assert "support_eur" in csv_text
    assert "q_baseline_mvar" in csv_text
    assert "Paid windows" in md_text
