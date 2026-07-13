"""
SBX Phase 2 — tests for ``sbx_h.contract`` and the approved additive
``Measurement.tie_line_p_mw`` extension (STATUS_SBX.md gate G2).

Acceptance (plan v2 §4 Phase 2):
* golden test 4 through the contract path: the default contract's q_std
  reproduces the base-case corridor flow within max(0.5 Mvar, 1 %);
* contract immutability enforced (mutation attempt raises);
* contract/corridor misalignment fails fast;
* ``measure_zone_tso`` populates tie-line P at the in-zone endpoint in
  load convention, aligned with the existing tie-line Q field.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pandapower as pp
import pytest

from core.measurement import measure_zone_tso
from network.ieee39.build import build_ieee39_net
from network.zone_partition import fixed_zone_partition_ieee39
from sbx_h.config import SBXConfig
from sbx_h.contract import build_default_contract, q_std_mvar
from sbx_h.corridor import build_corridor_registry
from sbx_h.fail import SBXError


@pytest.fixture(scope="module")
def case():
    net, _meta = build_ieee39_net(scenario="wind_replace")
    zone_map, _ = fixed_zone_partition_ieee39(net)
    registry = build_corridor_registry(net, zone_map)
    pp.runpp(net)
    return net, zone_map, registry


def _base_p_q(net, corridor):
    p_a, q_a = [], []
    for ln in corridor.lines:
        side = ("from" if int(net.line.at[ln.line_idx, "from_bus"]) == ln.bus_a
                else "to")
        p_a.append(float(net.res_line.at[ln.line_idx, f"p_{side}_mw"]))
        q_a.append(float(net.res_line.at[ln.line_idx, f"q_{side}_mvar"]))
    return p_a, q_a


def test_default_contract_reproduces_base_flow(case):
    net, _zone_map, registry = case
    cfg = SBXConfig()
    for key, corr in registry.items():
        contract = build_default_contract(corr, net, cfg)
        p_a, q_a = _base_p_q(net, corr)
        q_std = q_std_mvar(contract, corr, p_a)
        q_meas = sum(q_a)
        tol = max(0.5, 0.01 * abs(q_meas))
        assert abs(q_std - q_meas) <= tol, (
            f"corridor {key}: q_std={q_std:.3f} vs q_meas={q_meas:.3f} Mvar "
            f"(tol {tol:.3f})"
        )


def test_contract_immutability(case):
    net, _zone_map, registry = case
    corr = next(iter(registry.values()))
    contract = build_default_contract(corr, net, SBXConfig())
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.q_band_mvar = 99.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.v_std_a_pu = (1.0,) * contract.n_lines


def test_contract_quantum_is_rate_scaled(case):
    net, _zone_map, registry = case
    corr = next(iter(registry.values()))
    # Plan-v2 §5 reference timing pinned explicitly (the SBXConfig
    # DEFAULTS are Manuel's experimental knobs and may differ).
    contract = build_default_contract(
        corr, net, SBXConfig(k_sched=5,
                             dq_quant_rate_mvar_per_15min=10.0))
    # Phase 0 semantics: 5 iterations x 180 s = 15 min -> quantum = rate.
    assert contract.t_cycle_min == pytest.approx(15.0)
    assert contract.dq_quant_mvar == pytest.approx(
        contract.dq_quant_rate_mvar_per_15min
    )
    # Short-cycle ablation scaling: 3-min cycle -> quantum = rate / 5.
    short = dataclasses.replace(contract, k_sched=1, t_cycle_min=3.0)
    assert short.dq_quant_mvar == pytest.approx(
        contract.dq_quant_rate_mvar_per_15min / 5.0
    )


def test_contract_corridor_mismatch_raises(case):
    net, _zone_map, registry = case
    corridors = list(registry.values())
    contract = build_default_contract(corridors[0], net, SBXConfig())
    with pytest.raises(SBXError, match="different area pairs|line lists"):
        q_std_mvar(contract, corridors[1], [0.0] * corridors[1].n_lines)


def test_contract_requires_converged_base_case(case):
    # build_ieee39_net converges a power flow internally, so an
    # incomplete result table has to be forced to exercise the guard.
    import copy

    net, _zone_map, registry = case
    stale_net = copy.deepcopy(net)
    stale_net.res_bus = stale_net.res_bus.iloc[0:0]
    corr = next(iter(registry.values()))
    with pytest.raises(SBXError, match="converged power flow"):
        build_default_contract(corr, stale_net, SBXConfig())


def test_measure_zone_tso_populates_tie_line_p(case):
    net, _zone_map, registry = case
    corr = registry[(1, 2)]
    tie_lines = [ln.line_idx for ln in corr.lines]
    endpoints = [ln.bus_a for ln in corr.lines]  # zone-1 side
    zone_def = SimpleNamespace(
        zone_id=1,
        line_indices=[],
        pcc_trafo_indices=[],
        tso_der_indices=[],
        shunt_bus_indices=[],
        gen_indices=[],
        oltc_trafo_indices=[],
        tie_line_indices=tie_lines,
        tie_line_endpoint_buses=endpoints,
    )
    meas = measure_zone_tso(net, zone_def, it=0)
    assert meas.tie_line_p_mw.shape == meas.tie_line_q_mvar.shape
    for k, ln in enumerate(corr.lines):
        side = ("from" if int(net.line.at[ln.line_idx, "from_bus"]) == ln.bus_a
                else "to")
        p_expect = float(net.res_line.at[ln.line_idx, f"p_{side}_mw"])
        q_expect = float(net.res_line.at[ln.line_idx, f"q_{side}_mvar"])
        assert meas.tie_line_p_mw[k] == pytest.approx(p_expect)
        assert meas.tie_line_q_mvar[k] == pytest.approx(q_expect)


def test_v3_schedule_lookup_and_time_requirement(case):
    """SBX v3: planning-anchored v_std schedules — lookup, q_std jump at
    an interval boundary, and the no-silent-fallback time requirement."""
    import dataclasses

    from sbx_h.contract import q_std_mvar as _q_std

    net, _zone_map, registry = case
    cfg = SBXConfig()
    key = (1, 3)
    corr = registry[key]
    base = build_default_contract(corr, net, cfg)
    p_a, _q_a = _base_p_q(net, corr)

    va0, vb0 = base.v_std_a_pu, base.v_std_b_pu
    va1 = tuple(v + 0.002 for v in va0)   # hour 2: raised A-side plan
    sched = ((0.0, va0, vb0), (7200.0, va1, vb0))
    c = build_default_contract(corr, net, cfg, v_std_schedule=sched)

    # Piecewise-constant lookup.
    assert c.v_std_at(0.0) == (va0, vb0)
    assert c.v_std_at(7199.9) == (va0, vb0)
    assert c.v_std_at(7200.0) == (va1, vb0)
    assert c.v_std_at(1e9) == (va1, vb0)

    # q_std jumps at the interval boundary (stiff ties: 2 mpu on the
    # A side moves the standard by several Mvar).
    q0 = _q_std(c, corr, p_a, time_s=0.0)
    q1 = _q_std(c, corr, p_a, time_s=7200.0)
    assert abs(q1 - q0) > 1.0
    # ... and matches the constant contract at t = 0.
    assert q0 == pytest.approx(_q_std(base, corr, p_a), abs=1e-9)

    # Schedule-bearing contracts refuse a time-less q_std evaluation.
    with pytest.raises(SBXError, match="scenario time"):
        _q_std(c, corr, p_a)

    # Planning-derived hourly band (v3 closing piece): lookup + the
    # constant-field consistency rule.
    band_sched = ((0.0, 26.0), (7200.0, 41.0))
    cb = build_default_contract(corr, net, cfg,
                                v_std_schedule=sched,
                                q_band_schedule=band_sched)
    assert cb.q_band_mvar == 26.0          # t = 0 view
    assert cb.q_band_at(0.0) == 26.0
    assert cb.q_band_at(7199.9) == 26.0
    assert cb.q_band_at(7200.0) == 41.0
    assert base.q_band_at(1e9) == base.q_band_mvar   # constant contract
    with pytest.raises(SBXError, match="strictly ascending"):
        build_default_contract(corr, net, cfg, v_std_schedule=sched,
                               q_band_schedule=((0.0, 26.0),
                                                (0.0, 41.0)))
    with pytest.raises(SBXError, match="start at t = 0"):
        build_default_contract(corr, net, cfg, v_std_schedule=sched,
                               q_band_schedule=((60.0, 26.0),))

    # Validation: first interval must start at 0; constant fields must
    # equal the first interval; intervals strictly ascending.
    with pytest.raises(SBXError, match="start at t = 0"):
        build_default_contract(corr, net, cfg,
                               v_std_schedule=((60.0, va0, vb0),))
    with pytest.raises(SBXError, match="strictly ascending"):
        build_default_contract(
            corr, net, cfg,
            v_std_schedule=((0.0, va0, vb0), (0.0, va1, vb0)))
    with pytest.raises(SBXError, match="first interval"):
        dataclasses.replace(base, v_std_schedule=((0.0, va1, vb0),))


def test_measurement_default_tie_line_p_is_empty():
    from core.measurement import Measurement

    m = Measurement(
        iteration=0,
        bus_indices=np.array([], dtype=np.int64),
        voltage_magnitudes_pu=np.array([], dtype=np.float64),
        branch_indices=np.array([], dtype=np.int64),
        current_magnitudes_ka=np.array([], dtype=np.float64),
        interface_transformer_indices=np.array([], dtype=np.int64),
        interface_q_hv_side_mvar=np.array([], dtype=np.float64),
        der_indices=np.array([], dtype=np.int64),
        der_q_mvar=np.array([], dtype=np.float64),
        der_p_mw=np.array([], dtype=np.float64),
        oltc_indices=np.array([], dtype=np.int64),
        oltc_tap_positions=np.array([], dtype=np.int64),
        shunt_indices=np.array([], dtype=np.int64),
        shunt_states=np.array([], dtype=np.int64),
        gen_indices=np.array([], dtype=np.int64),
        gen_vm_pu=np.array([], dtype=np.float64),
    )
    assert m.tie_line_p_mw.size == 0
