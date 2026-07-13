"""
SBX Phase 1 — golden tests for ``sbx_h.tie_line_model`` and ``sbx_h.corridor``
(plan v2 §3, mandatory gate).

1. Every tie line, converged base case:
   |q_flow(v_a^meas, v_b^meas, p^meas) − q^meas| ≤ max(0.5 Mvar, 1 %) —
   validates extraction, per-unit conversion, end mapping and signs.
2. Round trips: ``v_sched_for_q`` → ``q_flow`` ≤ 1e−6 Mvar; corridor:
   ``corridor_solve_dv`` → ``corridor_q_flow`` likewise.
3. Sensitivities vs a perturbed pandapower re-run: ≤ 5 % relative for
   small perturbations (absolute floor 0.1 Mvar guards near-zero ΔQ).
4. Base-case consistency: with default contract voltages (base case
   rounded to 1e−5 pu), q_std reproduces the base-case corridor flow
   within the golden tolerance.  Rounding history (STATUS_SBX.md §1.3):
   the plan's 1e−3 pu shifted q_std by up to 4 Mvar (|b| ≈ 40–75 pu →
   4–7.5 Mvar per mpu); Manuel's first choice 1e−4 pu failed marginally
   on corridor (1,3) (−0.562 vs 0.500 Mvar tol, worst-case rounding on
   the stiff single-line corridor); 1e−5 pu passes everywhere with
   ≥ 85 % margin (worst deviation 0.067 Mvar).

Base case: ``build_ieee39_net(scenario="wind_replace")`` (the experiment
scenario), bare TN network, fixed 3-area partition — the same partition
BME uses (v2.2 item 6).
"""
from __future__ import annotations

import copy

import numpy as np
import pandapower as pp
import pytest

from network.ieee39.build import build_ieee39_net
from network.zone_partition import fixed_zone_partition_ieee39
from sbx_h.contract import V_STD_DECIMALS
from sbx_h.corridor import build_corridor_registry, corridor_q_flow, \
    corridor_sensitivities
from sbx_h.tie_line_model import q_flow, sensitivities, v_sched_for_q

# Ground truth of the fixed 3-area partition (matches
# tests/test_boundary_topology.py): corridor -> {line_idx: (bus_a, bus_b)}.
EXPECTED_CORRIDORS = {
    (1, 2): {2: (1, 2), 14: (38, 8)},
    (1, 3): {25: (26, 16)},
    (2, 3): {5: (2, 17), 18: (13, 14)},
}


@pytest.fixture(scope="module")
def case():
    net, _meta = build_ieee39_net(scenario="wind_replace")
    zone_map, _ = fixed_zone_partition_ieee39(net)
    registry = build_corridor_registry(net, zone_map)
    pp.runpp(net)
    return net, registry


def _measured(net, line):
    """(v_a, v_b, p_a, q_a) at the reference end A of one corridor line."""
    v_a = float(net.res_bus.at[line.bus_a, "vm_pu"])
    v_b = float(net.res_bus.at[line.bus_b, "vm_pu"])
    if int(net.line.at[line.line_idx, "from_bus"]) == line.bus_a:
        p_a = float(net.res_line.at[line.line_idx, "p_from_mw"])
        q_a = float(net.res_line.at[line.line_idx, "q_from_mvar"])
    else:
        p_a = float(net.res_line.at[line.line_idx, "p_to_mw"])
        q_a = float(net.res_line.at[line.line_idx, "q_to_mvar"])
    return v_a, v_b, p_a, q_a


def _corridor_measurements(net, corridor):
    v_a, v_b, p_a, q_a = [], [], [], []
    for ln in corridor.lines:
        va, vb, pa, qa = _measured(net, ln)
        v_a.append(va)
        v_b.append(vb)
        p_a.append(pa)
        q_a.append(qa)
    return v_a, v_b, p_a, q_a


# ---------------------------------------------------------------------------
#  Registry structure (ground-truth cross-check)
# ---------------------------------------------------------------------------


def test_registry_matches_ground_truth(case):
    _net, registry = case
    assert set(registry.keys()) == set(EXPECTED_CORRIDORS.keys())
    for key, expected_lines in EXPECTED_CORRIDORS.items():
        corr = registry[key]
        got = {ln.line_idx: (ln.bus_a, ln.bus_b) for ln in corr.lines}
        assert got == expected_lines


# ---------------------------------------------------------------------------
#  Golden test 1 — q_flow reproduces the measured base-case flow
# ---------------------------------------------------------------------------


def test_golden_1_base_case_q_flow(case):
    net, registry = case
    for corr in registry.values():
        for ln in corr.lines:
            v_a, v_b, p_a, q_a = _measured(net, ln)
            q_model = q_flow(v_a, v_b, p_a, ln.params)
            tol = max(0.5, 0.01 * abs(q_a))
            assert abs(q_model - q_a) <= tol, (
                f"line {ln.line_idx}: q_model={q_model:.4f} Mvar vs "
                f"q_meas={q_a:.4f} Mvar (tol {tol:.3f})"
            )


# ---------------------------------------------------------------------------
#  Golden test 2 — round trips
# ---------------------------------------------------------------------------


def test_golden_2_line_round_trip(case):
    net, registry = case
    for corr in registry.values():
        for ln in corr.lines:
            v_a, _v_b, p_a, q_a = _measured(net, ln)
            for q_target in (q_a, q_a + 5.0, q_a - 5.0):
                v_b_solved = v_sched_for_q(v_a, q_target, p_a, ln.params)
                q_back = q_flow(v_a, v_b_solved, p_a, ln.params)
                assert abs(q_back - q_target) <= 1e-6, (
                    f"line {ln.line_idx}: round trip error "
                    f"{abs(q_back - q_target):.2e} Mvar at target "
                    f"{q_target:.2f} Mvar"
                )


# ---------------------------------------------------------------------------
#  Golden test 3 — sensitivities vs perturbed pandapower re-run
# ---------------------------------------------------------------------------


def test_golden_3_sensitivities_vs_rerun(case):
    net, registry = case
    net_pert = copy.deepcopy(net)
    net_pert.load["p_mw"] = net_pert.load["p_mw"] * 1.01
    net_pert.load["q_mvar"] = net_pert.load["q_mvar"] * 1.01
    pp.runpp(net_pert)

    for corr in registry.values():
        for ln in corr.lines:
            v_a0, v_b0, p_a0, q_a0 = _measured(net, ln)
            v_a1, v_b1, p_a1, q_a1 = _measured(net_pert, ln)
            s_a, s_b, s_p = sensitivities(v_a0, v_b0, p_a0, ln.params)
            dq_pred = (s_a * (v_a1 - v_a0) + s_b * (v_b1 - v_b0)
                       + s_p * (p_a1 - p_a0))
            dq_act = q_a1 - q_a0
            tol = max(0.05 * abs(dq_act), 0.1)
            assert abs(dq_pred - dq_act) <= tol, (
                f"line {ln.line_idx}: Δq_pred={dq_pred:.4f} vs "
                f"Δq_act={dq_act:.4f} Mvar (tol {tol:.4f}; "
                f"Δv_a={v_a1 - v_a0:.2e}, Δv_b={v_b1 - v_b0:.2e}, "
                f"Δp={p_a1 - p_a0:.3f} MW)"
            )


def test_golden_3_corridor_sums_consistent(case):
    """Corridor per-side sums equal the sums of the per-line triplets."""
    net, registry = case
    for corr in registry.values():
        v_a, v_b, p_a, _ = _corridor_measurements(net, corr)
        per_line, s_corr_a, s_corr_b = corridor_sensitivities(
            corr, v_a, v_b, p_a
        )
        assert len(per_line) == corr.n_lines
        assert s_corr_a == pytest.approx(sum(s[0] for s in per_line))
        assert s_corr_b == pytest.approx(sum(s[1] for s in per_line))


# ---------------------------------------------------------------------------
#  Golden test 4 — base-case consistency of the contract standard
# ---------------------------------------------------------------------------


def test_golden_4_contract_default_consistency(case):
    net, registry = case
    report = []
    failures = []
    for key, corr in registry.items():
        v_a, v_b, p_a, q_a = _corridor_measurements(net, corr)
        v_std_a = [round(v, V_STD_DECIMALS) for v in v_a]
        v_std_b = [round(v, V_STD_DECIMALS) for v in v_b]
        q_corr_meas = sum(q_a)
        q_std = corridor_q_flow(corr, v_std_a, v_std_b, p_a)
        tol = max(0.5, 0.01 * abs(q_corr_meas))
        dev = q_std - q_corr_meas
        report.append(
            f"corridor {key}: q_std={q_std:.3f} Mvar, "
            f"q_meas={q_corr_meas:.3f} Mvar, dev={dev:+.3f} (tol {tol:.3f})"
        )
        if abs(dev) > tol:
            failures.append(key)
    assert not failures, (
        "contract-default q_std deviates from the base-case corridor flow "
        "beyond the golden tolerance (1e-5 pu contract rounding):\n  "
        + "\n  ".join(report)
    )
