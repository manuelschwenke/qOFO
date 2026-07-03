"""
BME Phase 2 — tests for :class:`controller.common_objective.CommonObjective`.

Spec references (docs/BME_STATUS.md; BME spec §3.3, §3.4, §3.5, §5 Phase 2):
* partition invariant Σ_i Φ_i(v) == Φ_global(v) at the base point and at
  ≥ 20 randomised operating points (fixed seed);
* finite-difference validation of μ_i = dΦ_i/dv_b for the REAL Φ_i
  (losses + band) over every adjacent boundary bus — own ports exercise
  the internal-response chain, far tie endpoints exercise the direct
  tie-share terms;
* φ_band hinge behaviour: value and one-sided gradients at the edges;
* finite-difference validation of the Convention-A port-frozen own
  gradient ∂Φ_i/∂u_i |_{v_b fixed} for the actuator primitives
  (Q injection, V_gen, OLTC tap, shunt step).

The FD oracle extends the Phase 1 port sub-network: zone closure plus the
far endpoints of the zone's ties, with EVERY adjacent boundary bus pinned
to the plant operating point by a voltage source. This is the test
oracle's privilege; CommonObjective itself is area-local.
"""
from __future__ import annotations

import copy

import numpy as np
import pandapower as pp
import pytest
from pandapower.toolbox import select_subnet

from controller.common_objective import CommonObjective
from network.boundary_topology import BoundaryTopology
from network.ieee39.build import build_ieee39_net
from network.zone_partition import fixed_zone_partition_ieee39
from sensitivity.jacobian import JacobianSensitivities
from sensitivity.marginal_computer import MarginalComputer


@pytest.fixture(scope="module")
def case():
    net, _ = build_ieee39_net()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)
    zone_map, _ = fixed_zone_partition_ieee39(net)
    topo = BoundaryTopology(net, zone_map)
    sens = JacobianSensitivities(net)
    computers = {
        z: MarginalComputer(sens, topo, z) for z in topo.zone_ids
    }
    return net, topo, sens, computers


@pytest.fixture(scope="module")
def objectives(case):
    """Three rungs: losses-only (w_band = 0 ablation, D2), a tight
    heavily weighted band that is active at the operating point, and the
    TS-level scope of Q7 (vn_kv ≥ 220 → 345 kV network only; machine
    trafos and generator terminal buses excluded)."""
    _, topo, _, _ = case
    return {
        "loss_only": CommonObjective(topo, w_band=0.0),
        "tight_band": CommonObjective(
            topo, w_band=100.0, v_soft_min=0.99, v_soft_max=1.01,
        ),
        "ts_only": CommonObjective(
            topo, w_band=100.0, v_soft_min=0.99, v_soft_max=1.01,
            vn_kv_min=220.0,
        ),
    }


def _build_adjacent_subnet(net, topo, zone):
    """Zone closure + far tie endpoints, every adjacent boundary bus
    pinned to the plant operating point by a voltage source."""
    keep = sorted(
        set(topo.zone_buses(zone)) | set(topo.adjacent_boundary(zone))
    )
    sub = select_subnet(net, buses=keep, include_results=False)
    for p in topo.adjacent_boundary(zone):
        if len(sub.ext_grid) and (sub.ext_grid.bus == p).any():
            continue
        pp.create_ext_grid(
            sub, bus=p,
            vm_pu=float(net.res_bus.at[p, "vm_pu"]),
            va_degree=float(net.res_bus.at[p, "va_degree"]),
            name=f"BME_PORT_{p}",
        )
    pp.runpp(sub, run_control=False, calculate_voltage_angles=True)
    return sub


def _phi_of_perturbed(sub0, mutate, obj, zone):
    """Deep-copy the pinned sub-network, apply ``mutate``, re-run the
    power flow and evaluate Φ_zone."""
    sub = copy.deepcopy(sub0)
    mutate(sub)
    pp.runpp(sub, run_control=False, calculate_voltage_angles=True)
    return obj.phi_zone(sub, zone).total


# ======================================================================
#  φ_band hinge (§3.3)
# ======================================================================

def test_phi_band_hinge_value_and_gradient(case):
    _, topo, _, _ = case
    w = 2.5
    obj = CommonObjective(
        topo, w_band=w, v_soft_min=0.97, v_soft_max=1.03,
    )
    # Zero inside the band and exactly at the edges
    for v in (0.97, 0.99, 1.0, 1.03):
        assert obj.phi_band(v) == 0.0
        assert obj.phi_band_grad(v) == 0.0
    # Quadratic outside
    d = 0.004
    assert obj.phi_band(1.03 + d) == pytest.approx(w * d * d, rel=1e-12)
    assert obj.phi_band(0.97 - d) == pytest.approx(w * d * d, rel=1e-12)
    assert obj.phi_band_grad(1.03 + d) == pytest.approx(2 * w * d, rel=1e-12)
    assert obj.phi_band_grad(0.97 - d) == pytest.approx(-2 * w * d, rel=1e-12)
    # C¹ at the edges: gradient continuous ...
    eps = 1e-9
    assert abs(obj.phi_band_grad(1.03 + eps)) < 1e-6
    assert abs(obj.phi_band_grad(0.97 - eps)) < 1e-6
    # ... with one-sided curvature 2·w_band outside, 0 inside
    h = 1e-5
    upper_out = (obj.phi_band_grad(1.03 + h) - obj.phi_band_grad(1.03)) / h
    upper_in = (obj.phi_band_grad(1.03) - obj.phi_band_grad(1.03 - h)) / h
    assert upper_out == pytest.approx(2 * w, rel=1e-9)
    assert upper_in == 0.0
    lower_out = (obj.phi_band_grad(0.97) - obj.phi_band_grad(0.97 - h)) / h
    lower_in = (obj.phi_band_grad(0.97 + h) - obj.phi_band_grad(0.97)) / h
    assert lower_out == pytest.approx(2 * w, rel=1e-9)
    assert lower_in == 0.0


def test_constructor_validation(case):
    _, topo, _, _ = case
    with pytest.raises(ValueError, match="must be ≥ 0"):
        CommonObjective(topo, w_band=-1.0)
    with pytest.raises(ValueError, match="v_soft_min"):
        CommonObjective(topo, w_band=1.0, v_soft_min=1.03, v_soft_max=0.97)


# ======================================================================
#  Partition invariant (§3.3, the Phase 2 correctness invariant)
# ======================================================================

@pytest.mark.parametrize("which", ["loss_only", "tight_band", "ts_only"])
def test_partition_invariant_randomised(case, objectives, which):
    """Σ_i Φ_i == Φ_global at the base point and 20 randomised operating
    points (loads ±10 %, generator setpoints ±0.01 pu, fixed seed)."""
    net, topo, _, _ = case
    obj = objectives[which]
    rng = np.random.default_rng(123)
    for k in range(21):
        net2 = copy.deepcopy(net)
        if k > 0:
            net2.load["p_mw"] *= 1.0 + rng.uniform(-0.1, 0.1, len(net2.load))
            net2.load["q_mvar"] *= 1.0 + rng.uniform(-0.1, 0.1, len(net2.load))
            net2.gen["vm_pu"] += rng.uniform(-0.01, 0.01, len(net2.gen))
            pp.runpp(net2, run_control=False, calculate_voltage_angles=True)
        total = sum(
            obj.phi_zone(net2, z).total for z in topo.zone_ids
        )
        glob = obj.phi_global(net2)
        assert abs(total - glob) <= 1e-9 * max(1.0, abs(glob)), (
            f"[{which}] operating point {k}: Σ Φ_i = {total!r} "
            f"vs Φ_global = {glob!r}"
        )


def test_phi_global_loss_matches_res_tables(case, objectives):
    """Independent oracle: with w_band = 0, Φ_global is exactly the sum
    of the res-table branch losses."""
    net, _, _, _ = case
    obj = objectives["loss_only"]
    expected = float(net.res_line.pl_mw.sum()) + float(
        net.res_trafo.pl_mw.sum()
    )
    if hasattr(net, "trafo3w") and len(net.trafo3w):
        expected += float(net.res_trafo3w.pl_mw.sum())
    assert obj.phi_global(net) == pytest.approx(expected, rel=1e-12)


def test_phi_zone_manual_tie_split(case, objectives):
    """Zone 1's loss equals interior branch losses plus HALF the losses
    of its ties (lines 2, 14, 25 — audit A3), recomputed by hand."""
    net, topo, _, _ = case
    obj = objectives["loss_only"]
    tie_lines_z1 = {2, 14, 25}
    manual = 0.0
    for li in net.line.index:
        zf = topo.bus_owner(int(net.line.at[li, "from_bus"]))
        zt = topo.bus_owner(int(net.line.at[li, "to_bus"]))
        pl = float(net.res_line.at[li, "pl_mw"])
        if zf == zt == 1:
            manual += pl
        elif int(li) in tie_lines_z1:
            manual += 0.5 * pl
    for t in net.trafo.index:
        if topo.bus_owner(int(net.trafo.at[t, "hv_bus"])) == 1:
            manual += float(net.res_trafo.at[t, "pl_mw"])
    got = obj.phi_zone(net, 1)
    assert got.loss_mw == pytest.approx(manual, rel=1e-12)
    assert got.band_penalty == 0.0


def test_phi_zone_without_powerflow_raises(case, objectives):
    net, _, _, _ = case
    obj = objectives["loss_only"]
    stripped = copy.deepcopy(net)
    stripped.res_bus = stripped.res_bus.iloc[0:0]
    with pytest.raises(ValueError, match="power flow"):
        obj.phi_global(stripped)


# ======================================================================
#  μ = dΦ/dv_b against the port finite difference (§3.4)
# ======================================================================

@pytest.mark.parametrize("which", ["loss_only", "tight_band", "ts_only"])
def test_mu_matches_adjacent_fd(case, objectives, which):
    """Money test of Phase 2: μ from the area-local machinery equals the
    finite difference dΦ_i/dv_b at EVERY adjacent boundary bus. Own
    ports exercise the internal-response chain plus direct terms; far
    tie endpoints exercise the pure direct tie-share terms."""
    net, topo, _, computers = case
    obj = objectives[which]
    delta = 3e-4
    for zone in topo.zone_ids:
        comp = computers[zone]
        mu = obj.gradients(comp).mu()
        sub0 = _build_adjacent_subnet(net, topo, zone)
        # Consistency: the pinned sub-network reproduces Φ_i itself.
        assert obj.phi_zone(sub0, zone).total == pytest.approx(
            obj.phi_zone(net, zone).total, rel=1e-4,
        )
        for b in comp.adjacent:
            fd = []
            for sign in (+1.0, -1.0):
                def bump(sub, b=b, sign=sign):
                    mask = sub.ext_grid.bus == b
                    assert mask.any()
                    eg = sub.ext_grid.index[mask][0]
                    sub.ext_grid.at[eg, "vm_pu"] = (
                        float(sub.ext_grid.at[eg, "vm_pu"]) + sign * delta
                    )
                fd.append(_phi_of_perturbed(sub0, bump, obj, zone))
            dphi_fd = (fd[0] - fd[1]) / (2.0 * delta)
            mu_b = mu[topo.registry_pos[b]]
            scale = max(abs(dphi_fd), abs(mu_b), 1e-9)
            assert abs(dphi_fd - mu_b) <= 0.02 * scale + 1e-6, (
                f"[{which}] zone {zone}, boundary bus {b}: "
                f"μ={mu_b:.6e} vs FD={dphi_fd:.6e}"
            )


def test_mu_sparsity_real_phi(case, objectives):
    """μ entries at non-adjacent boundary buses are EXACTLY zero for the
    real Φ_i (§3.4 sparsity, enforced not approximate)."""
    _, topo, _, computers = case
    obj = objectives["tight_band"]
    for zone in topo.zone_ids:
        comp = computers[zone]
        mu = obj.gradients(comp).mu()
        adjacent = set(comp.adjacent)
        for b in topo.registry:
            if b not in adjacent:
                assert mu[topo.registry_pos[b]] == 0.0


# ======================================================================
#  Convention-A port-frozen own gradient (§3.5)
# ======================================================================

@pytest.mark.parametrize("which", ["loss_only", "tight_band"])
def test_grad_q_injection_fd(case, objectives, which):
    """∂Φ_i/∂Q |_{v_b fixed}: inject ±2 Mvar at interior PQ buses of the
    pinned sub-network and compare the central difference."""
    net, topo, _, computers = case
    obj = objectives[which]
    dq = 2.0
    for zone in topo.zone_ids:
        comp = computers[zone]
        grads = obj.gradients(comp)
        sub0 = _build_adjacent_subnet(net, topo, zone)
        pq = comp.interior_pq_buses
        for bus in (pq[0], pq[len(pq) // 2]):
            tangent = grads.d_q_injection(bus)
            fd = []
            for sign in (+1.0, -1.0):
                def bump(sub, bus=bus, sign=sign):
                    pp.create_sgen(sub, bus=bus, p_mw=0.0,
                                   q_mvar=sign * dq, name="BME_FD")
                fd.append(_phi_of_perturbed(sub0, bump, obj, zone))
            dphi_fd = (fd[0] - fd[1]) / (2.0 * dq)
            scale = max(abs(dphi_fd), abs(tangent), 1e-9)
            assert abs(dphi_fd - tangent) <= 0.05 * scale + 1e-5, (
                f"[{which}] zone {zone}, bus {bus}: "
                f"d_q={tangent:.6e} vs FD={dphi_fd:.6e}"
            )


@pytest.mark.parametrize("which", ["loss_only", "tight_band", "ts_only"])
def test_grad_vgen_fd(case, objectives, which):
    """∂Φ_i/∂V_gen |_{v_b fixed}: perturb one non-slack generator
    setpoint per zone by ±0.002 pu in the pinned sub-network."""
    net, topo, _, computers = case
    obj = objectives[which]
    dv = 0.002
    for zone in topo.zone_ids:
        comp = computers[zone]
        grads = obj.gradients(comp)
        sub0 = _build_adjacent_subnet(net, topo, zone)
        gen_idx = None
        for g in net.gen.index:
            if bool(net.gen.at[g, "slack"]):
                continue
            if topo.bus_owner(int(net.gen.at[g, "bus"])) == zone:
                gen_idx = int(g)
                break
        assert gen_idx is not None, f"zone {zone}: no non-slack gen found"
        tangent = grads.d_vgen(gen_idx)
        fd = []
        for sign in (+1.0, -1.0):
            def bump(sub, g=gen_idx, sign=sign):
                sub.gen.at[g, "vm_pu"] = (
                    float(sub.gen.at[g, "vm_pu"]) + sign * dv
                )
            fd.append(_phi_of_perturbed(sub0, bump, obj, zone))
        dphi_fd = (fd[0] - fd[1]) / (2.0 * dv)
        scale = max(abs(dphi_fd), abs(tangent), 1e-9)
        assert abs(dphi_fd - tangent) <= 0.05 * scale + 1e-5, (
            f"[{which}] zone {zone}, gen {gen_idx}: "
            f"d_vgen={tangent:.6e} vs FD={dphi_fd:.6e}"
        )


@pytest.mark.parametrize(
    "zone,trafo", [(1, 9), (1, 0), (2, 1), (3, 6)],
)
def test_grad_tap_fd(case, objectives, zone, trafo):
    """∂Φ_i/∂s |_{v_b fixed} per whole tap step: secant over ±1 step in
    the pinned sub-network vs the tangent (≤ 15 %, Phase 1 precedent for
    whole-step discrete moves). Trafo 0 covers the port-hv edge case
    (hv bus 1 is a boundary port)."""
    net, topo, _, computers = case
    obj = objectives["tight_band"]
    comp = computers[zone]
    grads = obj.gradients(comp)
    sub0 = _build_adjacent_subnet(net, topo, zone)
    tangent = grads.d_tap_2w(trafo)
    fd = []
    for sign in (+1, -1):
        def bump(sub, t=trafo, sign=sign):
            sub.trafo.at[t, "tap_pos"] = (
                float(sub.trafo.at[t, "tap_pos"]) + sign
            )
        fd.append(_phi_of_perturbed(sub0, bump, obj, zone))
    secant = (fd[0] - fd[1]) / 2.0
    scale = max(abs(secant), abs(tangent), 1e-9)
    assert abs(secant - tangent) <= 0.15 * scale + 1e-5, (
        f"zone {zone}, trafo {trafo}: d_tap={tangent:.6e} "
        f"vs secant={secant:.6e}"
    )


def test_grad_shunt_fd(case, objectives):
    """∂Φ_i/∂s |_{v_b fixed} per shunt step: synthetic 20 Mvar bank at an
    interior PQ bus, secant over step ±1 vs the tangent (≤ 15 %)."""
    net, topo, _, computers = case
    obj = objectives["tight_band"]
    q_step = 20.0
    for zone in topo.zone_ids:
        comp = computers[zone]
        grads = obj.gradients(comp)
        sub0 = _build_adjacent_subnet(net, topo, zone)
        bus = comp.interior_pq_buses[0]
        tangent = grads.d_shunt(bus, q_step)
        fd = []
        for sign in (+1, -1):
            def bump(sub, bus=bus, sign=sign):
                pp.create_shunt(sub, bus=bus, q_mvar=q_step,
                                step=sign, name="BME_FD")
            fd.append(_phi_of_perturbed(sub0, bump, obj, zone))
        secant = (fd[0] - fd[1]) / 2.0
        scale = max(abs(secant), abs(tangent), 1e-9)
        assert abs(secant - tangent) <= 0.15 * scale + 1e-5, (
            f"zone {zone}, bus {bus}: d_shunt={tangent:.6e} "
            f"vs secant={secant:.6e}"
        )


def test_grad_tap_fd_ts_scope(case, objectives):
    """Q7: for a machine trafo (345/10.5 kV, outside the TS scope) the
    explicit ∂P_ℓ/∂τ term is weighted to zero — the tap stays an
    actuator whose only Φ effect is the indirect response of the 345 kV
    network. FD must still match."""
    net, topo, _, computers = case
    obj = objectives["ts_only"]
    zone, trafo = 1, 9
    comp = computers[zone]
    grads = obj.gradients(comp)
    sub0 = _build_adjacent_subnet(net, topo, zone)
    tangent = grads.d_tap_2w(trafo)
    fd = []
    for sign in (+1, -1):
        def bump(sub, t=trafo, sign=sign):
            sub.trafo.at[t, "tap_pos"] = (
                float(sub.trafo.at[t, "tap_pos"]) + sign
            )
        fd.append(_phi_of_perturbed(sub0, bump, obj, zone))
    secant = (fd[0] - fd[1]) / 2.0
    scale = max(abs(secant), abs(tangent), 1e-9)
    assert abs(secant - tangent) <= 0.15 * scale + 1e-5, (
        f"d_tap={tangent:.6e} vs secant={secant:.6e}"
    )


def test_ts_scope_manual_values(case, objectives):
    """Q7 value semantics on IEEE 39: with vn_kv_min = 220 the loss term
    is the 345 kV network only (all lines plus the two 345/345 kV
    interconnecting trafos), and the band covers only 345 kV buses."""
    net, _, _, _ = case
    obj = objectives["ts_only"]

    in_scope_trafos = [
        t for t in net.trafo.index
        if float(net.trafo.at[t, "vn_lv_kv"]) >= 220.0
    ]
    assert in_scope_trafos, "expected 345/345 kV trafos in IEEE 39"
    expected_loss = float(net.res_line.pl_mw.sum()) + float(
        sum(net.res_trafo.at[t, "pl_mw"] for t in in_scope_trafos)
    )
    expected_band = sum(
        obj.phi_band(float(net.res_bus.at[b, "vm_pu"]))
        for b in net.bus.index
        if bool(net.bus.at[b, "in_service"])
        and float(net.bus.at[b, "vn_kv"]) >= 220.0
    )
    assert obj.phi_global(net) == pytest.approx(
        obj.w_loss * expected_loss + expected_band, rel=1e-12,
    )
    # And the scope genuinely excludes something: the include-all rung
    # counts the machine-trafo losses on top.
    full = objectives["tight_band"].phi_global(net)
    assert full > obj.phi_global(net)


# ======================================================================
#  Locality, conventions and fail-fast behaviour
# ======================================================================

def test_foreign_actuator_raises(case, objectives):
    """A zone may only evaluate port-frozen responses of its OWN
    actuators (§3.9 locality)."""
    net, topo, _, computers = case
    obj = objectives["loss_only"]
    grads = obj.gradients(computers[1])
    foreign = computers[2].interior_pq_buses[0]
    with pytest.raises(ValueError, match="belongs to zone"):
        grads.d_q_injection(foreign)


def test_q_injection_at_pv_bus_raises(case, objectives):
    """A reactive injection at a pinned PV bus has no Jacobian channel —
    fail fast rather than returning a silent zero."""
    net, topo, _, computers = case
    obj = objectives["loss_only"]
    for g in net.gen.index:
        if bool(net.gen.at[g, "slack"]):
            continue
        bus = int(net.gen.at[g, "bus"])
        zone = topo.bus_owner(bus)
        grads = obj.gradients(computers[zone])
        with pytest.raises(ValueError, match="no voltage state"):
            grads.d_q_injection(bus)
        break


def test_pcc_column_negation(case, objectives):
    """Q_PCC_set uses the load convention: exactly the negated injection
    column (mirrors controller and RestrictedSensitivityProvider)."""
    _, _, _, computers = case
    obj = objectives["tight_band"]
    grads = obj.gradients(computers[2])
    bus = computers[2].interior_pq_buses[0]
    assert grads.d_pcc_set(bus) == -grads.d_q_injection(bus)


def test_slack_machine_not_an_actuator(case, objectives):
    """The slack machine's V is not a zone input (Phase 1 convention)."""
    net, topo, _, computers = case
    obj = objectives["loss_only"]
    slack = [g for g in net.gen.index if bool(net.gen.at[g, "slack"])]
    assert slack
    zone = topo.bus_owner(int(net.gen.at[slack[0], "bus"]))
    grads = obj.gradients(computers[zone])
    with pytest.raises(ValueError, match="slack machine"):
        grads.d_vgen(int(slack[0]))


def test_topology_mismatch_raises(case, objectives):
    """CommonObjective and MarginalComputer must share one topology."""
    net, _, _, computers = case
    zone_map, _ = fixed_zone_partition_ieee39(net)
    other_topo = BoundaryTopology(net, zone_map)
    other_obj = CommonObjective(other_topo, w_band=0.0)
    with pytest.raises(ValueError, match="same BoundaryTopology"):
        other_obj.gradients(computers[1])
