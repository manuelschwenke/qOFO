"""
BME Phase 1 — tests for
:class:`sensitivity.marginal_computer.MarginalComputer`.

Spec references (docs/BME_STATUS.md; BME spec §3.2, §3.4, §5 Phase 1):
* the §3.2 separator CONSEQUENCE holds on the 3-area IEEE 39 case: with
  the zone's own boundary voltages held fixed (ports = voltage sources),
  the zone's internal power flow reproduces the full-network operating
  point;
* finite-difference validation of μ: perturbing one adjacent boundary
  voltage as a port reproduces dΦ/dv_b for a synthetic quadratic Φ;
* sparsity: μ entries at non-adjacent boundary buses are EXACTLY zero,
  and direct terms outside the adjacent set raise.

The FD oracle builds a per-zone port sub-network (zone closure buses
only, ports replaced by voltage sources at the plant operating point) —
this is the test oracle's privilege; the MarginalComputer itself only
uses area-local Jacobian entries.
"""
from __future__ import annotations

import copy

import numpy as np
import pandapower as pp
import pytest
from pandapower.toolbox import select_subnet

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


def _build_port_subnet(net, topo, zone):
    """Zone closure sub-network with every port pinned to the plant
    operating point by a voltage source (test oracle for §3.2)."""
    keep = sorted(set(topo.zone_buses(zone)))
    sub = select_subnet(net, buses=keep, include_results=False)
    ports = topo.own_boundary(zone)
    for p in ports:
        if len(sub.ext_grid) and (sub.ext_grid.bus == p).any():
            continue  # the plant slack already pins this port
        pp.create_ext_grid(
            sub, bus=p,
            vm_pu=float(net.res_bus.at[p, "vm_pu"]),
            va_degree=float(net.res_bus.at[p, "va_degree"]),
            name=f"BME_PORT_{p}",
        )
    pp.runpp(sub, run_control=False, calculate_voltage_angles=True)
    return sub


def test_separator_consequence_port_subnet_reproduces_plant(case):
    """§3.2's consequence, asserted numerically: fixing the zone's own
    boundary voltages decouples its internal subproblem — the port
    sub-network reproduces the plant voltages at every interior bus."""
    net, topo, _, computers = case
    for zone in topo.zone_ids:
        sub = _build_port_subnet(net, topo, zone)
        for b in computers[zone].interior_pq_buses:
            assert abs(
                float(sub.res_bus.at[b, "vm_pu"])
                - float(net.res_bus.at[b, "vm_pu"])
            ) < 1e-6, f"zone {zone}, bus {b}: port subnet diverges"


def _synthetic_phi_and_grad(net_like, buses, seed):
    """Synthetic quadratic Φ = Σ w_n (V_n − a_n)² over ``buses`` with a
    fixed-seed weight/anchor draw. Returns (phi_fn, grad_at_plant)."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 2.0, size=len(buses))
    v0 = np.array([
        float(net_like.res_bus.at[b, "vm_pu"]) for b in buses
    ])
    a = v0 - rng.uniform(-0.02, 0.02, size=len(buses))

    def phi(some_net):
        v = np.array([
            float(some_net.res_bus.at[b, "vm_pu"]) for b in buses
        ])
        return float(np.sum(w * (v - a) ** 2))

    grad = 2.0 * w * (v0 - a)  # ∇_{v_int} Φ at the plant operating point
    return phi, grad


def test_mu_matches_port_finite_difference(case):
    """Money test of Phase 1: μ from the internal reduced Jacobian equals
    the port finite difference dΦ/dv_b for every zone and every port."""
    net, topo, _, computers = case
    delta = 3e-4
    for zone in topo.zone_ids:
        comp = computers[zone]
        phi, grad = _synthetic_phi_and_grad(
            net, comp.interior_pq_buses, seed=42 + zone,
        )
        mu = comp.mu(grad)
        sub0 = _build_port_subnet(net, topo, zone)
        for p in comp.ports:
            fd = []
            for sign in (+1.0, -1.0):
                sub = copy.deepcopy(sub0)
                mask = sub.ext_grid.bus == p
                assert mask.any()
                eg = sub.ext_grid.index[mask][0]
                sub.ext_grid.at[eg, "vm_pu"] = (
                    float(sub.ext_grid.at[eg, "vm_pu"]) + sign * delta
                )
                pp.runpp(
                    sub, run_control=False, calculate_voltage_angles=True,
                )
                fd.append(phi(sub))
            dphi_fd = (fd[0] - fd[1]) / (2.0 * delta)
            mu_p = mu[topo.registry_pos[p]]
            scale = max(abs(dphi_fd), abs(mu_p), 1e-9)
            assert abs(dphi_fd - mu_p) <= 0.02 * scale + 1e-7, (
                f"zone {zone}, port {p}: μ={mu_p:.6e} vs FD={dphi_fd:.6e}"
            )


def test_mu_sparsity_exact_zeros(case):
    """μ entries at boundary buses not adjacent to the zone are EXACTLY
    zero (§3.4) — enforced, not approximate."""
    net, topo, _, computers = case
    for zone in topo.zone_ids:
        comp = computers[zone]
        rng = np.random.default_rng(7)
        grad = rng.normal(size=len(comp.interior_pq_buses))
        mu = comp.mu(grad)
        adjacent = set(comp.adjacent)
        for b in topo.registry:
            if b not in adjacent:
                assert mu[topo.registry_pos[b]] == 0.0


def test_direct_terms(case):
    """Direct terms land on their registry slot; far endpoints of own
    ties are admissible; non-adjacent buses raise."""
    net, topo, _, computers = case
    comp = computers[1]  # zone 1: adjacent = [1, 2, 8, 16, 26, 38]
    grad = np.zeros(len(comp.interior_pq_buses))
    mu = comp.mu(grad, grad_direct={2: 1.5, 26: -0.5})
    assert mu[topo.registry_pos[2]] == 1.5    # far endpoint of tie 2
    assert mu[topo.registry_pos[26]] == -0.5  # own port
    with pytest.raises(ValueError, match="adjacent"):
        comp.mu(grad, grad_direct={13: 1.0})  # zone-2/3 tie bus: not ours


def test_wrong_grad_length_raises(case):
    _, _, _, computers = case
    comp = computers[2]
    with pytest.raises(ValueError, match="grad_v_int"):
        comp.mu(np.zeros(3))


def test_unknown_zone_raises(case):
    net, topo, sens, _ = case
    with pytest.raises(ValueError, match="unknown zone"):
        MarginalComputer(sens, topo, 99)
