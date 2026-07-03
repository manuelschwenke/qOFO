"""
BME Phase 1 — tests for
:class:`sensitivity.boundary_sensitivity.RestrictedSensitivityProvider`.

Spec references (docs/BME_STATUS.md; BME spec §3.5, §3.9, §5 Phase 1):
* finite-difference validation of H_{b,i} columns on the 3-area IEEE 39
  case — one column per actuator class, INCLUDING one whole tap step;
* pinned boundary buses (slack, IEEE 39 / 0-idx 38) yield exactly-zero rows;
* the access-restriction wrapper raises on out-of-scope access (§3.9).

FD conventions follow the repository's existing sensitivity tests
(tests/test_jacobian_qtie.py): perturb the plant net, re-run the power
flow, compare against the analytical column with explicit tolerances.
Whole-integer steps (tap, shunt) compare a secant against a tangent, so
their tolerance is looser than for the continuous perturbations.
"""
from __future__ import annotations

import copy

import numpy as np
import pandapower as pp
import pytest

from network.boundary_topology import BoundaryTopology
from network.ieee39.build import build_ieee39_net
from network.zone_partition import fixed_zone_partition_ieee39
from sensitivity.boundary_sensitivity import (
    RestrictedSensitivityProvider,
    ZoneInputSpec,
)
from sensitivity.jacobian import JacobianSensitivities

DER_TEST_BUS = 4     # 0-idx (IEEE 5), zone-2 interior PQ bus
SHUNT_TEST_BUS = 20  # 0-idx (IEEE 21), zone-3 interior PQ bus
SHUNT_Q_STEP_MVAR = 50.0


@pytest.fixture(scope="module")
def case():
    """IEEE 39 3-area case with one test DER (zone 2) and one test shunt
    (zone 3) so every actuator class of H_{b,i} has an FD handle."""
    net, meta = build_ieee39_net()
    der_idx = pp.create_sgen(
        net, bus=DER_TEST_BUS, p_mw=50.0, q_mvar=0.0, name="BME_TEST_DER",
    )
    shunt_idx = pp.create_shunt(
        net, bus=SHUNT_TEST_BUS, q_mvar=SHUNT_Q_STEP_MVAR, step=0,
        max_step=1, name="BME_TEST_SHUNT",
    )
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)
    zone_map, _ = fixed_zone_partition_ieee39(net)
    topo = BoundaryTopology(net, zone_map)
    sens = JacobianSensitivities(net)

    # Mirror the runner's ZoneDefinition conventions: the slack machine
    # and its trafo are NOT zone actuators (its terminal bus is the ppc
    # reference bus — no Jacobian column; see runner l. 1849-1851 and
    # IMPLEMENTATION_STATUS known issue 1).
    slack_gens = {
        int(g) for g in net.gen.index
        if "slack" in net.gen.columns and bool(net.gen.at[g, "slack"])
    }
    slack_trafos = {
        int(t) for t, g in zip(
            meta.machine_trafo_indices, meta.machine_trafo_gen_map,
        ) if int(g) in slack_gens
    }

    def gens_in_zone(z):
        return tuple(
            int(g) for g in net.gen.index
            if int(g) not in slack_gens
            and topo.bus_owner(int(net.gen.at[g, "bus"])) == z
        )

    def oltcs_in_zone(z):
        return tuple(
            int(t) for t in meta.machine_trafo_indices
            if int(t) not in slack_trafos
            and topo.bus_owner(int(net.trafo.at[t, "hv_bus"])) == z
        )

    specs = {
        1: ZoneInputSpec(zone_id=1, gen_indices=gens_in_zone(1),
                         oltc_trafo_indices=oltcs_in_zone(1)),
        2: ZoneInputSpec(zone_id=2, der_bus_indices=(DER_TEST_BUS,),
                         gen_indices=gens_in_zone(2),
                         oltc_trafo_indices=oltcs_in_zone(2)),
        3: ZoneInputSpec(zone_id=3, gen_indices=gens_in_zone(3),
                         oltc_trafo_indices=oltcs_in_zone(3),
                         shunt_bus_indices=(SHUNT_TEST_BUS,),
                         shunt_q_steps_mvar=(SHUNT_Q_STEP_MVAR,)),
    }
    provider = RestrictedSensitivityProvider(sens, topo, specs)
    return net, topo, provider, specs, der_idx, shunt_idx


def _fd_boundary_response(net, topo, mutate):
    """Δv_b (registry order) from re-running the PF after ``mutate(net2)``."""
    net2 = copy.deepcopy(net)
    mutate(net2)
    pp.runpp(net2, run_control=False, calculate_voltage_angles=True)
    return np.array([
        float(net2.res_bus.at[b, "vm_pu"]) - float(net.res_bus.at[b, "vm_pu"])
        for b in topo.registry
    ])


def _column(provider, specs, zone, kind, ident):
    H_b = provider.h_b(zone)
    labels = specs[zone].column_labels()
    return H_b[:, labels.index((kind, ident))]


def test_shapes_and_pinned_rows(case):
    """No boundary bus is pinned on this case: the system slack is a
    ``slack=True`` gen at a 10.5 kV terminal bus BEHIND boundary bus 38
    (machine trafo), so bus 38 itself is a PQ bus with a live V state.
    (Corrects the Phase 0 note that assumed the slack pinned bus 38.)"""
    net, topo, provider, specs, _, _ = case
    assert provider.pinned_boundary_buses == []
    for z in (1, 2, 3):
        H_b = provider.h_b(z)
        assert H_b.shape == (len(topo.registry), specs[z].n_columns)
        # At least one non-trivial entry per zone.
        assert np.max(np.abs(H_b)) > 0.0


def test_fd_vgen_column(case):
    """FD validation of a V_gen column: perturb one AVR setpoint."""
    net, topo, provider, specs, _, _ = case
    delta = 2e-4
    for zone in (1, 2, 3):
        gen = specs[zone].gen_indices[0]
        col = _column(provider, specs, zone, "vgen", gen)

        def mutate(n, g=gen):
            n.gen.at[g, "vm_pu"] = float(n.gen.at[g, "vm_pu"]) + delta

        fd = _fd_boundary_response(net, topo, mutate) / delta
        scale = max(np.max(np.abs(col)), 1e-9)
        assert np.max(np.abs(fd - col)) <= 0.05 * scale + 1e-5, (
            f"zone {zone} V_gen column (gen {gen}): "
            f"max err {np.max(np.abs(fd - col)):.3e}, scale {scale:.3e}"
        )


def test_fd_der_column(case):
    """FD validation of a Q_DER column: perturb the test DER's Q."""
    net, topo, provider, specs, _, _ = case
    col = _column(provider, specs, 2, "der", DER_TEST_BUS)
    dq = 5.0  # Mvar

    def mutate(n):
        s = n.sgen.index[n.sgen.name == "BME_TEST_DER"][0]
        n.sgen.at[s, "q_mvar"] = float(n.sgen.at[s, "q_mvar"]) + dq

    fd = _fd_boundary_response(net, topo, mutate) / dq
    scale = max(np.max(np.abs(col)), 1e-12)
    assert np.max(np.abs(fd - col)) <= 0.05 * scale + 1e-8, (
        f"Q_DER column: max err {np.max(np.abs(fd - col)):.3e}"
    )


def test_fd_oltc_tap_step_column(case):
    """FD validation of an OLTC column over ONE WHOLE tap step (spec
    Phase 1: 'perturb each actuator incl. one tap step'). Secant vs
    tangent ⇒ looser tolerance."""
    net, topo, provider, specs, _, _ = case
    for zone in (1, 2, 3):
        trafo = specs[zone].oltc_trafo_indices[0]
        col = _column(provider, specs, zone, "oltc", trafo)

        def mutate(n, t=trafo):
            n.trafo.at[t, "tap_pos"] = int(n.trafo.at[t, "tap_pos"]) + 1

        fd = _fd_boundary_response(net, topo, mutate)  # per one step
        scale = max(np.max(np.abs(col)), 1e-9)
        assert np.max(np.abs(fd - col)) <= 0.15 * scale + 2e-5, (
            f"zone {zone} OLTC column (trafo {trafo}): "
            f"max err {np.max(np.abs(fd - col)):.3e}, scale {scale:.3e}"
        )


def test_fd_shunt_step_column(case):
    """FD validation of a shunt column over one whole step (0 → 1)."""
    net, topo, provider, specs, _, shunt_idx = case
    col = _column(provider, specs, 3, "shunt", SHUNT_TEST_BUS)

    def mutate(n):
        n.shunt.at[shunt_idx, "step"] = 1

    fd = _fd_boundary_response(net, topo, mutate)  # per one step
    scale = max(np.max(np.abs(col)), 1e-9)
    assert np.max(np.abs(fd - col)) <= 0.15 * scale + 2e-5, (
        f"shunt column: max err {np.max(np.abs(fd - col)):.3e}"
    )
    # Reactor step (positive q_mvar, load convention) must LOWER voltages.
    assert col[topo.registry_pos[14]] < 0.0


def test_access_restriction_raises(case):
    """§3.9: out-of-scope access raises; a zone view exposes only its own
    H_{b,i}."""
    _, _, provider, _, _, _ = case
    with pytest.raises(PermissionError, match="not registered"):
        provider.h_b(99)
    with pytest.raises(PermissionError, match="not registered"):
        provider.view(99)
    view = provider.view(1)
    assert view.zone_id == 1
    H_b = view.h_b()
    assert H_b.shape[0] == 9
    # The view exposes no other read surface (h_b_stacked is the
    # D7-revised complex-boundary read — same informational scope).
    public = [a for a in dir(view) if not a.startswith("_")]
    assert set(public) == {"h_b", "h_b_stacked", "zone_id"}


def test_h_b_returns_copy(case):
    """Mutating a returned H_{b,i} must not poison the provider cache."""
    _, _, provider, _, _, _ = case
    a = provider.h_b(1)
    a[:] = 123.0
    b = provider.h_b(1)
    assert not np.any(b == 123.0)


def test_h_b_stacked_consistent_with_h_b(case):
    """D7 revision cross-check: the stacked assembly (generic full
    state-response path) must reproduce the magnitude rows of the
    legacy helper-based h_b to numerical identity, for every zone and
    every column class."""
    _, topo, provider, _, _, _ = case
    n_b = len(topo.registry)
    for z in topo.zone_ids:
        stacked = provider.h_b_stacked(z)
        legacy = provider.h_b(z)
        assert stacked.shape == (2 * n_b, legacy.shape[1])
        np.testing.assert_allclose(
            stacked[:n_b, :], legacy, rtol=1e-9, atol=1e-12,
        )
        # The angle block must be non-trivial (losses depend on it).
        assert np.max(np.abs(stacked[n_b:, :])) > 0.0
