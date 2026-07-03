"""
BME Phase 4 — the hard-gate gradient identity tests (spec §3.5, §5).

Test 2 of §3.5 ("the money test of the whole design"): on the 3-area
IEEE 39 case, the stacked distributed gradients

    g_i^bme = ∂Φ_i/∂u_i |_{v_b fixed}  +  H_{b,i}ᵀ · Σ_{all j} μ_j

(Convention A, d = 0, β = 1 — the neighbour marginals travel through the
REAL CoordinationBus/MarginalReceiver chain, the self-marginal is added
locally) must equal the finite-difference gradient of the GLOBAL Φ with
respect to the stacked u, at randomised operating points. Continuous
columns are compared with tight tolerance; discrete columns (whole tap /
shunt steps) as secant-vs-tangent, matching the Phase 1/2 precedent.

Test 1 of §3.5 (single-area identity): with a one-zone partition there
is no boundary — the port-frozen operators become TOTAL-response
operators, μ is the empty vector and the price term vanishes; the
distributed gradient must equal dΦ/du outright.

Objective configuration: Q7 (TS-level scope, vn_kv ≥ 220) with an active
tight band — the configuration the BME experiments will run.
"""
from __future__ import annotations

import copy

import numpy as np
import pandapower as pp
import pytest

from controller.bme_gradient import BMEGradientAssembler, pcc_hv_buses
from controller.common_objective import CommonObjective
from core.coordination_bus import (
    CoordinationBus,
    MarginalReceiver,
    MarginalSignal,
)
from network.boundary_topology import BoundaryTopology
from network.ieee39.build import build_ieee39_net
from network.zone_partition import fixed_zone_partition_ieee39
from sensitivity.boundary_sensitivity import (
    RestrictedSensitivityProvider,
    ZoneInputSpec,
)
from sensitivity.jacobian import JacobianSensitivities
from sensitivity.marginal_computer import MarginalComputer

# FD step sizes (continuous chosen small enough that a band-edge kink
# inside the central-difference window stays within tolerance)
DQ_MVAR = 1.5
DV_PU = 0.0015
TOL_CONT = 0.05    # relative, continuous columns
TOL_DISC = 0.15    # relative, whole-step discrete columns
ATOL = 1e-4        # MW-scale absolute floor


@pytest.fixture(scope="module")
def base():
    net, _ = build_ieee39_net()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)
    zone_map, _ = fixed_zone_partition_ieee39(net)
    topo = BoundaryTopology(net, zone_map)
    obj = CommonObjective(
        topo, w_band=100.0, v_soft_min=0.99, v_soft_max=1.01,
        vn_kv_min=220.0,
    )
    return net, zone_map, topo, obj


def _make_op(net, seed):
    """Randomised operating point (None → the base point)."""
    net2 = copy.deepcopy(net)
    if seed is not None:
        rng = np.random.default_rng(seed)
        net2.load["p_mw"] *= 1.0 + rng.uniform(-0.08, 0.08, len(net2.load))
        net2.load["q_mvar"] *= 1.0 + rng.uniform(-0.08, 0.08, len(net2.load))
        net2.gen.loc[~net2.gen.slack, "vm_pu"] += rng.uniform(
            -0.008, 0.008, int((~net2.gen.slack).sum())
        )
        pp.runpp(net2, run_control=False, calculate_voltage_angles=True)
    return net2


def _zone_gens(net, topo):
    out = {z: [] for z in topo.zone_ids}
    for g in net.gen.index:
        if bool(net.gen.at[g, "slack"]):
            continue
        out[topo.bus_owner(int(net.gen.at[g, "bus"]))].append(int(g))
    return out


def _ts_interior_pq(comp, net):
    return [
        b for b in comp.interior_pq_buses
        if float(net.bus.at[b, "vn_kv"]) >= 220.0
    ]


def _build_specs(net, topo, computers, *, full: bool):
    """Deterministic ZoneInputSpecs. ``full`` covers every column class
    (incl. the PCC stand-in and the port-hv trafo 0 edge case); the
    reduced variant keeps the 10-point sweep cheap."""
    gens = _zone_gens(net, topo)
    oltc_full = {1: (9, 0), 2: (1,), 3: (6,)}
    oltc_reduced = {1: (9,), 2: (1,), 3: (6,)}
    specs = {}
    for z in topo.zone_ids:
        ts_pq = _ts_interior_pq(computers[z], net)
        assert len(ts_pq) >= 3
        if full:
            specs[z] = ZoneInputSpec(
                zone_id=z,
                der_bus_indices=tuple(ts_pq[:2]),
                # PCC stand-in: the 345/345 kV trafo 3 (hv bus in zone 2)
                # exercises the load-convention column on the base net;
                # real 3W couplers land with the runner nets.
                pcc_trafo_indices=(3,) if z == 2 else (),
                gen_indices=(gens[z][0],),
                oltc_trafo_indices=oltc_full[z],
                shunt_bus_indices=(ts_pq[2],),
                shunt_q_steps_mvar=(20.0,),
            )
        else:
            specs[z] = ZoneInputSpec(
                zone_id=z,
                der_bus_indices=(ts_pq[0],),
                gen_indices=(gens[z][0],),
                oltc_trafo_indices=oltc_reduced[z],
            )
    return specs


def _distributed_gradients(net_op, topo, obj, specs):
    """The full distributed chain at one operating point: Jacobian →
    marginal computers → zone gradients → μ exchange over the REAL bus
    (d = 0, β = 1) → assembled g_i^bme per zone."""
    sens = JacobianSensitivities(net_op)
    computers = {z: MarginalComputer(sens, topo, z) for z in topo.zone_ids}
    provider = RestrictedSensitivityProvider(sens, topo, specs)
    grads = {z: obj.gradients(computers[z]) for z in topo.zone_ids}
    assemblers = {
        z: BMEGradientAssembler(specs[z], grads[z], provider.view(z))
        for z in topo.zone_ids
    }

    # Stacked complex-boundary coordinates (D7 revised): 2|B| entries,
    # [Vm_b | θ_b]; the v_b snapshot carries the same stacking.
    bus = CoordinationBus(
        topo.zone_ids, 2 * len(topo.registry), delay_steps=0,
    )
    receivers = {
        z: MarginalReceiver(z, bus, beta=1.0, start_step=0)
        for z in topo.zone_ids
    }
    mus = {z: assemblers[z].mu() for z in topo.zone_ids}
    v_b = np.concatenate([
        [float(net_op.res_bus.at[b, "vm_pu"]) for b in topo.registry],
        [np.deg2rad(float(net_op.res_bus.at[b, "va_degree"]))
         for b in topo.registry],
    ])
    for z in topo.zone_ids:
        bus.publish_marginal(
            MarginalSignal(zone_id=z, step=0, mu=mus[z], v_b_meas=v_b)
        )
    g = {}
    for z in topo.zone_ids:
        out = receivers[z].update(0)
        assert out.coordinated  # d = 0: no cold-start window
        g[z] = assemblers[z].g_bme(mus[z] + out.mu_neighbour_sum)
    return g


def _phi_perturbed(net_op, obj, mutate):
    net2 = copy.deepcopy(net_op)
    mutate(net2)
    pp.runpp(net2, run_control=False, calculate_voltage_angles=True)
    return obj.phi_global(net2)


def _fd_column(net_op, obj, kind, ident, spec):
    """Central FD of the GLOBAL Φ on the FULL plant net for one input
    column. Returns (value, is_discrete)."""
    if kind == "der":
        f = [
            _phi_perturbed(net_op, obj, lambda n, s=s: pp.create_sgen(
                n, bus=int(ident), p_mw=0.0, q_mvar=s * DQ_MVAR))
            for s in (+1.0, -1.0)
        ]
        return (f[0] - f[1]) / (2.0 * DQ_MVAR), False
    if kind == "pcc":
        hv_bus = pcc_hv_buses(net_op, spec)[
            list(spec.pcc_trafo_indices).index(int(ident))
        ]
        f = [
            _phi_perturbed(net_op, obj, lambda n, s=s: pp.create_load(
                n, bus=hv_bus, p_mw=0.0, q_mvar=s * DQ_MVAR))
            for s in (+1.0, -1.0)
        ]
        return (f[0] - f[1]) / (2.0 * DQ_MVAR), False
    if kind == "vgen":
        def bump(n, s):
            n.gen.at[int(ident), "vm_pu"] = (
                float(n.gen.at[int(ident), "vm_pu"]) + s * DV_PU
            )
        f = [
            _phi_perturbed(net_op, obj, lambda n, s=s: bump(n, s))
            for s in (+1.0, -1.0)
        ]
        return (f[0] - f[1]) / (2.0 * DV_PU), False
    if kind == "oltc":
        def tap(n, s):
            n.trafo.at[int(ident), "tap_pos"] = (
                float(n.trafo.at[int(ident), "tap_pos"]) + s
            )
        f = [
            _phi_perturbed(net_op, obj, lambda n, s=s: tap(n, s))
            for s in (+1, -1)
        ]
        return (f[0] - f[1]) / 2.0, True
    if kind == "shunt":
        q_step = spec.shunt_q_steps_mvar[
            list(spec.shunt_bus_indices).index(int(ident))
        ]
        f = [
            _phi_perturbed(net_op, obj, lambda n, s=s: pp.create_shunt(
                n, bus=int(ident), q_mvar=q_step, step=s))
            for s in (+1, -1)
        ]
        return (f[0] - f[1]) / 2.0, True
    raise ValueError(kind)


def _assert_identity(net_op, topo, obj, specs, tag):
    g = _distributed_gradients(net_op, topo, obj, specs)
    for z in topo.zone_ids:
        labels = specs[z].column_labels()
        for i, (kind, ident) in enumerate(labels):
            fd, discrete = _fd_column(net_op, obj, kind, ident, specs[z])
            tol = TOL_DISC if discrete else TOL_CONT
            scale = max(abs(fd), abs(g[z][i]), 1e-9)
            assert abs(fd - g[z][i]) <= tol * scale + ATOL, (
                f"[{tag}] zone {z}, column {i} ({kind} {ident}): "
                f"g_bme={g[z][i]:.6e} vs FD={fd:.6e}"
            )


# ======================================================================
#  §3.5 test 2 — distributed == centralised (HARD GATE)
# ======================================================================

@pytest.mark.parametrize("seed", [None, 101, 102])
def test_identity_full_column_coverage(base, seed):
    """Every u-column class (DER, PCC stand-in, V_gen, two OLTCs incl.
    the port-hv trafo 0, shunt) at the base point + 2 randomised
    operating points."""
    net, _, topo, obj = base
    net_op = _make_op(net, seed)
    sens = JacobianSensitivities(net_op)
    computers = {z: MarginalComputer(sens, topo, z) for z in topo.zone_ids}
    specs = _build_specs(net_op, topo, computers, full=True)
    _assert_identity(net_op, topo, obj, specs, f"full/seed={seed}")


@pytest.mark.parametrize("seed", list(range(10)))
def test_identity_ten_randomised_points(base, seed):
    """Spec §5 Phase 4: ≥ 10 randomised operating points (reduced column
    set: one DER, one V_gen, one OLTC per zone)."""
    net, _, topo, obj = base
    net_op = _make_op(net, seed)
    sens = JacobianSensitivities(net_op)
    computers = {z: MarginalComputer(sens, topo, z) for z in topo.zone_ids}
    specs = _build_specs(net_op, topo, computers, full=False)
    _assert_identity(net_op, topo, obj, specs, f"reduced/seed={seed}")


# ======================================================================
#  §3.5 test 1 — single-area identity
# ======================================================================

def test_single_area_identity(base):
    """One-zone partition: no boundary, no ports, no price — the
    'port-frozen' own gradient IS the total gradient dΦ/du and μ is the
    empty vector."""
    net, zone_map, _, _ = base
    all_buses = sorted(b for buses in zone_map.values() for b in buses)
    topo1 = BoundaryTopology(net, {1: all_buses})
    assert topo1.registry == []
    obj1 = CommonObjective(
        topo1, w_band=100.0, v_soft_min=0.99, v_soft_max=1.01,
        vn_kv_min=220.0,
    )
    sens = JacobianSensitivities(net)
    comp = MarginalComputer(sens, topo1, 1)
    grads = obj1.gradients(comp)
    assert grads.mu().shape == (0,)

    assert grads.mu_stacked().shape == (0,)

    ts_pq = _ts_interior_pq(comp, net)
    gen = _zone_gens(net, topo1)[1][0]
    checks = [
        ("der", ts_pq[0], grads.d_q_injection(ts_pq[0]), False),
        ("vgen", gen, grads.d_vgen(gen), False),
        ("oltc", 9, grads.d_tap_2w(9), True),
    ]
    spec = ZoneInputSpec(
        zone_id=1, der_bus_indices=(ts_pq[0],), gen_indices=(gen,),
        oltc_trafo_indices=(9,),
    )
    for kind, ident, tangent, discrete in checks:
        fd, _ = _fd_column(net, obj1, kind, ident, spec)
        tol = TOL_DISC if discrete else TOL_CONT
        scale = max(abs(fd), abs(tangent), 1e-9)
        assert abs(fd - tangent) <= tol * scale + ATOL, (
            f"single-area {kind} {ident}: g={tangent:.6e} vs "
            f"FD={fd:.6e}"
        )


# ======================================================================
#  Assembler validation
# ======================================================================

def test_assembler_validation(base):
    net, _, topo, obj = base
    sens = JacobianSensitivities(net)
    computers = {z: MarginalComputer(sens, topo, z) for z in topo.zone_ids}
    specs = _build_specs(net, topo, computers, full=False)
    provider = RestrictedSensitivityProvider(sens, topo, specs)
    grads = obj.gradients(computers[1])
    with pytest.raises(ValueError, match="zone mismatch"):
        BMEGradientAssembler(specs[2], grads, provider.view(1))
    asm = BMEGradientAssembler(specs[1], grads, provider.view(1))
    with pytest.raises(ValueError, match="stacked boundary coordinate"):
        asm.g_bme(np.zeros(3))
