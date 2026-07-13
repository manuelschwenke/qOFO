"""
SBX-H v6 — protocol tests for ``sbx_h.scheduler``.

Drives the v6 cycle (metering → settlement → schedule application →
escalation indicator) on the REAL IEEE 39 corridor registry and
contracts with synthetic measurements, so every transition is
deterministic and analytically checkable: steady state, q_std tracking
the persistence ``p_sched``, planned-support schedule switching,
violation-indicator recording, escalation, and directional hold/sag
support-energy settlement.  The v5 deal-protocol tests live in
``_archive/sbx_h_v5/``.
"""
from __future__ import annotations

import numpy as np
import pandapower as pp
import pytest

from network.ieee39.build import build_ieee39_net
from network.zone_partition import fixed_zone_partition_ieee39
from sbx_h.config import SBXConfig
from sbx_h.contract import build_default_contract, with_planned_support
from sbx_h.corridor import build_corridor_registry
from sbx_h.fail import SBXError
from sbx_h.scheduler import AreaStepInput, SBXScheduler

BOUND_OK = (0.90, 1.10)     # base-case voltages are comfortably inside
BOUND_UNDER = (1.04, 1.10)  # forces an undervoltage indicator

#: PINNED protocol-test reference config (defaults are Manuel's live
#: knobs and must not drift the tests): 6-min cycles, immediate
#: indicator, escalation after 2 flagged boundaries.
REF_CFG = dict(k_sched=2, n_need=1, escalation_cycles=2)


def ref_config(**overrides) -> SBXConfig:
    """Reference SBXConfig for the protocol tests (see REF_CFG)."""
    return SBXConfig(**{**REF_CFG, **overrides})


@pytest.fixture(scope="module")
def plant():
    net, _ = build_ieee39_net(scenario="wind_replace")
    zone_map, _ = fixed_zone_partition_ieee39(net)
    registry = build_corridor_registry(net, zone_map)
    pp.runpp(net)
    base_p = {}
    base_q = {}
    for key, corr in registry.items():
        p, q = [], []
        for ln in corr.lines:
            side = ("from" if int(net.line.at[ln.line_idx, "from_bus"])
                    == ln.bus_a else "to")
            p.append(float(net.res_line.at[ln.line_idx, f"p_{side}_mw"]))
            q.append(float(net.res_line.at[ln.line_idx, f"q_{side}_mvar"]))
        base_p[key] = p
        base_q[key] = q
    zone_v = {
        z: ([int(b) for b in buses],
            [float(net.res_bus.at[b, "vm_pu"]) for b in buses])
        for z, buses in zone_map.items()
    }
    return net, zone_map, registry, base_p, base_q, zone_v


class Harness:
    """Synthetic closed-world driver for the v6 scheduler.

    Feeds the BASE-CASE per-line P/Q and terminal voltages every tick
    (a stationary plant at the contract operating point), so q_meas ≈
    q_std and the steady state is in-band by construction.  Tests
    perturb ``q_offset``, terminal offsets, bounds, or planned
    support as needed."""

    def __init__(self, plant, config=None, support=None):
        net, zone_map, registry, base_p, base_q, zone_v = plant
        self.cfg = config or ref_config()
        self.registry = registry
        self.contracts = {}
        for key, corr in registry.items():
            contract = build_default_contract(corr, net, self.cfg)
            if support is not None and key in support:
                for t0, t1, dva, dvb in support[key]:
                    contract = with_planned_support(
                        contract, t0, t1, dv_a_pu=dva, dv_b_pu=dvb)
            self.contracts[key] = contract
        self.sched = SBXScheduler(self.cfg, registry, self.contracts)
        self.zone_v = zone_v
        self.base_p = base_p
        self.base_q = base_q
        self.bounds = {z: BOUND_OK for z in self.sched.area_ids}
        #: Additive per-corridor offset on the fed corridor Q [Mvar]
        #: (spread evenly over the lines) — creates deviations.
        self.q_offset = {key: 0.0 for key in registry}
        #: Additive terminal-voltage measurement offsets [pu].
        self.v_offset_a = {key: 0.0 for key in registry}
        self.v_offset_b = {key: 0.0 for key in registry}
        self.it = -1
        self.refs = self.sched.initial_references(time_s=0.0)

    def step(self):
        self.it += 1
        if self.sched.is_cycle_boundary(self.it):
            time_s = self.it * self.cfg.tso_period_s
            self.refs = self.sched.run_cycle(self.it, time_s=time_s)
        area_inputs = {}
        for z in self.sched.area_ids:
            buses, v = self.zone_v[z]
            lo, hi = self.bounds[z]
            area_inputs[z] = AreaStepInput(
                bus_indices=tuple(buses),
                v_meas_pu=tuple(v),
                v_min_pu=(lo,) * len(buses),
                v_max_pu=(hi,) * len(buses),
            )
        tie_q = {
            k: [q + self.q_offset[k] / len(qs) for q in qs]
            for k, qs in self.base_q.items()
        }
        self.sched.record_step(
            self.it, area_inputs,
            tie_p_mw=self.base_p,
            tie_q_mvar=tie_q,
            tie_v_a_pu={
                k: [
                    v + self.v_offset_a[k]
                    for v in self.sched.corridor_state(k).v_std_a_act
                ]
                for k in self.base_p
            },
            tie_v_b_pu={
                k: [
                    v + self.v_offset_b[k]
                    for v in self.sched.corridor_state(k).v_std_b_act
                ]
                for k in self.base_p
            },
        )

    def run_cycles(self, n):
        for _ in range(n * self.cfg.k_sched):
            self.step()


def test_steady_state_in_band_refs_at_v_std(plant):
    h = Harness(plant)
    h.run_cycles(4)
    for key, corr in h.registry.items():
        st = h.sched.corridor_state(key)
        contract = h.contracts[key]
        for k, ln in enumerate(corr.lines):
            assert st.refs_a[ln.bus_a] == pytest.approx(
                contract.v_std_a_pu[k])
            assert st.refs_b[ln.bus_b] == pytest.approx(
                contract.v_std_b_pu[k])
        recs = h.sched.records[key]
        assert len(recs) >= 3
        # The stationary plant sits inside the band (settled cycles):
        # deviation = q_meas − q_std ≈ golden-test-4 residual ≪ band.
        for r in recs[1:]:
            assert not r.beyond_band, (key, r.cycle, r.deviation_mvar)
            assert not r.escalation
        # Settlements were produced, but no side sagged: no support pay.
        for s in h.sched.settlements[key]:
            assert s.support_eur == 0.0
            assert abs(sum(s.payments_eur.values())) < 1e-9


def test_violation_indicator_and_escalation(plant):
    h = Harness(plant)
    h.bounds[1] = BOUND_UNDER
    h.run_cycles(5)                     # boundaries at cycles 1..4
    key = (1, 2)
    recs = h.sched.records[key]
    assert all(r.need_a for r in recs), "indicator must be set for z1"
    # escalation_cycles = 2 → flagged boundaries 3, 4 escalate.
    esc_areas = {z for _, z in h.sched.escalations}
    assert esc_areas == {1}
    esc_cycles = sorted(c for c, z in h.sched.escalations if z == 1)
    assert esc_cycles[0] == 3


def test_flow_offset_without_sag_escalates_but_is_not_paid(plant):
    """A beyond-band flow offset remains an operational escalation
    signal, but without a sag/hold pair it creates no payment."""
    h = Harness(plant)
    key = (1, 2)
    h.q_offset[key] = 4.0 * h.cfg.q_band_mvar
    h.run_cycles(5)
    settled = h.sched.records[key][1:]
    assert all(record.beyond_band for record in settled)
    assert any(record.escalation for record in settled)
    settlements = h.sched.settlements[key]
    assert all(item.support_state == "none" for item in settlements)
    assert all(item.support_eur == 0.0 for item in settlements)


def test_b_sags_a_holds_and_directional_q_pays_a(plant):
    """End-to-end: B is below schedule, A holds, and excess Q flows
    A to B, so B pays A for delivered support energy."""
    h = Harness(plant)
    key = (1, 2)
    h.v_offset_b[key] = -0.006
    h.q_offset[key] = 4.0 * h.cfg.q_band_mvar
    h.run_cycles(4)
    paid = [
        item for item in h.sched.settlements[key]
        if item.support_eur > 0.0
    ]
    assert paid
    for item in paid:
        assert item.support_state == "b_sags_a_holds"
        assert item.b_sags and item.a_holds
        assert not item.a_sags
        assert item.support_direction == "a_to_b"
        assert item.support_payer == key[1]
        assert item.support_payee == key[0]
        assert item.payments_eur[key[1]] < 0.0
        assert item.payments_eur[key[0]] > 0.0
        assert abs(sum(item.payments_eur.values())) < 1e-9


def test_planned_support_switches_schedule_and_q_std(plant):
    """v6 planned support: +2 mpu on B's terminals of corridor (1, 2)
    from minute 12 to 24 — agreed in advance.  The references and the
    implied q_std must switch at the boundaries and switch back."""
    t0, t1 = 12 * 60.0, 24 * 60.0
    key = (1, 2)
    h = Harness(plant, support={key: [(t0, t1, 0.0, +0.002)]})
    h.run_cycles(6)                     # boundaries at 6-min steps
    corr = h.registry[key]
    contract = h.contracts[key]
    base_vb = contract.v_std_schedule[0][2]

    # Reconstruct the reference trajectory from the records/state is
    # per-cycle; check the ACTIVE schedule lookups directly:
    va_in, vb_in = contract.v_std_at(t0)
    assert all(vb_in[k] == pytest.approx(base_vb[k] + 0.002, abs=1e-9)
               for k in range(corr.n_lines))
    va_out, vb_out = contract.v_std_at(t1)
    assert tuple(vb_out) == tuple(base_vb)

    # q_std of the cycles inside the window differs from outside in the
    # direction of MORE export from B's side (= less export from A):
    recs = h.sched.records[key]
    q_by_cycle = {r.cycle: r.q_std_mvar for r in recs}
    # boundary at cycle c applies schedule at t = c*6 min; the RECORD
    # stores the elapsed q_std, so cycles 3..4 (t = 12..24 min active)
    # show up in records of cycles 4..5.
    inside = q_by_cycle[4]
    outside = q_by_cycle[2]
    assert inside != pytest.approx(outside, abs=1e-6)
    # Raising B's terminals pushes corridor flow toward A: q (export
    # from A) becomes MORE NEGATIVE at the reference end.
    assert inside < outside


def test_run_cycle_off_boundary_raises(plant):
    h = Harness(plant)
    h.run_cycles(1)
    with pytest.raises(SBXError, match="boundary"):
        h.sched.run_cycle(3, time_s=0.0)   # 3 % k_sched(2) != 0
