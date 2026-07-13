"""
SBX Phase 5 — unit tests for ``sbx_h.scheduler`` (protocol logic level).

Drives the six-step cycle on the REAL IEEE 39 corridor registry and
contracts with synthetic measurements and synthetic per-area capability
data, so every protocol transition is deterministic and analytically
checkable: steady state, request → unilateral deal → acting-side refs,
same-cycle deals on two corridors of one area (v2.2 item 5, logic
level), scarcity, unwind with paid-first ordering and dwell, and the
Step-4 invariant.  The plant-in-the-loop behaviour is covered by the
closed-loop smoke test (``tests/sbx_h/smoke_sbx_closed_loop.py``).
"""
from __future__ import annotations

import numpy as np
import pandapower as pp
import pytest

from network.ieee39.build import build_ieee39_net
from network.zone_partition import fixed_zone_partition_ieee39
from sbx_h.config import SBXConfig
from sbx_h.contract import build_default_contract
from sbx_h.corridor import build_corridor_registry
from sbx_h.fail import SBXError
from sbx_h.matching import KIND_SCARCITY, KIND_UNILATERAL
from sbx_h.scheduler import AreaCycleData, AreaStepInput, SBXScheduler

BOUND_OK = (0.90, 1.10)     # base-case voltages are comfortably inside
BOUND_UNDER = (1.04, 1.10)  # forces undervoltage need (import)
N_U = 3                     # synthetic actuators per area

#: PINNED protocol-test reference timing AND semantics.  The SBXConfig
#: DEFAULTS are Manuel's live experimental knobs (they have changed
#: three times) and moved to the v5 mechanism on 2026-07-10; the
#: protocol tests here encode the v4 reference behaviour and must not
#: drift.  6-min cycles, one 12-Mvar quantum per cycle, immediate
#: need/release; v4 request trigger (need flag alone), single-quantum
#: requests, magnitude-classifier delivery gate, unconditional tier-2
#: billing.  The v5 additions are covered by test_v5_redesign.py.
REF_CFG = dict(k_sched=2, dq_quant_rate_mvar_per_15min=30.0,
               n_need=1, m_release=1,
               require_exhaustion_to_request=False,
               request_sizing="single",
               delivery_check="magnitude",
               tier2_requires_delivery=False)


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
    base_v = {}
    for key, corr in registry.items():
        p = []
        for ln in corr.lines:
            side = ("from" if int(net.line.at[ln.line_idx, "from_bus"])
                    == ln.bus_a else "to")
            p.append(float(net.res_line.at[ln.line_idx, f"p_{side}_mw"]))
        base_p[key] = p
    zone_v = {
        z: ([int(b) for b in buses],
            [float(net.res_bus.at[b, "vm_pu"]) for b in buses])
        for z, buses in zone_map.items()
    }
    return net, zone_map, registry, base_p, zone_v


class Harness:
    """Synthetic closed-world driver for the scheduler protocol."""

    def __init__(self, plant, config=None):
        net, zone_map, registry, base_p, zone_v = plant
        self.cfg = config or ref_config()
        self.registry = registry
        self.contracts = {
            key: build_default_contract(corr, net, self.cfg)
            for key, corr in registry.items()
        }
        self.sched = SBXScheduler(self.cfg, registry, self.contracts)
        self.zone_v = zone_v
        self.base_p = base_p
        self.bounds = {z: BOUND_OK for z in self.sched.area_ids}
        #: v5 tests: when True, the fed terminal-voltage measurements
        #: FOLLOW the scheduler's current references (a perfectly
        #: tracking plant); default False = constant v_std feeds (the
        #: legacy behaviour — a plant that never moves).
        self.track_refs = False
        self.it = -1
        self.refs = self.sched.initial_references()

    def cycle_data(self):
        data = {}
        for z in self.sched.area_ids:
            buses, v = self.zone_v[z]
            lo, hi = self.bounds[z]
            n_v = len(buses)
            term_buses = sorted({
                (ln.bus_a if z == corr.area_a else ln.bus_b)
                for key in self.sched.corridors_of_area[z]
                for corr in (self.registry[key],)
                for ln in corr.lines
            })
            # Synthetic local model: each terminal bus responds to one
            # actuator with 1e-3 pu per unit; monitored buses respond
            # weakly, so the voltage box never binds in these tests.
            h_rows = {bus: np.zeros(N_U) for bus in term_buses}
            for k, bus in enumerate(term_buses):
                h_rows[bus][k % N_U] = 1.0e-3
            data[z] = AreaCycleData(
                u_now=np.zeros(N_U),
                u_min=np.full(N_U, -500.0),
                u_max=np.full(N_U, +500.0),
                v_bus_indices=tuple(buses),
                v_meas_pu=np.asarray(v),
                v_min_pu=np.full(n_v, 0.5),   # capability box kept slack
                v_max_pu=np.full(n_v, 1.5),
                h_loc=np.zeros((n_v, N_U)),
                terminal_h_rows=h_rows,
                dv_dq_import_by_corridor={
                    key: +2.0e-4
                    for key in self.sched.corridors_of_area[z]
                },
            )
        return data

    def step(self):
        self.it += 1
        if self.sched.is_cycle_boundary(self.it):
            self.refs = self.sched.run_cycle(self.it, self.cycle_data())
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
        if self.track_refs:
            tie_v_a = {k: [self.sched.corridor_state(k).refs_a[ln.bus_a]
                           for ln in self.registry[k].lines]
                       for k in self.base_p}
            tie_v_b = {k: [self.sched.corridor_state(k).refs_b[ln.bus_b]
                           for ln in self.registry[k].lines]
                       for k in self.base_p}
        else:
            tie_v_a = {k: list(self.contracts[k].v_std_a_pu)
                       for k in self.base_p}
            tie_v_b = {k: list(self.contracts[k].v_std_b_pu)
                       for k in self.base_p}
        self.sched.record_step(
            self.it, area_inputs,
            tie_p_mw=self.base_p,
            tie_q_mvar={k: [0.0] * len(p) for k, p in self.base_p.items()},
            tie_v_a_pu=tie_v_a,
            tie_v_b_pu=tie_v_b,
        )

    def run_cycles(self, n):
        for _ in range(n * self.cfg.k_sched):
            self.step()


def test_steady_state_no_deals(plant):
    h = Harness(plant)
    h.run_cycles(3)
    for key, corr in h.registry.items():
        st = h.sched.corridor_state(key)
        assert st.surplus_mvar == 0.0
        recs = h.sched.records[key]
        assert len(recs) >= 2
        assert all(r.deal.dq_deal_mvar == 0.0 for r in recs)
        # q_std tracks the persistence forecast, refs stay at v_std.
        contract = h.contracts[key]
        for k, ln in enumerate(corr.lines):
            assert st.refs_a[ln.bus_a] == pytest.approx(
                contract.v_std_a_pu[k])
            assert st.refs_b[ln.bus_b] == pytest.approx(
                contract.v_std_b_pu[k])


def test_stressed_area_deals_on_both_corridors_same_cycle(plant):
    """v2.2 item 5 (logic level): area 1 requests on (1,2) AND (1,3)."""
    h = Harness(plant)
    h.bounds[1] = BOUND_UNDER
    h.run_cycles(2)   # need persists >= n_need steps within cycle 1
    for key in ((1, 2), (1, 3)):
        recs = h.sched.records[key]
        executed = [r for r in recs if r.deal.dq_deal_mvar != 0.0]
        assert executed, f"no deal on corridor {key}"
        first = executed[0]
        assert first.deal.kind == KIND_UNILATERAL and first.deal.paid
        assert first.deal.requester == 1
        # Import need of area 1: end A -> more negative corridor flow.
        assert first.deal.dq_deal_mvar == pytest.approx(
            -h.contracts[key].dq_quant_mvar)
        st = h.sched.corridor_state(key)
        assert st.surplus_mvar < 0.0
        assert st.surplus_paid_mvar == pytest.approx(st.surplus_mvar)
    # Corridor (2,3) stays undisturbed.
    assert h.sched.corridor_state((2, 3)).surplus_mvar == 0.0
    # Both deals landed in the SAME cycle (parallel corridors, v2.2).
    c12 = [r.cycle for r in h.sched.records[(1, 2)]
           if r.deal.dq_deal_mvar != 0.0][0]
    c13 = [r.cycle for r in h.sched.records[(1, 3)]
           if r.deal.dq_deal_mvar != 0.0][0]
    assert c12 == c13


def test_acting_side_is_exporting_end_and_invariant_holds(plant):
    h = Harness(plant)
    h.bounds[1] = BOUND_UNDER
    h.run_cycles(2)
    key = (1, 2)
    corr = h.registry[key]
    contract = h.contracts[key]
    rec = [r for r in h.sched.records[key] if r.deal.dq_deal_mvar != 0.0][0]
    # surplus < 0 -> B exports the surplus -> acting side = area 2.
    assert rec.acting_area == 2
    assert rec.dv_pu != 0.0
    st = h.sched.corridor_state(key)
    for k, ln in enumerate(corr.lines):
        assert st.refs_a[ln.bus_a] == pytest.approx(contract.v_std_a_pu[k])
        assert st.refs_b[ln.bus_b] == pytest.approx(
            contract.v_std_b_pu[k] + rec.dv_pu)


def test_opposite_needs_scarcity(plant):
    h = Harness(plant)
    h.bounds[1] = BOUND_UNDER          # area 1 wants import
    h.bounds[2] = BOUND_UNDER          # area 2 wants import too
    h.run_cycles(2)
    # Corridor (1,2): requests -q (A) and +q (B) -> scarcity, no deal.
    recs = h.sched.records[(1, 2)]
    assert any(r.deal.kind == KIND_SCARCITY for r in recs)
    assert h.sched.corridor_state((1, 2)).surplus_mvar == 0.0
    assert len(h.sched.scarcity_events) >= 1


def test_unwind_paid_first_with_dwell_and_return_to_v_std(plant):
    # delivery_gate off: the harness feeds tie_q = 0 (non-delivery by
    # construction); this test exercises the v2/v3 unwind mechanics.
    cfg = ref_config(delivery_gate=False)
    h = Harness(plant, cfg)
    h.bounds[1] = BOUND_UNDER
    h.run_cycles(3)                     # builds paid surplus
    key = (1, 2)
    st = h.sched.corridor_state(key)
    surplus_peak = abs(st.surplus_mvar)
    assert surplus_peak > 0.0
    # Remove the stress; flags clear -> dwell m_release cycles -> unwind
    # one quantum per cycle until zero.
    h.bounds[1] = BOUND_OK
    quantum = h.contracts[key].dq_quant_mvar
    max_cycles = int(np.ceil(surplus_peak / quantum)) + cfg.m_release + 2
    h.run_cycles(max_cycles)
    st = h.sched.corridor_state(key)
    assert st.surplus_mvar == 0.0
    assert st.surplus_paid_mvar == 0.0 and st.surplus_unpaid_mvar == 0.0
    # Refs back at v_std on both ends (invariant: no deviating end).
    contract = h.contracts[key]
    corr = h.registry[key]
    for k, ln in enumerate(corr.lines):
        assert st.refs_a[ln.bus_a] == pytest.approx(contract.v_std_a_pu[k])
        assert st.refs_b[ln.bus_b] == pytest.approx(contract.v_std_b_pu[k])
    # Unwind respected the dwell: no unwind before m_release clear cycles.
    recs = h.sched.records[key]
    first_clear = next(i for i, r in enumerate(recs)
                       if not r.need_a and r.surplus_mvar != 0.0
                       and r.deal.dq_deal_mvar == 0.0)
    unwind_cycles = [i for i, r in enumerate(recs) if r.unwound_mvar != 0.0]
    assert unwind_cycles, "no unwind recorded"
    assert unwind_cycles[0] - first_clear >= cfg.m_release - 1


def test_contract_cap_blocks_runaway_surplus(plant):
    # delivery_gate off: with the harness's tie_q = 0 feed the v4 gate
    # would (correctly) stop the ratchet long before the cap — this
    # test exercises the contract-cap rejection itself.
    cfg = ref_config(delivery_gate=False)
    h = Harness(plant, cfg)
    h.bounds[1] = BOUND_UNDER
    # Need never clears: surplus must saturate at the contract cap.
    n_cycles = int(cfg.dq_contract_max_mvar
                   / h.contracts[(1, 2)].dq_quant_mvar) + 4
    h.run_cycles(n_cycles + 1)
    st = h.sched.corridor_state((1, 2))
    assert abs(st.surplus_mvar) <= cfg.dq_contract_max_mvar + 1e-9
    from sbx_h.matching import REASON_CONTRACT_CAP
    assert any(r.deal.reason == REASON_CONTRACT_CAP
               for r in h.sched.records[(1, 2)])


def test_run_cycle_off_boundary_raises(plant):
    h = Harness(plant)
    with pytest.raises(SBXError, match="off the cycle boundary"):
        h.sched.run_cycle(3, h.cycle_data())
