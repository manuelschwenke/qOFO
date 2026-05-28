"""
Tests for the w-shift write paths in
``experiments.helpers.plant_io.apply_zone_tso_controls`` and
``apply_dso_controls``.

Under the w-shift / V_ref-reanchored DER actuator, the OFO output's
DER block is

  1. reanchored: ``net.sgen.qv_vref_anchor_pu`` is set to the most
     recently measured bus voltage from ``net.res_bus.vm_pu``;
  2. written:    ``net.sgen.q_set_mvar`` receives the OFO command.

The plant-side ``QVLocalLoop`` then reads both columns on the next
``pp.runpp(run_control=True)`` iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandapower as pp
import pytest

from controller.dso_controller import DSOControllerConfig
from experiments.helpers.plant_io import (
    apply_dso_controls,
    apply_zone_tso_controls,
)


# ---------------------------------------------------------------------------
#  Tiny helpers — minimal surrogates for ZoneDefinition and ControllerOutput
# ---------------------------------------------------------------------------

@dataclass
class _StubZoneDef:
    """Minimal duck-type for ZoneDefinition used by apply_zone_tso_controls."""
    tso_der_indices: List[int] = field(default_factory=list)
    pcc_trafo_indices: List[int] = field(default_factory=list)
    gen_indices: List[int] = field(default_factory=list)
    gridforming_gen_indices: List[int] = field(default_factory=list)
    oltc_trafo_indices: List[int] = field(default_factory=list)
    shunt_bus_indices: List[int] = field(default_factory=list)


@dataclass
class _StubOut:
    u_new: np.ndarray


def _build_2sgen_net() -> tuple[pp.pandapowerNet, list[int], int]:
    """Two-sgen network with q_mode columns prepopulated.  Returns the net,
    the sgen indices, and the load-bus index."""
    net = pp.create_empty_network()
    b_slack = pp.create_bus(net, vn_kv=110.0)
    b_load = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, bus=b_slack, vm_pu=1.0)
    pp.create_line_from_parameters(
        net, from_bus=b_slack, to_bus=b_load,
        length_km=10.0, r_ohm_per_km=0.1, x_ohm_per_km=0.3,
        c_nf_per_km=10.0, max_i_ka=1.0,
    )
    pp.create_load(net, bus=b_load, p_mw=20.0, q_mvar=10.0)
    s0 = pp.create_sgen(net, bus=b_load, p_mw=10.0, q_mvar=0.0,
                       sn_mva=100.0, name="der_a")
    s1 = pp.create_sgen(net, bus=b_load, p_mw=5.0, q_mvar=0.0,
                       sn_mva=50.0, name="der_b")
    net.sgen["op_diagram"] = "STATCOM"
    net.sgen["q_mode"] = "qv"
    net.sgen["qv_slope_pu"] = 0.07
    net.sgen["qv_vref_pu"] = 1.00
    net.sgen["qv_deadband_pu"] = 0.005
    net.sgen["cosphi"] = 1.0
    net.sgen["cosphi_sign"] = -1
    net.sgen["q_set_mvar"] = 0.0
    net.sgen["qv_vref_anchor_pu"] = float("nan")
    return net, [int(s0), int(s1)], int(b_load)


# ---------------------------------------------------------------------------
#  TSO apply step
# ---------------------------------------------------------------------------

class TestApplyZoneTsoControlsWshift:
    def test_writes_q_set_mvar(self):
        net, sgens, _ = _build_2sgen_net()
        # Stage a PF so net.res_bus has a vm_pu the apply step can read.
        pp.runpp(net, run_control=False)
        zone = _StubZoneDef(tso_der_indices=sgens)
        out = _StubOut(u_new=np.array([3.5, -1.0]))
        apply_zone_tso_controls(net, zone, out)
        assert net.sgen.at[sgens[0], "q_set_mvar"] == pytest.approx(3.5)
        assert net.sgen.at[sgens[1], "q_set_mvar"] == pytest.approx(-1.0)
        # q_mvar untouched (still 0 from init); the plant-side QVLocalLoop
        # is responsible for writing the realised Q on the next PF.
        assert net.sgen.at[sgens[0], "q_mvar"] == pytest.approx(0.0)

    def test_reanchors_vref_to_measured_bus_voltage(self):
        net, sgens, bus = _build_2sgen_net()
        pp.runpp(net, run_control=False)
        v_meas = float(net.res_bus.at[bus, "vm_pu"])
        zone = _StubZoneDef(tso_der_indices=sgens)
        out = _StubOut(u_new=np.array([0.0, 0.0]))
        apply_zone_tso_controls(net, zone, out)
        for s in sgens:
            assert net.sgen.at[s, "qv_vref_anchor_pu"] == pytest.approx(
                v_meas, abs=1e-12,
            )

    def test_creates_columns_if_absent(self):
        """Defensive: even on a network where ``tag_der_q_modes`` was
        not called, the apply step creates the columns as needed."""
        net, sgens, _ = _build_2sgen_net()
        net.sgen = net.sgen.drop(columns=["q_set_mvar", "qv_vref_anchor_pu"])
        pp.runpp(net, run_control=False)
        zone = _StubZoneDef(tso_der_indices=sgens)
        out = _StubOut(u_new=np.array([2.0, -0.5]))
        apply_zone_tso_controls(net, zone, out)
        assert "q_set_mvar" in net.sgen.columns
        assert "qv_vref_anchor_pu" in net.sgen.columns
        assert net.sgen.at[sgens[0], "q_set_mvar"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
#  DSO apply step
# ---------------------------------------------------------------------------

class TestApplyDsoControlsWshift:
    def _make_cfg(self, sgens, **kw) -> DSOControllerConfig:
        defaults = dict(
            interface_trafo_indices=[],
            der_indices=sgens,
            voltage_bus_indices=[],
            current_line_indices=[],
            oltc_trafo_indices=[],
            shunt_bus_indices=[],
            shunt_q_steps_mvar=[],
        )
        defaults.update(kw)
        return DSOControllerConfig(**defaults)

    def test_writes_q_set_mvar(self):
        net, sgens, _ = _build_2sgen_net()
        pp.runpp(net, run_control=False)
        cfg = self._make_cfg(sgens)
        out = _StubOut(u_new=np.array([4.0, -2.0]))
        apply_dso_controls(net, cfg, out)
        assert net.sgen.at[sgens[0], "q_set_mvar"] == pytest.approx(4.0)
        assert net.sgen.at[sgens[1], "q_set_mvar"] == pytest.approx(-2.0)

    def test_reanchors_vref_to_measured_bus_voltage(self):
        net, sgens, bus = _build_2sgen_net()
        pp.runpp(net, run_control=False)
        v_meas = float(net.res_bus.at[bus, "vm_pu"])
        cfg = self._make_cfg(sgens)
        out = _StubOut(u_new=np.array([0.0, 0.0]))
        apply_dso_controls(net, cfg, out)
        for s in sgens:
            assert net.sgen.at[s, "qv_vref_anchor_pu"] == pytest.approx(
                v_meas, abs=1e-12,
            )

    def test_creates_columns_if_absent(self):
        net, sgens, _ = _build_2sgen_net()
        net.sgen = net.sgen.drop(columns=["q_set_mvar", "qv_vref_anchor_pu"])
        pp.runpp(net, run_control=False)
        cfg = self._make_cfg(sgens)
        out = _StubOut(u_new=np.array([1.5, 0.5]))
        apply_dso_controls(net, cfg, out)
        assert "q_set_mvar" in net.sgen.columns
        assert "qv_vref_anchor_pu" in net.sgen.columns
        assert net.sgen.at[sgens[0], "q_set_mvar"] == pytest.approx(1.5)
