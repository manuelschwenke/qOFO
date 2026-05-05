"""
Tests for the Q_cor write paths in
``experiments.helpers.plant_io.apply_zone_tso_controls`` and
``apply_dso_controls`` (refactor_v2 commit 5).

When ``use_q_cor_actuator=True``, the OFO output's DER block is
written into ``net.sgen.q_cor_mvar`` instead of ``q_mvar``
(direct-Q legacy) or ``vm_pu_ref`` (legacy Q-shim).
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


def _build_2sgen_net() -> tuple[pp.pandapowerNet, list[int]]:
    """Two-sgen network with q_mode columns prepopulated."""
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
    net.sgen["qv_deadband_pu"] = 0.0
    net.sgen["cosphi"] = 1.0
    net.sgen["cosphi_sign"] = -1
    net.sgen["q_cor_mvar"] = 0.0
    return net, [int(s0), int(s1)]


# ---------------------------------------------------------------------------
#  TSO apply step
# ---------------------------------------------------------------------------

class TestApplyTsoQCor:
    def test_legacy_writes_q_mvar(self):
        net, sgens = _build_2sgen_net()
        zone = _StubZoneDef(tso_der_indices=sgens)
        out = _StubOut(u_new=np.array([3.5, -1.0]))
        # Legacy: use_q_cor_actuator default False
        apply_zone_tso_controls(net, zone, out)
        assert net.sgen.at[sgens[0], "q_mvar"] == pytest.approx(3.5)
        assert net.sgen.at[sgens[1], "q_mvar"] == pytest.approx(-1.0)
        # q_cor_mvar untouched
        assert net.sgen.at[sgens[0], "q_cor_mvar"] == pytest.approx(0.0)
        assert net.sgen.at[sgens[1], "q_cor_mvar"] == pytest.approx(0.0)

    def test_q_cor_mode_writes_q_cor_mvar(self):
        net, sgens = _build_2sgen_net()
        zone = _StubZoneDef(tso_der_indices=sgens)
        out = _StubOut(u_new=np.array([3.5, -1.0]))
        apply_zone_tso_controls(net, zone, out, use_q_cor_actuator=True)
        # q_cor_mvar populated
        assert net.sgen.at[sgens[0], "q_cor_mvar"] == pytest.approx(3.5)
        assert net.sgen.at[sgens[1], "q_cor_mvar"] == pytest.approx(-1.0)
        # q_mvar untouched (still 0 from init)
        assert net.sgen.at[sgens[0], "q_mvar"] == pytest.approx(0.0)
        assert net.sgen.at[sgens[1], "q_mvar"] == pytest.approx(0.0)

    def test_q_cor_creates_column_if_absent(self):
        """Defensive: even on a network where tag_der_q_modes was not
        called, the apply step creates ``q_cor_mvar`` as needed."""
        net, sgens = _build_2sgen_net()
        # Strip the column to simulate an unmigrated network
        net.sgen = net.sgen.drop(columns=["q_cor_mvar"])
        zone = _StubZoneDef(tso_der_indices=sgens)
        out = _StubOut(u_new=np.array([2.0, -0.5]))
        apply_zone_tso_controls(net, zone, out, use_q_cor_actuator=True)
        assert "q_cor_mvar" in net.sgen.columns
        assert net.sgen.at[sgens[0], "q_cor_mvar"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
#  DSO apply step
# ---------------------------------------------------------------------------

class TestApplyDsoQCor:
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

    def test_legacy_direct_q_writes_q_mvar(self):
        net, sgens = _build_2sgen_net()
        cfg = self._make_cfg(sgens, use_qv_local_loop=False,
                             use_q_cor_actuator=False)
        out = _StubOut(u_new=np.array([3.0, -0.5]))
        apply_dso_controls(net, cfg, out)
        assert net.sgen.at[sgens[0], "q_mvar"] == pytest.approx(3.0)
        assert net.sgen.at[sgens[1], "q_mvar"] == pytest.approx(-0.5)
        # q_cor_mvar untouched
        assert net.sgen.at[sgens[0], "q_cor_mvar"] == pytest.approx(0.0)

    def test_q_cor_mode_writes_q_cor_mvar(self):
        net, sgens = _build_2sgen_net()
        cfg = self._make_cfg(sgens, use_qv_local_loop=False,
                             use_q_cor_actuator=True)
        out = _StubOut(u_new=np.array([4.0, -2.0]))
        apply_dso_controls(net, cfg, out)
        # q_cor_mvar populated
        assert net.sgen.at[sgens[0], "q_cor_mvar"] == pytest.approx(4.0)
        assert net.sgen.at[sgens[1], "q_cor_mvar"] == pytest.approx(-2.0)
        # q_mvar and vm_pu_ref untouched
        assert net.sgen.at[sgens[0], "q_mvar"] == pytest.approx(0.0)

    def test_q_cor_takes_precedence_over_qv_local_loop(self):
        """When BOTH use_qv_local_loop=True and use_q_cor_actuator=True
        are set on the same cfg, the Q_cor branch wins (it is checked
        first in the chained if).  This is the migration story: a
        runner that flips on q_cor without first turning off qv_local_loop
        should still write q_cor_mvar."""
        net, sgens = _build_2sgen_net()
        cfg = self._make_cfg(sgens, use_qv_local_loop=True,
                             qv_apply_mode="q_shim",
                             use_q_cor_actuator=True)
        out = _StubOut(u_new=np.array([1.5, 0.5]))
        apply_dso_controls(net, cfg, out)
        assert net.sgen.at[sgens[0], "q_cor_mvar"] == pytest.approx(1.5)
        # vm_pu_ref must NOT have been written by the legacy Q-shim branch.
        if "vm_pu_ref" in net.sgen.columns:
            # Default vm_pu_ref column gets created by apply_dso_controls
            # when use_qv_local_loop=True; it should equal the default 1.03.
            assert net.sgen.at[sgens[0], "vm_pu_ref"] == pytest.approx(1.03)


# ---------------------------------------------------------------------------
#  TSO config flag validation (mirrors the DSO flag validation)
# ---------------------------------------------------------------------------

class TestTsoConfigFlag:
    def test_default_is_legacy(self):
        from controller.tso_controller import TSOControllerConfig
        cfg = TSOControllerConfig(
            der_indices=[],
            pcc_trafo_indices=[],
            pcc_dso_controller_ids=[],
            gen_indices=[],
            voltage_bus_indices=[],
            current_line_indices=[],
            oltc_trafo_indices=[],
            shunt_bus_indices=[],
            shunt_q_steps_mvar=[],
        )
        assert cfg.use_q_cor_actuator is False

    def test_q_cor_mode_accepted(self):
        from controller.tso_controller import TSOControllerConfig
        cfg = TSOControllerConfig(
            der_indices=[],
            pcc_trafo_indices=[],
            pcc_dso_controller_ids=[],
            gen_indices=[],
            voltage_bus_indices=[],
            current_line_indices=[],
            oltc_trafo_indices=[],
            shunt_bus_indices=[],
            shunt_q_steps_mvar=[],
            use_q_cor_actuator=True,
        )
        assert cfg.use_q_cor_actuator is True
        assert hasattr(cfg, "qv_slope_pu")
        assert cfg.qv_slope_pu == pytest.approx(0.07)
