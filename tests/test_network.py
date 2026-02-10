#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the Network Module
=============================

Covers:
* ``build_tuda_net`` — topology, element counts, metadata, convergence.
* ``split_network``  — TN/DN separation, boundary elements, convergence.
* ``validate_split`` — round-trip accuracy of the split.

Author: Manuel Schwenke
"""

from __future__ import annotations

import numpy as np
import pandapower as pp
import pytest

from network.build_tuda_net import build_tuda_net, NetworkMetadata
from network.split_tn_dn_net import (
    CouplerPowerFlow,
    SplitResult,
    split_network,
    validate_split,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def combined():
    """Build the combined network once for all tests in this module."""
    net, meta = build_tuda_net(pv_nodes=True)
    return net, meta


@pytest.fixture(scope="module")
def split_result(combined):
    """Split the combined network once for all tests in this module."""
    net, meta = combined
    return split_network(net, meta, dn_slack_coupler_index=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  build_tuda_net TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildTudaNet:
    """Tests for the combined network builder."""

    def test_returns_converged_network(self, combined):
        net, _ = combined
        assert net.converged, "Initial power flow must converge."

    def test_returns_network_metadata(self, combined):
        _, meta = combined
        assert isinstance(meta, NetworkMetadata)

    def test_metadata_is_frozen(self, combined):
        _, meta = combined
        with pytest.raises(AttributeError):
            meta.coupler_hv_buses = [999]

    # --- Bus counts ---

    def test_ehv_bus_count(self, combined):
        net, _ = combined
        ehv = net.bus[net.bus["vn_kv"] == 380.0]
        assert len(ehv) == 7, f"Expected 7 EHV buses, got {len(ehv)}."

    def test_hv_bus_count(self, combined):
        net, _ = combined
        hv = net.bus[net.bus["vn_kv"] == 110.0]
        assert len(hv) == 10, f"Expected 10 HV buses, got {len(hv)}."

    def test_tertiary_bus_count(self, combined):
        net, meta = combined
        ter = net.bus[net.bus["vn_kv"] == 20.0]
        assert len(ter) == 3, f"Expected 3 tertiary buses, got {len(ter)}."
        assert len(meta.coupler_lv_buses) == 3

    def test_gen_terminal_bus_exists(self, combined):
        net, _ = combined
        gen_term = net.bus[net.bus["subnet"] == "GEN_TERM"]
        assert len(gen_term) >= 1, "At least one generator terminal bus expected."

    # --- Element counts ---

    def test_three_coupler_transformers(self, combined):
        net, meta = combined
        assert len(meta.coupler_trafo3w_indices) == 3
        assert len(net.trafo3w) >= 3

    def test_machine_transformers_exist(self, combined):
        _, meta = combined
        assert len(meta.machine_trafo_indices) >= 1, (
            "At least one machine transformer expected with pv_nodes=True."
        )

    def test_tn_shunt_exists(self, combined):
        _, meta = combined
        assert len(meta.tn_shunt_indices) >= 1

    def test_tertiary_shunts_exist(self, combined):
        _, meta = combined
        assert len(meta.tertiary_shunt_indices) >= 1

    def test_no_placeholder_transformers_remain(self, combined):
        net, _ = combined
        names = net.trafo["name"].astype(str).tolist()
        for n in names:
            assert "placeholder" not in n.lower(), (
                f"Placeholder transformer '{n}' was not removed."
            )

    # --- Voltage plausibility ---

    def test_ehv_voltages_plausible(self, combined):
        net, _ = combined
        ehv_buses = net.bus.index[net.bus["vn_kv"] == 380.0]
        vm = net.res_bus.loc[ehv_buses, "vm_pu"]
        assert vm.min() >= 0.90, f"EHV V_min = {vm.min():.4f} is too low."
        assert vm.max() <= 1.15, f"EHV V_max = {vm.max():.4f} is too high."

    def test_hv_voltages_plausible(self, combined):
        net, _ = combined
        hv_buses = net.bus.index[net.bus["vn_kv"] == 110.0]
        vm = net.res_bus.loc[hv_buses, "vm_pu"]
        assert vm.min() >= 0.85, f"HV V_min = {vm.min():.4f} is too low."
        assert vm.max() <= 1.15, f"HV V_max = {vm.max():.4f} is too high."

    # --- Metadata consistency ---

    def test_coupler_bus_indices_exist(self, combined):
        net, meta = combined
        for bus_list in [meta.coupler_hv_buses, meta.coupler_mv_buses, meta.coupler_lv_buses]:
            for b in bus_list:
                assert b in net.bus.index, f"Bus {b} not in network."

    def test_coupler_trafo3w_indices_exist(self, combined):
        net, meta = combined
        for tidx in meta.coupler_trafo3w_indices:
            assert tidx in net.trafo3w.index, f"Trafo3w {tidx} not in network."


# ═══════════════════════════════════════════════════════════════════════════════
#  split_network TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplitNetwork:
    """Tests for the TN/DN network split."""

    def test_returns_split_result(self, split_result):
        assert isinstance(split_result, SplitResult)

    def test_tn_net_converged(self, split_result):
        assert split_result.tn_net.converged, "TN power flow must converge."

    def test_dn_net_converged(self, split_result):
        assert split_result.dn_net.converged, "DN power flow must converge."

    # --- TN network structure ---

    def test_tn_has_no_dn_buses(self, split_result):
        tn = split_result.tn_net
        dn_buses = tn.bus[tn.bus["subnet"].astype(str) == "DN"]
        assert dn_buses.empty, "TN network must not contain DN buses."

    def test_tn_has_no_trafo3w(self, split_result):
        tn = split_result.tn_net
        assert tn.trafo3w.empty, "TN network must not contain 3W transformers."

    def test_tn_has_ext_grid(self, split_result):
        tn = split_result.tn_net
        assert not tn.ext_grid.empty, "TN must retain its external grid."

    def test_tn_boundary_sgen_count(self, split_result):
        assert len(split_result.tn_boundary_sgen_indices) == 3

    # --- DN network structure ---

    def test_dn_has_no_tn_lines(self, split_result):
        dn = split_result.dn_net
        tn_lines = [
            i for i in dn.line.index
            if str(dn.line.at[i, "name"]).startswith("TN|")
        ]
        assert len(tn_lines) == 0, "DN network must not contain TN lines."

    def test_dn_has_trafo3w(self, split_result):
        dn = split_result.dn_net
        assert not dn.trafo3w.empty, "DN network must retain 3W transformers."

    def test_dn_has_slack(self, split_result):
        dn = split_result.dn_net
        assert not dn.ext_grid.empty, "DN must have a slack bus."

    def test_dn_boundary_sgen_count(self, split_result):
        assert len(split_result.dn_boundary_sgen_indices) == 3

    # --- Coupler flows ---

    def test_coupler_flows_count(self, split_result):
        assert len(split_result.coupler_flows) == 3

    def test_coupler_flows_are_frozen(self, split_result):
        cf = split_result.coupler_flows[0]
        assert isinstance(cf, CouplerPowerFlow)
        with pytest.raises(AttributeError):
            cf.p_hv_mw = 999.0

    # --- Invalid arguments ---

    def test_invalid_slack_index_raises(self, combined):
        net, meta = combined
        with pytest.raises(ValueError, match="out of range"):
            split_network(net, meta, dn_slack_coupler_index=99)


# ═══════════════════════════════════════════════════════════════════════════════
#  validate_split TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateSplit:
    """Tests for the split validation function."""

    def test_validation_passes(self, combined, split_result):
        net, meta = combined
        ok = validate_split(net, meta, split_result)
        assert ok, "Split validation must pass with default tolerances."

    def test_tn_voltages_match_combined(self, combined, split_result):
        """Spot-check: TN bus voltages must be very close to the combined network."""
        net, _ = combined
        tn = split_result.tn_net
        common = sorted(set(tn.bus.index) & set(net.res_bus.index))
        vm_combined = net.res_bus.loc[common, "vm_pu"].values
        vm_tn = tn.res_bus.loc[common, "vm_pu"].values
        np.testing.assert_allclose(vm_tn, vm_combined, atol=5e-4)

    def test_dn_slack_voltage_matches(self, combined, split_result):
        """DN slack bus voltage must equal the combined operating point."""
        net, _ = combined
        dn = split_result.dn_net
        slack_cf = split_result.coupler_flows[split_result.dn_slack_coupler_index]
        vm_dn = float(dn.res_bus.at[slack_cf.hv_bus, "vm_pu"])
        vm_ref = float(net.res_bus.at[slack_cf.hv_bus, "vm_pu"])
        np.testing.assert_allclose(vm_dn, vm_ref, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
#  build_tuda_net WITH DIFFERENT OPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildOptions:
    """Test build_tuda_net with non-default parameters."""

    def test_no_pv_nodes(self):
        net, meta = build_tuda_net(pv_nodes=False)
        assert net.converged
        assert net.gen.empty, "No generators expected when pv_nodes=False."
        assert len(meta.machine_trafo_indices) == 0

    def test_custom_load_scaling(self):
        net, _ = build_tuda_net(load_scaling=0.5)
        assert net.converged
        # All loads should be approximately 25 MW (50 * 0.5)
        assert np.allclose(net.load["p_mw"].values, 25.0)
