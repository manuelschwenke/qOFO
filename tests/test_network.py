#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the Network Module
=============================

Covers:
* ``build_tuda_net`` — topology, element counts, metadata, convergence.

Author: Manuel Schwenke
"""

from __future__ import annotations

import numpy as np
import pandapower as pp
import pytest

from network.build_tuda_net import build_tuda_net, NetworkMetadata


# ═══════════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def combined():
    """Build the combined network once for all tests in this module."""
    net, meta = build_tuda_net(pv_nodes=True)
    return net, meta



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
        # Only the first 10 DN loads (HV/MV substations) are affected by
        # load_scaling; EHV loads added afterwards use fixed power values.
        dn_loads = net.load[net.load["subnet"] == "DN"]
        assert len(dn_loads) == 10
        assert np.allclose(dn_loads["p_mw"].values, 25.0)
