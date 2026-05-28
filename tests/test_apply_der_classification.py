"""
Integration tests for ``apply_der_classification`` (sgen → gen promotion).

These tests build a real IEEE 39-bus network with the wind_replace scenario
and HV sub-networks attached, then verify that the classification step
correctly promotes grid-forming sgens into gens, prunes the TSO/DSO DER
registries, populates the new grid-forming registries, and produces a
network that still converges in PF.
"""

from __future__ import annotations

import pytest

import pandapower as pp

from core.der_classification import DERClassification, DERMode
from network.ieee39.build import (
    apply_der_classification,
    build_ieee39_net,
)
from network.ieee39.hv_networks import add_hv_networks


@pytest.fixture()
def base_net_meta():
    """Build a fresh wind_replace IEEE39 + HV sub-networks per test.

    Function-scoped because ``apply_der_classification`` mutates the
    network (drops sgens, creates gens). Tests would interfere otherwise.
    """
    net, meta = build_ieee39_net(scenario="wind_replace", verbose=False)
    meta = add_hv_networks(net, meta, verbose=False)
    return net, meta


class TestDefaultClassification:
    def test_promotes_all_tso_der_indices(self, base_net_meta):
        net, meta = base_net_meta
        # Snapshot before promotion
        n_tso_before = len(meta.tso_der_indices)
        n_sgen_before = len(net.sgen)
        n_gen_before = len(net.gen)
        original_tso_ids = set(meta.tso_der_indices)
        original_dso_ids = set(meta.dso_der_indices)

        # Default classification promotes all TSO DER, leaves DSO alone.
        meta2 = apply_der_classification(net, meta, overrides=None)

        # TSO DER list pruned to empty (every entry was grid-forming).
        assert len(meta2.tso_der_indices) == 0
        # DSO list unchanged
        assert set(meta2.dso_der_indices) == original_dso_ids

        # New grid-forming gen registry has same length as the original
        # TSO DER list.
        assert len(meta2.tso_grid_forming_gen_indices) == n_tso_before
        assert len(meta2.tso_grid_forming_gen_buses) == n_tso_before
        # No DSO promotions under default classification.
        assert len(meta2.dso_grid_forming_gen_indices) == 0

        # net.sgen lost exactly the promoted units, net.gen gained them.
        assert len(net.sgen) == n_sgen_before - n_tso_before
        assert len(net.gen) == n_gen_before + n_tso_before

        # Classification carries the promotion records for the original IDs.
        cls = meta2.der_classification
        assert cls is not None
        for sgen_id in original_tso_ids:
            assert cls.is_grid_forming(sgen_id)
            gen_idx = cls.gen_idx(sgen_id)
            assert gen_idx in net.gen.index

    def test_promoted_gens_have_correct_attributes(self, base_net_meta):
        net, meta = base_net_meta
        meta2 = apply_der_classification(net, meta, overrides=None)
        cls = meta2.der_classification
        for sgen_id in meta.tso_der_indices:
            gen_idx = cls.gen_idx(sgen_id)
            row = net.gen.loc[gen_idx]
            # Voltage setpoint defaults to 1.03 pu
            assert row["vm_pu"] == pytest.approx(1.03)
            # STATCOM op_diagram → ±S_n Q-circle
            assert row["max_q_mvar"] == pytest.approx(row["sn_mva"])
            assert row["min_q_mvar"] == pytest.approx(-row["sn_mva"])
            # Slack weight zero (does not absorb P-imbalance)
            assert row["slack_weight"] == pytest.approx(0.0)
            # Controllable (kept for OPF use)
            assert bool(row["controllable"]) is True
            # Subnet tag carries forward
            if "subnet" in net.gen.columns:
                assert row["subnet"] == "TN"

    def test_pf_converges_with_enforce_q_lims(self, base_net_meta):
        net, meta = base_net_meta
        # Promotion runs PF internally; subsequent PF should also converge.
        apply_der_classification(net, meta, overrides=None)
        pp.runpp(
            net, run_control=False, calculate_voltage_angles=True,
            enforce_q_lims=True, max_iteration=100,
        )
        assert net.converged


class TestOverrides:
    def test_override_flips_tso_der_to_grid_following(self, base_net_meta):
        net, meta = base_net_meta
        # Pick the first TSO DER and flip it to grid-following
        keep_as_sgen = int(meta.tso_der_indices[0])
        overrides = {keep_as_sgen: "grid_following"}

        meta2 = apply_der_classification(net, meta, overrides=overrides)

        # The flipped DER stays in net.sgen and in tso_der_indices
        assert keep_as_sgen in meta2.tso_der_indices
        assert keep_as_sgen in net.sgen.index
        # All other TSO DERs were still promoted
        for s in meta.tso_der_indices:
            if s == keep_as_sgen:
                continue
            assert s not in meta2.tso_der_indices
            assert s not in net.sgen.index

        cls = meta2.der_classification
        assert cls.is_grid_following(keep_as_sgen)

    def test_override_flips_dso_der_to_grid_forming(self, base_net_meta):
        net, meta = base_net_meta
        # Pick the first DSO DER and flip it to grid-forming
        promote_dso = int(meta.dso_der_indices[0])
        overrides = {promote_dso: "grid_forming"}

        meta2 = apply_der_classification(net, meta, overrides=overrides)

        # That DSO unit was promoted out of net.sgen into net.gen
        assert promote_dso not in net.sgen.index
        assert promote_dso not in meta2.dso_der_indices
        # Recorded in dso_grid_forming_gen_indices
        cls = meta2.der_classification
        gen_idx = cls.gen_idx(promote_dso)
        assert gen_idx in meta2.dso_grid_forming_gen_indices
        assert gen_idx in net.gen.index


class TestDERClassificationOnMeta:
    def test_meta_carries_classification(self, base_net_meta):
        net, meta = base_net_meta
        meta2 = apply_der_classification(net, meta, overrides=None)
        assert isinstance(meta2.der_classification, DERClassification)
        # Every original TSO DER is in the classification map
        for s in meta.tso_der_indices:
            assert meta2.der_classification.mode[int(s)] == DERMode.GRID_FORMING
        for s in meta.dso_der_indices:
            assert meta2.der_classification.mode[int(s)] == DERMode.GRID_FOLLOWING
