"""
Tests for trip/restore of *existing* loads via ContingencyEvent.

Covers the three addressing modes resolved by
:func:`experiments.helpers.contingency.prepare_load_contingencies`:

  1. ``element_index`` — pre-set integer index into ``net.load``.
  2. ``name``          — exact match against ``net.load["name"]``.
  3. Legacy dormant pattern: ``(bus, p_mw, q_mvar)`` + ``"connect"``
     creates a new dormant load that subsequent ``trip``/``restore``
     events at the same key target.

The same ``_apply_contingency`` function downstream handles all three;
this file exercises the resolution step plus the apply step end-to-end.
"""

from __future__ import annotations

import pandapower as pp
import pytest

from experiments.helpers.contingency import (
    _apply_contingency,
    prepare_load_contingencies,
)
from experiments.helpers.records import ContingencyEvent


# ---------------------------------------------------------------------------
#  Test fixtures
# ---------------------------------------------------------------------------


def _build_tiny_net() -> pp.pandapowerNet:
    """Two-bus pandapower net with three named loads at known indices."""
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=110.0, name="Bus_A")
    b1 = pp.create_bus(net, vn_kv=110.0, name="Bus_B")
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0)
    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b1, length_km=10.0,
        r_ohm_per_km=0.1, x_ohm_per_km=0.3,
        c_nf_per_km=10.0, max_i_ka=1.0,
    )
    pp.create_load(net, bus=b0, p_mw=10.0, q_mvar=2.0, name="L_const")
    pp.create_load(net, bus=b1, p_mw=20.0, q_mvar=4.0, name="L_profile")
    pp.create_load(net, bus=b1, p_mw=5.0,  q_mvar=1.0, name="L_extra")
    # Add the base_p_mw/base_q_mvar columns that prepare_load_contingencies
    # writes into when creating dormant loads (mirrors snapshot_base_values).
    net.load["base_p_mw"] = net.load["p_mw"].astype(float)
    net.load["base_q_mvar"] = net.load["q_mvar"].astype(float)
    return net


# ---------------------------------------------------------------------------
#  Mode 1: trip/restore by element_index
# ---------------------------------------------------------------------------


def test_trip_restore_existing_load_by_index() -> None:
    net = _build_tiny_net()
    target_idx = int(net.load.index[0])

    events = [
        ContingencyEvent(minute=10, element_type="load",
                         element_index=target_idx, action="trip"),
        ContingencyEvent(minute=20, element_type="load",
                         element_index=target_idx, action="restore"),
    ]

    prepare_load_contingencies(net, events, verbose=0)

    assert events[0].element_index == target_idx
    assert events[1].element_index == target_idx

    _apply_contingency(net, events[0], verbose=0)
    assert bool(net.load.at[target_idx, "in_service"]) is False

    _apply_contingency(net, events[1], verbose=0)
    assert bool(net.load.at[target_idx, "in_service"]) is True


# ---------------------------------------------------------------------------
#  Mode 2: trip/restore by name
# ---------------------------------------------------------------------------


def test_trip_restore_existing_load_by_name() -> None:
    net = _build_tiny_net()
    target_idx = int(net.load.index[net.load["name"] == "L_extra"][0])

    events = [
        ContingencyEvent(minute=10, element_type="load",
                         name="L_extra", action="trip"),
        ContingencyEvent(minute=20, element_type="load",
                         name="L_extra", action="restore"),
    ]

    prepare_load_contingencies(net, events, verbose=0)

    assert events[0].element_index == target_idx
    assert events[1].element_index == target_idx

    _apply_contingency(net, events[0], verbose=0)
    assert bool(net.load.at[target_idx, "in_service"]) is False

    _apply_contingency(net, events[1], verbose=0)
    assert bool(net.load.at[target_idx, "in_service"]) is True


# ---------------------------------------------------------------------------
#  Mode 3: legacy dormant-load pattern still works
# ---------------------------------------------------------------------------


def test_legacy_dormant_connect_trip_pattern() -> None:
    net = _build_tiny_net()
    n_loads_before = len(net.load)

    events = [
        ContingencyEvent(minute=10, element_type="load",
                         bus=0, p_mw=50.0, q_mvar=10.0, action="connect"),
        ContingencyEvent(minute=20, element_type="load",
                         bus=0, p_mw=50.0, q_mvar=10.0, action="trip"),
    ]

    prepare_load_contingencies(net, events, verbose=0)

    assert len(net.load) == n_loads_before + 1
    new_idx = events[0].element_index
    assert new_idx >= n_loads_before
    assert events[1].element_index == new_idx
    # Dormant load starts out-of-service
    assert bool(net.load.at[new_idx, "in_service"]) is False

    # Apply "connect" -> in_service True
    _apply_contingency(net, events[0], verbose=0)
    assert bool(net.load.at[new_idx, "in_service"]) is True

    # Apply "trip" -> in_service False
    _apply_contingency(net, events[1], verbose=0)
    assert bool(net.load.at[new_idx, "in_service"]) is False


# ---------------------------------------------------------------------------
#  Mixed: existing-trip + dormant-connect coexist in one prepare() call
# ---------------------------------------------------------------------------


def test_mixed_existing_and_dormant_events() -> None:
    net = _build_tiny_net()
    existing_idx = int(net.load.index[net.load["name"] == "L_profile"][0])
    n_loads_before = len(net.load)

    events = [
        # Existing load by name
        ContingencyEvent(minute=10, element_type="load",
                         name="L_profile", action="trip"),
        # Dormant load via legacy pattern
        ContingencyEvent(minute=15, element_type="load",
                         bus=1, p_mw=30.0, q_mvar=5.0, action="connect"),
        ContingencyEvent(minute=25, element_type="load",
                         bus=1, p_mw=30.0, q_mvar=5.0, action="trip"),
    ]

    prepare_load_contingencies(net, events, verbose=0)

    # Existing-load event resolved to L_profile's index, untouched count
    assert events[0].element_index == existing_idx
    # Dormant-load events resolved to a brand-new index
    new_idx = events[1].element_index
    assert events[2].element_index == new_idx
    assert new_idx != existing_idx
    assert len(net.load) == n_loads_before + 1


# ---------------------------------------------------------------------------
#  Edge cases — all of these should raise
# ---------------------------------------------------------------------------


def test_raises_when_name_matches_zero_rows() -> None:
    net = _build_tiny_net()
    events = [ContingencyEvent(minute=10, element_type="load",
                               name="does_not_exist", action="trip")]
    with pytest.raises(ValueError, match="no net.load row matches"):
        prepare_load_contingencies(net, events, verbose=0)


def test_raises_when_name_matches_multiple_rows() -> None:
    net = _build_tiny_net()
    # Force a duplicate name to trigger the ambiguity guard
    dup_idx = pp.create_load(
        net, bus=0, p_mw=1.0, q_mvar=0.5, name="L_const",
    )
    events = [ContingencyEvent(minute=10, element_type="load",
                               name="L_const", action="trip")]
    with pytest.raises(ValueError, match="matches multiple"):
        prepare_load_contingencies(net, events, verbose=0)
    # Defensive: make sure we exercised the duplicate path
    assert dup_idx in net.load.index


def test_raises_when_element_index_out_of_range() -> None:
    net = _build_tiny_net()
    bad_idx = int(net.load.index.max()) + 1000
    events = [ContingencyEvent(minute=10, element_type="load",
                               element_index=bad_idx, action="trip")]
    with pytest.raises(ValueError, match="not in"):
        prepare_load_contingencies(net, events, verbose=0)


def test_raises_when_connect_combined_with_explicit_index() -> None:
    net = _build_tiny_net()
    target_idx = int(net.load.index[0])
    events = [ContingencyEvent(minute=10, element_type="load",
                               element_index=target_idx, action="connect")]
    with pytest.raises(ValueError, match="contradiction"):
        prepare_load_contingencies(net, events, verbose=0)


def test_raises_when_connect_combined_with_name() -> None:
    net = _build_tiny_net()
    events = [ContingencyEvent(minute=10, element_type="load",
                               name="L_extra", action="connect")]
    with pytest.raises(ValueError, match="contradiction"):
        prepare_load_contingencies(net, events, verbose=0)
