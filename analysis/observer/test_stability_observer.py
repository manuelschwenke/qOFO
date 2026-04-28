"""
Tests for stability_observer.

Uses a minimal ``MockCoordinator`` that mimics the interface the observer
actually consumes — ``get_H_block(z, z)`` returning a sensitivity matrix.
This lets us test the observer without pulling in the full MultiTSOCoordinator
and its pandapower dependency.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import tempfile
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from analysis.observer.stability_tuning import BlockLayout
from analysis.observer.stability_observer import (
    StabilityObserver,
    ObservationRecord,
    ZoneTrajectory,
    write_trajectory_report,
)


# --------------------------------------------------------------------------- #
#  Mock objects
# --------------------------------------------------------------------------- #

@dataclass
class MockZoneDef:
    """Minimum fields the observer reads off a zone definition."""
    tso_der_indices: list = field(default_factory=list)
    pcc_trafo_indices: list = field(default_factory=list)
    gen_indices: list = field(default_factory=list)
    oltc_trafo_indices: list = field(default_factory=list)
    v_bus_indices: list = field(default_factory=list)
    line_indices: list = field(default_factory=list)


class MockCoordinator:
    """Provides ``get_H_block`` with profile-driven H matrices."""

    def __init__(self, n_zones=3, seed=0):
        self.rng = np.random.default_rng(seed)
        self.zones = list(range(1, n_zones + 1))
        self._H_blocks = {}
        self._zone_dims = {}   # (n_v, m) per zone

    def set_zone_dim(self, z, n_v, n_der, n_pcc, n_gen, n_oltc):
        self._zone_dims[z] = (n_v, n_der + n_pcc + n_gen + n_oltc)

    def advance_profile(self, t_idx):
        """Generate a new H for each zone as though the profile advanced."""
        for z in self.zones:
            n_v, m = self._zone_dims[z]
            # Introduce time-dependent amplitude to mimic load/PV variation.
            scale = 1.0 + 0.3 * np.sin(2 * np.pi * t_idx / 50.0)
            # Line currents also live in the H row block, append zeros.
            H = self.rng.standard_normal((n_v, m)) * 0.01 * scale
            self._H_blocks[(z, z)] = H

    def get_H_block(self, from_z, to_z):
        return self._H_blocks.get((from_z, to_z))


# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #

def _make_test_setup(n_v=12, n_der=3, n_pcc=2, n_gen=3, n_oltc=3, n_zones=3):
    coord = MockCoordinator(n_zones=n_zones, seed=42)
    zone_defs = {}
    for z in coord.zones:
        zone_defs[z] = MockZoneDef(
            tso_der_indices=list(range(n_der)),
            pcc_trafo_indices=list(range(n_pcc)),
            gen_indices=list(range(n_gen)),
            oltc_trafo_indices=list(range(n_oltc)),
            v_bus_indices=list(range(n_v)),
        )
        coord.set_zone_dim(z, n_v, n_der, n_pcc, n_gen, n_oltc)
    return coord, zone_defs


def test_observer_records_each_refresh():
    coord, zone_defs = _make_test_setup()
    obs = StabilityObserver(coord, zone_defs, g_v=1.0, g_q=0.5,
                            safety_margin=0.3)

    n_steps = 20
    for t_idx in range(n_steps):
        coord.advance_profile(t_idx)
        obs.record(time_s=t_idx * 900.0)   # 15-min steps

    for z in coord.zones:
        traj = obs.trajectories[z]
        assert len(traj.records) == n_steps, (
            f"zone {z}: expected {n_steps} records, got {len(traj.records)}"
        )
        gw_traj = traj.gw_block_trajectory()
        assert gw_traj.shape == (n_steps, 4)
    print(f"[OK] Observer recorded {n_steps} snapshots for "
          f"{len(coord.zones)} zones")


def test_trajectory_aggregation_ordering():
    coord, zone_defs = _make_test_setup()
    obs = StabilityObserver(coord, zone_defs, g_v=1.0, g_q=0.5)
    for t_idx in range(50):
        coord.advance_profile(t_idx)
        obs.record(time_s=t_idx * 900.0)

    for z in coord.zones:
        traj = obs.trajectories[z]
        gw_max = traj.aggregate(statistic="max")
        gw_p95 = traj.aggregate(statistic="percentile", percentile=95.0)
        gw_mean = traj.aggregate(statistic="mean")
        assert np.all(gw_max >= gw_p95 - 1e-9), (
            "max must dominate p95 element-wise"
        )
        assert np.all(gw_p95 >= gw_mean - 1e-9), (
            "p95 must dominate mean element-wise"
        )
    print("[OK] Aggregation preserves max >= p95 >= mean ordering")


def test_observer_skips_missing_zones():
    """If the coordinator returns None for a zone, it must be skipped silently."""
    coord, zone_defs = _make_test_setup(n_zones=3)
    obs = StabilityObserver(coord, zone_defs, g_v=1.0, g_q=0.5)

    # Advance profile but deliberately omit zone 2 from _H_blocks.
    for t_idx in range(10):
        coord.advance_profile(t_idx)
        coord._H_blocks.pop((2, 2), None)
        obs.record(time_s=t_idx * 900.0)

    # Zones 1 and 3 should have records; zone 2 should be empty.
    assert len(obs.trajectories[1].records) == 10
    assert len(obs.trajectories[2].records) == 0
    assert len(obs.trajectories[3].records) == 10
    print("[OK] Observer gracefully skips zones without H block")


def test_summary_structure():
    coord, zone_defs = _make_test_setup()
    # Pin method="block": the OLTC>V_gen>PCC>DER assertion below relies on
    # ratio_priors, which only the block-scalar tuner enforces. The Gershgorin
    # LMI tuner picks per-actuator floors that need not preserve this ordering.
    obs = StabilityObserver(coord, zone_defs, g_v=1.0, g_q=0.5, method="block")
    for t_idx in range(10):
        coord.advance_profile(t_idx)
        obs.record(time_s=t_idx * 900.0)

    summary = obs.summary(statistic="max")
    for z in coord.zones:
        entry = summary[z]
        assert entry["n_samples"] == 10
        assert "gw_per_block_max" in entry
        assert set(entry["gw_per_block_max"].keys()) == {
            "DER", "PCC", "V_gen", "OLTC",
        }
        # Physical-prior ordering holds on the aggregate as well:
        # OLTC > V_gen > PCC > DER because ratio_priors enforces it.
        v = entry["gw_per_block_max"]
        assert v["OLTC"] > v["V_gen"] > v["PCC"] > v["DER"]
        # Haeberle floor present and positive
        assert "gw_haberle_max" in entry
        assert entry["gw_haberle_max"] > 0.0
    print("[OK] Summary structure correct, block-ratio ordering preserved")


def test_write_results_end_to_end():
    """Dump JSON + PNGs to a temp dir and check all files exist."""
    coord, zone_defs = _make_test_setup()
    obs = StabilityObserver(coord, zone_defs, g_v=1.0, g_q=0.5)
    for t_idx in range(25):
        coord.advance_profile(t_idx)
        obs.record(time_s=t_idx * 900.0)

    with tempfile.TemporaryDirectory() as td:
        obs.write_results(td, plot=True)
        json_path = os.path.join(td, "stability_observer.json")
        assert os.path.isfile(json_path)
        import json
        with open(json_path) as fh:
            payload = json.load(fh)
        assert "aggregations" in payload
        assert set(payload["aggregations"].keys()) == {
            "max", "percentile", "mean",
        }

        for z in coord.zones:
            png = os.path.join(td, f"stability_observer_zone{z}.png")
            assert os.path.isfile(png), f"missing plot for zone {z}"

        # Markdown report
        md_path = os.path.join(td, "report.md")
        write_trajectory_report(md_path, obs.trajectories)
        with open(md_path) as fh:
            md = fh.read()
        for z in coord.zones:
            assert f"Zone {z}" in md
    print("[OK] JSON + PNG + markdown artefacts written correctly")


def test_realistic_trajectory_shape():
    """Check that a sinusoidal load scaling produces a realistic time series."""
    coord, zone_defs = _make_test_setup(n_zones=1)
    obs = StabilityObserver(coord, zone_defs, g_v=1.0, g_q=0.5,
                            safety_margin=0.2)

    for t_idx in range(96):   # 24 h at 15-min steps
        coord.advance_profile(t_idx)
        obs.record(time_s=t_idx * 900.0)

    traj = obs.trajectories[1]
    op_norms = traj.op_norm_trajectory()
    # The sine amplitude shows up in ||M||_op - coefficient of variation
    # should be non-trivial (>5%).
    cv = op_norms.std() / op_norms.mean()
    assert cv > 0.05, f"CV of ||M||_op = {cv:.3f}, expected >0.05 variability"
    print(f"[OK] Realistic trajectory: CV(||M||_op) = {cv:.3f}")


if __name__ == "__main__":
    print("Running stability_observer tests...\n")
    test_observer_records_each_refresh()
    test_trajectory_aggregation_ordering()
    test_observer_skips_missing_zones()
    test_summary_structure()
    test_write_results_end_to_end()
    test_realistic_trajectory_shape()
    print("\nAll tests passed.")
