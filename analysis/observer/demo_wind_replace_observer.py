"""
End-to-End Demo: Stability Observer on IEEE 39 / wind_replace Geometry
======================================================================

Simulates what the observer will see during a real run of
``experiments/000_M_TSO_M_DSO.py`` with the ``wind_replace`` scenario.

Zone geometry from ``network/ieee39/wind_replace.py`` +
``network/ieee39/constants.py``:

  Zone 1:
    - synchronous gens: {7}           (term 37 = IEEE G9, 830 MW)
    - STATCOM WPs:      {bus 1, 24}   (ex-G0 + ex-G6)
    - DSOs:             none in current SUBNET_DEFS (DSO_5 is commented out)
  Zone 2:
    - synchronous gens: {1}           (term 31 = IEEE G3, 650 MW)
    - STATCOM WPs:      {bus 5}       (ex-slack, half capacity)
    - DSOs:             3 (DSO_1, DSO_2, DSO_3 — attached via
                          3W trafos at IEEE buses 5/7/8, 4/12/14, 10/11/13)
  Zone 3:
    - synchronous gens: {4, 5}        (term 34, 39 = IEEE G6, G8)
    - STATCOM WPs:      {bus 18, 18}  (ex-G2 + ex-G3)
    - DSOs:             1 (DSO_4 at IEEE buses 21/23/24)

A 24-hour profile is simulated at 15-minute controller cadence (96 steps).
Load + PV + wind scaling is driven by a sinusoidal approximation of the
SimBench profiles.  The H matrix for each zone drifts with the operating
point — mimicking what ``coordinator.compute_cross_sensitivities`` would
produce at every refresh.

Outputs land in ``/mnt/user-data/outputs/``:
  - ``wind_replace_observer_zone{1,2,3}.png``  per-zone trajectory + histogram
  - ``wind_replace_observer.json``             machine-readable summary
  - ``wind_replace_observer_report.md``        markdown report
"""
import sys
sys.path.insert(0, ".")

import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .stability_integration_ieee39 import (
    attach_observer,
    write_observer_results_alongside_report,
    derive_tuned_gw,
)


# --------------------------------------------------------------------------- #
#  Mock objects mirroring the real coordinator / zone_def interfaces
# --------------------------------------------------------------------------- #

@dataclass
class MockZoneDef:
    """Mirrors network.ieee39 ZoneDefinition fields read by the observer."""
    tso_der_indices: List[int] = field(default_factory=list)
    pcc_trafo_indices: List[int] = field(default_factory=list)
    gen_indices: List[int] = field(default_factory=list)
    oltc_trafo_indices: List[int] = field(default_factory=list)
    v_bus_indices: List[int] = field(default_factory=list)
    line_indices: List[int] = field(default_factory=list)


class MockConfig:
    """Mirrors MultiTSOConfig fields read by attach_observer."""
    def __init__(self, g_v: float, g_q: float, result_dir: str) -> None:
        self.g_v = g_v
        self.g_q = g_q
        self.result_dir = result_dir


class WindReplaceCoordinator:
    """
    Produces per-zone H blocks that drift with a profile-driven operating
    point.  Matches the wind_replace scenario geometry.
    """
    # Wind park reactive-power magnitudes (Mvar), indexed by bus.  Used to
    # modulate the DER-column scaling of H with profile time.
    def __init__(self, zone_dims: Dict[int, Dict[str, int]], seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.zone_dims = zone_dims
        self._H_blocks: Dict = {}
        # Fix a base random Jacobian per zone (different but consistent
        # across refreshes); only the scaling drifts with time.
        self._H_base: Dict = {}
        for z, d in zone_dims.items():
            n_v = d["n_v"]
            n_i = d["n_i"]
            m = d["n_der"] + d["n_pcc"] + d["n_gen"] + d["n_oltc"]
            # Realistic column scales (dV/du at a typical monitored bus).
            # These match the orders of magnitude reported in Ortmann 2020,
            # Klein-Helmkamp 2024 and sensitivity analyses on IEEE 39.
            #   DER Q  (Mvar)  ~ 3e-5 pu/Mvar (remote bus via impedance)
            #   PCC Q  (Mvar)  ~ 5e-5 pu/Mvar (stiffer coupling via 3W trafo)
            #   V_gen  (pu)    ~ 0.3 pu/pu    (attenuated through grid)
            #   OLTC   (tap)   ~ 5e-3 pu/tap  (0.6% per tap, attenuated)
            col_scale = np.concatenate([
                np.full(d["n_der"],  3e-5),
                np.full(d["n_pcc"],  5e-5),
                np.full(d["n_gen"],  3e-1),
                np.full(d["n_oltc"], 5e-3),
            ])
            H_V = self.rng.standard_normal((n_v, m)) * col_scale
            H_I = self.rng.standard_normal((n_i, m)) * (col_scale * 0.1)
            self._H_base[z] = np.vstack([H_V, H_I])

    def advance_profile(self, t_idx: int, step_count: int = 96) -> None:
        """
        Drift each zone's H with a daily profile.  Scaling function mimics
        the net effect of load/PV/wind profiles on the Jacobian.
        """
        phase = 2 * np.pi * t_idx / step_count
        load_scale = 1.0 + 0.30 * np.sin(phase - np.pi / 2)  # peak evening
        pv_scale   = max(0.0, np.sin(phase - np.pi) + 0.3)   # midday peak
        wind_scale = 1.0 + 0.4 * np.sin(phase * 2.3)         # faster var

        for z, H_base in self._H_base.items():
            # Different zones are differently sensitive to profiles.
            if z == 1:   # heavy wind, moderate load
                s = 1.0 + 0.15 * (wind_scale - 1) + 0.10 * (load_scale - 1)
            elif z == 2: # moderate wind, DSO-heavy
                s = 1.0 + 0.20 * (load_scale - 1) + 0.05 * (pv_scale - 0.5)
            else:         # Zone 3: strong wind
                s = 1.0 + 0.25 * (wind_scale - 1) + 0.10 * (load_scale - 1)
            self._H_blocks[(z, z)] = H_base * s

    def get_H_block(self, from_z: int, to_z: int):
        return self._H_blocks.get((from_z, to_z))


# --------------------------------------------------------------------------- #
#  Build wind_replace zone geometry
# --------------------------------------------------------------------------- #

def build_wind_replace_zones() -> Dict[int, MockZoneDef]:
    """Per-zone counts derived from wind_replace.py + SUBNET_DEFS."""
    # Zone 1: 1 gen (term 37), 2 STATCOM WPs, 0 DSOs, 1 machine-OLTC
    z1 = MockZoneDef(
        tso_der_indices=[0, 1],              # 2 STATCOM WPs
        pcc_trafo_indices=[],                # no DSO in zone 1
        gen_indices=[7],                     # 1 synchronous gen
        oltc_trafo_indices=[0],              # 1 machine OLTC
        v_bus_indices=list(range(11)),       # 11 zone-1 buses
        line_indices=list(range(12)),        # lines inside zone 1
    )
    # Zone 2: 1 synchronous gen, 1 STATCOM WP, 3 DSOs (each ~3 PCC trafos),
    # 1 machine-OLTC
    z2 = MockZoneDef(
        tso_der_indices=[2],                 # 1 STATCOM WP (ex-slack)
        pcc_trafo_indices=list(range(3)),    # 3 DSOs × 1 3W trafo each
        gen_indices=[1],                     # 1 synchronous gen (IEEE G3)
        oltc_trafo_indices=[1],              # 1 machine OLTC
        v_bus_indices=list(range(14)),       # 14 zone-2 buses
        line_indices=list(range(14)),
    )
    # Zone 3: 2 synchronous gens, 2 STATCOM WPs (same bus 18), 1 DSO,
    # 2 machine-OLTCs
    z3 = MockZoneDef(
        tso_der_indices=[3, 4],              # 2 STATCOM WPs
        pcc_trafo_indices=[3],               # 1 DSO (DSO_4)
        gen_indices=[4, 5],                  # 2 synchronous gens
        oltc_trafo_indices=[2, 3],           # 2 machine OLTCs
        v_bus_indices=list(range(14)),       # 14 zone-3 buses
        line_indices=list(range(15)),
    )
    return {1: z1, 2: z2, 3: z3}


# --------------------------------------------------------------------------- #
#  Run the end-to-end simulation
# --------------------------------------------------------------------------- #

def main():
    result_dir = "/mnt/user-data/outputs"
    os.makedirs(result_dir, exist_ok=True)

    zone_defs = build_wind_replace_zones()

    # Coordinator with zone dimensions matching wind_replace geometry.
    zone_dims = {}
    for z, zd in zone_defs.items():
        zone_dims[z] = {
            "n_v":    len(zd.v_bus_indices),
            "n_i":    len(zd.line_indices),
            "n_der":  len(zd.tso_der_indices),
            "n_pcc":  len(zd.pcc_trafo_indices),
            "n_gen":  len(zd.gen_indices),
            "n_oltc": len(zd.oltc_trafo_indices),
        }
    coordinator = WindReplaceCoordinator(zone_dims, seed=42)

    # MultiTSOConfig values taken from scenario B in the experiment script
    # (line 2079: g_v=150000.0, g_q=200).
    config = MockConfig(g_v=150000.0, g_q=200.0, result_dir=result_dir)

    # Attach observer — the integration would look exactly like this
    # in the real experiment script.
    observer = attach_observer(coordinator, zone_defs, config, verbose=1)

    # Simulate 24 hours at 15-minute controller cadence.
    n_steps = 96
    print(f"  Simulating {n_steps} refresh steps (24 h at 15-min cadence) ...")
    for t_idx in range(n_steps):
        coordinator.advance_profile(t_idx, step_count=n_steps)
        time_s = t_idx * 900.0   # 15 minutes in seconds
        observer.record(time_s=time_s)

    # Write observer results using the standard helper.
    write_observer_results_alongside_report(
        observer, result_dir,
        basename="wind_replace_observer",
        verbose=1,
    )

    # Derive tuning recommendation (same as what the integration diff does).
    tuned_p95 = derive_tuned_gw(observer, statistic="percentile",
                                percentile=95.0)
    tuned_max = derive_tuned_gw(observer, statistic="max")

    print()
    print("=" * 72)
    print("  Tuning recommendation from observed profile trajectory")
    print("=" * 72)
    print(f"{'zone':>6s}  {'block':>6s}  {'mean':>10s}  {'p95':>10s}  {'max':>10s}")
    for z in sorted(observer.trajectories):
        traj = observer.trajectories[z]
        if not traj.records:
            continue
        gw_mean = traj.aggregate(statistic="mean")
        gw_p95  = traj.aggregate(statistic="percentile", percentile=95.0)
        gw_max  = traj.aggregate(statistic="max")
        for k, name in enumerate(traj.layout.names):
            sl = traj.layout.block_slice(k)
            if sl.stop == sl.start:
                continue
            print(
                f"  Z{z:>2d}   {name:>4s}   "
                f"{gw_mean[sl][0]:>10.2f}  "
                f"{gw_p95[sl][0]:>10.2f}  "
                f"{gw_max[sl][0]:>10.2f}"
            )
        print("-" * 72)
    print()
    print("Ready-to-paste block for ZoneDefinition defaults (p95 tuning):")
    for z, vals in sorted(tuned_p95.per_zone.items()):
        print(f"  # Zone {z}")
        parts = []
        for name, attr in (("DER", "g_w_der"), ("PCC", "g_w_pcc"),
                           ("V_gen", "g_w_gen"), ("OLTC", "g_w_oltc")):
            if name in vals:
                parts.append(f"{attr}={vals[name]:.2f}")
        print("  " + ",  ".join(parts) + ",")
    print()

    print("=" * 72)
    print("  CAVEAT about these numerical values")
    print("=" * 72)
    print("  These values come from a SYNTHETIC Jacobian with guessed column")
    print("  scales.  They calibrate the code path and are NOT a valid estimate")
    print("  of the spectral-gap floor for your real IEEE 39 + wind_replace")
    print("  operating point.")
    print()
    print("  The correct procedure is to run the observer DURING YOUR ACTUAL")
    print("  simulation.  coordinator.get_H_block() will return the real H")
    print("  matrix computed from the pandapower power-flow Jacobian at the")
    print("  current profile-scaled operating point.  Only those numbers can")
    print("  tell you whether your g_w defaults sit above or below the")
    print("  spectral-gap floor.")
    print()
    print("  If the observer reports g_w_spectral_gap_min >> your current g_w,")
    print("  your system is in the 'box-regularised' regime where the MIQP's")
    print("  actuator bounds and OLTC quantisation provide the implicit")
    print("  regularisation the pure spectral-gap analysis does not see.  This")
    print("  is a known and accepted mode of operation (Caduff MSc 2021,")
    print("  Ortmann PhD 2023); the empirical closed-loop validation IS the")
    print("  stability evidence in that case.")


if __name__ == "__main__":
    main()
