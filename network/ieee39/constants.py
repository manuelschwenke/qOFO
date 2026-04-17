"""
network/ieee39/constants.py
============================
Module-level constants for the IEEE 39-bus network builder and HV
sub-network attachment.

All constants are extracted from ``build_ieee39_net.py`` with leading
underscores removed for public access.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
#  Distributed slack (primary frequency response)
# ---------------------------------------------------------------------------
# 0-indexed generator indices (rows in ``net.gen``) that participate in the
# distributed slack.  Non-listed generators get ``slack_weight = 0`` and the
# ``net.ext_grid`` is always at ``0`` (angle reference only).
#
# The weight each participating generator receives is proportional to its
# ``sn_mva`` (uniform droop in per-unit of rated capacity).  Set this list
# to ``()`` to fall back to legacy single-ext-grid behaviour.

DISTRIBUTED_SLACK_GEN_INDICES: Tuple[int, ...] = tuple(range(9))


# ---------------------------------------------------------------------------
#  SimBench profile peak magnitudes (used for the 50/50 load split)
# ---------------------------------------------------------------------------
# The HS4 and HS5 simbench time-series peak at roughly 33 % of their nominal
# scaling.  When every 345 kV load is split 50 % constant + 50 % profile,
# the profile-driven half is scaled by ``0.5 / peak`` so that (constant half
# + profile half at peak) equals the original nominal load.

PROFILE_PEAK_HS4: float = 0.329
PROFILE_PEAK_HS5: float = 0.329


# ---------------------------------------------------------------------------
#  IEEE 39-bus reference line lengths (km)
# ---------------------------------------------------------------------------
#
# Source: CloudPSS IEEE 39-bus documentation.
# Keyed by (from_bus_1idx, to_bus_1idx).
# Used by _fix_line_lengths() to replace the default length_km=1.0 in
# pandapower case39() with realistic distances.

LINE_LENGTHS_KM: Dict[Tuple[int, int], float] = {
    (1, 2): 275.5,   (1, 39): 167.6,  (2, 3): 101.2,   (2, 25): 57.6,
    (3, 4): 142.8,   (3, 18): 89.1,   (4, 5): 85.8,    (4, 14): 86.5,
    (5, 6): 17.4,    (5, 8): 75.1,    (6, 7): 61.7,    (6, 11): 55.0,
    (7, 8): 30.8,    (8, 9): 243.3,   (9, 39): 167.6,  (10, 11): 28.8,
    (10, 13): 28.8,  (13, 14): 67.7,  (14, 15): 145.4, (15, 16): 63.0,
    (16, 17): 59.7,  (16, 19): 130.7, (16, 21): 90.5,  (16, 24): 39.5,
    (17, 18): 55.0,  (17, 27): 116.0, (21, 22): 93.8,  (22, 23): 64.3,
    (23, 24): 234.6, (25, 26): 216.5, (26, 27): 98.5,  (26, 28): 317.7,
    (26, 29): 418.9, (28, 29): 101.2,
}


# ── TUDA HV network topology (reused for all sub-networks) ───────────────

HV_LINE_TOPOLOGY: List[Tuple[int, int, float]] = [
    # (from_bus_no, to_bus_no, length_km)  -- matches build_tuda_net._create_lines
    (0, 1, 15),  (1, 2, 25),  (2, 3, 20),  (3, 4, 30),
    (4, 5, 40),  (5, 6, 30),  (2, 6, 20),  (6, 7, 15),
    (7, 8, 10),  (8, 9, 20),  (6, 9, 15),
]

# TUDA DER data: (hv_bus_no, p_mw, profile_name)
# TUDA_WIND_PARKS: List[Tuple[int, float, str]] = [
#     (4,  60.0, "WP7"),
#     (5, 130.0, "WP10"),
#     (6, 110.0, "WP7"),
#     (9, 110.0, "WP10"),
# ]
TUDA_WIND_PARKS: List[Tuple[int, float, str]] = [
    (4,  40.0, "WP7"),
    (5, 70.0, "WP10"),
    (6, 60.0, "WP7"),
    (9, 50.0, "WP10"),
]

# TUDA PV plants: (hv_bus_no, p_mw)  -- all use profile "PV3"
# TUDA_PV_PLANTS: List[Tuple[int, float]] = [
#     (3, 100.0),
#     (4,  60.0),
#     (5,  40.0),
#     (7,  30.0),
# ]
TUDA_PV_PLANTS: List[Tuple[int, float]] = [
    (3, 80.0),
    (4,  50.0),
    (5,  40.0),
    (7,  30.0),
]

# STATCOM-capable wind park at each HV coupling bus (MVA rating)
HV_COUPLING_WP_MVA: float = 80.0

# Multiplicative factor for the Q portion of HV sub-network loads.
# The reference Q for each DSO is derived from the pooled IEEE 39
# coupling-bus loads; this factor scales only the HV-internal Q
# (the TN-side load reduction stays at the original level).
# At 1.0 the HV networks carry exactly the IEEE 39 Q share.
# Increase to enlarge voltage spread within the 110 kV grids.
HV_Q_LOAD_FACTOR: float = 1.25

# Zone-3 buses for EHV profile assignment (0-indexed pandapower)
ZONE3_BUSES_0IDX: Set[int] = set(range(14, 24)) | {32, 33, 34, 35}

# ── Sub-network configuration table ──────────────────────────────────────
#
# Each entry: (net_id, zone, ieee_buses_1idx, hv_coupling, line_scale, gen_type)
# ieee_buses_1idx are 1-indexed IEEE bus labels matching the picture.

SUBNET_DEFS: List[dict] = [
    dict(net_id="DSO_1", zone=2,
         ieee_1idx=(7, 8, 5),    hv_buses=(3, 0, 8), scale=1.00, gen="mixed"), # 0.75
    # DSO_2 disabled — PF diverges with 3 HV sub-networks (investigate coupling buses)
    dict(net_id="DSO_2", zone=2,
         ieee_1idx=(12, 14, 4),   hv_buses=(3, 0, 8), scale=1.00, gen="mixed"),
    dict(net_id="DSO_3", zone=2,
         ieee_1idx=(11, 10, 13), hv_buses=(3, 0, 8), scale=1.00, gen="mixed"), # 0.75
    # dict(net_id="DSO_4", zone=3,
    #      ieee_1idx=(24, 21, 23), hv_buses=(3, 0, 8), scale=2.00, gen="pv"),
    # dict(net_id="DSO_5", zone=1,
    #      ieee_1idx=(27, 26, 25), hv_buses=(3, 0, 8), scale=3.00, gen="wind"),
]
