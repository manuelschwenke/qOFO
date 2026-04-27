"""
network/ieee39/meta.py
======================
Dataclass definitions for IEEE 39-bus network metadata and HV sub-network
tracking information.

These are pure data containers with no pandapower dependency, making them
safe to import in contexts where only structural metadata is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
#  Network metadata
# ---------------------------------------------------------------------------

@dataclass
class IEEE39NetworkMeta:
    """
    Immutable index catalogue for the IEEE 39-bus network.

    All index lists refer to pandapower element indices (integer row labels
    in the respective DataFrame, e.g. net.sgen.index).

    The separation into TN (345 kV, "subnet = TN") and DN (20 kV,
    "subnet = DN") mirrors the convention used in the TU-Darmstadt benchmark
    (build_tuda_net.py) so that the rest of the code-base can be reused.
    """

    # ── Transmission-network (345 kV) ────────────────────────────────────────
    tn_bus_indices: Tuple[int, ...]
    """All 345 kV bus indices (the 39 original buses)."""

    tn_line_indices: Tuple[int, ...]
    """All line indices (between 345 kV buses)."""

    # ── Generators ───────────────────────────────────────────────────────────
    gen_indices: Tuple[int, ...]
    """pandapower ``net.gen`` indices for every synchronous machine in the
    model.  Since :func:`network.ieee39.helpers.swap_slack_to_bus38`
    replaced the legacy ``ext_grid`` with a ``slack=True`` gen at bus 38,
    this tuple contains all 10 machines (9 original PV gens + the slack-
    enabled ex-G10 at bus 38) and every one of them is controllable by
    the TSO OFO loop.
    """

    gen_bus_indices: Tuple[int, ...]
    """Bus index of each generator (terminal bus — may be 10.5 kV if machine
    trafos are present, or 345 kV if directly connected)."""

    # ── TSO-level DER sgens ───────────────────────────────────────────────────
    tso_der_indices: Tuple[int, ...]
    """Indices of sgen elements co-located at generator buses.
    These represent the TSO's controllable reactive-power actuators.
    """

    tso_der_buses: Tuple[int, ...]
    """Bus indices of the TSO DER sgens (same order as tso_der_indices)."""

    # ── Generators: grid buses and machine transformers (optional) ─────────────
    gen_grid_bus_indices: Tuple[int, ...] = field(default_factory=tuple)
    """Original 345 kV grid bus of each generator (before machine trafo
    insertion).  Used for zone partitioning.  Same as gen_bus_indices when
    no machine transformers exist."""

    machine_trafo_indices: Tuple[int, ...] = field(default_factory=tuple)
    """Indices in ``net.trafo`` of 2W machine transformers connecting
    generator terminals to the grid.  OLTC actuators in the TSO OFO."""

    machine_trafo_gen_map: Tuple[int, ...] = field(default_factory=tuple)
    """For each machine trafo, the ``net.gen`` index of its generator
    (same order as machine_trafo_indices)."""

    # ── DSO feeders (populated by add_dso_feeders) ────────────────────────────
    dso_pcc_trafo_indices: Tuple[int, ...] = field(default_factory=tuple)
    """Indices of the 2-winding PCC trafos connecting each DSO feeder (345/20 kV)."""

    dso_pcc_hv_buses: Tuple[int, ...] = field(default_factory=tuple)
    """345 kV buses at which DSO feeders are attached."""

    dso_lv_buses: Tuple[int, ...] = field(default_factory=tuple)
    """20 kV distribution buses (one per DSO feeder)."""

    dso_der_indices: Tuple[int, ...] = field(default_factory=tuple)
    """sgen indices for DERs inside the DSO feeders (20 kV level)."""

    dso_der_buses: Tuple[int, ...] = field(default_factory=tuple)
    """Bus indices of DSO DERs (same order as dso_der_indices)."""

    dso_shunt_indices: Tuple[int, ...] = field(default_factory=tuple)
    """Shunt indices inside DSO feeders (one switchable shunt per feeder)."""

    dso_shunt_buses: Tuple[int, ...] = field(default_factory=tuple)
    """Bus indices of DSO shunts."""

    dn_bus_indices: Tuple[int, ...] = field(default_factory=tuple)
    """All 20 kV distribution bus indices."""

    dn_line_indices: Tuple[int, ...] = field(default_factory=tuple)
    """Line indices within distribution networks (empty for simple feeders)."""

    hv_networks: Tuple = field(default_factory=tuple)
    """HVNetworkInfo objects for attached 110 kV sub-networks (see add_hv_networks)."""


# ---------------------------------------------------------------------------
#  110 kV HV sub-network tracking
# ---------------------------------------------------------------------------

@dataclass
class HVNetworkInfo:
    """Tracking information for one attached 110 kV sub-network.

    Each HV sub-network replicates the TUDA 110 kV distribution topology
    (10 buses, 11 lines) and is coupled to the IEEE 39-bus 345 kV
    transmission network via 3-winding 345/110/20 kV transformers at three
    coupling points.
    """
    net_id: str
    """Unique sub-network identifier (e.g. ``"DSO_1"``)."""

    bus_indices: Tuple[int, ...]
    """Pandapower bus indices of the 10 HV (110 kV) buses."""

    line_indices: Tuple[int, ...]
    """Pandapower line indices within this sub-network."""

    sgen_indices: Tuple[int, ...]
    """Sgen indices for DER plants (PV / wind) in this sub-network."""

    load_indices: Tuple[int, ...]
    """Load indices (HV/MV substations, 10 loads)."""

    coupling_trafo_indices: Tuple[int, ...]
    """3-winding transformer indices (``net.trafo3w``) coupling
    345 kV TN to 110 kV HV with a 20 kV tertiary winding."""

    coupling_ieee_buses: Tuple[int, ...]
    """IEEE 39-bus TN bus (0-indexed) at the HV side of each coupling trafo."""

    coupling_hv_bus_indices: Tuple[int, ...]
    """HV bus index at the LV side of each coupling transformer."""

    zone: int = 0
    """IEEE zone this sub-network belongs to."""

    line_length_scale: float = 1.0
    """Scale factor applied to all line lengths relative to TUDA base."""

    total_ref_p_mw: float = 0.0
    """Total reference active power (sum of replaced IEEE loads) [MW]."""

    total_ref_q_mvar: float = 0.0
    """Total reference reactive power [Mvar]."""

    gen_type: str = "mixed"
    """Generation type: 'mixed', 'pv', or 'wind'."""
