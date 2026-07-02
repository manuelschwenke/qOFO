"""
network/boundary_topology.py
============================
Boundary topology for the Boundary Marginal Exchange (BME) coordination
scheme — spec §3.1–§3.3 (see ``docs/BME_STATUS.md``).

Symbol map (code ↔ spec)
------------------------
* ``registry``            ↔ B, the global boundary bus set (§3.1), in a fixed
                            order (ascending pandapower bus index) that every
                            boundary-indexed vector uses.
* ``B_pairs[(i, j)]``     ↔ B_ij, boundary buses of ties between zones i < j.
* ``TieLine``             ↔ one inter-zone tie e = (i, j) with fixed
                            orientation zone_i < zone_j.
* ``bus_owner`` /
  ``tie_loss_shares``     ↔ ownership convention (§3.3, DECISION D1:
                            tie-line losses split 50/50; every bus owned by
                            exactly one zone).
* ``assert_separator``    ↔ vertex-separator assumption (§3.2): removing B
                            disconnects the zone interiors. Hard error on
                            violation — the fix is to enlarge B, never to
                            weaken the check.

Bus ownership is *closure-based*: after removing B from the connectivity
graph, every remaining component is assigned to the unique zone whose
partition buses it contains. Buses absent from the zone partition (e.g.
10.5 kV generator-terminal buses, DN feeder buses) are thereby owned by
the zone they are electrically embedded in. Components containing buses
of two or more zones violate the separator assumption and raise.

Fail-fast throughout: missing preconditions raise ``ValueError`` with a
precise message; there are no silent defaults.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-02 (BME Phase 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import pandapower as pp


@dataclass(frozen=True)
class TieLine:
    """One inter-zone tie line e = (i, j), orientation fixed zone_i < zone_j.

    Attributes
    ----------
    line_idx :
        Pandapower ``net.line`` index of the tie.
    zone_i, zone_j :
        The two zone identifiers, with ``zone_i < zone_j``.
    bus_i, bus_j :
        Boundary endpoint bus in ``zone_i`` / ``zone_j`` respectively.
    """

    line_idx: int
    zone_i: int
    zone_j: int
    bus_i: int
    bus_j: int


class BoundaryTopology:
    """Boundary registry, per-pair subsets, ownership and separator check.

    Built once from the plant net and the zone partition; immutable
    afterwards (no re-detection at runtime — topology changes require a
    rebuild, consistent with the cached-sensitivity philosophy).

    Parameters
    ----------
    net :
        Pandapower network (in-service topology is read; no power-flow
        results are required).
    zone_map :
        ``{zone_id: [bus indices]}`` — the TN zone partition. Buses not
        listed (generator terminals, DN feeders) are absorbed into their
        electrical zone by closure. Every listed bus must exist in
        ``net.bus``.
    tie_loss_split :
        Fraction of each tie line's losses owned by ``zone_i`` (D1 —
        default 0.5, i.e. 50/50).
    """

    def __init__(
        self,
        net: pp.pandapowerNet,
        zone_map: Dict[int, List[int]],
        *,
        tie_loss_split: float = 0.5,
    ) -> None:
        if not zone_map:
            raise ValueError("zone_map must contain at least one zone")
        if not (0.0 <= tie_loss_split <= 1.0):
            raise ValueError(
                f"tie_loss_split must lie in [0, 1], got {tie_loss_split}"
            )
        self.zone_ids: List[int] = sorted(zone_map.keys())
        self._tie_loss_split = float(tie_loss_split)

        bus_set = set(int(b) for b in net.bus.index)
        self._bus_zone_partition: Dict[int, int] = {}
        for z, buses in zone_map.items():
            for b in buses:
                b = int(b)
                if b not in bus_set:
                    raise ValueError(
                        f"zone_map assigns bus {b} to zone {z}, but the bus "
                        f"does not exist in net.bus"
                    )
                if b in self._bus_zone_partition:
                    raise ValueError(
                        f"bus {b} assigned to two zones "
                        f"({self._bus_zone_partition[b]} and {z})"
                    )
                self._bus_zone_partition[b] = int(z)

        # ── Tie detection: in-service lines with endpoints in two zones ──
        self.ties: List[TieLine] = []
        for li in net.line.index:
            if not bool(net.line.at[li, "in_service"]):
                continue
            fb = int(net.line.at[li, "from_bus"])
            tb = int(net.line.at[li, "to_bus"])
            zf = self._bus_zone_partition.get(fb)
            zt = self._bus_zone_partition.get(tb)
            if zf is None or zt is None or zf == zt:
                continue
            if zf < zt:
                self.ties.append(TieLine(int(li), zf, zt, fb, tb))
            else:
                self.ties.append(TieLine(int(li), zt, zf, tb, fb))
        self.ties.sort(key=lambda t: t.line_idx)

        # ── Cross-zone non-line branches are a separator violation ──────
        self._assert_no_cross_zone_nonline_branch(net)

        # ── Boundary registry B: all tie endpoints, fixed ascending order ─
        boundary = set()
        for t in self.ties:
            boundary.add(t.bus_i)
            boundary.add(t.bus_j)
        self.registry: List[int] = sorted(boundary)
        self.registry_pos: Dict[int, int] = {
            b: k for k, b in enumerate(self.registry)
        }

        # ── B_ij per unordered zone pair ─────────────────────────────────
        self.B_pairs: Dict[Tuple[int, int], List[int]] = {}
        for t in self.ties:
            key = (t.zone_i, t.zone_j)
            self.B_pairs.setdefault(key, [])
            for b in (t.bus_i, t.bus_j):
                if b not in self.B_pairs[key]:
                    self.B_pairs[key].append(b)
        for key in self.B_pairs:
            self.B_pairs[key] = sorted(self.B_pairs[key])

        # ── Closure-based ownership + separator assertion ────────────────
        self._graph = self._build_graph(net)
        self._bus_owner: Dict[int, int] = {}
        self._assign_ownership_and_assert_separator()

        # ── Per-zone views ───────────────────────────────────────────────
        self._own_boundary: Dict[int, List[int]] = {
            z: sorted(
                {t.bus_i for t in self.ties if t.zone_i == z}
                | {t.bus_j for t in self.ties if t.zone_j == z}
            )
            for z in self.zone_ids
        }
        self._adjacent_boundary: Dict[int, List[int]] = {}
        for z in self.zone_ids:
            adj = set(self._own_boundary[z])
            for t in self.ties:
                if t.zone_i == z:
                    adj.add(t.bus_j)
                elif t.zone_j == z:
                    adj.add(t.bus_i)
            self._adjacent_boundary[z] = sorted(adj)

    # ------------------------------------------------------------------
    #  Graph construction and separator machinery
    # ------------------------------------------------------------------

    @staticmethod
    def _build_graph(net: pp.pandapowerNet) -> nx.Graph:
        """Connectivity graph over in-service buses and branches.

        Covers lines, 2W trafos, 3W trafos (pairwise terminal edges),
        impedances and closed bus-bus switches. Raises on element tables
        this check does not model (DC lines), rather than silently
        ignoring them.
        """
        g = nx.Graph()
        for b in net.bus.index:
            if bool(net.bus.at[b, "in_service"]):
                g.add_node(int(b))
        for li in net.line.index:
            if bool(net.line.at[li, "in_service"]):
                g.add_edge(
                    int(net.line.at[li, "from_bus"]),
                    int(net.line.at[li, "to_bus"]),
                )
        for t in net.trafo.index:
            if bool(net.trafo.at[t, "in_service"]):
                g.add_edge(
                    int(net.trafo.at[t, "hv_bus"]),
                    int(net.trafo.at[t, "lv_bus"]),
                )
        if hasattr(net, "trafo3w") and len(net.trafo3w):
            for t in net.trafo3w.index:
                if not bool(net.trafo3w.at[t, "in_service"]):
                    continue
                hv = int(net.trafo3w.at[t, "hv_bus"])
                mv = int(net.trafo3w.at[t, "mv_bus"])
                lv = int(net.trafo3w.at[t, "lv_bus"])
                g.add_edge(hv, mv)
                g.add_edge(hv, lv)
                g.add_edge(mv, lv)
        if hasattr(net, "impedance") and len(net.impedance):
            for i in net.impedance.index:
                if bool(net.impedance.at[i, "in_service"]):
                    g.add_edge(
                        int(net.impedance.at[i, "from_bus"]),
                        int(net.impedance.at[i, "to_bus"]),
                    )
        if hasattr(net, "switch") and len(net.switch):
            for s in net.switch.index:
                if (
                    str(net.switch.at[s, "et"]) == "b"
                    and bool(net.switch.at[s, "closed"])
                ):
                    g.add_edge(
                        int(net.switch.at[s, "bus"]),
                        int(net.switch.at[s, "element"]),
                    )
        if hasattr(net, "dcline") and len(net.dcline):
            raise ValueError(
                "BoundaryTopology does not model dcline elements; extend "
                "_build_graph before using nets with DC lines."
            )
        return g

    def _assert_no_cross_zone_nonline_branch(
        self, net: pp.pandapowerNet
    ) -> None:
        """§3.2: every inter-zone branch must be a tie *line* whose
        endpoints enter B. A cross-zone trafo/impedance would leave a
        coupling path outside B — enlarge B (i.e. re-partition), never
        weaken the check."""
        offenders: List[str] = []

        def _z(b: int):
            return self._bus_zone_partition.get(int(b))

        for t in net.trafo.index:
            if not bool(net.trafo.at[t, "in_service"]):
                continue
            zh, zl = _z(net.trafo.at[t, "hv_bus"]), _z(net.trafo.at[t, "lv_bus"])
            if zh is not None and zl is not None and zh != zl:
                offenders.append(f"trafo {t} (zones {zh}–{zl})")
        if hasattr(net, "trafo3w") and len(net.trafo3w):
            for t in net.trafo3w.index:
                if not bool(net.trafo3w.at[t, "in_service"]):
                    continue
                zs = {
                    _z(net.trafo3w.at[t, c])
                    for c in ("hv_bus", "mv_bus", "lv_bus")
                }
                zs.discard(None)
                if len(zs) > 1:
                    offenders.append(f"trafo3w {t} (zones {sorted(zs)})")
        if hasattr(net, "impedance") and len(net.impedance):
            for i in net.impedance.index:
                if not bool(net.impedance.at[i, "in_service"]):
                    continue
                zf = _z(net.impedance.at[i, "from_bus"])
                zt = _z(net.impedance.at[i, "to_bus"])
                if zf is not None and zt is not None and zf != zt:
                    offenders.append(f"impedance {i} (zones {zf}–{zt})")
        if offenders:
            raise ValueError(
                "Separator assumption violated: cross-zone non-line "
                f"branches found: {offenders}. The boundary set B only "
                "collects tie-LINE endpoints; enlarge B / fix the "
                "partition instead of weakening this check (spec §3.2)."
            )

    def _assign_ownership_and_assert_separator(self) -> None:
        """Assign every in-service bus to exactly one owner zone and
        assert the vertex-separator property in one pass.

        Method: remove B from the connectivity graph; every remaining
        component must contain partition buses of at most one zone
        (separator assertion). Components with partition buses take that
        zone; orphan components (no partition bus — e.g. a feeder hanging
        off a boundary bus) take the owner of the boundary buses they
        attach to, which must be unique. Boundary buses own themselves via
        their partition zone.
        """
        g_wo_b = self._graph.copy()
        g_wo_b.remove_nodes_from(self.registry)

        for comp in nx.connected_components(g_wo_b):
            zones_present = {
                self._bus_zone_partition[b]
                for b in comp
                if b in self._bus_zone_partition
            }
            if len(zones_present) > 1:
                raise ValueError(
                    "Separator assumption violated (spec §3.2): removing "
                    f"the boundary buses B={self.registry} leaves a "
                    f"component spanning zones {sorted(zones_present)} "
                    f"(component buses: {sorted(comp)}). Enlarge B / fix "
                    "the partition instead of weakening this check."
                )
            if len(zones_present) == 1:
                owner = zones_present.pop()
            else:
                # Orphan component: attaches only to boundary buses.
                attach_owners = set()
                for b in comp:
                    for nb in self._graph.neighbors(b):
                        if nb in self.registry_pos:
                            attach_owners.add(
                                self._bus_zone_partition[nb]
                            )
                if len(attach_owners) != 1:
                    raise ValueError(
                        f"Cannot assign owner zone to component "
                        f"{sorted(comp)}: it contains no partition bus and "
                        f"attaches to boundary buses of zones "
                        f"{sorted(attach_owners)} (need exactly one)."
                    )
                owner = attach_owners.pop()
            for b in comp:
                self._bus_owner[int(b)] = int(owner)

        for b in self.registry:
            self._bus_owner[b] = self._bus_zone_partition[b]

        unowned = [
            b for b in self._graph.nodes if b not in self._bus_owner
        ]
        if unowned:
            raise ValueError(
                f"Buses without an owner zone after closure: {unowned}. "
                "Every in-service bus must be owned by exactly one zone "
                "(spec §3.3 / D1)."
            )

    # ------------------------------------------------------------------
    #  Public views
    # ------------------------------------------------------------------

    def own_boundary(self, zone: int) -> List[int]:
        """Zone's own tie endpoints (its share of B), ascending."""
        self._require_zone(zone)
        return list(self._own_boundary[zone])

    def adjacent_boundary(self, zone: int) -> List[int]:
        """Boundary buses adjacent to the zone: own endpoints plus the
        far endpoints of its ties (the support of μ_zone, §3.4)."""
        self._require_zone(zone)
        return list(self._adjacent_boundary[zone])

    def bus_owner(self, bus: int) -> int:
        """Owning zone of a bus (closure-based, D1)."""
        b = int(bus)
        if b not in self._bus_owner:
            raise ValueError(f"bus {b} has no owner (not in the net?)")
        return self._bus_owner[b]

    def zone_buses(self, zone: int) -> List[int]:
        """All buses owned by the zone (interior closure + own boundary)."""
        self._require_zone(zone)
        return sorted(
            b for b, z in self._bus_owner.items() if z == zone
        )

    def interior_buses(self, zone: int) -> List[int]:
        """Zone-owned buses excluding its own boundary buses."""
        own_b = set(self._own_boundary[zone])
        return [b for b in self.zone_buses(zone) if b not in own_b]

    def tie_loss_shares(self) -> Dict[int, Dict[int, float]]:
        """Per tie line: ``{owner zone: loss share}`` (D1, default 50/50)."""
        s = self._tie_loss_split
        return {
            t.line_idx: {t.zone_i: s, t.zone_j: 1.0 - s}
            for t in self.ties
        }

    def zone_ties(self, zone: int) -> List[TieLine]:
        """Ties incident to the zone."""
        self._require_zone(zone)
        return [
            t for t in self.ties if zone in (t.zone_i, t.zone_j)
        ]

    def _require_zone(self, zone: int) -> None:
        if zone not in self.zone_ids:
            raise ValueError(
                f"unknown zone {zone}; known zones: {self.zone_ids}"
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"BoundaryTopology(zones={self.zone_ids}, |B|={len(self.registry)}, "
            f"ties={[t.line_idx for t in self.ties]})"
        )
