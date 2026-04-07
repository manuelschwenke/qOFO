"""
DER Mapping Module
==================

Maps individual DERs (pandapower sgen elements) to their bus locations,
enabling per-DER control variables while factoring the sensitivity
matrix as H_der = H_bus @ E for full-rank stability analysis.

The incidence matrix E has shape (n_unique_bus, n_der) with E[b,d] = 1
if DER d is connected at unique bus b.  This factorisation ensures that
H_bus retains full column rank (one column per unique bus) while the
MIQP can operate with per-DER decision variables and individual weights.

Author: Manuel Schwenke
Date: 2026-04-07
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DERMapping:
    """
    Maps individual DERs (sgens) to their bus locations.

    This class supports per-DER modelling where multiple DERs may share
    a single bus.  It provides:

    - The DER-to-bus incidence matrix ``E`` for the factorisation
      ``H_der = H_bus @ E``.
    - Per-DER cost weights for constructing a non-uniform diagonal
      ``G_w`` in the MIQP objective.
    - Convenience methods for aggregation (DER -> bus) and
      disaggregation (bus -> DER).

    Attributes
    ----------
    sgen_indices : Tuple[int, ...]
        Pandapower sgen indices, one per DER.
    bus_indices : Tuple[int, ...]
        Bus index for each DER (may contain duplicates when multiple
        DERs share a bus).
    s_rated_mva : NDArray[np.float64]
        Per-DER rated apparent power [MVA].
    p_max_mw : NDArray[np.float64]
        Per-DER maximum active power (installed capacity) [MW].
    weights : NDArray[np.float64]
        Per-DER cost / priority weights for the MIQP objective.
        Higher weight means higher cost of using that DER's reactive
        power.  Default is uniform (all ones).
    """

    sgen_indices: Tuple[int, ...]
    bus_indices: Tuple[int, ...]
    s_rated_mva: NDArray[np.float64]
    p_max_mw: NDArray[np.float64]
    weights: NDArray[np.float64]

    # ------------------------------------------------------------------
    #  Derived (cached) properties
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate inputs."""
        n = len(self.sgen_indices)
        if len(self.bus_indices) != n:
            raise ValueError(
                f"bus_indices length ({len(self.bus_indices)}) must match "
                f"sgen_indices length ({n})"
            )
        if len(self.s_rated_mva) != n:
            raise ValueError(
                f"s_rated_mva length ({len(self.s_rated_mva)}) must match "
                f"sgen_indices length ({n})"
            )
        if len(self.p_max_mw) != n:
            raise ValueError(
                f"p_max_mw length ({len(self.p_max_mw)}) must match "
                f"sgen_indices length ({n})"
            )
        if len(self.weights) != n:
            raise ValueError(
                f"weights length ({len(self.weights)}) must match "
                f"sgen_indices length ({n})"
            )

    @property
    def n_der(self) -> int:
        """Total number of DERs."""
        return len(self.sgen_indices)

    @property
    def unique_bus_indices(self) -> List[int]:
        """Deduplicated bus indices, preserving first-seen order."""
        seen: set[int] = set()
        result: List[int] = []
        for b in self.bus_indices:
            if b not in seen:
                seen.add(b)
                result.append(b)
        return result

    @property
    def n_unique_bus(self) -> int:
        """Number of unique DER buses."""
        return len(self.unique_bus_indices)

    @property
    def E(self) -> NDArray[np.float64]:
        """
        DER-to-bus incidence matrix of shape (n_unique_bus, n_der).

        ``E[b, d] = 1`` if DER *d* is connected at unique bus *b*.
        This supports the factorisation ``H_der = H_bus @ E`` where
        ``H_bus`` has one column per unique bus.
        """
        unique = self.unique_bus_indices
        bus_to_row = {bus: row for row, bus in enumerate(unique)}
        E = np.zeros((len(unique), self.n_der), dtype=np.float64)
        for d, bus in enumerate(self.bus_indices):
            E[bus_to_row[bus], d] = 1.0
        return E

    def bus_to_der_indices(self, bus: int) -> List[int]:
        """Return list of DER indices (position in sgen_indices) at *bus*."""
        return [d for d, b in enumerate(self.bus_indices) if b == bus]

    # ------------------------------------------------------------------
    #  Aggregation / disaggregation helpers
    # ------------------------------------------------------------------

    def aggregate_to_bus(
        self, u_der: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Aggregate per-DER values to per-unique-bus by summation.

        Parameters
        ----------
        u_der : NDArray[np.float64]
            Per-DER values of length ``n_der``.

        Returns
        -------
        u_bus : NDArray[np.float64]
            Per-unique-bus values of length ``n_unique_bus``.
        """
        return self.E @ u_der

    def disaggregate_to_der(
        self,
        u_bus: NDArray[np.float64],
        method: str = "capacity_weighted",
    ) -> NDArray[np.float64]:
        """
        Distribute per-bus values to individual DERs.

        Parameters
        ----------
        u_bus : NDArray[np.float64]
            Per-unique-bus values of length ``n_unique_bus``.
        method : str
            ``'capacity_weighted'`` (default): distribute proportional
            to ``s_rated_mva``.  ``'equal'``: distribute evenly.

        Returns
        -------
        u_der : NDArray[np.float64]
            Per-DER values of length ``n_der``.
        """
        unique = self.unique_bus_indices
        bus_to_row = {bus: row for row, bus in enumerate(unique)}
        u_der = np.zeros(self.n_der, dtype=np.float64)

        for row, bus in enumerate(unique):
            der_ids = self.bus_to_der_indices(bus)
            if not der_ids:
                continue

            bus_val = u_bus[row]

            if method == "capacity_weighted":
                caps = np.array(
                    [self.s_rated_mva[d] for d in der_ids],
                    dtype=np.float64,
                )
                total_cap = caps.sum()
                if total_cap > 0:
                    for d, cap in zip(der_ids, caps):
                        u_der[d] = bus_val * cap / total_cap
                else:
                    # Fallback to equal distribution
                    for d in der_ids:
                        u_der[d] = bus_val / len(der_ids)
            else:  # equal
                for d in der_ids:
                    u_der[d] = bus_val / len(der_ids)

        return u_der

    def get_bus_aggregated_s_rated(self) -> NDArray[np.float64]:
        """Sum of s_rated_mva per unique bus."""
        unique = self.unique_bus_indices
        result = np.zeros(len(unique), dtype=np.float64)
        for d, bus in enumerate(self.bus_indices):
            row = unique.index(bus)
            result[row] += self.s_rated_mva[d]
        return result

    def get_bus_aggregated_p_max(self) -> NDArray[np.float64]:
        """Sum of p_max_mw per unique bus."""
        unique = self.unique_bus_indices
        result = np.zeros(len(unique), dtype=np.float64)
        for d, bus in enumerate(self.bus_indices):
            row = unique.index(bus)
            result[row] += self.p_max_mw[d]
        return result
