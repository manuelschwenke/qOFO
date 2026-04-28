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
    #
    # ``E``, ``unique_bus_indices``, and ``n_unique_bus`` are pure
    # functions of ``bus_indices`` (which is immutable because the
    # dataclass is ``frozen=True``).  They were previously ``@property``
    # methods that recomputed on every access, which became a per-step
    # hot path after the per-DER H refactor (commit 65513fb): the
    # base-controller step loop reaches into ``mapping.E`` through
    # ``_expand_H_to_der_level`` every iteration, so a Python dedup +
    # dict build + ``np.zeros`` allocation ran on every call.
    #
    # We now materialise all three once in ``__post_init__`` and stash
    # them on the instance via ``object.__setattr__`` (required because
    # frozen dataclasses forbid normal attribute assignment).  The
    # public names remain the same and the ``@property`` contract is
    # preserved — callers still do ``mapping.E`` etc.

    def __post_init__(self) -> None:
        """Validate inputs and precompute derived caches."""
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

        # ---- Deduplicated bus list, preserving first-seen order ----
        seen: set[int] = set()
        unique: List[int] = []
        for b in self.bus_indices:
            if b not in seen:
                seen.add(b)
                unique.append(b)

        # ---- Incidence matrix E of shape (n_unique_bus, n_der) ----
        bus_to_row = {bus: row for row, bus in enumerate(unique)}
        E = np.zeros((len(unique), n), dtype=np.float64)
        for d, bus in enumerate(self.bus_indices):
            E[bus_to_row[bus], d] = 1.0

        # Frozen dataclass → must bypass __setattr__
        object.__setattr__(self, "_unique_bus_indices", unique)
        object.__setattr__(self, "_n_unique_bus", len(unique))
        object.__setattr__(self, "_E", E)

    @property
    def n_der(self) -> int:
        """Total number of DERs."""
        return len(self.sgen_indices)

    @property
    def unique_bus_indices(self) -> List[int]:
        """Deduplicated bus indices, preserving first-seen order.

        Cached in ``__post_init__`` — returns the same list instance on
        every call.  Do not mutate the returned list.
        """
        return self._unique_bus_indices  # type: ignore[attr-defined]

    @property
    def n_unique_bus(self) -> int:
        """Number of unique DER buses (cached)."""
        return self._n_unique_bus  # type: ignore[attr-defined]

    @property
    def E(self) -> NDArray[np.float64]:
        """
        DER-to-bus incidence matrix of shape (n_unique_bus, n_der).

        ``E[b, d] = 1`` if DER *d* is connected at unique bus *b*.
        This supports the factorisation ``H_der = H_bus @ E`` where
        ``H_bus`` has one column per unique bus.

        Cached in ``__post_init__`` — returns the same ndarray on every
        call.  Do not mutate the returned array.
        """
        return self._E  # type: ignore[attr-defined]

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
