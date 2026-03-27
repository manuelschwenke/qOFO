#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
network/zone_partition.py
=========================
Spectral clustering of a pandapower network into N geographically coherent
control zones, suitable for multi-TSO OFO decomposition.

Algorithm
---------
1. **Admittance-weighted graph**: Build adjacency matrix A where
   A[i, j] = |b_ij| (line susceptance magnitude) for each line/trafo between
   buses i and j.  This gives stronger coupling to lines with low impedance
   (i.e. short, high-capacity lines that carry more power and create tighter
   electrical coupling between buses).

2. **Normalised Laplacian**:
       L_sym = D^{-1/2} (D - A) D^{-1/2}
   where D = diag(row sums of A).  The normalised form makes cluster sizes
   more balanced than the unnormalised Laplacian.

3. **Eigenvectors**: Compute the N smallest eigenvectors of L_sym.
   For N=3 zones the relevant eigenvectors are v₁ (trivial, constant),
   v₂ (Fiedler vector, separates the network into 2 parts), and
   v₃ (further refines the partition into 3 parts).

4. **k-means**: Run k-means on the rows of V = [v₂, v₃] (one row per bus).
   Each row is a 2-D embedding of the bus in the spectral space; buses that
   are electrically close cluster together.

5. **Zone assignment**: Return two dicts — zone_id → [bus_indices] and
   bus → zone_id — for convenient lookup in the controller setup.

Interpretation for IEEE 39-bus
--------------------------------
For the IEEE 39-bus (New England) network the algorithm typically recovers
three zones matching the geographic partition of the system:

    Zone 0 (NW):  buses in the north-western area (generators G30–G33)
    Zone 1 (NE):  buses in the north-eastern area (generators G34–G36)
    Zone 2 (SE):  buses in the south-eastern area (generators G37–G39, G10)

The exact bus assignment is data-driven and reproducible given the same
network data.  Zone IDs are assigned by k-means centroid order and may
vary between runs (set ``random_state`` to fix).  The helper function
:func:`relabel_zones_by_bus_count` sorts zone IDs by ascending bus count
for deterministic labelling.

Coupling across zone boundaries
---------------------------------
The admittance matrix captures tie-line coupling.  Buses connected to
multiple zones by strong tie-lines may end up in different zones on
different runs.  The cross-sensitivity matrix H_ij in the
MultiTSOCoordinator quantifies the actual electrical coupling numerically.

Usage
-----
    from network.zone_partition import spectral_zone_partition

    zone_map, bus_zone = spectral_zone_partition(net, n_zones=3)
    # zone_map[0] → [bus_0, bus_3, bus_7, …]
    # bus_zone[5] → 1  (bus 5 belongs to zone 1)

    # Pick Zone-2 load buses for DSO feeders
    zone2_load_buses = [
        b for b in zone_map[2]
        if len(net.load[net.load["bus"] == b]) > 0
    ]

Author: Manuel Schwenke / Claude Code
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandapower as pp
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Pure-numpy k-means helper
# ---------------------------------------------------------------------------

def _kmeans_numpy(
    X: NDArray,
    k: int,
    *,
    n_init: int = 20,
    max_iter: int = 300,
    random_state: int = 42,
) -> NDArray:
    """
    Lloyd's k-means algorithm implemented in pure NumPy.

    Avoids scikit-learn / scipy DLL conflicts on Windows.  Fast enough for the
    small spectral embeddings used here (n_buses ≤ 100, k ≤ 5).

    Parameters
    ----------
    X : NDArray, shape (n, d)
        Data matrix.
    k : int
        Number of clusters.
    n_init : int
        Number of independent random restarts.
    max_iter : int
        Maximum Lloyd iterations per restart.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    labels : NDArray of int, shape (n,)
        Cluster assignment for each row of X (values in 0..k-1).
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    best_labels: NDArray | None = None
    best_inertia = np.inf

    for _ in range(n_init):
        # k-means++ initialisation: pick first centroid at random, then
        # greedily pick remaining centroids with probability ∝ distance².
        init_idx  = [int(rng.integers(n))]
        for _ in range(k - 1):
            dists = np.array([
                min(np.sum((X[i] - X[c]) ** 2) for c in init_idx)
                for i in range(n)
            ])
            dists /= dists.sum()
            init_idx.append(int(rng.choice(n, p=dists)))
        centroids = X[init_idx].copy()   # (k, d)

        labels = np.zeros(n, dtype=int)
        for _it in range(max_iter):
            # Assignment step: each point → nearest centroid
            diffs   = X[:, None, :] - centroids[None, :, :]  # (n, k, d)
            sq_dist = np.sum(diffs ** 2, axis=2)              # (n, k)
            new_labels = np.argmin(sq_dist, axis=1)           # (n,)

            if np.array_equal(new_labels, labels):
                break   # converged
            labels = new_labels

            # Update step: recompute centroids
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centroids[c] = X[mask].mean(axis=0)

        inertia = float(np.sum(
            (X - centroids[labels]) ** 2
        ))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels  = labels.copy()

    return best_labels  # type: ignore[return-value]


# ---------------------------------------------------------------------------
#  Core function
# ---------------------------------------------------------------------------

def spectral_zone_partition(
    net: pp.pandapowerNet,
    n_zones: int = 3,
    *,
    weight: str = "susceptance",
    include_trafos: bool = True,
    only_tn_buses: bool = True,
    random_state: int = 42,
    n_init: int = 20,
    verbose: bool = False,
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Partition the network buses into ``n_zones`` zones via spectral clustering.

    Parameters
    ----------
    net : pp.pandapowerNet
        Converged pandapower network.  Buses and lines must be in service.
    n_zones : int
        Number of zones (clusters).  3 is the natural choice for IEEE 39-bus.
    weight : str
        Edge weight for the graph.  ``"susceptance"`` (default) uses
        1/X (where X is series reactance in Ω); ``"uniform"`` sets all
        weights to 1 (unweighted graph).
    include_trafos : bool
        If True (default), also include 2-winding transformers as edges.
        trafo3w are included only if both buses are in the partition.
    only_tn_buses : bool
        If True (default), partition only buses tagged ``subnet = "TN"``.
        DSO buses (``subnet = "DN"``) are excluded because they are not
        independent TSO control zones.
    random_state : int
        k-means random seed for reproducibility.
    n_init : int
        Number of k-means restarts (more → more stable result).
    verbose : bool
        Print partition summary to stdout.

    Returns
    -------
    zone_map : Dict[int, List[int]]
        Maps zone_id (0, 1, …, n_zones-1) to sorted list of bus indices.
    bus_zone : Dict[int, int]
        Maps bus index to its assigned zone_id.

    Raises
    ------
    ValueError
        If fewer than n_zones buses are available for partitioning.
    """
    # ── Select buses to partition ─────────────────────────────────────────────
    if only_tn_buses and "subnet" in net.bus.columns:
        buses = sorted(
            int(b) for b in net.bus.index
            if str(net.bus.at[b, "subnet"]) == "TN"
               and bool(net.bus.at[b, "in_service"])
        )
    else:
        buses = sorted(
            int(b) for b in net.bus.index if bool(net.bus.at[b, "in_service"])
        )

    n_buses = len(buses)
    if n_buses < n_zones:
        raise ValueError(
            f"Only {n_buses} buses available but {n_zones} zones requested."
        )

    # Map bus index → position in the local adjacency matrix
    bus_to_pos: Dict[int, int] = {b: k for k, b in enumerate(buses)}
    bus_set = set(buses)

    # ── Build weighted adjacency matrix A ─────────────────────────────────────
    #
    # A[i, j] = sum of |susceptance| of all branches between bus i and bus j.
    # Using susceptance (1/X) rather than resistance (1/R) or unit weight
    # emphasises lines that have low reactance (tight coupling, short lines).
    A = np.zeros((n_buses, n_buses), dtype=np.float64)

    def _add_edge(bus_i: int, bus_j: int, w: float) -> None:
        """Add undirected weighted edge between two bus indices."""
        if bus_i in bus_to_pos and bus_j in bus_to_pos:
            pi, pj = bus_to_pos[bus_i], bus_to_pos[bus_j]
            A[pi, pj] += w
            A[pj, pi] += w

    # Lines (most branches in IEEE 39-bus)
    for li in net.line.index:
        if not net.line.at[li, "in_service"]:
            continue
        bi = int(net.line.at[li, "from_bus"])
        bj = int(net.line.at[li, "to_bus"])
        if bi not in bus_set or bj not in bus_set:
            continue

        if weight == "susceptance":
            # Compute susceptance from per-unit reactance.
            # net.line stores impedance in Ohm; convert to per-unit using
            # the base impedance Z_base = V_n^2 / S_base.
            # For graph clustering we only need relative weights, so we use
            # the raw 1/x_ohm value (consistent across all lines at 345 kV).
            x_ohm = float(net.line.at[li, "x_ohm_per_km"]) * float(net.line.at[li, "length_km"])
            w = 1.0 / max(x_ohm, 1e-6)
        else:
            w = 1.0
        _add_edge(bi, bj, w)

    # 2-winding transformers (some case39 versions include trafos)
    if include_trafos and not net.trafo.empty:
        for ti in net.trafo.index:
            if not net.trafo.at[ti, "in_service"]:
                continue
            bi = int(net.trafo.at[ti, "hv_bus"])
            bj = int(net.trafo.at[ti, "lv_bus"])
            if bi not in bus_set or bj not in bus_set:
                continue
            if weight == "susceptance":
                # Use 1/vk (short-circuit impedance) as proxy for susceptance
                vk = float(net.trafo.at[ti, "vk_percent"]) / 100.0
                w = 1.0 / max(vk, 1e-4)
            else:
                w = 1.0
            _add_edge(bi, bj, w)

    # ── Normalised Laplacian L_sym = D^{-1/2} (D - A) D^{-1/2} ──────────────
    d = A.sum(axis=1)                         # degree vector
    d_safe = np.maximum(d, 1e-12)             # avoid division by zero
    d_inv_sqrt = 1.0 / np.sqrt(d_safe)

    # D^{-1/2} (D - A) D^{-1/2} = I - D^{-1/2} A D^{-1/2}
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(n_buses) - (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]

    # ── Spectral decomposition ────────────────────────────────────────────────
    #
    # We need the n_zones smallest eigenvectors of the normalised Laplacian.
    # np.linalg.eigh (symmetric solver) calls LAPACK dsyev which can crash with
    # an MKL DLL conflict in some conda environments on Windows.  We therefore
    # use the general np.linalg.eig and sort the real parts of the eigenvalues
    # manually — L_sym is real-symmetric so all eigenvalues are real and the
    # imaginary parts are numerical noise.
    raw_vals, raw_vecs = np.linalg.eig(L_sym)
    # Take real parts (imaginary parts are at machine-epsilon level)
    raw_vals = raw_vals.real
    raw_vecs = raw_vecs.real
    # Sort eigenvectors by ascending eigenvalue
    sort_idx    = np.argsort(raw_vals)
    eigenvalues = raw_vals[sort_idx]
    eigenvectors = raw_vecs[:, sort_idx]   # columns are eigenvectors

    # eigenvectors[:, k] is the k-th eigenvector (ascending eigenvalue order)
    # Skip the first eigenvector (eigenvalue ≈ 0, trivial all-ones vector).
    # Use eigenvectors 1 … n_zones-1 for the embedding.
    V = eigenvectors[:, 1:n_zones]       # shape (n_buses, n_zones - 1)

    if verbose:
        print(f"[zone_partition] Smallest {n_zones} eigenvalues of L_sym:")
        for k in range(min(n_zones + 1, len(eigenvalues))):
            print(f"  λ_{k} = {eigenvalues[k]:.6f}")

    # ── k-means clustering in spectral space ──────────────────────────────────
    #
    # Each bus is represented by its coordinates in the (n_zones-1)-dimensional
    # spectral embedding V.  Buses that are electrically close have similar
    # Fiedler-vector components and cluster naturally.
    #
    # We use a pure-numpy Lloyd's algorithm to avoid third-party clustering
    # library conflicts (scikit-learn / scipy DLL issues on Windows).
    # The outer loop repeats n_init random initialisations and keeps the best
    # result (lowest within-cluster sum of squares) for stability.
    labels: NDArray = _kmeans_numpy(V, n_zones, n_init=n_init, random_state=random_state)

    # ── Build output dicts ────────────────────────────────────────────────────
    zone_map: Dict[int, List[int]] = {z: [] for z in range(n_zones)}
    bus_zone: Dict[int, int] = {}

    for k, bus in enumerate(buses):
        zone_id = int(labels[k])
        zone_map[zone_id].append(bus)
        bus_zone[bus] = zone_id

    for z in range(n_zones):
        zone_map[z].sort()

    if verbose:
        print(f"[zone_partition] Zone assignment ({n_zones} zones):")
        for z in range(n_zones):
            print(f"  Zone {z}: {len(zone_map[z])} buses → {zone_map[z]}")

    return zone_map, bus_zone


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def relabel_zones_by_generator_count(
    zone_map: Dict[int, List[int]],
    bus_zone: Dict[int, int],
    gen_bus_indices: List[int],
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Relabel zones in descending order of generator count.

    Zone 0 → most generators (likely the slack area).
    Zone 2 → fewest generators (chosen for DSO feeder replacement).

    This gives a deterministic labelling independent of k-means initialisation.

    Parameters
    ----------
    zone_map : dict
        Output of :func:`spectral_zone_partition`.
    bus_zone : dict
        Output of :func:`spectral_zone_partition`.
    gen_bus_indices : list[int]
        Bus indices of all generators (from ``IEEE39NetworkMeta.gen_bus_indices``).

    Returns
    -------
    zone_map, bus_zone : relabelled dicts with the same structure.
    """
    gen_set = set(gen_bus_indices)
    # Count generators per zone
    gen_count = {
        z: sum(1 for b in buses if b in gen_set)
        for z, buses in zone_map.items()
    }
    # Sort zones by descending generator count (Zone 0 = most generators)
    old_to_new = {
        old_z: new_z
        for new_z, (old_z, _)
        in enumerate(sorted(gen_count.items(), key=lambda kv: -kv[1]))
    }
    new_zone_map = {old_to_new[z]: sorted(buses) for z, buses in zone_map.items()}
    new_bus_zone = {b: old_to_new[z] for b, z in bus_zone.items()}
    return new_zone_map, new_bus_zone


def get_zone_lines(
    net: pp.pandapowerNet,
    zone_bus_set: set,
) -> List[int]:
    """
    Return line indices whose BOTH endpoints lie within ``zone_bus_set``.

    Lines with exactly one endpoint in the zone are "tie-lines" and are
    intentionally excluded; they are the source of cross-zone coupling and
    belong to neither zone's internal network.

    Parameters
    ----------
    net : pandapowerNet
    zone_bus_set : set of int
        Bus indices belonging to the zone.

    Returns
    -------
    List of line indices (int).
    """
    lines = []
    for li in net.line.index:
        if not net.line.at[li, "in_service"]:
            continue
        fb = int(net.line.at[li, "from_bus"])
        tb = int(net.line.at[li, "to_bus"])
        if fb in zone_bus_set and tb in zone_bus_set:
            lines.append(int(li))
    return sorted(lines)


def get_tie_lines(
    net: pp.pandapowerNet,
    zone_a_buses: set,
    zone_b_buses: set,
) -> List[int]:
    """
    Return line indices that cross the boundary between zone A and zone B.

    These lines are the physical source of the off-diagonal coupling blocks
    H_ij in the multi-TSO formulation.

    Parameters
    ----------
    net : pandapowerNet
    zone_a_buses : set of int
    zone_b_buses : set of int

    Returns
    -------
    List of tie-line indices.
    """
    ties = []
    for li in net.line.index:
        if not net.line.at[li, "in_service"]:
            continue
        fb = int(net.line.at[li, "from_bus"])
        tb = int(net.line.at[li, "to_bus"])
        a_to_b = (fb in zone_a_buses and tb in zone_b_buses)
        b_to_a = (tb in zone_a_buses and fb in zone_b_buses)
        if a_to_b or b_to_a:
            ties.append(int(li))
    return sorted(ties)
