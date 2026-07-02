"""
sensitivity/marginal_computer.py
================================
Area-local boundary marginal machinery — BME spec §3.4 (see
``docs/BME_STATUS.md``).

Symbol map (code ↔ spec)
------------------------
* ``mu(...)``            ↔ μ_j = dΦ_j/dv_b ∈ R^{|B|} (§3.4), assembled as
                           μ_j = (∂v_int,j/∂v_b)ᵀ · ∇_{v_int,j} Φ_j
                                 + ∂Φ_j/∂v_b (direct terms),
                           embedded into the global boundary registry order
                           with exactly-zero entries outside the zone's
                           adjacent boundary set (sparsity property, §3.4).
* ``response_v()``       ↔ ∂v_int,j/∂v_b — the internal voltage-magnitude
                           response to the zone's own boundary-port
                           magnitudes, from the area-internal reduced
                           Jacobian with ports held fixed (Schur block).
* ``response_full()``    ↔ ∂x_int,j/∂v_b for the full internal state
                           (θ and V) — needed by the Phase 2 loss gradient,
                           which depends on angles as well.
* ``ports``              ↔ the zone's own boundary buses (its share of B).
* ``adjacent``           ↔ support of μ_j: own ports ∪ far endpoints of the
                           zone's ties (far endpoints receive *direct*
                           contributions only, e.g. the owned tie-loss
                           share; with own ports held fixed there is no
                           internal response to a far-end voltage).

Locality note (§3.9)
--------------------
All matrix entries extracted here are mismatch-equation derivatives at the
zone's *interior* buses — they involve only branches among zone-owned buses
and the zone's own boundary ports. Although they are read from the shared
Jacobian object, the information content is area-local: no entry depends on
a non-adjacent zone's network data.

Magnitudes only (D7): ports are perturbed in voltage magnitude with the
port angle held fixed. The angle dependence of tie quantities is the
documented limitation of the scheme.

Fail-fast: a singular interior block, an unknown zone, a gradient of the
wrong length, or a direct term outside the adjacent set raises.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-02 (BME Phase 1)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from network.boundary_topology import BoundaryTopology
from sensitivity.jacobian import JacobianSensitivities
from sensitivity.index_helper import (
    get_jacobian_indices,
    get_jacobian_indices_ppc,
    get_ppc_trafo3w_branch_indices,
    pp_bus_to_ppc_bus,
)


class MarginalComputer:
    """Internal-response operators and μ assembly for one TSO zone.

    Parameters
    ----------
    sensitivities :
        The zone's Jacobian source. The extracted entries are area-local
        (see module header); the object is the shared full-network
        instance in the runner's default mode.
    topology :
        Boundary topology (registry, ownership, adjacency).
    zone_id :
        The zone this computer belongs to.
    """

    def __init__(
        self,
        sensitivities: JacobianSensitivities,
        topology: BoundaryTopology,
        zone_id: int,
    ) -> None:
        if zone_id not in topology.zone_ids:
            raise ValueError(
                f"unknown zone {zone_id}; known: {topology.zone_ids}"
            )
        self._sens = sensitivities
        self._topo = topology
        self.zone_id = int(zone_id)

        self.ports: List[int] = topology.own_boundary(zone_id)
        if not self.ports:
            raise ValueError(
                f"zone {zone_id} has no boundary ports — it cannot "
                "participate in boundary-marginal exchange."
            )
        self.adjacent: List[int] = topology.adjacent_boundary(zone_id)

        self._build_interior_block()

    # ------------------------------------------------------------------
    #  Interior block construction
    # ------------------------------------------------------------------

    def _build_interior_block(self) -> None:
        net = self._sens.net
        J = self._sens.J
        n_theta = self._sens.n_theta

        interior_pp = self._topo.interior_buses(self.zone_id)
        # Only buses present in this net (topology may have been built on
        # the plant net; the sensitivity net is its converged deep copy).
        missing = [b for b in interior_pp if b not in net.bus.index]
        if missing:
            raise ValueError(
                f"zone {self.zone_id}: interior buses {missing} not found "
                "in the sensitivity net — topology and Jacobian must be "
                "built from the same network."
            )

        rows_int: List[int] = []          # mismatch-equation rows
        cols_int: List[int] = []          # state columns
        state_labels: List[Tuple[str, int]] = []
        interior_pq_buses: List[int] = []

        def _add_bus_states(theta_idx: Optional[int], v_idx: Optional[int],
                            label_bus: int, is_aux: bool) -> None:
            if theta_idx is None and v_idx is None:
                # Voltage source strictly inside the interior (slack):
                # pinned, no states/equations — document-and-skip is
                # correct (its V is not a port of this computation).
                return
            if theta_idx is not None:
                rows_int.append(theta_idx)            # P-mismatch row
                cols_int.append(theta_idx)            # θ state column
                state_labels.append(
                    ("theta_aux" if is_aux else "theta", label_bus)
                )
            if v_idx is not None:
                rows_int.append(n_theta + v_idx)      # Q-mismatch row
                cols_int.append(n_theta + v_idx)      # V state column
                state_labels.append(
                    ("v_aux" if is_aux else "v", label_bus)
                )
                if not is_aux:
                    interior_pq_buses.append(label_bus)

        for b in interior_pp:
            theta_idx, v_idx = get_jacobian_indices(net, int(b))
            _add_bus_states(theta_idx, v_idx, int(b), is_aux=False)

        # Auxiliary star buses of zone-owned 3W transformers carry their
        # own mismatch equations and must be part of the interior block.
        if hasattr(net, "trafo3w") and len(net.trafo3w):
            for t in net.trafo3w.index:
                if not bool(net.trafo3w.at[t, "in_service"]):
                    continue
                terminals = [
                    int(net.trafo3w.at[t, c])
                    for c in ("hv_bus", "mv_bus", "lv_bus")
                ]
                owners = {self._topo.bus_owner(b) for b in terminals}
                if owners != {self.zone_id}:
                    continue
                star_ppc = self._trafo3w_star_ppc_bus(net, int(t))
                theta_idx, v_idx = get_jacobian_indices_ppc(net, star_ppc)
                _add_bus_states(theta_idx, v_idx, star_ppc, is_aux=True)

        if not rows_int:
            raise ValueError(
                f"zone {self.zone_id}: empty interior block — no interior "
                "bus contributes state variables."
            )

        rows = np.asarray(rows_int, dtype=np.int64)
        cols = np.asarray(cols_int, dtype=np.int64)
        J_int = J[np.ix_(rows, cols)]

        # Port columns: ∂g_int/∂V_port, one column per own boundary bus.
        port_cols = np.zeros((len(rows), len(self.ports)), dtype=np.float64)
        for k, p in enumerate(self.ports):
            theta_idx, v_idx = get_jacobian_indices(net, int(p))
            if v_idx is not None:
                port_cols[:, k] = J[rows, n_theta + v_idx]
            else:
                # Pinned port (slack or PV boundary bus): its magnitude is
                # an exogenous voltage-source input; the mismatch
                # derivative is assembled from the admittance matrix.
                ppc_idx = pp_bus_to_ppc_bus(net, int(p))
                dg = self._sens._compute_dg_dVgen(ppc_idx)
                port_cols[:, k] = dg[rows]

        try:
            self._R = -np.linalg.solve(J_int, port_cols)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                f"zone {self.zone_id}: interior Jacobian block is singular "
                f"({e}). Every interior island must connect to a boundary "
                "port or an internal voltage source — check the separator "
                "partition."
            )

        self._state_labels = state_labels
        self._interior_pq_buses = interior_pq_buses
        self._v_row_positions = [
            i for i, (kind, _) in enumerate(state_labels) if kind == "v"
        ]

    @staticmethod
    def _trafo3w_star_ppc_bus(net, t3w_idx: int) -> int:
        """Internal (ppc) index of a 3W transformer's star-point bus:
        the terminal shared by its HV and MV equivalent branches."""
        hv_br, mv_br, _, _ = get_ppc_trafo3w_branch_indices(net, t3w_idx)
        branch = net._ppc["branch"]
        hv_ends = {int(np.real(branch[hv_br, 0])), int(np.real(branch[hv_br, 1]))}
        mv_ends = {int(np.real(branch[mv_br, 0])), int(np.real(branch[mv_br, 1]))}
        shared = hv_ends & mv_ends
        if len(shared) != 1:
            raise ValueError(
                f"cannot identify star bus of trafo3w {t3w_idx}: "
                f"HV/MV branch endpoints {hv_ends} / {mv_ends}"
            )
        return shared.pop()

    # ------------------------------------------------------------------
    #  Public operators
    # ------------------------------------------------------------------

    @property
    def interior_pq_buses(self) -> List[int]:
        """Interior pandapower buses with a voltage state (PQ), in the
        row order of :meth:`response_v`."""
        return list(self._interior_pq_buses)

    def response_v(self) -> NDArray[np.float64]:
        """∂v_int/∂v_port ∈ R^{n_int_pq × n_ports} (rows aligned with
        :attr:`interior_pq_buses`, columns with :attr:`ports`)."""
        return self._R[self._v_row_positions, :].copy()

    def response_full(
        self,
    ) -> Tuple[NDArray[np.float64], List[Tuple[str, int]]]:
        """Full internal state response ∂x_int/∂v_port with labels
        ``("theta"|"v"|"theta_aux"|"v_aux", pp bus or ppc star bus)``.
        Needed once the loss gradient (angle-dependent) enters in
        Phase 2."""
        return self._R.copy(), list(self._state_labels)

    def mu(
        self,
        grad_v_int: NDArray[np.float64],
        grad_direct: Optional[Dict[int, float]] = None,
    ) -> NDArray[np.float64]:
        """Assemble μ_zone ∈ R^{|B|} in registry order (§3.4).

        Parameters
        ----------
        grad_v_int :
            ∇_{v_int} Φ_zone aligned with :attr:`interior_pq_buses`.
        grad_direct :
            Direct terms ∂Φ_zone/∂v_b per boundary bus (own ports and/or
            far endpoints of own ties). Keys outside the zone's adjacent
            boundary set raise — the sparsity property is enforced, not
            assumed.

        Returns
        -------
        NDArray
            μ_zone with exactly-zero entries outside
            :attr:`adjacent` (sparsity, §3.4).
        """
        grad_v_int = np.asarray(grad_v_int, dtype=np.float64)
        if grad_v_int.shape != (len(self._interior_pq_buses),):
            raise ValueError(
                f"grad_v_int has shape {grad_v_int.shape}; expected "
                f"({len(self._interior_pq_buses)},) aligned with "
                "interior_pq_buses."
            )
        out = np.zeros(len(self._topo.registry), dtype=np.float64)
        R_v = self._R[self._v_row_positions, :]
        for k, p in enumerate(self.ports):
            out[self._topo.registry_pos[p]] = float(
                R_v[:, k] @ grad_v_int
            )
        if grad_direct:
            adjacent = set(self.adjacent)
            for bus, val in grad_direct.items():
                b = int(bus)
                if b not in adjacent:
                    raise ValueError(
                        f"direct gradient term at bus {b} is outside zone "
                        f"{self.zone_id}'s adjacent boundary set "
                        f"{sorted(adjacent)} — μ must be exactly zero "
                        "there (§3.4 sparsity)."
                    )
                out[self._topo.registry_pos[b]] += float(val)
        return out
