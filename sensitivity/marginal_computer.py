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
* ``mu_x(...)``          ↔ μ_j assembled from a gradient over the FULL
                           internal state (θ and V, aux states included) —
                           required once the loss part of Φ_j (angle
                           dependent) enters in Phase 2.
* ``frozen_input_response`` /
  ``response_to_*``      ↔ ∂x_int,j/∂u_j |_{v_b fixed} (§3.5 Convention A):
                           the zone-internal state response to the zone's
                           own inputs with ALL boundary voltages held
                           fixed, Δx_int = -J_int⁻¹ · ∂g_int/∂u. This is
                           the "port-frozen" operator behind
                           g_j^own = ∂Φ_j/∂u_j |_{v_b fixed}.
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


def dg_dtau_2w_tolerant(
    sens: JacobianSensitivities, trafo_idx: int
) -> Tuple[NDArray[np.float64], float]:
    """Mismatch derivative ∂g/∂τ for a 2W transformer tap, tolerating a
    terminal at the reference bus (mirrors the accumulate-only-existing-
    rows behaviour of ``compute_dV_ds_2w``; the existing
    ``_compute_dg_dtau_2w`` helper raises there, which the slack machine
    trafo — an ordinary OLTC actuator in the runner's ZoneDefinition —
    must not).

    Returns ``(dg_dtau, delta_tau)``; the state response per tap step is
    ``Δx = -J⁻¹ · dg_dtau · Δτ``.
    """
    net = sens.net
    if trafo_idx not in net.trafo.index:
        raise ValueError(f"Transformer {trafo_idx} not found in network.")
    from sensitivity.index_helper import get_ppc_trafo_index
    ppc_br = get_ppc_trafo_index(net, trafo_idx)
    if ppc_br is None:
        raise ValueError(
            f"Could not find pypower branch index for transformer "
            f"{trafo_idx}."
        )
    hv_bus = net.trafo.at[trafo_idx, "hv_bus"]
    lv_bus = net.trafo.at[trafo_idx, "lv_bus"]
    V_i = net.res_bus.at[hv_bus, "vm_pu"]
    V_j = net.res_bus.at[lv_bus, "vm_pu"]
    theta = (
        np.deg2rad(net.res_bus.at[hv_bus, "va_degree"])
        - np.deg2rad(net.res_bus.at[lv_bus, "va_degree"])
    )
    s0 = net.trafo.at[trafo_idx, "tap_pos"]
    delta_tau = net.trafo.at[trafo_idx, "tap_step_percent"] / 100.0
    tau = 1.0 + s0 * delta_tau
    y_pu = 1.0 / complex(
        net._ppc["branch"][ppc_br, 2], net._ppc["branch"][ppc_br, 3]
    )
    g, b = y_pu.real, y_pu.imag

    theta_i_idx, v_i_idx = get_jacobian_indices(net, hv_bus)
    theta_j_idx, v_j_idx = get_jacobian_indices(net, lv_bus)
    if theta_i_idx is None and theta_j_idx is None:
        raise ValueError(
            f"trafo {trafo_idx}: neither terminal carries a Jacobian "
            "state — no tap response can be formed."
        )

    dg = np.zeros(sens.x_size, dtype=np.float64)
    dPi = (V_i * V_j * (g * np.cos(theta) + b * np.sin(theta)) / tau**2
           - 2 * g * V_i**2 / tau**3)
    dPj = V_j * V_i * (g * np.cos(theta) - b * np.sin(theta)) / tau**2
    dQi = (V_i * V_j * (g * np.sin(theta) - b * np.cos(theta)) / tau**2
           + 2 * b * V_i**2 / tau**3)
    dQj = V_j * V_i * (-g * np.sin(theta) - b * np.cos(theta)) / tau**2
    if theta_i_idx is not None:
        dg[theta_i_idx] += dPi
    if theta_j_idx is not None:
        dg[theta_j_idx] += dPj
    if v_i_idx is not None:
        dg[sens.n_theta + v_i_idx] += dQi
    if v_j_idx is not None and (sens.n_theta + v_j_idx) < sens.x_size:
        dg[sens.n_theta + v_j_idx] += dQj
    return dg, float(delta_tau)


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
        if not self.ports and topology.registry:
            # A portless zone in a MULTI-zone topology is an isolated
            # area — an error. The single-area case (empty registry) is
            # the legitimate degenerate mode: nothing is frozen, the
            # "port-frozen" operators become TOTAL-response operators and
            # μ is the empty vector (spec §3.5 single-area identity).
            raise ValueError(
                f"zone {zone_id} has no boundary ports although the "
                "topology has boundary buses — it cannot participate in "
                "boundary-marginal exchange."
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

        # Port columns: ∂g_int/∂V_port and ∂g_int/∂θ_port, one column per
        # own boundary bus and coordinate (complex boundary — D7 REVISED
        # 2026-07-02: the Phase 4 identity test demonstrated that the
        # magnitude-only channel misses the boundary-angle term of the
        # loss objective; Manuel's pre-authorised fallback applies).
        port_cols = np.zeros((len(rows), len(self.ports)), dtype=np.float64)
        port_cols_th = np.zeros(
            (len(rows), len(self.ports)), dtype=np.float64
        )
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
            if theta_idx is not None:
                port_cols_th[:, k] = J[rows, theta_idx]
            else:
                raise ValueError(
                    f"zone {self.zone_id}: boundary port {p} is the "
                    "reference bus — its angle has no mismatch "
                    "derivative column; extend the port machinery "
                    "before using such a partition."
                )

        try:
            self._R = -np.linalg.solve(J_int, port_cols)
            self._R_th = -np.linalg.solve(J_int, port_cols_th)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                f"zone {self.zone_id}: interior Jacobian block is singular "
                f"({e}). Every interior island must connect to a boundary "
                "port or an internal voltage source — check the separator "
                "partition."
            )

        # Kept for the port-frozen input responses (§3.5 Convention A):
        # Δx_int = -J_int⁻¹ · ∂g_int/∂u, with ∂g_int/∂u the rows_int slice
        # of the full mismatch-derivative vector.
        self._J_int = J_int
        self._rows_int = rows

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
    def sens(self) -> JacobianSensitivities:
        """The zone's Jacobian source (shared full-network instance in
        the runner's default mode; extracted entries are area-local)."""
        return self._sens

    @property
    def topology(self) -> BoundaryTopology:
        return self._topo

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
        grad_x = np.zeros(self._R.shape[0], dtype=np.float64)
        grad_x[self._v_row_positions] = grad_v_int
        return self.mu_x(grad_x, grad_direct)

    def mu_x(
        self,
        grad_x_int: NDArray[np.float64],
        grad_direct: Optional[Dict[int, float]] = None,
    ) -> NDArray[np.float64]:
        """Assemble μ_zone ∈ R^{|B|} from a gradient over the FULL
        internal state (§3.4, angle-dependent Φ terms included).

        Parameters
        ----------
        grad_x_int :
            ∇_{x_int} Φ_zone aligned with the state labels of
            :meth:`response_full` (θ in rad, V in pu; aux star states
            included).
        grad_direct :
            Direct terms ∂Φ_zone/∂v_b per boundary bus, as in :meth:`mu`.

        Returns
        -------
        NDArray
            μ_zone with exactly-zero entries outside :attr:`adjacent`
            (sparsity, §3.4).
        """
        return self._chain_ports(self._R, grad_x_int, grad_direct)

    def mu_x_stacked(
        self,
        grad_x_int: NDArray[np.float64],
        grad_direct_v: Optional[Dict[int, float]] = None,
        grad_direct_theta: Optional[Dict[int, float]] = None,
    ) -> NDArray[np.float64]:
        """Complex-boundary marginal μ_zone ∈ R^{2|B|} in the stacked
        coordinate order ``[dΦ/dVm_b (registry) | dΦ/dθ_b (registry)]``
        (D7 revision: the exchanged signal carries BOTH channels; the
        magnitude-only :meth:`mu_x` remains for diagnostics and the
        Phase 1/2 test oracles).
        """
        v_part = self._chain_ports(self._R, grad_x_int, grad_direct_v)
        th_part = self._chain_ports(
            self._R_th, grad_x_int, grad_direct_theta
        )
        return np.concatenate([v_part, th_part])

    def _chain_ports(
        self,
        R_block: NDArray[np.float64],
        grad_x_int: NDArray[np.float64],
        grad_direct: Optional[Dict[int, float]],
    ) -> NDArray[np.float64]:
        grad_x_int = np.asarray(grad_x_int, dtype=np.float64)
        if grad_x_int.shape != (R_block.shape[0],):
            raise ValueError(
                f"grad_x_int has shape {grad_x_int.shape}; expected "
                f"({R_block.shape[0]},) aligned with the state labels of "
                "response_full()."
            )
        out = np.zeros(len(self._topo.registry), dtype=np.float64)
        for k, p in enumerate(self.ports):
            out[self._topo.registry_pos[p]] = float(
                R_block[:, k] @ grad_x_int
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

    # ------------------------------------------------------------------
    #  Port-frozen input responses (§3.5 Convention A)
    # ------------------------------------------------------------------

    def frozen_input_response(
        self, dg_full: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Δx_int = -J_int⁻¹ · ∂g_int/∂u for a full-length mismatch
        derivative vector ``∂g/∂u`` (Jacobian state ordering
        ``[P_PV, P_PQ, Q_PQ]``); rows outside the zone's interior block
        are dropped — that is exactly the "boundary voltages held fixed"
        operation (the pinned ports absorb their own mismatch rows).

        Returns the internal state response aligned with the state
        labels of :meth:`response_full`.
        """
        dg_full = np.asarray(dg_full, dtype=np.float64)
        if dg_full.shape != (self._sens.x_size,):
            raise ValueError(
                f"dg_full has shape {dg_full.shape}; expected "
                f"({self._sens.x_size},) in Jacobian state ordering."
            )
        return -np.linalg.solve(self._J_int, dg_full[self._rows_int])

    def _require_owned_bus(self, bus: int, what: str) -> None:
        owner = self._topo.bus_owner(int(bus))
        if owner != self.zone_id:
            raise ValueError(
                f"{what} at bus {bus} belongs to zone {owner}, not zone "
                f"{self.zone_id} — a zone may only evaluate port-frozen "
                "responses of its OWN actuators (§3.9 locality)."
            )

    def response_to_q_injection(self, bus: int) -> NDArray[np.float64]:
        """Port-frozen state response to a reactive power injection at a
        zone-owned bus, per Mvar (generator convention: Q > 0 injects).

        A boundary-port bus yields the exact zero response (its voltage
        is held fixed; the effect travels through H_{b,i} and the price
        term instead)."""
        self._require_owned_bus(bus, "Q injection")
        net = self._sens.net
        _, v_idx = get_jacobian_indices(net, int(bus))
        if v_idx is None:
            raise ValueError(
                f"Q injection at bus {bus}: the bus has no voltage state "
                "in the Jacobian (PV or slack bus) — a reactive injection "
                "there is absorbed by the local voltage source."
            )
        dg = np.zeros(self._sens.x_size, dtype=np.float64)
        # g = S_calc - S_inj: +1 Mvar injection lowers the Q-mismatch row.
        dg[self._sens.n_theta + v_idx] = -1.0 / float(net.sn_mva)
        return self.frozen_input_response(dg)

    def response_to_vgen(self, gen_terminal_bus: int) -> NDArray[np.float64]:
        """Port-frozen state response to a pinned-bus voltage magnitude
        setpoint at a zone-owned generator terminal bus, per pu.

        Covers PV buses AND the reference bus: the runner's
        ``ZoneDefinition`` includes the slack machine's AVR setpoint as
        an ordinary actuator, and the slack magnitude is an exogenous
        power-flow input whose mismatch derivative ∂g/∂V_ref is
        well-defined (the Phase 1 "no Jacobian column at the reference
        bus" note concerned the missing STATE column, not this input
        channel)."""
        self._require_owned_bus(gen_terminal_bus, "V_gen setpoint")
        net = self._sens.net
        _, v_idx = get_jacobian_indices(net, int(gen_terminal_bus))
        if v_idx is not None:
            raise ValueError(
                f"V_gen at bus {gen_terminal_bus}: the bus has a voltage "
                "state (PQ) — not a pinned generator bus."
            )
        ppc_bus = pp_bus_to_ppc_bus(net, int(gen_terminal_bus))
        dg = self._sens._compute_dg_dVgen(ppc_bus)
        return self.frozen_input_response(dg)

    def response_to_tap_2w(self, trafo_idx: int) -> NDArray[np.float64]:
        """Port-frozen state response to a two-winding transformer tap
        step at a zone-owned transformer, per whole tap step."""
        net = self._sens.net
        if trafo_idx not in net.trafo.index:
            raise ValueError(f"trafo {trafo_idx} not in net.trafo")
        side = str(net.trafo.at[trafo_idx, "tap_side"])
        if side != "hv":
            raise ValueError(
                f"trafo {trafo_idx}: tap_side '{side}' is not supported — "
                "the ∂g/∂τ assembly (mirroring compute_dV_ds_2w) assumes "
                "an hv-side tap ratio."
            )
        for c in ("hv_bus", "lv_bus"):
            self._require_owned_bus(
                int(net.trafo.at[trafo_idx, c]), f"OLTC trafo {trafo_idx}"
            )
        # Tolerant assembly: the slack machine trafo (lv terminal at the
        # reference bus) is an ordinary OLTC actuator in the runner's
        # ZoneDefinition; the existing _compute_dg_dtau_2w raises there.
        dg, delta_tau = dg_dtau_2w_tolerant(self._sens, int(trafo_idx))
        return self.frozen_input_response(dg) * delta_tau
