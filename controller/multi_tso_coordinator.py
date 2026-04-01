#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
controller/multi_tso_coordinator.py
====================================
MultiTSOCoordinator: Orchestrates N decentralised TSOControllers for a
multi-zone transmission network and computes cross-zone coupling matrices
for stability analysis.

Architecture overview
---------------------
The system is divided into N TSO zones (e.g. N=3 for the IEEE 39-bus).
Each zone has its own :class:`~controller.tso_controller.TSOController`.
Coordination is DECENTRALISED: every zone solves its MIQP independently
using only its local sensitivity matrix H_ii.  The coordinator additionally
computes the off-diagonal (cross-zone) sensitivity blocks H_ij and assembles
the full preconditioned system matrix M_sys for post-hoc stability analysis.

Mathematical notation (from Schwenke / CIGRE 2026)
----------------------------------------------------
For zone i with control variables u_i, outputs y_i = H_ii u_i + Σ_{j≠i} H_ij u_j:

    Local curvature:        C_ii = H_ii^T  Q_obj,i  H_ii
    Cross curvature:        C_ij = H_ii^T  Q_obj,i  H_ij

    Preconditioned blocks:
        M_TSO,ii = G_w,i^{-½}  C_ii  G_w,i^{-½}    (diagonal block)
        M_TSO,ij = G_w,i^{-½}  C_ij  G_w,j^{-½}    (off-diagonal block)

    Full system matrix:
        M_sys = block_matrix([[M_TSO,ij]])           (symmetric, NNU × NNU)

    Contraction condition per zone (diagonal-dominance criterion):
        0 < α_i · (λ_max(M_TSO,ii) + Σ_{j≠i} ||M_TSO,ij||₂) < 2

If this holds for all i, the decentralised iteration converges globally.
See :mod:`analysis.multi_zone_stability` for the full analysis.

Row / column ordering in H
--------------------------
The TSOController assembles its sensitivity matrix with columns ordered as:

    u_i = [ Q_DER_i  |  Q_PCC_set_i  |  V_gen_i  |  s_OLTC_i  |  s_shunt_i ]

and rows ordered as:

    y_i = [ V_bus_i  |  I_line_i ]

The coordinator replicates this ordering when assembling the cross-sensitivity
blocks so that the column indexing is consistent with the per-zone MIQP.

Usage
-----
    from controller.multi_tso_coordinator import (
        ZoneDefinition, MultiTSOCoordinator
    )

    zone0 = ZoneDefinition(zone_id=0, ...)
    zone1 = ZoneDefinition(zone_id=1, ...)
    zone2 = ZoneDefinition(zone_id=2, ...)  # has DSO

    coordinator = MultiTSOCoordinator(
        zones=[zone0, zone1, zone2],
        net=net,
        verbose=1,
    )

    # After setting up each zone's TSOController externally:
    coordinator.register_tso_controller(0, tso_z0)
    coordinator.register_tso_controller(1, tso_z1)
    coordinator.register_tso_controller(2, tso_z2)

    # Each simulation step:
    tso_outputs = coordinator.step(measurements, step_index)
    coupling_info = coordinator.last_coupling_diagnostics

Author: Manuel Schwenke / Claude Code
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandapower as pp
from numpy.typing import NDArray

from controller.base_controller import OFOParameters
from controller.tso_controller import TSOController, TSOControllerConfig
from core.measurement import Measurement
from sensitivity.jacobian import JacobianSensitivities


# ---------------------------------------------------------------------------
#  ZoneDefinition — describes one TSO zone's topology and actuators
# ---------------------------------------------------------------------------

@dataclass
class ZoneDefinition:
    """
    All index sets and tuning parameters for one TSO zone.

    A ZoneDefinition is created once during the setup phase (after spectral
    zone partitioning) and then passed to the MultiTSOCoordinator.  It is
    immutable after creation; changes to the network topology require
    rebuilding the definition.

    Fields
    ------
    zone_id : int
        Integer zone label (0, 1, 2, …).

    bus_indices : List[int]
        All pandapower bus indices belonging to this zone.

    gen_indices : List[int]
        ``net.gen`` row indices of generators inside this zone.
        The controller issues AVR voltage setpoints for these.

    gen_bus_indices : List[int]
        Terminal bus index for each generator (same order as gen_indices).
        For IEEE 39-bus without machine trafos this equals the generator bus.

    tso_der_indices : List[int]
        ``net.sgen`` indices of TSO-DER sgens in this zone.
        These are the primary continuous Q actuators.

    tso_der_buses : List[int]
        Bus index for each TSO DER (same order as tso_der_indices).

    v_bus_indices : List[int]
        Buses whose voltage is MONITORED and penalised in the objective.
        Typically all zone buses (or only a critical subset).

    line_indices : List[int]
        Lines fully inside this zone whose current is monitored.

    pcc_trafo_indices : List[int]
        2W or 3W transformer indices connecting to DSO areas (Zone 2 only).
        Empty for zones without a subordinate DSO.

    pcc_dso_ids : List[str]
        Controller IDs of the DSOs attached at each PCC (same order).

    shunt_bus_indices, shunt_q_steps_mvar : List
        Switchable shunts inside the zone (may be empty).

    v_setpoint_pu : float
        Nominal voltage setpoint for all monitored buses.

    alpha : float
        OFO step-size gain α_i for this zone.

    g_v : float
        Voltage tracking weight in the objective (g_v in Q_obj diagonal).

    g_w_der : float
        Regularisation penalty on DER Q changes (g_w for Q_DER columns).

    g_w_gen : float
        Regularisation penalty on generator AVR changes.

    g_w_pcc : float
        Regularisation penalty on PCC setpoint changes (Zone 2 only).

    oltc_trafo_indices : List[int]
        Machine-transformer OLTC indices in ``net.trafo`` for this zone.

    g_w_oltc : float
        Regularisation penalty on machine-transformer OLTC tap changes.
    """
    zone_id:            int
    bus_indices:        List[int]
    gen_indices:        List[int]
    gen_bus_indices:    List[int]
    tso_der_indices:    List[int]
    tso_der_buses:      List[int]
    v_bus_indices:      List[int]
    line_indices:       List[int]
    line_max_i_ka:      List[float]
    pcc_trafo_indices:  List[int]   = field(default_factory=list)
    pcc_dso_ids:        List[str]   = field(default_factory=list)
    shunt_bus_indices:  List[int]   = field(default_factory=list)
    shunt_q_steps_mvar: List[float] = field(default_factory=list)
    oltc_trafo_indices: List[int]   = field(default_factory=list)
    v_setpoint_pu:      float = 1.02
    v_min_pu:           float = 0.95
    v_max_pu:           float = 1.05
    alpha:              float = 1.0
    g_v:                float = 1.0
    g_w_der:            float = 100.0
    g_w_gen:            float = 5e8
    g_w_pcc:            float = 100.0
    g_w_oltc:           float = 10.0

    def n_controls(self) -> int:
        """Total number of control variables for this zone."""
        return (
            len(self.tso_der_indices)      # Q_DER
            + len(self.pcc_trafo_indices)   # Q_PCC_set
            + len(self.gen_indices)         # V_gen
            + len(self.oltc_trafo_indices)  # s_OLTC (machine trafos)
        )

    def n_outputs(self) -> int:
        """Total number of output variables (rows in H_ii)."""
        return len(self.v_bus_indices) + len(self.line_indices)

    def q_obj_diagonal(self) -> NDArray[np.float64]:
        """
        Per-output objective weight vector Q_obj for this zone.

        Row ordering: [V_bus (g_v each) | I_line (0 — constraint only)].
        Current rows get weight 0 because currents are treated as hard
        constraints in the MIQP, not tracked in the objective.
        """
        n_v = len(self.v_bus_indices)
        n_i = len(self.line_indices)
        return np.concatenate([
            np.full(n_v, self.g_v),
            np.zeros(n_i),
        ])

    def gw_diagonal(self) -> NDArray[np.float64]:
        """
        Regularisation weight vector G_w (diagonal of G_w matrix) for this zone.

        Column ordering: [Q_DER | Q_PCC_set | V_gen | s_OLTC].
        """
        n_der = len(self.tso_der_indices)
        n_pcc = len(self.pcc_trafo_indices)
        n_gen = len(self.gen_indices)
        n_oltc = len(self.oltc_trafo_indices)
        return np.concatenate([
            np.full(n_der, self.g_w_der),
            np.full(n_pcc, self.g_w_pcc),
            np.full(n_gen, self.g_w_gen),
            np.full(n_oltc, self.g_w_oltc),
        ])


# ---------------------------------------------------------------------------
#  MultiTSOCoordinator
# ---------------------------------------------------------------------------

class MultiTSOCoordinator:
    """
    Orchestrates N decentralised TSOControllers and computes cross-zone coupling.

    The coordinator owns:
    * A list of :class:`ZoneDefinition` objects (one per zone).
    * A dict mapping zone_id → :class:`~controller.tso_controller.TSOController`.
    * The full-network :class:`~sensitivity.jacobian.JacobianSensitivities`
      object used to compute cross-sensitivity blocks H_ij.

    On each TSO step, the coordinator:
    1. Calls each zone's TSOController.step() independently (decentralised).
    2. Recomputes H_ij after the power-flow update (if requested).
    3. Assembles M_TSO,ii and M_TSO,ij blocks.
    4. Checks the diagonal-dominance contraction condition per zone.
    5. Returns all per-zone outputs and coupling diagnostics.

    Cross-sensitivity computation
    ------------------------------
    H_ij (zone i outputs, zone j inputs) is assembled column-by-column from:

    * DER Q columns:  ``∂V_i / ∂Q_DER_j``
          via ``compute_dV_dQ_der(der_bus_indices=zone_j_der_buses,
                                  observation_bus_indices=zone_i_v_buses)``

    * Generator AVR columns:  ``∂V_i / ∂V_gen_j``
          via ``compute_dV_dVgen_matrix(gen_bus_indices_pp=zone_j_gen_buses,
                                        observation_bus_indices=zone_i_v_buses)``

    * PCC setpoint columns:  ``∂V_i / ∂Q_PCC_set_j``
          via ``compute_dV_dQ_der`` at the PCC HV buses of zone j
          (negated for load convention, same as in TSOController._build_sensitivity_matrix).

    Current limitation:
        H_ij is recomputed once per TSO period (not cached between steps).
        This is sufficient for quasi-static stability analysis.
    """

    def __init__(
        self,
        zones: List[ZoneDefinition],
        net: pp.pandapowerNet,
        *,
        verbose: int = 0,
    ) -> None:
        """
        Initialise the coordinator.

        Parameters
        ----------
        zones : List[ZoneDefinition]
            One entry per TSO zone, ordered by zone_id.
        net : pp.pandapowerNet
            The COMBINED network (plant model).  This is used ONLY for
            cross-sensitivity computation (Jacobian queries).  Measurements
            and control application are handled externally by the simulation
            loop.
        verbose : int
            0 = silent, 1 = per-step summary, 2 = full diagnostic printout.
        """
        self.zones = {z.zone_id: z for z in zones}
        """Dict mapping zone_id → ZoneDefinition."""

        self.net = net
        """Reference to the combined pandapower network."""

        self.verbose = verbose

        self._controllers: Dict[int, TSOController] = {}
        """zone_id → TSOController (registered via register_tso_controller)."""

        # ── Cross-sensitivity storage ─────────────────────────────────────────
        # H_blocks[(i, j)] = H_ij matrix shape (n_outputs_i, n_controls_j)
        # M_blocks[(i, j)] = M_TSO,ij matrix (same shape as H blocks after
        #                     similarity transform with G_w)
        self._H_blocks: Dict[Tuple[int, int], NDArray[np.float64]] = {}
        self._M_blocks: Dict[Tuple[int, int], NDArray[np.float64]] = {}

        # ── Diagnostic storage (populated during step()) ──────────────────────
        self.last_coupling_diagnostics: Dict[int, Dict] = {}
        """Per-zone stability diagnostics from the most recent step.
        Keys: zone_id.  Values: dict with keys
        'lambda_max_Mii', 'coupling_sum', 'contraction_lhs', 'stable'.
        """

    # ─────────────────────────────────────────────────────────────────────────
    #  Registration
    # ─────────────────────────────────────────────────────────────────────────

    def register_tso_controller(
        self,
        zone_id: int,
        controller: TSOController,
    ) -> None:
        """
        Attach a fully configured TSOController to a zone.

        Must be called for every zone before ``step()`` can be used.

        Parameters
        ----------
        zone_id : int
        controller : TSOController
            Controller already initialised with sensitivities and actuator bounds.
        """
        if zone_id not in self.zones:
            raise KeyError(f"No ZoneDefinition registered for zone_id={zone_id}.")
        self._controllers[zone_id] = controller

    # ─────────────────────────────────────────────────────────────────────────
    #  Cross-sensitivity computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_cross_sensitivities(self) -> None:
        """
        Compute all H_ij cross-sensitivity blocks from the full-network Jacobian.

        This should be called:
          * Once at initialisation (after the first converged power flow).
          * After each TSO step where controls have been applied and a new
            power flow has been run (to keep H_ij consistent with the
            current operating point).

        For each ordered pair (i, j):
            * If i == j: H_ii (local diagonal block)
            * If i ≠ j: H_ij (off-diagonal coupling block)

        H_ij has shape (n_outputs_i, n_controls_j) with the column ordering:
            [Q_DER_j | Q_PCC_j | V_gen_j | s_OLTC_j]

        Row ordering (same for all zones): [V_bus_i | I_line_i].

        After calling this method, ``_H_blocks[(i, j)]`` is populated for
        all (i, j) pairs.
        """
        jac = JacobianSensitivities(self.net)
        n_zones = len(self.zones)
        zone_ids = sorted(self.zones.keys())

        if self.verbose >= 2:
            print("[MultiTSOCoordinator] Computing cross-sensitivity blocks H_ij ...")

        for i in zone_ids:
            zi = self.zones[i]
            n_v_i = len(zi.v_bus_indices)
            n_li  = len(zi.line_indices)

            for j in zone_ids:
                zj = self.zones[j]
                n_der_j  = len(zj.tso_der_indices)
                n_pcc_j  = len(zj.pcc_trafo_indices)
                n_gen_j  = len(zj.gen_indices)
                n_oltc_j = len(zj.oltc_trafo_indices)
                n_col    = n_der_j + n_pcc_j + n_gen_j + n_oltc_j

                # Output shape of this block
                H_ij = np.zeros((n_v_i + n_li, n_col), dtype=np.float64)

                # ── DER Q columns: ∂V_i / ∂Q_DER_j ──────────────────────────
                # Note: compute_dV_dQ_der only works for PQ buses.
                # In the IEEE 39-bus setup DERs sit at PV generator buses —
                # those buses have fixed voltage and are excluded from the
                # reduced Jacobian.  We catch the ValueError and leave the
                # DER Q columns as zero (the AVR columns dominate anyway).
                if n_der_j > 0:
                    try:
                        dV_dQder, obs_map, der_map = jac.compute_dV_dQ_der(
                            der_bus_indices=list(zj.tso_der_buses),
                            observation_bus_indices=zi.v_bus_indices,
                        )
                        # Map Jacobian output back to zone i's row ordering
                        for k_obs, obs_bus in enumerate(obs_map):
                            if obs_bus in zi.v_bus_indices:
                                row = zi.v_bus_indices.index(obs_bus)
                                for k_der, der_bus in enumerate(der_map):
                                    col = zj.tso_der_buses.index(der_bus)
                                    H_ij[row, col] = dV_dQder[k_obs, k_der]

                        # ∂I_i / ∂Q_DER_j
                        if n_li > 0:
                            dI_dQder, line_map, der_map_i = \
                                jac.compute_dI_dQ_der_matrix(
                                    line_indices=zi.line_indices,
                                    der_bus_indices=list(zj.tso_der_buses),
                                )
                            for k_line, line_idx in enumerate(line_map):
                                if line_idx in zi.line_indices:
                                    row = n_v_i + zi.line_indices.index(line_idx)
                                    for k_der, der_bus in enumerate(der_map_i):
                                        col = zj.tso_der_buses.index(der_bus)
                                        H_ij[row, col] = dI_dQder[k_line, k_der]

                    except ValueError:
                        # All DER buses are PV/slack — ∂V/∂Q_DER is zero in
                        # the linearised model (voltage is fixed by the AVR).
                        # H_ij DER columns stay zero; AVR columns are filled below.
                        if self.verbose >= 1:
                            print(
                                f"  [coordinator] Zone {j} DER buses are PV buses; "
                                "DER Q cross-sensitivity = 0 (AVR dominates)."
                            )

                # ── PCC Q setpoint columns: ∂V_i / ∂Q_PCC_set_j ─────────────
                # The PCC setpoint effect is approximated as a Q injection at
                # the PCC HV bus (load convention: positive = into trafo from HV).
                # Negated relative to generator convention (same as TSOController).
                if n_pcc_j > 0:
                    pcc_hv_buses_j = []
                    for t in zj.pcc_trafo_indices:
                        if t in self.net.trafo.index:
                            pcc_hv_buses_j.append(int(self.net.trafo.at[t, "hv_bus"]))
                        elif hasattr(self.net, "trafo3w") and t in self.net.trafo3w.index:
                            pcc_hv_buses_j.append(int(self.net.trafo3w.at[t, "hv_bus"]))

                    if pcc_hv_buses_j:
                        dV_dQ_pcc, obs_map_p, pcc_map = jac.compute_dV_dQ_der(
                            der_bus_indices=pcc_hv_buses_j,
                            observation_bus_indices=zi.v_bus_indices,
                        )
                        for k_obs, obs_bus in enumerate(obs_map_p):
                            if obs_bus in zi.v_bus_indices:
                                row = zi.v_bus_indices.index(obs_bus)
                                for k_pcc, pcc_bus in enumerate(pcc_map):
                                    if pcc_bus in pcc_hv_buses_j:
                                        col = n_der_j + pcc_hv_buses_j.index(pcc_bus)
                                        # Negate: load convention vs. generator convention
                                        H_ij[row, col] = -dV_dQ_pcc[k_obs, k_pcc]

                # ── AVR columns: ∂V_i / ∂V_gen_j ─────────────────────────────
                if n_gen_j > 0:
                    gen_terminal_buses_j = [
                        int(self.net.gen.at[g, "bus"])
                        for g in zj.gen_indices
                    ]
                    dV_dVgen, obs_map_g, gen_map = jac.compute_dV_dVgen_matrix(
                        gen_bus_indices_pp=gen_terminal_buses_j,
                        observation_bus_indices=zi.v_bus_indices,
                    )
                    for k_obs, obs_bus in enumerate(obs_map_g):
                        if obs_bus in zi.v_bus_indices:
                            row = zi.v_bus_indices.index(obs_bus)
                            for k_gen, gen_bus in enumerate(gen_map):
                                if gen_bus in gen_terminal_buses_j:
                                    col = n_der_j + n_pcc_j + gen_terminal_buses_j.index(gen_bus)
                                    H_ij[row, col] = dV_dVgen[k_obs, k_gen]

                    if n_li > 0:
                        dI_dVgen, line_map_g, gen_map_i = \
                            jac.compute_dI_dVgen_matrix(
                                line_indices=zi.line_indices,
                                gen_bus_indices_pp=gen_terminal_buses_j,
                            )
                        for k_line, line_idx in enumerate(line_map_g):
                            if line_idx in zi.line_indices:
                                row = n_v_i + zi.line_indices.index(line_idx)
                                for k_gen, gen_bus in enumerate(gen_map_i):
                                    if gen_bus in gen_terminal_buses_j:
                                        col = n_der_j + n_pcc_j + gen_terminal_buses_j.index(gen_bus)
                                        H_ij[row, col] = dI_dVgen[k_line, k_gen]

                # ── Machine-trafo OLTC columns: ∂V_i / ∂s_OLTC_j ────────────
                if n_oltc_j > 0:
                    oltc_indices_j = list(zj.oltc_trafo_indices)
                    dV_ds, obs_map_s, trafo_map_s = jac.compute_dV_ds_2w_matrix(
                        trafo_indices=oltc_indices_j,
                        observation_bus_indices=zi.v_bus_indices,
                    )
                    col_offset = n_der_j + n_pcc_j + n_gen_j
                    for k_obs, obs_bus in enumerate(obs_map_s):
                        if obs_bus in zi.v_bus_indices:
                            row = zi.v_bus_indices.index(obs_bus)
                            for k_t, t_idx in enumerate(trafo_map_s):
                                if t_idx in oltc_indices_j:
                                    col = col_offset + oltc_indices_j.index(t_idx)
                                    H_ij[row, col] = dV_ds[k_obs, k_t]

                    if n_li > 0:
                        dI_ds, line_map_s, trafo_map_si = \
                            jac.compute_dI_ds_2w_matrix(
                                line_indices=zi.line_indices,
                                trafo_indices=oltc_indices_j,
                            )
                        for k_line, line_idx in enumerate(line_map_s):
                            if line_idx in zi.line_indices:
                                row = n_v_i + zi.line_indices.index(line_idx)
                                for k_t, t_idx in enumerate(trafo_map_si):
                                    if t_idx in oltc_indices_j:
                                        col = col_offset + oltc_indices_j.index(t_idx)
                                        H_ij[row, col] = dI_ds[k_line, k_t]

                self._H_blocks[(i, j)] = H_ij

        if self.verbose >= 2:
            for (i, j), H in self._H_blocks.items():
                print(f"  H_({i},{j}): shape={H.shape}  ||H||_F={np.linalg.norm(H):.4g}")

    # ─────────────────────────────────────────────────────────────────────────
    #  M_sys assembly and stability diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def compute_M_blocks(self) -> None:
        """
        Assemble the preconditioned system blocks M_TSO,ij from H_blocks.

        Requires ``compute_cross_sensitivities()`` to have been called first.

        For each (i, j) pair:

            C_ij = H_ii^T  Q_obj,i  H_ij        (cross curvature)
            M_ij = G_w,i^{-½}  C_ij  G_w,j^{-½} (preconditioned)

        When i == j: C_ii = H_ii^T Q_obj,i H_ii  (local curvature)
                     M_ii = G_w,i^{-½} C_ii G_w,i^{-½} (same as existing single-zone M).

        The result is stored in ``_M_blocks[(i, j)]``.
        """
        zone_ids = sorted(self.zones.keys())

        for i in zone_ids:
            zi = self.zones[i]
            q_obj_i = zi.q_obj_diagonal()    # per-output weights for zone i
            gw_i    = zi.gw_diagonal()       # per-control regularisation for zone i
            gw_i_inv_sqrt = 1.0 / np.sqrt(np.maximum(gw_i, 1e-12))

            H_ii = self._H_blocks.get((i, i))
            if H_ii is None:
                raise RuntimeError(
                    f"H_({i},{i}) not computed.  Call compute_cross_sensitivities first."
                )

            # Q^{1/2} weighted H_ii for efficient C_ii = (Q^{1/2} H_ii)^T (Q^{1/2} H_ii)
            q_sqrt_i = np.sqrt(np.maximum(q_obj_i, 0.0))
            QH_ii = q_sqrt_i[:, None] * H_ii   # (n_y_i, n_u_i)

            for j in zone_ids:
                zj = self.zones[j]
                gw_j = zj.gw_diagonal()
                gw_j_inv_sqrt = 1.0 / np.sqrt(np.maximum(gw_j, 1e-12))

                H_ij = self._H_blocks.get((i, j))
                if H_ij is None:
                    continue

                # Cross curvature: C_ij = H_ii^T Q_obj,i H_ij
                # Efficient form: C_ij = (Q^{1/2} H_ii)^T  (Q^{1/2} H_ij)
                QH_ij = q_sqrt_i[:, None] * H_ij   # (n_y_i, n_u_j)
                C_ij = QH_ii.T @ QH_ij              # (n_u_i, n_u_j)

                # Preconditioned block: M_ij = G_w,i^{-½} C_ij G_w,j^{-½}
                M_ij = (gw_i_inv_sqrt[:, None] * C_ij) * gw_j_inv_sqrt[None, :]
                self._M_blocks[(i, j)] = M_ij

    def check_contraction(self) -> Dict[int, Dict]:
        """
        Evaluate the diagonal-dominance contraction condition for each zone.

        Condition (Schwenke / CIGRE 2026, eq. for multi-TSO):

            0  <  α_i · (λ_max(M_TSO,ii) + Σ_{j≠i} ||M_TSO,ij||₂)  <  2

        If this holds for all i, the decentralised OFO iterations converge.

        Returns
        -------
        diagnostics : Dict[int, Dict]
            zone_id → {
              'lambda_max_Mii':  float,  # largest eigenvalue of local M_ii
              'alpha_max_local': float,  # 2 / lambda_max_Mii (single-zone bound)
              'coupling_sum':    float,  # Σ_{j≠i} ||M_ij||₂
              'contraction_lhs': float,  # α_i * (λ_max + Σ||M_ij||₂)
              'stable':          bool,   # 0 < contraction_lhs < 2
              'warnings':        List[str],
            }
        """
        diagnostics: Dict[int, Dict] = {}
        zone_ids = sorted(self.zones.keys())

        for i in zone_ids:
            zi = self.zones[i]
            M_ii = self._M_blocks.get((i, i))
            if M_ii is None:
                diagnostics[i] = {'stable': False, 'warnings': ['M_ii not computed']}
                continue

            # ── Local: λ_max(M_ii) ────────────────────────────────────────────
            eig_ii_all = np.linalg.eigvalsh(M_ii)
            lambda_max_all = float(np.maximum(eig_ii_all[-1], 0.0))
            # Filter near-zero eigenvalues (null-space from co-located DERs)
            eig_tol = 1e-10 * max(lambda_max_all, 1e-14)
            eig_ii = eig_ii_all[eig_ii_all > eig_tol]
            lambda_max = float(eig_ii[-1]) if len(eig_ii) > 0 else 0.0
            alpha_max_local = (2.0 / lambda_max) if lambda_max > 1e-14 else np.inf

            # ── Coupling sum: Σ_{j≠i} ||M_ij||₂ ─────────────────────────────
            # ||M_ij||₂ = spectral norm (largest singular value of M_ij)
            coupling_norms: Dict[int, float] = {}
            for j in zone_ids:
                if j == i:
                    continue
                M_ij = self._M_blocks.get((i, j))
                if M_ij is None or M_ij.size == 0:
                    coupling_norms[j] = 0.0
                    continue
                # Spectral norm via largest singular value
                sv_max = float(np.linalg.norm(M_ij, ord=2))
                coupling_norms[j] = sv_max

            coupling_sum = sum(coupling_norms.values())

            # ── Diagonal-dominance criterion ──────────────────────────────────
            contraction_lhs = zi.alpha * (lambda_max + coupling_sum)
            stable = (contraction_lhs > 0.0) and (contraction_lhs < 2.0)

            warnings: List[str] = []
            if contraction_lhs >= 2.0:
                warnings.append(
                    f"Zone {i}: contraction_lhs = {contraction_lhs:.4f} ≥ 2.0 — "
                    f"INSTABILITY RISK.  Reduce α_i or increase g_w_i."
                )
            elif contraction_lhs > 1.5:
                warnings.append(
                    f"Zone {i}: contraction_lhs = {contraction_lhs:.4f} — "
                    f"marginal stability (> 1.5), consider reducing α_i."
                )
            if coupling_sum > lambda_max:
                warnings.append(
                    f"Zone {i}: coupling_sum ({coupling_sum:.4f}) > "
                    f"λ_max_Mii ({lambda_max:.4f}) — off-diagonal dominates; "
                    f"DSO cascade can improve diagonal dominance."
                )

            if self.verbose >= 1:
                print(
                    f"[MultiTSO] Zone {i}: λ_max(M_ii)={lambda_max:.4f}  "
                    f"Σ‖M_ij‖₂={coupling_sum:.4f}  "
                    f"α·(λ+Σ)={contraction_lhs:.4f}  "
                    f"{'OK' if stable else 'UNSTABLE'}"
                )
            for w in warnings:
                print(f"  WARNING: {w}")

            diagnostics[i] = {
                'lambda_max_Mii':  lambda_max,
                'alpha_max_local': alpha_max_local,
                'coupling_sum':    coupling_sum,
                'coupling_norms':  coupling_norms,
                'contraction_lhs': contraction_lhs,
                'stable':          stable,
                'warnings':        warnings,
            }

        self.last_coupling_diagnostics = diagnostics
        return diagnostics

    def assemble_M_sys(self) -> NDArray[np.float64]:
        """
        Assemble the full block system matrix M_sys.

        M_sys is the (N * n_u_max) × (N * n_u_max) matrix with blocks M_TSO,ij.
        In general the blocks have different sizes (each zone may have a
        different number of control variables), so M_sys is assembled by
        concatenating block rows/columns.

        Returns
        -------
        M_sys : NDArray[np.float64]
            Block matrix shape (Σ n_u_i, Σ n_u_i).
        """
        zone_ids = sorted(self.zones.keys())
        n_per_zone = [self.zones[z].n_controls() for z in zone_ids]
        n_total = sum(n_per_zone)

        M_sys = np.zeros((n_total, n_total), dtype=np.float64)

        # Fill block by block
        row_offset = 0
        for i_idx, i in enumerate(zone_ids):
            col_offset = 0
            for j_idx, j in enumerate(zone_ids):
                M_ij = self._M_blocks.get((i, j))
                if M_ij is not None:
                    r0, r1 = row_offset, row_offset + n_per_zone[i_idx]
                    c0, c1 = col_offset, col_offset + n_per_zone[j_idx]
                    # Guard against shape mismatches (can occur if zones differ
                    # in size between local H_ii and cross H_ij)
                    actual_r = min(M_ij.shape[0], r1 - r0)
                    actual_c = min(M_ij.shape[1], c1 - c0)
                    M_sys[r0:r0+actual_r, c0:c0+actual_c] = M_ij[:actual_r, :actual_c]
                col_offset += n_per_zone[j_idx]
            row_offset += n_per_zone[i_idx]

        return M_sys

    # ─────────────────────────────────────────────────────────────────────────
    #  Control step (decentralised)
    # ─────────────────────────────────────────────────────────────────────────

    def step(
        self,
        measurements: Dict[int, Measurement],
        step_index: int,
        *,
        recompute_cross_sensitivities: bool = False,
    ) -> Dict[int, object]:
        """
        Execute one decentralised TSO step for all zones.

        Each zone's TSOController.step() is called independently.
        There is NO explicit coupling-compensation between zones; the
        convergence is guaranteed by the diagonal-dominance condition
        (see :meth:`check_contraction`).

        Parameters
        ----------
        measurements : Dict[int, Measurement]
            zone_id → Measurement object from the combined plant network.
        step_index : int
            Current simulation step number (used for logging).
        recompute_cross_sensitivities : bool
            If True, recompute H_ij blocks after this step.  Set this to True
            on the first step and periodically (e.g. every 10 steps) to keep
            the cross-sensitivity consistent with the current operating point.

        Returns
        -------
        outputs : Dict[int, ControllerOutput]
            zone_id → ControllerOutput from each zone's TSOController.
        """
        if len(self._controllers) < len(self.zones):
            raise RuntimeError(
                f"Not all zones have a registered TSOController. "
                f"Expected {len(self.zones)}, got {len(self._controllers)}."
            )

        outputs = {}
        for zone_id, controller in self._controllers.items():
            meas = measurements.get(zone_id)
            if meas is None:
                raise KeyError(f"No measurement provided for zone_id={zone_id}.")
            outputs[zone_id] = controller.step(meas)

        # Optionally refresh cross-sensitivity and re-run stability diagnostics
        if recompute_cross_sensitivities:
            self.compute_cross_sensitivities()
            self.compute_M_blocks()
            self.check_contraction()

        return outputs

    # ─────────────────────────────────────────────────────────────────────────
    #  Utility accessors
    # ─────────────────────────────────────────────────────────────────────────

    def get_H_block(self, i: int, j: int) -> Optional[NDArray[np.float64]]:
        """Return H_ij (or None if not yet computed)."""
        return self._H_blocks.get((i, j))

    def get_M_block(self, i: int, j: int) -> Optional[NDArray[np.float64]]:
        """Return M_TSO,ij (or None if not yet computed)."""
        return self._M_blocks.get((i, j))

    def invalidate_sensitivity_cache(self) -> None:
        """
        Clear all cached H_ij and M_ij blocks and each zone's TSOController cache.

        Call this after a contingency (topology change) to force recomputation
        at the next TSO step.
        """
        self._H_blocks.clear()
        self._M_blocks.clear()
        for ctrl in self._controllers.values():
            ctrl.invalidate_sensitivity_cache()

    def update_network(self, net: pp.pandapowerNet) -> None:
        """
        Replace the network reference (e.g. after a contingency or topology change).

        Also updates the JacobianSensitivities inside each TSOController.
        """
        self.net = net
        for ctrl in self._controllers.values():
            ctrl.sensitivities = JacobianSensitivities(net)
        self.invalidate_sensitivity_cache()
