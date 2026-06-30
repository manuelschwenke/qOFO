"""
Central (single-controller) OFO Module
======================================

This module defines :class:`CentralOFOController`, the **single centralized**
Online Feedback Optimisation controller used as the CIGRE-2026 V5 best-case
upper-bound reference (``MultiTSOConfig.control_scope == 'central'``).

Unlike the distributed cascaded framework (one
:class:`controller.tso_controller.TSOController` per zone + one
:class:`controller.dso_controller.DSOController` per HV sub-network, coordinated
by capability / setpoint messages), the central controller owns **every**
actuator and observes **every** measurement across the whole interconnection in
a single MIQP:

* Actuators (one action vector ``u``):
    ``[ Q_DER | V_gen | s_OLTC2w | s_shunt | s_OLTC3w ]``
  i.e. all TSO+DSO DER reactive setpoints (w-shift ``q_set``), all synchronous
  generator AVR setpoints, all 2-winding machine-transformer OLTCs, all
  TSO-owned shunts, and all 3-winding coupler OLTCs at the TS–STS interfaces.
* Outputs (one ``y``):
    ``[ V_bus | I_line | Q_gen ]`` over every monitored TN+HV bus / line and the
  synchronous-machine reactive-capability soft band.

There is no interface-Q or tie-Q tracking term: with the problem no longer
decomposed across zones/sub-networks, those reactive flows are internal to the
single controller.  The sole objective is **voltage tracking**, with separate
per-bus weights for the EHV/TN buses (``g_v``) and the HV/STS buses
(``central_dso_g_v``).

Implementation
--------------
``CentralOFOController`` subclasses :class:`TSOController`, which is already a
near-superset (DER w-shift transform ``T'``, generator-Q capability soft rows,
2W OLTCs, shunts, voltage tracking, arbitrary index sets).  The only capability
``TSOController`` lacks is **3W coupler OLTC actuation**; the subclass appends a
trailing 3W-OLTC integer block to the control vector and the corresponding
voltage / current sensitivity columns to ``H`` (computed via the existing 3W
Jacobian helpers in :mod:`sensitivity.jacobian`).  The generator-Q rows w.r.t.
3W taps are treated as second-order and left at zero.

Author: Manuel Schwenke / Claude Code
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from controller.base_controller import OFOParameters
from controller.tso_controller import TSOController, TSOControllerConfig
from core.actuator_bounds import ActuatorBounds
from core.measurement import Measurement
from core.network_state import NetworkState
from sensitivity.jacobian import JacobianSensitivities


@dataclass
class CentralControllerConfig(TSOControllerConfig):
    """Configuration for the single centralized controller (CIGRE V5).

    Extends :class:`TSOControllerConfig` with the 3-winding coupler OLTC
    actuator block and a per-bus voltage-tracking weight vector.

    Notes
    -----
    * ``pcc_trafo_indices`` and ``tie_line_indices`` are expected to be empty
      (no interface-Q / tie-Q tracking in the monolithic formulation); the
      inherited fields are kept only so the parent's H / gradient machinery
      degenerates cleanly.
    * The 3W-OLTC block is **appended last** in the control vector:
      ``[ Q_DER | Q_PCC(0) | V_gen | s_OLTC2w | s_shunt | s_OLTC3w ]``.
    """

    oltc_trafo3w_indices: List[int] = field(default_factory=list)
    """Pandapower ``net.trafo3w`` indices of the TS–STS coupler OLTCs the
    central controller actuates (integer taps).  Their tap positions are read
    from / written to ``net.trafo3w`` (a separate index space from the 2W
    machine OLTCs in :attr:`oltc_trafo_indices`)."""

    g_v_per_bus: Optional[NDArray[np.float64]] = None
    """Per-bus voltage-tracking weight vector, length = ``voltage_bus_indices``.
    EHV/TN buses carry ``g_v`` and HV/STS buses carry ``central_dso_g_v`` (set
    by the runner).  When ``None``, the scalar :attr:`g_v` is broadcast to all
    buses."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.g_v_per_bus is not None:
            n_v = len(self.voltage_bus_indices)
            if len(self.g_v_per_bus) != n_v:
                raise ValueError(
                    f"g_v_per_bus length ({len(self.g_v_per_bus)}) must match "
                    f"voltage_bus_indices length ({n_v})"
                )


class CentralOFOController(TSOController):
    """Single OFO controller spanning all actuators / measurements (CIGRE V5).

    See the module docstring for the full action / output layout.  Reuses the
    entire :class:`TSOController` machinery and adds (i) a trailing 3W coupler
    OLTC integer block and (ii) per-bus voltage weights in the objective.
    """

    def __init__(
        self,
        controller_id: str,
        params: OFOParameters,
        config: CentralControllerConfig,
        network_state: NetworkState,
        actuator_bounds: ActuatorBounds,
        sensitivities: JacobianSensitivities,
    ) -> None:
        super().__init__(
            controller_id=controller_id,
            params=params,
            config=config,
            network_state=network_state,
            actuator_bounds=actuator_bounds,
            sensitivities=sensitivities,
        )
        self.config: CentralControllerConfig = config
        # Per-bus voltage weight vector (fallback: scalar g_v broadcast).
        n_v = len(config.voltage_bus_indices)
        if config.g_v_per_bus is not None:
            self.g_v_per_bus = np.asarray(config.g_v_per_bus, dtype=np.float64)
        else:
            self.g_v_per_bus = np.full(n_v, float(config.g_v), dtype=np.float64)

    # ------------------------------------------------------------------
    #  Control-vector structure: append the 3W-OLTC integer block
    # ------------------------------------------------------------------

    def _n_oltc3w(self) -> int:
        return len(self.config.oltc_trafo3w_indices)

    def _get_control_structure(self) -> Tuple[int, int, List[int]]:
        n_cont, n_int, int_idx = super()._get_control_structure()
        n_oltc3w = self._n_oltc3w()
        if n_oltc3w == 0:
            return n_cont, n_int, int_idx
        # The 3W block is appended after the parent's full control vector.
        start = n_cont + n_int
        int_idx_full = list(int_idx) + list(range(start, start + n_oltc3w))
        return n_cont, n_int + n_oltc3w, int_idx_full

    def _get_oltc_integer_indices(self) -> List[int]:
        base = list(super()._get_oltc_integer_indices())
        n_oltc3w = self._n_oltc3w()
        if n_oltc3w == 0:
            return base
        n_cont, n_int, _ = super()._get_control_structure()
        start = n_cont + n_int
        return base + list(range(start, start + n_oltc3w))

    def _extract_control_values(
        self, measurement: Measurement,
    ) -> NDArray[np.float64]:
        u_base = super()._extract_control_values(measurement)
        n_oltc3w = self._n_oltc3w()
        if n_oltc3w == 0:
            return u_base
        taps = np.zeros(n_oltc3w, dtype=np.float64)
        for k, t in enumerate(self.config.oltc_trafo3w_indices):
            idx = np.where(measurement.oltc3w_indices == t)[0]
            if len(idx) == 0:
                raise ValueError(
                    f"3W OLTC {t} not found in measurement.oltc3w_indices"
                )
            taps[k] = float(measurement.oltc3w_tap_positions[idx[0]])
        return np.concatenate([u_base, taps])

    def _compute_input_bounds(
        self,
        tso_dso_interface_q_current: NDArray[np.float64],
        der_p_current: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        u_lower, u_upper = super()._compute_input_bounds(
            tso_dso_interface_q_current, der_p_current,
        )
        n_oltc3w = self._n_oltc3w()
        if n_oltc3w == 0:
            return u_lower, u_upper
        # 3W coupler tap mechanical limits (constant columns on net.trafo3w).
        # The per-step ±int_max_step clamp and wall-clock OLTC cooldown are
        # applied generically in BaseOFOController.step() over all integer
        # indices (which include the 3W positions via _get_oltc_integer_indices).
        net = self.sensitivities.net
        lo3 = np.array(
            [int(net.trafo3w.at[t, "tap_min"]) for t in self.config.oltc_trafo3w_indices],
            dtype=np.float64,
        )
        hi3 = np.array(
            [int(net.trafo3w.at[t, "tap_max"]) for t in self.config.oltc_trafo3w_indices],
            dtype=np.float64,
        )
        return np.concatenate([u_lower, lo3]), np.concatenate([u_upper, hi3])

    # ------------------------------------------------------------------
    #  Sensitivity matrix: append 3W-OLTC columns to the parent H
    # ------------------------------------------------------------------

    def _build_sensitivity_matrix(self) -> NDArray[np.float64]:
        if self._H_cache is not None:
            return self._H_cache
        # super() builds the [V | I | Q_gen] H over the parent's actuator
        # columns, caches it into self._H_cache, AND constructs a
        # SensitivityUpdater bound to that 56-column base matrix.  We append
        # the 3W coupler-OLTC columns and overwrite the cache with the full
        # matrix.  Crucially, the parent's per-step shunt-V² rescaler
        # (TSOController.step -> self._H_cache = self._sensitivity_updater.update(...))
        # is built around the *base* matrix and would otherwise clobber the
        # appended 3W columns every step; the centralized V5 problem has no
        # shunts, so we disable the updater and keep the frozen full H (the
        # same "cached sensitivities" assumption as the distributed variants).
        base_H = super()._build_sensitivity_matrix()
        self._sensitivity_updater = None
        if self._n_oltc3w() == 0:
            return base_H
        extra = self._build_oltc3w_columns(n_rows=base_H.shape[0])
        full_H = np.hstack([base_H, extra])
        self._H_cache = full_H
        return full_H

    def _build_oltc3w_columns(self, n_rows: int) -> NDArray[np.float64]:
        """Build the trailing 3W coupler-OLTC columns of H.

        Row layout (central case: no Q_PCC / Q_tie rows):
            ``[ V_bus (n_v) | I_line (n_i) | Q_gen (n_gen) ]``.
        V-rows from :meth:`compute_dV_ds_trafo3w_matrix`, I-rows from
        :meth:`compute_dI_ds_3w_matrix`; the Q_gen rows w.r.t. a coupler tap
        are second-order and left at zero.
        """
        cfg = self.config
        t3w = list(cfg.oltc_trafo3w_indices)
        n_oltc3w = len(t3w)
        cols = np.zeros((n_rows, n_oltc3w), dtype=np.float64)
        if n_oltc3w == 0:
            return cols

        n_v = len(cfg.voltage_bus_indices)
        n_i = len(cfg.current_line_indices)
        # Output-row layout is [ V | Q_PCC | I | Q_gen | ... ]; for the
        # centralized problem Q_PCC is empty, but compute the offset honestly.
        i_row_start = n_v + len(cfg.pcc_trafo_indices)
        jac = self.sensitivities
        col_pos = {int(t): j for j, t in enumerate(t3w)}

        # --- V rows: ∂V_obs / ∂s_3w ---
        if n_v > 0:
            try:
                dV, obs_map, t3w_map = jac.compute_dV_ds_trafo3w_matrix(
                    trafo3w_indices=t3w,
                    observation_bus_indices=cfg.voltage_bus_indices,
                )
                v_pos = {int(b): i for i, b in enumerate(cfg.voltage_bus_indices)}
                for jj, t in enumerate(t3w_map):
                    cj = col_pos.get(int(t))
                    if cj is None:
                        continue
                    for ii, b in enumerate(obs_map):
                        ri = v_pos.get(int(b))
                        if ri is not None:
                            cols[ri, cj] = dV[ii, jj]
            except (ValueError, KeyError, IndexError):
                # Coupler OOS / Jacobian gap -> leave 3W V-columns at zero.
                pass

        # --- I rows: ∂I_line / ∂s_3w ---
        if n_i > 0:
            try:
                dI, line_map, t3w_map_i = jac.compute_dI_ds_3w_matrix(
                    line_indices=cfg.current_line_indices,
                    trafo3w_indices=t3w,
                )
                i_pos = {int(l): i for i, l in enumerate(cfg.current_line_indices)}
                for jj, t in enumerate(t3w_map_i):
                    cj = col_pos.get(int(t))
                    if cj is None:
                        continue
                    for ii, l in enumerate(line_map):
                        ri = i_pos.get(int(l))
                        if ri is not None:
                            cols[i_row_start + ri, cj] = dI[ii, jj]
            except (ValueError, KeyError, IndexError):
                pass

        return cols

    # ------------------------------------------------------------------
    #  Objective: voltage tracking only, per-bus weights (EHV vs HV)
    # ------------------------------------------------------------------

    def voltage_curvature_inputs(
        self,
    ) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Voltage rows of H and per-bus ``g_v`` for curvature analysis.

        The monolithic objective is voltage-tracking only, output ordering
        ``[ V_bus | I_line | Q_gen ]``, so the voltage block is the leading
        ``n_v`` rows and the weights are the per-bus :attr:`g_v_per_bus`
        (``g_v`` on TN buses, ``central_dso_g_v`` on HV buses).  This is the
        exact ``(H_v, g_v)`` the V5 curvature probe uses.  See
        :meth:`BaseOFOController.voltage_curvature_inputs`.
        """
        n_v = len(self.config.voltage_bus_indices)
        if n_v == 0:
            return None
        H = self._expand_H_to_der_level(self._build_sensitivity_matrix())
        H_v = np.ascontiguousarray(H[:n_v, :], dtype=np.float64)
        g_v_vec = np.asarray(self.g_v_per_bus, dtype=np.float64)
        return H_v, g_v_vec

    def _compute_objective_gradient(
        self, measurement: Measurement,
    ) -> NDArray[np.float64]:
        """∇f for the monolithic voltage-tracking objective.

        ``∇f = 2 · (w ⊙ (V - V_set))^T · ∂V/∂u`` where ``w`` is the per-bus
        weight vector (``g_v`` on TN buses, ``central_dso_g_v`` on HV buses).
        No interface-Q / tie-Q tracking terms (the problem is not decomposed).
        DER usage regularisation is handled by ``g_u`` inside
        ``build_miqp_problem``.
        """
        grad_f = np.zeros(self.n_controls, dtype=np.float64)
        n_v = len(self.config.voltage_bus_indices)
        if self.config.v_setpoints_pu is None or n_v == 0:
            return grad_f

        v_current = np.zeros(n_v, dtype=np.float64)
        for j, bus_idx in enumerate(self.config.voltage_bus_indices):
            meas_idx = np.where(measurement.bus_indices == bus_idx)[0]
            if len(meas_idx) == 0:
                raise ValueError(f"Bus {bus_idx} not found in measurement")
            v_current[j] = measurement.voltage_magnitudes_pu[meas_idx[0]]

        v_error = v_current - self.config.v_setpoints_pu

        H = self._expand_H_to_der_level(self._build_sensitivity_matrix())
        dV_du = H[:n_v, :]
        grad_f += 2.0 * ((self.g_v_per_bus * v_error) @ dV_du)
        return grad_f
