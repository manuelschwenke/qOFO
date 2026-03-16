"""
Sensitivity Updater Module
===========================

This module provides the ``SensitivityUpdater`` class that updates
state-dependent columns of the sensitivity matrix H between full Jacobian
recomputations.

Currently one type of update is supported:

1. **Shunt columns** (MSR / MSC): Shunts are constant-susceptance devices
   whose reactive power injection is ``Q = B · V²``.  The shunt columns
   of H are rescaled by ``(V_measured / V_cached)²`` — a pure column
   scaling that is O(n_shunts × n_outputs).

Author: Manuel Schwenke
Date: 2026-02-13

References
----------
[1] Schwenke et al., PSCC 2026, Section II (Linearised Power System Model)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional

from core.measurement import Measurement
from sensitivity.jacobian import JacobianSensitivities


class SensitivityUpdater:
    """
    Updates state-dependent columns of the sensitivity matrix H.

    The updater stores an immutable base copy of H and always rescales
    from that base to avoid floating-point drift.

    Parameters
    ----------
    H : NDArray[np.float64]
        The sensitivity matrix from ``build_sensitivity_matrix_H`` (or the
        final assembled H from the controller).
    mappings : dict
        The mappings dict from ``build_sensitivity_matrix_H``.  Must contain
        ``'shunt_buses'``, ``'shunt_cached_v_pu'``, ``'der_buses'``,
        ``'oltc_trafos'``, ``'oltc_trafo3w'``.
    sensitivities : JacobianSensitivities
        Reference to the Jacobian sensitivity calculator.
    update_interval_min : int, optional
        Minimum number of iterations between updates.  Default: 1
        (every call triggers an update).
    col_shunt_start : int, optional
        Override the shunt column offset in H.  If ``None``, computed as
        ``n_der + n_oltc2w + n_oltc3w`` from ``mappings``.
    """

    def __init__(
        self,
        H: NDArray[np.float64],
        mappings: dict,
        sensitivities: JacobianSensitivities,
        update_interval_min: int = 1,
        col_shunt_start: Optional[int] = None,
    ) -> None:
        self._H_base = H.copy()
        self._H_current = H.copy()
        self._mappings = mappings
        self._sensitivities = sensitivities
        self._update_interval = update_interval_min
        self._last_update_iteration = -999  # force first update

        # ------------------------------------------------------------------
        # Shunt column setup
        # ------------------------------------------------------------------
        n_der = len(mappings.get('der_buses', []))
        n_oltc2w = len(mappings.get('oltc_trafos', []))
        n_oltc3w = len(mappings.get('oltc_trafo3w', []))
        self._n_shunt = len(mappings.get('shunt_buses', []))

        if col_shunt_start is not None:
            self._col_shunt_start = col_shunt_start
        else:
            self._col_shunt_start = n_der + n_oltc2w + n_oltc3w

        shunt_cached_v = mappings.get('shunt_cached_v_pu', np.array([]))
        self._v_cached_sq = shunt_cached_v ** 2 if len(shunt_cached_v) > 0 else np.array([])

    @property
    def n_shunts(self) -> int:
        """Number of shunt columns managed by this updater."""
        return self._n_shunt

    @property
    def current_H(self) -> NDArray[np.float64]:
        """Return the current (possibly updated) H matrix."""
        return self._H_current

    def update(
        self,
        measurement: Measurement,
        current_iteration: int,
    ) -> NDArray[np.float64]:
        """
        Return an updated H matrix with state-dependent columns corrected.

        If the update interval has not elapsed, returns the previously
        computed H without recomputing.

        Parameters
        ----------
        measurement : Measurement
            Current system measurements (bus voltages for shunt rescaling).
        current_iteration : int
            Current iteration / minute index for interval checking.

        Returns
        -------
        H : NDArray[np.float64]
            The (potentially updated) sensitivity matrix.
        """
        # Check interval
        if (current_iteration - self._last_update_iteration) < self._update_interval:
            return self._H_current

        # Start from the immutable base
        self._H_current[:, :] = self._H_base[:, :]

        # ------------------------------------------------------------------
        # Shunt column rescaling
        # ------------------------------------------------------------------
        if self._n_shunt > 0 and len(self._v_cached_sq) > 0:
            self._update_shunt_columns(measurement)

        self._last_update_iteration = current_iteration
        return self._H_current

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_shunt_columns(self, measurement: Measurement) -> None:
        """Rescale shunt columns by (V_measured / V_cached)²."""
        shunt_buses = self._mappings['shunt_buses']
        v_measured_sq = np.ones(self._n_shunt)

        for i, bus in enumerate(shunt_buses):
            idx = np.where(measurement.bus_indices == bus)[0]
            if len(idx) > 0:
                v_measured_sq[i] = measurement.voltage_magnitudes_pu[idx[0]] ** 2
            else:
                # If voltage not available, assume no change (ratio = 1)
                v_measured_sq[i] = self._v_cached_sq[i]

        ratios = v_measured_sq / self._v_cached_sq  # shape (n_shunt,)

        col_start = self._col_shunt_start
        col_end = col_start + self._n_shunt
        self._H_current[:, col_start:col_end] *= ratios[np.newaxis, :]
