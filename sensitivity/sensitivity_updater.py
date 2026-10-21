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

import warnings

import numpy as np
from numpy.typing import NDArray
from typing import Optional

from core.measurement import Measurement
from sensitivity.jacobian import JacobianSensitivities


def voltage_rescaling_ratios(
    v_measured_sq: NDArray[np.float64],
    v_cached_sq: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the shunt rescaling factors ``(V_measured / V_cached)²``.

    A constant-susceptance bank injects ``q_step · V²`` per step, so every
    sensitivity of a bank, computed at the cached voltage ``V_cached``, is
    brought to the measured voltage by this factor (thesis
    ``eq:shunt_rescaling``).  Shared by :class:`SensitivityUpdater` (shunt
    columns of an MIQP's H) and the switched-shunt integrator in the
    multi-zone runner.

    Repeated Newton-Raphson solves can leave round-off-level voltage
    differences even when the physical operating point is unchanged; ratios
    within a 1e-10 relative V² tolerance of one are set to exactly one so the
    sensitivities are not needlessly changed.

    Parameters
    ----------
    v_measured_sq, v_cached_sq : NDArray[np.float64]
        Squared measured and cached bus-voltage magnitudes [pu²], same shape.
    """
    ratios = (np.atleast_1d(np.asarray(v_measured_sq, dtype=np.float64))
              / np.atleast_1d(np.asarray(v_cached_sq, dtype=np.float64)))
    ratios[np.isclose(ratios, 1.0, rtol=1e-10, atol=0.0)] = 1.0
    return ratios


def bank_voltage_ratio(
    measurement: Measurement,
    cached_net,
    bus_idx: int,
) -> float:
    """Return ``(V_measured / V_cached)²`` at one shunt-bank bus.

    ``V_cached`` is read from ``cached_net.res_bus`` -- the linearisation point
    of the controller model that evaluates the bank's sensitivities -- and
    ``V_measured`` from ``measurement``.  If the bus is not in the measurement,
    a warning is issued and 1.0 is returned (the same fallback as
    :class:`SensitivityUpdater`), i.e. the cached voltage is kept.
    """
    idx = np.where(measurement.bus_indices == int(bus_idx))[0]
    if len(idx) == 0:
        warnings.warn(
            f"Shunt bus {bus_idx} not in measurement; voltage rescaling of its "
            f"sensitivities skipped (cached voltage kept).",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1.0
    v_meas = float(measurement.voltage_magnitudes_pu[idx[0]])
    v_cached = float(cached_net.res_bus.at[int(bus_idx), "vm_pu"])
    return float(voltage_rescaling_ratios(
        np.array([v_meas ** 2]), np.array([v_cached ** 2]),
    )[0])


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

        # shape (n_shunt,); identity guard inside (see voltage_rescaling_ratios)
        ratios = voltage_rescaling_ratios(v_measured_sq, self._v_cached_sq)

        col_start = self._col_shunt_start
        col_end = col_start + self._n_shunt
        self._H_current[:, col_start:col_end] *= ratios[np.newaxis, :]
