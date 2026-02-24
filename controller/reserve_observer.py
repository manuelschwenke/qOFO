"""
Reserve Observer Module
=======================

Superordinate observer that monitors DER reactive power contributions
to each TSO-DSO interface (3W coupler) and decides whether the shunt
reactor at the interface's tertiary winding should step in.

The observer uses the Jacobian sensitivity sub-matrix
``∂Q_interface / ∂Q_DER`` (shape ``[n_interfaces, n_der]``) to
compute each DER fleet's effective contribution to each interface
reactive power flow.  Hysteresis logic is applied **per interface**
(not aggregate), and a 1:1 mapping between interface trafos and
tertiary shunts is assumed: when the DER contribution at interface *j*
exceeds the engage threshold, shunt *j* is forced ON.

A hysteresis band (engage threshold > release threshold) prevents
chattering.

Author: Manuel Schwenke / Claude Code
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class ReserveObserverConfig:
    """
    Configuration for a single Reserve Observer instance.

    Attributes
    ----------
    q_threshold_mvar : float
        Per-interface DER Q contribution above which shunt engagement
        is forced at that interface's tertiary winding.
        For example, 15.0 means that once the DER fleet's Jacobian-
        weighted reactive power contribution to an interface exceeds
        15 Mvar (in absorbing direction), the corresponding shunt is
        forced ON.
    q_release_mvar : float
        Per-interface DER Q contribution below which the engaged shunt
        may be released.  Must be strictly less than *q_threshold_mvar*
        to provide hysteresis.
    shunt_q_steps_mvar : List[float]
        Signed reactive power per state change for each shunt [Mvar].
        Positive = reactor (absorbs Q), negative = capacitor (injects Q).
        Must have the same length as the number of interface transformers
        (1:1 mapping).
    """
    q_threshold_mvar: float = 15.0
    q_release_mvar: float = 5.0
    shunt_q_steps_mvar: List[float] = field(default_factory=list)
    cooldown_min: int = 3
    """Minimum minutes between consecutive engage/release actions on the
    same interface.  Prevents overreaction to transient conditions."""

    def __post_init__(self) -> None:
        if self.q_release_mvar >= self.q_threshold_mvar:
            raise ValueError(
                f"q_release_mvar ({self.q_release_mvar}) must be less than "
                f"q_threshold_mvar ({self.q_threshold_mvar})"
            )
        if self.cooldown_min < 0:
            raise ValueError(
                f"cooldown_min must be non-negative, got {self.cooldown_min}"
            )


@dataclass
class ShuntOverride:
    """Bound overrides returned by :meth:`ReserveObserver.evaluate`."""
    force_engage: List[int] = field(default_factory=list)
    force_release: List[int] = field(default_factory=list)


class ReserveObserver:
    """
    Per-interface DER reactive power reserve observer.

    Monitors the Jacobian-weighted DER reactive power contribution to
    each TSO-DSO interface and decides on shunt engagement at the
    corresponding tertiary winding.

    There is a **1:1 mapping** between interface transformers (3W
    couplers) and tertiary shunts: interface *j* has exactly one shunt
    at index *j*.

    Usage in the cascade loop::

        # After DSO H matrix is built, extract sub-matrix
        n_iface = len(dso_config.interface_trafo_indices)
        n_der = len(dso_config.der_bus_indices)
        dQ_dQder = dso._H_cache[:n_iface, :n_der]

        # Evaluate
        override = reserve_obs.evaluate(der_q_mvar, shunt_states, dQ_dQder)

        # Apply overrides to the MIQP shunt bounds
        for idx in override.force_engage:
            dso.set_shunt_overrides({idx: (1, 1)})
        for idx in override.force_release:
            dso.set_shunt_overrides({idx: (0, 0)})

    The observer tracks an internal ``_engaged`` flag per interface to
    implement the hysteresis logic.
    """

    def __init__(self, config: ReserveObserverConfig) -> None:
        self.config = config
        n_interfaces = len(config.shunt_q_steps_mvar)
        self._engaged: List[bool] = [False] * n_interfaces
        # Per-interface minute of last engage/release action (for cooldown).
        # Initialised to -∞ so the first action is never blocked.
        self._last_action_min: List[int] = [-999] * n_interfaces

    def evaluate(
            self,
            der_q_mvar: NDArray[np.float64],
            shunt_states: NDArray[np.int64],
            dQ_dQder: Optional[NDArray[np.float64]] = None,
            minute: int = 0,
    ) -> ShuntOverride:
        """
        Evaluate per-interface DER Q burden and return shunt overrides.

        Parameters
        ----------
        der_q_mvar : NDArray[np.float64]
            Current DER reactive power setpoints [Mvar].
            Shape ``(n_der,)``.
        shunt_states : NDArray[np.int64]
            Current shunt switching states (0 = off, 1 = on).
            Shape ``(n_shunts,)``.
        dQ_dQder : NDArray[np.float64]
            Jacobian sub-matrix ``∂Q_interface / ∂Q_DER`` of shape
            ``(n_interfaces, n_der)``. Each row *j* maps DER Q changes
            to the reactive power flow at interface *j*. Must be provided;
            the aggregate fallback is not supported in this implementation
            as it does not respect per-interface sign conventions.
        minute : int, optional
            Current simulation minute (used for cooldown tracking).

        Returns
        -------
        ShuntOverride
            Lists of shunt indices (0-based within the DSO shunt vector)
            to force-engage or force-release.

        Raises
        ------
        ValueError
            If ``dQ_dQder`` is ``None``. The aggregate fallback is
            explicitly unsupported to prevent silently incorrect results.
        """
        if dQ_dQder is None:
            print('NONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            raise ValueError(
                "dQ_dQder must be provided. The aggregate fallback (dQ_dQder=None) "
                "is not supported because it does not respect per-interface sign "
                "conventions and would produce silently incorrect shunt decisions."
            )

        override = ShuntOverride()
        n_interfaces = len(self.config.shunt_q_steps_mvar)

        # Per-interface DER Q contribution:
        #   q_contribution[j] = dQ_dQder[j, :] @ der_q_mvar
        # With negative sensitivities and negative DER Q (absorption),
        # q_contribution is POSITIVE — representing the magnitude of
        # DER-driven Q flow at the interface.
        q_contribution = dQ_dQder @ der_q_mvar  # shape (n_interfaces,)

        for j in range(n_interfaces):
            q_step = self.config.shunt_q_steps_mvar[j]
            currently_on = int(np.round(shunt_states[j])) != 0

            # --- FIX 3: Reconcile internal engaged-state with the actual plant
            # state BEFORE applying hysteresis logic. If the shunt was switched
            # ON externally (e.g. by the MIQP solver in a previous step), the
            # observer must track it; otherwise a release decision made below
            # would be immediately overwritten on the next call.
            if currently_on and not self._engaged[j]:
                self._engaged[j] = True

            # Compute burden (positive = DERs need relief in the shunt's
            # compensation direction):
            #
            #   q_contribution[j] = dQ_dQder[j,:] @ der_q_mvar
            #
            # In the TUDa network the sensitivities dQ_interface/dQ_DER are
            # negative (a DER absorbing more Q causes more Q to flow from TN
            # into DN).  When DERs absorb Q (der_q < 0), the product of two
            # negatives makes q_contribution POSITIVE.  That positive value
            # is exactly the magnitude of DER-driven Q flow at the interface
            # that a reactor should relieve.
            #
            # For a reactor (q_step > 0): burden = +q_contribution
            #   → positive when DERs drive Q flow (need relief).
            # For a capacitor (q_step < 0): burden = -q_contribution
            #   → positive when DERs inject Q (need relief).
            if q_step > 0:
                burden = q_contribution[j]
            else:
                burden = -q_contribution[j]

            # Cooldown: skip switching decisions if the last action on this
            # interface was less than cooldown_min minutes ago.
            in_cooldown = (minute - self._last_action_min[j]) < self.config.cooldown_min

            if not self._engaged[j]:
                # Engage when burden exceeds the threshold
                if burden > self.config.q_threshold_mvar and not in_cooldown:
                    self._engaged[j] = True
                    self._last_action_min[j] = minute
                    if not currently_on:
                        override.force_engage.append(j)
            else:
                # Release when burden drops below the release threshold
                # (hysteresis band prevents chattering)
                if burden < self.config.q_release_mvar and not in_cooldown:
                    self._engaged[j] = False
                    self._last_action_min[j] = minute
                    if currently_on:
                        override.force_release.append(j)
                else:
                    # Stay engaged — reinforce if somehow switched off externally.
                    if not currently_on:
                        override.force_engage.append(j)

        return override

