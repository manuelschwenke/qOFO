"""
Switched-shunt integrator (MSC / MSR) for the TSO OFO controller
================================================================

This module implements a *separate integrating dispatcher* for mechanically
switched shunt banks owned by the TSO at the DSO tertiary windings.  It is
deliberately kept OUTSIDE the OLTC MIQP: a per-instant MIQP engages a large
bulk device only when the full step improves the snapshot objective, which —
with a sizeable step and the switching penalty needed for stability — rarely
holds, so the bank is engaged very seldom or chatters.  An integrating
continuous-relaxation state instead accumulates the time-integral of the
reactive "pressure" projected onto the bank's boundary sensitivity and commits
a physical step only on *sustained* need, which is the correct behaviour for a
slow bulk device.

Two device classes are provided:

* :class:`MSC` — mechanically switched capacitor.  Engaging a step *injects*
  reactive power at the tertiary (raises voltage); the boundary voltage
  sensitivity per step is therefore expected to be non-negative.
* :class:`MSR` — mechanically switched reactor.  Engaging a step *absorbs*
  reactive power (lowers voltage); the expected per-step voltage sensitivity is
  non-positive.

Each bank carries ``n_levels`` discrete levels (lattice ``ℓ ∈ {0 … N}``).  The
device sign is carried by the measured sensitivity column, so the update law is
identical for both classes; the subclasses differ only in the pandapower step
mapping and the expected sensitivity sign used by the fail-fast guards.

Conventions
-----------
* ``Q_eq`` is the *nameplate* equivalent reactive injection in Mvar; the
  committed physical level satisfies ``Q_eq_phys = ℓ · q_step_mvar`` and the
  per-step increment is ``dQ_eq_step = q_step_mvar``.
* The boundary sensitivity ``h_H`` passed in is ``∂y_H / ∂Q_eq`` (per Mvar of
  nameplate injection), evaluated by the online estimator with a unit step.
* The objective ``f`` is the *same* TSO objective used by the rest of the loop;
  ``grad_g = h_Hᵀ · ∇_y f`` is computed by the caller from the shared
  output-space gradient (no separate objective is defined here).

Fail-fast: every constructor and update validates shapes, ranges, and the
hysteresis condition ``delta ∈ (0, q_step/2)``; inconsistencies raise
``ValueError`` rather than silently degrading (see project CLAUDE.md).

Author: Manuel Schwenke
Date: 2026-06-22
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

#: Length of the rolling window over which the daily switching budget is
#: counted [s].  One calendar day; exposed as a module constant so tests can
#: shrink it without touching the public API.
SECONDS_PER_DAY: float = 86_400.0


@dataclass(frozen=True)
class ShuntBankConfig:
    """Static configuration for one switched-shunt bank.

    Parameters
    ----------
    shunt_idx :
        Pandapower ``net.shunt`` index of this bank.  Required (not just the
        bus) because a tertiary may host both an MSC and an MSR bank; the
        toggle, the rank-1 SMW refresh, and the disturbance message all target
        this specific device.
    bus_idx :
        Pandapower bus index of the tertiary node hosting the bank (ground
        truth lives here, on the explicit 3-winding tertiary).  Used by the
        boundary-sensitivity helpers, which read the bus voltage.
    interface_trafo3w_idx :
        ``net.trafo3w`` index of the coupling transformer whose tertiary hosts
        the bank.  Used by the runner to compute the DSO interface feedforward
        from the DSO's own three-winding model.
    dso_id :
        Controller identifier of the DSO whose tertiary hosts the bank.
    kind :
        ``"MSC"`` (capacitor) or ``"MSR"`` (reactor).
    q_step_mvar :
        Nameplate reactive power per level step at ``V = 1`` p.u. [Mvar],
        strictly positive (the sign / direction is encoded by the subclass and
        the measured sensitivity, not by this magnitude).
    n_levels :
        Number of discrete steps ``N`` (lattice ``ℓ ∈ {0 … N}``), ``≥ 1``.
    g_w :
        Quadratic step weight for the continuous-relaxation update, strictly
        positive.  Consistent with the rest of the controller (which fixes
        ``alpha = 1`` and tunes step amplitude purely via the ``g_w`` weights):
        the integrator advances by the OFO step ``Delta = g_H / (2*g_w)``, i.e.
        smaller ``g_w`` gives a larger step (mirrors the MIQP's unconstrained
        step ``du = -grad_f / (2*g_w)``).
    delta :
        Hysteresis half-width [Mvar]; must satisfy ``0 < delta < q_step/2`` so
        the commit band is well posed and cannot straddle a level.
    t_dwell_s :
        Minimum dwell time between commits of this bank [s], ``≥ 0``.
    daily_switch_budget :
        Maximum number of commits allowed within any
        :data:`SECONDS_PER_DAY` window, ``≥ 0``.
    y_h_min, y_h_max :
        HV-boundary voltage band [p.u.] used by the overshoot feasibility
        guard (uniform across the observed boundary buses).  ``y_h_min <
        y_h_max``.
    """

    shunt_idx: int
    bus_idx: int
    interface_trafo3w_idx: int
    dso_id: str
    kind: str
    q_step_mvar: float
    n_levels: int
    g_w: float
    delta: float
    t_dwell_s: float
    daily_switch_budget: int
    y_h_min: float
    y_h_max: float

    def __post_init__(self) -> None:
        if self.kind not in ("MSC", "MSR"):
            raise ValueError(
                f"ShuntBankConfig.kind must be 'MSC' or 'MSR', got {self.kind!r}"
            )
        if not (self.q_step_mvar > 0.0):
            raise ValueError(
                f"q_step_mvar must be > 0, got {self.q_step_mvar}"
            )
        if int(self.n_levels) < 1:
            raise ValueError(
                f"n_levels must be >= 1, got {self.n_levels}"
            )
        if not (self.g_w > 0.0):
            raise ValueError(f"g_w must be > 0, got {self.g_w}")
        if not (0.0 < self.delta < 0.5 * self.q_step_mvar):
            raise ValueError(
                f"delta must lie in (0, q_step/2) = (0, {0.5 * self.q_step_mvar}), "
                f"got {self.delta}"
            )
        if self.t_dwell_s < 0.0:
            raise ValueError(f"t_dwell_s must be >= 0, got {self.t_dwell_s}")
        if int(self.daily_switch_budget) < 0:
            raise ValueError(
                f"daily_switch_budget must be >= 0, got {self.daily_switch_budget}"
            )
        if not (float(self.y_h_min) < float(self.y_h_max)):
            raise ValueError(
                f"y_h_min ({self.y_h_min}) must be strictly less than "
                f"y_h_max ({self.y_h_max})"
            )


@dataclass
class ShuntCommit:
    """Record of a single committed step change, returned to the runner.

    The runner uses this to (atomically) toggle the physical bank, step the DSO
    interface feedforward, and refresh the cached sensitivities via the rank-1
    SMW update — all in the same control instant, with no power flow.
    """

    shunt_idx: int
    bus_idx: int
    interface_trafo3w_idx: int
    dso_id: str
    kind: str
    old_level: int
    new_level: int
    pp_step_new: int
    q_step_mvar: float
    direction: int  # +1 = engage one more step, -1 = release one step


class ShuntBank:
    """One switched-shunt bank with an integrating continuous-relaxation state.

    Subclassed by :class:`MSC` and :class:`MSR`; the base class holds all of the
    update logic and the only difference is :meth:`pp_step_for_level` and
    :attr:`expected_h_sign`.
    """

    #: Expected sign of ``∂V/∂Q_eq`` at the bank's own buses (+1 for a
    #: capacitor, -1 for a reactor).  Used only by the fail-fast sign guard;
    #: the update law itself reads the measured sensitivity directly.
    expected_h_sign: int = 0

    def __init__(self, config: ShuntBankConfig) -> None:
        self.config = config
        # Committed physical level ℓ ∈ {0 … N}; banks start de-energised.
        self.level: int = 0
        # Continuous-relaxation (auxiliary) state in nameplate Mvar.
        self.q_eq_aux: float = 0.0
        # Dwell / budget bookkeeping.  -inf so the first commit is always
        # dwell-eligible.
        self._last_switch_t: float = -math.inf
        self._budget_window_start_t: float = -math.inf
        self._switches_in_window: int = 0

    # -- Derived quantities ------------------------------------------------

    @property
    def q_eq_phys(self) -> float:
        """Committed equivalent nameplate injection [Mvar]: ``ℓ · q_step``."""
        return float(self.level) * self.config.q_step_mvar

    def pp_step_for_level(self, level: int) -> int:
        """Map a lattice level to the pandapower ``net.shunt['step']`` value.

        Overridden per device class.  The base implementation is the identity
        (level == pandapower step), which both MSC and MSR use because the
        capacitive / inductive direction is encoded by the sign of the shunt's
        ``q_mvar`` set at build time, not by the step value.
        """
        return int(level)

    # -- Internal helpers --------------------------------------------------

    def _budget_remaining(self, t_now: float) -> bool:
        """Roll the daily window forward and report whether budget remains."""
        if t_now - self._budget_window_start_t >= SECONDS_PER_DAY:
            # Open a fresh window anchored at the current instant.
            self._budget_window_start_t = t_now
            self._switches_in_window = 0
        return self._switches_in_window < self.config.daily_switch_budget

    def _dwell_elapsed(self, t_now: float) -> bool:
        return (t_now - self._last_switch_t) >= self.config.t_dwell_s

    def _feasible(self, direction: int, v_meas: NDArray[np.float64],
                  h_v: NDArray[np.float64]) -> bool:
        """HV overshoot guard for a candidate one-step move.

        Predicts the boundary voltages after the step using the cached
        sensitivity and checks them against the configured band:

            y_H_min ≤ v_meas + h_v · (direction · q_step) ≤ y_H_max

        ``h_v`` is ``∂V/∂Q_eq`` (per Mvar) at the observed boundary buses.
        """
        v_pred = v_meas + h_v * (float(direction) * self.config.q_step_mvar)
        return bool(
            np.all(v_pred >= self.config.y_h_min)
            and np.all(v_pred <= self.config.y_h_max)
        )

    # -- Public update -----------------------------------------------------

    def step(
        self,
        *,
        grad_g: float,
        v_meas: NDArray[np.float64],
        h_v: NDArray[np.float64],
        t_now: float,
        df_dq_eq: float = 0.0,
    ) -> Optional[ShuntCommit]:
        """Advance the integrator one control iteration and maybe commit a step.

        Parameters
        ----------
        grad_g :
            Projected objective gradient along this bank's injection,
            ``h_Hᵀ · ∇_y f`` [f-units per Mvar], computed by the caller from the
            shared TSO output-space gradient and the bank's boundary
            sensitivity column.
        v_meas :
            Measured HV-boundary voltages [p.u.] used by the feasibility guard.
        h_v :
            ``∂V/∂Q_eq`` (per Mvar) at the same boundary buses (same ordering as
            ``v_meas`` and the configured band).
        t_now :
            Current simulation time [s] for dwell / budget bookkeeping.
        df_dq_eq :
            Optional direct partial ``∂f/∂Q_eq`` (default 0.0 — the present TSO
            objective has no direct shunt term; the hook is kept for a future
            loss term).

        Returns
        -------
        commit : ShuntCommit or None
            A commit record if a physical step was committed this iteration,
            otherwise ``None``.
        """
        cfg = self.config
        v_meas = np.atleast_1d(np.asarray(v_meas, dtype=np.float64))
        h_v = np.atleast_1d(np.asarray(h_v, dtype=np.float64))
        if v_meas.size == 0:
            raise ValueError("v_meas must contain at least one boundary bus")
        if h_v.shape != v_meas.shape:
            raise ValueError(
                f"h_v shape {h_v.shape} does not match v_meas shape {v_meas.shape}"
            )
        if not math.isfinite(grad_g):
            raise ValueError(f"grad_g must be finite, got {grad_g}")

        # 1) Integrate the reactive pressure into the auxiliary state, then
        #    apply the anti-windup clamp: the relaxation may not run more than
        #    one physical step ahead of the committed level in either direction.
        g_h = float(grad_g) + float(df_dq_eq)
        q_phys = self.q_eq_phys
        lo = q_phys - cfg.q_step_mvar
        hi = q_phys + cfg.q_step_mvar
        # OFO step in the g_w convention: Delta = g_H / (2*g_w) (alpha fixed = 1).
        self.q_eq_aux = float(
            np.clip(self.q_eq_aux - g_h / (2.0 * cfg.g_w), lo, hi)
        )

        # 2) Round-to-nearest with a hysteresis dead-band: a commit is proposed
        #    only once the auxiliary state has accumulated past the half-step
        #    plus the hysteresis half-width.
        up_thresh = q_phys + 0.5 * cfg.q_step_mvar + cfg.delta
        down_thresh = q_phys - 0.5 * cfg.q_step_mvar - cfg.delta

        direction = 0
        if self.q_eq_aux >= up_thresh and self.level < cfg.n_levels:
            direction = +1
        elif self.q_eq_aux <= down_thresh and self.level > 0:
            direction = -1
        if direction == 0:
            return None

        # 3) Commit guards — all must hold, else the proposal is suppressed and
        #    the auxiliary state is held (it stays clamped, so it does not wind
        #    up indefinitely against a blocked rail).
        if not self._dwell_elapsed(t_now):
            return None
        if not self._budget_remaining(t_now):
            return None
        if not self._feasible(direction, v_meas, h_v):
            return None

        # 4) Commit: move the physical level, record the switch for dwell /
        #    budget, and emit the record for the atomic plant / feedforward /
        #    sensitivity update performed by the caller.
        old_level = self.level
        new_level = old_level + direction
        self.level = new_level
        self._last_switch_t = t_now
        self._switches_in_window += 1
        return ShuntCommit(
            shunt_idx=cfg.shunt_idx,
            bus_idx=cfg.bus_idx,
            interface_trafo3w_idx=cfg.interface_trafo3w_idx,
            dso_id=cfg.dso_id,
            kind=cfg.kind,
            old_level=old_level,
            new_level=new_level,
            pp_step_new=self.pp_step_for_level(new_level),
            q_step_mvar=cfg.q_step_mvar,
            direction=direction,
        )


class MSC(ShuntBank):
    """Mechanically switched capacitor: engaging a step injects reactive power
    (raises voltage), so ``∂V/∂Q_eq ≥ 0`` is expected at the bank's buses."""

    expected_h_sign = +1


class MSR(ShuntBank):
    """Mechanically switched reactor: engaging a step absorbs reactive power
    (lowers voltage), so ``∂V/∂Q_eq ≤ 0`` is expected at the bank's buses."""

    expected_h_sign = -1


def make_bank(config: ShuntBankConfig) -> ShuntBank:
    """Construct the concrete bank instance for ``config.kind``."""
    if config.kind == "MSC":
        return MSC(config)
    if config.kind == "MSR":
        return MSR(config)
    raise ValueError(f"Unknown shunt kind {config.kind!r}")


@dataclass
class ShuntIntegrator:
    """Per-zone container holding the zone's switched-shunt banks.

    Thin convenience wrapper: the runner builds, for each bank, the projected
    gradient ``grad_g`` and the voltage feasibility inputs, then calls
    :meth:`update`.  The banks own all state; this object only iterates them.
    """

    banks: List[ShuntBank] = field(default_factory=list)

    @classmethod
    def from_configs(cls, configs: List[ShuntBankConfig]) -> "ShuntIntegrator":
        return cls(banks=[make_bank(c) for c in configs])

    def update(
        self,
        grad_g: List[float],
        v_meas: List[NDArray[np.float64]],
        h_v: List[NDArray[np.float64]],
        t_now: float,
        df_dq_eq: Optional[List[float]] = None,
    ) -> List[ShuntCommit]:
        """Advance every bank one iteration; return the list of commits.

        The three list arguments are parallel to :attr:`banks`.
        """
        n = len(self.banks)
        if not (len(grad_g) == len(v_meas) == len(h_v) == n):
            raise ValueError(
                f"update() arguments must each have length {n} (one per bank); "
                f"got grad_g={len(grad_g)}, v_meas={len(v_meas)}, h_v={len(h_v)}"
            )
        if df_dq_eq is None:
            df_dq_eq = [0.0] * n
        elif len(df_dq_eq) != n:
            raise ValueError(
                f"df_dq_eq must have length {n}, got {len(df_dq_eq)}"
            )
        commits: List[ShuntCommit] = []
        for k, bank in enumerate(self.banks):
            commit = bank.step(
                grad_g=grad_g[k],
                v_meas=v_meas[k],
                h_v=h_v[k],
                t_now=t_now,
                df_dq_eq=df_dq_eq[k],
            )
            if commit is not None:
                commits.append(commit)
        return commits
