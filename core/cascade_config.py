"""
Centralized Cascade Simulation Configuration
=============================================

Defines :class:`CascadeConfig`, a flat dataclass that holds **every**
tunable parameter for the cascaded TSO-DSO OFO controller.  All
parameters that were previously hardcoded inside ``run_cascade()`` or
scattered across function arguments are now declared here with their
defaults.

Usage::

    from core.cascade_config import CascadeConfig

    config = CascadeConfig(
        v_setpoint_pu=1.05,
        n_minutes=720,
        g_v=100000,
        ...
    )
    result = run_cascade(config)

Author: Claude (generated)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from core.profiles import DEFAULT_PROFILES_CSV
from controller.reserve_observer import ReserveObserverConfig


# ═══════════════════════════════════════════════════════════════════════════════
#  ContingencyEvent (re-exported for convenience — canonical definition
#  remains in run_S_TSO_M_DSO.py so we avoid circular imports)
# ═══════════════════════════════════════════════════════════════════════════════

# We import ContingencyEvent lazily inside to_dict / from_dict to avoid
# circular dependency with run_S_TSO_M_DSO.py.


# ═══════════════════════════════════════════════════════════════════════════════
#  CascadeConfig
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CascadeConfig:
    """
    Complete configuration for a cascaded TSO-DSO OFO simulation.

    Every tunable parameter is exposed as a field with a sensible default.
    Construct one instance, pass it to ``run_cascade(config)``, and all
    parameters are set from this single object.

    The defaults match the values that were previously hardcoded in
    ``run_cascade()``.
    """

    # ── Simulation timing & environment ────────────────────────────────────
    v_setpoint_pu: float = 1.03
    """TSO voltage setpoint for transmission-level monitored buses [p.u.].
    Also used for ext_grid vm_pu and generator AVR initialisation."""

    dso_v_setpoint_pu: Optional[float] = 1.03
    """DSO voltage setpoint for distribution-level monitored buses [p.u.].
    If ``None`` (default), the TSO value ``v_setpoint_pu`` is used for
    both controllers (backward-compatible behaviour)."""

    n_minutes: int = 120
    """Total simulation duration [minutes].  Ignored when ``n_seconds`` is set."""

    tso_period_min: int = 3
    """TSO controller fires every N minutes.  Ignored when ``tso_period_s`` is set."""

    dso_period_min: int = 1
    """DSO controller fires every N minutes.  Ignored when ``dso_period_s`` is set."""

    # ── Seconds-based timing overrides (take precedence over _min fields) ──
    n_seconds: Optional[int] = None
    """Total simulation duration [seconds].  Overrides ``n_minutes`` when set."""

    tso_period_s: Optional[float] = None
    """TSO controller period [seconds].  Overrides ``tso_period_min`` when set."""

    dso_period_s: Optional[float] = None
    """DSO controller period [seconds].  Overrides ``dso_period_min`` when set."""

    start_time: datetime = field(default_factory=lambda: datetime(2016, 6, 10, 0, 0))
    """Simulation start time (for profile lookup)."""

    profiles_csv: str = DEFAULT_PROFILES_CSV
    """Path to the time-series profiles CSV file."""

    use_profiles: bool = True
    """Whether to apply time-varying load/generation profiles."""

    verbose: int = 1
    """Verbosity level (0=silent, 1=progress, 2=detailed)."""

    live_plot: bool = True
    """Whether to show real-time plots during simulation."""

    # ── TSO objective weights ──────────────────────────────────────────────
    g_v: float = 1000.0
    """TSO voltage tracking penalty weight:  g_v · Σ(V − V_set)²."""

    # ── DSO objective weights ──────────────────────────────────────────────
    g_q: float = 1.0
    """DSO interface Q tracking penalty:  g_q · Σ(Q − Q_set)²."""

    g_qi: float = 0.0
    """DSO integral Q-tracking weight (leaky integrator).  When > 0,
    accumulates Q-interface errors over iterations, building pressure
    for discrete switching actions (OLTC, shunts).  Default 0.0 (disabled)."""

    lambda_qi: float = 0.9
    """Decay factor for the DSO leaky integrator (0 ≤ λ ≤ 1).
    1.0 = pure integration (no decay), 0.9 = gradual decay."""

    q_integral_max_mvar: float = 50.0
    """Anti-windup clamp for the DSO integral accumulator [Mvar]."""

    dso_g_v: float = 100.0
    """DSO voltage tracking weight (soft secondary objective)."""

    # ── OFO algorithm parameters ──────────────────────────────────────────
    alpha: float = 1.0
    """OFO step-size gain.  α = 1 means the MIQP change Δu is applied
    directly without scaling."""

    g_z: float = 1e12
    """Slack variable penalty in the MIQP (z^T G_z z) for voltage and
    interface-Q outputs.  Very large to discourage constraint violation."""

    gz_tso_current: float = 1e3
    """Slack variable penalty for TSO current-limit outputs.  Much smaller
    than ``g_z`` because branch currents are only weakly controllable via
    reactive-power actuators.  A too-large value causes oscillations when
    current limits are hit."""

    gz_dso_current: float = 1e3
    """Slack variable penalty for DSO current-limit outputs."""

    # ── TSO g_w diagonal (per-actuator change damping) ────────────────────
    gw_tso_q_der: float = 0.4
    """g_w for TSO DER Q changes [Mvar]."""

    gw_tso_q_pcc: float = 0.2
    """g_w for TSO PCC Q-setpoint changes [Mvar]."""

    gw_tso_v_gen: float = 5e6
    """g_w for generator AVR setpoint changes [p.u.].  Very high = cautious."""

    gw_tso_oltc: float = 40.0
    """g_w for machine transformer OLTC tap changes."""

    gw_tso_shunt: float = 1000.0
    """g_w for TSO shunt switching."""

    gw_oltc_cross_tso: float = 0.0
    """OLTC cross-coupling weight for TSO:  g_cross · 𝟏𝟏ᵀ  sub-block.
    Penalises simultaneous OLTC tapping.  0 = disabled."""

    # ── DSO g_w diagonal (per-actuator change damping) ────────────────────
    gw_dso_q_der: float = 4.0
    """g_w for DSO DER Q changes [Mvar]."""

    gw_dso_oltc: float = 100.0
    """g_w for coupler (3W) OLTC tap changes."""

    gw_dso_shunt: float = 3000.0
    """g_w for DSO shunt switching."""

    gw_oltc_cross_dso: float = 0.0
    """OLTC cross-coupling weight for DSO.  0 = disabled."""

    # ── Per-actuator overrides (optional vectors) ─────────────────────────
    gw_tso_override: Optional[NDArray[np.float64]] = None
    """Complete g_w vector for TSO. If set, overrides build_gw_tso_diag()."""

    gw_dso_override: Optional[NDArray[np.float64]] = None
    """Complete g_w vector for DSO. If set, overrides build_gw_dso_diag()."""

    gu_tso_override: Optional[NDArray[np.float64]] = None
    """Complete g_u vector for TSO."""

    gu_dso_override: Optional[NDArray[np.float64]] = None
    """Complete g_u vector for DSO."""

    # ── g_u usage penalties (per-actuator absolute level) ─────────────────
    gu_tso_q_der: float = 0.0
    """TSO DER Q usage regularisation (penalises |Q| level, not change)."""

    gu_tso_q_pcc: float = 0.0
    gu_tso_v_gen: float = 0.0
    gu_tso_oltc: float = 0.0
    gu_tso_oltc: float = 0.0
    gu_tso_shunt: float = 0.0

    gu_dso_q_der: float = 0.0
    """DSO DER Q usage regularisation."""

    gu_dso_oltc: float = 0.0
    gu_dso_shunt: float = 0.0

    # ── Generator capability curve parameters (Milano §12.2.1) ────────────
    gen_xd_pu: float = 1.2
    """Direct-axis synchronous reactance [p.u.]."""

    gen_i_f_max_pu: float = 2.65
    """Maximum field current [p.u.] (turbo-generator)."""

    gen_beta: float = 0.15
    """Under-excitation limit slope."""

    gen_q0_pu: float = 0.4
    """Under-excitation limit Q-axis offset [p.u.]."""

    # ── Reserve Observer ──────────────────────────────────────────────────
    enable_reserve_observer: bool = True
    """Whether the per-interface reserve observer is active."""

    reserve_q_threshold_mvar: float = 40.0
    """DER Q contribution above which shunt engagement is forced [Mvar]."""

    reserve_q_release_mvar: float = -40.0
    """DER Q contribution below which the engaged shunt may be released."""

    reserve_cooldown_min: int = 15
    """Minimum minutes between consecutive engage/release actions.
    Ignored when ``reserve_cooldown_s`` is set."""

    reserve_cooldown_s: Optional[float] = None
    """Minimum seconds between consecutive engage/release actions.
    Overrides ``reserve_cooldown_min`` when set."""

    # ── DSO OLTC initialisation ───────────────────────────────────────────
    dso_oltc_init_tol_pu: float = 0.01
    """Tolerance band for pandapower DiscreteTapControl initialisation."""

    # ── Achieved-Value Tracking ────────────────────────────────────────────
    k_t_avt: float = 1.0
    """Achieved-Value Tracking factor for TSO PCC-Q reset.
    1.0 = full reset to measured Q (recommended), 0.0 = disabled."""

    # ── Dwell-time stability analysis ────────────────────────────────────
    dwell_time_epsilon: float = 1
    """Convergence tolerance for the dwell-time stability formula.
    Used to compute the minimum cooldown T_dwell such that the
    continuous sub-problem contracts the discrete perturbation
    below epsilon."""

    int_cooldown: int = 6
    """Number of OFO iterations a discrete actuator is locked after switching.
    Used both by the controller (BaseOFOController) and the stability
    analysis (compared against the theoretically required T_dwell)."""

    int_max_step: int = 1
    """Maximum discrete step per OFO iteration (e.g. 1 tap for OLTCs)."""

    # ── Contingency events ────────────────────────────────────────────────
    contingencies: List = field(default_factory=list)
    """List of ContingencyEvent objects to inject during simulation.
    Typed as List (not List[ContingencyEvent]) to avoid circular import."""

    # ═══════════════════════════════════════════════════════════════════════
    #  Derived properties
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def effective_dso_v_setpoint_pu(self) -> float:
        """Return the DSO voltage setpoint, falling back to ``v_setpoint_pu``."""
        return self.dso_v_setpoint_pu if self.dso_v_setpoint_pu is not None else self.v_setpoint_pu

    @property
    def effective_tso_period_s(self) -> float:
        """TSO period in seconds (uses ``tso_period_s`` if set, else ``tso_period_min * 60``)."""
        return self.tso_period_s if self.tso_period_s is not None else self.tso_period_min * 60.0

    @property
    def effective_dso_period_s(self) -> float:
        """DSO period in seconds (uses ``dso_period_s`` if set, else ``dso_period_min * 60``)."""
        return self.dso_period_s if self.dso_period_s is not None else self.dso_period_min * 60.0

    @property
    def effective_n_seconds(self) -> int:
        """Total simulation duration in seconds."""
        return self.n_seconds if self.n_seconds is not None else self.n_minutes * 60

    @property
    def effective_reserve_cooldown_s(self) -> float:
        """Reserve observer cooldown in seconds."""
        return self.reserve_cooldown_s if self.reserve_cooldown_s is not None else self.reserve_cooldown_min * 60.0

    @property
    def effective_sim_step_s(self) -> float:
        """Simulation timestep in seconds (smallest controller period)."""
        return min(self.effective_tso_period_s, self.effective_dso_period_s)

    @property
    def uses_sub_minute_timing(self) -> bool:
        """True if any period is sub-minute (seconds-based)."""
        return self.effective_sim_step_s < 60.0

    # ═══════════════════════════════════════════════════════════════════════
    #  Builder methods — construct runtime numpy vectors from scalar fields
    # ═══════════════════════════════════════════════════════════════════════

    def build_gw_tso_diag(
        self, n_der: int, n_pcc: int, n_gen: int, n_oltc: int, n_shunt: int,
    ) -> NDArray[np.float64]:
        """Build the TSO g_w diagonal vector: [Q_DER | Q_PCC | V_gen | s_OLTC | s_shunt]."""
        if self.gw_tso_override is not None:
            return self.gw_tso_override
        return np.concatenate([
            np.full(n_der,   self.gw_tso_q_der),
            np.full(n_pcc,   self.gw_tso_q_pcc),
            np.full(n_gen,   self.gw_tso_v_gen),
            np.full(n_oltc,  self.gw_tso_oltc),
            np.full(n_shunt, self.gw_tso_shunt),
        ])

    def build_gw_dso_diag(
        self, n_der: int, n_oltc: int, n_shunt: int,
    ) -> NDArray[np.float64]:
        """Build the DSO g_w diagonal vector: [Q_DER | s_OLTC | s_shunt]."""
        if self.gw_dso_override is not None:
            return self.gw_dso_override
        return np.concatenate([
            np.full(n_der,   self.gw_dso_q_der),
            np.full(n_oltc,  self.gw_dso_oltc),
            np.full(n_shunt, self.gw_dso_shunt),
        ])

    def build_gu_tso(
        self, n_der: int, n_pcc: int, n_gen: int, n_oltc: int, n_shunt: int,
    ) -> NDArray[np.float64]:
        """Build the TSO g_u vector (usage/level penalties)."""
        if self.gu_tso_override is not None:
            return self.gu_tso_override
        return np.concatenate([
            np.full(n_der,   self.gu_tso_q_der),
            np.full(n_pcc,   self.gu_tso_q_pcc),
            np.full(n_gen,   self.gu_tso_v_gen),
            np.full(n_oltc,  self.gu_tso_oltc),
            np.full(n_shunt, self.gu_tso_shunt),
        ])

    def build_gu_dso(
        self, n_der: int, n_oltc: int, n_shunt: int,
    ) -> NDArray[np.float64]:
        """Build the DSO g_u vector (usage/level penalties)."""
        if self.gu_dso_override is not None:
            return self.gu_dso_override
        return np.concatenate([
            np.full(n_der,   self.gu_dso_q_der),
            np.full(n_oltc,  self.gu_dso_oltc),
            np.full(n_shunt, self.gu_dso_shunt),
        ])

    def build_gz_tso(self, n_v: int, n_i: int) -> NDArray[np.float64]:
        """Build per-output g_z vector for the TSO controller.

        Output ordering: [ V_bus | I_line ]
        """
        return np.concatenate([
            np.full(n_v, self.g_z),
            np.full(n_i, self.gz_tso_current),
        ])

    def build_gz_dso(self, n_iface: int, n_v: int, n_i: int) -> NDArray[np.float64]:
        """Build per-output g_z vector for the DSO controller.

        Output ordering: [ Q_interface | V_bus | I_line ]
        """
        return np.concatenate([
            np.full(n_iface, self.g_z),
            np.full(n_v,     self.g_z),
            np.full(n_i,     self.gz_dso_current),
        ])

    def build_reserve_observer_config(
        self, shunt_q_steps_mvar: List[float],
    ) -> ReserveObserverConfig:
        """Build a ReserveObserverConfig from flat config fields + runtime shunt data."""
        return ReserveObserverConfig(
            q_threshold_mvar=self.reserve_q_threshold_mvar,
            q_release_mvar=self.reserve_q_release_mvar,
            shunt_q_steps_mvar=shunt_q_steps_mvar,
            cooldown_min=self.reserve_cooldown_min,
            cooldown_s=self.reserve_cooldown_s,
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  Serialisation
    # ═══════════════════════════════════════════════════════════════════════

    def to_dict(self) -> Dict:
        """
        Convert to a JSON-serializable dictionary.

        Handles non-JSON types: datetime → ISO string,
        ContingencyEvent list → list of plain dicts.
        """
        d: Dict = {}

        # Simulation
        d["v_setpoint_pu"] = self.v_setpoint_pu
        d["dso_v_setpoint_pu"] = self.dso_v_setpoint_pu
        d["n_minutes"] = self.n_minutes
        d["tso_period_min"] = self.tso_period_min
        d["dso_period_min"] = self.dso_period_min
        if self.n_seconds is not None:
            d["n_seconds"] = self.n_seconds
        if self.tso_period_s is not None:
            d["tso_period_s"] = self.tso_period_s
        if self.dso_period_s is not None:
            d["dso_period_s"] = self.dso_period_s
        d["start_time"] = self.start_time.isoformat()
        d["profiles_csv"] = self.profiles_csv
        d["use_profiles"] = self.use_profiles
        d["verbose"] = self.verbose
        d["live_plot"] = self.live_plot

        # Objective weights
        d["g_v"] = self.g_v
        d["g_q"] = self.g_q
        d["g_qi"] = self.g_qi
        d["lambda_qi"] = self.lambda_qi
        d["q_integral_max_mvar"] = self.q_integral_max_mvar
        d["dso_g_v"] = self.dso_g_v

        # OFO
        d["alpha"] = self.alpha
        d["g_z"] = self.g_z
        d["gz_tso_current"] = self.gz_tso_current
        d["gz_dso_current"] = self.gz_dso_current

        # TSO g_w
        d["gw_tso_q_der"] = self.gw_tso_q_der
        d["gw_tso_q_pcc"] = self.gw_tso_q_pcc
        d["gw_tso_v_gen"] = self.gw_tso_v_gen
        d["gw_tso_oltc"] = self.gw_tso_oltc
        d["gw_tso_shunt"] = self.gw_tso_shunt
        d["gw_oltc_cross_tso"] = self.gw_oltc_cross_tso

        # DSO g_w
        d["gw_dso_q_der"] = self.gw_dso_q_der
        d["gw_dso_oltc"] = self.gw_dso_oltc
        d["gw_dso_shunt"] = self.gw_dso_shunt
        d["gw_oltc_cross_dso"] = self.gw_oltc_cross_dso

        # g_u
        d["gu_tso_q_der"] = self.gu_tso_q_der
        d["gu_tso_q_pcc"] = self.gu_tso_q_pcc
        d["gu_tso_v_gen"] = self.gu_tso_v_gen
        d["gu_tso_oltc"] = self.gu_tso_oltc
        d["gu_tso_shunt"] = self.gu_tso_shunt
        d["gu_dso_q_der"] = self.gu_dso_q_der
        d["gu_dso_oltc"] = self.gu_dso_oltc
        d["gu_dso_shunt"] = self.gu_dso_shunt

        # Generator
        d["gen_xd_pu"] = self.gen_xd_pu
        d["gen_i_f_max_pu"] = self.gen_i_f_max_pu
        d["gen_beta"] = self.gen_beta
        d["gen_q0_pu"] = self.gen_q0_pu

        # Reserve Observer
        d["enable_reserve_observer"] = self.enable_reserve_observer
        d["reserve_q_threshold_mvar"] = self.reserve_q_threshold_mvar
        d["reserve_q_release_mvar"] = self.reserve_q_release_mvar
        d["reserve_cooldown_min"] = self.reserve_cooldown_min
        if self.reserve_cooldown_s is not None:
            d["reserve_cooldown_s"] = self.reserve_cooldown_s

        # DSO OLTC init
        d["dso_oltc_init_tol_pu"] = self.dso_oltc_init_tol_pu

        # Achieved-Value Tracking
        d["k_t_avt"] = self.k_t_avt

        # Contingencies
        d["contingencies"] = [
            {
                "minute": c.minute,
                "element_type": c.element_type,
                "element_index": c.element_index,
                "action": c.action,
                **({"time_s": c.time_s} if c.time_s is not None else {}),
            }
            for c in self.contingencies
        ]

        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict) -> "CascadeConfig":
        """
        Reconstruct a CascadeConfig from a dictionary (e.g. loaded from JSON).

        Handles datetime parsing and ContingencyEvent reconstruction.
        """
        from run.records import ContingencyEvent

        kwargs = dict(d)  # shallow copy

        # Parse datetime
        if "start_time" in kwargs and isinstance(kwargs["start_time"], str):
            kwargs["start_time"] = datetime.fromisoformat(kwargs["start_time"])

        # Parse contingencies
        if "contingencies" in kwargs:
            kwargs["contingencies"] = [
                ContingencyEvent(**c) for c in kwargs["contingencies"]
            ]

        # Remove any metadata keys that aren't fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        return cls(**kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> "CascadeConfig":
        """Reconstruct from a JSON string."""
        return cls.from_dict(json.loads(json_str))
