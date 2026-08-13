"""
Actuator Bounds Module
======================

This module defines the ActuatorBounds class, which calculates operating-point-
dependent bounds for all actuators controlled by an OFO controller.

The bounds are used as input constraints in the MIQP optimisation problem.
DER reactive power bounds depend on the current active power output via the
capability curve defined in grid codes (e.g., VDE-AR-N 4120).

Synchronous generator limits are modelled using the detailed capability
curve from Milano (2010), §12.2.1, comprising three thermal constraints:
    (i)   Stator current limit   –  p² + q² ≤ s_max²
    (ii)  Rotor current limit    –  p² + (q + v²/xd)² ≤ (v·i_f_max/xd)²
    (iii) Under-excitation limit –  q ≥ −q₀(v) + β·p_max

References
----------
[1] F. Milano, *Power System Modelling and Scripting*, Springer, 2010,
    Chapter 12, Eqs. (12.7)–(12.11).

Author: Manuel Schwenke
Date: 2025-02-05
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import math
import numpy as np
from numpy.typing import NDArray


#: Ablation override for the DER operating diagram, in pu of S_n.
#: ``None`` (default) uses the real diagram (STATCOM circle / VDE box).
#: When set to a float ``x``, EVERY DER gets the symmetric box
#: ``[-x*S_n, +x*S_n]`` regardless of ``op_diagram`` and P.
#:
#: This is a *diagnostic* knob, not a modelling choice: it answers "does the
#: capability constraint cause the DSO's steady-state interface-Q offset?"
#: by removing the constraint entirely.  Introduced 2026-07-21 after a
#: per-park audit found 12 of 44 parks pinned at an individual limit while
#: the *aggregate* headroom looked ample (DSO_1 134.1 of 168.1 Mvar with 5/10
#: parks saturated), i.e. aggregate numbers hid the binding constraint.
#:
#: MUST be honoured identically by the controller's bound computation
#: (``ActuatorBounds._compute_single_der_q_capability``) and by the plant's
#: clip (``controller.der_qv_local_loop._qv_capability``).  If the two ever
#: disagree the controller optimises against a plant it does not have --
#: the exact failure mode that produced the Gate-E STATCOM bug.
DER_Q_CAPABILITY_OVERRIDE_PU: Optional[float] = None


def set_der_q_capability_override(value: Optional[float]) -> None:
    """Set (or clear with ``None``) the global DER capability override."""
    global DER_Q_CAPABILITY_OVERRIDE_PU
    DER_Q_CAPABILITY_OVERRIDE_PU = (
        None if value is None else float(value)
    )


#: Diagnostic override of the DER Q(V) *deadband* [pu], applied to BOTH plants
#: (static QVLocalLoop via ``net.sgen.qv_deadband_pu`` and RMS QVPRE via
#: ``_anchor_qv_precontrollers``).  The nominal 0.01 pu deadband puts the parks
#: at their dead-zone edge at the profiled operating point, where the droop is
#: multi-valued (with the ZIP loads): the two solvers settle in different
#: basins and the DSO_4 interface-Q setpoints diverge (2026-07-24 finding).
#: Setting this to 0 removes the dead zone so the droop is single-valued and
#: both plants converge to the same equilibrium.  ``None`` keeps each park's
#: own snapshot/config deadband.
DER_QV_DEADBAND_OVERRIDE_PU: Optional[float] = None


def set_der_qv_deadband_override(value: Optional[float]) -> None:
    """Set (or clear with ``None``) the global DER Q(V) deadband override."""
    global DER_QV_DEADBAND_OVERRIDE_PU
    DER_QV_DEADBAND_OVERRIDE_PU = (
        None if value is None else float(value)
    )


#: Per-park DER Q(V) deadband [pu], keyed by pandapower sgen index.
#:
#: The RMS plant anchors its Q(V) pre-controllers from the *exported snapshot*,
#: which ``export.make_snapshots`` writes using ``MultiTSOConfig()`` DEFAULTS --
#: not the deadband of the run being executed.  The blanket scalar above was
#: therefore the only channel by which a per-run deadband reached PowerFactory,
#: and being a single number it cannot express ``delta_TS != delta_DS``.
#:
#: This map carries the per-level (and per-park-override) values that
#: ``tag_der_q_modes`` has already written into ``net.sgen.qv_deadband_pu`` on
#: the static side, so both plants are driven from ONE column rather than from
#: two independently-derived numbers.  Precedence in ``pf.plant``:
#: blanket scalar > this map > snapshot value.
DER_QV_DEADBAND_BY_SGEN_PU: Dict[int, float] = {}


def set_der_qv_deadband_by_sgen(mapping: Optional[Dict[int, float]]) -> None:
    """Publish the per-sgen Q(V) deadband map (``None``/empty clears it)."""
    global DER_QV_DEADBAND_BY_SGEN_PU
    DER_QV_DEADBAND_BY_SGEN_PU = (
        {} if not mapping else {int(k): float(v) for k, v in mapping.items()}
    )


#: Per-park DER Q(V) DROOP [pu], keyed by pandapower sgen index.
#:
#: Same defect, same remedy as the deadband map above: the RMS plant anchors
#: its Q(V) pre-controllers from the exported snapshot, which
#: ``export.make_snapshots`` writes using ``MultiTSOConfig()`` DEFAULTS.  So
#: ``--der-slope`` changed the static plant (via ``tag_der_q_modes`` ->
#: ``net.sgen.qv_slope_pu``) while the RMS plant silently kept the snapshot's
#: 0.06, and a droop sweep would have compared two different plants.  Harmless
#: until now only because both defaults happened to be 0.06.
#:
#: NOTE ON THE NAME: ``qv_slope_pu`` is the DROOP, not a gain.  It enters as a
#: divisor of the voltage error -- static ``R = S_n/slope`` [Mvar/pu_v], RMS
#: ``Kdroop = 1/slope`` -- so ``slope = 0.06`` means 0.06 pu of voltage
#: deviation commands full rated Q, i.e. a 6 % droop.  The grid code permits
#: 5-15 %.
DER_QV_SLOPE_BY_SGEN_PU: Dict[int, float] = {}


def set_der_qv_slope_by_sgen(mapping: Optional[Dict[int, float]]) -> None:
    """Publish the per-sgen Q(V) droop map (``None``/empty clears it)."""
    global DER_QV_SLOPE_BY_SGEN_PU
    DER_QV_SLOPE_BY_SGEN_PU = (
        {} if not mapping else {int(k): float(v) for k, v in mapping.items()}
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Synchronous Generator Capability Curve  (Milano §12.2.1)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeneratorParameters:
    """
    Physical parameters of a synchronous generator for capability-curve
    computation.

    All electrical quantities are in *per-unit on the machine MVA base*
    unless noted otherwise.

    Attributes
    ----------
    s_rated_mva : float
        Rated apparent power [MVA].  Used to convert between p.u. and
        physical units.
    p_max_mw : float
        Maximum active power output [MW].  Equals the turbine rating.
    p_min_mw : float
        Minimum technical active power output [MW].  Turbine minimum load.
        Typical: 30–40 % of p_max_mw for steam turbines.  Default 0.0.
    xd_pu : float
        Direct-axis synchronous reactance [p.u.].
        Typical: 1.0–1.8 for turbo-generators, 0.6–1.2 for salient-pole.
    i_f_max_pu : float
        Maximum field current [p.u.].
        Typical: 2.6–2.73 for turbo-generators (Milano eq. 12.10),
                 1.7–1.79 for salient-pole (Milano eq. 12.9).
    beta : float
        Under-excitation limit slope parameter [p.u./p.u.].
        Typical: 0.1–0.2 (Milano p. 293).
    q0_pu : float
        Under-excitation limit offset at nominal voltage [p.u.].
        Typical: ≈ 0.4 (Milano p. 293).
        Voltage-dependent: q₀(v) ≈ q0_pu · v² (proportional to V²).
    """
    s_rated_mva: float
    p_max_mw: float
    p_min_mw: float = 0.0
    xd_pu: float = 1.8
    i_f_max_pu: float = 2.7
    beta: float = 0.15
    q0_pu: float = 0.4

    def __post_init__(self) -> None:
        # Coherence of the (xd, i_f_max) pair.  The rotor-limit circle of
        # ``compute_generator_q_limits`` is centred at -v²/xd with radius
        # v·i_f_max/xd, so at p = 0, v = 1 it reaches (i_f_max - 1)/xd.  If that
        # exceeds the stator radius the rotor circle lies wholly outside the
        # stator circle and the field limit never binds — the capability curve
        # silently degenerates to stator + UEL.  Fail fast instead (a mixed
        # turbo/salient-pole pairing such as xd = 1.2 with i_f_max = 2.65 is the
        # typical way this happens).
        if self.xd_pu <= 0.0:
            raise ValueError(f"xd_pu must be > 0, got {self.xd_pu}")
        if (self.i_f_max_pu - 1.0) / self.xd_pu >= 1.0:
            raise ValueError(
                f"Incoherent capability parameters: xd_pu={self.xd_pu}, "
                f"i_f_max_pu={self.i_f_max_pu} give a rotor limit of "
                f"{(self.i_f_max_pu - 1.0) / self.xd_pu:.3f} p.u. at p=0, which "
                f"does not bind inside the stator circle.  Require "
                f"xd_pu > i_f_max_pu - 1."
            )

    @classmethod
    def round_rotor(
        cls, s_rated_mva: float, p_max_mw: float, p_min_mw: float = 0.0,
    ) -> "GeneratorParameters":
        """Large turbogenerator (cylindrical rotor): steam/gas and nuclear sets.

        xd = 1.8 p.u., i_f_max = 2.7 p.u. (Milano eq. 12.10, ranges 1.0-1.8 and
        2.6-2.73).  Rotor limit binds at (2.7-1)/1.8 = 0.94 p.u. at zero load.
        """
        return cls(
            s_rated_mva=s_rated_mva, p_max_mw=p_max_mw, p_min_mw=p_min_mw,
            xd_pu=1.8, i_f_max_pu=2.7, beta=0.15, q0_pu=0.4,
        )

    @classmethod
    def salient_pole(
        cls, s_rated_mva: float, p_max_mw: float, p_min_mw: float = 0.0,
    ) -> "GeneratorParameters":
        """Salient-pole machine (hydro sets).

        xd = 1.0 p.u., i_f_max = 1.79 p.u. (Milano eq. 12.9, ranges 0.6-1.2 and
        1.7-1.79).  Rotor limit binds at (1.79-1)/1.0 = 0.79 p.u. at zero load.
        Note that xd = 0.6 with i_f_max = 1.7 would NOT bind; keep xd >= 0.8.
        """
        return cls(
            s_rated_mva=s_rated_mva, p_max_mw=p_max_mw, p_min_mw=p_min_mw,
            xd_pu=1.0, i_f_max_pu=1.79, beta=0.15, q0_pu=0.4,
        )


#: ``net.gen["type"]`` values (from ``network.ieee39.constants.GEN_NAMEPLATE``)
#: that denote salient-pole machines.  Everything else — including the
#: aggregated "Equivalent" slack anchor — is treated as round rotor.
SALIENT_POLE_GEN_TYPES = frozenset({"Hydro"})


def generator_parameters_for_type(
    gen_type: str, s_rated_mva: float, p_max_mw: float, p_min_mw: float = 0.0,
) -> GeneratorParameters:
    """Select the capability parameter set from the machine type label."""
    if str(gen_type) in SALIENT_POLE_GEN_TYPES:
        return GeneratorParameters.salient_pole(s_rated_mva, p_max_mw, p_min_mw)
    return GeneratorParameters.round_rotor(s_rated_mva, p_max_mw, p_min_mw)


def compute_generator_q_limits(
    params: GeneratorParameters,
    p_mw: float,
    v_pu: float = 1.0,
) -> tuple[float, float]:
    """
    Compute the reactive power limits of a synchronous generator at a
    given operating point using the detailed capability curve.

    The three constraints (all in p.u. on the machine base) are:

    1. **Stator current limit** (Milano eq. 12.7):
       p² + q² ≤ s_max²

    2. **Rotor current limit** (Milano eq. 12.8):
       p² + (q + v²/xd)² ≤ (v · i_f_max / xd)²

    3. **Under-excitation limit** (Milano eq. 12.11):
       q ≥ −q₀(v) + β · p      (line of slope β in P; incline ≈ arctan β)

    Parameters
    ----------
    params : GeneratorParameters
        Machine parameters.
    p_mw : float
        Current active power output [MW].
    v_pu : float
        Terminal voltage magnitude [p.u.].  Default 1.0.

    Returns
    -------
    q_min_mvar : float
        Minimum reactive power (under-excited / absorbing) [Mvar].
    q_max_mvar : float
        Maximum reactive power (over-excited / injecting) [Mvar].
    """
    s_base = params.s_rated_mva
    if s_base <= 0:
        return 0.0, 0.0

    # Convert to per-unit on machine base
    p_pu = p_mw / s_base
    s_max_pu = 1.0  # by definition of s_rated

    xd = params.xd_pu
    i_f_max = params.i_f_max_pu

    # ------------------------------------------------------------------
    # q_max: minimum of stator limit and rotor limit (both give upper Q)
    # ------------------------------------------------------------------

    # (1) Stator current limit:  q_max_stator = sqrt(s_max² − p²)
    disc_stator = s_max_pu ** 2 - p_pu ** 2
    if disc_stator > 0:
        q_max_stator = math.sqrt(disc_stator)
    else:
        q_max_stator = 0.0

    # (2) Rotor current limit:
    #     p² + (q + v²/xd)² ≤ (v·i_f_max/xd)²
    #     => q ≤ -v²/xd + sqrt((v·i_f_max/xd)² − p²)
    rotor_radius = v_pu * i_f_max / xd
    disc_rotor = rotor_radius ** 2 - p_pu ** 2
    if disc_rotor > 0:
        q_max_rotor = -v_pu ** 2 / xd + math.sqrt(disc_rotor)
    else:
        q_max_rotor = -v_pu ** 2 / xd  # degenerate: p exceeds rotor circle

    q_max_pu = min(q_max_stator, q_max_rotor)

    # ------------------------------------------------------------------
    # q_min: maximum of stator limit (lower) and under-excitation limit
    # ------------------------------------------------------------------

    # (1) Stator current limit (lower branch):  q_min_stator = -sqrt(s_max² − p²)
    q_min_stator = -q_max_stator  # symmetric for stator

    # (3) Under-excitation limit (Milano eq. 12.11):
    #     q ≥ -q₀(v) + β · p   (straight line of slope β in the P-Q plane;
    #     under-excitation incline ≈ arctan β — uses the RUNNING p, not p_max,
    #     so the limit tilts with active power instead of staying flat)
    #     q₀(v) ≈ q0_pu · v²  (voltage-dependent offset)
    q0_v = params.q0_pu * v_pu ** 2
    q_min_ue = -q0_v + params.beta * p_pu

    q_min_pu = max(q_min_stator, q_min_ue)

    # Ensure q_min ≤ q_max
    if q_min_pu > q_max_pu:
        mid = 0.5 * (q_min_pu + q_max_pu)
        q_min_pu = mid
        q_max_pu = mid

    # Convert back to Mvar
    q_min_mvar = q_min_pu * s_base
    q_max_mvar = q_max_pu * s_base

    return q_min_mvar, q_max_mvar


class ActuatorBounds:
    """
    Calculator for operating-point-dependent actuator bounds.

    This class computes the lower and upper bounds for all actuators at the
    current operating point.  For DERs, the reactive power bounds depend on
    the current active power output.  For synchronous generators, the Q
    bounds depend on P and terminal voltage via the detailed capability
    curve (Milano §12.2.1).  For discrete actuators (OLTCs, shunts), the
    bounds are fixed.

    Attributes
    ----------
    der_indices : NDArray[np.int64]
        Indices of controllable DERs.
    der_s_rated_mva : NDArray[np.float64]
        Rated apparent power of each DER in MVA.
    der_p_max_mw : NDArray[np.float64]
        Maximum active power (installed capacity) of each DER in MW.
    oltc_indices : NDArray[np.int64]
        Indices of controllable OLTCs.
    oltc_tap_min : NDArray[np.int64]
        Minimum tap position for each OLTC.
    oltc_tap_max : NDArray[np.int64]
        Maximum tap position for each OLTC.
    shunt_indices : NDArray[np.int64]
        Indices of controllable shunts.
    shunt_q_mvar : NDArray[np.float64]
        Rated reactive power of each shunt in Mvar (positive for reactors).
    gen_params : list[GeneratorParameters] | None
        Per-generator physical parameters for capability-curve computation.
        ``None`` if no synchronous generators are modelled.
    """

    def __init__(
        self,
        der_indices: NDArray[np.int64],
        der_s_rated_mva: NDArray[np.float64],
        der_p_max_mw: NDArray[np.float64],
        oltc_indices: NDArray[np.int64],
        oltc_tap_min: NDArray[np.int64],
        oltc_tap_max: NDArray[np.int64],
        shunt_indices: NDArray[np.int64],
        shunt_q_mvar: NDArray[np.float64],
        gen_params: Optional[list[GeneratorParameters]] = None,
        der_op_diagrams: Optional[list[str]] = None,
    ) -> None:
        """
        Initialise ActuatorBounds with static actuator parameters.

        Parameters
        ----------
        der_indices : NDArray[np.int64]
            Indices of controllable DERs.
        der_s_rated_mva : NDArray[np.float64]
            Rated apparent power of each DER in MVA.
        der_p_max_mw : NDArray[np.float64]
            Maximum active power of each DER in MW.
        oltc_indices : NDArray[np.int64]
            Indices of controllable OLTCs.
        oltc_tap_min : NDArray[np.int64]
            Minimum tap position for each OLTC.
        oltc_tap_max : NDArray[np.int64]
            Maximum tap position for each OLTC.
        shunt_indices : NDArray[np.int64]
            Indices of controllable shunts.
        shunt_q_mvar : NDArray[np.float64]
            Rated reactive power of each shunt in Mvar.
        gen_params : list[GeneratorParameters] | None, optional
            Per-generator physical parameters for the detailed capability
            curve.  If ``None`` (default), generator Q bounds are not
            available and :meth:`compute_gen_q_bounds` will raise.
        der_op_diagrams : list[str] | None, optional
            Operating diagram type per DER.  Supported values:
            ``'VDE-AR-N-4120-v2'`` (default) — piecewise-linear with dead
            zone below P/S_n = 0.1.
            ``'STATCOM'`` — full circle diagram Q = ±sqrt(S_n² - P²),
            no dead zone at P = 0.  Models Type-4 wind parks with
            STATCOM-class full-converter capability.
        """
        self.der_indices = der_indices
        self.der_s_rated_mva = der_s_rated_mva
        self.der_p_max_mw = der_p_max_mw
        self.oltc_indices = oltc_indices
        self.oltc_tap_min = oltc_tap_min
        self.oltc_tap_max = oltc_tap_max
        self.shunt_indices = shunt_indices
        self.shunt_q_mvar = shunt_q_mvar
        self.gen_params = gen_params
        n_der = len(der_indices)
        if der_op_diagrams is not None:
            self.der_op_diagrams = list(der_op_diagrams)
        else:
            self.der_op_diagrams = ['VDE-AR-N-4120-v2'] * n_der

    def compute_der_q_bounds(
        self,
        der_p_current_mw: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute DER reactive power bounds based on current active power output.

        The bounds are derived from the VDE-AR-N 4120 Variant 2 capability
        curve for HV-connected DERs.  Both the P normalisation and Q limits
        use the rated apparent power S_n as reference, matching the standard
        definition where the diagram axes are P/S_n and Q/S_n.

        Parameters
        ----------
        der_p_current_mw : NDArray[np.float64]
            Current active power output of each DER in MW.

        Returns
        -------
        q_min_mvar : NDArray[np.float64]
            Minimum reactive power for each DER in Mvar (underexcited/consuming).
        q_max_mvar : NDArray[np.float64]
            Maximum reactive power for each DER in Mvar (overexcited/producing).

        Notes
        -----
        Sign convention:
            - Positive Q: overexcited (producing/injecting reactive power)
            - Negative Q: underexcited (consuming/absorbing reactive power)

        VDE-AR-N 4120 Variant 2 breakpoints (in p.u. of S_n):
            - P/S_n < 0.1:  Q = 0 (dead zone)
            - P/S_n = 0.1:  Q in [-0.10, +0.10] * S_n
            - P/S_n = 0.2:  Q in [-0.33, +0.41] * S_n
            - P/S_n >= 0.2: Q in [-0.33, +0.41] * S_n
        """
        n_der = len(self.der_indices)
        q_min_mvar = np.zeros(n_der)
        q_max_mvar = np.zeros(n_der)

        for i in range(n_der):
            p_current = der_p_current_mw[i]
            s_rated = self.der_s_rated_mva[i]

            # Normalise P by S_rated (VDE convention: P/S_n)
            if s_rated > 0.0:
                p_ratio = abs(p_current) / s_rated
            else:
                p_ratio = 0.0

            # Compute Q capability based on operating diagram type
            op_diag = self.der_op_diagrams[i]
            q_min, q_max = self._compute_single_der_q_capability(
                p_ratio=p_ratio,
                s_rated_mva=s_rated,
                op_diagram=op_diag,
            )

            q_min_mvar[i] = q_min
            q_max_mvar[i] = q_max

        return q_min_mvar, q_max_mvar
    
    def _compute_single_der_q_capability(
        self,
        p_ratio: float,
        s_rated_mva: float,
        op_diagram: str = 'VDE-AR-N-4120-v2',
    ) -> tuple[float, float]:
        """
        Compute Q capability for a single DER based on its P ratio.

        Supports two operating diagram types:

        **VDE-AR-N-4120-v2** (default) — piecewise-linear with dead zone:

        ====== ================ ================
        P/Sn     Q_min/Sn         Q_max/Sn
        ====== ================ ================
        < 0.1    0.0              0.0
        0.1      -0.10            +0.10
        0.2      -0.33            +0.41
        >= 0.2   -0.33            +0.41
        ====== ================ ================

        **STATCOM** — full circle diagram (no dead zone):

            Q_max = +sqrt(S_n² - P²)
            Q_min = -sqrt(S_n² - P²)

        Models Type-4 (full converter) wind parks with STATCOM-class
        grid-forming inverters.  Full rated apparent power is available
        as reactive power when P = 0.

        Parameters
        ----------
        p_ratio : float
            Ratio of current active power to rated apparent power (P/S_n).
        s_rated_mva : float
            Rated apparent power of the DER in MVA.
        op_diagram : str
            Operating diagram type: ``'VDE-AR-N-4120-v2'`` or ``'STATCOM'``.

        Returns
        -------
        q_min : float
            Minimum reactive power in Mvar (underexcited/capacitive).
        q_max : float
            Maximum reactive power in Mvar (overexcited/inductive).
        """
        if DER_Q_CAPABILITY_OVERRIDE_PU is not None:
            q = float(DER_Q_CAPABILITY_OVERRIDE_PU) * s_rated_mva
            return -q, q

        if op_diagram == 'STATCOM':
            # Full circle diagram: Q = ±sqrt(S_n² - P²)
            # p_ratio = |P| / S_n, so P_pu² = p_ratio²
            p_pu_sq = min(p_ratio ** 2, 1.0)
            q_pu = math.sqrt(max(1.0 - p_pu_sq, 0.0))
            q_max = q_pu * s_rated_mva
            q_min = -q_pu * s_rated_mva
            return q_min, q_max

        # VDE-AR-N 4120 Variant 2 breakpoints (p.u. of S_rated)
        #   P:     [0.0,  0.1,  0.2,  1.0]
        #   Q_min: [0.0, -0.10, -0.33, -0.33]
        #   Q_max: [0.0, +0.10, +0.41, +0.41]

        if p_ratio < 0.1:
            # Dead zone: no Q capability below 10 % active power
            q_min = 0.0
            q_max = 0.0
        elif p_ratio < 0.2:
            # Transition region: linear ramp from (0.1 → 0.2)
            t = (p_ratio - 0.1) / 0.1  # 0 at P=0.1, 1 at P=0.2
            q_min = (-0.10 + t * (-0.33 - (-0.10))) * s_rated_mva
            q_max = ( 0.10 + t * ( 0.41 -   0.10))  * s_rated_mva
        else:
            # Full capability: constant above P = 0.2
            q_min = -0.33 * s_rated_mva
            q_max =  0.41 * s_rated_mva

        return q_min, q_max
    
    def get_oltc_tap_bounds(
        self,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Get the fixed tap position bounds for all OLTCs.
        
        Returns
        -------
        tap_min : NDArray[np.int64]
            Minimum tap position for each OLTC.
        tap_max : NDArray[np.int64]
            Maximum tap position for each OLTC.
        """
        return self.oltc_tap_min.copy(), self.oltc_tap_max.copy()
    
    def get_shunt_state_bounds(
        self,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Get the fixed state bounds for all shunts.
        
        Shunts have three possible states:
            -1: Capacitor (reactive power production)
             0: Off
            +1: Reactor (reactive power consumption)
        
        Returns
        -------
        state_min : NDArray[np.int64]
            Minimum state for each shunt (always -1).
        state_max : NDArray[np.int64]
            Maximum state for each shunt (always +1).
        """
        n_shunts = len(self.shunt_indices)
        state_min = np.full(n_shunts, -1, dtype=np.int64)
        state_max = np.full(n_shunts, +1, dtype=np.int64)
        return state_min, state_max
    
    def compute_gen_q_bounds(
        self,
        gen_p_mw: NDArray[np.float64],
        gen_v_pu: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute synchronous generator Q bounds from the detailed
        capability curve (Milano §12.2.1).

        The bounds depend on the current active power dispatch *and*
        the terminal voltage, accounting for stator current, rotor
        current, and under-excitation thermal limits.

        Parameters
        ----------
        gen_p_mw : NDArray[np.float64]
            Current active power output of each generator [MW].
        gen_v_pu : NDArray[np.float64]
            Terminal voltage magnitude of each generator [p.u.].

        Returns
        -------
        q_min_mvar : NDArray[np.float64]
            Minimum reactive power for each generator [Mvar].
        q_max_mvar : NDArray[np.float64]
            Maximum reactive power for each generator [Mvar].

        Raises
        ------
        RuntimeError
            If ``gen_params`` was not provided at construction.
        """
        if self.gen_params is None:
            raise RuntimeError(
                "Generator parameters not configured. "
                "Pass gen_params to ActuatorBounds constructor."
            )
        n_gen = len(self.gen_params)
        q_min = np.zeros(n_gen, dtype=np.float64)
        q_max = np.zeros(n_gen, dtype=np.float64)
        for i, gp in enumerate(self.gen_params):
            q_min[i], q_max[i] = compute_generator_q_limits(
                gp, p_mw=gen_p_mw[i], v_pu=gen_v_pu[i],
            )
        return q_min, q_max

    @property
    def n_ders(self) -> int:
        """Return the number of controllable DERs."""
        return len(self.der_indices)

    @property
    def n_gens(self) -> int:
        """Return the number of synchronous generators."""
        return len(self.gen_params) if self.gen_params is not None else 0

    @property
    def n_oltcs(self) -> int:
        """Return the number of controllable OLTCs."""
        return len(self.oltc_indices)

    @property
    def n_shunts(self) -> int:
        """Return the number of controllable shunts."""
        return len(self.shunt_indices)
