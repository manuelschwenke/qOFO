"""
Base Controller Module
======================

This module defines the abstract base class for OFO controllers.

The base controller implements the common OFO iteration logic:
    u^{k+1} = u^k + σ(u^k, d^k, y^k)

where σ is the solution to the MIQP optimisation problem.

The controller maintains:
- A cached NetworkState for Jacobian-based sensitivity computation
- An ActuatorBounds instance for operating-point-dependent limits
- An MIQPSolver instance for solving the optimisation problem

References
----------
[1] Schwenke et al., PSCC 2026, Eq. (22) - OFO iteration update (alpha=1)
[2] Schwenke et al., PSCC 2026, Eq. (23)-(27) - MIQP formulation

Author: Manuel Schwenke
Date: 2025-02-06
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Union
import numpy as np
from numpy.typing import NDArray

from core.network_state import NetworkState
from core.measurement import Measurement
from core.actuator_bounds import ActuatorBounds
from core.der_mapping import DERMapping
from sensitivity.jacobian import JacobianSensitivities
from optimisation.miqp_solver import (
    MIQPSolver,
    MIQPProblem,
    MIQPResult,
    build_miqp_problem,
)


@dataclass(frozen=True)
class OFOParameters:
    """
    Tuning parameters for the OFO controller.
    
    These parameters control the behaviour of the OFO iteration and the
    MIQP objective function weighting.
    
    Attributes
    ----------
    g_w : float or NDArray[np.float64]
        Weight for control variable changes (w^T G_w w term).
        Penalises large changes in setpoints per iteration.
        Either a scalar (uniform for all variables), a 1-D array of
        length n_total with per-variable diagonal weights, or a 2-D
        symmetric (n_total x n_total) matrix for coupled weights
        (e.g. OLTC cross-penalties).
    g_u : float or NDArray[np.float64]
        Weight for control variable usage (regularisation).
        Penalises deviation from zero/neutral setpoints.
        Either a scalar (uniform for all variables) or an array of
        length n_total with per-variable weights.  Set to 0 for
        actuators that should not be regularised (e.g. OLTC, shunt).
    g_z : float or NDArray[np.float64]
        Weight for slack variables (soft constraint violations).
        Higher values enforce output constraints more strictly.
        Either a scalar (uniform for all outputs) or a 1-D array
        of length n_outputs with per-output weights.  Use lower
        values for outputs that cannot be tightly controlled
        (e.g. branch currents in a reactive-power controller).
    max_iter_per_step : int
        Maximum MIQP solver iterations per OFO step.
    solver_verbose : bool
        Whether to print solver output for debugging.
    """
    g_w: Union[float, NDArray[np.float64]]
    g_z: Union[float, NDArray[np.float64]]
    g_u: Union[float, NDArray[np.float64]] = 0.0
    alpha: float = 1.0
    """OFO step-size for continuous actuators.  Decoupled from g_w so that
    stability (controlled by alpha) and MIQP action amplitude (controlled
    by g_w) are independent.  Discrete actuators always use alpha=1.
    Set to 2/lambda_max(M) × safety_factor after stability analysis."""
    max_iter_per_step: int = 100
    solver_verbose: bool = False
    int_max_step: int = 1
    int_cooldown: int = 6

    def __post_init__(self) -> None:
        """Validate parameters after initialisation."""
        g_w_arr = np.asarray(self.g_w)
        if g_w_arr.ndim <= 1 and np.any(g_w_arr < 0):
            raise ValueError(f"g_w diagonal must be non-negative, got {self.g_w}")
        if g_w_arr.ndim == 2 and g_w_arr.shape[0] != g_w_arr.shape[1]:
            raise ValueError(
                f"g_w matrix must be square, got shape {g_w_arr.shape}"
            )
        g_u_arr = np.asarray(self.g_u)
        if np.any(g_u_arr < 0):
            raise ValueError(f"g_u must be non-negative, got {self.g_u}")
        g_z_arr = np.asarray(self.g_z)
        if np.any(g_z_arr < 0):
            raise ValueError(f"g_z must be non-negative, got {self.g_z}")


@dataclass
class ControllerOutput:
    """
    Output of a single OFO controller iteration.
    
    This class contains the results of one call to `controller.step()`,
    including the new setpoints and diagnostic information.
    
    Attributes
    ----------
    iteration : int
        The OFO iteration index after this step.
    u_new : NDArray[np.float64]
        New control variable values after the update.
        Order: [continuous variables, integer variables].
    u_continuous : NDArray[np.float64]
        New continuous control variable values (e.g., DER Q setpoints).
    u_integer : NDArray[np.int64]
        New integer control variable values (e.g., OLTC tap positions).
    y_predicted : NDArray[np.float64]
        Predicted output values after applying new setpoints.
    sigma : NDArray[np.float64]
        The optimisation result σ^k (change direction).
    z_slack : NDArray[np.float64]
        Slack variable values (constraint violations).
    objective_value : float
        MIQP objective function value at the solution.
    solver_status : str
        Status string from the MIQP solver.
    solve_time_s : float
        Solver computation time in seconds.
    """
    iteration: int
    u_new: NDArray[np.float64]
    u_continuous: NDArray[np.float64]
    u_integer: NDArray[np.int64]
    y_predicted: NDArray[np.float64]
    sigma: NDArray[np.float64]
    z_slack: NDArray[np.float64]
    objective_value: float
    solver_status: str
    solve_time_s: float
    
    @property
    def is_optimal(self) -> bool:
        """Check if the solver found an optimal solution."""
        return self.solver_status == 'optimal'
    
    @property
    def is_feasible(self) -> bool:
        """Check if the solver found a feasible solution."""
        return self.solver_status in ('optimal', 'optimal_inaccurate')


class BaseOFOController(ABC):
    """
    Abstract base class for Online Feedback Optimisation controllers.
    
    This class implements the common OFO iteration logic and provides
    abstract methods that must be implemented by concrete subclasses
    (TSOController, DSOController) for problem-specific behaviour.
    
    The OFO update rule is:
        u^{k+1} = u^k + σ^k
    
    where σ^k is the solution to the MIQP problem:
        σ^k = argmin g(w, z, Δs)
        s.t.  input and output constraints
    
    Attributes
    ----------
    controller_id : str
        Unique identifier for this controller instance.
    params : OFOParameters
        Tuning parameters for the OFO algorithm.
    network_state : NetworkState
        Cached network state for sensitivity computation.
    actuator_bounds : ActuatorBounds
        Operating-point-dependent actuator limits.
    sensitivities : JacobianSensitivities
        Jacobian-based sensitivity calculator.
    solver : MIQPSolver
        MIQP solver instance.
    iteration : int
        Current OFO iteration index.
    u_current : NDArray[np.float64]
        Current control variable values.
    """
    
    def __init__(
        self,
        controller_id: str,
        params: OFOParameters,
        network_state: NetworkState,
        actuator_bounds: ActuatorBounds,
        sensitivities: JacobianSensitivities,
    ) -> None:
        """
        Initialise the base OFO controller.
        
        Parameters
        ----------
        controller_id : str
            Unique identifier for this controller.
        params : OFOParameters
            Tuning parameters for the OFO algorithm.
        network_state : NetworkState
            Cached network state from a converged power flow.
        actuator_bounds : ActuatorBounds
            Actuator limits calculator.
        sensitivities : JacobianSensitivities
            Jacobian-based sensitivity calculator.
        
        Raises
        ------
        ValueError
            If any required parameter is None or invalid.
        """
        if controller_id is None or controller_id == "":
            raise ValueError("controller_id must be a non-empty string")
        if params is None:
            raise ValueError("params must not be None")
        if network_state is None:
            raise ValueError("network_state must not be None")
        if actuator_bounds is None:
            raise ValueError("actuator_bounds must not be None")
        if sensitivities is None:
            raise ValueError("sensitivities must not be None")
        
        self.controller_id = controller_id
        self.params = params
        self.network_state = network_state
        self.actuator_bounds = actuator_bounds
        self.sensitivities = sensitivities
        
        # Initialise MIQP solver
        self.solver = MIQPSolver(
            verbose=params.solver_verbose,
            time_limit_s=60.0,
            mip_gap=1e-6,
        )
        
        # Initialise iteration counter
        self.iteration = 0

        # Initialise control variables (to be set by subclass)
        self._u_current: Optional[NDArray[np.float64]] = None
        self._n_continuous: int = 0
        self._n_integer: int = 0
        self._integer_indices: List[int] = []

        # ---- Caches populated lazily in step() / initialise() ----
        # Expanded per-DER H (shape n_out × n_controls).  Tracked by the
        # identity of the bus-level H it was built from so we can reuse it
        # as long as SensitivityUpdater mutates the base matrix in place.
        self._H_der_cache: Optional[NDArray[np.float64]] = None
        self._H_der_cache_base_id: Optional[int] = None
        # Pre-computed per-variable weight vectors (None when no DER mapping).
        self._g_w_vector_cache: Optional[NDArray[np.float64]] = None
        self._g_u_vector_cache: Optional[NDArray[np.float64]] = None
        # Vectorised continuous/integer index arrays for the step loop.
        self._cont_idx_arr: Optional[NDArray[np.int64]] = None
        self._int_idx_arr: Optional[NDArray[np.int64]] = None
        # (alpha_vec removed: step-size is absorbed into per-actuator g_w)
    
    @property
    def u_current(self) -> NDArray[np.float64]:
        """Get current control variable values."""
        if self._u_current is None:
            raise RuntimeError(
                "Controller not initialised. Call initialise() first."
            )
        return self._u_current.copy()
    
    @property
    def n_continuous(self) -> int:
        """Number of continuous control variables."""
        return self._n_continuous
    
    @property
    def n_integer(self) -> int:
        """Number of integer control variables."""
        return self._n_integer
    
    @property
    def n_controls(self) -> int:
        """Total number of control variables."""
        return self._n_continuous + self._n_integer
    
    def initialise(self, measurement: Measurement) -> None:
        """
        Initialise the controller from the current system state.

        This method must be called before the first call to step().
        It extracts initial control variable values from measurements
        and sets up the internal state.

        Parameters
        ----------
        measurement : Measurement
            Current system measurements.

        Raises
        ------
        ValueError
            If the measurement is incompatible with the controller.
        """
        # Extract initial control values from measurement
        u_init = self._extract_control_values(measurement)

        # Set internal state
        self._u_current = u_init.copy()
        self.iteration = measurement.iteration

        # Set dimensions (to be defined by subclass in _get_control_structure)
        self._n_continuous, self._n_integer, self._integer_indices = \
            self._get_control_structure()

        # Integer switching logic
        self._int_cooldown = self.params.int_cooldown
        self._int_max_step = self.params.int_max_step
        self._int_lock_until: dict[int, int] = {}   # idx -> iteration when lock expires

        # ---- Vectorised continuous/integer index arrays for step() ----
        # Used to replace `for i in range(n_controls): if i in integer_indices`
        # Python loops with pure numpy indexing.
        n_total = self._n_continuous + self._n_integer
        mask = np.ones(n_total, dtype=bool)
        if self._integer_indices:
            mask[list(self._integer_indices)] = False
        self._cont_idx_arr = np.flatnonzero(mask).astype(np.int64)
        self._int_idx_arr = np.asarray(
            self._integer_indices, dtype=np.int64,
        )
        # (alpha_vec removed: step-size is now absorbed into per-actuator g_w)

        # ---- Precompute per-variable MIQP weight vectors ----
        # When a DER mapping with per-DER weights is active, both g_w and
        # g_u get scaled per-DER.  All inputs are static for the life of
        # the controller, so we compute once here instead of rebuilding
        # the arrays on every step.
        self._g_w_vector_cache, self._g_u_vector_cache = \
            self._compute_per_variable_weights()

        # Expanded H cache is invalidated on re-init (new operating point).
        self._H_der_cache = None
        self._H_der_cache_base_id = None

        # g_z semantics (enforced in the MIQP solver):
        #   g_z = 0  → hard output constraints (no slack variable z)
        #   g_z > 0  → soft output constraints (z penalised by z^T G_z z)
        # Use g_z > 0 for transient robustness when the problem may be
        # temporarily infeasible (e.g., after large disturbances).

    def step(self, measurement: Measurement) -> ControllerOutput:
        """
        Execute one OFO iteration.

        This method performs the core OFO update:
            1. Extract current outputs y^k from measurements
            2. Compute objective gradient ∇f
            3. Compute sensitivity matrix H
            4. Build and solve MIQP problem
            5. Apply update: u^{k+1} = u^k + σ^k
        
        Parameters
        ----------
        measurement : Measurement
            Current system measurements at iteration k.
        
        Returns
        -------
        ControllerOutput
            Results of this iteration including new setpoints.
        
        Raises
        ------
        RuntimeError
            If the controller is not initialised or the solver fails.
        ValueError
            If the measurement is incompatible.
        """
        if self._u_current is None:
            raise RuntimeError(
                "Controller not initialised. Call initialise() first."
            )
        
        # Step 1: Extract current outputs from measurements
        y_current = self._extract_outputs(measurement)
        
        # Step 2: Get current actuator P values for capability calculation and current Q_TSO_DSO values
        der_p_current = self._extract_der_active_power(measurement)
        tso_dso_interface_q_current = self._extract_trafo_reactive_power(measurement)
        
        # Step 3: Compute input bounds (operating-point-dependent)
        u_lower, u_upper = self._compute_input_bounds(tso_dso_interface_q_current, der_p_current)

        # Step 3b: Integer variables may change by at most ±N per iteration.
        # This prevents multi-step jumps (e.g. OLTC jumping 5 taps at once).
        # Standard value is 1, but can be increased via params.int_max_step.
        for idx in self._integer_indices:
            u_lower[idx] = max(u_lower[idx], self._u_current[idx] - self._int_max_step)
            u_upper[idx] = min(u_upper[idx], self._u_current[idx] + self._int_max_step)

        # Step 3c: Enforce integer cooldown — lock recently-switched integers
        for idx in self._integer_indices:
            if idx in self._int_lock_until and self.iteration < self._int_lock_until[idx]:
                u_lower[idx] = self._u_current[idx]
                u_upper[idx] = self._u_current[idx]

        # Step 4: Get output bounds
        y_lower, y_upper = self._get_output_limits()
        
        # Step 5: Compute objective gradient
        grad_f = self._compute_objective_gradient(measurement)
        
        # Step 6: Build sensitivity matrix H
        #   _build_sensitivity_matrix returns the bus-level H (full rank).
        #   If a DER mapping is active, expand to per-DER via H_bus @ E.
        #   The expansion result is cached across steps as long as the
        #   bus-level matrix identity does not change (see
        #   _expand_H_to_der_level).
        H_bus = self._build_sensitivity_matrix()
        H = self._expand_H_to_der_level(H_bus)

        # Step 6b: Per-variable weight vectors are computed once in
        # initialise() and stored on the instance — just fetch them.
        g_w_vector, g_u_vector = self._get_per_variable_weights()

        # Step 7: Build and solve MIQP problem
        #   alpha scales continuous variables only; discrete actuators
        #   use alpha=1 (they cannot move "a fraction of a step").
        alpha = self.params.alpha
        problem = build_miqp_problem(
            alpha=alpha,
            u_current=self._u_current,
            y_current=y_current,
            H=H,
            grad_f=grad_f,
            u_lower=u_lower,
            u_upper=u_upper,
            y_lower=y_lower,
            y_upper=y_upper,
            g_w=self.params.g_w,
            g_u=self.params.g_u,
            g_z=self.params.g_z,
            integer_indices=self._integer_indices,
            g_w_vector=g_w_vector,
            g_u_vector=g_u_vector,
        )

        result = self.solver.solve(problem)

        if not result.is_feasible:
            raise RuntimeError(
                f"MIQP solver failed at iteration {self.iteration}: "
                f"{result.status}"
            )

        # Step 8: Reassemble sigma in original (unreordered) variable
        # order using the precomputed continuous/integer index arrays.
        # Replaces three Python `for i in range(n_controls): if i in
        # self._integer_indices` loops, which were O(n_controls × n_int)
        # and dominated the per-step cost after the per-DER refactor.
        cont_idx = self._cont_idx_arr
        int_idx = self._int_idx_arr

        sigma = np.zeros(self.n_controls, dtype=np.float64)
        sigma[cont_idx] = alpha * result.w_continuous
        if int_idx.size > 0:
            sigma[int_idx] = result.w_integer.astype(np.float64)

        # Step 9: OFO update.
        #   Continuous: u_new = u + α·w   (alpha controls step size)
        #   Discrete:   u_new = u + w     (full step, cannot move fractionally)
        u_new = self._u_current + sigma
        if int_idx.size > 0:
            # Note: u_new[int_idx] = np.round(u_new[int_idx]) is the
            # correct in-place write — fancy-indexed assignment writes
            # back, while np.round(..., out=u_new[int_idx]) would write
            # into a temporary copy.
            u_new[int_idx] = np.round(u_new[int_idx])

        # Step 10: Predict new outputs.
        #   Δy ≈ H · σ
        y_predicted = y_current + H @ sigma

        # Step 11: Cooldown bookkeeping for integer switches.
        if int_idx.size > 0:
            switched = u_new[int_idx] != self._u_current[int_idx]
            for j in np.flatnonzero(switched):
                idx = int(int_idx[j])
                self._int_lock_until[idx] = (
                    self.iteration + 1 + self._int_cooldown
                )

        self._u_current = u_new.copy()
        self.iteration += 1

        # Step 12: Continuous and integer slices in original order.
        u_continuous = u_new[cont_idx].copy()
        if int_idx.size > 0:
            u_integer = u_new[int_idx].astype(np.int64)
        else:
            u_integer = np.array([], dtype=np.int64)

        return ControllerOutput(
            iteration=self.iteration,
            u_new=u_new,
            u_continuous=u_continuous,
            u_integer=u_integer,
            y_predicted=y_predicted,
            sigma=sigma,
            z_slack=result.z,
            objective_value=result.objective_value,
            solver_status=result.status,
            solve_time_s=result.solve_time_s,
        )
    
    def reset(self) -> None:
        """
        Reset the controller to uninitialised state.
        
        After calling reset(), initialise() must be called before step().
        """
        self._u_current = None
        self.iteration = 0
    
    # =========================================================================
    # Abstract methods to be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def _get_control_structure(self) -> Tuple[int, int, List[int]]:
        """
        Define the structure of control variables.
        
        Returns
        -------
        n_continuous : int
            Number of continuous control variables.
        n_integer : int
            Number of integer control variables.
        integer_indices : List[int]
            Indices of integer variables in the combined control vector.
        """
        pass
    
    @abstractmethod
    def _extract_control_values(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """
        Extract current control variable values from measurements.
        
        Parameters
        ----------
        measurement : Measurement
            Current system measurements.
        
        Returns
        -------
        u : NDArray[np.float64]
            Current control variable values.
        """
        pass
    
    @abstractmethod
    def _extract_outputs(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """
        Extract current output values from measurements.
        
        Parameters
        ----------
        measurement : Measurement
            Current system measurements.
        
        Returns
        -------
        y : NDArray[np.float64]
            Current output values (voltages, currents, Q flows).
        """
        pass
    
    @abstractmethod
    def _extract_der_active_power(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """
        Extract current DER active power outputs for capability calculation.
        
        Parameters
        ----------
        measurement : Measurement
            Current system measurements.
        
        Returns
        -------
        p_der : NDArray[np.float64]
            DER active power outputs in MW.
        """
        pass

    @abstractmethod
    def _extract_trafo_reactive_power(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """
        Extract current trafo ractive power flow for capability calculation.

        Parameters
        ----------
        measurement : Measurement
            Current system measurements.

        Returns
        -------
        q_tso_dso : NDArray[np.float64]
            Trafo reactive power flows in Mvar.
        """
        pass

    
    @abstractmethod
    def _compute_input_bounds(
        self,
        tso_dso_interface_q_current: NDArray[np.float64],
        der_p_current: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute operating-point-dependent input bounds.
        
        Parameters
        ----------
        der_p_current : NDArray[np.float64]
            Current DER active power outputs in MW.
        
        Returns
        -------
        u_lower : NDArray[np.float64]
            Lower bounds on control variables.
        u_upper : NDArray[np.float64]
            Upper bounds on control variables.
        """
        pass
    
    @abstractmethod
    def _get_output_limits(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get output constraint limits (voltage bands, current limits).
        
        Returns
        -------
        y_lower : NDArray[np.float64]
            Lower bounds on outputs.
        y_upper : NDArray[np.float64]
            Upper bounds on outputs.
        """
        pass
    
    @abstractmethod
    def _compute_objective_gradient(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """
        Compute the objective function gradient ∇f.
        
        This is problem-specific and differs between TSO and DSO controllers.
        
        Parameters
        ----------
        measurement : Measurement
            Current system measurements.
        
        Returns
        -------
        grad_f : NDArray[np.float64]
            Objective gradient vector.
        """
        pass
    
    @abstractmethod
    def _build_sensitivity_matrix(self) -> NDArray[np.float64]:
        """
        Build the bus-level input-output sensitivity matrix H_bus.

        Uses the JacobianSensitivities instance to compute dy/du.
        DER columns correspond to **unique buses** (one column per bus),
        ensuring full column rank.

        Returns
        -------
        H_bus : NDArray[np.float64]
            Bus-level sensitivity matrix of shape
            ``(n_outputs, n_controls_bus)``.
        """
        pass

    # ------------------------------------------------------------------
    #  Per-DER expansion and weight construction
    # ------------------------------------------------------------------

    def _get_der_mapping(self) -> Optional[DERMapping]:
        """
        Return the DER mapping from the subclass config, if any.

        Subclasses should override this if they support per-DER
        modelling.  The default returns ``None`` (bus-level).
        """
        return None

    def _get_n_der_bus(self) -> int:
        """
        Return the number of DER-related columns in H_bus.

        Subclasses must override this to return the count of unique
        DER bus columns that precede other actuator columns in H.
        """
        raise NotImplementedError

    def _expand_H_to_der_level(
        self, H_bus: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Expand H_bus to per-DER columns via the incidence matrix E.

        If no DER mapping is active, returns H_bus unchanged.

        The factorisation is:
            H_der[:, :n_der] = H_bus[:, :n_bus_der] @ E
            H_der[:, n_der:] = H_bus[:, n_bus_der:]   (non-DER cols)

        A cache of the expanded matrix is kept and reused across steps
        as long as the **identity** of the bus-level matrix does not
        change.  The ``SensitivityUpdater`` mutates ``_H_current`` in
        place on every step (only a scalar column rescaling for shunts)
        and returns the same ndarray, so identity-based invalidation is
        sufficient: the cached per-DER expansion is automatically reused
        until ``invalidate_sensitivity_cache()`` clears it.

        Parameters
        ----------
        H_bus : NDArray[np.float64]
            Bus-level sensitivity matrix with shape
            ``(n_outputs, n_controls_bus)``.

        Returns
        -------
        H : NDArray[np.float64]
            Sensitivity matrix with shape ``(n_outputs, n_controls)``.
            If a DER mapping is active, ``n_controls > n_controls_bus``
            because each bus with multiple DERs gets expanded.
        """
        mapping = self._get_der_mapping()
        if mapping is None:
            return H_bus

        base_id = id(H_bus)
        if (
            self._H_der_cache is not None
            and self._H_der_cache_base_id == base_id
        ):
            # Overwrite the DER block from the (possibly rescaled)
            # H_bus[:, :n_bus_der] @ E and the rest block from
            # H_bus[:, n_bus_der:], without allocating a new matrix.
            n_bus_der = mapping.n_unique_bus
            n_der = mapping.n_der
            E = mapping.E
            np.matmul(
                H_bus[:, :n_bus_der], E,
                out=self._H_der_cache[:, :n_der],
            )
            self._H_der_cache[:, n_der:] = H_bus[:, n_bus_der:]
            return self._H_der_cache

        # First-time (or post-invalidation) build.
        n_bus_der = mapping.n_unique_bus
        E = mapping.E
        H_bus_der = H_bus[:, :n_bus_der]
        H_bus_rest = H_bus[:, n_bus_der:]
        H_der_part = H_bus_der @ E
        H = np.hstack([H_der_part, H_bus_rest])
        self._H_der_cache = H
        self._H_der_cache_base_id = base_id
        return H

    def _compute_per_variable_weights(
        self,
    ) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
        """
        Compute per-variable MIQP weight vectors **once** at init time.

        When a DER mapping with non-uniform weights is active, this
        constructs ``g_w_vector`` and ``g_u_vector`` arrays where the
        first ``n_der`` entries are scaled by the per-DER weights and
        the remaining entries use the scalar defaults.

        All inputs (``self.params.g_w``, ``self.params.g_u``,
        ``mapping.weights``) are static for the life of the controller,
        so this function is called from ``initialise()`` and the result
        is cached on ``self._g_w_vector_cache`` /
        ``self._g_u_vector_cache``.  ``step()`` then reads the cached
        refs via ``_get_per_variable_weights()`` instead of rebuilding
        the arrays on every iteration.

        Returns
        -------
        g_w_vector : NDArray or None
            Per-variable change weights, or None for uniform.
        g_u_vector : NDArray or None
            Per-variable usage weights, or None for uniform.
        """
        mapping = self._get_der_mapping()
        if mapping is None:
            return None, None

        n_total = self.n_controls
        n_der = mapping.n_der

        # Extract base scalar values from params (handle Union types)
        g_w_scalar = np.asarray(self.params.g_w)
        if g_w_scalar.ndim == 0:
            g_w_base = float(g_w_scalar)
        else:
            # g_w is already an array or matrix; use mean of diagonal
            # as base for DER weight scaling
            if g_w_scalar.ndim == 2:
                g_w_base = float(np.mean(np.diag(g_w_scalar)[:n_der]))
            else:
                g_w_base = float(np.mean(g_w_scalar[:n_der]))

        g_u_scalar = np.asarray(self.params.g_u)
        if g_u_scalar.ndim == 0:
            g_u_base = float(g_u_scalar)
        else:
            g_u_base = float(np.mean(g_u_scalar[:n_der]))

        # Build g_w per variable: DER entries scaled by per-DER weights
        g_w_vec = np.broadcast_to(
            np.asarray(self.params.g_w, dtype=np.float64),
            (n_total,),
        ).copy()
        g_w_vec[:n_der] = g_w_base * mapping.weights

        # Build g_u per variable: same treatment
        g_u_vec = np.broadcast_to(
            np.asarray(self.params.g_u, dtype=np.float64),
            (n_total,),
        ).copy()
        g_u_vec[:n_der] = g_u_base * mapping.weights

        return g_w_vec, g_u_vec

    def _get_per_variable_weights(
        self,
    ) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
        """Return the cached per-variable weight vectors (or None/None)."""
        return self._g_w_vector_cache, self._g_u_vector_cache

    def get_bus_level_sensitivity(
        self,
    ) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]:
        """
        Return the bus-level sensitivity matrix and incidence matrix.

        This is intended for closed-loop stability / eigenvalue
        analysis, where full column rank of H is required.

        Returns
        -------
        H_bus : NDArray[np.float64]
            Bus-level sensitivity matrix (full column rank in the
            DER columns).
        E : NDArray[np.float64] or None
            DER-to-bus incidence matrix of shape
            ``(n_unique_bus, n_der)``, or ``None`` if no DER mapping
            is active.
        """
        H_bus = self._build_sensitivity_matrix()
        mapping = self._get_der_mapping()
        E = mapping.E if mapping is not None else None
        return H_bus, E
