"""
Base Controller Module
======================

This module defines the abstract base class for OFO controllers.

The base controller implements the common OFO iteration logic:
    u^{k+1} = u^k + α · σ(u^k, d^k, y^k)

where σ is the solution to the MIQP optimisation problem.

The controller maintains:
- A cached NetworkState for Jacobian-based sensitivity computation
- An ActuatorBounds instance for operating-point-dependent limits
- An MIQPSolver instance for solving the optimisation problem

References
----------
[1] Schwenke et al., PSCC 2026, Eq. (22) - OFO iteration update
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
    alpha : float
        Step size (gain) for the OFO update. Must be positive.
        Larger values lead to faster but potentially less stable convergence.
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
    alpha: float
    g_w: Union[float, NDArray[np.float64]]
    g_z: Union[float, NDArray[np.float64]]
    g_u: Union[float, NDArray[np.float64]] = 0.0
    max_iter_per_step: int = 100
    solver_verbose: bool = False
    int_max_step: int = 1
    int_cooldown: int = 6

    def __post_init__(self) -> None:
        """Validate parameters after initialisation."""
        if np.any(self.alpha) <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
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
        u^{k+1} = u^k + α · σ^k
    
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
    
    def step(self, measurement: Measurement) -> ControllerOutput:
        """
        Execute one OFO iteration.
        
        This method performs the core OFO update:
            1. Extract current outputs y^k from measurements
            2. Compute objective gradient ∇f
            3. Compute sensitivity matrix H
            4. Build and solve MIQP problem
            5. Apply update: u^{k+1} = u^k + α · σ^k
        
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
        H_bus = self._build_sensitivity_matrix()
        H = self._expand_H_to_der_level(H_bus)

        # Step 6b: Build per-variable weight vectors if DER mapping is active
        g_w_vector, g_u_vector = self._build_per_variable_weights()

        # Step 7: Build and solve MIQP problem
        problem = build_miqp_problem(
            alpha=self.params.alpha,
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


        # Step 8: Compute full sigma (combining continuous and integer)
        sigma = np.zeros(self.n_controls)
        
        # Reorder back from [continuous, integer] to original order
        continuous_indices = [
            i for i in range(self.n_controls) 
            if i not in self._integer_indices
        ]
        
        for i, ci in enumerate(continuous_indices):
            sigma[ci] = result.w_continuous[i]
        for i, ii in enumerate(self._integer_indices):
            sigma[ii] = float(result.w_integer[i])

        # Step 9: Apply OFO update
        #   Continuous: u^{k+1} = u^k + α · σ^k   (gradient micro-step)
        #   Integer:    u^{k+1} = u^k + σ^k        (direct state change)
        u_new = self._u_current.copy()
        for i in range(self.n_controls):
            if i in self._integer_indices:
                u_new[i] += sigma[i]                     # direct
            else:
                u_new[i] += self.params.alpha * sigma[i]  # scaled

        # Round integer variables
        for idx in self._integer_indices:
            u_new[idx] = np.round(u_new[idx])

        # Step 10: Predict new outputs
        #   Δy ≈ α · H_c · σ_c  +  H_i · σ_i
        y_predicted = y_current.copy()
        for i in range(self.n_controls):
            if i in self._integer_indices:
                y_predicted += H[:, i] * sigma[i]
            else:
                y_predicted += self.params.alpha * H[:, i] * sigma[i]
        
        # Step 11: Update internal state and set cooldown for switched integers
        for idx in self._integer_indices:
            if u_new[idx] != self._u_current[idx]:
                self._int_lock_until[idx] = self.iteration + 1 + self._int_cooldown

        self._u_current = u_new.copy()
        self.iteration += 1

        # Step 12: Extract continuous and integer parts
        u_continuous = np.array([
            u_new[i] for i in range(self.n_controls) 
            if i not in self._integer_indices
        ])
        u_integer = np.array([
            int(np.round(u_new[i])) for i in self._integer_indices
        ], dtype=np.int64)
        
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

        n_bus_der = mapping.n_unique_bus
        E = mapping.E  # (n_unique_bus, n_der)

        # Split H_bus into DER-bus columns and the rest
        H_bus_der = H_bus[:, :n_bus_der]           # (n_out, n_bus_der)
        H_bus_rest = H_bus[:, n_bus_der:]           # (n_out, n_other)

        # Expand DER columns: H_der_part = H_bus_der @ E
        H_der_part = H_bus_der @ E                  # (n_out, n_der)

        # Reassemble
        return np.hstack([H_der_part, H_bus_rest])

    def _build_per_variable_weights(
        self,
    ) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
        """
        Build per-variable weight vectors for the MIQP objective.

        When a DER mapping with non-uniform weights is active, this
        constructs ``g_w_vector`` and ``g_u_vector`` arrays where the
        first ``n_der`` entries are scaled by the per-DER weights and
        the remaining entries use the scalar defaults.

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
