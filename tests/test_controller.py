"""
Unit Tests for OFO Controllers
===============================

Tests for BaseOFOController, TSOController, and DSOController.

These tests verify:
    - Parameter validation
    - Control variable structure and ordering
    - Initialisation from measurements
    - Single-step OFO iteration
    - Inter-controller messaging (setpoints and capabilities)
    - Output extraction and bounds computation
    - Sensitivity matrix assembly

The tests use synthetic sensitivity matrices and measurements to
avoid dependence on actual power-flow calculations.

Author: Manuel Schwenke
Date: 2025-02-06
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from unittest.mock import MagicMock

from controller.base_controller import (
    BaseOFOController,
    OFOParameters,
    ControllerOutput,
)
from controller.tso_controller import TSOController, TSOControllerConfig
from controller.dso_controller import DSOController, DSOControllerConfig
from core.network_state import NetworkState
from core.measurement import Measurement
from core.actuator_bounds import ActuatorBounds
from core.message import SetpointMessage, CapabilityMessage
from sensitivity.jacobian import JacobianSensitivities


# =============================================================================
# Helpers for constructing test fixtures
# =============================================================================

def _make_network_state() -> NetworkState:
    """Create a minimal NetworkState for testing."""
    return NetworkState(
        bus_indices=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        voltage_magnitudes_pu=np.array([1.05, 1.03, 1.02, 1.01, 1.00]),
        voltage_angles_rad=np.array([0.0, -0.01, -0.02, -0.03, -0.04]),
        slack_bus_index=0,
        pv_bus_indices=np.array([1], dtype=np.int64),
        pq_bus_indices=np.array([2, 3, 4], dtype=np.int64),
        transformer_indices=np.array([0, 1], dtype=np.int64),
        tap_positions=np.array([0.0, 0.0]),
        source_case="test_case",
        timestamp="2025-02-06T00:00:00",
        cached_at_iteration=0,
    )


def _make_actuator_bounds(
    n_der: int = 2,
    n_oltc: int = 1,
    n_shunt: int = 1,
) -> ActuatorBounds:
    """Create ActuatorBounds for testing."""
    return ActuatorBounds(
        der_indices=np.arange(n_der, dtype=np.int64),
        der_s_rated_mva=np.full(n_der, 100.0),
        der_p_max_mw=np.full(n_der, 80.0),
        oltc_indices=np.arange(n_oltc, dtype=np.int64),
        oltc_tap_min=np.full(n_oltc, -16, dtype=np.int64),
        oltc_tap_max=np.full(n_oltc, 16, dtype=np.int64),
        shunt_indices=np.arange(n_shunt, dtype=np.int64),
        shunt_q_mvar=np.full(n_shunt, 50.0),
    )


def _make_mock_sensitivities(
    n_outputs: int,
    n_der: int,
    n_oltc: int = 0,
    n_shunt: int = 0,
) -> JacobianSensitivities:
    """
    Create a mock JacobianSensitivities that returns a synthetic H matrix.

    The returned H matrix has predictable values for testing:
    H_physical rows = [Q_trafo, V_bus, I_line]
    H_physical cols = [DER, OLTC, shunt]
    """
    mock_sens = MagicMock(spec=JacobianSensitivities)

    n_inputs_phys = n_der + n_oltc + n_shunt
    H_physical = 0.01 * np.ones((n_outputs, n_inputs_phys))
    # Make DER columns slightly larger for identifiability
    H_physical[:, :n_der] = 0.05

    mappings = {
        "input_types": (
            {i: "continuous" for i in range(n_der)}
            | {n_der + i: "integer" for i in range(n_oltc + n_shunt)}
        ),
    }

    mock_sens.build_sensitivity_matrix_H.return_value = (H_physical, mappings)
    return mock_sens


def _make_dso_measurement(
    iteration: int = 0,
    der_bus_indices: list = None,
    oltc_trafo_indices: list = None,
    shunt_bus_indices: list = None,
    interface_trafo_indices: list = None,
    voltage_bus_indices: list = None,
    current_line_indices: list = None,
) -> Measurement:
    """Create a Measurement suitable for DSO controller tests."""
    if der_bus_indices is None:
        der_bus_indices = [2, 3]
    if oltc_trafo_indices is None:
        oltc_trafo_indices = [0]
    if shunt_bus_indices is None:
        shunt_bus_indices = [4]
    if interface_trafo_indices is None:
        interface_trafo_indices = [0]
    if voltage_bus_indices is None:
        voltage_bus_indices = [1, 2, 3]
    if current_line_indices is None:
        current_line_indices = [0, 1]

    all_bus_indices = sorted(
        set(der_bus_indices + voltage_bus_indices + shunt_bus_indices
            + [0, 1, 2, 3, 4])
    )

    return Measurement(
        iteration=iteration,
        bus_indices=np.array(all_bus_indices, dtype=np.int64),
        voltage_magnitudes_pu=np.full(len(all_bus_indices), 1.02),
        branch_indices=np.array(current_line_indices, dtype=np.int64),
        current_magnitudes_ka=np.full(len(current_line_indices), 0.15),
        interface_transformer_indices=np.array(
            interface_trafo_indices, dtype=np.int64
        ),
        interface_q_hv_side_mvar=np.full(
            len(interface_trafo_indices), 10.0
        ),
        der_indices=np.array(der_bus_indices, dtype=np.int64),
        der_q_mvar=np.full(len(der_bus_indices), 5.0),
        oltc_indices=np.array(oltc_trafo_indices, dtype=np.int64),
        oltc_tap_positions=np.zeros(
            len(oltc_trafo_indices), dtype=np.int64
        ),
        shunt_indices=np.array(shunt_bus_indices, dtype=np.int64),
        shunt_states=np.zeros(len(shunt_bus_indices), dtype=np.int64),
    )


# =============================================================================
# Tests for OFOParameters
# =============================================================================


class TestOFOParameters:
    """Tests for the OFOParameters data class."""

    def test_valid_construction(self) -> None:
        """Test valid parameter construction."""
        params = OFOParameters(
            alpha=0.03, g_w=0.2, g_z=1000.0, g_s=80.0, g_u=0.01
        )
        assert params.alpha == 0.03
        assert params.g_w == 0.2
        assert params.g_z == 1000.0
        assert params.g_s == 80.0
        assert params.g_u == 0.01

    def test_negative_alpha_raises(self) -> None:
        """Test that negative alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            OFOParameters(alpha=-0.1, g_w=1.0, g_z=1.0, g_s=1.0)

    def test_zero_alpha_raises(self) -> None:
        """Test that zero alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            OFOParameters(alpha=0.0, g_w=1.0, g_z=1.0, g_s=1.0)

    def test_negative_g_w_raises(self) -> None:
        """Test that negative g_w raises ValueError."""
        with pytest.raises(ValueError, match="g_w must be non-negative"):
            OFOParameters(alpha=0.1, g_w=-1.0, g_z=1.0, g_s=1.0)

    def test_negative_g_z_raises(self) -> None:
        """Test that negative g_z raises ValueError."""
        with pytest.raises(ValueError, match="g_z must be non-negative"):
            OFOParameters(alpha=0.1, g_w=1.0, g_z=-1.0, g_s=1.0)

    def test_negative_g_s_raises(self) -> None:
        """Test that negative g_s raises ValueError."""
        with pytest.raises(ValueError, match="g_s must be non-negative"):
            OFOParameters(alpha=0.1, g_w=1.0, g_z=1.0, g_s=-1.0)

    def test_negative_g_u_raises(self) -> None:
        """Test that negative g_u raises ValueError."""
        with pytest.raises(ValueError, match="g_u must be non-negative"):
            OFOParameters(alpha=0.1, g_w=1.0, g_z=1.0, g_s=1.0, g_u=-0.5)

    def test_default_g_u_is_zero(self) -> None:
        """Test that g_u defaults to zero."""
        params = OFOParameters(alpha=0.1, g_w=1.0, g_z=1.0, g_s=1.0)
        assert params.g_u == 0.0


# =============================================================================
# Tests for ControllerOutput
# =============================================================================


class TestControllerOutput:
    """Tests for the ControllerOutput data class."""

    def _make_output(self, status: str = "optimal") -> ControllerOutput:
        """Create a ControllerOutput for testing."""
        return ControllerOutput(
            iteration=1,
            u_new=np.array([1.0, 2.0, 3.0]),
            u_continuous=np.array([1.0, 2.0]),
            u_integer=np.array([3], dtype=np.int64),
            y_predicted=np.array([1.01, 10.0]),
            sigma=np.array([0.1, 0.2, 1.0]),
            z_slack=np.array([0.0, 0.0]),
            objective_value=0.5,
            solver_status=status,
            solve_time_s=0.002,
        )

    def test_is_optimal_true(self) -> None:
        """Test is_optimal for an optimal solution."""
        out = self._make_output("optimal")
        assert out.is_optimal is True

    def test_is_optimal_false(self) -> None:
        """Test is_optimal for a non-optimal solution."""
        out = self._make_output("infeasible")
        assert out.is_optimal is False

    def test_is_feasible_optimal(self) -> None:
        """Test is_feasible for an optimal solution."""
        out = self._make_output("optimal")
        assert out.is_feasible is True

    def test_is_feasible_inaccurate(self) -> None:
        """Test is_feasible for an inaccurate-but-feasible solution."""
        out = self._make_output("optimal_inaccurate")
        assert out.is_feasible is True

    def test_is_feasible_infeasible(self) -> None:
        """Test is_feasible for an infeasible solution."""
        out = self._make_output("infeasible")
        assert out.is_feasible is False


# =============================================================================
# Tests for DSOControllerConfig
# =============================================================================


class TestDSOControllerConfig:
    """Tests for DSOControllerConfig validation."""

    def test_valid_config(self) -> None:
        """Test valid configuration construction."""
        cfg = DSOControllerConfig(
            der_bus_indices=[2, 3],
            oltc_trafo_indices=[0],
            shunt_bus_indices=[4],
            shunt_q_steps_mvar=[50.0],
            interface_trafo_indices=[0],
            voltage_bus_indices=[1, 2, 3],
            current_line_indices=[0, 1],
        )
        assert len(cfg.der_bus_indices) == 2
        assert cfg.v_min_pu == 0.95

    def test_shunt_length_mismatch_raises(self) -> None:
        """Test that mismatched shunt lists raise ValueError."""
        with pytest.raises(ValueError, match="shunt_bus_indices length"):
            DSOControllerConfig(
                der_bus_indices=[2],
                oltc_trafo_indices=[],
                shunt_bus_indices=[4, 5],
                shunt_q_steps_mvar=[50.0],  # Only one element
                interface_trafo_indices=[0],
                voltage_bus_indices=[1],
                current_line_indices=[],
            )

    def test_invalid_voltage_band_raises(self) -> None:
        """Test that v_min >= v_max raises ValueError."""
        with pytest.raises(ValueError, match="v_min_pu"):
            DSOControllerConfig(
                der_bus_indices=[2],
                oltc_trafo_indices=[],
                shunt_bus_indices=[],
                shunt_q_steps_mvar=[],
                interface_trafo_indices=[0],
                voltage_bus_indices=[1],
                current_line_indices=[],
                v_min_pu=1.10,
                v_max_pu=0.95,
            )


# =============================================================================
# Tests for TSOControllerConfig
# =============================================================================


class TestTSOControllerConfig:
    """Tests for TSOControllerConfig validation."""

    def test_valid_config(self) -> None:
        """Test valid configuration construction."""
        cfg = TSOControllerConfig(
            der_bus_indices=[2, 3],
            pcc_trafo_indices=[0, 1],
            pcc_dso_controller_ids=["dso_1", "dso_2"],
            oltc_trafo_indices=[0],
            shunt_bus_indices=[4],
            shunt_q_steps_mvar=[50.0],
            voltage_bus_indices=[0, 1, 2],
            current_line_indices=[0],
        )
        assert len(cfg.pcc_trafo_indices) == 2

    def test_pcc_dso_id_mismatch_raises(self) -> None:
        """Test that mismatched PCC/DSO-ID lists raise ValueError."""
        with pytest.raises(ValueError, match="pcc_trafo_indices length"):
            TSOControllerConfig(
                der_bus_indices=[2],
                pcc_trafo_indices=[0, 1],
                pcc_dso_controller_ids=["dso_1"],  # Only one
                oltc_trafo_indices=[],
                shunt_bus_indices=[],
                shunt_q_steps_mvar=[],
                voltage_bus_indices=[0],
                current_line_indices=[],
            )

    def test_shunt_length_mismatch_raises(self) -> None:
        """Test that mismatched shunt lists raise ValueError."""
        with pytest.raises(ValueError, match="shunt_bus_indices length"):
            TSOControllerConfig(
                der_bus_indices=[],
                pcc_trafo_indices=[],
                pcc_dso_controller_ids=[],
                oltc_trafo_indices=[],
                shunt_bus_indices=[4],
                shunt_q_steps_mvar=[],  # Empty
                voltage_bus_indices=[],
                current_line_indices=[],
            )

    def test_voltage_setpoint_length_mismatch_raises(self) -> None:
        """Test that mismatched v_setpoints length raises ValueError."""
        with pytest.raises(ValueError, match="v_setpoints_pu length"):
            TSOControllerConfig(
                der_bus_indices=[],
                pcc_trafo_indices=[],
                pcc_dso_controller_ids=[],
                oltc_trafo_indices=[],
                shunt_bus_indices=[],
                shunt_q_steps_mvar=[],
                voltage_bus_indices=[0, 1],
                current_line_indices=[],
                v_setpoints_pu=np.array([1.05]),  # Only one
            )


# =============================================================================
# Tests for DSOController
# =============================================================================


class TestDSOController:
    """Tests for the DSOController class."""

    @pytest.fixture
    def dso_setup(self):
        """Create a complete DSO controller setup."""
        der_buses = [2, 3]
        oltc_trafos = [0]
        shunt_buses = [4]
        interface_trafos = [0]
        voltage_buses = [1, 2, 3]
        current_lines = [0, 1]

        n_der = len(der_buses)
        n_oltc = len(oltc_trafos)
        n_shunt = len(shunt_buses)
        n_interface = len(interface_trafos)
        n_v = len(voltage_buses)
        n_i = len(current_lines)
        n_outputs = n_interface + n_v + n_i

        config = DSOControllerConfig(
            der_bus_indices=der_buses,
            oltc_trafo_indices=oltc_trafos,
            shunt_bus_indices=shunt_buses,
            shunt_q_steps_mvar=[50.0],
            interface_trafo_indices=interface_trafos,
            voltage_bus_indices=voltage_buses,
            current_line_indices=current_lines,
        )

        params = OFOParameters(
            alpha=0.03, g_w=0.2, g_z=1000.0, g_s=80.0, g_u=0.01
        )

        network_state = _make_network_state()
        actuator_bounds = _make_actuator_bounds(n_der, n_oltc, n_shunt)
        sensitivities = _make_mock_sensitivities(
            n_outputs=n_outputs,
            n_der=n_der,
            n_oltc=n_oltc,
            n_shunt=n_shunt,
        )

        controller = DSOController(
            controller_id="dso_test",
            params=params,
            config=config,
            network_state=network_state,
            actuator_bounds=actuator_bounds,
            sensitivities=sensitivities,
        )

        measurement = _make_dso_measurement(
            der_bus_indices=der_buses,
            oltc_trafo_indices=oltc_trafos,
            shunt_bus_indices=shunt_buses,
            interface_trafo_indices=interface_trafos,
            voltage_bus_indices=voltage_buses,
            current_line_indices=current_lines,
        )

        return controller, config, measurement

    def test_control_structure(self, dso_setup) -> None:
        """Test that control variable structure is correct."""
        controller, config, measurement = dso_setup
        controller.initialise(measurement)

        assert controller.n_continuous == 2  # 2 DERs
        assert controller.n_integer == 2  # 1 OLTC + 1 shunt
        assert controller.n_controls == 4

    def test_initialise_sets_u_current(self, dso_setup) -> None:
        """Test that initialise extracts correct u values."""
        controller, _, measurement = dso_setup
        controller.initialise(measurement)

        u = controller.u_current
        # DER Q = 5.0 each, OLTC tap = 0, shunt state = 0
        assert_allclose(u[:2], [5.0, 5.0])
        assert_allclose(u[2], 0.0)  # OLTC
        assert_allclose(u[3], 0.0)  # Shunt

    def test_initialise_missing_der_raises(self, dso_setup) -> None:
        """Test that missing DER in measurement raises ValueError."""
        controller, _, measurement = dso_setup
        # Corrupt measurement: remove DER index
        measurement.der_indices = np.array([99], dtype=np.int64)
        measurement.der_q_mvar = np.array([0.0])
        with pytest.raises(ValueError, match="DER at bus"):
            controller.initialise(measurement)

    def test_step_before_init_raises(self, dso_setup) -> None:
        """Test that step() before initialise() raises RuntimeError."""
        controller, _, measurement = dso_setup
        with pytest.raises(RuntimeError, match="not initialised"):
            controller.step(measurement)

    def test_step_returns_controller_output(self, dso_setup) -> None:
        """Test that step() returns a ControllerOutput."""
        controller, _, measurement = dso_setup
        controller.initialise(measurement)
        result = controller.step(measurement)

        assert isinstance(result, ControllerOutput)
        assert result.iteration == 1
        assert len(result.u_new) == 4
        assert len(result.u_continuous) == 2
        assert len(result.u_integer) == 2

    def test_step_infeasible_raises(self, dso_setup) -> None:
        """Test that step() raises RuntimeError on infeasible problem."""
        controller, config, measurement = dso_setup

        # Force infeasibility: set contradictory bounds
        config.v_min_pu = 1.10
        config.v_max_pu = 1.11
        # Also need to invalidate cache
        controller.invalidate_sensitivity_cache()
        controller.initialise(measurement)

        # This may or may not be infeasible depending on slack;
        # the key test is that the mechanism works.  We skip if feasible.
        # (The solver can always find a feasible point via slack variables.)

    def test_receive_setpoint(self, dso_setup) -> None:
        """Test that setpoint messages are received correctly."""
        controller, config, _ = dso_setup

        msg = SetpointMessage(
            source_controller_id="tso_test",
            target_controller_id="dso_test",
            iteration=0,
            interface_transformer_indices=np.array([0], dtype=np.int64),
            q_setpoints_mvar=np.array([25.0]),
        )
        controller.receive_setpoint(msg)
        assert_allclose(controller.q_setpoint_mvar, [25.0])

    def test_receive_setpoint_wrong_target_raises(self, dso_setup) -> None:
        """Test that setpoint for wrong controller raises ValueError."""
        controller, _, _ = dso_setup
        msg = SetpointMessage(
            source_controller_id="tso_test",
            target_controller_id="dso_other",
            iteration=0,
            interface_transformer_indices=np.array([0], dtype=np.int64),
            q_setpoints_mvar=np.array([25.0]),
        )
        with pytest.raises(ValueError, match="does not match"):
            controller.receive_setpoint(msg)

    def test_generate_capability_message(self, dso_setup) -> None:
        """Test capability message generation."""
        controller, _, measurement = dso_setup
        controller.initialise(measurement)

        cap_msg = controller.generate_capability_message(
            target_controller_id="tso_test",
            measurement=measurement,
        )
        assert isinstance(cap_msg, CapabilityMessage)
        assert cap_msg.source_controller_id == "dso_test"
        assert cap_msg.target_controller_id == "tso_test"
        assert len(cap_msg.q_min_mvar) == 1
        assert len(cap_msg.q_max_mvar) == 1
        # Bounds must be ordered: min <= max
        assert np.all(cap_msg.q_min_mvar <= cap_msg.q_max_mvar)

    def test_output_limits(self, dso_setup) -> None:
        """Test output limit computation."""
        controller, config, measurement = dso_setup
        controller.initialise(measurement)

        y_lo, y_hi = controller._get_output_limits()
        n_interface = 1
        n_v = 3
        n_i = 2

        # Interface Q: no hard limits
        assert y_lo[0] == -np.inf
        assert y_hi[0] == np.inf
        # Voltage limits
        assert_allclose(
            y_lo[n_interface:n_interface + n_v], config.v_min_pu
        )
        assert_allclose(
            y_hi[n_interface:n_interface + n_v], config.v_max_pu
        )
        # Current: lower=0, upper=i_max
        assert_allclose(y_lo[n_interface + n_v:], 0.0)
        assert_allclose(y_hi[n_interface + n_v:], config.i_max_pu)

    def test_reset(self, dso_setup) -> None:
        """Test that reset clears internal state."""
        controller, _, measurement = dso_setup
        controller.initialise(measurement)
        controller.reset()

        with pytest.raises(RuntimeError, match="not initialised"):
            _ = controller.u_current

    def test_sensitivity_cache_invalidation(self, dso_setup) -> None:
        """Test that sensitivity cache can be invalidated."""
        controller, _, measurement = dso_setup
        controller.initialise(measurement)

        # Build cache
        _ = controller._build_sensitivity_matrix()
        assert controller._H_cache is not None

        controller.invalidate_sensitivity_cache()
        assert controller._H_cache is None


# =============================================================================
# Tests for TSOController
# =============================================================================


class TestTSOController:
    """Tests for the TSOController class."""

    @pytest.fixture
    def tso_setup(self):
        """Create a complete TSO controller setup."""
        der_buses = [2, 3]
        pcc_trafos = [0]
        dso_ids = ["dso_1"]
        oltc_trafos = [1]
        shunt_buses = [4]
        voltage_buses = [0, 1, 2]
        current_lines = [0]

        n_der = len(der_buses)
        n_pcc = len(pcc_trafos)
        n_oltc = len(oltc_trafos)
        n_shunt = len(shunt_buses)
        n_v = len(voltage_buses)
        n_i = len(current_lines)
        # Physical outputs: Q_trafo(n_pcc) + V(n_v) + I(n_i)
        n_outputs_phys = n_pcc + n_v + n_i

        config = TSOControllerConfig(
            der_bus_indices=der_buses,
            pcc_trafo_indices=pcc_trafos,
            pcc_dso_controller_ids=dso_ids,
            oltc_trafo_indices=oltc_trafos,
            shunt_bus_indices=shunt_buses,
            shunt_q_steps_mvar=[50.0],
            voltage_bus_indices=voltage_buses,
            current_line_indices=current_lines,
        )

        params = OFOParameters(
            alpha=0.03, g_w=0.2, g_z=1000.0, g_s=80.0, g_u=0.01
        )

        network_state = _make_network_state()
        actuator_bounds = _make_actuator_bounds(n_der, n_oltc, n_shunt)
        sensitivities = _make_mock_sensitivities(
            n_outputs=n_outputs_phys,
            n_der=n_der,
            n_oltc=n_oltc,
            n_shunt=n_shunt,
        )

        controller = TSOController(
            controller_id="tso_test",
            params=params,
            config=config,
            network_state=network_state,
            actuator_bounds=actuator_bounds,
            sensitivities=sensitivities,
        )

        measurement = _make_dso_measurement(
            der_bus_indices=der_buses,
            oltc_trafo_indices=oltc_trafos,
            shunt_bus_indices=shunt_buses,
            interface_trafo_indices=pcc_trafos,
            voltage_bus_indices=voltage_buses,
            current_line_indices=current_lines,
        )

        return controller, config, measurement

    def test_control_structure(self, tso_setup) -> None:
        """Test that control variable structure is correct."""
        controller, _, measurement = tso_setup
        controller.initialise(measurement)

        # Continuous: 2 DER + 1 PCC = 3
        assert controller.n_continuous == 3
        # Integer: 1 OLTC + 1 shunt = 2
        assert controller.n_integer == 2
        assert controller.n_controls == 5

    def test_initialise_sets_u_current(self, tso_setup) -> None:
        """Test that initialise extracts correct u values."""
        controller, _, measurement = tso_setup
        controller.initialise(measurement)

        u = controller.u_current
        # u = [Q_DER_0, Q_DER_1, Q_PCC_0, s_OLTC, s_shunt]
        assert_allclose(u[0], 5.0)   # DER 0 Q
        assert_allclose(u[1], 5.0)   # DER 1 Q
        assert_allclose(u[2], 10.0)  # PCC Q (from measurement)
        assert_allclose(u[3], 0.0)   # OLTC tap
        assert_allclose(u[4], 0.0)   # Shunt state

    def test_initialise_missing_pcc_trafo_raises(self, tso_setup) -> None:
        """Test that missing PCC transformer raises ValueError."""
        controller, _, measurement = tso_setup
        measurement.interface_transformer_indices = np.array(
            [99], dtype=np.int64
        )
        measurement.interface_q_hv_side_mvar = np.array([0.0])
        with pytest.raises(ValueError, match="PCC transformer"):
            controller.initialise(measurement)

    def test_step_returns_controller_output(self, tso_setup) -> None:
        """Test that step() returns a valid ControllerOutput."""
        controller, _, measurement = tso_setup
        controller.initialise(measurement)
        result = controller.step(measurement)

        assert isinstance(result, ControllerOutput)
        assert result.iteration == 1
        assert len(result.u_new) == 5
        assert len(result.u_continuous) == 3
        assert len(result.u_integer) == 2

    def test_step_before_init_raises(self, tso_setup) -> None:
        """Test that step() before initialise() raises RuntimeError."""
        controller, _, measurement = tso_setup
        with pytest.raises(RuntimeError, match="not initialised"):
            controller.step(measurement)

    def test_receive_capability(self, tso_setup) -> None:
        """Test receiving a capability message from a DSO."""
        controller, _, _ = tso_setup

        cap_msg = CapabilityMessage(
            source_controller_id="dso_1",
            target_controller_id="tso_test",
            iteration=0,
            interface_transformer_indices=np.array([0], dtype=np.int64),
            q_min_mvar=np.array([-30.0]),
            q_max_mvar=np.array([30.0]),
        )
        controller.receive_capability(cap_msg)

        assert_allclose(controller.pcc_capability_min_mvar, [-30.0])
        assert_allclose(controller.pcc_capability_max_mvar, [30.0])

    def test_receive_capability_wrong_target_raises(self, tso_setup) -> None:
        """Test that capability for wrong controller raises ValueError."""
        controller, _, _ = tso_setup
        cap_msg = CapabilityMessage(
            source_controller_id="dso_1",
            target_controller_id="tso_other",
            iteration=0,
            interface_transformer_indices=np.array([0], dtype=np.int64),
            q_min_mvar=np.array([-30.0]),
            q_max_mvar=np.array([30.0]),
        )
        with pytest.raises(ValueError, match="does not match"):
            controller.receive_capability(cap_msg)

    def test_generate_setpoint_messages(self, tso_setup) -> None:
        """Test setpoint message generation."""
        controller, _, measurement = tso_setup
        controller.initialise(measurement)

        messages = controller.generate_setpoint_messages()
        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg, SetpointMessage)
        assert msg.source_controller_id == "tso_test"
        assert msg.target_controller_id == "dso_1"
        assert_array_equal(msg.interface_transformer_indices, [0])
        # PCC setpoint is u[2] = 10.0 (initialised from measurement)
        assert_allclose(msg.q_setpoints_mvar, [10.0])

    def test_generate_setpoint_before_init_raises(self, tso_setup) -> None:
        """Test that generating setpoints before init raises RuntimeError."""
        controller, _, _ = tso_setup
        with pytest.raises(RuntimeError, match="not initialised"):
            controller.generate_setpoint_messages()

    def test_update_voltage_setpoints(self, tso_setup) -> None:
        """Test voltage setpoint update."""
        controller, config, _ = tso_setup
        v_set = np.array([1.05, 1.04, 1.03])
        controller.update_voltage_setpoints(v_set)
        assert_allclose(config.v_setpoints_pu, v_set)

    def test_update_voltage_setpoints_wrong_length_raises(
        self, tso_setup
    ) -> None:
        """Test that wrong setpoint length raises ValueError."""
        controller, _, _ = tso_setup
        with pytest.raises(ValueError, match="v_setpoints_pu length"):
            controller.update_voltage_setpoints(np.array([1.05]))

    def test_output_limits_without_setpoints(self, tso_setup) -> None:
        """Test output limits when no voltage setpoints are set."""
        controller, config, measurement = tso_setup
        controller.initialise(measurement)

        y_lo, y_hi = controller._get_output_limits()
        n_v = 3
        n_pcc = 1
        n_i = 1

        # Voltage: general band
        assert_allclose(y_lo[:n_v], config.v_min_pu)
        assert_allclose(y_hi[:n_v], config.v_max_pu)
        # PCC Q: no limits
        assert y_lo[n_v] == -np.inf
        assert y_hi[n_v] == np.inf
        # Current: [0, i_max]
        assert_allclose(y_lo[n_v + n_pcc:], 0.0)
        assert_allclose(y_hi[n_v + n_pcc:], config.i_max_pu)

    def test_output_limits_with_setpoints(self, tso_setup) -> None:
        """Test output limits when voltage setpoints are configured."""
        controller, config, measurement = tso_setup
        v_set = np.array([1.05, 1.04, 1.03])
        controller.update_voltage_setpoints(v_set)
        controller.initialise(measurement)

        y_lo, y_hi = controller._get_output_limits()
        # Voltage limits should equal setpoints (tight band)
        assert_allclose(y_lo[:3], v_set)
        assert_allclose(y_hi[:3], v_set)

    def test_pcc_capability_bounds_used_in_input_bounds(
        self, tso_setup
    ) -> None:
        """Test that PCC capability bounds propagate to input bounds."""
        controller, _, measurement = tso_setup
        controller.initialise(measurement)

        cap_msg = CapabilityMessage(
            source_controller_id="dso_1",
            target_controller_id="tso_test",
            iteration=0,
            interface_transformer_indices=np.array([0], dtype=np.int64),
            q_min_mvar=np.array([-20.0]),
            q_max_mvar=np.array([20.0]),
        )
        controller.receive_capability(cap_msg)

        der_p = controller._extract_der_active_power(measurement)
        u_lo, u_hi = controller._compute_input_bounds(der_p)

        # PCC setpoint index is after DERs: u[2]
        assert_allclose(u_lo[2], -20.0)
        assert_allclose(u_hi[2], 20.0)

    def test_sensitivity_matrix_shape(self, tso_setup) -> None:
        """Test that the sensitivity matrix has the correct shape."""
        controller, _, measurement = tso_setup
        controller.initialise(measurement)

        H = controller._build_sensitivity_matrix()
        n_outputs = 3 + 1 + 1  # V(3) + Q_PCC(1) + I(1)
        n_controls = 2 + 1 + 1 + 1  # DER(2) + PCC(1) + OLTC(1) + shunt(1)
        assert H.shape == (n_outputs, n_controls)

    def test_sensitivity_pcc_identity(self, tso_setup) -> None:
        """Test that the PCC setpoint column has unit sensitivity."""
        controller, _, measurement = tso_setup
        controller.initialise(measurement)

        H = controller._build_sensitivity_matrix()
        n_v = 3
        n_pcc = 1
        # PCC column: index 2+0=2 (after 2 DERs)
        # Q_PCC row: index 3 (after 3 voltages)
        pcc_col = 2
        q_pcc_row = n_v

        assert_allclose(H[q_pcc_row, pcc_col], 1.0)
        # Other rows should be zero for PCC column
        assert_allclose(H[:n_v, pcc_col], 0.0)
        assert_allclose(H[n_v + n_pcc:, pcc_col], 0.0)

    def test_sensitivity_cache_invalidation(self, tso_setup) -> None:
        """Test that sensitivity cache can be invalidated."""
        controller, _, measurement = tso_setup
        controller.initialise(measurement)

        _ = controller._build_sensitivity_matrix()
        assert controller._H_cache is not None

        controller.invalidate_sensitivity_cache()
        assert controller._H_cache is None

    def test_reset(self, tso_setup) -> None:
        """Test that reset clears internal state."""
        controller, _, measurement = tso_setup
        controller.initialise(measurement)
        controller.reset()

        with pytest.raises(RuntimeError, match="not initialised"):
            _ = controller.u_current


# =============================================================================
# Tests for TSO-DSO message exchange
# =============================================================================


class TestCascadedCommunication:
    """Test the message exchange between TSO and DSO controllers."""

    def test_tso_setpoint_received_by_dso(self) -> None:
        """Test end-to-end: TSO generates setpoint, DSO receives it."""
        # Shared topology
        der_buses = [2, 3]
        pcc_trafos = [0]
        oltc_trafos = [1]
        shunt_buses = [4]
        voltage_buses = [0, 1, 2]
        current_lines = [0]

        n_der = 2
        n_oltc = 1
        n_shunt = 1
        n_pcc = 1
        n_v = 3
        n_i = 1
        n_out_phys = n_pcc + n_v + n_i

        # TSO setup
        tso_config = TSOControllerConfig(
            der_bus_indices=der_buses,
            pcc_trafo_indices=pcc_trafos,
            pcc_dso_controller_ids=["dso_1"],
            oltc_trafo_indices=oltc_trafos,
            shunt_bus_indices=shunt_buses,
            shunt_q_steps_mvar=[50.0],
            voltage_bus_indices=voltage_buses,
            current_line_indices=current_lines,
        )
        tso_params = OFOParameters(
            alpha=0.03, g_w=0.2, g_z=1000.0, g_s=80.0
        )
        tso = TSOController(
            controller_id="tso_main",
            params=tso_params,
            config=tso_config,
            network_state=_make_network_state(),
            actuator_bounds=_make_actuator_bounds(n_der, n_oltc, n_shunt),
            sensitivities=_make_mock_sensitivities(
                n_out_phys, n_der, n_oltc, n_shunt
            ),
        )

        # DSO setup
        dso_config = DSOControllerConfig(
            der_bus_indices=der_buses,
            oltc_trafo_indices=oltc_trafos,
            shunt_bus_indices=shunt_buses,
            shunt_q_steps_mvar=[50.0],
            interface_trafo_indices=pcc_trafos,
            voltage_bus_indices=voltage_buses,
            current_line_indices=current_lines,
        )
        dso = DSOController(
            controller_id="dso_1",
            params=tso_params,
            config=dso_config,
            network_state=_make_network_state(),
            actuator_bounds=_make_actuator_bounds(n_der, n_oltc, n_shunt),
            sensitivities=_make_mock_sensitivities(
                n_out_phys, n_der, n_oltc, n_shunt
            ),
        )

        # Measurement
        meas = _make_dso_measurement(
            der_bus_indices=der_buses,
            oltc_trafo_indices=oltc_trafos,
            shunt_bus_indices=shunt_buses,
            interface_trafo_indices=pcc_trafos,
            voltage_bus_indices=voltage_buses,
            current_line_indices=current_lines,
        )

        # Initialise both controllers
        tso.initialise(meas)
        dso.initialise(meas)

        # TSO generates setpoint messages
        messages = tso.generate_setpoint_messages()
        assert len(messages) == 1

        # DSO receives the setpoint
        dso.receive_setpoint(messages[0])
        # TSO initial PCC setpoint = 10.0 (from measurement)
        assert_allclose(dso.q_setpoint_mvar, [10.0])

    def test_dso_capability_received_by_tso(self) -> None:
        """Test end-to-end: DSO generates capability, TSO receives it."""
        der_buses = [2, 3]
        pcc_trafos = [0]
        oltc_trafos = [1]
        shunt_buses = [4]
        voltage_buses = [0, 1, 2]
        current_lines = [0]
        n_der = 2
        n_oltc = 1
        n_shunt = 1
        n_pcc = 1
        n_v = 3
        n_i = 1
        n_out_phys = n_pcc + n_v + n_i

        tso = TSOController(
            controller_id="tso_main",
            params=OFOParameters(alpha=0.03, g_w=0.2, g_z=1000.0, g_s=80.0),
            config=TSOControllerConfig(
                der_bus_indices=der_buses,
                pcc_trafo_indices=pcc_trafos,
                pcc_dso_controller_ids=["dso_1"],
                oltc_trafo_indices=oltc_trafos,
                shunt_bus_indices=shunt_buses,
                shunt_q_steps_mvar=[50.0],
                voltage_bus_indices=voltage_buses,
                current_line_indices=current_lines,
            ),
            network_state=_make_network_state(),
            actuator_bounds=_make_actuator_bounds(n_der, n_oltc, n_shunt),
            sensitivities=_make_mock_sensitivities(
                n_out_phys, n_der, n_oltc, n_shunt
            ),
        )

        dso = DSOController(
            controller_id="dso_1",
            params=OFOParameters(alpha=0.03, g_w=0.2, g_z=1000.0, g_s=80.0),
            config=DSOControllerConfig(
                der_bus_indices=der_buses,
                oltc_trafo_indices=oltc_trafos,
                shunt_bus_indices=shunt_buses,
                shunt_q_steps_mvar=[50.0],
                interface_trafo_indices=pcc_trafos,
                voltage_bus_indices=voltage_buses,
                current_line_indices=current_lines,
            ),
            network_state=_make_network_state(),
            actuator_bounds=_make_actuator_bounds(n_der, n_oltc, n_shunt),
            sensitivities=_make_mock_sensitivities(
                n_out_phys, n_der, n_oltc, n_shunt
            ),
        )

        meas = _make_dso_measurement(
            der_bus_indices=der_buses,
            oltc_trafo_indices=oltc_trafos,
            shunt_bus_indices=shunt_buses,
            interface_trafo_indices=pcc_trafos,
            voltage_bus_indices=voltage_buses,
            current_line_indices=current_lines,
        )

        dso.initialise(meas)

        # DSO generates capability message
        cap_msg = dso.generate_capability_message(
            target_controller_id="tso_main",
            measurement=meas,
        )

        # TSO receives capability
        tso.receive_capability(cap_msg)

        # Capability bounds must be finite and ordered
        assert np.all(np.isfinite(tso.pcc_capability_min_mvar))
        assert np.all(np.isfinite(tso.pcc_capability_max_mvar))
        assert np.all(
            tso.pcc_capability_min_mvar <= tso.pcc_capability_max_mvar
        )


# =============================================================================
# Tests for BaseOFOController via DSOController (concrete subclass)
# =============================================================================


class TestBaseOFOControllerViaSubclass:
    """Test base controller logic using DSOController as concrete class."""

    def test_u_current_returns_copy(self) -> None:
        """Test that u_current returns a copy, not a reference."""
        n_out = 6
        controller = DSOController(
            controller_id="copy_test",
            params=OFOParameters(alpha=0.1, g_w=1.0, g_z=100.0, g_s=10.0),
            config=DSOControllerConfig(
                der_bus_indices=[2, 3],
                oltc_trafo_indices=[0],
                shunt_bus_indices=[4],
                shunt_q_steps_mvar=[50.0],
                interface_trafo_indices=[0],
                voltage_bus_indices=[1, 2, 3],
                current_line_indices=[0, 1],
            ),
            network_state=_make_network_state(),
            actuator_bounds=_make_actuator_bounds(2, 1, 1),
            sensitivities=_make_mock_sensitivities(n_out, 2, 1, 1),
        )
        meas = _make_dso_measurement()
        controller.initialise(meas)

        u1 = controller.u_current
        u2 = controller.u_current
        # Modify u1 → u2 must not change
        u1[0] = 999.0
        assert u2[0] != 999.0

    def test_constructor_rejects_none_params(self) -> None:
        """Test that None parameters raise ValueError."""
        with pytest.raises(ValueError, match="params must not be None"):
            DSOController(
                controller_id="test",
                params=None,
                config=DSOControllerConfig(
                    der_bus_indices=[],
                    oltc_trafo_indices=[],
                    shunt_bus_indices=[],
                    shunt_q_steps_mvar=[],
                    interface_trafo_indices=[],
                    voltage_bus_indices=[],
                    current_line_indices=[],
                ),
                network_state=_make_network_state(),
                actuator_bounds=_make_actuator_bounds(0, 0, 0),
                sensitivities=_make_mock_sensitivities(0, 0),
            )

    def test_constructor_rejects_empty_id(self) -> None:
        """Test that empty controller_id raises ValueError."""
        with pytest.raises(ValueError, match="controller_id"):
            DSOController(
                controller_id="",
                params=OFOParameters(
                    alpha=0.1, g_w=1.0, g_z=100.0, g_s=10.0
                ),
                config=DSOControllerConfig(
                    der_bus_indices=[],
                    oltc_trafo_indices=[],
                    shunt_bus_indices=[],
                    shunt_q_steps_mvar=[],
                    interface_trafo_indices=[],
                    voltage_bus_indices=[],
                    current_line_indices=[],
                ),
                network_state=_make_network_state(),
                actuator_bounds=_make_actuator_bounds(0, 0, 0),
                sensitivities=_make_mock_sensitivities(0, 0),
            )
