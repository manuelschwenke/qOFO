"""
Tests for AVR saturation handling in the TSO controller (Feature B).

Covers the mode classification state machine and the AVR reset that
runs on free→saturated transitions.  Uses a minimal TSOController set
up with real ActuatorBounds (for the Milano PQ-curve) but a MagicMock
JacobianSensitivities — the methods under test do not consult the
sensitivity cache, so no real power-flow is needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from controller.base_controller import OFOParameters
from controller.tso_controller import TSOController, TSOControllerConfig
from core.actuator_bounds import ActuatorBounds, GeneratorParameters
from core.measurement import Measurement
from core.network_state import NetworkState
from sensitivity.jacobian import JacobianSensitivities


def _make_network_state() -> NetworkState:
    return NetworkState(
        bus_indices=np.array([0, 1, 2], dtype=np.int64),
        voltage_magnitudes_pu=np.array([1.02, 1.03, 1.01]),
        voltage_angles_rad=np.zeros(3),
        slack_bus_index=0,
        pv_bus_indices=np.array([1], dtype=np.int64),
        pq_bus_indices=np.array([2], dtype=np.int64),
        transformer_indices=np.array([], dtype=np.int64),
        tap_positions=np.array([]),
        source_case="sat_test",
        timestamp="2026-04-17T00:00:00",
        cached_at_iteration=0,
    )


def _make_gen_params(n_gen: int = 1) -> list:
    """One synchronous machine with a typical turbo-generator capability."""
    return [
        GeneratorParameters(
            s_rated_mva=100.0,
            p_max_mw=90.0,
            p_min_mw=10.0,
            xd_pu=1.8,
            i_f_max_pu=2.7,
            beta=0.15,
            q0_pu=0.4,
        )
        for _ in range(n_gen)
    ]


def _make_actuator_bounds(n_gen: int = 1) -> ActuatorBounds:
    """Minimal ActuatorBounds populated with gen_params for Q-curve lookup."""
    return ActuatorBounds(
        der_indices=np.array([0], dtype=np.int64),
        der_s_rated_mva=np.array([100.0]),
        der_p_max_mw=np.array([80.0]),
        oltc_indices=np.array([], dtype=np.int64),
        oltc_tap_min=np.array([], dtype=np.int64),
        oltc_tap_max=np.array([], dtype=np.int64),
        shunt_indices=np.array([], dtype=np.int64),
        shunt_q_mvar=np.array([]),
        gen_params=_make_gen_params(n_gen),
    )


def _make_measurement(q_mvar: float, gen_vm_pu: float = 1.03) -> Measurement:
    """Single-gen measurement; P constant at 50 MW and V at 1.03 pu by default."""
    return Measurement(
        iteration=0,
        bus_indices=np.array([0, 1, 2], dtype=np.int64),
        voltage_magnitudes_pu=np.array([1.02, gen_vm_pu, 1.01]),
        branch_indices=np.array([], dtype=np.int64),
        current_magnitudes_ka=np.array([]),
        interface_transformer_indices=np.array([], dtype=np.int64),
        interface_q_hv_side_mvar=np.array([]),
        der_indices=np.array([0], dtype=np.int64),
        der_q_mvar=np.array([0.0]),
        der_p_mw=np.array([10.0]),
        oltc_indices=np.array([], dtype=np.int64),
        oltc_tap_positions=np.array([], dtype=np.int64),
        shunt_indices=np.array([], dtype=np.int64),
        shunt_states=np.array([], dtype=np.int64),
        gen_indices=np.array([0], dtype=np.int64),
        gen_vm_pu=np.array([gen_vm_pu]),
        gen_p_mw=np.array([50.0]),
        gen_q_mvar=np.array([q_mvar]),
    )


def _make_tso() -> TSOController:
    """Build a minimal TSOController for method-level testing."""
    config = TSOControllerConfig(
        der_indices=[0],
        pcc_trafo_indices=[],
        pcc_dso_controller_ids=[],
        oltc_trafo_indices=[],
        shunt_bus_indices=[],
        shunt_q_steps_mvar=[],
        voltage_bus_indices=[1],
        current_line_indices=[],
        gen_indices=[0],
        gen_bus_indices=[1],
        gen_vm_min_pu=0.95,
        gen_vm_max_pu=1.07,
        # tight hysteresis band so tests can drive transitions cleanly
        sat_eps_enter_mvar=2.0,
        sat_eps_exit_mvar=10.0,
        enable_saturation_mode=True,
    )
    params = OFOParameters(g_w=0.2, g_z=1000.0, g_u=0.0)
    sens = MagicMock(spec=JacobianSensitivities)
    sens.net = MagicMock()
    sens.net.sgen = pd.DataFrame({"bus": [2]}, index=[0])
    sens.net.trafo = pd.DataFrame()
    sens.net.trafo3w = pd.DataFrame()
    sens.net.gen = pd.DataFrame({"bus": [1], "in_service": [True]}, index=[0])
    return TSOController(
        controller_id="tso_sat_test",
        params=params,
        config=config,
        network_state=_make_network_state(),
        actuator_bounds=_make_actuator_bounds(n_gen=1),
        sensitivities=sens,
    )


class TestClassifySaturationModes:
    def test_free_generator_stays_free_inside_band(self) -> None:
        """Gen at mid-Q stays in free mode (0)."""
        tso = _make_tso()
        q_min, q_max = tso.actuator_bounds.compute_gen_q_bounds(
            np.array([50.0]), np.array([1.03]),
        )
        # Mid-range Q should be well away from either limit.
        q_mid = 0.5 * (q_min[0] + q_max[0])
        tso._classify_saturation_modes(_make_measurement(q_mvar=float(q_mid)))
        assert tso._sat_mode[0] == 0

    def test_enter_upper_saturation(self) -> None:
        """Q within eps_enter of q_max → mode flips to +1."""
        tso = _make_tso()
        q_min, q_max = tso.actuator_bounds.compute_gen_q_bounds(
            np.array([50.0]), np.array([1.03]),
        )
        # Cross inside the enter band (q_max − 0.1 * eps_enter).
        q = float(q_max[0]) - 0.1 * tso.config.sat_eps_enter_mvar
        tso._classify_saturation_modes(_make_measurement(q_mvar=q))
        assert tso._sat_mode[0] == +1

    def test_enter_lower_saturation(self) -> None:
        """Q within eps_enter of q_min → mode flips to -1."""
        tso = _make_tso()
        q_min, q_max = tso.actuator_bounds.compute_gen_q_bounds(
            np.array([50.0]), np.array([1.03]),
        )
        q = float(q_min[0]) + 0.1 * tso.config.sat_eps_enter_mvar
        tso._classify_saturation_modes(_make_measurement(q_mvar=q))
        assert tso._sat_mode[0] == -1

    def test_hysteresis_no_chatter_between_enter_and_exit(self) -> None:
        """Once saturated, Q must retreat past eps_exit (> eps_enter) to unlock.

        We first saturate, then step Q back into the band between
        (q_max − eps_exit) and (q_max − eps_enter) — the mode must stay +1.
        """
        tso = _make_tso()
        q_min, q_max = tso.actuator_bounds.compute_gen_q_bounds(
            np.array([50.0]), np.array([1.03]),
        )
        # Step 1: saturate
        q_sat = float(q_max[0]) - 0.1 * tso.config.sat_eps_enter_mvar
        tso._classify_saturation_modes(_make_measurement(q_mvar=q_sat))
        assert tso._sat_mode[0] == +1
        # Step 2: inside hysteresis band (still saturated)
        q_mid = float(q_max[0]) - 0.5 * (
            tso.config.sat_eps_enter_mvar + tso.config.sat_eps_exit_mvar
        )
        tso._classify_saturation_modes(_make_measurement(q_mvar=q_mid))
        assert tso._sat_mode[0] == +1

    def test_desaturate_below_exit_threshold(self) -> None:
        """Q drops past (q_max − eps_exit) → mode returns to free."""
        tso = _make_tso()
        q_min, q_max = tso.actuator_bounds.compute_gen_q_bounds(
            np.array([50.0]), np.array([1.03]),
        )
        # Saturate first
        tso._classify_saturation_modes(
            _make_measurement(q_mvar=float(q_max[0]) - 0.1)
        )
        assert tso._sat_mode[0] == +1
        # De-saturate: well below q_max − eps_exit
        q_free = float(q_max[0]) - 2.0 * tso.config.sat_eps_exit_mvar
        tso._classify_saturation_modes(_make_measurement(q_mvar=q_free))
        assert tso._sat_mode[0] == 0


class TestApplyAvrModeReset:
    def test_reset_on_free_to_saturated_transition(self) -> None:
        """u_current[V_gen] realigns to measured V_gen on mode change 0 → ±1."""
        tso = _make_tso()
        tso._u_current = np.array([0.0, 1.04])  # [Q_der, V_gen]
        # Simulate classification producing a transition
        previous_modes = np.array([0], dtype=np.int8)
        tso._sat_mode = np.array([+1], dtype=np.int8)

        meas = _make_measurement(q_mvar=100.0, gen_vm_pu=1.05)
        tso.apply_avr_mode_reset(meas, previous_modes)
        # avr_start = n_der (1) + n_pcc (0) = 1
        assert tso._u_current[1] == pytest.approx(1.05)

    def test_no_reset_when_mode_unchanged(self) -> None:
        """Free→Free or Sat→Sat: u_current is untouched."""
        tso = _make_tso()
        tso._u_current = np.array([0.0, 1.04])
        previous_modes = np.array([0], dtype=np.int8)
        tso._sat_mode = np.array([0], dtype=np.int8)
        meas = _make_measurement(q_mvar=50.0, gen_vm_pu=1.05)
        tso.apply_avr_mode_reset(meas, previous_modes)
        assert tso._u_current[1] == pytest.approx(1.04)

    def test_no_reset_on_saturated_to_free_transition(self) -> None:
        """Sat → free: commanded V_gen is already at achieved value; no reset needed."""
        tso = _make_tso()
        tso._u_current = np.array([0.0, 1.04])
        previous_modes = np.array([+1], dtype=np.int8)
        tso._sat_mode = np.array([0], dtype=np.int8)
        meas = _make_measurement(q_mvar=50.0, gen_vm_pu=1.05)
        tso.apply_avr_mode_reset(meas, previous_modes)
        assert tso._u_current[1] == pytest.approx(1.04)
