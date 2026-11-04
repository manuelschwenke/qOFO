"""
Unit tests for the STS coupler hold after a TS bank commit
==========================================================

``BaseOFOController.hold_oltcs(n)`` pins every OLTC integer for the next
``n`` calls of ``step()`` through the iteration lock of ``int_cooldown``;
the runner calls it on the hosting STS when
``MultiTSOConfig.sts_coupler_hold_after_bank > 0`` and the TS-MIQP commits a
bank there.  Shunt integers and continuous actuators stay free.

Date: 2026-11-04
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import controller.base_controller as base_mod
from configs.config import MultiTSOConfig
from controller.base_controller import OFOParameters
from controller.dso_controller import DSOController, DSOControllerConfig
from tests.test_controller import (  # noqa: E402
    _make_actuator_bounds,
    _make_dso_measurement,
    _make_mock_sensitivities,
    _make_network_state,
)


@pytest.fixture
def dso():
    """DSO controller with one coupler OLTC and one shunt, no wall-clock lock."""
    der_buses = [2, 3]
    oltc_trafos = [0]
    shunt_buses = [4]
    interface_trafos = [0]
    voltage_buses = [1, 2, 3]
    current_lines = [0, 1]
    n_outputs = len(interface_trafos) + len(voltage_buses) + len(current_lines)

    config = DSOControllerConfig(
        der_indices=der_buses,
        oltc_trafo_indices=oltc_trafos,
        shunt_bus_indices=shunt_buses,
        shunt_q_steps_mvar=[50.0],
        interface_trafo_indices=interface_trafos,
        voltage_bus_indices=voltage_buses,
        current_line_indices=current_lines,
    )
    controller = DSOController(
        controller_id="dso_test_hold",
        params=OFOParameters(g_w=0.2, g_z=1000.0, g_u=0.01, int_cooldown=5),
        config=config,
        network_state=_make_network_state(),
        actuator_bounds=_make_actuator_bounds(2, 1, 1),
        sensitivities=_make_mock_sensitivities(
            n_outputs=n_outputs, n_der=2, n_oltc=1, n_shunt=1,
            der_indices=der_buses, oltc_trafo_indices=oltc_trafos,
            interface_trafo_indices=interface_trafos,
        ),
    )
    measurement = _make_dso_measurement(
        der_indices=der_buses,
        oltc_trafo_indices=oltc_trafos,
        shunt_bus_indices=shunt_buses,
        interface_trafo_indices=interface_trafos,
        voltage_bus_indices=voltage_buses,
        current_line_indices=current_lines,
    )
    return controller, measurement


@pytest.fixture
def bounds_log(monkeypatch):
    """Record the input bounds every ``step()`` hands to the MIQP builder."""
    log = []
    real = base_mod.build_miqp_problem

    def spy(*args, **kwargs):
        log.append((np.array(kwargs["u_lower"]), np.array(kwargs["u_upper"])))
        return real(*args, **kwargs)

    monkeypatch.setattr(base_mod, "build_miqp_problem", spy)
    return log


def _oltc_and_shunt(controller):
    oltc = sorted(controller._oltc_int_indices)
    shunt = [i for i in controller._integer_indices if i not in oltc]
    return oltc, shunt


class TestHoldOltcs:

    def test_requires_initialisation(self, dso) -> None:
        controller, _ = dso
        with pytest.raises(RuntimeError):
            controller.hold_oltcs(1)

    def test_locks_oltcs_only(self, dso) -> None:
        controller, measurement = dso
        controller.initialise(measurement)
        oltc, shunt = _oltc_and_shunt(controller)
        assert oltc and shunt
        k = controller.iteration
        controller.hold_oltcs(3)
        assert all(controller._int_lock_until[i] == k + 3 for i in oltc)
        assert all(i not in controller._int_lock_until for i in shunt)

    def test_non_positive_is_noop(self, dso) -> None:
        controller, measurement = dso
        controller.initialise(measurement)
        controller.hold_oltcs(0)
        controller.hold_oltcs(-2)
        assert controller._int_lock_until == {}

    def test_longer_lock_is_kept(self, dso) -> None:
        controller, measurement = dso
        controller.initialise(measurement)
        oltc, _ = _oltc_and_shunt(controller)
        k = controller.iteration
        controller._int_lock_until[oltc[0]] = k + 6     # e.g. after a tap move
        controller.hold_oltcs(2)
        assert controller._int_lock_until[oltc[0]] == k + 6

    def test_shorter_lock_is_extended(self, dso) -> None:
        controller, measurement = dso
        controller.initialise(measurement)
        oltc, _ = _oltc_and_shunt(controller)
        k = controller.iteration
        controller._int_lock_until[oltc[0]] = k + 1
        controller.hold_oltcs(4)
        assert controller._int_lock_until[oltc[0]] == k + 4

    def test_hold_one_pins_exactly_the_next_step(self, dso, bounds_log) -> None:
        """n = 1: the step of the commit instant is pinned, the next is free."""
        controller, measurement = dso
        controller.initialise(measurement)
        oltc, shunt = _oltc_and_shunt(controller)
        controller.hold_oltcs(1)
        controller.step(measurement)
        lo, hi = bounds_log[-1]
        assert all(lo[i] == hi[i] for i in oltc)
        assert all(lo[i] < hi[i] for i in shunt)
        # The held tap could not move, so no cooldown lock follows the hold.
        controller.step(measurement)
        lo, hi = bounds_log[-1]
        assert all(lo[i] < hi[i] for i in oltc)


class TestConfigField:

    def test_default_is_off(self) -> None:
        assert MultiTSOConfig().sts_coupler_hold_after_bank == 0

    @staticmethod
    def _miqp(hold):
        return dataclasses.replace(MultiTSOConfig(), shunt_dispatch="miqp",
                                   install_tso_tertiary_shunts=True,
                                   sts_coupler_hold_after_bank=hold)

    @pytest.mark.parametrize("bad", [-1, 1.5, "2"])
    def test_validation_rejects(self, bad) -> None:
        with pytest.raises(ValueError, match="sts_coupler_hold_after_bank"):
            self._miqp(bad).validate_integrator_mode()

    def test_validation_accepts(self) -> None:
        self._miqp(6).validate_integrator_mode()
