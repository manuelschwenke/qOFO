"""Regression tests for the active SBX-H v6 configuration."""

from sbx_h.config import SBXConfig


def test_terminal_schedule_does_not_receive_implicit_priority() -> None:
    """The base contract must not give terminal errors implicit priority."""
    assert SBXConfig().w_track_factor == 1.0
