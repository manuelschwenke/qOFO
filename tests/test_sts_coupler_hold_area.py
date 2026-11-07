"""
Tests for the area-wide STS coupler hold ("hold over dispatch")
==============================================================

``sts_coupler_hold_scope = 'area'`` (2026-11-06): when the TS-MIQP of an area
commits a bank or a machine-transformer tap, every STS of that area holds its
coupler OLTCs for ``sts_coupler_hold_after_bank`` STS steps.  The runner block
calls :func:`experiments.runners._multi_tso_helpers.taps_changed` and
:func:`hold_area_couplers`; the lock itself is ``hold_oltcs`` of the base
controller, covered by ``tests/test_sts_coupler_hold.py``.

Author: Manuel Schwenke / Claude Code (2026-11-06)
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from configs.config import MultiTSOConfig
from experiments.runners._multi_tso_helpers import hold_area_couplers, taps_changed
from tests.test_sts_coupler_hold import dso  # noqa: F401  (fixture: initialised DSO controller)


class _FakeSTS:
    def __init__(self) -> None:
        self.calls = []

    def hold_oltcs(self, n: int) -> None:
        self.calls.append(n)


# ── configuration ────────────────────────────────────────────────────────────

def test_default_scope_is_host():
    assert MultiTSOConfig().sts_coupler_hold_scope == "host"


@pytest.mark.parametrize("scope", ["host", "area"])
def test_scope_accepted(scope):
    dataclasses.replace(MultiTSOConfig(), sts_coupler_hold_scope=scope,
                        sts_coupler_hold_after_bank=9).validate_integrator_mode()


@pytest.mark.parametrize("bad", ["zone", "", None])
def test_scope_rejected_even_without_banks(bad):
    cfg = dataclasses.replace(MultiTSOConfig(), sts_coupler_hold_scope=bad,
                              shunt_dispatch="off")
    with pytest.raises(ValueError, match="sts_coupler_hold_scope"):
        cfg.validate_integrator_mode()


# ── commit detection ─────────────────────────────────────────────────────────

def test_taps_changed():
    assert not taps_changed([0, 1, 1], np.array([0.0, 1.0, 1.0]))
    assert taps_changed([0, 1, 1], np.array([0.0, 2.0, 1.0]))
    assert not taps_changed([0, 1], np.array([0.4, 1.49]))   # rounding as in plant writes
    assert not taps_changed([], np.array([]))                  # area without MT OLTCs


# ── hold dispatch ────────────────────────────────────────────────────────────

def test_only_areas_with_a_commit_are_held():
    s1, s2, s3, s4 = (_FakeSTS() for _ in range(4))
    ctrls = {"DSO_1": s1, "DSO_2": s2, "DSO_3": s3, "DSO_4": s4}
    ids = {2: ["DSO_1", "DSO_1", "DSO_1", "DSO_2", "DSO_2", "DSO_2", "DSO_3", "DSO_3", "DSO_3"],
           3: ["DSO_4", "DSO_4", "DSO_4"], 1: []}
    held = hold_area_couplers({1: True, 2: True, 3: False}, ids, ctrls, 9)
    assert held == {2: ["DSO_1", "DSO_2", "DSO_3"]}
    assert s1.calls == s2.calls == s3.calls == [9]      # once per STS, not per coupler
    assert s4.calls == []


def test_local_sts_and_zero_hold_do_nothing():
    s1 = _FakeSTS()
    assert hold_area_couplers({2: True}, {2: ["DSO_1", "DSO_X"]}, {"DSO_1": s1}, 0) == {}
    assert s1.calls == []
    held = hold_area_couplers({2: True}, {2: ["DSO_1", "DSO_X"]}, {"DSO_1": s1}, 9)
    assert held == {2: ["DSO_1"]}                        # DSO_X has no controller (local)


def test_hold_with_real_controller_sets_the_lock(dso):
    """With the real controller: n = 9 locks the OLTC integers until
    iteration + 9 and leaves the shunt integer free."""
    ctrl, meas = dso
    ctrl.initialise(meas)
    held = hold_area_couplers({2: True}, {2: ["S"]}, {"S": ctrl}, 9)
    assert held == {2: ["S"]}
    for idx in ctrl._oltc_int_indices:
        assert ctrl._int_lock_until[idx] == ctrl.iteration + 9
    assert len(ctrl._int_lock_until) == len(ctrl._oltc_int_indices)
