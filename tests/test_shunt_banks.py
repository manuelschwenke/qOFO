"""
Unit tests for the switched-shunt bank model (MSC / MSR).

The dispatch decision itself belongs to the zone MIQP; what is tested here is
what the bank contributes to it — the committed level, the directions the
device currently permits (level limits and minimum holding time), the
predicted-voltage guard, and the commit record the runner applies. No
pandapower network is built, so these are fast and isolate the model from the
plant.

Author: Manuel Schwenke
Date: 2026-10-22
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from controller.shunt_banks import (  # noqa: E402
    MSC,
    MSR,
    ShuntBankConfig,
    make_bank,
)


def _base_cfg(**overrides) -> ShuntBankConfig:
    """A nominal MSC config; override individual fields per test."""
    kw = dict(
        shunt_idx=0,
        bus_idx=10,
        interface_trafo3w_idx=0,
        dso_id="DSO_1",
        kind="MSC",
        q_step_mvar=50.0,
        n_levels=3,
        t_dwell_s=0.0,
        y_h_min=0.90,
        y_h_max=1.10,
    )
    kw.update(overrides)
    return ShuntBankConfig(**kw)


# One observed HV bus at 1.00 p.u., +0.0005 p.u. per Mvar of injection, so one
# 50 Mvar step moves it by 0.025 p.u.
_V = np.array([1.00])
_H = np.array([0.0005])


# ---------------------------------------------------------------------------
#  Config fail-fast
# ---------------------------------------------------------------------------

def test_config_rejects_bad_kind_and_q_step():
    with pytest.raises(ValueError, match="kind"):
        _base_cfg(kind="SVC")
    with pytest.raises(ValueError, match="q_step_mvar"):
        _base_cfg(q_step_mvar=0.0)


def test_config_rejects_bad_levels_and_dwell():
    with pytest.raises(ValueError, match="n_levels"):
        _base_cfg(n_levels=0)
    with pytest.raises(ValueError, match="t_dwell_s"):
        _base_cfg(t_dwell_s=-1.0)


def test_config_rejects_bad_band():
    with pytest.raises(ValueError, match="y_h_min"):
        _base_cfg(y_h_min=1.10, y_h_max=0.90)


# ---------------------------------------------------------------------------
#  Permitted directions: the bounds the MIQP is given
# ---------------------------------------------------------------------------

def test_both_directions_available_in_the_interior():
    bank = MSC(_base_cfg())
    bank.level = 1
    assert sorted(bank.available_directions(0.0)) == [-1, 1]


def test_level_limits_remove_a_direction():
    bank = MSC(_base_cfg())
    assert bank.available_directions(0.0) == [1]          # de-energised
    bank.level = bank.config.n_levels
    assert bank.available_directions(0.0) == [-1]          # fully engaged


def test_holding_time_blocks_every_direction():
    bank = MSC(_base_cfg(t_dwell_s=1800.0))
    bank.commit_selected(+1, 0.0)
    assert bank.available_directions(1799.0) == []
    assert sorted(bank.available_directions(1800.0)) == [-1, 1]


def test_first_commit_is_always_outside_the_holding_time():
    bank = MSC(_base_cfg(t_dwell_s=1800.0))
    assert bank.available_directions(0.0) == [1]
    assert math.isinf(bank._last_switch_t)


def test_non_finite_time_is_rejected():
    bank = MSC(_base_cfg())
    with pytest.raises(ValueError, match="finite"):
        bank.available_directions(float("nan"))


# ---------------------------------------------------------------------------
#  Commit record
# ---------------------------------------------------------------------------

def test_commit_advances_level_and_clock():
    bank = MSC(_base_cfg(t_dwell_s=600.0))
    commit = bank.commit_selected(+1, 120.0)
    assert (commit.old_level, commit.new_level, commit.direction) == (0, 1, +1)
    assert commit.pp_step_new == 1
    assert commit.shunt_idx == 0 and commit.dso_id == "DSO_1"
    assert commit.q_step_mvar == 50.0 and commit.kind == "MSC"
    assert bank.level == 1
    assert bank.q_eq_phys == 50.0
    assert bank._last_switch_t == 120.0


def test_commit_rejects_a_direction_the_device_forbids():
    bank = MSC(_base_cfg(t_dwell_s=1800.0))
    bank.commit_selected(+1, 0.0)
    with pytest.raises(ValueError, match="holding time or a level limit"):
        bank.commit_selected(+1, 900.0)        # inside the holding time
    bank.level = bank.config.n_levels
    with pytest.raises(ValueError, match="holding time or a level limit"):
        bank.commit_selected(+1, 100_000.0)    # at the level limit


def test_release_decrements_the_level():
    bank = MSR(_base_cfg(kind="MSR"))
    bank.level = 2
    commit = bank.commit_selected(-1, 0.0)
    assert (commit.old_level, commit.new_level) == (2, 1)
    assert bank.level == 1


# ---------------------------------------------------------------------------
#  Predicted-voltage guard
# ---------------------------------------------------------------------------

def test_guard_blocks_overshoot_and_admits_inside_the_band():
    # 1.09 + 0.025 = 1.115 > y_h_max; the release direction stays admissible.
    bank = MSC(_base_cfg())
    bank.level = 1
    assert not bank._feasible(+1, np.array([1.09]), _H)
    assert bank._feasible(-1, np.array([1.09]), _H)
    assert bank._feasible(+1, _V, _H)


def test_guard_checks_every_observed_bus():
    bank = MSC(_base_cfg())
    v = np.array([1.00, 1.09])
    h = np.array([0.0005, 0.0005])
    assert not bank._feasible(+1, v, h)


# ---------------------------------------------------------------------------
#  Device classes
# ---------------------------------------------------------------------------

def test_make_bank_dispatch_and_expected_signs():
    assert isinstance(make_bank(_base_cfg(kind="MSC")), MSC)
    assert isinstance(make_bank(_base_cfg(kind="MSR")), MSR)
    assert MSC(_base_cfg()).expected_h_sign == +1
    assert MSR(_base_cfg(kind="MSR")).expected_h_sign == -1
    # The lattice level maps identically onto the pandapower step for both
    # classes: the capacitive / inductive direction is the sign of q_mvar.
    for kind in ("MSC", "MSR"):
        bank = make_bank(_base_cfg(kind=kind))
        assert [bank.pp_step_for_level(ell) for ell in (0, 1, 2)] == [0, 1, 2]


def test_make_bank_rejects_unknown_kind():
    cfg = _base_cfg()
    # Bypass the config's own guard to reach the factory's.
    object.__setattr__(cfg, "kind", "SVC")
    with pytest.raises(ValueError, match="Unknown shunt kind"):
        make_bank(cfg)


def test_q_eq_phys_tracks_the_level():
    bank = make_bank(_base_cfg(q_step_mvar=25.0))
    assert bank.q_eq_phys == 0.0
    bank.level = 2
    assert bank.q_eq_phys == 50.0
    assert replace(bank.config, n_levels=4).n_levels == 4
