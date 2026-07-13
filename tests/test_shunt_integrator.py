"""
Unit tests for the switched-shunt integrator (MSC / MSR).

These exercise the pure integrating dispatch logic in
``controller.shunt_integrator`` with synthetic gradient / sensitivity inputs —
no pandapower network is built, so the tests are fast and isolate the commit
logic (anti-windup clamp, hysteresis quantiser, dwell, daily budget, HV
feasibility guard) from the plant.

Author: Manuel Schwenke
Date: 2026-06-22
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from controller.shunt_integrator import (  # noqa: E402
    MSC,
    MSR,
    SECONDS_PER_DAY,
    ShuntBankConfig,
    ShuntIntegrator,
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
        g_w=2.5,  # step = g_H/(2*g_w) = 0.2*g_H
        delta=5.0,
        t_dwell_s=0.0,
        daily_switch_budget=10,
        y_h_min=0.90,
        y_h_max=1.10,
    )
    kw.update(overrides)
    return ShuntBankConfig(**kw)


# Nominal measurement / sensitivity inputs: one observed HV bus at 1.00 p.u.,
# +0.0005 p.u. per Mvar (so one 50 Mvar step ≈ +0.025 p.u. for an MSC).
_V = np.array([1.00])
_H = np.array([0.0005])


# ---------------------------------------------------------------------------
#  Config fail-fast
# ---------------------------------------------------------------------------

def test_config_rejects_bad_delta():
    # delta must be in (0, q_step/2) = (0, 25)
    with pytest.raises(ValueError):
        _base_cfg(delta=25.0)
    with pytest.raises(ValueError):
        _base_cfg(delta=0.0)


def test_config_rejects_bad_kind_and_q_step():
    with pytest.raises(ValueError):
        _base_cfg(kind="XYZ")
    with pytest.raises(ValueError):
        _base_cfg(q_step_mvar=-1.0)
    with pytest.raises(ValueError):
        _base_cfg(n_levels=0)


def test_config_rejects_bad_band():
    with pytest.raises(ValueError):
        _base_cfg(y_h_min=1.10, y_h_max=0.90)
    with pytest.raises(ValueError):
        _base_cfg(y_h_min=1.05, y_h_max=1.05)


# ---------------------------------------------------------------------------
#  Integrator behaviour: sustained need commits, brief transient does not
# ---------------------------------------------------------------------------

def test_transient_does_not_commit():
    bank = MSC(_base_cfg())
    # One sub-threshold iteration of reactive pressure (under-voltage → MSC
    # gradient is negative → auxiliary state rises).
    c = bank.step(grad_g=-30.0, v_meas=_V, h_v=_H, t_now=0.0)
    assert c is None
    assert 0.0 < bank.q_eq_aux < 30.0
    # Pressure then vanishes (voltage recovered): the integrator must not creep
    # to a commit on its own.
    for i in range(10):
        assert bank.step(grad_g=0.0, v_meas=_V, h_v=_H, t_now=float(i + 1)) is None
    assert bank.level == 0


def test_sustained_need_commits_one_step():
    bank = MSC(_base_cfg())
    commit = None
    for i in range(20):
        c = bank.step(grad_g=-30.0, v_meas=_V, h_v=_H, t_now=float(i))
        if c is not None:
            commit = c
            break
    assert commit is not None
    assert commit.direction == +1
    assert commit.new_level == 1
    assert commit.pp_step_new == 1
    assert bank.level == 1


# ---------------------------------------------------------------------------
#  Dwell time (anti-chatter)
# ---------------------------------------------------------------------------

def test_dwell_blocks_second_commit():
    bank = MSC(_base_cfg(t_dwell_s=600.0))
    # Place the bank one step in, with the auxiliary state already past the
    # next up-threshold, and a recent switch at t = 1000 s.
    bank.level = 1
    bank.q_eq_aux = 100.0
    bank._last_switch_t = 1000.0
    # 100 s after the last switch (< 600 s dwell): blocked.
    assert bank.step(grad_g=-30.0, v_meas=_V, h_v=_H, t_now=1100.0) is None
    assert bank.level == 1
    # 700 s after the last switch (≥ 600 s dwell): allowed.
    c = bank.step(grad_g=-30.0, v_meas=_V, h_v=_H, t_now=1700.0)
    assert c is not None
    assert bank.level == 2


# ---------------------------------------------------------------------------
#  HV feasibility (overshoot) guard
# ---------------------------------------------------------------------------

def test_feasibility_guard_blocks_overshoot():
    bank = MSC(_base_cfg())
    bank.q_eq_aux = 100.0  # well past the up-threshold
    # Observed voltage already near the upper rail: a +1 step would predict
    # 1.08 + 0.0005·50 = 1.105 p.u. > 1.10 → must be blocked.
    v_hi = np.array([1.08])
    assert bank.step(grad_g=-30.0, v_meas=v_hi, h_v=_H, t_now=0.0) is None
    assert bank.level == 0
    # With headroom (v = 1.00) the same proposal commits.
    c = bank.step(grad_g=-30.0, v_meas=_V, h_v=_H, t_now=0.0)
    assert c is not None
    assert bank.level == 1


# ---------------------------------------------------------------------------
#  Daily switching budget
# ---------------------------------------------------------------------------

def test_daily_budget_caps_commits_and_resets_next_day():
    bank = MSC(_base_cfg(daily_switch_budget=1))
    bank.q_eq_aux = 100.0
    c1 = bank.step(grad_g=-30.0, v_meas=_V, h_v=_H, t_now=10.0)
    assert c1 is not None and bank.level == 1
    # Second commit same day → blocked by budget.
    bank.q_eq_aux = 200.0
    c2 = bank.step(grad_g=-30.0, v_meas=_V, h_v=_H, t_now=20.0)
    assert c2 is None and bank.level == 1
    # A day later → window resets, commit allowed again.
    bank.q_eq_aux = 200.0
    c3 = bank.step(grad_g=-30.0, v_meas=_V, h_v=_H, t_now=20.0 + SECONDS_PER_DAY)
    assert c3 is not None and bank.level == 2


# ---------------------------------------------------------------------------
#  Anti-windup clamp
# ---------------------------------------------------------------------------

def test_antiwindup_clamp_bounds_aux():
    bank = MSC(_base_cfg())
    # Hammer with an enormous sustained gradient; the auxiliary state must stay
    # within one physical step of the committed level at all times.
    for i in range(200):
        bank.step(grad_g=-1.0e6, v_meas=_V, h_v=_H, t_now=float(i))
        assert bank.q_eq_aux <= bank.q_eq_phys + bank.config.q_step_mvar + 1e-9
        assert bank.q_eq_aux >= bank.q_eq_phys - bank.config.q_step_mvar - 1e-9
    # And it cannot exceed the top of the lattice.
    assert bank.level == bank.config.n_levels


# ---------------------------------------------------------------------------
#  MSR class
# ---------------------------------------------------------------------------

def test_msr_sign_and_engagement():
    assert MSC(_base_cfg()).expected_h_sign == +1
    cfg = _base_cfg(kind="MSR")
    bank = MSR(cfg)
    assert bank.expected_h_sign == -1
    # Over-voltage → reactor gradient is also negative in the level coordinate
    # (h_v < 0, grad_y > 0 → grad_g < 0), so the reactor engages.  Drive with a
    # negative grad_g and the matching negative voltage sensitivity.
    h_msr = np.array([-0.0005])
    commit = None
    for i in range(20):
        c = bank.step(grad_g=-30.0, v_meas=_V, h_v=h_msr, t_now=float(i))
        if c is not None:
            commit = c
            break
    assert commit is not None and commit.kind == "MSR" and bank.level == 1


def test_make_bank_dispatch():
    assert isinstance(make_bank(_base_cfg(kind="MSC")), MSC)
    assert isinstance(make_bank(_base_cfg(kind="MSR")), MSR)


# ---------------------------------------------------------------------------
#  Container
# ---------------------------------------------------------------------------

def test_integrator_update_length_mismatch_raises():
    integ = ShuntIntegrator.from_configs([_base_cfg(), _base_cfg(kind="MSR")])
    with pytest.raises(ValueError):
        integ.update(grad_g=[-30.0], v_meas=[_V, _V], h_v=[_H, _H], t_now=0.0)


def test_integrator_update_returns_commits():
    integ = ShuntIntegrator.from_configs([_base_cfg()])
    # Pre-load the single bank past threshold so one update commits.
    integ.banks[0].q_eq_aux = 100.0
    commits = integ.update(
        grad_g=[-30.0], v_meas=[_V], h_v=[_H], t_now=0.0,
    )
    assert len(commits) == 1
    assert commits[0].direction == +1
