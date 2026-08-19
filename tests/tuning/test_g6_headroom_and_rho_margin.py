"""
The two barriers added 2026-08-19, and the backwards compatibility they must keep.

Both exist because a safeguard was applied at one stage and left unenforced at
the next, and phase B walked straight through the gap:

* **g6_ds_headroom.** ``guard_deficit_ds_pu`` entered the search as a *filter*
  criterion. A candidate that improves ``f_ts`` and ``f_q`` while degrading
  ``f_ds`` is non-dominated, so it is accepted -- and phase B duly sold DSO
  voltage margin until a bus left [0.90, 1.10] (worst headroom +0.0200 ->
  -0.0003 pu, 0/12 -> 1/12 windows outside the corridor). A filter criterion
  *prices* margin; only a barrier *forbids selling* it.
* **rho_margin on the barrier.** ``--rho-margin 0.031`` shrinks the target the
  lambda* CALIBRATION selects against and leaves the declared ceiling alone, so
  g3 kept comparing against 1.5. Phase B walked ``lambda_tso`` 0.15 -> 0.2518,
  reaching rho = 1.4771: inside the declared ceiling, outside the margined
  1.4549 -- i.e. it spent exactly the transfer allowance the calibration had
  set aside.

See ``docs/daily_log/08_2026/2026-08-18_dso4_voltage_relief.md`` sections 10-11.
"""

from __future__ import annotations

import pytest

from tuning.metrics import TrajectoryMetrics
from tuning.objectives_v2 import (
    CONSTRAINT_NAMES,
    ConstraintLimits,
    feasibility_constraints,
)


class _Res:
    def __init__(self, m: TrajectoryMetrics) -> None:
        self.metrics, self.scenario_name, self.failure_reason = m, "w", ""


class _Cfg:
    v_min_pu, v_max_pu, v_setpoint_pu = 0.90, 1.10, 1.03


def _m(rho: float = 1.0, head: float = 0.05) -> TrajectoryMetrics:
    return TrajectoryMetrics(
        rho_emp_p95=rho, ds_headroom_min_pu=head, n_records=1080,
        voltage_excess_pu=0.0, tap_ops_per_h_tso=1.0, tap_ops_per_h_dso=1.0,
        tap_reversals_per_h_tso=0.1, tap_reversals_per_h_dso=0.1,
    )


def _g(limits, *metrics):
    v = feasibility_constraints([_Res(m) for m in metrics], _Cfg(), limits)
    return dict(zip(CONSTRAINT_NAMES, v))


# ── backwards compatibility ────────────────────────────────────────────────

def test_g6_is_appended_so_existing_indices_do_not_shift():
    """Readers that index the vector positionally must keep working."""
    assert CONSTRAINT_NAMES[:6] == (
        "g1_diverged", "g2_corridor", "g3_contraction",
        "g4_settling", "g5a_tap_ops", "g5b_tap_reversals",
    )
    assert CONSTRAINT_NAMES[6] == "g6_ds_headroom"


def test_defaults_reproduce_the_pre_2026_08_19_behaviour():
    """Both additions are inert at their defaults, or every prior study moves."""
    d = ConstraintLimits()
    assert d.rho_margin == 0.0
    assert d.ds_headroom_pu is None
    g = _g(d, _m(rho=1.4771, head=-0.0003))
    assert g["g3_contraction"] == pytest.approx(1.4771 - d.rho_emp_p95)
    assert g["g6_ds_headroom"] == -1.0          # disabled, never binding


def test_absent_headroom_metric_is_not_a_violation():
    """A metric that predates the field must not fail the trial.

    Same rule g4 already applies: no evidence is not evidence of a violation,
    otherwise every old row fails for a plumbing reason.
    """
    g = _g(ConstraintLimits(ds_headroom_pu=0.01),
           _m(head=float("nan")))
    assert g["g6_ds_headroom"] == -1.0


# ── g6 does the job the filter could not ───────────────────────────────────

def test_g6_rejects_the_incumbent_that_sold_the_margin():
    """The measured phase-B incumbent must now be infeasible."""
    g = _g(ConstraintLimits(ds_headroom_pu=0.01), _m(head=-0.0003))
    assert g["g6_ds_headroom"] > 0


def test_g6_accepts_the_headroom_respecting_selection():
    g = _g(ConstraintLimits(ds_headroom_pu=0.01), _m(head=+0.0126))
    assert g["g6_ds_headroom"] < 0


def test_g6_takes_the_worst_scenario_not_the_mean():
    """One bad window must fail the candidate; the constraint is a max."""
    g = _g(ConstraintLimits(ds_headroom_pu=0.01),
           _m(head=+0.05), _m(head=+0.05), _m(head=-0.001))
    assert g["g6_ds_headroom"] == pytest.approx(0.01 - (-0.001))
    assert g["g6_ds_headroom"] > 0


@pytest.mark.parametrize("req", [0.005, 0.010, 0.015, 0.020])
def test_g6_is_exactly_the_shortfall(req):
    g = _g(ConstraintLimits(ds_headroom_pu=req), _m(head=0.012))
    assert g["g6_ds_headroom"] == pytest.approx(req - 0.012)


# ── rho margin now binds the search, not only the calibration ──────────────

def test_rho_margin_tightens_the_barrier():
    """rho = 1.4771 is inside 1.5 but outside 1.5/1.031 = 1.4549."""
    loose = _g(ConstraintLimits(rho_emp_p95=1.5), _m(rho=1.4771))
    tight = _g(ConstraintLimits(rho_emp_p95=1.5, rho_margin=0.031), _m(rho=1.4771))
    assert loose["g3_contraction"] < 0          # the gap phase B walked through
    assert tight["g3_contraction"] > 0


def test_rho_margin_matches_the_calibration_formula():
    """Same ``target / (1 + margin)`` the lambda* selection uses."""
    lim = ConstraintLimits(rho_emp_p95=1.5, rho_margin=0.031)
    g = _g(lim, _m(rho=1.40))
    assert g["g3_contraction"] == pytest.approx(1.40 - 1.5 / 1.031)


def test_lambda_star_point_survives_the_tightened_barrier():
    """lambda* = 0.15 measured rho = 1.3256 -- it must remain feasible."""
    lim = ConstraintLimits(rho_emp_p95=1.5, rho_margin=0.031, ds_headroom_pu=0.01)
    g = _g(lim, _m(rho=1.3256, head=+0.0200))
    assert g["g3_contraction"] < 0
    assert g["g6_ds_headroom"] < 0
