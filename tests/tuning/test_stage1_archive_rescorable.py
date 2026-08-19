"""
The Stage-1 per-scenario archive must be re-scorable offline.

Why this test exists
--------------------
On 2026-08-18 a DS-voltage criterion (``guard_deficit_ds_pu``, "is the
subordinate network riding its bound?") was added to the filter.  It could not
be applied to any of the 110 archived tier-1 trials: ``stage_1_search`` receives
``(res, records)`` per scenario but persists only ``score.per_scenario``, and
that dict carried ``f_ts / f_q / f_ds / tap stats / rho`` and no DS voltage
envelope at all.  Re-scoring would have cost minutes; re-simulating the bank
costs ~9 h wall.

The fix is to persist the whole flat ``TrajectoryMetrics`` under
``per_scenario[name]["metrics"]``.  This test locks that, so the next criterion
change is a re-score and not a re-run.

It deliberately asserts on *coverage of the metric vector*, not on a hand-listed
set of field names: a list would have to be extended by whoever adds the next
metric, which is exactly the failure mode being prevented.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from tuning.metrics import TrajectoryMetrics
from tuning_mc.metrics import score_candidate


class _Res:
    """Minimal stand-in for ``tuning.runner.RunResult``."""

    def __init__(self, name: str, metrics: TrajectoryMetrics) -> None:
        self.scenario_name = name
        self.metrics = metrics
        self.wall_time_s = 0.0
        self.failure_reason = ""


def _metrics(**over) -> TrajectoryMetrics:
    base = dict(
        v_rms_ts=0.006, v_worst_ts=0.04, v_band_excess_ts=0.005,
        v_rms_ds=0.005, v_worst_ds=0.07, v_band_excess_ds=0.02,
        guard_deficit_ds_pu=0.0223, ds_headroom_min_pu=-0.0012,
        voltage_excess_pu=0.04, itae_q_pcc=3.0,
        tap_ops_per_h_tso=1.3, tap_ops_per_h_dso=2.3,
        tap_reversals_per_h_tso=0.17, tap_reversals_per_h_dso=0.0,
        rho_emp_p95=0.5, n_records=1080, duration_s=21600.0,
    )
    base.update(over)
    return TrajectoryMetrics(**base)


@pytest.fixture()
def scored():
    class _Cfg:
        v_min_pu, v_max_pu = 0.90, 1.10
        v_setpoint_pu = 1.03
    results = [_Res("win_a", _metrics()),
               _Res("win_b", _metrics(guard_deficit_ds_pu=0.0, v_worst_ds=0.03))]
    return score_candidate(results, _Cfg()), results


def test_archive_carries_the_whole_metric_vector(scored):
    """Every TrajectoryMetrics field survives into per_scenario[...]['metrics'].

    Coverage-based on purpose -- see the module docstring.
    """
    score, results = scored
    expected = {f.name for f in dataclasses.fields(TrajectoryMetrics)}
    for res in results:
        stored = score.per_scenario[res.scenario_name]["metrics"]
        missing = expected - set(stored)
        assert not missing, (
            f"{sorted(missing)} dropped from the Stage-1 archive; a criterion "
            f"built on them could not be re-scored without re-simulating"
        )


def test_ds_voltage_is_recoverable_from_the_archive(scored):
    """The specific fields the 2026-08-18 re-run needed are present and exact."""
    score, results = scored
    for res in results:
        stored = score.per_scenario[res.scenario_name]["metrics"]
        for field in ("guard_deficit_ds_pu", "ds_headroom_min_pu",
                      "v_worst_ds", "v_band_excess_ds", "voltage_excess_pu"):
            assert stored[field] == pytest.approx(getattr(res.metrics, field))


def test_a_ds_criterion_can_be_recomputed_offline(scored):
    """Re-scoring from the archive reproduces scoring from the live objects.

    This is the property that makes a criterion change cheap: swapping ``f_ds``
    for a different DS metric must be doable from the JSON alone.
    """
    score, results = scored
    # "Re-score" offline: pick a different DS criterion straight out of the
    # archive, exactly as an offline pass over the eval JSONs would.
    offline = [score.per_scenario[r.scenario_name]["metrics"]["guard_deficit_ds_pu"]
               for r in results]
    live = [r.metrics.guard_deficit_ds_pu for r in results]
    assert offline == pytest.approx(live)


def test_existing_keys_are_untouched(scored):
    """The addition is purely additive -- existing readers must not break."""
    score, _ = scored
    legacy = {"f_ts", "f_q", "f_ds", "f_total", "tap_ops_per_h_tso",
              "tap_ops_per_h_dso", "tap_reversals_per_h_tso",
              "tap_reversals_per_h_dso", "rho_emp_p95", "feasible"}
    for entry in score.per_scenario.values():
        assert legacy <= set(entry)


def test_archive_survives_a_json_round_trip(scored):
    """What is asserted above must actually reach disk.

    ``stage_1_search`` writes the payload with ``json.dumps(..., indent=1)``
    (default ``allow_nan=True``) and reads it back with ``json.loads``, so
    non-finite entries round-trip as NaN rather than raising.
    """
    score, results = scored
    payload = json.loads(json.dumps(score.as_dict()))
    for res in results:
        stored = payload["per_scenario"][res.scenario_name]["metrics"]
        assert stored["guard_deficit_ds_pu"] == pytest.approx(
            res.metrics.guard_deficit_ds_pu)

    # and a NaN-carrying metric vector must not break the write
    nan_score = score_candidate(
        [_Res("win_nan", _metrics(ds_headroom_min_pu=float("nan")))],
        type("C", (), {"v_min_pu": 0.9, "v_max_pu": 1.1,
                       "v_setpoint_pu": 1.03})(),
    )
    rt = json.loads(json.dumps(nan_score.as_dict()))
    assert math.isnan(rt["per_scenario"]["win_nan"]["metrics"]["ds_headroom_min_pu"])
