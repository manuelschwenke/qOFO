"""
Guards for the 2026-08-18 Stage-1 re-run.

Two independent traps were found while preparing it, both of the same shape as
the ones ``bank_fingerprint`` and ``stage0_fingerprint`` already document:
something that changes what a stored number *means*, without changing anything
the cache validates against.

1. **Objective change vs. the evaluation cache.**  Adding the ``guard`` DS
   criterion touched only ``tuning_mc/metrics.py`` and ``tuning/metrics.py``.
   The cache is keyed on ``(scenario_set, knob hash)`` and validated against the
   scenario bank and the design rule -- neither notices.  All 110 archived
   tier-1 results would have been replayed with their old ``f_ds`` mixed into a
   new front.

2. **A fixed voltage relief vs. a searched ``dso_g_v``.**  ``dso_g_v_ratio`` is
   a search coordinate, so ``dso_g_v`` moves per trial.  Writing the per-area
   relief as absolute numbers would let the OLTC loop gain
   ``dso_g_v / g_w_dso_oltc`` drift as the search walked -- and that ratio is
   exactly what keeps the integer tap out of a limit cycle (measured: 50.5
   reversals/h when it is broken, against 0.00 at baseline).

See ``docs/daily_log/08_2026/2026-08-18_dso4_voltage_relief.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from configs.config import apply_dso_v_relief
from tuning._io import load_config_yaml
from tuning_mc.stage_1_search import (
    BOUNDS,
    DEFAULT_BASELINE,
    DSO_V_RELIEF_AREAS,
    DSO_V_RELIEF_FACTORS,
    X0,
    bank_fingerprint,
    build_config,
    scoring_fingerprint,
)


# ── 1. scoring fingerprint ──────────────────────────────────────────────────

def test_criterion_changes_the_scoring_fingerprint():
    assert scoring_fingerprint("v_rms") != scoring_fingerprint("guard")


def test_scoring_fingerprint_is_stable_for_one_criterion():
    assert scoring_fingerprint("guard") == scoring_fingerprint("guard")


def test_scoring_fingerprint_is_independent_of_the_bank_stamp():
    """They must be separate stamps, or one change masks the other.

    The metric edit that motivated this left ``bank_fingerprint`` untouched --
    that is precisely why a second stamp was needed.
    """
    assert scoring_fingerprint("guard") != bank_fingerprint()


def test_unstamped_cache_entries_cannot_match():
    """Rows written before the stamp existed must be treated as stale.

    ``bank_ok`` deliberately accepts ``None`` (legacy rows predate that stamp
    too), so the scoring check must NOT copy that leniency -- an unstamped row
    is exactly the pre-change row whose ``f_ds`` is unusable.
    """
    legacy_row = {"f_ts": 1.0, "f_q": 1.0, "f_ds": 1.0}
    assert legacy_row.get("_scoring_fingerprint") != scoring_fingerprint("guard")
    assert legacy_row.get("_scoring_fingerprint") != scoring_fingerprint("v_rms")


# ── 2. relief under a moving dso_g_v ────────────────────────────────────────

@pytest.fixture(scope="module")
def baseline():
    if not Path(DEFAULT_BASELINE).exists():
        pytest.skip(f"campaign baseline not present: {DEFAULT_BASELINE}")
    return load_config_yaml(Path(DEFAULT_BASELINE))


def test_campaign_default_reproduces_earlier_campaigns(baseline):
    """An empty mapping is a no-op, so pre-2026-08-18 runs are unchanged."""
    assert DSO_V_RELIEF_FACTORS == {}
    cfg = build_config({"dso_g_v_ratio": 1.0}, {}, baseline)
    assert not cfg.dso_g_v_per_area
    assert not cfg.dso_g_w_class


@pytest.mark.parametrize("ratio", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_relief_holds_the_loop_gain_across_the_search_range(baseline, ratio):
    """The invariant must survive every value ``dso_g_v_ratio`` can take."""
    relief = {"DSO_2": 20.0, "DSO_4": 20.0}
    cfg = build_config({"dso_g_v_ratio": ratio}, {"g_w_dso_oltc": 200.0},
                       baseline, dso_v_relief=relief)
    for dso_id, factor in relief.items():
        r_gv = cfg.dso_g_v_per_area[dso_id] / cfg.dso_g_v
        r_gw = cfg.dso_g_w_class[dso_id]["dso_oltc"] / cfg.g_w_dso_oltc
        assert r_gv == pytest.approx(factor)
        assert r_gw == pytest.approx(factor)
        assert r_gv / r_gw == pytest.approx(1.0), (
            f"{dso_id}: OLTC loop gain moved by {r_gv / r_gw:.4f}x at "
            f"dso_g_v_ratio={ratio}; the integer tap will limit-cycle"
        )


def test_relief_derives_from_the_searched_value_not_the_baseline(baseline):
    """Doubling the knob must double the area's absolute authority."""
    relief = {"DSO_4": 20.0}
    lo = build_config({"dso_g_v_ratio": 1.0}, {}, baseline, dso_v_relief=relief)
    hi = build_config({"dso_g_v_ratio": 2.0}, {}, baseline, dso_v_relief=relief)
    assert hi.dso_g_v == pytest.approx(2.0 * lo.dso_g_v)
    assert (hi.dso_g_v_per_area["DSO_4"]
            == pytest.approx(2.0 * lo.dso_g_v_per_area["DSO_4"]))


def test_relief_is_applied_after_the_designed_oltc_weight(baseline):
    """The factor must multiply Stage 0's designed weight, not the baseline's."""
    designed = 977.0
    cfg = build_config({"dso_g_v_ratio": 1.0}, {"g_w_dso_oltc": designed},
                       baseline, dso_v_relief={"DSO_4": 20.0})
    assert cfg.g_w_dso_oltc == pytest.approx(designed)
    assert cfg.dso_g_w_class["DSO_4"]["dso_oltc"] == pytest.approx(designed * 20.0)


def test_relief_leaves_unlisted_areas_alone(baseline):
    cfg = build_config({"dso_g_v_ratio": 1.0}, {}, baseline,
                       dso_v_relief={"DSO_4": 20.0})
    assert set(cfg.dso_g_v_per_area) == {"DSO_4"}


def test_helper_rejects_a_nonpositive_factor(baseline):
    with pytest.raises(ValueError, match="must be > 0"):
        apply_dso_v_relief(baseline, {"DSO_4": 0.0})


# ── 3. the searched dso_v_authority coordinate (option B) ───────────────────

def test_authority_is_gated_not_a_default_coordinate():
    """Addressable but not searched unless asked for.

    Same gating as ``lambda_tso_z*``: in BOUNDS so ``--x0`` and the promotion
    flag can reach it, absent from X0 so every earlier campaign reproduces.
    Phase B is a compass search -- an extra live direction costs two
    evaluations on every poll, so this must be opt-in.
    """
    assert "dso_v_authority" in BOUNDS
    assert "dso_v_authority" not in X0


def test_authority_lower_bound_is_the_unrelieved_plant():
    """The incumbent must always be able to walk back to 'no relief'."""
    lo, hi = BOUNDS["dso_v_authority"]
    assert lo == 1.0
    assert hi > 20.0          # the measured operating point must be interior


@pytest.mark.parametrize("auth", [1.0, 5.0, 20.0, 100.0])
@pytest.mark.parametrize("ratio", [0.5, 2.0])
def test_authority_knob_holds_the_loop_gain(baseline, auth, ratio):
    """The searched factor must not drift the OLTC loop gain either.

    Two coordinates move here at once -- ``dso_g_v_ratio`` and
    ``dso_v_authority`` -- which is exactly the combination a compass poll
    produces, and exactly where an absolute relief would break.
    """
    cfg = build_config({"dso_g_v_ratio": ratio, "dso_v_authority": auth},
                       {"g_w_dso_oltc": 200.0}, baseline)
    if auth == 1.0:
        assert not cfg.dso_g_v_per_area      # 1.0 == no relief
        return
    for dso_id in DSO_V_RELIEF_AREAS:
        r_gv = cfg.dso_g_v_per_area[dso_id] / cfg.dso_g_v
        r_gw = cfg.dso_g_w_class[dso_id]["dso_oltc"] / cfg.g_w_dso_oltc
        assert r_gv == pytest.approx(auth)
        assert r_gv / r_gw == pytest.approx(1.0)


def test_authority_knob_overrides_the_fixed_mapping(baseline):
    """A searched value must win over DSO_V_RELIEF_FACTORS, not merge with it."""
    cfg = build_config({"dso_g_v_ratio": 1.0, "dso_v_authority": 7.0}, {},
                       baseline, dso_v_relief={"DSO_4": 20.0})
    assert set(cfg.dso_g_v_per_area) == set(DSO_V_RELIEF_AREAS)
    for dso_id in DSO_V_RELIEF_AREAS:
        assert (cfg.dso_g_v_per_area[dso_id]
                == pytest.approx(cfg.dso_g_v * 7.0))


def test_authority_acts_only_on_the_spread_limited_areas():
    """DSO_1 and DSO_3 must stay out -- they measured no gain and a real cost."""
    assert set(DSO_V_RELIEF_AREAS) == {"DSO_2", "DSO_4"}
