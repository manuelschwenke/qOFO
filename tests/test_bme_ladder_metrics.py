# -*- coding: utf-8 -*-
"""
Unit tests for the 011_BME_LADDER derived-metrics module (spec §6,
Phase 6 item 5): AR(2) oscillation indicator, gap-to-oracle columns,
Phulpin normalised overcost, D2 band-violation fix.

Symbol map (spec §3): Φ = common objective [MW]; Φ_i = zone share
(D1 ownership); B = boundary registry; the AR(2) pole is the §6
oscillation indicator on boundary voltages.
"""
from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ladder = importlib.import_module("experiments.011_BME_LADDER")


# ── AR(2) dominant pole ────────────────────────────────────────────────

def test_ar2_damped_cosine_recovers_pole():
    rho, omega = 0.97, 0.3
    k = np.arange(600, dtype=float)
    y = rho ** k * np.cos(omega * k)
    mod, cx, cyc = ladder._ar2_dominant_pole(y)
    assert cx, "damped cosine must yield a complex pair"
    # Finite-sample Yule-Walker bias on a decaying record is real —
    # tolerance is loose but excludes the real-pole failure mode.
    assert mod == pytest.approx(rho, abs=0.05)
    assert cyc == pytest.approx(omega / (2 * np.pi), rel=0.15)


def test_ar2_sustained_oscillation_pole_near_unit_circle():
    k = np.arange(400, dtype=float)
    y = np.cos(0.25 * k)
    mod, cx, _ = ladder._ar2_dominant_pole(y)
    assert cx
    assert mod == pytest.approx(1.0, abs=0.02)


def test_ar2_constant_and_ramp_are_not_oscillations():
    mod_c, cx_c, _ = ladder._ar2_dominant_pole(np.full(200, 1.03))
    assert (mod_c, cx_c) == (0.0, False)
    # Linear drift (load ramp) must NOT masquerade as a pole at +1.
    mod_r, cx_r, _ = ladder._ar2_dominant_pole(np.linspace(0.98, 1.06, 200))
    assert not cx_r
    assert mod_r < 0.5


def test_ar2_white_noise_small_modulus():
    rng = np.random.default_rng(7)
    mod, _, _ = ladder._ar2_dominant_pole(rng.normal(size=1000))
    assert mod < 0.5


def test_ar2_short_series_nan():
    mod, cx, cyc = ladder._ar2_dominant_pole(np.ones(5))
    assert np.isnan(mod) and not cx and np.isnan(cyc)


# ── Synthetic ladder records ───────────────────────────────────────────

def _rec(minute, phi, phi_zones, losses, v_boundary=None, tie_q=None):
    return SimpleNamespace(
        time_s=60.0 * minute,
        bme_phi_mw=phi,
        bme_phi_zone_mw=dict(phi_zones),
        total_losses_mw=losses,
        zone_v_min={1: 1.02, 2: 1.02, 3: 1.02},
        zone_v_max={1: 1.04, 2: 1.04, 3: 1.04},
        zone_oltc_taps={1: np.zeros(2)},
        zone_tie_q_mvar=dict(tie_q or {}),
        bme_v_boundary=dict(v_boundary or {}),
    )


def _flat_rung(n, phi, phi_zones, losses, osc_bus=None, osc_rho=1.0):
    recs = []
    for i in range(n):
        vb = {}
        if osc_bus is not None:
            vb = {osc_bus: 1.03 + 1e-3 * osc_rho ** i * np.cos(0.4 * i),
                  99: 1.03}
        recs.append(_rec(i, phi, phi_zones, losses, v_boundary=vb,
                         tie_q={(1, 2): 5.0}))
    return recs


@pytest.fixture()
def ladder_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder, "RESULT_DIR", tmp_path)
    return tmp_path


def test_derived_metrics_gap_and_fairness(ladder_dir):
    zones_none = {1: 20.0, 2: 12.0, 3: 8.0}      # Σ = 40
    zones_bme = {1: 19.0, 2: 11.0, 3: 6.0}       # Σ = 36
    zones_orc = {1: 18.0, 2: 11.0, 3: 6.0}       # Σ = 35
    data = {
        "none": _flat_rung(180, 40.0, zones_none, 47.0),
        "bme": _flat_rung(180, 36.0, zones_bme, 45.0),
        "oracle": _flat_rung(180, 35.0, zones_orc, 44.5),
    }
    rows = {r["rung"]: r for r in ladder.compute_derived_metrics(data)}

    bme = rows["bme"]
    assert bme["gap_oracle_terminal_pct"] == pytest.approx(
        100 * (36 / 35 - 1), abs=1e-3)
    assert bme["gap_oracle_integral_pct"] == pytest.approx(
        100 * (36 / 35 - 1), abs=1e-3)
    assert bme["gap_closure_pct"] == pytest.approx(80.0, abs=0.1)
    assert "gap_oracle_terminal_pct" not in rows["oracle"]
    assert "gap_closure_pct" not in rows["none"]

    # Phulpin normalised overcost: 100·(Φ_i − Φ_i^none)/|Φ_i^none|.
    assert bme["overcost_z1_pct"] == pytest.approx(-5.0)
    assert bme["overcost_z2_pct"] == pytest.approx(-100 / 12, abs=0.01)
    assert bme["overcost_z3_pct"] == pytest.approx(-25.0)
    assert bme["overcost_max_pct"] == pytest.approx(-5.0)
    # The oracle's 3-area metric partition makes it fairness-comparable.
    assert rows["oracle"]["overcost_max_pct"] == pytest.approx(
        -100 / 12, abs=0.01)
    assert "overcost_max_pct" not in rows["none"]

    # loss reduction vs none
    assert bme["loss_red_vs_none_pct"] == pytest.approx(
        100 * (1 - 45.0 / 47.0), abs=0.01)

    # CSV written with one row per rung
    with open(ladder_dir / "metrics_derived.csv", newline="") as f:
        assert {r["rung"] for r in csv.DictReader(f)} == set(data)


def test_oscillation_indicator_boundary_v_and_proxy(ladder_dir):
    zones = {1: 20.0, 2: 12.0, 3: 8.0}
    # Sustained boundary-voltage oscillation at bus 8 → pole ≈ 1.
    sustained = _flat_rung(180, 40.0, zones, 47.0, osc_bus=8, osc_rho=1.0)
    osc = ladder._oscillation_indicator(sustained)
    assert osc["osc_signal"] == "boundary_v:bus8"
    assert osc["osc_complex_pair"]
    assert osc["osc_pole_mod"] == pytest.approx(1.0, abs=0.03)
    # period = dt / (ω/2π) = 1 min · 2π/0.4 ≈ 15.7 min
    assert osc["osc_period_min"] == pytest.approx(2 * np.pi / 0.4, abs=1.0)

    # Damped oscillation → strictly smaller modulus.
    damped = _flat_rung(180, 40.0, zones, 47.0, osc_bus=8, osc_rho=0.9)
    assert (ladder._oscillation_indicator(damped)["osc_pole_mod"]
            < osc["osc_pole_mod"])

    # Pickles predating bme_v_boundary fall back to the tie-Q proxy.
    legacy = _flat_rung(180, 40.0, zones, 47.0)
    for r in legacy:
        del r.bme_v_boundary
    assert ladder._oscillation_indicator(legacy)["osc_signal"].startswith(
        "tie_q_proxy:tie1-2")


def test_band_violation_uses_d2_edges():
    zones = {1: 20.0, 2: 12.0, 3: 8.0}
    recs = _flat_rung(10, 40.0, zones, 47.0)
    # All voltages inside the D2 band (1.01, 1.05) → zero violation;
    # the pre-fix default band (0.97, 1.03) would report 1.0 here.
    assert ladder._band_violation_fraction(recs) == 0.0
