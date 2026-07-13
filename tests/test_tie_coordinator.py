"""
Unit tests for the horizontal TSO-TSO tie coordinator (gradient-exchange design).

Covers the combined-gradient descent of :class:`HorizontalTieCoordinator`
(``G = κγ_i − (1−κ)γ_j``; ``ΔV_ref ← ΔV_ref − α·G − anchor·DB``; per-zone
worsening cap ``grad_eps``) and the per-side setpoint messages.  Plant-free.

Author: Manuel Schwenke / Claude Code
Date: 2026-06-26
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from controller.tie_coordinator import (  # noqa: E402
    HorizontalTieCoordinator,
    TieCoordinatorConfig,
    TieLink,
    _deadband,
)
from core.message import TieCoordinationMessage  # noqa: E402


def _link(tie_id=10, zi=1, zj=2, bi=100, bj=200, vni=1.03, vnj=1.03):
    return TieLink(
        tie_id=tie_id, zone_i=zi, zone_j=zj, bus_i=bi, bus_j=bj,
        controller_i=f"tso_{zi}", controller_j=f"tso_{zj}",
        v_nom_i=vni, v_nom_j=vnj,
    )


def _coord(links=None, **cfg_kw):
    links = links if links is not None else [_link()]
    return HorizontalTieCoordinator(links, TieCoordinatorConfig(**cfg_kw))


# ── deadband ────────────────────────────────────────────────────────────

def test_deadband_soft_threshold():
    assert _deadband(0.0, 0.002) == 0.0
    assert _deadband(0.001, 0.002) == 0.0
    assert _deadband(0.05, 0.002) == pytest.approx(0.048)
    assert _deadband(-0.05, 0.002) == pytest.approx(-0.048)


# ── initial state ───────────────────────────────────────────────────────

def test_initial_state_zero():
    c = _coord()
    st = c.state()[10]
    assert st["dvref"] == 0.0
    assert st["grad_i"] == 0.0 and st["grad_j"] == 0.0 and st["grad_combined"] == 0.0
    assert c.iteration == 0


def test_duplicate_tie_id_rejected():
    with pytest.raises(ValueError):
        _coord([_link(tie_id=5), _link(tie_id=5, zi=2, zj=3)])


def test_v_anchor_is_schedule_midpoint():
    assert _link(vni=1.05, vnj=1.01).v_anchor == pytest.approx(1.03)


# ── descent ─────────────────────────────────────────────────────────────

def test_descent_moves_opposite_combined_gradient():
    # G = κγ_i − (1−κ)γ_j = 0.5·2 − 0.5·0 = 1.0 ; Δ = −α·G = −0.1
    c = _coord(grad_alpha=0.1, grad_eps=1e9, anchor=0.0, kappa=0.5, dvref_max=10.0)
    c.update({10: (2.0, 0.0)})
    st = c.state()[10]
    assert st["grad_combined"] == pytest.approx(1.0)
    assert st["grad_i"] == 2.0 and st["grad_j"] == 0.0
    assert c.dvref[10] == pytest.approx(-0.1)


def test_safeguard_caps_per_zone_worsening():
    # γ_i=1, γ_j=10 -> G=0.5−5=−4.5 ; Δ_raw=+4.5 ; dJ_i=1·0.5·4.5=2.25 > eps=1
    # -> Δ scaled to 4.5·(1/2.25)=2.0
    c = _coord(grad_alpha=1.0, grad_eps=1.0, anchor=0.0, kappa=0.5, dvref_max=10.0)
    c.update({10: (1.0, 10.0)})
    assert c.state()[10]["grad_combined"] == pytest.approx(-4.5)
    assert c.dvref[10] == pytest.approx(2.0)          # capped move
    # the capped move yields exactly the allowed worsening for zone i
    assert 1.0 * 0.5 * c.dvref[10] == pytest.approx(1.0)


def test_anchor_pulls_toward_zero_when_gradients_indifferent():
    c = _coord(grad_alpha=1e-12, grad_eps=1e9, anchor=0.5, deadband_v_pu=0.0,
               kappa=0.5, dvref_max=10.0)
    c.dvref[10] = 0.1
    c.update({10: (0.0, 0.0)})                         # G≈0 -> only the anchor acts
    assert c.dvref[10] == pytest.approx(0.05)          # 0.1 − 0.5·DB(0.1)=0.1−0.05


def test_dvref_clipped():
    c = _coord(grad_alpha=1.0, grad_eps=1e9, anchor=0.0, dvref_max=0.03)
    c.update({10: (-10.0, 0.0)})                       # Δ=+5 -> clipped to dvref_max
    assert c.dvref[10] == pytest.approx(0.03)


def test_unknown_tie_id_rejected():
    c = _coord()
    with pytest.raises(KeyError):
        c.update({999: (1.0, 1.0)})


# ── messages ────────────────────────────────────────────────────────────

def test_messages_split_difference_no_price():
    c = _coord(grad_alpha=1.0, grad_eps=1e9, anchor=0.0, kappa=0.5,
               links=[_link(vni=1.03, vnj=1.03)])
    c.update({10: (-0.04, 0.0)})   # G=−0.02 ; Δ=+0.02 -> dvref=0.02
    assert c.dvref[10] == pytest.approx(0.02)
    msgs = {m.target_controller_id: m for m in c.generate_messages()}
    mi, mj = msgs["tso_1"], msgs["tso_2"]
    assert float(mi.v_ref_pu[0]) == pytest.approx(1.03 + 0.5 * 0.02)
    assert float(mj.v_ref_pu[0]) == pytest.approx(1.03 - 0.5 * 0.02)
    assert float(mi.v_ref_pu[0] - mj.v_ref_pu[0]) == pytest.approx(0.02)
    assert not hasattr(mi, "price")


def test_messages_aggregate_multiple_ties_per_zone():
    links = [_link(tie_id=10, zi=1, zj=2, bi=100, bj=200),
             _link(tie_id=11, zi=2, zj=3, bi=201, bj=300)]
    c = _coord(links)
    c.update({10: (0.0, 0.0), 11: (0.0, 0.0)})
    msgs = {m.target_controller_id: m for m in c.generate_messages()}
    assert msgs["tso_1"].n_ties == 1 and msgs["tso_3"].n_ties == 1
    assert msgs["tso_2"].n_ties == 2
    assert set(int(b) for b in msgs["tso_2"].boundary_bus_indices) == {200, 201}


# ── validation ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw", [
    {"grad_alpha": 0.0}, {"grad_eps": -1.0}, {"anchor": -1.0},
    {"deadband_v_pu": -1e-3}, {"kappa": 1.5}, {"dvref_max": 0.0},
])
def test_config_validation(kw):
    with pytest.raises(ValueError):
        TieCoordinatorConfig(**kw)


def test_message_length_mismatch_rejected():
    with pytest.raises(ValueError):
        TieCoordinationMessage(
            source_controller_id="tie_coordinator", target_controller_id="tso_1",
            iteration=0,
            tie_line_indices=np.array([10, 11], dtype=np.int64),
            boundary_bus_indices=np.array([100], dtype=np.int64),  # too short
            v_ref_pu=np.array([1.0, 1.0], dtype=np.float64),
        )
