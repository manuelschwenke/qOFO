"""
Character test for the gradient-exchange tie coordinator.

Pins the claim:

  "The gradient-exchange coordinator is a BOUNDED JOINT voltage-profile
   optimiser, not a preferential ancillary-rescue mechanism.  Its ancillary
   service is cheap, mutually-beneficial, bounded boundary-voltage support."

Each clause maps to a test.  These are plant-free and deterministic: the claim
is a property of the coordinator's *update law* (controller/tie_coordinator.py),
which is exactly what is asserted here.  (The plant-level evidence — gen-trip vs.
compound-stress on IEEE-39 — lives in the 007 case study; this fixes the
mechanism so the operational reading cannot silently drift.)

The coordinator acts on ONE variable per tie — the agreed boundary-voltage
difference ΔV_ref — via
    G      = κ·γ_i − (1−κ)·γ_j                 # gradient of the JOINT objective
    Δ      = −grad_alpha·G − anchor·db(ΔV_ref)  # joint descent + subsidiarity
    cap:   dJ_i ≈ γ_i·κ·Δ ,  dJ_j ≈ −γ_j·(1−κ)·Δ ; shrink Δ so max(dJ_i,dJ_j) ≤ grad_eps
    ΔV_ref = clip(ΔV_ref + Δ, ±dvref_max)
so γ is a *boundary-voltage* marginal and ΔV_ref is a *boundary voltage* — the
"voltage-profile / boundary-voltage support" part of the claim is structural.

Run:  pytest tests/test_tie_coordinator_character.py -v
  or:  python tests/test_tie_coordinator_character.py   (prints the evidence table)

Author: Manuel Schwenke / Claude Code
Date: 2026-06-29
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from controller.tie_coordinator import (  # noqa: E402
    HorizontalTieCoordinator,
    TieCoordinatorConfig,
    TieLink,
)


def _link(tie_id: int = 10):
    return TieLink(
        tie_id=tie_id, zone_i=1, zone_j=2, bus_i=100, bus_j=200,
        controller_i="tso_1", controller_j="tso_2", v_nom_i=1.03, v_nom_j=1.03,
    )


def _coord(**cfg):
    return HorizontalTieCoordinator([_link()], TieCoordinatorConfig(**cfg))


def _dJ(g_i, g_j, kappa, delta):
    """Predicted per-zone objective change for a realised move ``delta`` (the
    same first-order model the coordinator's safeguard uses)."""
    return g_i * kappa * delta, -g_j * (1.0 - kappa) * delta


# ── Clause 1: BOUNDED ─────────────────────────────────────────────────────

class TestBounded:
    def test_dvref_never_exceeds_clip(self):
        """|ΔV_ref| ≤ dvref_max under a sustained one-sided push (the spread
        cannot run away — it is a *bounded* setpoint)."""
        c = _coord(grad_alpha=1.0, grad_eps=1e9, anchor=0.0, dvref_max=0.06)
        for _ in range(200):
            c.update({10: (-50.0, 0.0)})
            assert abs(c.dvref[10]) <= 0.06 + 1e-12

    def test_per_zone_worsening_capped_at_grad_eps(self):
        """A conflicting move pushes NEITHER zone uphill by more than grad_eps
        — the 'help only if it is cheap for me' bound."""
        kappa, eps = 0.5, 1.0
        c = _coord(grad_alpha=1.0, grad_eps=eps, anchor=0.0, kappa=kappa,
                   dvref_max=1e9)
        c.update({10: (-1.0, -40.0)})          # both want boundary up -> conflict
        dJ_i, dJ_j = _dJ(-1.0, -40.0, kappa, c.dvref[10])
        assert max(dJ_i, dJ_j) <= eps + 1e-9


# ── Clause 2: JOINT & SYMMETRIC (not preferential) ────────────────────────

class TestJointAndSymmetric:
    def test_descends_the_combined_gradient(self):
        """The move is −grad_alpha·G with G = κγ_i − (1−κ)γ_j — the gradient of
        the *joint* objective, not of either zone alone."""
        kappa = 0.5
        c = _coord(grad_alpha=0.01, grad_eps=1e9, anchor=0.0, kappa=kappa,
                   dvref_max=1e9)
        g_i, g_j = 2.0, -3.0
        c.update({10: (g_i, g_j)})
        G = kappa * g_i - (1.0 - kappa) * g_j
        assert c.state()[10]["grad_combined"] == pytest.approx(G)
        assert c.dvref[10] == pytest.approx(-0.01 * G)

    def test_symmetric_no_zone_preference(self):
        """κ=0.5: swapping the two zones' gradients flips the move's sign but not
        its magnitude — there is no built-in preference for the stressed zone."""
        cfg = dict(grad_alpha=0.01, grad_eps=1e9, anchor=0.0, kappa=0.5,
                   dvref_max=1e9)
        a = _coord(**cfg); a.update({10: (1.0, -5.0)})
        b = _coord(**cfg); b.update({10: (-5.0, 1.0)})
        assert a.dvref[10] == pytest.approx(-b.dvref[10])


# ── Clause 3: NOT A RESCUE (help saturates with neighbour stress) ─────────

class TestNotARescue:
    def test_help_saturates_as_neighbour_stress_grows(self):
        """Fix the helper's gradient; make the neighbour arbitrarily more
        stressed.  The move (and the helper's incurred cost) SATURATES at the
        grad_eps bound and does NOT grow with the neighbour's need.  This is the
        decisive 'bounded, not preferential rescue' property: the move is limited
        by what is cheap for the helper, not by how badly the neighbour needs it.
        """
        kappa, eps, g_i = 0.5, 1.0, -1.0       # helper i mildly wants boundary up
        cap = eps / (abs(g_i) * kappa)         # analytic saturation of |Δ|
        moves = []
        for b in (2.0, 5.0, 20.0, 100.0, 1e3, 1e6):   # neighbour ever more stressed
            c = _coord(grad_alpha=1.0, grad_eps=eps, anchor=0.0, kappa=kappa,
                       dvref_max=1e9)
            c.update({10: (g_i, -b)})          # both want up -> conflict
            moves.append(abs(c.dvref[10]))
        assert max(moves) <= cap + 1e-9                  # never exceeds the cap
        assert moves[-1] == pytest.approx(cap, rel=1e-6)  # deep stress -> at the cap
        assert abs(moves[-1] - moves[-2]) <= 1e-6 * cap   # flat tail (saturated)


# ── Clause 4: CHEAP / MUTUALLY-BENEFICIAL (strict at grad_eps=0) ──────────

class TestCheapMutuallyBeneficial:
    def test_grad_eps_zero_freezes_any_costly_move(self):
        """grad_eps=0: a move that would worsen EITHER zone is frozen — only
        strictly mutually-beneficial moves are allowed (the 'cheap' end)."""
        c = _coord(grad_alpha=1.0, grad_eps=0.0, anchor=0.0, kappa=0.5,
                   dvref_max=1e9)
        c.update({10: (-1.0, -40.0)})          # conflict -> would cost helper i
        assert c.dvref[10] == pytest.approx(0.0)

    def test_grad_eps_zero_takes_pareto_move_fully(self):
        """A strictly mutually-beneficial move (neither zone worsens) is taken in
        full even at grad_eps=0 — coordination is free when it costs no one."""
        c = _coord(grad_alpha=0.01, grad_eps=0.0, anchor=0.0, kappa=0.5,
                   dvref_max=1e9)
        g_i, g_j = 2.0, -2.0                   # ΔV_ref-down helps BOTH zones
        c.update({10: (g_i, g_j)})
        G = 0.5 * g_i - 0.5 * g_j
        assert c.dvref[10] == pytest.approx(-0.01 * G)   # full joint step, not frozen


# ── Runnable demonstration ────────────────────────────────────────────────

def _demo():
    """Print the saturation evidence for the 'not a rescue' clause."""
    kappa, eps, g_i = 0.5, 1.0, -1.0
    cap = eps / (abs(g_i) * kappa)
    print("\nHelp vs. neighbour stress  (helper gradient g_i = %.1f, grad_eps = %.1f)" % (g_i, eps))
    print("  the MOVE and HELPER COST saturate; the neighbour's gain keeps growing")
    print("  -> bounded support, not a proportional rescue.\n")
    print(f"  {'g_j (stress)':>14} {'|dV_ref| move':>14} {'helper cost dJ_i':>18} {'neighbour gain dJ_j':>20}")
    for b in (2.0, 5.0, 20.0, 100.0, 1e3, 1e6):
        c = _coord(grad_alpha=1.0, grad_eps=eps, anchor=0.0, kappa=kappa, dvref_max=1e9)
        c.update({10: (g_i, -b)})
        d = c.dvref[10]
        dJ_i, dJ_j = _dJ(g_i, -b, kappa, d)
        print(f"  {-b:>14.1f} {abs(d):>14.4f} {dJ_i:>18.4f} {dJ_j:>20.2f}")
    print(f"\n  analytic move cap = grad_eps/(|g_i|*kappa) = {cap:.4f}  (independent of g_j)")


if __name__ == "__main__":
    _demo()
