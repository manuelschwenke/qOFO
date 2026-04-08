"""
Tests for analysis/pso_tune_ofo.py
==================================

Validates the PSO joint TSO + DSO tuner on a synthetic 2-zone + 1-DSO
problem.  The test focuses on the invariants the runner relies on:

* PSO never regresses below the Gershgorin warm start
  (``pso_result.fitness <= warm_start_fitness``)
* Every per-actuator g_w respects its PSO floor
* The synthetic problem is small enough that PSO converges to a feasible
  solution (``fitness < 1``) within a tiny budget (15 particles, 30 iters)
* The decode/bound bookkeeping handles a TSO zone *with* PCC actuators
  AND a DSO with shunt actuators (covers all column blocks)
"""

import numpy as np
import pytest

from analysis.pso_tune_ofo import (
    DSOTuneInput,
    PSOTuningResult,
    _dso_cascade_decay,
    tune_pso_all,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Synthetic problem builders
# ═══════════════════════════════════════════════════════════════════════════════

def _make_psd_diag_block(n: int, seed: int) -> np.ndarray:
    """Build a tall H whose columns are independent (so H^T H is PD)."""
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((n + 4, n))
    return H


def _build_2zone_1dso():
    """A small 2-TSO-zone + 1-DSO test problem.

    Zone layout:
        Zone 1: 2 DER + 1 PCC + 1 GEN + 1 OLTC = 5 actuators
        Zone 2: 2 DER + 0 PCC + 2 GEN + 1 OLTC = 5 actuators
        DSO  1: 3 DER + 1 OLTC + 1 SHUNT       = 5 actuators
    """
    # ── TSO ─────────────────────────────────────────────────────────────
    # Cross-zone coupling is intentionally moderate so a feasible region
    # exists but the warm start is not already optimal.
    H11 = _make_psd_diag_block(5, seed=11)
    H22 = _make_psd_diag_block(5, seed=22)
    H12 = 0.15 * np.random.default_rng(12).standard_normal((H11.shape[0], 5))
    H21 = 0.15 * np.random.default_rng(21).standard_normal((H22.shape[0], 5))

    H_blocks = {(1, 1): H11, (1, 2): H12, (2, 1): H21, (2, 2): H22}
    Q_obj_list = [np.ones(H11.shape[0]) * 50.0, np.ones(H22.shape[0]) * 50.0]
    actuator_counts = [
        {'n_der': 2, 'n_pcc': 1, 'n_gen': 1, 'n_oltc': 1},
        {'n_der': 2, 'n_pcc': 0, 'n_gen': 2, 'n_oltc': 1},
    ]
    zone_ids = [1, 2]

    # ── DSO ─────────────────────────────────────────────────────────────
    H_dso = _make_psd_diag_block(5, seed=33)
    q_obj_dso = np.concatenate([
        np.full(2, 1.0),                # 2 interface Q rows
        np.full(H_dso.shape[0] - 4, 100.0),  # voltage rows
        np.zeros(2),                    # current rows (no weight)
    ])
    dso_inputs = [DSOTuneInput(
        dso_id="DSO_TEST",
        H=H_dso,
        q_obj_diag=q_obj_dso,
        n_der=3,
        n_oltc=1,
        n_shunt=1,
    )]

    floors_tso = {'der': 1e-3, 'pcc': 0.01, 'gen': 1e2, 'oltc': 1.0}
    floors_dso = {'der': 1e-3, 'oltc': 0.1, 'shunt': 10.0}

    return (
        H_blocks, Q_obj_list, actuator_counts, zone_ids,
        dso_inputs, floors_tso, floors_dso,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPSOInvariants:

    def test_no_regression_versus_warm_start(self):
        """PSO must never return a worse fitness than the warm start."""
        (H_blocks, Q_obj_list, actuator_counts, zone_ids,
         dso_inputs, floors_tso, floors_dso) = _build_2zone_1dso()

        result = tune_pso_all(
            H_blocks=H_blocks,
            Q_obj_list=Q_obj_list,
            actuator_counts=actuator_counts,
            zone_ids=zone_ids,
            dso_inputs=dso_inputs,
            floors_tso=floors_tso,
            floors_dso=floors_dso,
            swarm_size=15,
            max_iterations=30,
            seed=42,
            verbose=False,
        )
        assert isinstance(result, PSOTuningResult)
        assert np.isfinite(result.fitness)
        assert result.fitness <= result.warm_start_fitness + 1e-9

    def test_floors_respected(self):
        """Every per-actuator g_w must lie above its PSO floor."""
        (H_blocks, Q_obj_list, actuator_counts, zone_ids,
         dso_inputs, floors_tso, floors_dso) = _build_2zone_1dso()

        result = tune_pso_all(
            H_blocks=H_blocks,
            Q_obj_list=Q_obj_list,
            actuator_counts=actuator_counts,
            zone_ids=zone_ids,
            dso_inputs=dso_inputs,
            floors_tso=floors_tso,
            floors_dso=floors_dso,
            swarm_size=15,
            max_iterations=30,
            seed=7,
            verbose=False,
        )

        # ── TSO floors ──────────────────────────────────────────────────
        for ztr, counts in zip(result.tso_zones, actuator_counts):
            gw = ztr.g_w
            off = 0
            for type_name in ('der', 'pcc', 'gen', 'oltc'):
                n = counts.get(f'n_{type_name}', 0)
                if n <= 0:
                    continue
                block = gw[off:off + n]
                # Allow tiny floating-point slack
                assert np.all(block >= floors_tso[type_name] * (1 - 1e-9)), (
                    f"Zone {ztr.zone_id}: {type_name} block "
                    f"{block} below floor {floors_tso[type_name]}"
                )
                off += n

        # ── DSO floors ──────────────────────────────────────────────────
        for d in dso_inputs:
            gw_dso, _alpha, _rho, _margin = result.dso[d.dso_id]
            off = 0
            for type_name, n in (('der', d.n_der),
                                 ('oltc', d.n_oltc),
                                 ('shunt', d.n_shunt)):
                if n <= 0:
                    continue
                block = gw_dso[off:off + n]
                assert np.all(block >= floors_dso[type_name] * (1 - 1e-9)), (
                    f"DSO {d.dso_id}: {type_name} block "
                    f"{block} below floor {floors_dso[type_name]}"
                )
                off += n

    def test_feasible_on_synthetic(self):
        """On a small well-conditioned synthetic problem PSO should reach
        f < 1 within 30 iterations."""
        (H_blocks, Q_obj_list, actuator_counts, zone_ids,
         dso_inputs, floors_tso, floors_dso) = _build_2zone_1dso()

        result = tune_pso_all(
            H_blocks=H_blocks,
            Q_obj_list=Q_obj_list,
            actuator_counts=actuator_counts,
            zone_ids=zone_ids,
            dso_inputs=dso_inputs,
            floors_tso=floors_tso,
            floors_dso=floors_dso,
            swarm_size=15,
            max_iterations=30,
            seed=123,
            verbose=False,
        )
        assert result.fitness < 1.0, (
            f"PSO failed to reach feasibility on the synthetic problem: "
            f"fitness = {result.fitness:.4f}, "
            f"warnings = {result.feasibility_warnings}"
        )
        assert result.converged

    def test_history_length_and_keys(self):
        """The convergence history should have one entry per iteration
        and each entry should expose the metrics the JSON serialiser
        relies on."""
        (H_blocks, Q_obj_list, actuator_counts, zone_ids,
         dso_inputs, floors_tso, floors_dso) = _build_2zone_1dso()

        result = tune_pso_all(
            H_blocks=H_blocks,
            Q_obj_list=Q_obj_list,
            actuator_counts=actuator_counts,
            zone_ids=zone_ids,
            dso_inputs=dso_inputs,
            floors_tso=floors_tso,
            floors_dso=floors_dso,
            swarm_size=10,
            max_iterations=20,
            seed=1,
            verbose=False,
        )
        assert len(result.history) >= 1
        assert len(result.history) <= 20
        required = {
            'iter', 'fitness', 'rho_max_tso', 'rho_max_dso',
            'spectral', 'gamma', 'max_dso_decay',
            'alpha_eff_tso', 'alpha_eff_dso',
        }
        for entry in result.history:
            missing = required - set(entry.keys())
            assert not missing, f"Missing history keys: {missing}"


class TestDSOCascadeDecay:

    def test_well_conditioned_decay(self):
        """A diagonal H with uniform g_w should yield rho = 0 (perfect
        contraction)."""
        H = np.eye(4) * 2.0
        q = np.ones(4)
        g_w = np.full(4, 4.0)  # any uniform value
        rho, decay, alpha_opt = _dso_cascade_decay(H, q, g_w, n_inner=3)
        assert rho == pytest.approx(0.0, abs=1e-12)
        assert decay == pytest.approx(0.0, abs=1e-12)
        assert alpha_opt == pytest.approx(1.0)

    def test_known_two_eigenvalue_case(self):
        """Two distinct eigenvalues 1 and 4 → rho = (4-1)/(4+1) = 0.6."""
        # Build a H so that C = diag([1, 4])
        H = np.diag([1.0, 2.0])
        q = np.ones(2)
        g_w = np.ones(2)
        rho, decay, _ = _dso_cascade_decay(H, q, g_w, n_inner=2)
        assert rho == pytest.approx(0.6, abs=1e-9)
        assert decay == pytest.approx(0.36, abs=1e-9)
