"""
Tests for analysis/pso_tune_ofo.py (eigenvector-pump g_w tuner)
================================================================

Validates the deterministic pump on a synthetic 2-zone + 1-DSO problem.
"""

import numpy as np
import pytest

from analysis.pso_tune_ofo import (
    DSOTuneInput,
    TuneGwResult,
    _dso_cascade_decay,
    tune_gw,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Synthetic problem builder
# ═══════════════════════════════════════════════════════════════════════════════

def _make_psd_diag_block(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((n + 4, n))
    return H


def _build_2zone_1dso():
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

    H_dso = _make_psd_diag_block(5, seed=33)
    q_obj_dso = np.concatenate([
        np.full(2, 1.0),
        np.full(H_dso.shape[0] - 4, 100.0),
        np.zeros(2),
    ])
    dso_inputs = [DSOTuneInput(
        dso_id="DSO_TEST", H=H_dso, q_obj_diag=q_obj_dso,
        n_der=3, n_oltc=1, n_shunt=1,
    )]

    floors_tso = {'der': 1e-3, 'pcc': 0.01, 'gen': 1e2, 'oltc': 1.0}
    floors_dso = {'der': 1e-3, 'oltc': 0.1, 'shunt': 10.0}

    # Initial g_w (user's hand-tuned values)
    gw_tso_init = [np.ones(5) * 10.0, np.ones(5) * 10.0]
    gw_dso_init = [np.ones(5) * 5.0]

    return (
        H_blocks, Q_obj_list, actuator_counts, zone_ids,
        dso_inputs, floors_tso, floors_dso, gw_tso_init, gw_dso_init,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTuneGw:

    def test_user_init_respected(self):
        """Pump must never decrease g_w below the user's init values."""
        (H_blocks, Q_obj_list, actuator_counts, zone_ids,
         dso_inputs, floors_tso, floors_dso,
         gw_tso_init, gw_dso_init) = _build_2zone_1dso()

        # Use large init values — pump should keep them
        large_tso = [np.full(5, 1000.0), np.full(5, 1000.0)]
        large_dso = [np.full(5, 500.0)]

        result = tune_gw(
            H_blocks=H_blocks, Q_obj_list=Q_obj_list,
            actuator_counts=actuator_counts, zone_ids=zone_ids,
            dso_inputs=dso_inputs,
            gw_tso_init=large_tso, gw_dso_init=large_dso,
            floors_tso=floors_tso, floors_dso=floors_dso,
            verbose=False,
        )
        for idx, gw in enumerate(result.gw_tso):
            assert np.all(gw >= large_tso[idx] - 1e-9), (
                f"Zone {zone_ids[idx]}: pump decreased g_w below user init"
            )
        for idx, gw in enumerate(result.gw_dso):
            assert np.all(gw >= large_dso[idx] - 1e-9), (
                f"DSO {dso_inputs[idx].dso_id}: pump decreased g_w below user init"
            )

    def test_floors_respected(self):
        """Every per-actuator g_w must lie above its floor."""
        (H_blocks, Q_obj_list, actuator_counts, zone_ids,
         dso_inputs, floors_tso, floors_dso,
         gw_tso_init, gw_dso_init) = _build_2zone_1dso()

        result = tune_gw(
            H_blocks=H_blocks, Q_obj_list=Q_obj_list,
            actuator_counts=actuator_counts, zone_ids=zone_ids,
            dso_inputs=dso_inputs,
            gw_tso_init=gw_tso_init, gw_dso_init=gw_dso_init,
            floors_tso=floors_tso, floors_dso=floors_dso,
            verbose=False,
        )

        for ztr_gw, counts in zip(result.gw_tso, actuator_counts):
            off = 0
            for type_name in ('der', 'pcc', 'gen', 'oltc'):
                n = counts.get(f'n_{type_name}', 0)
                if n <= 0:
                    continue
                block = ztr_gw[off:off + n]
                assert np.all(block >= floors_tso[type_name] * (1 - 1e-9)), (
                    f"{type_name} block {block} below floor {floors_tso[type_name]}"
                )
                off += n

        for d, gw in zip(dso_inputs, result.gw_dso):
            off = 0
            for type_name, n in (('der', d.n_der), ('oltc', d.n_oltc), ('shunt', d.n_shunt)):
                if n <= 0:
                    continue
                block = gw[off:off + n]
                assert np.all(block >= floors_dso[type_name] * (1 - 1e-9)), (
                    f"DSO {d.dso_id}: {type_name} below floor {floors_dso[type_name]}"
                )
                off += n

    def test_feasible_on_synthetic(self):
        """On a small synthetic problem the pump should reach feasibility."""
        (H_blocks, Q_obj_list, actuator_counts, zone_ids,
         dso_inputs, floors_tso, floors_dso,
         gw_tso_init, gw_dso_init) = _build_2zone_1dso()

        result = tune_gw(
            H_blocks=H_blocks, Q_obj_list=Q_obj_list,
            actuator_counts=actuator_counts, zone_ids=zone_ids,
            dso_inputs=dso_inputs,
            gw_tso_init=gw_tso_init, gw_dso_init=gw_dso_init,
            floors_tso=floors_tso, floors_dso=floors_dso,
            spectral_target=1.95,
            verbose=False,
        )
        assert result.spectral_feasible, (
            f"Pump failed: lam_max_sys = {result.lam_max_sys:.4f}"
        )

    def test_returns_correct_types(self):
        """Result has the expected types and shapes."""
        (H_blocks, Q_obj_list, actuator_counts, zone_ids,
         dso_inputs, floors_tso, floors_dso,
         gw_tso_init, gw_dso_init) = _build_2zone_1dso()

        result = tune_gw(
            H_blocks=H_blocks, Q_obj_list=Q_obj_list,
            actuator_counts=actuator_counts, zone_ids=zone_ids,
            dso_inputs=dso_inputs,
            gw_tso_init=gw_tso_init, gw_dso_init=gw_dso_init,
            floors_tso=floors_tso, floors_dso=floors_dso,
            verbose=False,
        )
        assert isinstance(result, TuneGwResult)
        assert len(result.gw_tso) == 2
        assert len(result.gw_dso) == 1
        assert len(result.per_zone_kappa) == 2
        assert len(result.per_dso_lam_max) == 1
        assert isinstance(result.lam_max_sys, float)
        assert isinstance(result.pump_iterations_tso, int)


class TestDSOCascadeDecay:

    def test_well_conditioned_decay(self):
        H = np.eye(4) * 2.0
        q = np.ones(4)
        g_w = np.full(4, 4.0)
        rho, decay, alpha_opt = _dso_cascade_decay(H, q, g_w, n_inner=3)
        assert rho == pytest.approx(0.0, abs=1e-12)
        assert decay == pytest.approx(0.0, abs=1e-12)
        assert alpha_opt == pytest.approx(1.0)

    def test_known_two_eigenvalue_case(self):
        H = np.diag([1.0, 2.0])
        q = np.ones(2)
        g_w = np.ones(2)
        rho, decay, _ = _dso_cascade_decay(H, q, g_w, n_inner=2)
        assert rho == pytest.approx(0.6, abs=1e-9)
        assert decay == pytest.approx(0.36, abs=1e-9)
