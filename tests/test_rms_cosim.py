import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.helpers.rms_replay import (
    endpoint_comparison,
    interval_settling_table,
    rms_controlled_trajectories,
    static_controlled_trajectories,
)
from experiments.run_multi_system_ofo import make_config as make_reference_config
from experiments.run_comparison_rms_cosim_qss import (
    _markdown_table,
    _write_summary,
    make_gate_e_config,
)


def test_static_controlled_outputs_use_interval_endpoints():
    records = [
        SimpleNamespace(
            time_s=20.0,
            dso_trafo_q_actual_mvar={"DSO_1|trafo_7": 3.0},
            zone_v_mean={1: 1.01},
        ),
        SimpleNamespace(
            time_s=40.0,
            dso_trafo_q_actual_mvar={"DSO_1|trafo_7": 4.0},
            zone_v_mean={1: 1.02},
        ),
    ]

    out = static_controlled_trajectories(records)

    np.testing.assert_allclose(out["qSTS_t7"][0], [20.0, 40.0])
    np.testing.assert_allclose(out["qSTS_t7"][1], [3.0, 4.0])
    np.testing.assert_allclose(out["vZone_1"][1], [1.01, 1.02])


def test_rms_normalisation_builds_same_tn_pq_zone_mean():
    time = np.array([0.0, 1.0, 2.0])
    raw = {
        "qSTS_NC3W_DSO_1_t7": (time, np.array([1.0, 2.0, 3.0])),
        "u_TN_bus0": (time, np.array([1.00, 1.01, 1.02])),
        "u_TN_bus1": (time, np.array([1.02, 1.03, 1.04])),
    }
    doc = {
        "model": {
            "trafo3w": {"7": {}},
            "bus": {
                "0": {"subnet": "TN", "vn_kv": 345.0},
                "1": {"subnet": "TN", "vn_kv": 345.0},
                "2": {"subnet": "TN", "vn_kv": 10.5},
            },
            "gen": {},
        },
        "zone_map": {"1": [0, 1, 2]},
    }

    out = rms_controlled_trajectories(raw, doc)

    np.testing.assert_allclose(out["qSTS_t7"][1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(out["vZone_1"][1], [1.01, 1.02, 1.03])


def test_rms_normalisation_excludes_generator_bus_from_zone_mean():
    time = np.array([0.0, 1.0])
    raw = {
        "qSTS_NC3W_DSO_1_t7": (time, np.array([0.0, 0.0])),
        "u_TN_bus0": (time, np.array([1.0, 1.0])),
        "u_TN_bus1": (time, np.array([1.2, 1.2])),
    }
    doc = {
        "model": {
            "trafo3w": {"7": {}},
            "bus": {
                "0": {"subnet": "TN", "vn_kv": 345.0},
                "1": {"subnet": "TN", "vn_kv": 345.0},
            },
            "gen": {"0": {"bus": 1, "in_service": True}},
        },
        "zone_map": {"1": [0, 1]},
    }

    out = rms_controlled_trajectories(raw, doc)

    np.testing.assert_allclose(out["vZone_1"][1], [1.0, 1.0])


def test_interval_settling_detects_settled_and_censored_windows():
    time = np.arange(0.0, 40.1, 0.1)
    settled = np.where(
        time < 0.1,
        0.0,
        10.0 + np.exp(-(time - 0.1)) * (-10.0),
    )
    # Persistent alternating motion leaves the final sample outside the
    # final-tail mean's narrow voltage band.
    unsettled = (np.arange(time.size) % 2).astype(float)
    trajectories = {
        "qSTS_t1": (time, settled),
        "vZone_1": (time, 1.0 + 0.01 * unsettled),
    }

    table = interval_settling_table(
        trajectories,
        interval_s=20.0,
        total_s=40.0,
        voltage_abs_floor_pu=1e-5,
    )

    q_first = table[
        (table["signal"] == "qSTS_t1")
        & (table["interval_start_s"] == 0.0)
    ].iloc[0]
    assert bool(q_first["settled_within_interval"])
    assert 0.0 < q_first["settling_time_s"] < 20.0

    v_first = table[
        (table["signal"] == "vZone_1")
        & (table["interval_start_s"] == 0.0)
    ].iloc[0]
    assert not bool(v_first["settled_within_interval"])
    assert v_first["settling_time_s"] == 20.0


def test_endpoint_comparison_interpolates_rms_at_static_endpoints():
    static = {
        "qSTS_t1": (
            np.array([20.0, 40.0]),
            np.array([10.0, 20.0]),
        )
    }
    rms = {
        "qSTS_t1": (
            np.array([0.0, 20.0, 40.0]),
            np.array([0.0, 11.0, 18.0]),
        )
    }

    comparison = endpoint_comparison(static, rms)

    np.testing.assert_allclose(comparison["error"], [1.0, -2.0])
    np.testing.assert_allclose(comparison["abs_error"], [1.0, 2.0])


def test_cosim_config_only_overrides_reference_envelope():
    cfg = make_gate_e_config(40.0, verbose=0)
    expected = make_reference_config()
    expected.n_total_s = 40.0
    expected.tso_period_s = 180.0
    expected.dso_period_s = 20.0
    expected.dt_s = 20.0
    # Profiles ON and the +-1.0 pu capability override are part of the
    # co-simulation envelope (set in make_cosim_config since 2026-07-21);
    # this test had drifted out of date against them.
    expected.use_profiles = True
    expected.der_q_capability_override_pu = 1.0
    expected.contingencies = []
    expected.measurement_noise.enabled = False
    expected.enable_reachability_guard = False
    expected.live_plot_controller = False
    expected.live_plot_cascade = False
    expected.live_plot_system = False
    expected.live_plot_tracking = False
    expected.live_plot_sbx = False
    expected.verbose = 0

    assert cfg == expected
    assert cfg.coordination_mode == "sbx_h"
    assert cfg.sbx_config == make_reference_config().sbx_config
    assert cfg.install_tso_tertiary_shunts is True
    assert cfg.shunt_dispatch == "integrator"
    assert cfg.tso_shunt_kind == "msc_msr"


def test_markdown_table_has_no_optional_dependency():
    frame = pd.DataFrame(
        {"quantity": ["interface_q"], "max": [1.23456789], "ok": [True]}
    )

    table = _markdown_table(frame)

    assert "| quantity | max | ok |" in table
    assert "| interface_q | 1.23457 | True |" in table


def test_gate_e_summary_blocks_non_equivalent_qv_plant(tmp_path):
    settling = pd.DataFrame(
        {
            "settled_within_interval": [True],
            "signal_type": ["interface_q"],
        }
    )
    settling_by_type = pd.DataFrame(
        {
            "signal_type": ["interface_q"],
            "n_intervals": [1],
            "unsettled_intervals": [0],
            "max_settling_s": [1.0],
            "p95_settling_s": [1.0],
            "max_overshoot_fraction": [0.0],
        }
    )
    endpoint = pd.DataFrame(
        {
            "signal_type": ["interface_q"],
            "error": [0.0],
            "abs_error": [0.0],
        }
    )
    config = SimpleNamespace(
        coordination_mode="sbx",
        install_tso_tertiary_shunts=True,
        shunt_dispatch="integrator",
    )
    plant = SimpleNamespace(
        t=20.0,
        skipped_writes=[],
        der_qv_local_control_equivalent=False,
    )

    verdict = _write_summary(
        tmp_path, duration_s=20.0, n_static=1, n_rms=1,
        config=config, plant=plant, settling=settling,
        settling_by_type=settling_by_type, endpoint=endpoint,
    )

    summary = json.loads((tmp_path / "gate_e_summary.json").read_text())
    assert verdict is False
    assert summary["gate_e_settling_verdict"] == "PASS"
    assert summary["gate_e_validation_verdict"] == "BLOCKED_DER_QV_MISMATCH"
    assert summary["comparison_validity"] == "PROVISIONAL_NON_EQUIVALENT_PLANT_LAW"
