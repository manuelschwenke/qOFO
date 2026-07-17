"""Configuration regression tests for the thesis-facing SBX-H demo."""

import importlib


def test_single_demo_initializes_sbx_at_time_zero() -> None:
    demo = importlib.import_module("experiments.archived.014_SBX_SINGLE_DEMO")
    config = demo.make_config(
        minutes=2.0,
        stress_on_min=1.0,
        stress_off_min=None,
        sink_mvar=1.0,
        verbose=0,
    )

    assert config.sbx_warmup_s == 0.0
    assert config.sbx_config.q_band_mvar == 10.0
    assert config.sbx_support_intervals is None
    assert config.zone_v_setpoints_pu == {1: 1.03, 2: 1.03, 3: 1.03}
