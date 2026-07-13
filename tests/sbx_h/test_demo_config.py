"""Configuration regression tests for the thesis-facing SBX-H demo."""

import importlib


def test_single_demo_initializes_sbx_at_time_zero() -> None:
    demo = importlib.import_module("experiments.014_SBX_SINGLE_DEMO")
    config = demo.make_config(
        minutes=2.0,
        stress_on_min=1.0,
        stress_off_min=None,
        sink_mvar=1.0,
        verbose=0,
    )

    assert config.sbx_warmup_s == 0.0
