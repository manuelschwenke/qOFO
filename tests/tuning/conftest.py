from __future__ import annotations

from datetime import datetime

import pytest

from configs.multi_tso_config import MultiTSOConfig


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (selectable with `-m slow`; otherwise "
        "skipped via `-m 'not slow'`)",
    )


@pytest.fixture
def baseline_cfg() -> MultiTSOConfig:
    """Minimal valid ``MultiTSOConfig`` for fast tests.

    The numerical values reflect the default qOFO baseline used in
    ``main()``; they are only set explicitly so that the fixture is
    self-contained and immune to drift in the dataclass defaults.
    """
    return MultiTSOConfig(
        n_total_s=180.0,
        tso_period_s=180.0,
        dso_period_s=10.0,
        g_v=120000.0,
        g_q=200.0,
        dso_g_v=20000.0,
        g_w_der=10.0,
        g_w_gen=5e7,
        g_w_pcc=100.0,
        g_w_tso_oltc=100.0,
        g_w_dso_der=1000.0,
        g_w_dso_oltc=50.0,
        scenario="wind_replace",
        use_fixed_zones=True,
        run_stability_analysis=True,
        verbose=0,
        live_plot_controller=False,
        live_plot_cascade=False,
        live_plot_system=False,
        use_profiles=True,
        start_time=datetime(2016, 4, 15, 12, 0),
        contingencies=[],
    )
