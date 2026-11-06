"""unit_commitment_window_s: default off; D1/D2 commit over the 12-h design window (2026-11-05)."""
from datetime import datetime


def test_default_is_the_run_window():
    from configs.config import MultiTSOConfig
    assert MultiTSOConfig().unit_commitment_window_s is None


def test_deadband_runs_use_the_design_window():
    from experiments.ch_09_parameter_selection import ch_9_4_deadband as D
    cfg = D._config(1.0, datetime(2016, 7, 18, 14, 0), 5410.0)
    assert D.COMMIT_WINDOW_S == 12 * 3600.0
    assert cfg.unit_commitment_window_s == D.COMMIT_WINDOW_S
    assert cfg.n_total_s == 5410.0
