"""End-to-end CLI smoke test for the tuning workflow.

Marked ``@pytest.mark.slow`` — runs ``tuning.tune`` for 5 trials, then
``tuning.validate`` on 5 randomised scenarios, and verifies that both
HTML reports are produced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from configs.multi_tso_config import MultiTSOConfig
from tuning._io import load_tuned_params, save_config_yaml
from tuning.parameters import BO_DIMS
from tuning import tune as tune_cli
from tuning import validate as validate_cli


@pytest.mark.slow
def test_full_tune_then_validate(
    baseline_cfg: MultiTSOConfig,
    tmp_path: Path,
) -> None:
    """End-to-end: ``tune`` then ``validate``, both via the CLI."""
    baseline_yaml = tmp_path / "baseline.yaml"
    save_config_yaml(baseline_cfg, baseline_yaml)
    assert baseline_yaml.exists()

    params_yaml = tmp_path / "params.yaml"
    tune_html = tmp_path / "tuning.html"
    storage_url = f"sqlite:///{(tmp_path / 'smoke.db').as_posix()}"

    rc_tune = tune_cli.main([
        "--baseline", str(baseline_yaml),
        "--n-trials", "5",
        "--n-startup-trials", "3",
        "--study-name", "smoke",
        "--storage", storage_url,
        "--n-jobs", "1",
        "--output", str(params_yaml),
        "--report", str(tune_html),
        "--no-progress-bar",
        "--no-cache-ceilings",
    ])
    assert rc_tune == 0
    assert params_yaml.exists()
    assert tune_html.exists()

    p_loaded, m_loaded = load_tuned_params(params_yaml)
    assert set(p_loaded.keys()) == {p.name for p in BO_DIMS}
    assert m_loaded.get("study_name") == "smoke"
    assert m_loaded.get("n_trials", 0) >= 1

    validation_html = tmp_path / "validation.html"
    rc_validate = validate_cli.main([
        "--params", str(params_yaml),
        "--baseline", str(baseline_yaml),
        "--n-scenarios", "5",
        "--seed", "0",
        "--n-jobs", "1",
        "--report", str(validation_html),
    ])
    assert rc_validate == 0
    assert validation_html.exists()

    # The template must actually render (not just emit a skeleton).
    size = validation_html.stat().st_size
    assert size > 5_000, (
        f"validation report only {size} bytes — template did not render"
    )
