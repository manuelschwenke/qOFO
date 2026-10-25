"""The DSO controllers' voltage band (MultiTSOConfig.dso_v_min_pu / _max_pu).

Added 2026-10-25: an operating margin the DSOs keep inside the grid-code band.
Before it, the runner never passed a band, so every DSO controller held the
class default 0.90 .. 1.10 pu. These tests pin that the default still
reproduces that, that a band reaches every DSO controller, and that an
inverted band is refused.
"""
from __future__ import annotations

import dataclasses

import pytest

from configs.config import MultiTSOConfig
from controller.dso_controller import DSOControllerConfig


def test_default_band_equals_the_controller_default() -> None:
    cfg = MultiTSOConfig()
    fields = {f.name: f.default for f in dataclasses.fields(DSOControllerConfig)}
    assert (cfg.dso_v_min_pu, cfg.dso_v_max_pu) == (
        fields["v_min_pu"], fields["v_max_pu"]) == (0.90, 1.10)


@pytest.mark.parametrize("lo, hi", [(1.08, 0.92), (1.0, 1.0), (0.0, 1.1)])
def test_invalid_band_is_refused(lo: float, hi: float) -> None:
    with pytest.raises(ValueError, match="DSO voltage band"):
        MultiTSOConfig(dso_v_min_pu=lo, dso_v_max_pu=hi)


def test_project_runner_config_sets_the_operating_margin() -> None:
    from experiments.run_multi_system_ofo import make_config
    cfg = make_config()
    assert (cfg.dso_v_min_pu, cfg.dso_v_max_pu) == (0.92, 1.08)


def test_band_reaches_every_dso_controller() -> None:
    """Build the controllers (no time step) and read their configured band."""
    from configs.paramsets import apply_paramset
    from experiments.run_multi_system_ofo import make_config
    from tuning._sim_loader import get_run_multi_tso_dso

    cfg = dataclasses.replace(
        apply_paramset(make_config(), "stage1_start_10_2026_banks", verbose=False),
        dso_v_min_pu=0.93, dso_v_max_pu=1.07, n_total_s=60.0, contingencies=[],
        verbose=0, live_plot_controller=False, live_plot_cascade=False,
        live_plot_system=False)
    got = {}

    def hook(state):
        got.update(state)
        return True                           # stop before the time loop

    get_run_multi_tso_dso()(cfg, pre_loop_hook=hook)
    dsos = got["dso_controllers"]
    assert dsos
    for dso_id, ctrl in dsos.items():
        assert (ctrl.config.v_min_pu, ctrl.config.v_max_pu) == (0.93, 1.07), dso_id
