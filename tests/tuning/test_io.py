"""Unit tests for ``tuning/_io.py``."""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers.records import ContingencyEvent
from tuning._io import (
    jsonable,
    load_config_yaml,
    load_tuned_params,
    save_config_yaml,
    save_tuned_params,
)
from tuning.parameters import BO_DIMS


# ---------------------------------------------------------------------------
# 1. MultiTSOConfig YAML round-trip
# ---------------------------------------------------------------------------

def test_save_load_config_roundtrip(
    baseline_cfg: MultiTSOConfig,
    tmp_path: Path,
) -> None:
    cfg = dataclasses.replace(
        baseline_cfg,
        start_time=datetime(2017, 7, 4, 9, 0),
        contingencies=[
            ContingencyEvent(
                minute=10, element_type="gen", element_index=2,
                action="trip",
            ),
        ],
    )
    path = tmp_path / "cfg.yaml"
    save_config_yaml(cfg, path)
    assert path.exists()

    loaded = load_config_yaml(path)

    # All BO fields preserved
    for p in BO_DIMS:
        assert getattr(loaded, p.name) == pytest.approx(
            float(getattr(cfg, p.name))
        )

    # start_time round-trips as datetime (not str)
    assert isinstance(loaded.start_time, datetime)
    assert loaded.start_time == cfg.start_time

    # contingencies round-trip as ContingencyEvent
    assert len(loaded.contingencies) == 1
    ev = loaded.contingencies[0]
    assert isinstance(ev, ContingencyEvent)
    assert ev.element_type == "gen"
    assert ev.element_index == 2
    assert ev.action == "trip"
    assert ev.minute == 10


# ---------------------------------------------------------------------------
# 2. Tuned-params YAML round-trip
# ---------------------------------------------------------------------------

def test_save_load_tuned_params_roundtrip(tmp_path: Path) -> None:
    params = {p.name: float(i + 1) * 1.5 for i, p in enumerate(BO_DIMS)}
    meta = {
        "study_name": "smoke",
        "n_trials":   5,
        "best_value": 0.123,
        "ceilings":   {"g_w_der": 99.0, "g_v": 1e6},
        "timestamp":  "2026-04-27T12:00:00",
    }
    path = tmp_path / "tuned.yaml"
    save_tuned_params(params, meta, path)

    p_loaded, m_loaded = load_tuned_params(path)
    assert p_loaded == pytest.approx(params)
    assert m_loaded == meta


# ---------------------------------------------------------------------------
# 3. jsonable handles numpy / dataclass / datetime
# ---------------------------------------------------------------------------

def test_jsonable_handles_numpy() -> None:
    out = jsonable({
        "f64":     np.float64(1.5),
        "i64":     np.int64(7),
        "arr":     np.array([1, 2, 3]),
        "bool":    np.bool_(True),
        "dt":      datetime(2020, 1, 1, 12, 0),
        "nested": {"x": np.float32(0.25)},
        "list":    [np.int8(1), np.float64(2.5)],
    })

    # Plain Python types only
    assert isinstance(out["f64"], float) and out["f64"] == 1.5
    assert isinstance(out["i64"], int) and out["i64"] == 7
    assert isinstance(out["arr"], list) and out["arr"] == [1, 2, 3]
    assert isinstance(out["bool"], bool) and out["bool"] is True
    assert isinstance(out["dt"], str) and "2020-01-01" in out["dt"]
    assert isinstance(out["nested"]["x"], float)
    assert math.isclose(out["nested"]["x"], 0.25)
    assert isinstance(out["list"][0], int) and out["list"][0] == 1
    assert isinstance(out["list"][1], float) and out["list"][1] == 2.5
