from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pf.profile_playback import (
    UNITY_COLUMN,
    build_profile_schedule,
    referenced_profile_columns,
    write_measurement_file,
)
from pf.session import PFSessionError


def _profiles():
    index = pd.date_range("2016-04-01", periods=4, freq="20s")
    return pd.DataFrame(
        {
            "load_p": [1.0, 1.1, 1.2, 1.3],
            "load_q": [0.9, 0.8, 0.7, 0.6],
            "wind": [0.5, 0.6, 0.7, 0.8],
        },
        index=index,
    )


def test_schedule_matches_existing_dispatch_clock():
    profiles = _profiles()
    schedule = build_profile_schedule(
        profiles,
        columns=("load_p", "wind"),
        start_time=profiles.index[0].to_pydatetime(),
        dt_s=20.0,
        duration_s=60.0,
        transition_delay_s=0.51,
    )

    assert schedule.columns == (UNITY_COLUMN, "load_p", "wind")
    np.testing.assert_allclose(schedule.time_s, [0.0, 0.51, 20.51, 40.51])
    np.testing.assert_allclose(schedule.values[:, 0], 1.0)
    np.testing.assert_allclose(schedule.values[:, 1], [1.0, 1.1, 1.2, 1.3])
    np.testing.assert_allclose(schedule.values[:, 2], [0.5, 0.6, 0.7, 0.8])
    assert schedule.channel("wind") == 3


def test_measurement_file_is_plain_pf_format(tmp_path):
    profiles = _profiles()
    schedule = build_profile_schedule(
        profiles,
        columns=("load_p",),
        start_time=profiles.index[0].to_pydatetime(),
        dt_s=20.0,
        duration_s=40.0,
        transition_delay_s=0.51,
    )

    path = write_measurement_file(tmp_path / "profiles.txt", schedule)
    lines = path.read_text(encoding="ascii").splitlines()

    assert lines[0] == "2"
    assert lines[1] == "0 1 1"
    assert lines[2] == "0.51 1 1.1"
    assert lines[-1] == "20.51 1 1.2"


def test_referenced_columns_are_unique_and_ignore_empty_values():
    net = SimpleNamespace(
        load=pd.DataFrame(
            {
                "profile_p": ["load_p", None, "load_p"],
                "profile_q": ["load_q", np.nan, ""],
            }
        ),
        sgen=pd.DataFrame({"profile": ["wind", None]}),
    )

    assert referenced_profile_columns(net) == ("load_p", "load_q", "wind")


def test_schedule_fails_for_missing_or_nonfinite_channel():
    profiles = _profiles()
    start = profiles.index[0].to_pydatetime()
    with pytest.raises(PFSessionError, match="missing columns"):
        build_profile_schedule(
            profiles,
            columns=("absent",),
            start_time=start,
            dt_s=20.0,
            duration_s=20.0,
            transition_delay_s=0.51,
        )

    profiles.loc[profiles.index[1], "wind"] = np.nan
    with pytest.raises(PFSessionError, match="non-finite"):
        build_profile_schedule(
            profiles,
            columns=("wind",),
            start_time=start,
            dt_s=20.0,
            duration_s=20.0,
            transition_delay_s=0.51,
        )
