"""Runner: an outage event of a machine the unit commitment keeps off-line is
void, with its restore (2026-11-07).  Without the guard the restore started
the decommitted machine.

At 2016-12-16 18:45 (the OP-R of the step tests) the commitment keeps G3, G7
and G9 off-line.  A first one-step run reads the off-line set from its first
record; a second run schedules a trip and a restore of one of them.
"""
import dataclasses
import datetime as dt

import numpy as np
import pytest

from experiments.ch_11_case2_multisystem import variants as V
from experiments.helpers.records import ContingencyEvent
from experiments.runners.multi_tso_dso import run_multi_tso_dso

START = dt.datetime(2016, 12, 16, 18, 45)


def _cfg(tmp_path, seconds, events=()):
    cfg = V.resolved("L2", 1, list(events), noise_seed=1)
    return dataclasses.replace(cfg, start_time=START, n_total_s=float(seconds),
                               result_dir=str(tmp_path), verbose=0)


@pytest.mark.slow
def test_offline_machine_outage_is_void_and_the_machine_stays_off(tmp_path):
    first = run_multi_tso_dso(_cfg(tmp_path / "a", 20.0))
    off = tuple(first[0].uc_offline_gens)
    assert off, "OP-R must have a machine off-line"
    assert first[0].uc_void_events == ()
    g = int(off[0])
    ev = [ContingencyEvent(minute=0, time_s=20.0, element_type="gen",
                           element_index=g, action="trip"),
          ContingencyEvent(minute=0, time_s=40.0, element_type="gen",
                           element_index=g, action="restore")]
    log = run_multi_tso_dso(_cfg(tmp_path / "b", 60.0, ev))
    assert tuple(log[0].uc_offline_gens) == off
    assert log[0].uc_void_events == ((20.0, g, "trip"), (40.0, g, "restore"))
    for r in log:                                   # never back in service
        for z, idx in r.zone_gen_idx.items():
            idx = np.asarray(idx)
            if g in idx:
                assert not bool(np.asarray(r.zone_gen_in_service[z])[idx == g][0])
