"""Case-2 MC on the commitment base (2026-11-07).

* An outage of a machine the unit commitment keeps off-line is void with its
  restore; the episode class is counted on the events that apply, the draw is
  unchanged.
* ``--buffer K`` runs the paired variants on the next K valid episodes after
  the first N from the start; the window and the counts are unchanged.
"""
import argparse
import json

from experiments.ch_11_case2_multisystem.mc import run_mc as M
from experiments.helpers.records import ContingencyEvent


def _gen(m, g, action):
    return ContingencyEvent(minute=m, time_s=60.0 * m + 40.0, element_type="gen",
                            element_index=g, action=action)


def _line(m, li, action):
    return ContingencyEvent(minute=m, time_s=60.0 * m + 40.0, element_type="line",
                            element_index=li, action=action)


def test_an_outage_of_an_offline_machine_is_void_with_its_restore():
    ev = [_gen(30, 5, "trip"), _line(60, 7, "trip"), _gen(120, 5, "restore"),
          _line(150, 7, "restore")]
    eff, void = M.void_events(ev, offline=[5, 8])
    assert [(e.element_type, e.action) for e in void] == [("gen", "trip"), ("gen", "restore")]
    assert [(e.element_type, e.action) for e in eff] == [("line", "trip"), ("line", "restore")]
    assert M.episode_class(ev) == "multiple" and M.episode_class(eff) == "single"


def test_a_committed_machine_and_other_kinds_are_untouched():
    ev = [_gen(30, 4, "trip"), _gen(120, 4, "restore"),
          ContingencyEvent(minute=60, time_s=3640.0, element_type="load", bus=3,
                           p_mw=0.0, q_mvar=200.0, action="connect")]
    eff, void = M.void_events(ev, offline=[5])
    assert void == [] and len(eff) == 3
    assert M.void_events(ev, offline=())[1] == []


def test_applied_events_drops_the_void_rows_and_reads_older_records():
    rows = M.events_to_rows([_gen(30, 5, "trip"), _line(60, 7, "trip")])
    pre = {"events": rows, "events_void": rows[:1]}
    assert M.applied_events(pre) == rows[1:]
    assert M.applied_events({"events": rows}) == rows          # before 2026-11-07
    assert "(void: off-line)" in M._event_text(pre)


def _fake_campaign(tmp_path, monkeypatch, buffer, variants=("S1", "O2")):
    """Run ``_simulate`` with the pool, the prescreen and the report stubbed;
    return the (seed, variant) tasks it submitted."""
    valid = [M.MASTER_SEED + i for i in range(40)]
    monkeypatch.setattr(M, "enumerate_elements",
                        lambda: {"gens": [], "lines": [], "load_buses": [],
                                 "lines_excluded_islanding": []})
    monkeypatch.setattr(M, "_prescreen_until",
                        lambda d, out, args, n_need: valid[:n_need])
    monkeypatch.setattr(M, "report", lambda *a, **k: None)
    sent = []

    def pool(fn, jobs, n, label):
        for s, v, out, _ in jobs:
            sent.append((s, v))
            (tmp_path / "runs" / f"{s}_{v}.json").write_text(json.dumps(
                {"seed": s, "variant": v, "outcome": "ok"}), encoding="utf-8")
        return []

    monkeypatch.setattr(M, "_pool_map", pool)
    args = argparse.Namespace(n=5, n_o3=0, minutes=10, jobs=2, also_seeds="",
                              rule=M.WINDOW_RULE, max_valid=M.N_VALID_MAX,
                              buffer=buffer)
    M._simulate(tmp_path, args, list(variants))
    return sent


def test_buffer_runs_the_next_valid_episodes_in_the_first_wave(tmp_path, monkeypatch):
    sent = _fake_campaign(tmp_path, monkeypatch, buffer=3)
    for v in ("S1", "O2"):
        assert sorted(s for s, w in sent if w == v) == \
            [M.MASTER_SEED + i for i in range(8)]


def test_without_buffer_only_the_window_runs(tmp_path, monkeypatch):
    sent = _fake_campaign(tmp_path, monkeypatch, buffer=0)
    assert sorted(s for s, w in sent if w == "S1") == \
        [M.MASTER_SEED + i for i in range(5)]


def test_buffer_does_not_move_the_window():
    valid = [M.MASTER_SEED + i for i in range(20)]
    out = {(s, v): "ok" for s in valid[:8] for v in M.ALL_VARIANTS}
    out[(valid[2], "S1")] = "fail_abort"                  # replaced by valid[5]
    window, skipped = M.comparison_window(valid, out, 5)
    assert window == [valid[i] for i in (0, 1, 3, 4, 5)] and skipped == [valid[2]]
