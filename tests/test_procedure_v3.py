"""Procedure v3 (ch_9_11): engagement measure and the routing of screen arguments."""
from pathlib import Path

from experiments.ch_09_parameter_selection import ch_9_11_procedure_v3 as V3


def _row(dso_moves=(), shunt_moves=(), events=()):
    moves = [[t, "DSO_1|trafo_2[0]", 1] for t in dso_moves] + [[60.0, "1[0]", 1]]
    return {"moves": moves, "shunt_moves": [[t, "z2[0]", 1] for t in shunt_moves],
            "events_s": list(events)}


def test_engagement_counts_couplers_per_window_and_banks_after_events():
    pw = {w: _row(dso_moves=[100.0]) for w in V3.SEARCH}
    pw["W1"] = _row(dso_moves=[100.0], shunt_moves=[1000.0 + 3600.0], events=[1000.0])
    pw["W6"] = _row(dso_moves=[100.0], shunt_moves=[1000.0 + 7300.0], events=[1000.0])
    e = V3.engagement({"per_window": pw})
    assert e["sts_all"] is True
    assert e["bank_after_windows"] == 1          # W6's commit comes 2 h 1 min after
    assert e["bank"] is False
    pw["W4"] = _row(dso_moves=[100.0], shunt_moves=[500.0], events=[400.0])
    assert V3.engagement({"per_window": pw})["bank"] is True
    pw["W3"] = _row()                            # no coupler move in W3
    assert V3.engagement({"per_window": pw})["sts_all"] is False


def test_screen_arguments_by_stage():
    out, cj = Path("results/x"), Path("results/x/c.json")
    a = V3.screen_argv(out, cj, 4)
    assert a[a.index("--windows") + 1] == "W1,W3,W4,W6"
    assert a[a.index("--abort-tap-reversals") + 1] == str(V3.TOLERANCE + 1)
    assert a[a.index("--abort-bank-reversals") + 1] == "1"
    V3.STAGE["now"] = "complete"
    try:
        b = V3.screen_argv(out, cj, 4)
    finally:
        V3.STAGE["now"] = "search"
    assert b[b.index("--windows") + 1] == "W1,W2,W3,W4,W6"
    assert "--abort-tap-reversals" not in b
    c = V3.screen_argv(out, cj, 2, windows="W5")
    assert c[c.index("--windows") + 1] == "W5" and "--abort-tap-reversals" not in c
