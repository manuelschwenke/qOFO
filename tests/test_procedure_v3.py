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


def test_screen_arguments_amendment2():
    out, cj = Path("results/x"), Path("results/x/c.json")
    for now in ("search", "complete", "bank"):
        V3.STAGE["now"] = now
        try:
            a = V3.screen_argv(out, cj, 4)
        finally:
            V3.STAGE["now"] = "search"
        assert a[a.index("--startup-exempt-s") + 1] == f"{V3.STARTUP_EXEMPT_S:g}"
        assert ("--abort-tap-reversals" in a) == (now == "search")
        if now == "bank":
            assert a[a.index("--windows") + 1] == "W1,W3,W4,W6"
    c = V3.screen_argv(out, cj, 2, windows="W5")
    assert "--startup-exempt-s" in c


def test_startup_exemption_of_the_first_reversal_only():
    from tuning.stage2_search import _split_reversals, hunting_reversals
    moves = [(180.0, "1[0]", -1), (4500.0, "1[0]", 1), (5000.0, "1[0]", -1),
             (2000.0, "2[0]", 1), (2600.0, "2[0]", -1)]
    hunt, legit = hunting_reversals(moves)
    assert len(hunt) == 3 and not legit                       # off by default
    hunt, legit, start = _split_reversals(moves, (), 7200.0, 1.0, 900.0)
    assert [r["t0_s"] for r in start] == [180.0]              # first pair, first move < 15 min
    assert sorted(r["t0_s"] for r in hunt) == [2000.0, 4500.0]  # 2[0] starts after 15 min
    hunt, legit, start = _split_reversals(moves, (3000.0,), 7200.0, 1.0, 900.0)
    assert [r["t0_s"] for r in legit] == [180.0] and not start  # an event explains it first


def test_stage2c_grid_and_pick():
    g = V3.bank_grid(4.266, 38.22)
    assert len(g) == V3.N_2C - 1 and abs(g[0] - 4.266) < 1e-9 and g[-1] < 38.22
    assert V3.bank_grid(4.266, 4.3) == []
    r = lambda k, p, e: {"key": k, "weights": {"g_w_tso_shunt": p}, "aggregate": {"e_v_ts": e}}
    ranked = [r("lead", 38.2, 0.00882), r("a", 9.0, 0.00905), r("b", 13.0, 0.00895),
              r("c", 20.0, 0.00880)]
    assert V3.pick_lowest_price(ranked, 0.00882)["key"] == "b"   # a is outside the 2 % band
