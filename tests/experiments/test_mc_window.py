"""The comparison window of the Case-2 MC (top-up, amendment 2026-10-30)."""
from experiments.ch_11_case2_multisystem.mc import run_mc as M

V = M.ALL_VARIANTS


def _all(seed, outcome):
    return {(seed, v): outcome for v in V}


def test_declared_rule_takes_the_first_n_valid_seeds():
    valid = [1, 2, 3, 4, 5]
    out = {**_all(2, "fail_abort")}
    assert M.comparison_window(valid, out, 3, top_up=False) == ([1, 2, 3], [])


def test_an_episode_all_six_failed_on_is_replaced_by_the_next():
    valid = [1, 2, 3, 4, 5, 6]
    out = {**_all(1, "ok"), **_all(2, "fail_abort"), **_all(3, "ok"),
           **_all(4, "fail_pf"), **_all(5, "ok")}
    assert M.comparison_window(valid, out, 3) == ([1, 3, 5], [2, 4])


def test_failures_of_single_variants_are_never_replaced():
    valid = [1, 2, 3]
    out = {**_all(1, "fail_abort"), (1, "M"): "ok", **_all(2, "ok")}
    assert M.comparison_window(valid, out, 2) == ([1, 2], [])


def test_an_episode_not_yet_run_by_every_variant_counts_as_usable():
    valid = [1, 2, 3]
    out = {**_all(1, "fail_abort")}
    del out[(1, "M")]                       # five failures, M still missing
    assert M.comparison_window(valid, out, 2) == ([1, 2], [])


def test_the_window_is_short_when_the_valid_seeds_run_out():
    valid = [1, 2]
    out = {**_all(1, "fail_abort")}
    assert M.comparison_window(valid, out, 2) == ([2], [1])
