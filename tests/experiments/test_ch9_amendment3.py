"""Amendment 3 re-selection (ch_9_7_amendment3): engagement band, host split and
the take-over of finished settling checks."""
import json

import pytest

from experiments.ch_09_parameter_selection import ch_9_7_amendment3 as A


def test_the_band_is_closed_at_both_ends():
    assert A.admitted(0.02) and A.admitted(0.04) and A.admitted(0.0345)
    assert not A.admitted(0.0199) and not A.admitted(0.0401)


def test_hosts_alternate_so_a_short_batch_uses_both():
    assert A.hosts(2, 3) == ["local", "remote", "local", "remote", "remote"]
    assert A.hosts(2, 0) == ["local", "local"]
    assert A.hosts(0, 1) == ["remote"]


def _source_check(key):
    d = A.SOURCE / "expC" / f"cand_{key}"
    rd = A._latest(d)
    if rd is None or not (rd / "steps.csv").exists():
        pytest.skip(f"source run check of {key} not available here")
    w = json.loads((A.SOURCE / "expC" / f"cand_{key}_weights.json").read_text("utf-8"))
    return rd, w


def test_a_finished_check_of_the_same_weights_is_the_same_test():
    rd, w = _source_check("dd14ab9e4194")
    assert A.same_test(rd, w, A.reference_args())


def test_other_weights_are_not_the_same_test():
    rd, w = _source_check("dd14ab9e4194")
    assert not A.same_test(rd, {**w, "g_q": w["g_q"] * 1.001}, A.reference_args())


def test_other_test_arguments_are_not_the_same_test():
    rd, w = _source_check("dd14ab9e4194")
    ref = {**A.reference_args(), "max_step_mvar": 5.0}
    assert not A.same_test(rd, w, ref)
