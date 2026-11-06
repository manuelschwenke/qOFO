"""Scenario F / FM (declared 2026-11-05): E's first two events, then two load steps at two buses."""
from experiments.ch_11_case2_multisystem.ref import scenarios as sc


def test_F_extends_E_with_two_distinct_steps():
    e = sc.events("E")
    for name, q in (("F", 400.0), ("FM", 350.0)):
        ev = sc.events(name)
        assert [x.effective_time_s for x in ev[:2]] == [x.effective_time_s for x in e[:2]]
        steps = [x for x in ev if x.element_type == "load"]
        assert [x.q_mvar for x in steps] == [q, q] and [x.bus for x in steps] == [3, 7]
        # connect events with equal (bus, P, Q) would share one dormant load
        assert len({(x.bus, x.p_mw, x.q_mvar) for x in steps}) == 2
        assert steps[0].effective_time_s == e[2].effective_time_s
        assert steps[1].effective_time_s == 270 * 60 + 60   # 271:10 was off the step grid
        assert len({x.effective_time_s % sc.T_TS for x in ev}) == len(ev)   # distinct offsets
        assert sc.start_time(name) is None                                   # reference period


def test_every_scheduled_event_lies_on_the_step_grid():
    # The runner fires an event only at a step time; an off-grid event is silently skipped.
    for name in sc.SCENARIOS:
        for ev in sc.events(name):
            assert ev.effective_time_s % sc.DT_S == 0.0, (name, ev.effective_time_s)


def test_at_rejects_off_grid_times():
    import pytest
    with pytest.raises(ValueError, match="step grid"):
        sc._at(270, 70)
