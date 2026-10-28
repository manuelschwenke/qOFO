"""The event phase of the MC (2026-10-28) leaves the declared draw unchanged."""
import numpy as np

from experiments.ch_11_case2_multisystem.mc import run_mc as M


def _key(e):
    return (e.minute, e.element_type, getattr(e, "element_index", None),
            getattr(e, "bus", None), getattr(e, "q_mvar", None), e.action)


def test_phase_draw_keeps_the_declared_episodes():
    el = M.enumerate_elements()
    for seed in range(M.MASTER_SEED, M.MASTER_SEED + 40):
        rng = np.random.default_rng(seed)
        t_old = M.draw_start(rng)
        old = M.draw_events(rng, el)                      # the declared draw
        t_new, new = M.draw_episode(seed)
        assert t_new == t_old
        assert sorted(map(_key, new)) == sorted(map(_key, old))
        on = {}
        for e in new:
            ph = e.effective_time_s - 60.0 * e.minute
            assert ph in M.PHASE_GRID_S                   # never on a dispatch
            k = (e.element_type, getattr(e, "element_index", None), getattr(e, "bus", None))
            if e.action in ("trip", "connect") and k not in on:
                on[k] = ph
            else:                                         # the reversal keeps it
                assert ph == on[k]


def test_case2_configuration_and_noise_seed():
    cfg = M._config("O3", M.PROFILE_START, [], M.Path("."), 10, noise_seed=123)
    assert cfg.measurement_noise.seed == 123
    assert cfg.measurement_noise.apply_to_svr and cfg.measurement_noise.apply_to_der_voltage
    assert set(M.ALL_VARIANTS) == set(M.VARIANTS)
