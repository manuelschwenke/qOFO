"""The central controller keeps the bank-column V^2 rescaling and its 3W columns."""

import numpy as np

from controller.central_controller import _AppendFrozenColumns


class _Inner:
    def __init__(self):
        self.calls = 0
        self.n_shunts = 1

    def update(self, measurement, current_iteration):
        self.calls += 1
        return np.full((2, 3), float(current_iteration))


def test_update_rescales_the_base_and_reappends_the_frozen_columns():
    extra = np.array([[7.0], [8.0]])
    w = _AppendFrozenColumns(_Inner(), extra)
    H = w.update(None, 4)
    assert H.shape == (2, 4)
    np.testing.assert_array_equal(H[:, :3], 4.0)
    np.testing.assert_array_equal(H[:, 3:], extra)
    assert w.n_shunts == 1                       # other attributes pass through


def test_ts_devices_keep_the_cascade_lock_and_coupler_taps_the_global_one(monkeypatch):
    """A switched transmission tap or bank is locked ts_discrete_cooldown steps."""
    from types import SimpleNamespace

    from controller import central_controller as cc_mod
    from controller.tso_controller import TSOController

    # layout [Q_DER (2) | V_gen (1) | OLTC2w (1) | shunt (1) | OLTC3w (1)]
    c = object.__new__(cc_mod.CentralOFOController)
    c.config = SimpleNamespace(oltc_trafo_indices=[7], shunt_bus_indices=[9],
                               oltc_trafo3w_indices=[11])
    c.ts_discrete_cooldown = 53
    c.iteration = 10
    c._u_current = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    c._int_lock_until = {}
    monkeypatch.setattr(type(c), "n_controls", property(lambda self: 6))

    def fake_parent_step(self, measurement, sim_time_s=None):
        # the base controller's own bookkeeping: every switch locked c+1 ahead
        new = self._u_current.copy()
        new[3] += 1          # MT tap
        new[4] += 1          # bank
        new[5] += 1          # coupler tap
        for idx in (3, 4, 5):
            self._int_lock_until[idx] = self.iteration + 1 + 5
        self._u_current = new
        self.iteration += 1
        return "out"

    monkeypatch.setattr(TSOController, "step", fake_parent_step)
    assert c.ts_discrete_indices() == [3, 4]
    assert c.step(None, sim_time_s=0.0) == "out"
    assert c._int_lock_until[3] == 11 + 53       # free after (1+53) x 20 s = 18 min
    assert c._int_lock_until[4] == 11 + 53
    assert c._int_lock_until[5] == 10 + 1 + 5    # coupler tap: STS timescale
