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
