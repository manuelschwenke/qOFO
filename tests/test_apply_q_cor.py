"""
Deprecated: the Q_cor horizontal-shift actuator has been replaced by the
w-shift (vertical shift + V_ref reanchoring) actuator.  All tests in
this file have been migrated to :mod:`tests.test_apply_q_set`.

This stub remains so pytest collection does not error on a stale
filename; the active suite is :mod:`tests.test_apply_q_set`.
"""

import pytest

pytest.skip(
    "Superseded by tests/test_apply_q_set.py (w-shift replaces Q_cor).",
    allow_module_level=True,
)
