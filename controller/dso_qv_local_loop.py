"""
controller/dso_qv_local_loop.py — back-compat shim (refactor_v2 commit 3)
========================================================================

The plant-side Q(V) controller was renamed to
:mod:`controller.der_qv_local_loop` because it now serves both TSO- and
DSO-connected DERs (and now also dispatches the new
:class:`CosPhiConstLoop` for ``q_mode == "cosphi"``).

This module is preserved as a thin re-export so any caller that has not
yet migrated to the new path keeps working.  Slated for removal in
refactor_v2 commit 7.

New code should ``from controller.der_qv_local_loop import ...``.
"""

from controller.der_qv_local_loop import (  # noqa: F401
    CosPhiConstLoop,
    QVLocalLoop,
    _qv_capability,
    cache_per_sgen_svq,
    install_der_q_loops,
    install_qv_local_loops,
    remove_der_q_loops,
    remove_qv_local_loops,
)
