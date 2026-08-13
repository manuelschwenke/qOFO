"""
tuning/compat.py
================
Bridges between historical Optuna studies and the current decision space.

Why this exists
---------------
The persisted study database (``results/tuning/studies.db``) holds 19 studies
and ~1555 scenario-runs.  Every IEEE-39 study among them records a **9th**
parameter, ``tso_g_q_tie``, which:

* is **not** in :data:`tuning.parameters.BO_DIMS` any more, and
* is **not** a field of :class:`~configs.config.MultiTSOConfig` at all --
  the surviving field is ``tso_g_q_pcc``.

So replaying any of those studies' best-params through
:func:`tuning.parameters.apply_to_config` raises
``ValueError: Unknown BO params: ['tso_g_q_tie']``.

The tempting fix -- quietly dropping unknown keys -- would hide the fact that a
tuned value is being discarded, which is exactly the kind of silent degradation
that produced the defects this module documents.  Instead, retired parameters
are listed explicitly and :func:`sanitize_legacy_params` *returns* what it
dropped so the caller can report it.

Use this **only** in reporting / replay tooling.  The live objective must keep
using :func:`tuning.parameters.apply_to_config` unchanged, so that a genuinely
unknown key remains a hard error.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "LEGACY_PARAM_ALIASES",
    "LEGACY_FINGERPRINT",
    "sanitize_legacy_params",
]


#: Parameters that appear in historical studies but not in the current space.
#: ``None`` means "retired, no replacement" -- the value is dropped.  A string
#: value would mean "renamed to this field" and the value carried over.
LEGACY_PARAM_ALIASES: dict[str, str | None] = {
    # Present in every IEEE-39 study in studies.db.  Never a MultiTSOConfig
    # field under this name; ``tso_g_q_pcc`` is the surviving TSO
    # interface-Q objective weight, but the two are not interchangeable
    # (different normalisation), so the recorded values are not carried over.
    "tso_g_q_tie": None,
}


#: Stamped onto studies that predate search-space fingerprinting, so the
#: resume guard in :func:`tuning.tune.main` refuses them by construction.
LEGACY_FINGERPRINT = "LEGACY_PRE_VERSIONING"


def sanitize_legacy_params(
    params: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Split a historical param dict into (usable, dropped).

    Returns
    -------
    clean
        Keys that still exist in the current space, plus any renamed ones
        remapped to their current names.
    dropped
        Keys that were retired or are unknown, with their original values.
        **Report these.**  A caller that ignores the second element is
        reintroducing the silent-degradation problem this module exists to
        prevent.

    Examples
    --------
    >>> clean, dropped = sanitize_legacy_params(
    ...     {"g_v": 1e4, "tso_g_q_tie": 0.42})
    >>> sorted(clean)
    ['g_v']
    >>> dropped
    {'tso_g_q_tie': 0.42}
    """
    from tuning.parameters import BO_DIMS

    current = {p.name for p in BO_DIMS}
    clean: dict[str, float] = {}
    dropped: dict[str, Any] = {}

    for key, value in params.items():
        if key in current:
            clean[key] = float(value)
            continue
        if key in LEGACY_PARAM_ALIASES:
            target = LEGACY_PARAM_ALIASES[key]
            if target is None or target not in current:
                dropped[key] = value
            else:
                clean[target] = float(value)
            continue
        dropped[key] = value

    return clean, dropped
