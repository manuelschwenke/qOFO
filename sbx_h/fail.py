"""
sbx_h/fail.py
===========
Fail-fast helper for the SBX package (plan v2 §1 hard rule 1).

``rep1()`` — *report once, raise* — is the single failure channel used
throughout ``sbx``.  Phase 0 reconnaissance established that no such
helper pre-exists in the repository ("``rep1()``-style assertions" in
``docs/BME_SPEC.md`` §8 is a convention, not code); creating it here was
approved at the Phase 0 gate (STATUS_SBX.md, assumption A1 / gate G1).

No silent defaults, no exception swallowing, no fallback values: every
violated precondition in ``sbx`` goes through :func:`rep1` with the
diagnostics needed to reproduce the failure.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-07 (SBX Phase 1)
"""

from __future__ import annotations

from typing import NoReturn


class SBXError(RuntimeError):
    """Raised by :func:`rep1` for every SBX precondition violation."""


def rep1(message: str, **diagnostics: object) -> NoReturn:
    """Raise :class:`SBXError` with the message and formatted diagnostics.

    Parameters
    ----------
    message :
        Precise statement of the violated precondition (British English,
        no marketing language).
    **diagnostics :
        Named values that make the failure reproducible (indices,
        measured values, brackets, solver statuses ...).  Rendered as
        ``key=value`` pairs appended to the message.

    Raises
    ------
    SBXError
        Always.  This function never returns.
    """
    if diagnostics:
        detail = ", ".join(f"{k}={v!r}" for k, v in sorted(diagnostics.items()))
        raise SBXError(f"{message} [{detail}]")
    raise SBXError(message)
