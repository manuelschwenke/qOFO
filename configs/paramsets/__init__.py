"""Named parameter sets: JSON overlays on :func:`MultiTSOConfig`.

Why these are data and not functions
------------------------------------
``experiments/run_multi_system_ofo.py`` used to carry four config factories
(``make_config``, ``make_config_tuned``, ``make_config_dso_oltc_active``,
``make_config_per_area``).  Three of them were the same config with a
different *weight set*, and each re-stated the whole block, so a change to a
shared setting had to be made four times and usually was not.  Worse, each
one applied the DSO voltage relief itself, which made the call ORDER
load-bearing: :func:`configs.config.apply_dso_v_relief` read an existing
per-area ``dso_oltc`` entry as its base, so calling it twice squared the
factor -- a trap two of the factories had to work around by hand.

A parameter set is therefore a JSON file holding only the fields that
*differ* from the runner configuration, plus its provenance.  The runner
stays the single definition of everything else, and the relief is now a
config field (:attr:`MultiTSOConfig.dso_v_relief_factors`) re-derived on
every copy, so there is no ordering left to get wrong.

Format
------
::

    {
      "name": "tuned",
      "title": "one line",
      "description": ["free text", "..."],
      "provenance": {...},          # optional, not applied
      "fields": {"g_w_der": 10.2, ...}
    }

``fields`` keys must be :class:`MultiTSOConfig` field names; the loader
rejects anything else rather than silently dropping it.  JSON object keys are
always strings, so the per-zone maps (``zone_g_w_class``,
``zone_v_setpoints_pu``) are coerced back to integer keys, and
``precondition_exclude_classes`` back to a tuple.

Usage
-----
::

    python experiments/run_multi_system_ofo.py --params tuned

    from configs.paramsets import apply_paramset, available
    cfg = apply_paramset(make_config(), "tuned")

Author: Manuel Schwenke / Claude Code (2026-08-21)
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List

PARAMSET_DIR = Path(__file__).resolve().parent

#: Fields whose JSON object keys are integers in the dataclass.  JSON has no
#: integer keys, so they arrive as strings and would silently never match a
#: zone id.
_INT_KEYED_FIELDS = ("zone_g_w_class", "zone_v_setpoints_pu",
                     "zone_g_w_scale")

#: Fields the dataclass declares as tuples.
_TUPLE_FIELDS = ("precondition_exclude_classes",)


def available() -> List[str]:
    """Names of every parameter set on disk, sorted."""
    return sorted(p.stem for p in PARAMSET_DIR.glob("*.json"))


def load_paramset(name: str) -> Dict[str, Any]:
    """Parse one parameter set and return the whole document."""
    path = PARAMSET_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(
            f"unknown parameter set {name!r}; available: "
            f"{', '.join(available()) or '(none)'}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc.get("fields"), dict):
        raise SystemExit(f"{path}: no 'fields' object")
    return doc


def _coerce(field_name: str, value: Any) -> Any:
    if value is None:
        return None
    if field_name in _INT_KEYED_FIELDS and isinstance(value, dict):
        return {int(k): v for k, v in value.items()}
    if field_name in _TUPLE_FIELDS and isinstance(value, list):
        return tuple(value)
    return value


def apply_paramset(cfg, name: str, *, verbose: bool = True):
    """Return ``cfg`` with the named parameter set applied.

    ``dataclasses.replace`` is used rather than in-place assignment so that
    :meth:`MultiTSOConfig.__post_init__` re-derives everything that depends on
    the overridden weights -- in particular the per-DSO voltage relief, whose
    ``dso_g_v`` and ``g_w_dso_oltc`` bases a parameter set is very likely to
    move.  That re-derivation is idempotent, so it cannot compound.
    """
    doc = load_paramset(name)
    fields = doc["fields"]
    known = {f.name for f in dataclasses.fields(cfg)}
    unknown = sorted(set(fields) - known)
    if unknown:
        raise SystemExit(
            f"parameter set {name!r}: not MultiTSOConfig fields: "
            f"{', '.join(unknown)}")
    overlay = {k: _coerce(k, v) for k, v in fields.items()}
    out = dataclasses.replace(cfg, **overlay)
    if verbose:
        title = doc.get("title", name)
        print(f"[paramset] {name}: {title}")
        print(f"[paramset] {len(overlay)} field(s) overridden: "
              f"{', '.join(sorted(overlay))}")
    return out
