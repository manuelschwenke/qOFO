"""
tuning/_io.py
=============
YAML and JSON helpers for the tuning module.

Kept private; callers should go through the :mod:`tuning.tune` and
:mod:`tuning.validate` CLIs (or :mod:`tuning.reports`).
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from configs.config import MultiTSOConfig


# ---------------------------------------------------------------------------
# YAML load / save for MultiTSOConfig
# ---------------------------------------------------------------------------

def save_config_yaml(cfg: MultiTSOConfig, path: Path) -> None:
    """Save a :class:`MultiTSOConfig` to YAML.

    Non-trivial fields (datetime, list-of-:class:`ContingencyEvent`) are
    serialised as ISO strings / dicts respectively.  Numpy scalars are
    coerced to Python types via :func:`jsonable`.
    """
    d = dataclasses.asdict(cfg)
    if "start_time" in d and isinstance(d["start_time"], datetime):
        d["start_time"] = d["start_time"].isoformat()
    d = jsonable(d)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(d, f, sort_keys=True, default_flow_style=False)


#: Config fields holding a nested dataclass.  ``dataclasses.asdict`` flattens
#: these to plain dicts on save, and nothing reconstructed them on load, so a
#: save/load round-trip silently replaced e.g. ``cfg.sbx_config`` (an
#: ``SBXConfig``) with a ``dict`` — attribute access on it then raises
#: ``AttributeError`` deep inside a run.  ``sbx_config`` is declared
#: ``Optional[object]`` and ``measurement_noise`` is only reachable through a
#: default factory, so the type cannot be recovered from the annotation alone;
#: it is named explicitly here.
_NESTED_DATACLASS_FIELDS: dict[str, str] = {
    "measurement_noise": "configs.config:MeasurementNoiseConfig",
    "sbx_config":        "sbx_h.config:SBXConfig",
}

#: Fields declared as tuples.  YAML round-trips them as lists, and code that
#: relies on hashability or on ``isinstance(x, tuple)`` would then misbehave.
_TUPLE_FIELDS: frozenset[str] = frozenset({"precondition_exclude_classes"})

#: Mapping fields whose **keys** are not strings.  :func:`jsonable` stringifies
#: every dict key (``str(k)``), so a save/load round-trip turns ``1`` into
#: ``"1"`` and ``(14, 38)`` into ``"(14, 38)"``.  Every consumer looks these up
#: with the original key type and **fails silently** on a miss:
#:
#: * ``tie_thevenin_k`` -- ``build_tso_local_net._k_for`` looks up
#:   ``(int(line_idx), int(far_bus))``; a miss falls back to
#:   ``THEVENIN_K_DEFAULT``, so a YAML baseline that names the *measured*
#:   per-corridor impedances would quietly tune a different boundary model
#:   than the one it declares (measured 2026-08-13).
#: * ``zone_*`` / ``q_pcc_setpoints_mvar_per_dso`` -- looked up with ``int(z)``
#:   (e.g. ``multi_tso_dso.py`` ``zone_g_w_scale.get(int(_z), 1.0)``), so the
#:   override is dropped and the scalar default silently applies instead.
#:
#: Values are the key parsers; unparseable keys are left as-is rather than
#: raising, so an unrelated hand-written YAML cannot be made unloadable by this.
_INT_KEY_FIELDS: frozenset[str] = frozenset({
    "zone_v_setpoints_pu", "zone_v_min_pu", "zone_v_max_pu", "zone_g_v",
    "zone_g_z_voltage", "zone_g_w_scale", "zone_g_w_class",
    "zone_tso_g_res_sg",
    "zone_tso_g_res_der", "zone_tso_g_loss",
    "der_q_mode_overrides", "der_qv_vref_pu_overrides",
    "der_qv_slope_pu_overrides", "der_qv_deadband_pu_overrides",
    "der_cosphi_overrides", "der_cosphi_sign_overrides",
})

#: Mapping fields keyed by a tuple of ints, e.g. ``(line_idx, far_end_bus)``.
_TUPLE_INT_KEY_FIELDS: frozenset[str] = frozenset({"tie_thevenin_k"})


def _parse_int_key(key: Any) -> Any:
    if isinstance(key, str):
        try:
            return int(key)
        except ValueError:
            return key
    return key


def _parse_tuple_int_key(key: Any) -> Any:
    if not isinstance(key, str):
        return key
    text = key.strip()
    if not (text.startswith("(") and text.endswith(")")):
        return key
    try:
        return tuple(int(part) for part in text[1:-1].split(","))
    except ValueError:
        return key


def _restore_dict_keys(d: dict[str, Any]) -> None:
    """Undo :func:`jsonable`'s key stringification, in place."""
    for name in _INT_KEY_FIELDS:
        value = d.get(name)
        if isinstance(value, dict):
            d[name] = {_parse_int_key(k): v for k, v in value.items()}
    for name in _TUPLE_INT_KEY_FIELDS:
        value = d.get(name)
        if isinstance(value, dict):
            d[name] = {_parse_tuple_int_key(k): v for k, v in value.items()}

#: Fields removed from :class:`MultiTSOConfig` that may still appear in YAMLs
#: written by earlier runs.  ``MultiTSOConfig(**d)`` would raise ``TypeError``
#: on them, so they are dropped on load and old study directories / archived
#: baselines stay readable.
#: 2026-07-31 -- integral Q-tracking removed from the DSO controller.
_RETIRED_FIELDS: frozenset[str] = frozenset({
    "dso_g_qi",
    "dso_lambda_qi",
    "dso_q_integral_max_mvar",
})


def _rebuild_nested(name: str, value: Any) -> Any:
    """Rebuild a nested dataclass from its ``asdict`` form, if possible."""
    if not isinstance(value, dict):
        return value
    target = _NESTED_DATACLASS_FIELDS.get(name)
    if target is None:
        return value
    module_name, _, cls_name = target.partition(":")
    try:
        import importlib

        cls = getattr(importlib.import_module(module_name), cls_name)
        return cls(**value)
    except Exception:
        # Never let a provenance detail break loading a config; the round-trip
        # test in tests/tuning/test_io.py is what guards this path.
        return value


def load_config_yaml(path: Path) -> MultiTSOConfig:
    """Load a :class:`MultiTSOConfig` from YAML.

    Reverses :func:`save_config_yaml`.  Reconstructs ``datetime``,
    :class:`ContingencyEvent`, nested dataclasses
    (:data:`_NESTED_DATACLASS_FIELDS`), tuple-typed fields
    (:data:`_TUPLE_FIELDS`) and non-string mapping keys
    (:data:`_INT_KEY_FIELDS`, :data:`_TUPLE_INT_KEY_FIELDS`).  Retired fields
    (:data:`_RETIRED_FIELDS`) are dropped rather than passed to the
    constructor.
    """
    from experiments.helpers.records import ContingencyEvent

    with path.open("r") as f:
        d = yaml.safe_load(f)

    d = {k: v for k, v in d.items() if k not in _RETIRED_FIELDS}

    if "start_time" in d and isinstance(d["start_time"], str):
        d["start_time"] = datetime.fromisoformat(d["start_time"])
    if "contingencies" in d and isinstance(d["contingencies"], list):
        d["contingencies"] = [
            ContingencyEvent(**c) if isinstance(c, dict) else c
            for c in d["contingencies"]
        ]
    for name in _NESTED_DATACLASS_FIELDS:
        if name in d:
            d[name] = _rebuild_nested(name, d[name])
    for name in _TUPLE_FIELDS:
        if name in d and isinstance(d[name], list):
            d[name] = tuple(d[name])
    _restore_dict_keys(d)

    return MultiTSOConfig(**d)


# ---------------------------------------------------------------------------
# Tuned-params YAML (BO 8-dim subset only)
# ---------------------------------------------------------------------------

def save_tuned_params(
    params: dict[str, float],
    meta: dict[str, Any],
    path: Path,
) -> None:
    """Save the BO-tuned 8-dim params plus study metadata to YAML.

    Schema::

        params:
          g_v: 12345.0
          g_q: 200.0
          ...
        meta:
          study_name: ...
          n_trials: ...
          best_value: ...
          ceilings: {g_w_der: ..., ...}
          timestamp: ...
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": jsonable(dict(params)),
        "meta":   jsonable(dict(meta)),
    }
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=True, default_flow_style=False)


def load_tuned_params(
    path: Path,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Load tuned params; returns ``(params, meta)``."""
    with path.open("r") as f:
        payload = yaml.safe_load(f)
    return dict(payload["params"]), dict(payload.get("meta", {}))


# ---------------------------------------------------------------------------
# JSON helpers (for trial-level diagnostics)
# ---------------------------------------------------------------------------

def jsonable(obj: Any) -> Any:
    """Recursively convert numpy / dataclass / datetime objects into
    types that :mod:`json` and :mod:`yaml` can serialise."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj
