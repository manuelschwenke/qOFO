"""Capture and reuse one cached-model stability snapshot.

The expensive plant/sensitivity extraction runs once. Candidate G_w values
then re-precondition the same cached curvature without consulting the plant.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pickle
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from analysis.stability_analysis import (
    MultiZoneStabilityResult,
    analyse_multi_zone_stability,
)
from configs.config import MultiTSOConfig
from tuning._sim_loader import get_runner_module


CACHE_VERSION = 1
DEFAULT_CACHE_DIR = Path("results/stability_certificate/curvature_cache")
CONTINUOUS_BO_FIELDS = ("g_w_der", "g_w_pcc", "g_w_dso_der")


@dataclass(frozen=True)
class DSOModelSnapshot:
    dso_id: str
    H: np.ndarray
    Q: np.ndarray
    baseline_gw: np.ndarray
    gw_fields: tuple[str, ...]
    actuator_counts: dict[str, int]


@dataclass(frozen=True)
class CachedCurvatureSnapshot:
    cache_version: int
    cache_key: str
    generated_at: str
    zone_ids: tuple[int, ...]
    H_blocks: dict[tuple[int, int], np.ndarray]
    Q_obj_list: tuple[np.ndarray, ...]
    baseline_gw_list: tuple[np.ndarray, ...]
    tso_gw_fields: tuple[tuple[str, ...], ...]
    actuator_counts: tuple[dict[str, int], ...]
    dso_models: tuple[DSOModelSnapshot, ...]
    baseline_gw_parameters: dict[str, float]
    tso_period_s: float
    dso_period_s: float
    baseline_c3_gamma: float
    baseline_c3_certified: bool


def _config_cache_key(config: MultiTSOConfig) -> str:
    payload = dataclasses.asdict(config)
    for key in (
        "result_dir",
        "n_total_s",
        "run_stability_analysis",
        "stability_analysis_at_s",
        "verbose",
        "live_plot_controller",
        "live_plot_cascade",
        "live_plot_system",
        "live_plot_tracking",
        "live_plot_sbx",
        "contingencies",
    ):
        payload.pop(key, None)
    payload["snapshot_cache_version"] = CACHE_VERSION
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:20]


def _field_labels(counts: dict[str, int]) -> tuple[str, ...]:
    return (
        ("g_w_der",) * counts["n_der"]
        + ("g_w_pcc",) * counts["n_pcc"]
        + ("g_w_gen",) * counts["n_gen"]
        + ("g_w_tso_oltc",) * counts["n_oltc"]
        + ("g_w_tso_shunt",) * counts["n_shunt"]
    )


def _snapshot_from_runtime(
    result: MultiZoneStabilityResult,
    runtime: dict[str, Any],
    cache_key: str,
) -> CachedCurvatureSnapshot:
    config: MultiTSOConfig = runtime["config"]
    coordinator = runtime["coordinator"]
    zone_defs = runtime["zone_defs"]
    tso_controllers = runtime["tso_controllers"]
    dso_controllers = runtime["dso_controllers"]

    zone_ids = tuple(sorted(zone_defs))
    H_blocks = {
        key: np.asarray(coordinator.get_H_block(*key), dtype=float).copy()
        for key in coordinator._H_blocks
    }
    q_obj_list = tuple(
        np.asarray(zone_defs[zone].q_obj_diagonal(), dtype=float).copy()
        for zone in zone_ids
    )
    baseline_gw_list = tuple(
        np.asarray(tso_controllers[zone].params.g_w, dtype=float).copy()
        for zone in zone_ids
    )
    if any(abs(float(tso_controllers[zone].params.alpha) - 1.0) > 1e-12 for zone in zone_ids):
        raise ValueError("The stability BO requires the controller convention alpha = 1.")

    counts_list: list[dict[str, int]] = []
    labels_list: list[tuple[str, ...]] = []
    for zone in zone_ids:
        counts = {
            "n_der": len(zone_defs[zone].tso_der_indices),
            "n_pcc": len(zone_defs[zone].pcc_trafo_indices),
            "n_gen": len(zone_defs[zone].gen_indices),
            "n_oltc": len(zone_defs[zone].oltc_trafo_indices),
            "n_shunt": len(zone_defs[zone].shunt_bus_indices),
        }
        labels = _field_labels(counts)
        if len(labels) != len(tso_controllers[zone].params.g_w):
            raise ValueError(
                f"TSO zone {zone}: G_w dimension does not match actuator ordering."
            )
        counts_list.append(counts)
        labels_list.append(labels)

    dso_models: list[DSOModelSnapshot] = []
    for dso_id, controller in dso_controllers.items():
        dso_config = controller.config
        n_interfaces = len(dso_config.interface_trafo_indices)
        n_voltage = len(dso_config.voltage_bus_indices)
        n_current = len(dso_config.current_line_indices)
        Q = np.zeros(n_interfaces + n_voltage + n_current, dtype=float)
        Q[:n_interfaces] = float(config.g_q)
        if dso_config.v_setpoints_pu is not None and n_voltage:
            Q[n_interfaces : n_interfaces + n_voltage] = float(config.dso_g_v)
        H_bus = controller._build_sensitivity_matrix()
        H = np.asarray(controller._expand_H_to_der_level(H_bus), dtype=float)
        counts = {
            "n_der": len(dso_config.der_indices),
            "n_oltc": len(dso_config.interface_trafo_indices),
            "n_shunt": len(dso_config.shunt_bus_indices),
        }
        labels = (
            ("g_w_dso_der",) * counts["n_der"]
            + ("g_w_dso_oltc",) * counts["n_oltc"]
            + ("g_w_tso_shunt",) * counts["n_shunt"]
        )
        baseline_gw = np.asarray(controller.params.g_w, dtype=float).copy()
        if len(labels) != len(baseline_gw):
            raise ValueError(
                f"{dso_id}: G_w dimension does not match DSO actuator ordering."
            )
        if abs(float(controller.params.alpha) - 1.0) > 1e-12:
            raise ValueError(f"{dso_id}: alpha is not one.")
        dso_models.append(
            DSOModelSnapshot(
                dso_id=str(dso_id),
                H=H.copy(),
                Q=Q,
                baseline_gw=baseline_gw,
                gw_fields=labels,
                actuator_counts=counts,
            )
        )

    parameter_fields = {
        field
        for labels in labels_list
        for field in labels
    } | {
        field
        for model in dso_models
        for field in model.gw_fields
    }
    baseline_parameters = {
        field: float(getattr(config, field))
        for field in sorted(parameter_fields)
        if hasattr(config, field)
    }

    return CachedCurvatureSnapshot(
        cache_version=CACHE_VERSION,
        cache_key=cache_key,
        generated_at=datetime.now(timezone.utc).isoformat(),
        zone_ids=zone_ids,
        H_blocks=H_blocks,
        Q_obj_list=q_obj_list,
        baseline_gw_list=baseline_gw_list,
        tso_gw_fields=tuple(labels_list),
        actuator_counts=tuple(counts_list),
        dso_models=tuple(dso_models),
        baseline_gw_parameters=baseline_parameters,
        tso_period_s=float(config.tso_period_s),
        dso_period_s=float(config.dso_period_s),
        baseline_c3_gamma=float(result.c3_discrete.Gamma_spectral_radius),
        baseline_c3_certified=bool(result.c3_discrete.stable),
    )


def extract_cached_snapshot(config: MultiTSOConfig) -> CachedCurvatureSnapshot:
    """Run the short stability extraction and capture reusable matrices."""

    module = get_runner_module()
    original = module._run_delayed_stability_analysis
    cache_key = _config_cache_key(config)
    captured: list[CachedCurvatureSnapshot] = []

    def wrapper(*args: Any, **kwargs: Any) -> MultiZoneStabilityResult:
        result = original(*args, **kwargs)
        captured.append(_snapshot_from_runtime(result, kwargs, cache_key))
        return result

    with tempfile.TemporaryDirectory(prefix="ofo_curvature_snapshot_") as tmp:
        short_config = dataclasses.replace(
            config,
            n_total_s=float(config.tso_period_s) + 1.0,
            stability_analysis_at_s=0.0,
            run_stability_analysis=True,
            contingencies=[],
            verbose=0,
            live_plot_controller=False,
            live_plot_cascade=False,
            live_plot_system=False,
            load_tuned_params_path=None,
            result_dir=tmp,
        )
        module._run_delayed_stability_analysis = wrapper
        try:
            module.run_multi_tso_dso(short_config)
        finally:
            module._run_delayed_stability_analysis = original

    if not captured:
        raise RuntimeError("The one-step run did not produce a curvature snapshot.")
    return captured[0]


def load_or_extract_snapshot(
    config: MultiTSOConfig,
    *,
    use_cache: bool = True,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> CachedCurvatureSnapshot:
    """Load a local trusted cache entry or perform one extraction."""

    key = _config_cache_key(config)
    path = cache_dir / f"snapshot_{key}.pkl"
    if use_cache and path.exists():
        with path.open("rb") as handle:
            snapshot = pickle.load(handle)
        if (
            isinstance(snapshot, CachedCurvatureSnapshot)
            and snapshot.cache_version == CACHE_VERSION
            and snapshot.cache_key == key
        ):
            return snapshot

    snapshot = extract_cached_snapshot(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return snapshot


def _rescale_vector(
    baseline: np.ndarray,
    labels: tuple[str, ...],
    snapshot: CachedCurvatureSnapshot,
    candidate: MultiTSOConfig,
) -> np.ndarray:
    result = np.asarray(baseline, dtype=float).copy()
    for index, field in enumerate(labels):
        base_value = snapshot.baseline_gw_parameters[field]
        candidate_value = float(getattr(candidate, field))
        if base_value <= 0.0 or candidate_value <= 0.0:
            raise ValueError(f"{field} must remain strictly positive.")
        result[index] *= candidate_value / base_value
    return result


def candidate_gw_lists(
    snapshot: CachedCurvatureSnapshot,
    candidate: MultiTSOConfig,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Return candidate TSO and DSO G_w vectors with class scaling."""

    baseline_gen = snapshot.baseline_gw_parameters.get("g_w_gen")
    if baseline_gen is not None and not np.isclose(
        float(candidate.g_w_gen), baseline_gen, rtol=0.0, atol=0.0
    ):
        raise ValueError(
            "g_w_gen is fixed for stability BO and must equal the baseline value "
            f"{baseline_gen:g}."
        )

    tso = tuple(
        _rescale_vector(base, labels, snapshot, candidate)
        for base, labels in zip(
            snapshot.baseline_gw_list,
            snapshot.tso_gw_fields,
            strict=True,
        )
    )
    dso = tuple(
        _rescale_vector(model.baseline_gw, model.gw_fields, snapshot, candidate)
        for model in snapshot.dso_models
    )
    return tso, dso


def rebuild_stability_result(
    snapshot: CachedCurvatureSnapshot,
    candidate: MultiTSOConfig,
) -> MultiZoneStabilityResult:
    """Rebuild preconditioned curvature and all C1/C2/C3 diagnostics."""

    tso_gw, dso_gw = candidate_gw_lists(snapshot, candidate)
    dso_data = [
        {
            "H": model.H,
            "Q": model.Q,
            "G_w": gw,
            "id": model.dso_id,
            "alpha": 1.0,
            "actuator_counts": model.actuator_counts,
        }
        for model, gw in zip(snapshot.dso_models, dso_gw, strict=True)
    ]
    return analyse_multi_zone_stability(
        H_blocks=snapshot.H_blocks,
        Q_obj_list=list(snapshot.Q_obj_list),
        G_w_list=list(tso_gw),
        zone_ids=list(snapshot.zone_ids),
        zone_names=[f"Zone {zone}" for zone in snapshot.zone_ids],
        actuator_counts=list(snapshot.actuator_counts),
        alpha=1.0,
        verbose=False,
        dso_data=dso_data,
        tso_period_s=snapshot.tso_period_s,
        dso_period_s=snapshot.dso_period_s,
    )
