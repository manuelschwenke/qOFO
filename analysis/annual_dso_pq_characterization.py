"""
Annual P-Q characterization of the IEEE39-underlaid synthetic DS models.

The study runs four electrically independent AC power flows per native
15-minute profile sample, one for each DSO.  Every 345 kV primary terminal is
represented by a stiff 1.03 p.u. source, with equal distributed-slack weights
over the three sources belonging to one DSO.
All distribution-system DER retain their exogenous active-power profiles.
The DSO coupling-transformer OLTCs regulate their 110 kV terminals around
1.03 p.u. with pandapower ``DiscreteTapControl``.  DER reactive power is
either zero (unity power factor) or follows a selected fixed inductive power
factor.

Sign convention
---------------
``p_interface_hv_mw`` and ``q_interface_hv_mvar`` are sums of pandapower
``res_trafo3w.p_hv_mw`` and ``res_trafo3w.q_hv_mvar`` over the three coupling
transformers of one DSO.  Positive values therefore denote import from the
stiff primary sources into the distribution system.

Outputs
-------
``annual_pq_timeseries.csv``
    Full time-series results, one row per timestamp and DSO.
``annual_pq_scatter.csv``
    Compact combined interface-P/Q table.
``pq_scatter_DSO_<n>.csv``
    Two-column, header-bearing files intended for direct PGFPlots/TikZ use.
``annual_pq_summary.csv``
    Per-DSO extrema, quantiles, import/export energy and voltage statistics.
``annual_pq_characterization.png``
    Diagnostic scatter plot with common axes.
``run_metadata.json`` and ``README.md``
    Reproducibility metadata and sign/column documentation.

Run
---
``python -m analysis.annual_dso_pq_characterization``
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.control import DiscreteTapControl
from pandapower.control.run_control import (
    ctrl_variables_default,
    run_control,
)

from core.profiles import DEFAULT_PROFILES_CSV, load_profiles, snapshot_base_values
from network.ieee39.dso_overrides import (
    apply_dso_overrides as _apply_dso_overrides,
)
from network.ieee39 import add_hv_networks, build_ieee39_net
from network.ieee39.scenarios import SCENARIO_REGISTRY


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "annual_dso_pq_characterization_isolated"
)
DEFAULT_OLTC_VREF_PU = 1.03
DEFAULT_PRIMARY_VM_PU = 1.03
SUPPORTED_DER_COSPHI: Tuple[float, ...] = (1.0, 0.98, 0.95)
DER_COSPHI_SIGN = -1
StudyCase = Tuple[pp.pandapowerNet, object]

RESULT_FIELDS: Tuple[str, ...] = (
    "p_interface_hv_mw",
    "q_interface_hv_mvar",
    "p_load_mw",
    "q_load_mvar",
    "p_der_mw",
    "q_der_mvar",
    "p_net_demand_mw",
    "q_net_demand_mvar",
    "p_loss_including_couplers_mw",
    "q_passive_network_mvar",
    "v_min_pu",
    "v_mean_pu",
    "v_max_pu",
    "line_loading_max_percent",
    "coupler_loading_max_percent",
    "system_v_min_pu",
    "system_v_max_pu",
    "p_coupler_1_mw",
    "q_coupler_1_mvar",
    "p_coupler_2_mw",
    "q_coupler_2_mvar",
    "p_coupler_3_mw",
    "q_coupler_3_mvar",
    "tap_coupler_1",
    "tap_coupler_2",
    "tap_coupler_3",
)
FIELD_POS = {name: i for i, name in enumerate(RESULT_FIELDS)}


@dataclass(frozen=True)
class ProfileApplicationMap:
    """Vectorized equivalent of ``core.profiles.apply_profiles``."""

    profile_columns: Tuple[str, ...]
    load_p_profile_pos: np.ndarray
    load_q_profile_pos: np.ndarray
    sgen_p_profile_pos: np.ndarray
    load_base_p_mw: np.ndarray
    load_base_q_mvar: np.ndarray
    sgen_base_p_mw: np.ndarray
    dso_sgen_mask: np.ndarray

    @staticmethod
    def _positions(
        labels: Iterable[object],
        column_pos: Mapping[str, int],
    ) -> np.ndarray:
        return np.asarray(
            [
                column_pos.get(str(label), -1)
                if pd.notna(label)
                else -1
                for label in labels
            ],
            dtype=int,
        )

    @classmethod
    def from_net(
        cls,
        net: pp.pandapowerNet,
        profiles: pd.DataFrame,
        dso_sgen_indices: Sequence[int],
    ) -> "ProfileApplicationMap":
        column_pos = {str(col): i for i, col in enumerate(profiles.columns)}
        sgen_index_pos = {
            int(index): pos for pos, index in enumerate(net.sgen.index)
        }
        dso_sgen_mask = np.zeros(len(net.sgen), dtype=bool)
        for index in dso_sgen_indices:
            dso_sgen_mask[sgen_index_pos[int(index)]] = True

        def labels(table: pd.DataFrame, column: str) -> Sequence[object]:
            if column in table.columns:
                return table[column].tolist()
            return [np.nan] * len(table)

        return cls(
            profile_columns=tuple(str(c) for c in profiles.columns),
            load_p_profile_pos=cls._positions(
                labels(net.load, "profile_p"), column_pos
            ),
            load_q_profile_pos=cls._positions(
                labels(net.load, "profile_q"), column_pos
            ),
            sgen_p_profile_pos=cls._positions(
                labels(net.sgen, "profile"), column_pos
            ),
            load_base_p_mw=net.load["base_p_mw"].to_numpy(dtype=float),
            load_base_q_mvar=net.load["base_q_mvar"].to_numpy(dtype=float),
            sgen_base_p_mw=net.sgen["base_p_mw"].to_numpy(dtype=float),
            dso_sgen_mask=dso_sgen_mask,
        )

    @staticmethod
    def _scaled(
        base: np.ndarray,
        profile_pos: np.ndarray,
        profile_row: np.ndarray,
    ) -> np.ndarray:
        factor = np.ones(len(base), dtype=float)
        mask = profile_pos >= 0
        factor[mask] = profile_row[profile_pos[mask]]
        return base * factor

    def apply(
        self,
        net: pp.pandapowerNet,
        profile_row: np.ndarray,
        *,
        der_cosphi: float,
        der_cosphi_sign: int = DER_COSPHI_SIGN,
    ) -> None:
        """Apply one profile row and the selected fixed DSO-DER power factor."""
        net.load.loc[:, "p_mw"] = self._scaled(
            self.load_base_p_mw, self.load_p_profile_pos, profile_row
        )
        net.load.loc[:, "q_mvar"] = self._scaled(
            self.load_base_q_mvar, self.load_q_profile_pos, profile_row
        )
        p_sgen = self._scaled(
            self.sgen_base_p_mw, self.sgen_p_profile_pos, profile_row
        )
        net.sgen.loc[:, "p_mw"] = p_sgen

        q_sgen = np.zeros(len(net.sgen), dtype=float)
        if der_cosphi < 1.0:
            tan_phi = np.sqrt(1.0 / der_cosphi**2 - 1.0)
            q_sgen[self.dso_sgen_mask] = (
                float(der_cosphi_sign)
                * np.abs(p_sgen[self.dso_sgen_mask])
                * tan_phi
            )
        net.sgen.loc[:, "q_mvar"] = q_sgen


def _fill_profile_gaps(
    profiles: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Fill source gaps without altering complete profile columns.

    Interior gaps are linearly interpolated.  A trailing/leading gap is first
    copied from the same quarter-hour of the previous/next day (96 samples);
    forward/backward fill is only a final safety fallback.  The supplied data
    use this only for the short year-end tail of HS4/HS5 transmission-load
    profiles; DSO load and DER columns are complete.
    """
    filled = profiles.copy()
    audit: Dict[str, dict] = {}
    one_day = 96

    for column in filled.columns:
        original = filled[column].copy()
        series = original.interpolate(method="linear", limit_area="inside")

        missing = series.isna()
        if missing.any():
            previous_day = series.shift(one_day)
            series.loc[missing & previous_day.notna()] = previous_day

        missing = series.isna()
        if missing.any():
            next_day = series.shift(-one_day)
            series.loc[missing & next_day.notna()] = next_day

        series = series.ffill().bfill()
        if series.isna().any():
            raise ValueError(f"Profile {column!r} remains incomplete after fill")

        n_missing = int(original.isna().sum())
        audit[str(column)] = {
            "missing_source_samples": n_missing,
            "filled_samples": n_missing,
            "method": (
                "unchanged"
                if n_missing == 0
                else "interior linear; boundary same quarter-hour previous/next day"
            ),
        }
        filled[column] = series

    return filled, audit


def _install_dso_oltc_controllers(
    net: pp.pandapowerNet,
    hv_networks: Sequence[object],
    *,
    vm_set_pu: float,
) -> Tuple[int, ...]:
    """Regulate every DSO coupler's 110 kV terminal with a discrete OLTC."""
    controller_indices: List[int] = []
    for hv in hv_networks:
        for trafo3w_idx in hv.coupling_trafo_indices:
            controller = DiscreteTapControl.from_tap_step_percent(
                net,
                element_index=int(trafo3w_idx),
                vm_set_pu=float(vm_set_pu),
                side="mv",
                element="trafo3w",
                tol=1e-3,
            )
            controller_indices.append(int(controller.index))
    return tuple(controller_indices)


def _isolate_dso_with_stiff_primaries(
    source_net: pp.pandapowerNet,
    hv: object,
    *,
    primary_vm_pu: float,
    oltc_vref_pu: float,
) -> pp.pandapowerNet:
    """Return one DSO island supplied by three equal-weight stiff sources."""
    net = copy.deepcopy(source_net)

    # Remove the transmission system and every non-target DSO from the
    # electrical problem while preserving table indices used by HVNetworkInfo.
    for table_name in (
        "line",
        "trafo",
        "trafo3w",
        "load",
        "sgen",
        "gen",
        "ext_grid",
        "shunt",
        "switch",
    ):
        table = net[table_name]
        if len(table) and "in_service" in table.columns:
            table.loc[:, "in_service"] = False

    dso_lines = list(hv.line_indices) + list(hv.internal_aux_line_indices)
    net.line.loc[dso_lines, "in_service"] = True
    net.trafo3w.loc[list(hv.coupling_trafo_indices), "in_service"] = True
    net.load.loc[list(hv.load_indices), "in_service"] = True
    net.sgen.loc[list(hv.sgen_indices), "in_service"] = True

    for primary_bus in hv.coupling_ieee_buses:
        pp.create_ext_grid(
            net,
            bus=int(primary_bus),
            vm_pu=float(primary_vm_pu),
            va_degree=0.0,
            slack_weight=1.0 / len(hv.coupling_ieee_buses),
            name=f"{hv.net_id}|StiffPrimary_{primary_bus}",
        )

    controller_indices = _install_dso_oltc_controllers(
        net,
        (hv,),
        vm_set_pu=oltc_vref_pu,
    )
    net["annual_probe_oltc_controller_indices"] = controller_indices
    net["annual_probe_oltc_vref_pu"] = float(oltc_vref_pu)
    net["annual_probe_primary_vm_pu"] = float(primary_vm_pu)
    net["annual_probe_boundary"] = "isolated_dso_stiff_primary"
    return net


def _build_study_network(
    scenario: str,
    *,
    oltc_vref_pu: float,
    primary_vm_pu: float,
    dso_der_scale: Mapping[str, float] | None = None,
    dso_load_p_scale: Mapping[str, float] | None = None,
    dso_load_q_profile_base_mvar: Mapping[str, float] | None = None,
    dso_line_std_type: Mapping[str, str] | None = None,
) -> Tuple[Tuple[StudyCase, ...], object]:
    """Build four independent DSO cases with stiff 345 kV primaries."""
    source_net, meta = build_ieee39_net(scenario=scenario, verbose=False)
    meta = add_hv_networks(
        source_net,
        meta,
        install_tso_tertiary_shunts=False,
        verbose=False,
    )

    snapshot_base_values(source_net)
    # Several constructors deliberately pre-populate base columns for only
    # profile-controlled rows.  Complete any unrelated NaNs defensively.
    source_net.load["base_p_mw"] = pd.to_numeric(
        source_net.load["base_p_mw"], errors="coerce"
    ).fillna(source_net.load["p_mw"])
    source_net.load["base_q_mvar"] = pd.to_numeric(
        source_net.load["base_q_mvar"], errors="coerce"
    ).fillna(source_net.load["q_mvar"])
    source_net.sgen["base_p_mw"] = pd.to_numeric(
        source_net.sgen["base_p_mw"], errors="coerce"
    ).fillna(source_net.sgen["p_mw"])

    _apply_dso_overrides(
        source_net,
        meta.hv_networks,
        dso_der_scale=dso_der_scale,
        dso_load_p_scale=dso_load_p_scale,
        dso_load_q_profile_base_mvar=dso_load_q_profile_base_mvar,
        dso_line_std_type=dso_line_std_type,
    )

    # No DER-Q control and no switched compensation.
    source_net.sgen.loc[:, "q_mvar"] = 0.0
    if len(source_net.shunt):
        source_net.shunt.loc[:, "step"] = 0

    study_cases = tuple(
        (
            _isolate_dso_with_stiff_primaries(
                source_net,
                hv,
                primary_vm_pu=primary_vm_pu,
                oltc_vref_pu=oltc_vref_pu,
            ),
            hv,
        )
        for hv in meta.hv_networks
    )
    return study_cases, meta


def _run_power_flow(
    net: pp.pandapowerNet,
    *,
    warm_start: bool,
    recycle: bool,
) -> Tuple[bool, bool]:
    """
    Run one PF and retry from a fresh conversion if the warm start fails.

    Returns ``(converged, used_retry)``.
    """
    common = dict(
        algorithm="nr",
        calculate_voltage_angles=True,
        check_connectivity=True,
        distributed_slack=True,
        enforce_q_lims=False,
        max_iteration=50,
        run_control=True,
        voltage_depend_loads=True,
    )
    try:
        if warm_start and recycle:
            # pp.runpp with recycle enters the direct recycled-power-flow
            # branch before its controller branch. Route recycling through
            # run_control explicitly so the OLTC loop still executes. Profiles
            # change bus P/Q and OLTCs change taps, so both ppc components must
            # be refreshed.
            ctrl_variables = ctrl_variables_default(net)
            ctrl_variables["recycle_options"] = {
                "bus_pq": True,
                "trafo": True,
                "gen": False,
            }
            recycled_common = dict(common)
            recycled_common["run_control"] = False
            run_control(
                net,
                ctrl_variables=ctrl_variables,
                init="results",
                **recycled_common,
            )
        else:
            pp.runpp(
                net,
                init="results" if warm_start else "dc",
                **common,
            )
        if bool(net.converged):
            return True, False
    except Exception:
        pass

    try:
        pp.runpp(net, init="dc", **common)
        return bool(net.converged), True
    except Exception:
        return False, True


def _sum_numeric(table: pd.DataFrame, indices: Sequence[int], column: str) -> float:
    if not indices:
        return 0.0
    return float(pd.to_numeric(table.loc[list(indices), column]).sum())


def _extract_dso_row(
    net: pp.pandapowerNet,
    hv: object,
    *,
    converged: bool,
) -> np.ndarray:
    """Extract one DSO's operating point into the fixed result schema."""
    out = np.full(len(RESULT_FIELDS), np.nan, dtype=float)
    load_idx = tuple(int(i) for i in hv.load_indices)
    sgen_idx = tuple(int(i) for i in hv.sgen_indices)

    p_load = _sum_numeric(net.load, load_idx, "p_mw")
    q_load = _sum_numeric(net.load, load_idx, "q_mvar")
    p_der = _sum_numeric(net.sgen, sgen_idx, "p_mw")
    q_der = _sum_numeric(net.sgen, sgen_idx, "q_mvar")
    p_net = p_load - p_der
    q_net = q_load - q_der

    out[FIELD_POS["p_load_mw"]] = p_load
    out[FIELD_POS["q_load_mvar"]] = q_load
    out[FIELD_POS["p_der_mw"]] = p_der
    out[FIELD_POS["q_der_mvar"]] = q_der
    out[FIELD_POS["p_net_demand_mw"]] = p_net
    out[FIELD_POS["q_net_demand_mvar"]] = q_net

    if not converged:
        return out

    couplers = tuple(int(i) for i in hv.coupling_trafo_indices)
    p_each = net.res_trafo3w.loc[list(couplers), "p_hv_mw"].to_numpy(float)
    q_each = net.res_trafo3w.loc[list(couplers), "q_hv_mvar"].to_numpy(float)
    p_interface = float(p_each.sum())
    q_interface = float(q_each.sum())

    out[FIELD_POS["p_interface_hv_mw"]] = p_interface
    out[FIELD_POS["q_interface_hv_mvar"]] = q_interface
    out[FIELD_POS["p_loss_including_couplers_mw"]] = p_interface - p_net
    out[FIELD_POS["q_passive_network_mvar"]] = q_interface - q_net

    dso_buses = tuple(
        dict.fromkeys(
            [int(i) for i in hv.bus_indices]
            + [int(i) for i in hv.internal_aux_bus_indices]
        )
    )
    vm = net.res_bus.loc[list(dso_buses), "vm_pu"].to_numpy(float)
    out[FIELD_POS["v_min_pu"]] = float(np.min(vm))
    out[FIELD_POS["v_mean_pu"]] = float(np.mean(vm))
    out[FIELD_POS["v_max_pu"]] = float(np.max(vm))
    out[FIELD_POS["line_loading_max_percent"]] = float(
        net.res_line.loc[list(hv.line_indices), "loading_percent"].max()
    )
    out[FIELD_POS["coupler_loading_max_percent"]] = float(
        net.res_trafo3w.loc[list(couplers), "loading_percent"].max()
    )
    out[FIELD_POS["system_v_min_pu"]] = float(net.res_bus.vm_pu.min())
    out[FIELD_POS["system_v_max_pu"]] = float(net.res_bus.vm_pu.max())

    for pos, (p_value, q_value) in enumerate(zip(p_each, q_each), start=1):
        out[FIELD_POS[f"p_coupler_{pos}_mw"]] = float(p_value)
        out[FIELD_POS[f"q_coupler_{pos}_mvar"]] = float(q_value)
    for pos, trafo3w_idx in enumerate(couplers, start=1):
        out[FIELD_POS[f"tap_coupler_{pos}"]] = float(
            net.trafo3w.at[trafo3w_idx, "tap_pos"]
        )

    return out


def _save_checkpoint(
    path: Path,
    values: np.ndarray,
    converged: np.ndarray,
    retried: np.ndarray,
    completed_steps: int,
) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            values=values,
            converged=converged,
            retried=retried,
            completed_steps=np.asarray(completed_steps, dtype=int),
        )
    temp_path.replace(path)


def _load_checkpoint(
    path: Path,
    expected_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as saved:
        values = saved["values"]
        converged = saved["converged"]
        retried = saved["retried"]
        completed_steps = int(saved["completed_steps"])
    if values.shape != expected_shape:
        raise ValueError(
            f"Checkpoint shape {values.shape} does not match {expected_shape}"
        )
    return values, converged, retried, completed_steps


def _to_timeseries_frame(
    timestamps: pd.DatetimeIndex,
    dso_ids: Sequence[str],
    values: np.ndarray,
    converged: np.ndarray,
    retried: np.ndarray,
) -> pd.DataFrame:
    n_steps = len(timestamps)
    n_dso = len(dso_ids)
    result = pd.DataFrame(
        values.reshape(n_steps * n_dso, len(RESULT_FIELDS)),
        columns=RESULT_FIELDS,
    )
    result.insert(0, "dso_id", np.tile(np.asarray(dso_ids), n_steps))
    result.insert(0, "timestamp", timestamps.repeat(n_dso))
    result.insert(0, "step", np.repeat(np.arange(n_steps), n_dso))
    result.insert(3, "converged", converged.reshape(-1))
    result.insert(4, "pf_retry_used", retried.reshape(-1))
    return result


def _summarize(
    results: pd.DataFrame,
    *,
    dt_hours: float,
    installed_der: Mapping[str, float],
    reference_p_load: Mapping[str, float],
    reference_q_load: Mapping[str, float],
) -> pd.DataFrame:
    rows: List[dict] = []
    quantiles = (0.01, 0.05, 0.50, 0.95, 0.99)
    for dso_id, frame_all in results.groupby("dso_id", sort=False):
        frame = frame_all.loc[frame_all.converged].copy()
        p = frame["p_interface_hv_mw"]
        q = frame["q_interface_hv_mvar"]
        row = {
            "dso_id": dso_id,
            "n_samples": len(frame_all),
            "n_converged": len(frame),
            "n_failed": int((~frame_all.converged).sum()),
            "p_installed_der_mw": installed_der[dso_id],
            "p_reference_load_mw": reference_p_load[dso_id],
            "q_reference_load_mvar": reference_q_load[dso_id],
            "p_min_mw": float(p.min()),
            "p_mean_mw": float(p.mean()),
            "p_max_mw": float(p.max()),
            "q_min_mvar": float(q.min()),
            "q_mean_mvar": float(q.mean()),
            "q_max_mvar": float(q.max()),
            "export_fraction": float((p < 0.0).mean()),
            "energy_import_mwh": float(p.clip(lower=0.0).sum() * dt_hours),
            "energy_export_mwh": float((-p.clip(upper=0.0)).sum() * dt_hours),
            "p_loss_mean_mw": float(
                frame["p_loss_including_couplers_mw"].mean()
            ),
            "q_passive_network_mean_mvar": float(
                frame["q_passive_network_mvar"].mean()
            ),
            "v_min_annual_pu": float(frame["v_min_pu"].min()),
            "v_max_annual_pu": float(frame["v_max_pu"].max()),
            "samples_v_outside_0p9_1p1": int(
                ((frame["v_min_pu"] < 0.9) | (frame["v_max_pu"] > 1.1)).sum()
            ),
            "line_loading_max_percent": float(
                frame["line_loading_max_percent"].max()
            ),
            "coupler_loading_max_percent": float(
                frame["coupler_loading_max_percent"].max()
            ),
        }
        for quantile in quantiles:
            label = f"p{int(round(100 * quantile)):02d}"
            row[f"p_{label}_mw"] = float(p.quantile(quantile))
            row[f"q_{label}_mvar"] = float(q.quantile(quantile))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_scatter(
    results: pd.DataFrame,
    output_path: Path,
    *,
    der_cosphi: float,
    oltc_vref_pu: float,
    primary_vm_pu: float,
) -> None:
    valid = results.loc[results.converged].copy()
    dso_ids = list(valid.dso_id.drop_duplicates())
    colors = ["#1565C0", "#00897B", "#EF6C00", "#7B1FA2"]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    p_min = float(valid.p_interface_hv_mw.min())
    p_max = float(valid.p_interface_hv_mw.max())
    q_min = float(valid.q_interface_hv_mvar.min())
    q_max = float(valid.q_interface_hv_mvar.max())
    p_pad = max(0.03 * (p_max - p_min), 1.0)
    q_pad = max(0.03 * (q_max - q_min), 1.0)

    for ax, dso_id, color in zip(axes_flat, dso_ids, colors):
        frame = valid.loc[valid.dso_id == dso_id]
        ax.scatter(
            frame.p_interface_hv_mw,
            frame.q_interface_hv_mvar,
            s=2.0,
            alpha=0.16,
            color=color,
            edgecolors="none",
            rasterized=True,
        )
        ax.axhline(0.0, color="#666666", linewidth=0.6)
        ax.axvline(0.0, color="#666666", linewidth=0.6)
        ax.set_title(f"{dso_id} ({len(frame):,} converged samples)")
        ax.grid(True, color="#D9D9D9", linewidth=0.4, alpha=0.7)
        ax.set_xlim(p_min - p_pad, p_max + p_pad)
        ax.set_ylim(q_min - q_pad, q_max + q_pad)

    for ax in axes[-1, :]:
        ax.set_xlabel("P at TS-DS interface [MW]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Q at TS-DS interface [Mvar]")
    q_policy = (
        "DER Q = 0 Mvar"
        if der_cosphi == 1.0
        else f"DSO DER cos(phi) = {der_cosphi:.2f}, inductive"
    )
    fig.suptitle(
        "Annual synthetic-DS interface operating points\n"
        f"positive P/Q = import from TS; {q_policy}; "
        f"primary V = {primary_vm_pu:.3f} p.u.; "
        f"OLTC Vref = {oltc_vref_pu:.3f} p.u.",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _write_readme(
    output_path: Path,
    *,
    n_samples: int,
    n_power_flows: int,
    n_failed: int,
    scenario: str,
    dso_overrides: Mapping[str, Mapping[str, object]],
    dso_line_std_types: Mapping[str, Sequence[str]],
    dso_parallel_lines: Mapping[str, Mapping[str, int]],
    dso_reactive_load_models: Mapping[str, Mapping[str, object]],
    der_cosphi: float,
    oltc_vref_pu: float,
    primary_vm_pu: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    output_path.write_text(
        f"""# Annual synthetic-DS P-Q characterization

- Horizon: `{start}` to `{end}` ({n_samples:,} native 15-minute samples).
- Scenario: `{scenario}`.
- DSO-specific analysis overrides: {json.dumps(dso_overrides, sort_keys=True)}.
- Physical DSO line standard types: {json.dumps(dso_line_std_types, sort_keys=True)}.
- Physical parallel DSO corridors (circuit count greater than one):
  {json.dumps(dso_parallel_lines, sort_keys=True)}.
- Physical DSO reactive-load models:
  {json.dumps(dso_reactive_load_models, sort_keys=True)}.
- Plant: four electrically independent synthetic 110 kV DS models; the
  transmission system is not part of the annual characterization power flow.
- Primary boundary: three stiff 345 kV `ext_grid` sources per DSO at
  `{primary_vm_pu:.3f} p.u.` and 0 degrees, with equal distributed-slack
  weights of `1/3`.
- Convergence: `{n_power_flows - n_failed:,}` of `{n_power_flows:,}` DSO
  power flows solved; `{n_failed:,}` were explicitly flagged non-convergent.
- DER active power: exogenous `PV3`, `WP7`, and `WP10` profiles.
- DSO DER reactive power: inductive fixed power factor
  `cos(phi) = {der_cosphi:.2f}` with negative pandapower `q_mvar` absorption.
  Unity power factor therefore means `Q_DER = 0 Mvar`.
- OLTCs: pandapower `DiscreteTapControl` on all twelve DSO coupling
  transformers, controlling each 110 kV (`mv`) terminal around
  `{oltc_vref_pu:.3f} p.u.` using the physical 1.25% tap steps.
- Switched shunts: inactive; TSO tertiary shunts are not installed.
- Interface sign: positive P or Q is import from TS into the DS.

`annual_pq_timeseries.csv` is the auditable source table.  Its
`q_passive_network_mvar = q_interface_hv_mvar - q_net_demand_mvar` column
separates the passive line/transformer contribution from the time-varying
load Q.
`annual_pf_failures.csv` contains the affected timestamps and exogenous
P/Q inputs; failed rows are intentionally absent from the scatter files.

The four `pq_scatter_DSO_*.csv` files contain only `p_mw,q_mvar` and can be
used directly in PGFPlots, for example:

```latex
\\addplot[
  only marks,
  mark size=0.25pt,
  opacity=0.15
] table[
  x=p_mw,
  y=q_mvar,
  col sep=comma
] {{pq_scatter_DSO_1.csv}};
```
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles-csv",
        type=Path,
        default=Path(DEFAULT_PROFILES_CSV),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_REGISTRY),
        default="base_410",
        help="IEEE39 wind-replacement + per-DSO installed-capacity scenario.",
    )
    parser.add_argument(
        "--primary-vm-pu",
        type=float,
        default=DEFAULT_PRIMARY_VM_PU,
        help="Fixed 345 kV source voltage for all DSO primary terminals.",
    )
    parser.add_argument(
        "--oltc-vref-pu",
        type=float,
        default=DEFAULT_OLTC_VREF_PU,
        help="110 kV terminal setpoint for every DSO coupling OLTC.",
    )
    parser.add_argument(
        "--der-cosphi",
        type=float,
        choices=SUPPORTED_DER_COSPHI,
        default=1.0,
        help="Inductive fixed power factor for DSO wind/PV.",
    )
    parser.add_argument(
        "--dso-der-scale",
        action="append",
        type=_parse_dso_value,
        default=[],
        metavar="DSO_ID=FACTOR",
        help="Experiment-only DER nameplate multiplier; may be repeated.",
    )
    parser.add_argument(
        "--dso-load-p-scale",
        action="append",
        type=_parse_dso_value,
        default=[],
        metavar="DSO_ID=FACTOR",
        help="Experiment-only active-load base multiplier; may be repeated.",
    )
    parser.add_argument(
        "--dso-load-q-profile-base-mvar",
        action="append",
        type=_parse_dso_value,
        default=[],
        metavar="DSO_ID=MVAR",
        help="Profile-only Q base; zeros constant-Q rows; may be repeated.",
    )
    parser.add_argument(
        "--dso-line-std-type",
        action="append",
        type=_parse_dso_text,
        default=[],
        metavar="DSO_ID=STD_TYPE",
        help="Experiment-only line standard type; may be repeated.",
    )
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional smoke/benchmark limit; omit for the full profile year.",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-recycle", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def _parse_dso_value(value: str) -> Tuple[str, float]:
    try:
        dso_id, raw_value = value.split("=", maxsplit=1)
        return dso_id.strip(), float(raw_value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            f"Expected DSO_ID=value, received {value!r}"
        ) from exc


def _parse_dso_text(value: str) -> Tuple[str, str]:
    try:
        dso_id, text_value = value.split("=", maxsplit=1)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            f"Expected DSO_ID=value, received {value!r}"
        ) from exc
    if not dso_id.strip() or not text_value.strip():
        raise argparse.ArgumentTypeError(
            f"Expected non-empty DSO_ID=value, received {value!r}"
        )
    return dso_id.strip(), text_value.strip()


def _as_override_map(
    values: Sequence[Tuple[str, float]],
    *,
    option_name: str,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for dso_id, value in values:
        if dso_id in result:
            raise ValueError(f"Duplicate {option_name} entry for {dso_id}")
        result[dso_id] = float(value)
    return result


def _as_text_override_map(
    values: Sequence[Tuple[str, str]],
    *,
    option_name: str,
) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for dso_id, value in values:
        if dso_id in result:
            raise ValueError(f"Duplicate {option_name} entry for {dso_id}")
        result[dso_id] = str(value)
    return result


def main() -> None:
    args = parse_args()
    dso_der_scale = _as_override_map(
        args.dso_der_scale,
        option_name="--dso-der-scale",
    )
    dso_load_p_scale = _as_override_map(
        args.dso_load_p_scale,
        option_name="--dso-load-p-scale",
    )
    dso_load_q_profile_base_mvar = _as_override_map(
        args.dso_load_q_profile_base_mvar,
        option_name="--dso-load-q-profile-base-mvar",
    )
    dso_line_std_type = _as_text_override_map(
        args.dso_line_std_type,
        option_name="--dso-line-std-type",
    )
    dso_overrides = {
        "der_scale": dso_der_scale,
        "load_p_scale": dso_load_p_scale,
        "load_q_profile_base_mvar": dso_load_q_profile_base_mvar,
        "line_std_type": dso_line_std_type,
    }
    if args.output_dir is None:
        default_dir = DEFAULT_OUTPUT_DIR
        if args.scenario != "base_410":
            default_dir = default_dir.with_name(
                f"{default_dir.name}_{args.scenario}"
            )
        output_dir = default_dir.resolve()
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / ".annual_pq_checkpoint.npz"

    source_profiles = load_profiles(str(args.profiles_csv), timestep_min=15)
    profiles, profile_gap_audit = _fill_profile_gaps(source_profiles)
    if args.start is not None:
        profiles = profiles.loc[pd.Timestamp(args.start) :]
    if args.end is not None:
        profiles = profiles.loc[: pd.Timestamp(args.end)]
    if args.max_steps is not None:
        profiles = profiles.iloc[: args.max_steps]
    if profiles.empty:
        raise ValueError("Selected profile window is empty")

    step_seconds = profiles.index.to_series().diff().dropna().dt.total_seconds()
    if len(step_seconds) and not np.allclose(step_seconds.to_numpy(), 900.0):
        raise ValueError("Annual characterization requires regular 15-minute data")

    study_cases, _meta = _build_study_network(
        args.scenario,
        oltc_vref_pu=args.oltc_vref_pu,
        primary_vm_pu=args.primary_vm_pu,
        dso_der_scale=dso_der_scale,
        dso_load_p_scale=dso_load_p_scale,
        dso_load_q_profile_base_mvar=dso_load_q_profile_base_mvar,
        dso_line_std_type=dso_line_std_type,
    )
    dso_ids = [str(hv.net_id) for _net, hv in study_cases]
    n_steps = len(profiles)
    n_dso = len(dso_ids)
    shape = (n_steps, n_dso, len(RESULT_FIELDS))
    values = np.full(shape, np.nan, dtype=float)
    converged = np.zeros((n_steps, n_dso), dtype=bool)
    retried = np.zeros((n_steps, n_dso), dtype=bool)
    start_step = 0

    if args.resume and checkpoint_path.exists():
        values, converged, retried, start_step = _load_checkpoint(
            checkpoint_path, shape
        )
        print(f"Resuming from completed step {start_step:,}/{n_steps:,}")

    profile_maps = tuple(
        ProfileApplicationMap.from_net(
            net,
            profiles,
            tuple(int(index) for index in hv.sgen_indices),
        )
        for net, hv in study_cases
    )
    profile_values = profiles.to_numpy(dtype=float)
    installed_der = {
        str(hv.net_id): _sum_numeric(
            net.sgen, tuple(int(i) for i in hv.sgen_indices), "base_p_mw"
        )
        for net, hv in study_cases
    }
    reference_p_load = {
        str(hv.net_id): float(hv.total_ref_p_mw)
        for _net, hv in study_cases
    }
    reference_q_load = {
        str(hv.net_id): float(hv.total_ref_q_mvar)
        for _net, hv in study_cases
    }
    dso_line_std_types = {
        str(hv.net_id): sorted(
            str(value)
            for value in set(net.line.loc[list(hv.line_indices), "std_type"])
        )
        for net, hv in study_cases
    }
    dso_parallel_lines: Dict[str, Dict[str, int]] = {}
    dso_reactive_load_models: Dict[str, Dict[str, object]] = {}
    for net, hv in study_cases:
        dso_id = str(hv.net_id)
        line_rows = net.line.loc[list(hv.line_indices)]
        dso_parallel_lines[dso_id] = {
            str(row["name"]): int(row["parallel"])
            for _index, row in line_rows.iterrows()
            if int(row["parallel"]) > 1
        }

        load_rows = net.load.loc[list(hv.load_indices)]
        q_profile = load_rows["profile_q"]
        profiled = load_rows.loc[q_profile.notna()]
        constant = load_rows.loc[q_profile.isna()]
        profile_names = sorted(
            str(value) for value in set(profiled["profile_q"])
        )
        constant_base_q = float(
            pd.to_numeric(constant["base_q_mvar"]).sum()
        )
        profile_base_q = float(
            pd.to_numeric(profiled["base_q_mvar"]).sum()
        )
        dso_reactive_load_models[dso_id] = {
            "mode": "profile_only" if np.isclose(constant_base_q, 0.0) else "mixed",
            "constant_base_q_mvar": constant_base_q,
            "profile_base_q_mvar": profile_base_q,
            "profiles": profile_names,
        }

    t0 = time.perf_counter()
    last_converged = np.zeros(n_dso, dtype=bool)
    total_retries = int(retried[:start_step].sum())
    failed_power_flows = int((~converged[:start_step]).sum())

    print(
        f"Running {n_steps * n_dso:,} isolated-DSO power flows "
        f"({profiles.index[0]} to {profiles.index[-1]})"
    )
    for step in range(start_step, n_steps):
        for dso_pos, ((net, hv), profile_map) in enumerate(
            zip(study_cases, profile_maps)
        ):
            profile_map.apply(
                net,
                profile_values[step],
                der_cosphi=args.der_cosphi,
            )
            ok, used_retry = _run_power_flow(
                net,
                warm_start=bool(last_converged[dso_pos]),
                recycle=not args.no_recycle,
            )
            last_converged[dso_pos] = ok
            if used_retry:
                total_retries += 1
            if not ok:
                failed_power_flows += 1
            values[step, dso_pos, :] = _extract_dso_row(
                net, hv, converged=ok
            )
            converged[step, dso_pos] = ok
            retried[step, dso_pos] = used_retry

        completed = step + 1
        if args.checkpoint_every > 0 and (
            completed % args.checkpoint_every == 0 or completed == n_steps
        ):
            _save_checkpoint(
                checkpoint_path,
                values,
                converged,
                retried,
                completed,
            )

        if args.progress_every > 0 and (
            completed % args.progress_every == 0 or completed == n_steps
        ):
            elapsed = time.perf_counter() - t0
            processed = (completed - start_step) * n_dso
            rate = processed / elapsed if elapsed > 0 else np.nan
            remaining = (
                (n_steps - completed) * n_dso / rate if rate > 0 else np.nan
            )
            print(
                f"  {completed:>6,}/{n_steps:,} | "
                f"{rate:6.1f} PF/s | ETA {remaining / 60:5.1f} min | "
                f"failed={failed_power_flows} retries={total_retries}",
                flush=True,
            )

    results = _to_timeseries_frame(
        profiles.index,
        dso_ids,
        values,
        converged,
        retried,
    )
    timeseries_path = output_dir / "annual_pq_timeseries.csv"
    results.to_csv(
        timeseries_path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
        float_format="%.9f",
    )

    scatter = results.loc[
        results.converged,
        ["timestamp", "dso_id", "p_interface_hv_mw", "q_interface_hv_mvar"],
    ].rename(
        columns={
            "p_interface_hv_mw": "p_mw",
            "q_interface_hv_mvar": "q_mvar",
        }
    )
    scatter.to_csv(
        output_dir / "annual_pq_scatter.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
        float_format="%.9f",
    )
    for dso_id, frame in scatter.groupby("dso_id", sort=False):
        frame.loc[:, ["p_mw", "q_mvar"]].to_csv(
            output_dir / f"pq_scatter_{dso_id}.csv",
            index=False,
            float_format="%.9f",
        )

    failures = results.loc[
        ~results.converged,
        [
            "step", "timestamp", "dso_id", "p_load_mw", "q_load_mvar",
            "p_der_mw", "q_der_mvar", "p_net_demand_mw",
            "q_net_demand_mvar",
        ],
    ]
    failures.to_csv(
        output_dir / "annual_pf_failures.csv", index=False,
        date_format="%Y-%m-%d %H:%M:%S", float_format="%.9f",
    )

    summary = _summarize(
        results,
        dt_hours=0.25,
        installed_der=installed_der,
        reference_p_load=reference_p_load,
        reference_q_load=reference_q_load,
    )
    summary.to_csv(
        output_dir / "annual_pq_summary.csv",
        index=False,
        float_format="%.9f",
    )

    if not args.no_plot:
        _plot_scatter(
            results,
            output_dir / "annual_pq_characterization.png",
            der_cosphi=args.der_cosphi,
            oltc_vref_pu=args.oltc_vref_pu,
            primary_vm_pu=args.primary_vm_pu,
        )

    elapsed_s = time.perf_counter() - t0
    metadata = {
        "study": "annual synthetic DSO P-Q characterization",
        "scenario": args.scenario,
        "dso_overrides": dso_overrides,
        "dso_line_std_types": dso_line_std_types,
        "dso_parallel_lines": dso_parallel_lines,
        "dso_reactive_load_models": dso_reactive_load_models,
        "profiles_csv": str(Path(args.profiles_csv).resolve()),
        "start": str(profiles.index[0]),
        "end": str(profiles.index[-1]),
        "n_profile_samples": n_steps,
        "profile_resolution_minutes": 15,
        "n_dsos": n_dso,
        "n_power_flows": n_steps * n_dso,
        "n_failed_power_flows": failed_power_flows,
        "n_retry_steps": total_retries,
        "elapsed_seconds": elapsed_s,
        "power_flow": {
            "solver": "pandapower.runpp Newton-Raphson",
            "calculate_voltage_angles": True,
            "distributed_slack": True,
            "enforce_q_lims": False,
            "run_control": True,
            "fresh_initialization": "dc",
            "chronological_warm_start": "results",
            "warm_start": True,
            "recycle_bus_pq": not args.no_recycle,
            "recycle_trafo_parameters": not args.no_recycle,
        },
        "boundary_condition": {
            "model": "isolated DSO with three stiff primary sources",
            "primary_voltage_vm_pu": args.primary_vm_pu,
            "primary_voltage_va_degree": 0.0,
            "primary_sources_per_dso": 3,
            "distributed_slack": True,
            "slack_weight_per_source": 1.0 / 3.0,
            "transmission_system": "not included in the power flow",
        },
        "actuators": {
            "dso_der_q_policy": {
                "mode": "fixed_cosphi",
                "cosphi": args.der_cosphi,
                "q_sign": DER_COSPHI_SIGN,
                "q_sign_meaning": "negative q_mvar absorption (inductive)",
            },
            "oltc": {
                "controller": "pandapower DiscreteTapControl",
                "controlled_side": "mv (110 kV)",
                "vm_set_pu": args.oltc_vref_pu,
                "n_controllers": sum(
                    len(net["annual_probe_oltc_controller_indices"])
                    for net, _hv in study_cases
                ),
            },
            "switched_shunts": "inactive; TSO tertiary shunts not installed",
            "transmission_system": "not included",
        },
        "controlled_outputs": (
            "local 110 kV coupling-bus voltages via OLTC; interface P/Q "
            "remain uncontrolled characterization outputs"
        ),
        "interface_sign": (
            "positive p_interface_hv_mw/q_interface_hv_mvar denotes "
            "TS-to-DS import"
        ),
        "profile_gap_audit": profile_gap_audit,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    _write_readme(
        output_dir / "README.md",
        n_samples=n_steps,
        n_power_flows=n_steps * n_dso,
        n_failed=failed_power_flows,
        scenario=args.scenario,
        dso_overrides=dso_overrides,
        dso_line_std_types=dso_line_std_types,
        dso_parallel_lines=dso_parallel_lines,
        dso_reactive_load_models=dso_reactive_load_models,
        der_cosphi=args.der_cosphi,
        oltc_vref_pu=args.oltc_vref_pu,
        primary_vm_pu=args.primary_vm_pu,
        start=profiles.index[0],
        end=profiles.index[-1],
    )

    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Completed in {elapsed_s / 60:.2f} min: {timeseries_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
