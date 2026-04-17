"""
core/reporting.py
=================
Serialisation and human-readable reporting for the multi-zone TSO/DSO OFO run.

Two groups of helpers live here:

* ``write_tuned_params_json`` / ``load_and_apply_tuned_params``
  round-trip the per-controller ``g_w`` vectors to JSON so a subsequent run
  can warm-start from a previous stability-analysis checkpoint.
* ``write_stability_analysis_markdown`` renders the Theorem 3.3 report
  (C1/C2/C3 results plus per-controller actuator tables) as GitHub-flavored
  markdown.

These functions were extracted from ``experiments/000_M_TSO_M_DSO.py`` during
the April 2026 refactor to keep the runner focused on orchestration.
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime as _dt
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from analysis.stability_analysis import MultiZoneStabilityResult
    from network.ieee39.hv_networks import HVNetworkInfo


# Schema version for the tuned-params JSON format.  Bump whenever the layout
# of the written file changes in a backward-incompatible way.
TUNED_PARAMS_JSON_VERSION = 1


def write_tuned_params_json(
    json_path,
    *,
    time_s: float,
    zone_ids_sorted: List[int],
    zone_defs,
    tso_controllers,
    dso_controllers,
    hv_info_map: Dict[str, "HVNetworkInfo"],
    stab_result: Optional["MultiZoneStabilityResult"] = None,
    pump_result: Optional[Any] = None,
) -> None:
    """Serialise the tuned ``g_w`` values to a JSON file.

    The file can be loaded by a subsequent run via
    ``MultiTSOConfig.load_tuned_params_path``.

    Schema (version 1)::

        {
          "version": 1,
          "written_at": "...",
          "simulation_time_s": 3600.0,
          "global_metrics": {...},
          "tso_zones": {"1": {"g_w": [...], "actuator_counts": {...}}, ...},
          "dso_controllers": {"DSO_1": {"g_w": [...], "parent_zone": 2, ...}},
        }
    """
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

    payload: Dict[str, object] = {
        "version": TUNED_PARAMS_JSON_VERSION,
        "written_at": _dt.now().isoformat(timespec="seconds"),
        "simulation_time_s": float(time_s),
    }

    if stab_result is not None:
        c2 = stab_result.c2_continuous
        c3 = stab_result.c3_discrete
        payload["global_metrics"] = {
            "c1_satisfied": bool(stab_result.c1_satisfied),
            "c2_rho": float(c2.spectral_radius),
            "c2_satisfied": bool(stab_result.c2_satisfied),
            "c3_rho_gamma": float(c3.Gamma_spectral_radius),
            "c3_satisfied": bool(stab_result.c3_satisfied),
            "stable": bool(stab_result.stable),
        }

    # TSO zones
    tso_payload: Dict[str, object] = {}
    for z in zone_ids_sorted:
        ctrl = tso_controllers[z]
        zd = zone_defs[z]
        tso_payload[str(z)] = {
            "g_w": [float(x) for x in np.asarray(ctrl.params.g_w).ravel()],
            "actuator_counts": {
                "n_der":  int(len(zd.tso_der_indices)),
                "n_pcc":  int(len(zd.pcc_trafo_indices)),
                "n_gen":  int(len(zd.gen_indices)),
                "n_oltc": int(len(zd.oltc_trafo_indices)),
            },
        }
    payload["tso_zones"] = tso_payload

    # DSO controllers
    dso_payload: Dict[str, object] = {}
    for dso_id_key, dso_ctrl in dso_controllers.items():
        cfg_d = dso_ctrl.config
        parent_zone = (int(hv_info_map[dso_id_key].zone)
                       if dso_id_key in hv_info_map else None)
        dso_payload[str(dso_id_key)] = {
            "g_w": [float(x) for x in np.asarray(dso_ctrl.params.g_w).ravel()],
            "parent_zone": parent_zone,
            "actuator_counts": {
                "n_der":   int(len(cfg_d.der_indices)),
                "n_oltc":  int(len(cfg_d.interface_trafo_indices)),
                "n_shunt": int(len(cfg_d.shunt_bus_indices)),
            },
        }
    payload["dso_controllers"] = dso_payload

    if pump_result is not None:
        try:
            payload["pump_meta"] = {
                "alpha_tso":   float(getattr(pump_result, "alpha_tso", float("nan"))),
                "c1_feasible": bool(getattr(pump_result, "c1_feasible", False)),
                "c2_feasible": bool(getattr(pump_result, "c2_feasible", False)),
                "c3_feasible": bool(getattr(pump_result, "c3_feasible", False)),
                "feasible":    bool(getattr(pump_result, "feasible", False)),
            }
        except Exception:
            pass

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_and_apply_tuned_params(
    json_path: str,
    *,
    zone_defs,
    tso_controllers,
    dso_controllers,
    verbose: int,
) -> bool:
    """Load a tuned-params JSON file and apply its ``g_w`` values in place.

    Returns True if the file was loaded and applied, False if the file
    doesn't exist (silently skipped).  Raises ``ValueError`` if the schema
    version or actuator counts don't match the current network.
    """
    if not json_path or not os.path.exists(json_path):
        if verbose >= 1 and json_path:
            print(f"  [load_tuned_params] file not found, skipping: {json_path}")
        return False

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    version = int(payload.get("version", -1))
    if version != TUNED_PARAMS_JSON_VERSION:
        raise ValueError(
            f"Unsupported tuned-params JSON version {version} "
            f"(expected {TUNED_PARAMS_JSON_VERSION}) in {json_path}"
        )

    tso_payload: Dict[str, dict] = dict(payload.get("tso_zones", {}))
    dso_payload: Dict[str, dict] = dict(payload.get("dso_controllers", {}))

    for z in sorted(zone_defs.keys()):
        key = str(z)
        if key not in tso_payload:
            raise ValueError(
                f"Tuned-params file {json_path} is missing TSO zone {z}"
            )
        z_data = tso_payload[key]
        zd = zone_defs[z]

        expected_counts = {
            "n_der":  int(len(zd.tso_der_indices)),
            "n_pcc":  int(len(zd.pcc_trafo_indices)),
            "n_gen":  int(len(zd.gen_indices)),
            "n_oltc": int(len(zd.oltc_trafo_indices)),
        }
        got_counts = dict(z_data.get("actuator_counts", {}))
        if got_counts != expected_counts:
            raise ValueError(
                f"Tuned-params file {json_path} TSO zone {z} actuator "
                f"counts mismatch: expected {expected_counts}, got {got_counts}"
            )

        g_w_vec = np.asarray(z_data["g_w"], dtype=np.float64)
        n_expected = sum(expected_counts.values())
        if g_w_vec.size != n_expected:
            raise ValueError(
                f"Tuned-params file {json_path} TSO zone {z} g_w length "
                f"{g_w_vec.size} != expected {n_expected}"
            )

        tso_controllers[z].params = dataclasses.replace(
            tso_controllers[z].params, g_w=g_w_vec)

    for dso_id_key, dso_ctrl in dso_controllers.items():
        if dso_id_key not in dso_payload:
            raise ValueError(
                f"Tuned-params file {json_path} is missing DSO "
                f"controller {dso_id_key}"
            )
        d_data = dso_payload[dso_id_key]
        cfg_d = dso_ctrl.config

        expected_counts = {
            "n_der":   int(len(cfg_d.der_indices)),
            "n_oltc":  int(len(cfg_d.interface_trafo_indices)),
            "n_shunt": int(len(cfg_d.shunt_bus_indices)),
        }
        got_counts = dict(d_data.get("actuator_counts", {}))
        if got_counts != expected_counts:
            raise ValueError(
                f"Tuned-params file {json_path} DSO {dso_id_key} actuator "
                f"counts mismatch: expected {expected_counts}, got {got_counts}"
            )

        g_w_dso = np.asarray(d_data["g_w"], dtype=np.float64)
        n_expected = sum(expected_counts.values())
        if g_w_dso.size != n_expected:
            raise ValueError(
                f"Tuned-params file {json_path} DSO {dso_id_key} g_w length "
                f"{g_w_dso.size} != expected {n_expected}"
            )

        dso_ctrl.params = dataclasses.replace(
            dso_ctrl.params, g_w=g_w_dso)

    if verbose >= 1:
        written_at = str(payload.get("written_at", "?"))
        sim_time = payload.get("simulation_time_s")
        sim_tag = (f"{float(sim_time)/60.0:.1f} min"
                   if sim_time is not None else "?")
        print(f"  [load_tuned_params] Loaded {json_path}")
        print(f"    written at: {written_at}  (sim time: {sim_tag})")
        print(f"    applied to {len(tso_payload)} TSO zones "
              f"and {len(dso_payload)} DSO controllers")
    return True


def _lookup_trafo_name_bus(
    net,
    t_idx: int,
    *,
    prefer_3w: bool = True,
) -> Tuple[str, int]:
    """Resolve a transformer index to ``(name, hv_bus)``.

    Integer keys overlap between ``net.trafo`` (2W) and ``net.trafo3w``
    (every index 0..N exists in both tables), so the caller must specify
    which table to prefer:

    * ``prefer_3w=True``  -- PCC trafos and DSO interface trafos
      (always 3-winding HV-network couplers in this codebase).
    * ``prefer_3w=False`` -- TSO OLTC machine transformers (2-winding).
    """
    if prefer_3w:
        if hasattr(net, "trafo3w") and t_idx in net.trafo3w.index:
            nm = net.trafo3w.at[t_idx, "name"] or f"T3W_{t_idx}"
            bus = int(net.trafo3w.at[t_idx, "hv_bus"])
            return str(nm), bus
        if t_idx in net.trafo.index:
            nm = net.trafo.at[t_idx, "name"] or f"T2W_{t_idx}"
            bus = int(net.trafo.at[t_idx, "hv_bus"])
            return str(nm), bus
    else:
        if t_idx in net.trafo.index:
            nm = net.trafo.at[t_idx, "name"] or f"T2W_{t_idx}"
            bus = int(net.trafo.at[t_idx, "hv_bus"])
            return str(nm), bus
        if hasattr(net, "trafo3w") and t_idx in net.trafo3w.index:
            nm = net.trafo3w.at[t_idx, "name"] or f"T3W_{t_idx}"
            bus = int(net.trafo3w.at[t_idx, "hv_bus"])
            return str(nm), bus
    return f"Trafo_{t_idx}", -1


def _md_escape(name: str) -> str:
    """Escape ``|`` for GitHub-flavored markdown tables."""
    return str(name).replace("|", "\\|")


def write_stability_analysis_markdown(
    md_path,
    *,
    time_s: float,
    config,
    net,
    zone_ids_sorted: List[int],
    zone_defs,
    tso_controllers,
    dso_controllers,
    hv_info_map: Dict[str, "HVNetworkInfo"],
    stab_result: "MultiZoneStabilityResult",
) -> None:
    """Write the per-controller ``g_w`` tables plus the stability summary
    (Theorem 3.3) to ``md_path``.

    Four sections: header, global C1/C2/C3 pass-fail, per-zone compact
    table, and per-controller actuator tables.
    """
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)

    global_tag = "STABLE" if stab_result.stable else "UNSTABLE"
    c2 = stab_result.c2_continuous
    c3 = stab_result.c3_discrete

    lines: List[str] = []
    lines.append("# Multi-Zone OFO Stability Analysis (Theorem 3.3)")
    lines.append("")
    lines.append(f"- **Written at:** {_dt.now().isoformat(timespec='seconds')}")
    lines.append(f"- **Simulation time:** {time_s/60.0:.1f} min ({time_s:.0f} s)")
    lines.append(f"- **Verdict:** {global_tag}")
    lines.append(f"- **C1 (DSO inner loops):** {'pass' if stab_result.c1_satisfied else 'FAIL'}")
    lines.append(f"- **C2 (continuous):** rho(M_full^c) = {c2.spectral_radius:.4f}  "
                 f"[{'pass' if stab_result.c2_satisfied else 'FAIL'}]")
    lines.append(f"- **C3 (discrete):** rho(Gamma) = {c3.Gamma_spectral_radius:.4f}  "
                 f"[{'pass' if stab_result.c3_satisfied else 'FAIL'}]")
    lines.append("")

    # C1: DSO results
    lines.append("## C1 -- DSO Inner-Loop Stability")
    lines.append("")
    if stab_result.c1_dso:
        lines.append("| DSO | rho(M_cont) | Cascade margin | N_inner | Status |")
        lines.append("|---|---|---|---|---|")
        for r in stab_result.c1_dso:
            st = "pass" if r.stable else "FAIL"
            lines.append(f"| {r.dso_id} | {r.M_cont_spectral_radius:.4f} "
                         f"| {r.cascade_margin:.4f} | {r.N_inner:.0f} | {st} |")
    else:
        lines.append("No DSO data provided.")
    lines.append("")

    # C2: Continuous stability
    lines.append("## C2 -- Multi-Zone Continuous Stability")
    lines.append("")
    lines.append("| Zone | lam_max(M_ii^c) | kappa | rho_c | coupling | Status |")
    lines.append("|---|---|---|---|---|---|")
    for zr in stab_result.zones:
        kappa_str = ("inf" if c2.per_zone_kappa.get(zr.zone_id, np.inf) >= 1e6
                     else f"{c2.per_zone_kappa.get(zr.zone_id, 1.0):.3g}")
        lines.append(
            f"| Zone {zr.zone_id} "
            f"| {zr.lambda_max_c:.3g} "
            f"| {kappa_str} "
            f"| {zr.rho_c:.4f} "
            f"| {zr.coupling_sum_c:.4g} "
            f"| {'pass' if zr.rho_c < 1 else 'FAIL'} |"
        )
    lines.append("")
    lines.append(f"Global: rho(M_full^c) = {c2.spectral_radius:.4f}, "
                 f"small-gain gamma = {c2.small_gain_gamma:.4f}")
    lines.append("")

    # C3: Discrete small-gain
    lines.append("## C3 -- Discrete Small-Gain")
    lines.append("")
    lines.append(f"rho(Gamma) = {c3.Gamma_spectral_radius:.4f}  "
                 f"[{'pass' if c3.stable else 'FAIL'}]")
    lines.append("")
    if c3.g_min_required:
        lines.append("### G Sizing Rule (Corollary 3.2)")
        lines.append("")
        lines.append("| Zone | Actuator | g_w (current) | g_w (min) | Margin |")
        lines.append("|---|---|---|---|---|")
        for zi in sorted(c3.g_min_required.keys()):
            for name in sorted(c3.g_min_required[zi].keys()):
                g_cur = c3.g_current.get(zi, {}).get(name, 0.0)
                g_min = c3.g_min_required[zi][name]
                margin = c3.g_margin.get(zi, {}).get(name, 0.0)
                status = "OK" if margin >= 0 else "**VIOLATION**"
                lines.append(f"| Zone {zi} | {name} | {g_cur:.2f} "
                             f"| {g_min:.2f} | {margin:.2f} {status} |")
        lines.append("")

    # Warnings / Recommendations
    all_warnings = c2.warnings + c3.warnings
    for r in stab_result.c1_dso:
        all_warnings.extend(r.warnings)
    if all_warnings:
        lines.append("### Warnings")
        lines.append("")
        for w in all_warnings:
            lines.append(f"- {w}")
        lines.append("")
    if stab_result.recommendations:
        lines.append("### Recommendations")
        lines.append("")
        for rec in stab_result.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    # Per-TSO-zone actuator tables
    lines.append("## TSO controllers")
    lines.append("")
    for z in zone_ids_sorted:
        ctrl = tso_controllers[z]
        zd = zone_defs[z]
        gw_vec = ctrl.params.g_w
        lines.append(f"### TSO Zone {z}")
        lines.append("")
        lines.append("| Type | Name | Bus | g_w |")
        lines.append("|---|---|---|---|")

        off = 0
        for k, s_idx in enumerate(zd.tso_der_indices):
            nm = net.sgen.at[s_idx, "name"] or f"SGen_{s_idx}"
            bus = int(net.sgen.at[s_idx, "bus"])
            lines.append(f"| Q_DER | `{_md_escape(nm)}` | {bus} | {gw_vec[off+k]:.4g} |")
        off += len(zd.tso_der_indices)

        for k, t_idx in enumerate(zd.pcc_trafo_indices):
            nm, bus = _lookup_trafo_name_bus(net, int(t_idx), prefer_3w=True)
            lines.append(f"| Q_PCC | `{_md_escape(nm)}` | {bus} | {gw_vec[off+k]:.4g} |")
        off += len(zd.pcc_trafo_indices)

        for k, g_idx in enumerate(zd.gen_indices):
            nm = net.gen.at[g_idx, "name"] or f"Gen_{g_idx}"
            bus = int(net.gen.at[g_idx, "bus"])
            lines.append(f"| V_gen | `{_md_escape(nm)}` | {bus} | {gw_vec[off+k]:.4g} |")
        off += len(zd.gen_indices)

        for k, t_idx in enumerate(zd.oltc_trafo_indices):
            nm, bus = _lookup_trafo_name_bus(net, int(t_idx), prefer_3w=False)
            lines.append(f"| OLTC | `{_md_escape(nm)}` | {bus} | {gw_vec[off+k]:.4g} |")
        off += len(zd.oltc_trafo_indices)
        lines.append("")

    # Per-DSO actuator tables
    lines.append("## DSO controllers")
    lines.append("")
    for dso_id_key, dso_ctrl in dso_controllers.items():
        parent_zone = (hv_info_map[dso_id_key].zone
                       if dso_id_key in hv_info_map else "?")
        dso_cfg_out = dso_ctrl.config
        gw_dso = dso_ctrl.params.g_w

        lines.append(f"### DSO `{dso_id_key}`  (under TSO Zone {parent_zone})")
        lines.append("")
        lines.append("| Type | Name | Bus | g_w |")
        lines.append("|---|---|---|---|")

        off_d = 0
        for k, s_idx in enumerate(dso_cfg_out.der_indices):
            nm = net.sgen.at[s_idx, "name"] or f"SGen_{s_idx}"
            bus = int(net.sgen.at[s_idx, "bus"])
            lines.append(f"| DER | `{_md_escape(nm)}` | {bus} | {gw_dso[off_d+k]:.4g} |")
        off_d += len(dso_cfg_out.der_indices)

        for k, t_idx in enumerate(dso_cfg_out.interface_trafo_indices):
            nm, bus = _lookup_trafo_name_bus(net, int(t_idx), prefer_3w=True)
            lines.append(f"| OLTC | `{_md_escape(nm)}` | {bus} | {gw_dso[off_d+k]:.4g} |")
        off_d += len(dso_cfg_out.interface_trafo_indices)

        for k, sb_idx in enumerate(dso_cfg_out.shunt_bus_indices):
            lines.append(f"| Shunt | `Shunt_{int(sb_idx)}` | {int(sb_idx)} | "
                         f"{gw_dso[off_d+k]:.4g} |")
        off_d += len(dso_cfg_out.shunt_bus_indices)
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
