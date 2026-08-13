"""
Compare coupled-TS and stiff-primary DSO boundary conditions.

This script reproduces the investigation that identified depressed endogenous
IEEE 39 primary voltages as the cause of OLTC saturation in the 700 MW DSO
scenario.  It evaluates:

1. ``coupled``: the complete IEEE 39 + four-DSO network, with primary voltages
   determined by the transmission-system power flow;
2. ``isolated``: four independent DSO cases, each supplied through three stiff
   345 kV sources at 1.03 p.u. with equal distributed-slack weights.

For each boundary condition, the default DER policies are unity power factor
and inductive cos(phi) = 0.98 / 0.95.  All coupling-transformer OLTCs use
pandapower ``DiscreteTapControl`` with a 1.03 p.u. 110 kV-side reference.

Run from the repository root:

```
python -m network.ieee39.analysis.probe_dso_boundary_conditions
```
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pandapower as pp

from analysis.annual_dso_pq_characterization import (
    DEFAULT_OLTC_VREF_PU,
    DEFAULT_PRIMARY_VM_PU,
    DER_COSPHI_SIGN,
    ProfileApplicationMap,
    SUPPORTED_DER_COSPHI,
    _build_study_network,
    _fill_profile_gaps,
    _install_dso_oltc_controllers,
    _run_power_flow,
)
from core.profiles import (
    DEFAULT_PROFILES_CSV,
    load_profiles,
    snapshot_base_values,
)
from network.ieee39 import add_hv_networks, build_ieee39_net
from network.ieee39.scenarios import SCENARIO_REGISTRY


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[3]
    / "results"
    / "ieee39_dso_boundary_condition_probe"
)
BOUNDARY_CHOICES: Tuple[str, ...] = ("coupled", "isolated")


def _complete_base_columns(net: pp.pandapowerNet) -> None:
    """Populate auditable profile bases and disable non-profile Q actuators."""
    snapshot_base_values(net)
    net.load["base_p_mw"] = pd.to_numeric(
        net.load["base_p_mw"], errors="coerce"
    ).fillna(net.load["p_mw"])
    net.load["base_q_mvar"] = pd.to_numeric(
        net.load["base_q_mvar"], errors="coerce"
    ).fillna(net.load["q_mvar"])
    net.sgen["base_p_mw"] = pd.to_numeric(
        net.sgen["base_p_mw"], errors="coerce"
    ).fillna(net.sgen["p_mw"])
    net.sgen.loc[:, "q_mvar"] = 0.0
    if len(net.shunt):
        net.shunt.loc[:, "step"] = 0


def _build_coupled_case(
    scenario: str,
    *,
    oltc_vref_pu: float,
) -> Tuple[pp.pandapowerNet, object]:
    """Build the complete IEEE 39 + four-DSO diagnostic case."""
    net, meta = build_ieee39_net(scenario=scenario, verbose=False)
    meta = add_hv_networks(
        net,
        meta,
        install_tso_tertiary_shunts=False,
        verbose=False,
    )
    _complete_base_columns(net)
    net["boundary_probe_oltc_controller_indices"] = (
        _install_dso_oltc_controllers(
            net,
            meta.hv_networks,
            vm_set_pu=oltc_vref_pu,
        )
    )
    return net, meta


def _run_coupled_power_flow(
    net: pp.pandapowerNet,
    *,
    warm_start: bool,
    recycle: bool,
) -> Tuple[bool, bool]:
    """Run the coupled diagnostic without distributed slack."""
    common = dict(
        algorithm="nr",
        calculate_voltage_angles=True,
        check_connectivity=True,
        distributed_slack=False,
        enforce_q_lims=False,
        max_iteration=50,
        run_control=True,
        voltage_depend_loads=True,
    )
    try:
        kwargs = dict(common)
        kwargs["init"] = "results" if warm_start else "dc"
        if warm_start and recycle:
            kwargs["recycle"] = {
                "bus_pq": True,
                "trafo": False,
                "gen": False,
            }
        pp.runpp(net, **kwargs)
        if bool(net.converged):
            return True, False
    except Exception:
        pass

    try:
        pp.runpp(net, init="dc", **common)
        return bool(net.converged), True
    except Exception:
        return False, True


def _sum(table: pd.DataFrame, indices: Sequence[int], column: str) -> float:
    return float(table.loc[list(indices), column].sum())


def _result_row(
    *,
    boundary: str,
    der_cosphi: float,
    step: int,
    timestamp: pd.Timestamp,
    net: pp.pandapowerNet,
    hv: object,
    converged: bool,
    retry_used: bool,
) -> dict:
    """Extract one DSO diagnostic row."""
    trafo_indices = tuple(int(index) for index in hv.coupling_trafo_indices)
    taps = net.trafo3w.loc[list(trafo_indices), "tap_pos"].to_numpy(float)
    tap_min = net.trafo3w.loc[list(trafo_indices), "tap_min"].to_numpy(float)
    tap_max = net.trafo3w.loc[list(trafo_indices), "tap_max"].to_numpy(float)

    row = {
        "boundary": boundary,
        "der_cosphi": float(der_cosphi),
        "der_cosphi_sign": DER_COSPHI_SIGN,
        "step": int(step),
        "timestamp": timestamp,
        "dso_id": str(hv.net_id),
        "converged": bool(converged),
        "pf_retry_used": bool(retry_used),
        "p_load_mw": _sum(net.load, hv.load_indices, "p_mw"),
        "q_load_mvar": _sum(net.load, hv.load_indices, "q_mvar"),
        "p_der_mw": _sum(net.sgen, hv.sgen_indices, "p_mw"),
        "q_der_mvar": _sum(net.sgen, hv.sgen_indices, "q_mvar"),
        "p_interface_hv_mw": np.nan,
        "q_interface_hv_mvar": np.nan,
        "primary_v_min_pu": np.nan,
        "primary_v_mean_pu": np.nan,
        "primary_v_max_pu": np.nan,
        "secondary_v_min_pu": np.nan,
        "secondary_v_mean_pu": np.nan,
        "secondary_v_max_pu": np.nan,
        "dso_v_min_pu": np.nan,
        "dso_v_max_pu": np.nan,
        "tap_coupler_1": float(taps[0]),
        "tap_coupler_2": float(taps[1]),
        "tap_coupler_3": float(taps[2]),
        "oltc_at_limit": np.nan,
    }
    if not converged:
        return row

    primary_vm = net.res_bus.loc[
        list(hv.coupling_ieee_buses), "vm_pu"
    ].to_numpy(float)
    secondary_vm = net.res_bus.loc[
        list(hv.coupling_hv_bus_indices), "vm_pu"
    ].to_numpy(float)
    dso_buses = tuple(
        dict.fromkeys(
            [int(index) for index in hv.bus_indices]
            + [int(index) for index in hv.internal_aux_bus_indices]
        )
    )
    dso_vm = net.res_bus.loc[list(dso_buses), "vm_pu"].to_numpy(float)
    row.update(
        {
            "p_interface_hv_mw": _sum(
                net.res_trafo3w, trafo_indices, "p_hv_mw"
            ),
            "q_interface_hv_mvar": _sum(
                net.res_trafo3w, trafo_indices, "q_hv_mvar"
            ),
            "oltc_at_limit": bool(
                np.any(np.isclose(taps, tap_min) | np.isclose(taps, tap_max))
            ),
            "primary_v_min_pu": float(primary_vm.min()),
            "primary_v_mean_pu": float(primary_vm.mean()),
            "primary_v_max_pu": float(primary_vm.max()),
            "secondary_v_min_pu": float(secondary_vm.min()),
            "secondary_v_mean_pu": float(secondary_vm.mean()),
            "secondary_v_max_pu": float(secondary_vm.max()),
            "dso_v_min_pu": float(dso_vm.min()),
            "dso_v_max_pu": float(dso_vm.max()),
        }
    )
    return row


def _run_coupled(
    *,
    scenario: str,
    profiles: pd.DataFrame,
    der_cosphi: float,
    oltc_vref_pu: float,
    recycle: bool,
) -> List[dict]:
    net, meta = _build_coupled_case(
        scenario,
        oltc_vref_pu=oltc_vref_pu,
    )
    dso_sgens = tuple(
        int(index)
        for hv in meta.hv_networks
        for index in hv.sgen_indices
    )
    profile_map = ProfileApplicationMap.from_net(net, profiles, dso_sgens)
    profile_values = profiles.to_numpy(dtype=float)
    rows: List[dict] = []
    warm_start = False

    for step, timestamp in enumerate(profiles.index):
        profile_map.apply(
            net,
            profile_values[step],
            der_cosphi=der_cosphi,
        )
        converged, retry_used = _run_coupled_power_flow(
            net,
            warm_start=warm_start,
            recycle=recycle,
        )
        warm_start = converged
        rows.extend(
            _result_row(
                boundary="coupled",
                der_cosphi=der_cosphi,
                step=step,
                timestamp=timestamp,
                net=net,
                hv=hv,
                converged=converged,
                retry_used=retry_used,
            )
            for hv in meta.hv_networks
        )
    return rows


def _run_isolated(
    *,
    scenario: str,
    profiles: pd.DataFrame,
    der_cosphi: float,
    primary_vm_pu: float,
    oltc_vref_pu: float,
    recycle: bool,
) -> List[dict]:
    study_cases, _meta = _build_study_network(
        scenario,
        primary_vm_pu=primary_vm_pu,
        oltc_vref_pu=oltc_vref_pu,
    )
    profile_maps = tuple(
        ProfileApplicationMap.from_net(
            net,
            profiles,
            tuple(int(index) for index in hv.sgen_indices),
        )
        for net, hv in study_cases
    )
    profile_values = profiles.to_numpy(dtype=float)
    warm_start = np.zeros(len(study_cases), dtype=bool)
    rows: List[dict] = []

    for step, timestamp in enumerate(profiles.index):
        for pos, ((net, hv), profile_map) in enumerate(
            zip(study_cases, profile_maps)
        ):
            profile_map.apply(
                net,
                profile_values[step],
                der_cosphi=der_cosphi,
            )
            converged, retry_used = _run_power_flow(
                net,
                warm_start=bool(warm_start[pos]),
                recycle=recycle,
            )
            warm_start[pos] = converged
            rows.append(
                _result_row(
                    boundary="isolated",
                    der_cosphi=der_cosphi,
                    step=step,
                    timestamp=timestamp,
                    net=net,
                    hv=hv,
                    converged=converged,
                    retry_used=retry_used,
                )
            )
    return rows


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(
            ["boundary", "der_cosphi", "dso_id"],
            sort=False,
        )
        .agg(
            n_samples=("converged", "size"),
            n_converged=("converged", "sum"),
            n_retries=("pf_retry_used", "sum"),
            primary_v_min_pu=("primary_v_min_pu", "min"),
            primary_v_max_pu=("primary_v_max_pu", "max"),
            secondary_v_min_pu=("secondary_v_min_pu", "min"),
            secondary_v_max_pu=("secondary_v_max_pu", "max"),
            dso_v_min_pu=("dso_v_min_pu", "min"),
            dso_v_max_pu=("dso_v_max_pu", "max"),
            oltc_limit_fraction=("oltc_at_limit", "mean"),
            p_interface_min_mw=("p_interface_hv_mw", "min"),
            p_interface_max_mw=("p_interface_hv_mw", "max"),
            q_interface_min_mvar=("q_interface_hv_mvar", "min"),
            q_interface_max_mvar=("q_interface_hv_mvar", "max"),
        )
        .reset_index()
    )
    summary.insert(
        5,
        "n_failed",
        summary["n_samples"] - summary["n_converged"],
    )
    return summary


def _write_readme(
    path: Path,
    *,
    scenario: str,
    n_steps: int,
    boundaries: Sequence[str],
    cosphi_values: Sequence[float],
    primary_vm_pu: float,
    oltc_vref_pu: float,
) -> None:
    path.write_text(
        f"""# IEEE 39 / DSO boundary-condition probe

- Scenario: `{scenario}`.
- Samples: `{n_steps}` native 15-minute time steps.
- Boundaries: `{", ".join(boundaries)}`.
- DSO DER cos(phi): `{", ".join(str(v) for v in cosphi_values)}` with
  `q_sign = -1` (inductive absorption).
- Isolated primary sources: `{primary_vm_pu:.3f} p.u.`, 0 degrees, three
  equal distributed-slack weights of `1/3` per DSO.
- Coupling-transformer OLTC target: `{oltc_vref_pu:.3f} p.u.` on the 110 kV
  (`mv`) side.
- Fresh power-flow initialization: DC; chronological warm start: results.

`boundary_probe_timeseries.csv` contains one row per boundary, power factor,
timestamp, and DSO. `boundary_probe_summary.csv` provides convergence,
voltage, OLTC-limit, and interface-P/Q comparisons.
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_REGISTRY),
        default="rural_700",
    )
    parser.add_argument(
        "--profiles-csv",
        type=Path,
        default=Path(DEFAULT_PROFILES_CSV),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--boundaries",
        nargs="+",
        choices=BOUNDARY_CHOICES,
        default=list(BOUNDARY_CHOICES),
    )
    parser.add_argument(
        "--cosphi",
        nargs="+",
        type=float,
        choices=SUPPORTED_DER_COSPHI,
        default=list(SUPPORTED_DER_COSPHI),
    )
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument(
        "--primary-vm-pu",
        type=float,
        default=DEFAULT_PRIMARY_VM_PU,
    )
    parser.add_argument(
        "--oltc-vref-pu",
        type=float,
        default=DEFAULT_OLTC_VREF_PU,
    )
    parser.add_argument("--no-recycle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_profiles = load_profiles(
        str(args.profiles_csv),
        timestep_min=15,
    )
    profiles, gap_audit = _fill_profile_gaps(source_profiles)
    if args.start is not None:
        profiles = profiles.loc[pd.Timestamp(args.start) :]
    if args.max_steps is not None:
        profiles = profiles.iloc[: args.max_steps]
    if profiles.empty:
        raise ValueError("Selected profile window is empty")

    rows: List[dict] = []
    t0 = time.perf_counter()
    for boundary in args.boundaries:
        for cosphi in args.cosphi:
            print(
                f"Running boundary={boundary} cosphi={cosphi:.2f} "
                f"for {len(profiles)} samples",
                flush=True,
            )
            if boundary == "coupled":
                rows.extend(
                    _run_coupled(
                        scenario=args.scenario,
                        profiles=profiles,
                        der_cosphi=cosphi,
                        oltc_vref_pu=args.oltc_vref_pu,
                        recycle=not args.no_recycle,
                    )
                )
            else:
                rows.extend(
                    _run_isolated(
                        scenario=args.scenario,
                        profiles=profiles,
                        der_cosphi=cosphi,
                        primary_vm_pu=args.primary_vm_pu,
                        oltc_vref_pu=args.oltc_vref_pu,
                        recycle=not args.no_recycle,
                    )
                )

    results = pd.DataFrame(rows)
    results.to_csv(
        output_dir / "boundary_probe_timeseries.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
        float_format="%.9f",
    )
    summary = _summarize(results)
    summary.to_csv(
        output_dir / "boundary_probe_summary.csv",
        index=False,
        float_format="%.9f",
    )

    elapsed_s = time.perf_counter() - t0
    metadata: Dict[str, object] = {
        "study": "IEEE39/DSO boundary-condition and OLTC investigation",
        "scenario": args.scenario,
        "profiles_csv": str(args.profiles_csv.resolve()),
        "start": str(profiles.index[0]),
        "end": str(profiles.index[-1]),
        "n_profile_samples": len(profiles),
        "boundaries": list(args.boundaries),
        "der_cosphi_values": list(args.cosphi),
        "der_cosphi_sign": DER_COSPHI_SIGN,
        "primary_vm_pu_isolated": args.primary_vm_pu,
        "oltc_vref_pu": args.oltc_vref_pu,
        "isolated_distributed_slack": True,
        "isolated_slack_weight_per_source": 1.0 / 3.0,
        "fresh_initialization": "dc",
        "elapsed_seconds": elapsed_s,
        "profile_gap_audit": gap_audit,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    _write_readme(
        output_dir / "README.md",
        scenario=args.scenario,
        n_steps=len(profiles),
        boundaries=args.boundaries,
        cosphi_values=args.cosphi,
        primary_vm_pu=args.primary_vm_pu,
        oltc_vref_pu=args.oltc_vref_pu,
    )

    print(f"Completed in {elapsed_s:.1f} s: {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
