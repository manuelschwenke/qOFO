"""
008_PILOT_BUS_SELECT.py
=======================

Pilot-node selection for the classical SVR reference (thesis variant ``S1``,
``\\cref{ch:setup:references:svr}``).

One pilot node per TSO control area, chosen inside the fixed 3-area partition as
the bus of strongest reactive influence over its own area:

    p_a = argmax_{p in R_a}  sum_{j in R_a} | dV_j / dQ_p |

with ``R_a`` the area's TN **PQ** buses.  Restricting the candidate set to PQ
buses is not a filter added on top -- it falls out of the reduced Jacobian, and
it is the right set for two independent reasons:

  * generator terminal and slack buses are voltage-regulated, so a pilot error
    defined there would be trivially zero; and
  * ``R_a`` is exactly the set the TS-OFO holds voltage references for
    (``v_bus_indices_z`` in ``experiments/runners/multi_tso_dso.py``), so the
    selected pilot is guaranteed to lie inside the OFO's own reference set and
    the two schemes are scored on the same buses.

The network is built exactly as ``make_cigre_config()`` builds it -- wind-replace
IEEE 39 at ``rural_700`` with the four 110 kV HV sub-networks attached and no
tertiary shunts -- so the selection is made on the plant the campaigns run on.

Selection is made ONCE at the base case and is then held fixed, which is what
classical SVR does with its participation structure.

Output
------
``results/008_pilot_bus/pilot_buses.json``  -- machine-readable, consumed by the
``S1`` variant in ``005_CIGRE_MULTI.py``.
Console table -- reproduces the row set the thesis reports.

Run:
    python -m experiments.CIGRE_2026.008_PILOT_BUS_SELECT
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandapower as pp

from core.plant import PandapowerStaticPlant
from network.ieee39.build import build_ieee39_net, tag_der_q_modes
from network.ieee39.hv_networks import add_hv_networks
from network.zone_partition import fixed_zone_partition_ieee39
from controller.der_qv_local_loop import install_der_q_loops
from sensitivity.index_helper import get_jacobian_indices
from sensitivity.jacobian import JacobianSensitivities

OUT_DIR = Path("results/008_pilot_bus")

#: Installed-capacity scenario.  MUST match ``make_cigre_config().scenario`` in
#: 005_CIGRE_MULTI -- the pilot node is selected from dV/dQ at the base
#: operating point, and rural_700 loads the sub-networks very differently from
#: base_410 (700 MW installed DER per DSO against 410).  A mismatch here would
#: silently select pilots for a network the campaigns never run.
SCENARIO = "rural_700"


#: Which config the selection is made on.  This matters: the two build
#: MATERIALLY DIFFERENT plants.  The validated set installs tertiary shunts and
#: uses a Thevenin boundary equivalent; the paper set has neither.  The pilot
#: node is chosen from dV/dQ at the base operating point, so a selection made on
#: one is not valid for the other.
#:   "thesis" -- experiments.run_multi_system_ofo.make_config (validated set;
#:               what the dissertation ladder runs on)
#:   "cigre"  -- 005_CIGRE_MULTI.make_cigre_config (frozen paper set)
CONFIG_SOURCE = "thesis"


def _load_config(source: str):
    """Return the config whose plant the selection should be made on.

    Every plant-side parameter used below comes from here, so this script drifts
    from the campaigns only if the runner's SETUP SEQUENCE changes, not when a
    parameter value does.  If it ever does drift, the symptom is a diverging
    base case rather than a silently different one.
    """
    if source == "thesis":
        from experiments.run_multi_system_ofo import make_config
        cfg = make_config()
    elif source == "cigre":
        import importlib.util as _u
        import os as _os
        _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           "005_CIGRE_MULTI.py")
        _s = _u.spec_from_file_location("_m005_cfg", _p)
        _m = _u.module_from_spec(_s)
        _s.loader.exec_module(_m)
        cfg = _m.make_cigre_config()
    else:
        raise SystemExit(f"unknown config source {source!r}; use thesis|cigre")

    if cfg.scenario != SCENARIO:
        raise SystemExit(
            f"scenario mismatch: {source} config uses {cfg.scenario!r}, this "
            f"script {SCENARIO!r}. The pilot nodes must be selected on the "
            "network the campaigns actually run."
        )
    return cfg


# ---------------------------------------------------------------------------
#  Plant
# ---------------------------------------------------------------------------


def build_benchmark_net():
    """Rebuild the plant of the selected config, as ``run_multi_tso_dso`` does.

    Mirrors the runner's steps 1-2, INCLUDING its shunt branching: the
    validated set installs MSC/MSR banks and dispatches them through the
    ShuntIntegrator, the frozen paper set installs none.  Hardcoding either
    would select pilot nodes on a plant the campaigns never run -- which is the
    class of error this whole script exists downstream of.
    """
    cfg = _load_config(CONFIG_SOURCE)

    net, meta = build_ieee39_net(
        ext_grid_vm_pu=1.03,
        scenario=SCENARIO,
        verbose=False,
    )

    # Same branch the runner takes (multi_tso_dso.py: _shunt_mode handling).
    _shunt_mode = cfg.shunt_dispatch
    if _shunt_mode == "off" and cfg.install_tso_tertiary_shunts:
        _shunt_mode = "miqp"
    if _shunt_mode == "integrator":
        meta = add_hv_networks(
            net, meta,
            install_tso_tertiary_shunts=True,
            tso_shunt_kind="msc_msr",
            msc_n_levels=cfg.tso_shunt_msc_n_levels,
            msr_n_levels=cfg.tso_shunt_msr_n_levels,
            msc_q_step_mvar=cfg.tso_shunt_msc_q_step_mvar,
            msr_q_step_mvar=cfg.tso_shunt_msr_q_step_mvar,
            verbose=False,
        )
    else:
        meta = add_hv_networks(
            net, meta,
            install_tso_tertiary_shunts=cfg.install_tso_tertiary_shunts,
            tso_tertiary_shunt_q_mvar=cfg.tso_tertiary_shunt_q_mvar,
            verbose=False,
        )

    # The solve must use the runner's plant settings, not a bare runpp.  At
    # rural_700 the sub-networks carry 700 MW of installed DER, and a plain
    # Newton-Raphson without distributed slack, machine Q limits and the local
    # q(v) loops does not converge -- the base case is only well posed once the
    # autonomous voltage layer is present.  Reusing PandapowerStaticPlant keeps
    # this in step with the campaigns instead of duplicating its flags here.
    meta = tag_der_q_modes(
        net, meta,
        tso_q_mode=cfg.tso_q_mode,
        dso_q_mode=cfg.dso_q_mode,
        q_mode_overrides=cfg.der_q_mode_overrides,
        tso_qv_slope_pu=cfg.tso_qv_slope_pu,
        dso_qv_slope_pu=cfg.dso_qv_slope_pu,
        qv_slope_pu_overrides=cfg.der_qv_slope_pu_overrides,
        tso_qv_vref_pu=cfg.tso_qv_vref_pu,
        dso_qv_vref_pu=cfg.dso_qv_vref_pu,
        qv_vref_pu_overrides=cfg.der_qv_vref_pu_overrides,
    )

    _damp = min(float(cfg.qv_local_damping), 0.03)
    install_der_q_loops(
        net, list(meta.tso_der_indices),
        qv_damping=_damp,
        qv_max_step_frac=cfg.qv_local_max_step_frac,
        qv_tol_mvar=cfg.tso_qv_tol_mvar,
    )
    dso_sgens = [int(s) for s in meta.dso_der_indices]
    if dso_sgens:
        install_der_q_loops(
            net, dso_sgens,
            qv_damping=cfg.qv_local_damping,
            qv_max_step_frac=cfg.qv_local_max_step_frac,
            qv_tol_mvar=cfg.dso_qv_tol_mvar,
        )

    PandapowerStaticPlant(
        net,
        distributed_slack=cfg.distributed_slack,
        enforce_q_lims=cfg.enforce_q_lims_plant,
    ).advance(0.0)
    return net, meta


# ---------------------------------------------------------------------------
#  Selection
# ---------------------------------------------------------------------------


def candidate_buses(net, zone_buses: List[int]) -> List[Tuple[int, int]]:
    """Return ``(bus, v_idx)`` for every zone bus present in the reduced Jacobian.

    A bus with ``v_idx is None`` is PV or slack and is excluded automatically.
    """
    out: List[Tuple[int, int]] = []
    for b in zone_buses:
        _, v_idx = get_jacobian_indices(net, int(b))
        if v_idx is not None:
            out.append((int(b), int(v_idx)))
    return out


def select_pilot(
    dv_dq: np.ndarray,
    cands: List[Tuple[int, int]],
) -> Tuple[int, float, List[Tuple[int, float]]]:
    """Pick the pilot bus and return it with its score and the full ranking."""
    rows = [v for _, v in cands]
    n = dv_dq.shape[0]
    if rows and max(rows) >= n:
        raise IndexError(
            f"Jacobian row index {max(rows)} outside dV_dQ_reduced ({n} rows). "
            "The power flow and the sensitivity were built on different nets."
        )

    scored: List[Tuple[int, float]] = []
    for bus, col in cands:
        score = float(np.sum(np.abs(dv_dq[np.ix_(rows, [col])])))
        scored.append((bus, score))

    scored.sort(key=lambda t: t[1], reverse=True)
    best_bus, best_score = scored[0]
    return best_bus, best_score, scored


def main() -> None:
    net, meta = build_benchmark_net()
    zone_map, _bus_zone = fixed_zone_partition_ieee39(net, verbose=False)

    jac = JacobianSensitivities(net)
    dv_dq = jac.dV_dQ_reduced

    print(f"Reduced Jacobian dV/dQ : {dv_dq.shape[0]} x {dv_dq.shape[1]} (PQ buses)")
    print(f"Zones                  : {sorted(zone_map)}\n")

    result: Dict[str, Dict] = {}

    header = f"{'Zone':>4}  {'Pilot':>6}  {'kV':>6}  {'Score':>9}  {'|R_a|':>6}  {'Runner-up (score)':>22}"
    print(header)
    print("-" * len(header))

    for z in sorted(zone_map):
        cands = candidate_buses(net, zone_map[z])
        if not cands:
            raise ValueError(f"zone {z}: no PQ bus available as a pilot candidate")

        pilot, score, ranking = select_pilot(dv_dq, cands)
        vn_kv = float(net.bus.at[pilot, "vn_kv"])
        runner_up = ranking[1] if len(ranking) > 1 else (None, float("nan"))
        margin = (score / runner_up[1] - 1.0) * 100.0 if len(ranking) > 1 else float("nan")

        print(f"{z:>4}  {pilot:>6}  {vn_kv:>6.0f}  {score:>9.5f}  {len(cands):>6}  "
              f"{str(runner_up[0]):>6} ({runner_up[1]:.5f})")

        result[str(z)] = {
            "pilot_bus": pilot,
            "pilot_bus_vn_kv": vn_kv,
            "score": score,
            "n_candidates": len(cands),
            "candidates": [b for b, _ in cands],
            "ranking": [{"bus": b, "score": s} for b, s in ranking],
            "runner_up_bus": runner_up[0],
            "runner_up_score": runner_up[1],
            "margin_over_runner_up_pct": margin,
        }

    print()
    for z in sorted(zone_map):
        r = result[str(z)]
        print(f"  zone {z}: pilot bus {r['pilot_bus']} leads the runner-up by "
              f"{r['margin_over_runner_up_pct']:.1f} %")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"pilot_buses_{CONFIG_SOURCE}.json"
    payload = {
        "criterion": "argmax_p sum_{j in R_a} |dV_j/dQ_p| over TN PQ buses of the area",
        "scenario": SCENARIO,
        "config_source": CONFIG_SOURCE,
        "partition": "fixed 3-area IEEE 39 (network.zone_partition)",
        "note": "selected once at the base case and held fixed, as classical SVR does",
        "zones": result,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
