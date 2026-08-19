"""
tuning_mc/stage_0_coupling_decomposition.py
===========================================
Close the decomposition of the measured contraction intercept.

The problem
-----------
The lambda calibration measures ``rho_emp_p95 = 1.170 + 1.617 * lambda_tso`` and
the intercept is what decides feasibility -- it is the contraction the loop
retains when the continuous design gain goes to zero, and ``rho <= 1`` is
unreachable because of it.  Attributing that intercept is the whole question,
because what it is made of decides what could be done about it.

``stage_0_lambda_curve`` gave the first piece: with the preconditioned columns'
weights sent to infinity, the cached model's own floor is ~0.55 in the binding
zone -- about half the measured 1.170.  But that number is **local to one zone**,
and the criterion the plant is graded against is not:

    contraction_lhs_i = lambda_max(M_ii) + sum_{j != i} ||M_ij||_2

(:meth:`controller.multi_tso_coordinator.MultiTSOCoordinator.check_contraction`).
The coupling term is structurally invisible to a per-controller calculation, so
"the rest is coupling plus model gap" was a hypothesis, not a measurement.

What this module does
---------------------
It recomputes the coordinator's own blocks under two weight policies and reports
the criterion *exactly as the coordinator forms it*, coupling included:

``designed``   every weight as the campaign actually ran it (Stage 0's designed
               values, not the baseline config's -- these differ, and the tap
               weights are what the floor is inversely proportional to);
``floor``      the continuous columns (``Q_DER``, ``Q_PCC``) sent to infinity,
               which removes their rank-one terms from every ``M_ij`` exactly,
               leaving what the columns the curvature rule never prices
               contribute on their own.

The difference between the two policies' local terms and their coupling terms
splits the intercept three ways:

    measured intercept  =  local floor  +  coupling floor  +  (live H, p95 over
                           zones and time, and anything else the cache cannot see)

Only the first two are analytic.  The third is named rather than computed, and
its size is the honest statement of how far the cached model is from the plant.

Cost: one controller build (~60 s).  Nothing is simulated.

Usage::

    python -m tuning_mc.stage_0_coupling_decomposition \
        --x0 "lambda_tso=0.2,lambda_dso=1.0" --measured-intercept 1.1702

Author: Manuel Schwenke (with Claude Code), 2026-08-14
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_OUT = (_REPO_ROOT / "results" / "tuning_mc" / "campaign_0814"
               / "coupling_decomposition.json")


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def continuous_mask(zone) -> np.ndarray:
    """Columns the curvature rule scales, in the zone's own ordering.

    ``ZoneDefinition.gw_diagonal`` lays the vector out as
    ``[Q_DER | Q_PCC_set | V_gen | s_OLTC | s_shunt]``.  The rule scales the
    reactive-power classes only: ``gen`` is in ``precondition_exclude_classes``
    and the tap and shunt columns are integers, priced by the commit rule
    instead.  Those three are exactly the "fixed" set whose contribution the
    floor measures.
    """
    n_der = len(zone.tso_der_indices)
    n_pcc = len(zone.pcc_trafo_indices)
    n_gen = len(zone.gen_indices)
    n_oltc = len(zone.oltc_trafo_indices)
    n_shunt = len(zone.shunt_bus_indices)
    mask = np.zeros(n_der + n_pcc + n_gen + n_oltc + n_shunt, dtype=bool)
    mask[: n_der + n_pcc] = True
    return mask


def m_blocks(coord, gw_by_zone: dict[int, np.ndarray]) -> dict[tuple, np.ndarray]:
    """``M_ij = G_w,i^{-1/2} H_ii^T Q_i H_ij G_w,j^{-1/2}``, for supplied weights.

    Mirrors :meth:`MultiTSOCoordinator.compute_M_blocks` rather than calling it,
    so the zone objects are never mutated: this module must be able to evaluate
    a hypothetical weighting without leaving the coordinator in that state.
    """
    out: dict[tuple, np.ndarray] = {}
    zone_ids = sorted(coord.zones.keys())
    for i in zone_ids:
        zi = coord.zones[i]
        q_i = np.asarray(zi.q_obj_diagonal(), float)
        gw_i_inv_sqrt = 1.0 / np.sqrt(np.maximum(gw_by_zone[i], 1e-12))
        H_ii = coord._H_blocks.get((i, i))
        if H_ii is None:
            continue
        QH_ii = np.sqrt(np.maximum(q_i, 0.0))[:, None] * H_ii
        for j in zone_ids:
            H_ij = coord._H_blocks.get((i, j))
            if H_ij is None:
                continue
            gw_j_inv_sqrt = 1.0 / np.sqrt(np.maximum(gw_by_zone[j], 1e-12))
            QH_ij = np.sqrt(np.maximum(q_i, 0.0))[:, None] * H_ij
            C_ij = QH_ii.T @ QH_ij
            out[(i, j)] = (gw_i_inv_sqrt[:, None] * C_ij) * gw_j_inv_sqrt[None, :]
    return out


def criterion(coord, blocks: dict[tuple, np.ndarray]) -> dict[int, dict]:
    """Per zone: the coordinator's own ``lambda_max(M_ii) + sum ||M_ij||_2``."""
    zone_ids = sorted(coord.zones.keys())
    res: dict[int, dict] = {}
    for i in zone_ids:
        M_ii = blocks.get((i, i))
        if M_ii is None:
            continue
        eig = np.linalg.eigvalsh(M_ii)
        lam_all = float(max(eig[-1], 0.0))
        # Same near-zero filter the coordinator applies: co-located DERs leave a
        # null space that would otherwise report as a spurious mode.
        tol = 1e-10 * max(lam_all, 1e-14)
        kept = eig[eig > tol]
        lam = float(kept[-1]) if kept.size else 0.0
        coup = sum(float(np.linalg.norm(blocks[(i, j)], ord=2))
                   for j in zone_ids
                   if j != i and blocks.get((i, j)) is not None
                   and blocks[(i, j)].size)
        res[i] = {"lambda_max_Mii": lam, "coupling_sum": coup,
                  "contraction_lhs": lam + coup}
    return res


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    from tuning_mc.stage_1_search import (
        DEFAULT_BASELINE, X0, apply_x0_override, build_config, design_weights,
    )

    p = argparse.ArgumentParser(
        prog="tuning_mc.stage_0_coupling_decomposition")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--x0", default=None,
                   help="Design point, e.g. 'lambda_tso=0.2,lambda_dso=1.0'.")
    p.add_argument("--design-scenario", default="none")
    p.add_argument("--designs", type=Path,
                   default=_REPO_ROOT / "results" / "tuning_mc" / "stage1" / "designs")
    p.add_argument("--measured-intercept", type=float, default=None,
                   help="The intercept of the measured rho(lambda) fit, for the "
                        "residual line. Not used in any computation.")
    p.add_argument("--measured-rho", type=float, default=None,
                   help="Measured rho_emp_p95 at this design point, likewise.")
    p.add_argument("--with-coupling", action="store_true",
                   help="Recompute the cross-sensitivities with the "
                        "off-diagonal H_ij blocks RETAINED, instead of reusing "
                        "the runner's zeroed ones. Measures what neglecting the "
                        "physical connection to neighbouring areas costs in the "
                        "contraction criterion. Does not change the controller.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    if args.x0:
        apply_x0_override(args.x0)
    knobs = dict(X0)

    from tuning._io import load_config_yaml
    from tuning._sim_loader import get_run_multi_tso_dso

    baseline_cfg = load_config_yaml(Path(args.baseline))
    weights = design_weights(knobs, baseline=Path(args.baseline),
                             design_scenario=args.design_scenario,
                             workdir=Path(args.designs))
    # The campaign's OWN weights, not the baseline config's.  The floor scales as
    # 1/g_w on the tap columns, and Stage 0 designs g_w_tso_oltc = 3783 against
    # the config's 5000 -- evaluating the floor at the config value would
    # understate it by that ratio.
    cfg = build_config(knobs, weights, baseline_cfg)
    print(f"[decomp] knobs   : { {k: round(v, 5) for k, v in knobs.items()} }")
    print(f"[decomp] weights : { {k: round(v, 4) for k, v in weights.items()} }")

    captured: dict[str, Any] = {}

    def hook(state: dict[str, Any]) -> bool:
        captured["coordinator"] = state.get("coordinator")
        return True                      # abort before the time loop

    t0 = time.perf_counter()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        get_run_multi_tso_dso()(cfg, pre_loop_hook=hook)
    coord = captured.get("coordinator")
    if coord is None:
        raise SystemExit("[decomp] the runner exposed no coordinator; this "
                         "analysis needs the multi-TSO path.")
    print(f"[decomp] controllers built in {time.perf_counter() - t0:.0f} s")

    if args.with_coupling:
        # The runner builds ``_H_blocks`` with ``zero_offdiag=True`` whenever
        # ``local_sensitivities_tso`` is set, which is the shipped configuration.
        # Reusing those blocks makes ``coupling_sum`` identically zero -- not
        # because the areas are uncoupled, but because the controller is told
        # they are.  Recomputing with the off-diagonals retained measures what
        # that assumption costs: the criterion the *coordinator* would form if
        # it saw the physical connection to its neighbours.
        #
        # This does NOT change the controller.  It re-evaluates the contraction
        # of the same locally-designed loop against a less restricted model, so
        # the difference is the model gap, not a different design.
        with contextlib.redirect_stdout(io.StringIO()):
            coord.compute_cross_sensitivities(zero_offdiag=False)
    elif not getattr(coord, "_H_blocks", None):
        with contextlib.redirect_stdout(io.StringIO()):
            coord.compute_cross_sensitivities()
    zone_ids = sorted(coord.zones.keys())

    gw_designed = {i: np.asarray(coord.zones[i].gw_diagonal(), float)
                   for i in zone_ids}
    gw_floor = {}
    for i in zone_ids:
        g = gw_designed[i].copy()
        mask = continuous_mask(coord.zones[i])
        if mask.size != g.size:
            raise SystemExit(f"[decomp] zone {i}: column layout mismatch "
                             f"({mask.size} vs {g.size}); the gw_diagonal "
                             f"ordering has changed and this mask is stale.")
        g[mask] = np.inf
        gw_floor[i] = g

    res_des = criterion(coord, m_blocks(coord, gw_designed))
    res_flr = criterion(coord, m_blocks(coord, gw_floor))

    print(f"\n{'zone':>5}  {'--- as designed ---':^34}  {'--- floor (continuous -> inf) ---':^34}")
    print(f"{'':>5}  {'lam_max':>10}{'coupling':>11}{'lhs':>11}  "
          f"{'lam_max':>12}{'coupling':>11}{'lhs':>11}")
    for i in zone_ids:
        d, f = res_des.get(i), res_flr.get(i)
        if d is None or f is None:
            continue
        print(f"{i:>5}  {d['lambda_max_Mii']:>10.4f}{d['coupling_sum']:>11.4f}"
              f"{d['contraction_lhs']:>11.4f}  "
              f"{f['lambda_max_Mii']:>12.4f}{f['coupling_sum']:>11.4f}"
              f"{f['contraction_lhs']:>11.4f}")

    worst_des = max(res_des.values(), key=lambda v: v["contraction_lhs"])
    worst_flr = max(res_flr.values(), key=lambda v: v["contraction_lhs"])
    print(f"\n[decomp] worst zone, as designed: lhs = "
          f"{worst_des['contraction_lhs']:.4f} "
          f"(local {worst_des['lambda_max_Mii']:.4f} + coupling "
          f"{worst_des['coupling_sum']:.4f})")
    print(f"[decomp] worst zone, floor      : lhs = "
          f"{worst_flr['contraction_lhs']:.4f} "
          f"(local {worst_flr['lambda_max_Mii']:.4f} + coupling "
          f"{worst_flr['coupling_sum']:.4f})")

    payload: dict[str, Any] = {
        "knobs": knobs, "weights": weights,
        "designed": {str(k): v for k, v in res_des.items()},
        "floor": {str(k): v for k, v in res_flr.items()},
        "worst_designed": worst_des, "worst_floor": worst_flr,
    }

    if args.measured_intercept is not None:
        gap = args.measured_intercept - worst_flr["contraction_lhs"]
        payload["measured_intercept"] = args.measured_intercept
        payload["unexplained_intercept"] = gap
        print(f"\n[decomp] INTERCEPT DECOMPOSITION (measured "
              f"{args.measured_intercept:.4f})")
        print(f"    local, fixed columns      {worst_flr['lambda_max_Mii']:>8.4f}")
        print(f"    inter-zone coupling       {worst_flr['coupling_sum']:>8.4f}")
        print(f"    ------------------------- {'':>8}")
        print(f"    analytic, cached model    {worst_flr['contraction_lhs']:>8.4f}"
              f"   ({100 * worst_flr['contraction_lhs'] / args.measured_intercept:.0f} %"
              f" of measured)")
        print(f"    live H, p95 over zones\n"
              f"      and time, model gap     {gap:>8.4f}"
              f"   ({100 * gap / args.measured_intercept:.0f} %)")
    if args.measured_rho is not None:
        gap = args.measured_rho - worst_des["contraction_lhs"]
        payload["measured_rho"] = args.measured_rho
        payload["unexplained_rho"] = gap
        print(f"\n[decomp] AT THE DESIGN POINT (measured rho "
              f"{args.measured_rho:.4f})")
        print(f"    analytic, cached model    {worst_des['contraction_lhs']:>8.4f}")
        print(f"    unexplained               {gap:>8.4f}"
              f"   (factor {args.measured_rho / worst_des['contraction_lhs']:.2f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1, default=float),
                              encoding="utf-8")
    print(f"\n[decomp] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
