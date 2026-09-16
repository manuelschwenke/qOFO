r"""
tuning_mc/coupled_spectral_radius.py
====================================
Stage I-d: the contraction of the **assembled coupled** operator, rather than
the per-zone block bound.

Why this exists
---------------
The coordinator screens, per zone,

    lambda_max(M_ii) + sum_{j != i} ||M_ij||_2  <=  bound

which is a *sufficient* condition: it bounds the spectral radius of the
assembled operator ``M`` from above via a triangle inequality over the
off-diagonal blocks.  Under the shipped configuration
(``local_sensitivities_tso = True``) the off-diagonal blocks are zeroed before
the criterion is formed, so the coupling sum vanishes identically and the
criterion collapses to ``max_i lambda_max(M_ii)`` -- which is
``lambda_max(B)`` bit-for-bit.  Coupling is therefore neglected *in practice*
by the architecture, not by the theory.

Measured 2026-08-17 on the pre-reset design, retaining the off-diagonals moved
the worst-zone criterion from 1.3488 to 2.1773.  That is the *bound* moving,
not necessarily the contraction: the triangle inequality is loose whenever the
coupling blocks do not align with the dominant local mode.  So the question the
design actually needs answering is not what the block bound says but what
``rho(M)`` is, which is one eigenvalue computation on a matrix the blocks
already tile.

**This is a reported diagnostic, not a screen.**  ``A_cl = I - M`` is not
symmetric once the off-diagonals are retained, so ``lambda(B) < 2`` is neither
necessary nor sufficient for it; the meaningful tests are ``rho(A_cl) < 1``
(asymptotic decay) and the stronger ``||A_cl||_2 < 1`` (contraction at every
step), both on the observable subspace.  Chapter 9 records that the coupled
calculation was never run at the frozen weights -- its invocation omitted both
the selected coordinates and the frozen configuration factory -- so its numbers
were never assignable to the selected point.  This module exists to close that
gap, which means it must be pointed at a *parameter set*.

Rewritten 2026-09-10 with the weight reset: it previously took campaign search
knobs and rebuilt the weights through ``tuning_mc.stage_1_search``, both of
which are retired.  The block algebra (``m_blocks``, ``criterion``) is carried
over from the retired ``tuning_mc.stage_0_coupling_decomposition`` unchanged and
now lives here, so this module is self-contained.

Usage::

    python -m tuning_mc.coupled_spectral_radius --paramset ch9_selected \
        --start-time 2016-01-05T08:00
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Block algebra (carried over from stage_0_coupling_decomposition)
# ---------------------------------------------------------------------------

def m_blocks(coord, gw_by_zone: Dict[int, np.ndarray]) -> Dict[tuple, np.ndarray]:
    """``M_ij = G_w,i^{-1/2} H_ii^T Q_i H_ij G_w,j^{-1/2}`` for supplied weights.

    Mirrors ``MultiTSOCoordinator.compute_M_blocks`` rather than calling it, so
    the zone objects are never mutated: this module must be able to evaluate a
    hypothetical weighting without leaving the coordinator in that state.
    """
    out: Dict[tuple, np.ndarray] = {}
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


def criterion(coord, blocks: Dict[tuple, np.ndarray]) -> Dict[int, dict]:
    """Per zone: ``lambda_max(M_ii) + sum_{j != i} ||M_ij||_2``."""
    zone_ids = sorted(coord.zones.keys())
    res: Dict[int, dict] = {}
    for i in zone_ids:
        M_ii = blocks.get((i, i))
        if M_ii is None:
            continue
        eig = np.linalg.eigvalsh(M_ii)
        lam_all = float(max(eig[-1], 0.0))
        # Same near-zero filter the coordinator applies: co-located DER leave a
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


def assemble(blocks: Dict[tuple, np.ndarray], zone_ids: list) -> np.ndarray:
    """Stack the ``M_ij`` blocks into one square operator.

    Row block ``i`` is ``[M_i1 | M_i2 | ...]``.  The result is square because
    every zone contributes the same column count to every row block.
    """
    rows = []
    for i in zone_ids:
        rows.append(np.hstack([blocks[(i, j)] for j in zone_ids]))
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuning_mc.coupled_spectral_radius")
    p.add_argument("--paramset", default="ch9_selected",
                   help="Parameter set to evaluate. The weights are applied "
                        "before the criterion is formed; there is no fallback, "
                        "because a coupled figure quoted against the wrong "
                        "weights is what this module exists to stop.")
    p.add_argument("--start-time", default=None,
                   help="ISO timestamp of the operating point, e.g. "
                        "'2016-01-05T08:00'. H changes with loading, so the "
                        "coupling factor must be checked across conditions "
                        "rather than assumed constant.")
    p.add_argument("--condition", default=None,
                   help="Name from tuning_mc.scenarios_mc_v2.CONDITIONS "
                        "(W1..W6); an alternative to --start-time.")
    p.add_argument("--network", default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    # The STRUCTURAL configuration must come from the project factory -- shunt
    # dispatch, boundary equivalent, zone partition, profile use.  A bare
    # MultiTSOConfig() is a different plant (no tertiary shunts, PQ boundary
    # instead of Thevenin, profiles off) and its power flow does not even
    # converge at these conditions, so a coupled figure computed on it would
    # describe a system nobody runs.
    from experiments.run_multi_system_ofo import make_config
    from configs.paramsets import apply_paramset
    from tuning._sim_loader import get_run_multi_tso_dso

    cfg = apply_paramset(make_config(), args.paramset, verbose=False)

    overlay: Dict[str, Any] = dict(
        n_total_s=60.0, contingencies=[], verbose=0,
        live_plot_controller=False, live_plot_cascade=False,
        live_plot_system=False, run_stability_analysis=False,
        precondition_g_w=False,
    )
    if args.condition:
        from tuning_mc.scenarios_mc_v2 import CONDITIONS
        row = next((r for r in CONDITIONS if r[0] == args.condition), None)
        if row is None:
            raise SystemExit(f"unknown condition {args.condition!r}")
        overlay["start_time"] = row[1]
    elif args.start_time:
        from datetime import datetime
        overlay["start_time"] = datetime.fromisoformat(args.start_time)
    if args.network:
        overlay["scenario"] = args.network
    cfg = dataclasses.replace(cfg, **overlay)

    print(f"[rho] paramset {args.paramset}  operating point {cfg.start_time}  "
          f"network {cfg.scenario}")

    captured: Dict[str, Any] = {}

    def hook(state):
        captured.update(state)
        return True

    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        get_run_multi_tso_dso()(cfg, pre_loop_hook=hook)
    coord = captured.get("coordinator")
    if coord is None:
        raise SystemExit("[rho] no coordinator exposed")

    zone_ids = sorted(coord.zones.keys())
    gw = {i: np.asarray(coord.zones[i].gw_diagonal(), float) for i in zone_ids}

    out: Dict[str, Any] = {}
    for tag, zero in (("local", True), ("coupled", False)):
        with contextlib.redirect_stdout(io.StringIO()):
            coord.compute_cross_sensitivities(zero_offdiag=zero)
        blocks = m_blocks(coord, gw)
        crit = criterion(coord, blocks)
        M = assemble(blocks, zone_ids)
        n = M.shape[0]
        eig = np.linalg.eigvals(M)
        rho = float(np.max(np.abs(eig)))

        # The quantity that actually decides convergence.  The error iteration
        # is e+ = (I - M) e, so:
        #   rho(I - M) < 1   asymptotic decay of the LTI iteration;
        #   ||I - M||_2 < 1  one-step Euclidean contraction, and the only one of
        #                    the two that survives re-linearisation, because a
        #                    product of matrices each of norm < 1 contracts
        #                    whatever order they arrive in.
        # rho(M) < 2 is equivalent to rho(I - M) < 1 ONLY when M is symmetric
        # PSD.  The assembled coupled M is not symmetric, so the two part
        # company and rho(M) < 2 is neither necessary nor sufficient.
        # M is singular: actuator directions producing no output change (a null
        # space of H -- co-located DER are the usual cause) give M a zero
        # eigenvalue, hence I - M an eigenvalue of exactly 1.  Those modes
        # neither decay nor grow and are not a stability finding, so the test
        # runs on the observable subspace.
        A = np.eye(n) - M
        tol = 1e-8 * max(np.abs(eig).max(), 1e-30)
        nz = eig[np.abs(eig) > tol]
        n_null = int(n - nz.size)
        rho_cl_all = float(np.max(np.abs(np.linalg.eigvals(A))))
        rho_cl = float(np.max(np.abs(1.0 - nz))) if nz.size else 0.0

        U, sv, _ = np.linalg.svd(M)
        keep = sv > 1e-8 * max(sv.max(), 1e-30)
        P = U[:, keep] @ U[:, keep].conj().T
        nrm_cl = float(np.linalg.norm(P @ A @ P, 2))
        asym = float(np.linalg.norm(M - M.T, 2) / max(np.linalg.norm(M, 2), 1e-30))
        max_im = float(np.max(np.abs(np.imag(eig))))

        bound = max(v["contraction_lhs"] for v in crit.values())
        local_only = max(v["lambda_max_Mii"] for v in crit.values())
        out[tag] = {"rho_assembled": rho, "block_bound": bound,
                    "max_lambda_Mii": local_only, "dim": int(n),
                    "rho_I_minus_M": rho_cl, "rho_I_minus_M_all": rho_cl_all,
                    "norm_I_minus_M": nrm_cl, "n_null": n_null,
                    "asymmetry": asym, "max_abs_imag_eig": max_im,
                    "per_zone": {str(k): v for k, v in crit.items()}}
        print(f"[{tag:>7}] dim {n} (null {n_null})  rho(M)={rho:.4f}   "
              f"rho(I-M)|obs={rho_cl:.4f} {'OK  ' if rho_cl < 1 else 'FAIL'}   "
              f"||I-M||2|obs={nrm_cl:.4f} {'OK  ' if nrm_cl < 1 else 'FAIL'}   "
              f"asym={asym:.3f}")

    c, l = out["coupled"], out["local"]
    print()
    print(f"[rho] the block bound is loose by a factor "
          f"{c['block_bound'] / max(c['rho_assembled'], 1e-30):.2f} "
          f"({c['block_bound']:.4f} -> {c['rho_assembled']:.4f})")
    print(f"[rho] coupling raises the true contraction by "
          f"{100 * (c['rho_assembled'] / max(l['rho_assembled'], 1e-30) - 1):+.1f} % "
          f"({l['rho_assembled']:.4f} -> {c['rho_assembled']:.4f})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"paramset": args.paramset,
                   "start_time": str(cfg.start_time),
                   "scenario": cfg.scenario, **out}
        args.out.write_text(json.dumps(payload, indent=1, default=float),
                            encoding="utf-8")
        print(f"[rho] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
