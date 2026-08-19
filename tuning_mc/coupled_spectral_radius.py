"""
tuning_mc/coupled_spectral_radius.py
===================================
The contraction of the *assembled* coupled operator, rather than the block
bound.

Why this exists
---------------
The coordinator's criterion is

    per zone i:   lambda_max(M_ii) + sum_{j != i} ||M_ij||_2   <=  bound

which is a **sufficient** condition: it bounds the spectral radius of the
assembled block operator ``M`` from above via a triangle inequality over the
off-diagonal blocks.  Under the shipped configuration
(``local_sensitivities_tso = true``) the off-diagonal blocks are zeroed before
the criterion is formed, so the sum term vanishes and the criterion collapses to
``max_i lambda_max(M_ii)`` -- the quantity the whole calibration is measured
against.

Measured 2026-08-17, retaining the off-diagonals at the calibrated point moves
the worst-zone criterion from 1.3488 to 2.1773, i.e. above the OFO bound of 2.
That is the *bound* moving, not necessarily the contraction: the triangle
inequality is loose whenever the coupling blocks do not align with the dominant
local mode.

So the question the design actually needs answering is not "what does the block
bound say" but "what is ``rho(M)``".  That is one eigenvalue computation on a
matrix the decomposition module already assembles block by block, and it is
computed here.

Usage::

    python -m tuning_mc.coupled_spectral_radius \\
        --x0 "lambda_tso=0.30,engage_tso_pu=0.017,lambda_dso=1.6,tau=1.0,\\
engage_dso_pu=0.035,dso_g_v_ratio=1.5"
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def assemble(blocks: dict[tuple, np.ndarray], zone_ids: list[int]) -> np.ndarray:
    """Stack the ``M_ij`` blocks into one square operator.

    Row block ``i`` is ``[M_i1 | M_i2 | ... ]``.  The result is square because
    every zone contributes the same column count to every row block: ``M_ij`` is
    ``(n_i x n_j)`` and the blocks tile exactly.
    """
    rows = []
    for i in zone_ids:
        row = [blocks[(i, j)] for j in zone_ids]
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main(argv: list[str] | None = None) -> int:
    from tuning_mc.stage_0_coupling_decomposition import (
        continuous_mask, criterion, m_blocks,
    )

    p = argparse.ArgumentParser(prog="tuning_mc.coupled_spectral_radius")
    p.add_argument("--baseline", type=Path,
                   default=_REPO_ROOT / "tuning" / "scripts" / "configs"
                   / "baseline_ieee39_thevenin.yaml")
    p.add_argument("--knobs", default=None,
                   help="Design coordinates, e.g. "
                        "'lambda_tso=0.30,engage_tso_pu=0.017,...'. Stage 0 is "
                        "run for these and the weights it designs are applied "
                        "before the criterion is formed. Omitting this "
                        "evaluates the BASELINE config's shipped weights, which "
                        "is a different operating point and must not be quoted "
                        "as the tuned one.")
    p.add_argument("--design-scenario", default="none")
    p.add_argument("--start-time", default=None,
                   help="ISO timestamp at which to build the plant, e.g. "
                        "'2016-02-03T20:00'. The coupling factor is only known "
                        "to be stable across WEIGHTS at one operating point; "
                        "H changes with loading, so it has to be checked across "
                        "operating points before being used as a constant.")
    p.add_argument("--network", default=None)
    p.add_argument("--workdir", type=Path,
                   default=_REPO_ROOT / "results" / "tuning_mc"
                   / "campaign_0815" / "designs")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    import dataclasses

    from tuning._io import load_config_yaml
    from tuning_mc.stage_1_search import (
        APPLIED_FIELDS, X0, build_config, design_payload,
    )

    from tuning._sim_loader import get_run_multi_tso_dso

    cfg = load_config_yaml(Path(args.baseline))
    if args.knobs:
        knobs = dict(X0)
        for kv in args.knobs.split(","):
            name, _, val = kv.partition("=")
            knobs[name.strip()] = float(val)
        payload = design_payload(knobs, baseline=Path(args.baseline),
                                 design_scenario=args.design_scenario,
                                 workdir=args.workdir)
        block = payload["config_block"]
        weights = {f: float(block[f]["designed"]) for f in APPLIED_FIELDS}
        cfg = build_config(knobs, weights, cfg)
        print(f"[rho] knobs { {k: round(v, 5) for k, v in knobs.items()} }")
        print(f"[rho] weights { {k: round(v, 4) for k, v in weights.items()} }")
    else:
        print("[rho] NO --knobs given: evaluating the baseline config's "
              "shipped weights, not a tuned point.")
    overlay: dict[str, Any] = dict(
        n_total_s=60.0, contingencies=[], verbose=0,
        live_plot_controller=False, live_plot_cascade=False,
        live_plot_system=False, run_stability_analysis=False,
        precondition_g_w=False,
    )
    if args.start_time:
        from datetime import datetime
        overlay["start_time"] = datetime.fromisoformat(args.start_time)
    if args.network:
        overlay["scenario"] = args.network
    cfg = dataclasses.replace(cfg, **overlay)
    print(f"[rho] operating point: {cfg.start_time}  network {cfg.scenario}")
    captured: dict[str, Any] = {}

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

    out: dict[str, Any] = {}
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
        # PSD (then the eigenvalues of I - M are 1 - lambda_i, all real).  The
        # assembled coupled M is not symmetric, so the two part company and
        # rho(M) < 2 is neither necessary nor sufficient.
        # M is singular: actuator directions that produce no output change (a
        # null space of H -- co-located DERs are the usual cause) give M a zero
        # eigenvalue, hence I - M an eigenvalue of exactly 1.  Those modes
        # neither decay nor grow and are not a stability finding; the
        # coordinator applies the same filter.  The test must therefore be run
        # on the observable subspace, i.e. over the non-zero eigenvalues.
        A = np.eye(n) - M
        tol = 1e-8 * max(np.abs(eig).max(), 1e-30)
        nz = eig[np.abs(eig) > tol]
        n_null = int(n - nz.size)
        rho_cl_all = float(np.max(np.abs(np.linalg.eigvals(A))))
        rho_cl = float(np.max(np.abs(1.0 - nz))) if nz.size else 0.0

        # Transient bound on the observable subspace: project onto range(M)
        # using the singular vectors with non-negligible singular value.
        U, sv, _ = np.linalg.svd(M)
        keep = sv > 1e-8 * max(sv.max(), 1e-30)
        P = U[:, keep] @ U[:, keep].conj().T
        nrm_cl = float(np.linalg.norm(P @ A @ P, 2))
        asym = float(np.linalg.norm(M - M.T, 2) / max(np.linalg.norm(M, 2), 1e-30))
        in_disc = bool(np.all(np.abs(1.0 - eig) < 1.0))
        max_im = float(np.max(np.abs(np.imag(eig))))

        bound = max(v["contraction_lhs"] for v in crit.values())
        local_only = max(v["lambda_max_Mii"] for v in crit.values())
        out[tag] = {"rho_assembled": rho, "block_bound": bound,
                    "max_lambda_Mii": local_only, "dim": int(n),
                    "rho_I_minus_M": rho_cl, "rho_I_minus_M_all": rho_cl_all,
                    "norm_I_minus_M": nrm_cl, "n_null": n_null,
                    "asymmetry": asym, "eigs_in_unit_disc": in_disc,
                    "max_abs_imag_eig": max_im}
        print(f"[{tag:>7}] dim {n} (null {n_null})  rho(M)={rho:.4f}   "
              f"rho(I-M)|obs={rho_cl:.4f} {'OK  ' if rho_cl < 1 else 'FAIL'}   "
              f"||I-M||2|obs={nrm_cl:.4f} {'OK  ' if nrm_cl < 1 else 'FAIL'}   "
              f"asym={asym:.3f}")

    c, l = out["coupled"], out["local"]
    print()
    print(f"[rho] the block bound is loose by a factor "
          f"{c['block_bound'] / c['rho_assembled']:.2f} "
          f"({c['block_bound']:.4f} -> {c['rho_assembled']:.4f})")
    print(f"[rho] coupling raises the true contraction by "
          f"{100 * (c['rho_assembled'] / l['rho_assembled'] - 1):+.1f} % "
          f"({l['rho_assembled']:.4f} -> {c['rho_assembled']:.4f})")
    for b in (1.0, 1.5, 2.0):
        print(f"[rho] rho(M) coupled {'<=' if c['rho_assembled'] <= b else '> '} "
              f"{b:g}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"[rho] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
