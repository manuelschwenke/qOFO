"""
tuning_mc/stage_0_preconditioning.py
====================================
Stage 0 — analytic weight design from the cached sensitivities, for **both**
actuator kinds.

Continuous classes: the curvature rule (existing)
-------------------------------------------------
One unconstrained OFO tick gives ``e_{k+1} = (I - M) e_k`` with
``M = H_y G_w^{-1} H_y^T diag(g_y)``, so with ``a_i`` the i-th column of
``D_y^{1/2} H_y``

    M_sym = sum_i (1/g_w_i) a_i a_i^T .

Setting ``g_w_i ∝ ||a_i||^2`` equalises the per-actuator contributions
(conditioning) and one global ``kappa`` places ``lambda_max(M)`` on a target
(gain).  This is :func:`controller.gw_precondition.precondition_g_w`, used here
unchanged so Stage 0 reports exactly what the runner would apply.

Integer classes: the commit-threshold rule (new)
------------------------------------------------
The curvature rule does not transfer: the MIQP rounds an integer actuator onto a
lattice, so ``g_w`` is not a step size there.  It is a **price**, and what it
sets is the error at which one move pays for itself.

For a single move ``w_i = ±1`` on integer column ``i``, with weighted residual
``ytil = D_y^{1/2}(y - y*)`` the objective changes by

    dJ = ||ytil - a_i w_i||^2 - ||ytil||^2 = -2 w_i a_i^T ytil + ||a_i||^2 ,

minimised over the sign at ``w_i = sign(a_i^T ytil)``, so the move is taken iff

    2 |a_i^T ytil| - ||a_i||^2  >  g_w_i .

Writing ``p_i = |a_i^T ytil| / ||a_i||`` — the component of the weighted error
along that column — the move happens iff

    p_i  >  p*_i = ||a_i|| / 2  +  g_w_i / (2 ||a_i||)                      (T)

``p*_i`` is the **commit threshold**: the projected error at which this
transformer taps.  Inverting (T) gives the design rule

    g_w_i = ||a_i|| * (2 p*_i - ||a_i||)                                    (T')

Fixing one ``p*`` per class and applying (T') per column equalises the commit
threshold across transformers, which a single shared scalar does not: a
strong transformer (large ``||a_i||``) taps at a *lower* error than a weak one
unless it is priced proportionally higher.

Units.  ``p*`` is in weighted-residual units.  It is reported here in the
physical unit of whichever output block dominates that column: pu voltage
(divide by ``sqrt(g_v)``) or Mvar of interface-Q (divide by ``sqrt(g_q)``).  The
distinction is not cosmetic — for a DSO the interface-Q block outweighs voltage
~500x in priority terms, so a DSO tap threshold expressed in pu would be
meaningless.

Feasibility of a requested threshold.  (T') returns a positive weight only for
``p* > ||a_i||/2``: below that, one tap moves the output further than the
deadband being asked for, so no positive price can realise it.  The script
flags this rather than clipping silently.

Continuous classes, second rule: the move budget (this is what sets ``g_w_gen``)
-------------------------------------------------------------------------------
The curvature rule fixes the closed-loop *gain*.  It says nothing about how far
one actuator travels in one tick, and it is not applied to the AVR class at all
— ``precondition_exclude_classes = ("gen",)``, because a column carrying ~10^10x
a DER's energy would otherwise set ``kappa`` for everyone.  ``g_w_gen`` is
consequently the one weight in the file with no rule behind it.  This is the
rule.

The MIQP minimises ``w^T G_w w + grad_f^T w`` with ``G_w`` diagonal and
``grad_f = 2 A^T ytil`` (:mod:`optimisation.miqp_solver`, ``build_miqp_problem``
/ ``grad_f += 2 g_v (V - V_set) dV/du``).  The tracking term is **linear** in
``w``, so the continuous columns decouple; with ``alpha = 1`` and ``g_u = 0``
(both pinned by the runner for every TSO and DSO controller) the unconstrained
step of continuous column ``i`` is

    du_i = - (a_i^T ytil) / g_w_i                                          (S)

*exactly*, not to first order.  Actuator bounds and the output constraints can
only shrink it, so (S) is an upper bound on the realised move whenever the MIQP
is interior — which is the regime the weight is supposed to govern.

A move budget ``|du_i| <= du_max`` is therefore a statement about the largest
error the loop is designed to absorb in one tick.  Bounding every controlled bus
by ``|e_k| <= d_ref`` makes ``|a_i^T ytil|`` maximal at adversarial signs, which
is the l1 reading and the only one that is a *guarantee*:

    g_w_i  >=  g_v * (sum_k |h_ki|) * d_ref / du_max                       (S')

Two weaker readings are reported next to it, both special cases of the same
projection: ``max_k |h_ki|`` (only the strongest bus is off reference) and
``|sum_k h_ki|`` (every bus off by the same ``+d`` — the systematic offset an
AVR exists for, and the one to design on if the box bound is judged paranoid).
Inverting (S') gives ``d@limit``, the per-bus deviation at which the *shipped*
weight first permits a full ``du_max`` step.  That is the direct answer to "is
``g_w_gen = 1e9`` the right number": it is, iff ``d@limit`` sits comfortably
outside the voltage corridor the loop actually operates in.

Note where this differs from (T).  (T) carries a ``||a_i||^2`` self-cost, which
follows from modelling the objective as ``||ytil - A w||^2``.  The MIQP itself
linearises the tracking term and has no such quadratic, so (S) has no
counterpart to it; the two sections are not using the same objective model.

Output weights ``g_v`` / ``g_q`` / ``dso_g_v``: what can and cannot be derived
------------------------------------------------------------------------------
These are not step sizes and neither rule above sets them, but they are not
free either, and the report says exactly which part of them is arbitrary.

*Gauge.*  ``a_i^T ytil`` is linear in ``g_y``, so under
``(g_y, g_w) -> (c g_y, c g_w)`` every step (S), every commit test (T) and
``lambda_max(M) = lambda_max(H G_w^-1 H^T diag(g_y))`` are unchanged.  The
absolute level of ``g_v`` is therefore **not a tuned quantity** — raising it
with ``g_w`` fixed is the same experiment as lowering ``g_w``.  Two things are
observable: the ratio ``g_y/g_w`` (the loop gain, which the curvature rule
already sets) and the ratios *between output blocks*.  The gauge itself has one
legitimate criterion, numerical: choose ``c`` so the ``G_w`` diagonal straddles
1 rather than spanning ten decades around it.

*Trade-off.*  The block ratio is physical and is the only genuinely free
objective choice.  Read it as an inverse-square-tolerance (Bryson) pair,
``g_block = 1/sigma_block^2``: the weighted residual then counts tolerances
rather than mixed units, and fixing one tolerance fixes the other,

    g_q / g_v = (sigma_V / sigma_Q)^2 .

The report inverts this — it fixes ``sigma_V`` at ``--vtol-pu`` and prints the
``sigma_Q`` the *shipped* weights already imply, together with the exchange rate
"1 Mvar of interface-Q error is worth this much voltage error".  That is the
defensible form of the argument: the number to defend in writing is a
tolerance, not a weight.

*Realised balance.*  What the loop actually prioritises need not match what the
weights were meant to express.  The row analogue of the column-norm rule,

    E_block = sum_{k in block} g_y_k ||H[k,:]||^2 ,

is that block's share of the objective curvature and is reported per loop.  It
is where a claim such as "the DSO's interface-Q block outweighs voltage ~500x
in priority terms" has to come from, rather than from the weight ratio alone,
which ignores how much output each actuator can actually move.

Aggregation, and the per-area refinement
----------------------------------------
Both rules are per-column while ``MultiTSOConfig`` carries one scalar per class,
so the default report compresses the design onto the config shape and prints the
loss.  ``--per-area`` adds the intermediate level — one value per TSO zone and
per DSO area — together with the single-factor re-gain that realises it with the
shipped config (``zone_g_w_scale``, which multiplies one zone's whole ``g_w``
vector) and the residual spread that a single factor per area cannot absorb.
The factor is the log-least-squares optimum, i.e. the geometric mean of the
per-column ratio ``designed_col / global_class_scalar`` over that area.  DSO
areas are reported the same way but ``MultiTSOConfig`` has no ``dso_g_w_scale``
counterpart, so those numbers are informational only.

Caveats, stated rather than assumed
-----------------------------------
* (T) is a *single-move, single-tick* condition on a locally quadratic
  objective.  It ignores that the continuous actuators move in the same tick and
  may remove the error before the tap commits, and that the MIQP may prefer two
  taps at once (``int_max_step`` permitting).  It gives a defensible **base
  value**, not a certificate.
* ``H`` is the controller's cached sensitivity, so every number here inherits
  the cache's model error and the operating point at which it was taken.  The
  operating point is a CLI argument and is stamped on the output.
* The shunt integrator is deliberately not covered: in ``integrator`` dispatch
  its banks live outside the MIQP and its gain has its own hysteresis/dwell
  logic, which needs its own derivation.

Usage::

    python -m tuning_mc.stage_0_preconditioning
    python -m tuning_mc.stage_0_preconditioning --scenario v2_undervoltage_ramp \
        --lambda-tso 0.9 --lambda-dso 0.9 --tau 1.0 \
        --p-star-tso-pu 0.02 --p-star-dso-mvar 2.0 --out stage0.json
    python -m tuning_mc.stage_0_preconditioning --per-area \
        --max-move-gen-pu 0.001 --dref-pu 0.02

Author: Manuel Schwenke (with Claude Code), 2026-08-14
Revised 2026-08-14: per-area refinement (--per-area) and the continuous move
budget that designs ``g_w_gen`` (--max-move-gen-pu / --dref-pu).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_BASELINE = (_REPO_ROOT / "tuning" / "scripts" / "configs"
                    / "baseline_ieee39_thevenin.yaml")


# ---------------------------------------------------------------------------
# Row-block identification
# ---------------------------------------------------------------------------

def _voltage_row_slice(H_y: np.ndarray, H_v: np.ndarray | None) -> slice | None:
    """Locate the voltage rows inside the stacked objective block.

    The two controllers stack their outputs in opposite orders — TSO
    ``[V | Q_PCC | ...]``, DSO ``[Q_interface | V | ...]`` — so the slice is
    identified by matching against ``voltage_curvature_inputs`` rather than
    assumed.  Returns ``None`` if no voltage block is present or it cannot be
    matched.
    """
    if H_v is None or H_v.size == 0:
        return None
    n_v = H_v.shape[0]
    if n_v == 0 or n_v > H_y.shape[0]:
        return None
    if np.allclose(H_y[:n_v, :], H_v, rtol=1e-9, atol=1e-12):
        return slice(0, n_v)
    if np.allclose(H_y[-n_v:, :], H_v, rtol=1e-9, atol=1e-12):
        return slice(H_y.shape[0] - n_v, H_y.shape[0])
    return None


def _fixed_column_floor(H_y: np.ndarray, g_y: np.ndarray,
                        g_w: np.ndarray, pre_idx: np.ndarray) -> float:
    """``lambda_max(M)`` from the columns the curvature rule does not scale.

    Sending the preconditioned columns' weights to infinity removes their
    rank-one terms from ``M = sum_i g_w_i^{-1} a_i a_i^T`` exactly, leaving what
    the integer taps and the excluded AVR class contribute on their own.  That
    is the floor: no continuous weight can push the loop gain below it, so a
    design target under it is unreachable however the rule is scaled.

    Caveat inherited from the model: ``M`` treats an integer OLTC column as a
    continuous per-tick move, while the actuator steps at most one tap per
    cooldown.  The floor is therefore an **upper bound** on what the taps really
    contribute, and is weak evidence on its own -- which is why the campaign
    measures ``rho_emp_p95`` rather than stopping here.
    """
    from controller.gw_precondition import curvature_spectrum
    if pre_idx.size == 0:
        return float("nan")
    g_inf = np.asarray(g_w, float).copy()
    g_inf[np.asarray(pre_idx, dtype=int)] = np.inf
    return float(curvature_spectrum(H_y, g_y, g_inf).lambda_max)


def _analyse_controller(
    ctrl: Any,
    label: str,
    *,
    kind: str,
    area: Any,
    lambda_target: float,
    tau: float,
    p_star_pu: float,
    p_star_mvar: float,
    engage_pu: float,
    exclude_classes: tuple[str, ...],
    granularity: str,
    floor_frac: float,
    floor_scope: str = "preconditioned",
    d_ref_pu: float = 0.02,
    d_ref_q_mvar: float = 5.0,
    max_move_gen_pu: float = 1e-3,
    max_move_q_mvar: float | None = None,
) -> dict[str, Any]:
    """Continuous curvature rule + move budget + integer commit-threshold rule."""
    from controller.gw_precondition import precondition_g_w

    ident = {"label": label, "kind": kind, "area": area}
    vci = ctrl.objective_curvature_inputs()
    if vci is None:
        return {**ident, "status": "no_curvature"}
    H_y, g_y = np.asarray(vci[0], float), np.asarray(vci[1], float)

    try:
        vv = ctrl.voltage_curvature_inputs()
    except Exception:                                        # noqa: BLE001
        vv = None
    v_slice = _voltage_row_slice(H_y, np.asarray(vv[0], float) if vv else None)

    class_map = ctrl._actuator_class_indices()
    if not class_map:
        return {**ident, "status": "no_classes"}
    int_set = {int(i) for i in ctrl._integer_indices}

    # (S) is exact only for alpha = 1 and g_u = 0.  Both are pinned by the
    # runner for every controller it builds, but read them back rather than
    # assume: if either moves, the move budget below is quietly wrong.
    alpha = float(getattr(ctrl.params, "alpha", 1.0))
    g_u_raw = getattr(ctrl.params, "g_u", 0.0)
    g_u_max = float(np.max(np.abs(np.asarray(g_u_raw, float)))) \
        if np.size(g_u_raw) else 0.0

    g_w_cur, _ = ctrl._get_per_variable_weights()
    if g_w_cur is None:
        g_w_cur = np.broadcast_to(
            np.asarray(ctrl.params.g_w, float), (H_y.shape[1],))
    g_w_cur = np.asarray(g_w_cur, float).copy()

    # Role-based OLTC attenuation, mirroring the controller's own gradient.
    # ``DSOController._build_gradient`` scales the Q-tracking gradient on the
    # OLTC columns by ``gamma_oltc_q`` (0.0 by default: "OLTCs receive no
    # Q-tracking incentive and are driven only by voltage deviations"), while
    # ``objective_curvature_inputs`` returns the *raw* rows, which keep the full
    # dQ/ds_OLTC.  Using the raw rows here would credit a tap with a Q
    # improvement the optimiser never sees, and inflate its self-cost
    # ||a_i||^2.  Apply gamma to the non-voltage rows of the integer OLTC
    # columns so Stage 0 and the MIQP agree.
    gamma = float(getattr(getattr(ctrl, "config", None), "gamma_oltc_q", 1.0))
    H_eff = H_y.copy()
    if gamma < 1.0 and v_slice is not None:
        non_v = np.ones(H_y.shape[0], dtype=bool)
        non_v[v_slice] = False
        oltc_cols = [
            int(i)
            for c, idx in class_map.items() if c.endswith("oltc")
            for i in np.asarray(idx, dtype=int).tolist()
            if int(i) in int_set
        ]
        if non_v.any() and oltc_cols:
            H_eff[np.ix_(non_v, oltc_cols)] *= gamma

    # Weighted columns a_i = D_y^{1/2} H_eff[:, i], and their split by block.
    sqrt_g = np.sqrt(g_y)
    A = sqrt_g[:, None] * H_eff
    col_sq = np.einsum("ij,ij->j", A, A)

    v_rows = np.zeros(H_y.shape[0], dtype=bool)
    if v_slice is not None:
        v_rows[v_slice] = True
    col_sq_v = np.einsum("ij,ij->j", A[v_rows, :], A[v_rows, :]) \
        if v_rows.any() else np.zeros_like(col_sq)
    col_sq_o = col_sq - col_sq_v

    # Representative weights per block, for converting the threshold into a
    # physical unit.  Taken from the weight vector itself rather than from a
    # config field name, so this works for both controller kinds.
    g_v_typ = float(np.median(g_y[v_rows])) if v_rows.any() else float("nan")
    g_o_typ = float(np.median(g_y[~v_rows])) if (~v_rows).any() else float("nan")

    # ── Continuous block: the curvature rule ────────────────────────────────
    cont_classes = [
        c for c, idx in class_map.items()
        if c not in exclude_classes
        and not ({int(i) for i in np.asarray(idx).tolist()} & int_set)
    ]
    pre_idx = (np.concatenate([np.asarray(class_map[c], dtype=int)
                               for c in cont_classes])
               if cont_classes else np.zeros(0, dtype=int))
    cont: dict[str, Any] = {}
    if cont_classes:
        scales = {"der": math.sqrt(tau), "pcc": 1.0 / math.sqrt(tau)}
        # Floor scope.  ``precondition_g_w`` floors columns at
        # ``floor_frac * max_j ||a_j||^2`` with the max taken over *all*
        # columns.  On a TSO loop the excluded AVR column is ~1e7 while every
        # DER/PCC column is ~1 (``dV/dV_ref ~ 1 pu/pu`` against
        # ``dV/dQ ~ 3e-4 pu/Mvar``), so the floor lands two decades above the
        # columns being scaled and flattens all of them to one value -- the
        # conditioning half of the rule silently does nothing.  The floor is
        # documented as protecting near-uncontrollable directions *within the
        # scaled set*, so scope it there by rescaling the fraction rather than
        # by editing the shared module (which would change what every existing
        # study means).
        eff_floor_frac = floor_frac
        if floor_scope == "preconditioned":
            max_all = float(col_sq.max()) if col_sq.size else 0.0
            max_pre = float(col_sq[pre_idx].max()) if pre_idx.size else 0.0
            if max_all > 0.0 and max_pre > 0.0:
                eff_floor_frac = floor_frac * (max_pre / max_all)
        res = precondition_g_w(
            H_v=H_y, g_v=g_y, g_w_current=g_w_cur,
            class_index_map=class_map, preconditionable_classes=cont_classes,
            lambda_target=lambda_target, granularity=granularity,
            floor_frac=eff_floor_frac, mode="set",
            class_scale_overrides={k: v for k, v in scales.items()
                                   if k in cont_classes},
            lambda_scope="preconditioned",
        )
        for c in cont_classes:
            idx = np.asarray(class_map[c], dtype=int)
            w_new = np.asarray(res.g_w_new, float)[idx]
            cont[c] = {
                "n": int(idx.size),
                "g_w_current": float(np.mean(g_w_cur[idx])),
                "g_w_designed_mean": float(np.mean(w_new)),
                "g_w_designed_geomean": float(np.exp(np.mean(np.log(w_new)))),
                "g_w_designed_min": float(np.min(w_new)),
                "g_w_designed_max": float(np.max(w_new)),
                "g_w_designed_all": [float(v) for v in w_new],
            }
        cont_meta = {
            "floor_scope": floor_scope,
            "floor_frac_effective": float(eff_floor_frac),
            "kappa": float(res.kappa), "status": res.status,
            "lambda_target": float(lambda_target),
            "lambda_full_before": float(res.lambda_max_before),
            "lambda_full_after": float(res.lambda_max_after),
            # The analytic counterpart of the intercept the measured rho sweep
            # fits: lambda_max(M) contributed by the columns the curvature rule
            # does NOT scale -- the integer taps and the excluded AVR class --
            # so it is the lower bound no continuous weight can beat.  Reporting
            # the design target alone hides it, because the target is placed on
            # the PRECONDITIONED scope while the loop runs on the full column
            # set.  Computed here rather than read from ``res.lambda_floor``:
            # that field is hard-coded to 0.0 under ``lambda_scope =
            # 'preconditioned'`` (gw_precondition.py: "the fixed columns are out
            # of scope entirely, so there is no floor to be blocked by"), which
            # is right for the rule's own reachability test and wrong as a
            # description of the loop.
            "lambda_floor": _fixed_column_floor(H_y, g_y, g_w_cur, pre_idx),
            "cond_before": float(res.spectrum_before.cond),
            "cond_after": float(res.spectrum_after.cond),
        }
    else:
        cont_meta = {"status": "no_continuous_class"}

    # ── Continuous block: the per-step move budget (this designs g_w_gen) ───
    # Exact interior step, (S):  du_i = -(a_i^T ytil) / (g_w_i + alpha^2 g_u_i),
    # scaled by alpha for continuous columns (w_c = du/alpha).  With the runner's
    # alpha = 1, g_u = 0 this is du_i = -(a_i^T ytil)/g_w_i.  The projection is
    # evaluated for three shapes of a reference error, all of them the same
    # inner product against a different worst-case pattern:
    #
    #   1bus : only the column's strongest controlled bus is off by d_ref
    #          -> |a^T ytil| = g_v * max_k|h_ki| * d_ref
    #   sys  : every bus off by the same +d_ref (the systematic offset)
    #          -> |a^T ytil| = g_v * |sum_k h_ki| * d_ref
    #   box  : every bus off by at most d_ref, signs adversarial -- the max over
    #          the whole box ||e||_inf <= d_ref, hence the l1 norm and the only
    #          reading that is a guarantee
    #          -> |a^T ytil| = g_v * sum_k|h_ki| * d_ref
    #
    # Always 1bus <= box and sys <= box.  Non-voltage objective rows (a DSO's
    # interface-Q block; a TSO's only when tso_g_q_pcc > 0) enter the box reading
    # through their own reference error ``d_ref_q_mvar``.
    moves: list[dict[str, Any]] = []
    for cls, idx in class_map.items():
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0 or ({int(i) for i in idx.tolist()} & int_set):
            continue                       # integer classes have their own rule
        unit = "pu" if cls == "gen" else "Mvar"
        max_move = max_move_gen_pu if cls == "gen" else max_move_q_mvar
        # The curvature rule's answer for this class, when it covered it.
        w_curv = cont.get(cls, {}).get("g_w_designed_all")
        for pos, i in enumerate(idx.tolist()):
            h_v = H_eff[v_rows, i] if v_rows.any() else np.zeros(0)
            h_o = H_eff[~v_rows, i] if (~v_rows).any() else np.zeros(0)
            gv = g_v_typ if g_v_typ == g_v_typ else 0.0
            go = g_o_typ if g_o_typ == g_o_typ else 0.0

            # Projection per pu of voltage error, by error shape.
            pr_1bus = gv * float(np.abs(h_v).max()) if h_v.size else 0.0
            pr_sys = gv * float(abs(h_v.sum())) if h_v.size else 0.0
            pr_box = gv * float(np.abs(h_v).sum()) if h_v.size else 0.0
            # Projection per Mvar of other-channel error.
            pr_box_q = go * float(np.abs(h_o).sum()) if h_o.size else 0.0

            proj_1bus = pr_1bus * d_ref_pu
            proj_sys = pr_sys * d_ref_pu
            proj_box = pr_box * d_ref_pu + pr_box_q * d_ref_q_mvar

            g_w_i = float(g_w_cur[i])
            den = g_w_i + alpha * alpha * g_u_max
            scale = alpha / den if den > 0 else float("nan")
            g_w_curv = (float(np.asarray(w_curv, float)[pos])
                        if w_curv is not None and pos < len(w_curv) else float("nan"))

            # Weight required to hold the move at ``max_move`` for each shape.
            def _req(proj: float) -> float:
                if max_move is None or not (max_move > 0.0) or proj <= 0.0:
                    return float("nan")
                return alpha * proj / float(max_move) - alpha * alpha * g_u_max

            # d@limit: the per-bus voltage deviation at which the *shipped*
            # weight first permits a full ``max_move`` step under the box
            # reading, with the other-channel reference set to zero so the
            # number stays a pure voltage statement.
            if max_move is not None and max_move > 0.0 and pr_box > 0.0:
                d_limit = float(max_move) * den / (alpha * pr_box)
            else:
                d_limit = float("nan")

            moves.append({
                "class": cls, "col": int(i), "unit": unit,
                "g_w_current": g_w_i,
                "g_w_curvature": g_w_curv,
                "proj_per_pu_1bus": pr_1bus,
                "proj_per_pu_sys": pr_sys,
                "proj_per_pu_box": pr_box,
                "proj_per_mvar_box": pr_box_q,
                # realised step at the shipped weight, at the reference error
                "move_1bus": proj_1bus * scale,
                "move_sys": proj_sys * scale,
                "move_box": proj_box * scale,
                # realised step at the curvature-designed weight (NaN for gen,
                # which the curvature rule deliberately does not touch)
                "move_box_curvature": (alpha * proj_box / g_w_curv
                                       if g_w_curv == g_w_curv and g_w_curv > 0
                                       else float("nan")),
                "max_move": float(max_move) if max_move is not None else float("nan"),
                "d_at_limit_pu": d_limit,
                "g_w_for_move_1bus": _req(proj_1bus),
                "g_w_for_move_sys": _req(proj_sys),
                "g_w_for_move_box": _req(proj_box),
            })

    # ── Output weights: block balance, the ROW analogue of the column rule ──
    # The curvature rule equalises the column energies ||a_i||^2 across
    # actuators.  The same quantity summed over the ROWS of one output block,
    #
    #     E_block = sum_{k in block} g_y_k * ||H_eff[k,:]||^2 ,
    #
    # is that block's share of the objective curvature -- i.e. what the loop
    # actually spends its gain on, as opposed to what the weights were meant to
    # express.  Reported both over all columns and over the preconditioned ones
    # only, because a single excluded AVR column (energy ~10^10x a DER's) sets
    # the first number almost by itself and makes it useless as a statement
    # about DER/PCC-driven behaviour.
    all_cols = np.arange(A.shape[1], dtype=int)

    def _energy(mask: np.ndarray, cols: np.ndarray) -> float:
        if not mask.any() or cols.size == 0:
            return 0.0
        sub = A[np.ix_(mask, cols)]
        return float(np.einsum("ij,ij->", sub, sub))

    blocks: dict[str, Any] = {}
    for bname, bmask, bg in (("voltage", v_rows, g_v_typ),
                             ("other", ~v_rows, g_o_typ)):
        if not bmask.any():
            continue
        blocks[bname] = {
            "n_rows": int(bmask.sum()),
            "g_typ": float(bg),
            "energy_all": _energy(bmask, all_cols),
            "energy_preconditioned": _energy(bmask, pre_idx),
        }

    move_meta = {
        "alpha": alpha, "g_u_max": g_u_max,
        "exact": bool(alpha == 1.0 and g_u_max == 0.0),
        "d_ref_pu": float(d_ref_pu), "d_ref_q_mvar": float(d_ref_q_mvar),
        "max_move_gen_pu": float(max_move_gen_pu),
        "max_move_q_mvar": (float(max_move_q_mvar)
                            if max_move_q_mvar is not None else None),
    }

    # ── Integer block: the commit-threshold rule ────────────────────────────
    integers: list[dict[str, Any]] = []
    for cls, idx in class_map.items():
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0 or not ({int(i) for i in idx.tolist()} & int_set):
            continue
        for i in idx.tolist():
            a_norm = float(math.sqrt(max(col_sq[i], 0.0)))
            if a_norm <= 0.0:
                integers.append({"class": cls, "col": int(i),
                                 "status": "zero_sensitivity"})
                continue
            v_share = float(col_sq_v[i] / col_sq[i]) if col_sq[i] > 0 else 0.0
            volt_dominates = v_share >= 0.5
            g_unit = g_v_typ if volt_dominates else g_o_typ
            unit = "pu" if volt_dominates else "Mvar"
            p_star_target = (p_star_pu if volt_dominates else p_star_mvar)

            g_w_i = float(g_w_cur[i])
            # (T): projection threshold implied by the weight in the config.
            p_star_now = 0.5 * a_norm + g_w_i / (2.0 * a_norm)
            sq = math.sqrt(g_unit) if (g_unit == g_unit and g_unit > 0) else float("nan")
            step_phys = a_norm / sq if sq == sq else float("nan")
            thr_phys = p_star_now / sq if sq == sq else float("nan")

            # ── Engage deviation at the tap's strongest controlled bus ──────
            # The projection threshold is the invariant quantity, but the
            # operator states a requirement as "the tap should move once this
            # bus is d off its reference".  For a deviation d at voltage row j
            # and no other error, ytil has the single entry sqrt(g_v) d, so
            # a_i^T ytil = g_v h_ij d and the commit condition
            # ``2 |a_i^T ytil| > ||a_i||^2 + g_w`` becomes
            #
            #     g_w = 2 g_v |h_ij| d - ||a_i||^2 ,       d_now = (g_w + ||a_i||^2) / (2 g_v |h_ij|)
            #
            # exactly -- no dominance assumption, and ||a_i||^2 correctly keeps
            # the cost of whatever else the tap disturbs (interface-Q for a DSO).
            h_v = H_eff[v_rows, i] if v_rows.any() else np.zeros(0)
            if h_v.size and g_v_typ == g_v_typ and g_v_typ > 0:
                jmax = int(np.argmax(np.abs(h_v)))
                h_max = float(abs(h_v[jmax]))
            else:
                jmax, h_max = -1, 0.0
            if h_max > 0.0:
                denom = 2.0 * g_v_typ * h_max
                d_now = (g_w_i + col_sq[i]) / denom
                d_floor = col_sq[i] / denom            # g_w -> 0
                g_w_engage = denom * float(engage_pu) - col_sq[i]
            else:
                d_now = d_floor = g_w_engage = float("nan")

            # The threshold depends on the SHAPE of the deviation, and the
            # single-bus reading above is the pessimistic case.  The case an
            # OLTC exists for is a *systematic* offset: e = d*1 across the
            # zone, giving projection d*|sum h| and
            #     d = (g_w + ||a||^2) / (2 g_v |sum h|) .
            # It accounts for buses the tap pushes the wrong way (negative
            # entries partially cancel), so it is not simply the optimistic
            # bound -- for a tap that opposes part of its own zone it comes out
            # HIGHER than the single-bus reading.  Measured 2026-08-14: TSO
            # taps engage at 0.98-1.31 % under a systematic offset but 3.4-4.4 %
            # under a single-bus spike, which is why a voltage-only single-bus
            # reading badly misdescribes observed tap behaviour.
            h_sum = float(abs(h_v.sum())) if h_v.size else 0.0
            if h_sum > 0.0 and g_v_typ == g_v_typ and g_v_typ > 0:
                den_u = 2.0 * g_v_typ * h_sum
                d_now_uniform = (g_w_i + col_sq[i]) / den_u
                g_w_engage_uniform = den_u * float(engage_pu) - col_sq[i]
            else:
                d_now_uniform = g_w_engage_uniform = float("nan")

            # Other-channel reading (interface-Q for a DSO).  With
            # ``tso_g_q_pcc = 0`` a TSO has no such rows and this is NaN.
            h_o = H_eff[~v_rows, i] if (~v_rows).any() else np.zeros(0)
            o_max = float(np.abs(h_o).max()) if h_o.size else 0.0
            if o_max > 0.0 and g_o_typ == g_o_typ and g_o_typ > 0:
                engage_other = (g_w_i + col_sq[i]) / (2.0 * g_o_typ * o_max)
            else:
                engage_other = float("nan")

            # (T'): weight for the requested *projection* threshold.
            p_star_req = p_star_target * sq if sq == sq else float("nan")
            g_w_req = a_norm * (2.0 * p_star_req - a_norm)

            integers.append({
                "class": cls, "col": int(i),
                "a_norm_weighted": a_norm,
                "voltage_share": v_share,
                "unit": unit,
                "step_phys": step_phys,            # ||h_i||, norm over buses
                "max_bus_step_pu": h_max,          # strongest single bus
                "max_bus_row": jmax,
                "g_w_current": g_w_i,
                "threshold_current": thr_phys,     # (T), projection
                "threshold_requested": float(p_star_target),
                "g_w_designed": float(g_w_req),
                "feasible": bool(g_w_req > 0.0),
                # operator-facing view
                "engage_pu_current": float(d_now),
                "engage_pu_floor": float(d_floor),
                "engage_pu_requested": float(engage_pu),
                "g_w_for_engage": float(g_w_engage),
                "engage_feasible": bool(g_w_engage > 0.0),
                # systematic-offset reading (the design-relevant one)
                "engage_pu_uniform_current": float(d_now_uniform),
                "g_w_for_engage_uniform": float(g_w_engage_uniform),
                # other output channel, e.g. interface-Q for a DSO
                "engage_other_current": float(engage_other),
                "other_unit": "Mvar",
            })

    return {
        **ident, "status": "ok",
        "n_outputs": int(H_y.shape[0]), "n_actuators": int(H_y.shape[1]),
        "g_v_typ": g_v_typ, "g_other_typ": g_o_typ,
        "voltage_rows": int(v_rows.sum()),
        "continuous": cont, "continuous_meta": cont_meta,
        "moves": moves, "move_meta": move_meta,
        "blocks": blocks,
        "integers": integers,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuning_mc.stage_0_preconditioning")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--scenario", type=str, default="v2_quiet_spring",
                   help="Nominal operating point; H is cached at its start.")
    p.add_argument("--from-runner", type=str, default=None, metavar="FUNC",
                   help="Analyse the config built by "
                        "experiments.run_multi_system_ofo.FUNC() instead of the "
                        "YAML baseline + scenario overlay (e.g. "
                        "'make_config_per_area'). Use this whenever the output "
                        "is destined for that config: the design depends on H, "
                        "and H depends on the boundary equivalent, the zone "
                        "partition and the start time that config declares.")
    p.add_argument("--lambda-tso", type=float, default=None)
    p.add_argument("--lambda-tso-zone", type=str, default=None,
                   metavar="'1=..,2=..,3=..'",
                   help="Per-zone override of --lambda-tso, e.g. "
                        "'1=2.0,2=0.2,3=0.5'. Zones not listed keep "
                        "--lambda-tso. Motivated by the analytic per-zone "
                        "contraction (stage_0_coupling_decomposition): a single "
                        "global lambda_tso is set entirely by the worst zone, "
                        "leaving the others well inside their own limit. The "
                        "resulting design is only expressible through "
                        "zone_g_w_class -- the global g_w_<class> scalars carry "
                        "no area information -- so a caller that uses this MUST "
                        "apply the per-area block, not config_block.")
    p.add_argument("--lambda-dso", type=float, default=None)
    p.add_argument("--tau", type=float, default=None,
                   help="DER/PCC shape factor; 1.0 = the analytic rule. "
                        "Unset: taken from the config's precondition_* fields "
                        "so the design lands at the gain and shape that config "
                        "already declares, not at this script's defaults.")
    p.add_argument("--p-star-tso-pu", type=float, default=0.02,
                   help="Requested TSO tap commit threshold [pu voltage].")
    p.add_argument("--p-star-dso-mvar", type=float, default=1E6,
                   help="Requested DSO tap commit threshold [Mvar interface-Q].")
    p.add_argument("--engage-tso-pu", type=float, default=0.015,
                   help="TSO taps should engage once their strongest controlled "
                        "bus is this far from reference [pu].")
    p.add_argument("--engage-dso-pu", type=float, default=0.025,
                   help="Same for DSO taps [pu]; larger leaves more of the "
                        "tracking work to DER reactive power.")
    p.add_argument("--max-move-gen-pu", type=float, default=1e-3,
                   help="Largest AVR setpoint change one TSO step may command "
                        "[pu]. This is what sets g_w_gen; the curvature rule "
                        "does not touch the 'gen' class "
                        "(precondition_exclude_classes).")
    p.add_argument("--max-move-q-mvar", type=float, default=None,
                   help="Same budget for the Mvar-unit continuous classes "
                        "(der/pcc/dso_der). Default None: those keep the "
                        "curvature rule as their design authority and the move "
                        "budget is reported as a diagnostic only.")
    p.add_argument("--dref-pu", type=float, default=0.02,
                   help="Reference voltage error [pu]: the largest per-bus "
                        "deviation the loop is designed to absorb in one tick. "
                        "The move rule is linear in this, so halving it halves "
                        "every designed weight.")
    p.add_argument("--dref-q-mvar", type=float, default=5.0,
                   help="Reference error on the non-voltage objective rows "
                        "[Mvar] -- a DSO's interface-Q block, or a TSO's when "
                        "tso_g_q_pcc > 0. No effect on the TSO 'gen' rule at "
                        "the default tso_g_q_pcc = 0 (no such rows exist).")
    p.add_argument("--vtol-pu", type=float, default=0.01,
                   help="Voltage tolerance used to read the OUTPUT weights "
                        "(g_v / g_q / dso_g_v) as an inverse-square-tolerance "
                        "(Bryson) pair. Fixes the gauge; the Q tolerance the "
                        "shipped weights imply is then reported, not assumed.")
    p.add_argument("--all-columns", action="store_true",
                   help="List every actuator column in the move-budget table. "
                        "Default collapses any class with more than 3 columns "
                        "onto its binding (largest-move) column, which is the "
                        "one the design is set by.")
    p.add_argument("--per-area", action="store_true",
                   help="Also report the design per TSO zone / DSO area, with "
                        "the single-factor zone_g_w_scale that realises it and "
                        "the residual spread a single factor cannot absorb.")
    p.add_argument("--granularity", type=str, default="column",
                   choices=("column", "class"))
    p.add_argument("--floor-scope", type=str, default="preconditioned",
                   choices=("preconditioned", "all"),
                   help="'preconditioned' (default) floors columns relative to "
                        "the largest column BEING SCALED -- the documented "
                        "intent. 'all' reproduces the shipped behaviour, where "
                        "the excluded AVR column sets the floor and flattens "
                        "every DER/PCC weight to one value.")
    p.add_argument("--start-time", type=str, default=None,
                   help="ISO timestamp overriding the operating point, e.g. "
                        "'2016-01-05T08:00'.  H depends on it, so it decides "
                        "every weight this script emits.")
    p.add_argument("--network", type=str, default=None,
                   help="Network overriding the operating point (e.g. "
                        "'rural_700').")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    from tuning._io import load_config_yaml
    from tuning._sim_loader import get_run_multi_tso_dso
    from tuning.scenarios import tune_set_v2

    scenario = None
    if args.from_runner:
        import experiments.run_multi_system_ofo as _rmso
        builder = getattr(_rmso, args.from_runner, None)
        if builder is None:
            raise SystemExit(
                f"[stage0] experiments.run_multi_system_ofo has no "
                f"{args.from_runner!r}")
        cfg = builder()
        # The config declares its own operating point; do NOT overlay a tuning
        # scenario on top of it, that would analyse a different network.
        source = f"run_multi_system_ofo.{args.from_runner}()"
        # Per-area g_w overrides already in the config would make the "g_w now"
        # column area-dependent, which is exactly what we want to see, so they
        # are deliberately left in place.
    else:
        cfg = load_config_yaml(args.baseline)
        # ``--scenario none`` keeps the baseline's OWN operating point.  Without
        # it the overlay is unconditional, which silently relocates the design:
        # the shipped baseline declares 2016-01-05 08:00 on rural_700, while
        # ``v2_overvoltage_rural`` moves it to 2016-07-10 03:00 -- a summer night
        # at which DER reactive capability is EXACTLY ZERO (measured 2026-08-14,
        # tuning_mc.stage_1a_excitation: every DER below the VDE P/Sn = 0.1 dead
        # zone).  Deriving weights there designs the reactive allocation at an
        # operating point where the DER actuator cannot move at all.
        if str(args.scenario).strip().lower() in ("none", "baseline", ""):
            scenario = None
        else:
            scenario = next(
                (s for s in tune_set_v2() if s.name == args.scenario), None)
            if scenario is None:
                raise SystemExit(f"[stage0] Unknown scenario {args.scenario!r}")
            cfg = scenario.overlay_on(cfg)
        source = str(args.baseline)

    # Explicit operating-point overrides, applied last so they win over both
    # paths.  H -- and therefore every weight below -- depends on this point.
    if args.start_time or args.network:
        from datetime import datetime as _dt
        repl: dict[str, Any] = {}
        if args.start_time:
            repl["start_time"] = _dt.fromisoformat(args.start_time)
        if args.network:
            repl["scenario"] = args.network
        cfg = dataclasses.replace(cfg, **repl)
        scenario = None
    # Headless only.  Deliberately NOT tuning's FIXED_OVERRIDES: this analyses
    # the shipped configuration, and that overlay pins int_cooldown=1, which is
    # a tuning artefact rather than the operating point.
    cfg = dataclasses.replace(
        cfg, verbose=0, live_plot_controller=False, live_plot_cascade=False,
        live_plot_system=False, run_stability_analysis=False,
        precondition_g_w=False,
    )

    captured: dict[str, Any] = {}

    def hook(state: dict[str, Any]) -> bool:
        captured["tso"] = state.get("tso_controllers", {})
        captured["dso"] = state.get("dso_controllers", {})
        return True          # abort before the time loop: nothing is simulated

    print(f"[stage0] config source  : {source}")
    if scenario is not None:
        print(f"[stage0] operating point: {args.scenario} "
              f"(start {scenario.start_time}, network {scenario.scenario})")
    else:
        print(f"[stage0] operating point: start {cfg.start_time}, "
              f"boundary {getattr(cfg, 'tie_boundary_equivalent', '?')}, "
              f"local_H tso/dso "
              f"{getattr(cfg, 'local_sensitivities_tso', '?')}/"
              f"{getattr(cfg, 'local_sensitivities_dso', '?')}")
    t0 = time.perf_counter()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        get_run_multi_tso_dso()(cfg, pre_loop_hook=hook)
    print(f"[stage0] controllers built in {time.perf_counter() - t0:.0f} s")

    # ── Resolve lambda / tau from the config unless given explicitly ────────
    # The curvature rule reproduces what the preconditioner WOULD apply, so it
    # must use the same gain and shape the config declares.  Running with this
    # script's own defaults against a BO-tuned config silently designs a
    # different operating point: make_config_tuned carries lambda_tso=0.5012 and
    # class_scales {der: 0.13225, pcc: 7.5617} (= tau 0.017484), against
    # defaults of 0.9 and tau=1, which alone moves the PCC/DER weight ratio by
    # ~57x.
    _prec_lam = float(getattr(cfg, "precondition_lambda_target", 0.8) or 0.8)
    if args.lambda_tso is None:
        args.lambda_tso = float(
            getattr(cfg, "precondition_lambda_target_tso", None) or _prec_lam)
    if args.lambda_dso is None:
        args.lambda_dso = float(
            getattr(cfg, "precondition_lambda_target_dso", None) or _prec_lam)
    if args.tau is None:
        _cs = dict(getattr(cfg, "precondition_class_scales", {}) or {})
        # scales = {der: sqrt(tau), pcc: 1/sqrt(tau)}  ->  tau = scales[der]^2
        if "der" in _cs and float(_cs["der"]) > 0.0:
            args.tau = float(_cs["der"]) ** 2
            _gm = math.sqrt(float(_cs["der"]) * float(_cs.get("pcc", 0.0))) \
                if float(_cs.get("pcc", 0.0)) > 0 else float("nan")
            if _gm == _gm and abs(_gm - 1.0) > 0.05:
                print(f"[stage0] WARNING precondition_class_scales geometric "
                      f"mean is {_gm:.4g}, not 1: it is carrying loop GAIN as "
                      f"well as shape, and tau reproduces the shape only.")
        else:
            args.tau = 1.0
    print(f"[stage0] gain/shape   : lambda_tso={args.lambda_tso:g}  "
          f"lambda_dso={args.lambda_dso:g}  tau={args.tau:g}  "
          f"-> class_scales {{der: {math.sqrt(args.tau):.5g}, "
          f"pcc: {1.0 / math.sqrt(args.tau):.5g}}}")

    exclude = tuple(getattr(cfg, "precondition_exclude_classes", ()) or ())
    out: list[dict[str, Any]] = []
    move_kw = dict(
        d_ref_pu=args.dref_pu, d_ref_q_mvar=args.dref_q_mvar,
        max_move_gen_pu=args.max_move_gen_pu,
        max_move_q_mvar=args.max_move_q_mvar,
    )
    # Per-zone loop gain.  Parsed here rather than folded into args.lambda_tso
    # so that the scalar keeps its meaning ("the gain every zone gets unless
    # told otherwise") and the payload can record both.
    lam_by_zone: dict[int, float] = {}
    if args.lambda_tso_zone:
        for kv in args.lambda_tso_zone.split(","):
            z_txt, _, v_txt = kv.partition("=")
            if not v_txt:
                raise SystemExit(f"[stage0] bad --lambda-tso-zone entry {kv!r}; "
                                 f"expected 'zone=value'")
            lam_by_zone[int(z_txt.strip())] = float(v_txt)
        print(f"[stage0] per-zone lambda_tso: {lam_by_zone} "
              f"(zones not listed use {args.lambda_tso:g})")

    for z, ctrl in sorted(captured.get("tso", {}).items()):
        out.append(_analyse_controller(
            ctrl, f"TSO-z{z}", kind="tso", area=int(z),
            lambda_target=lam_by_zone.get(int(z), args.lambda_tso),
            tau=args.tau,
            p_star_pu=args.p_star_tso_pu, p_star_mvar=args.p_star_dso_mvar,
            engage_pu=args.engage_tso_pu,
            exclude_classes=exclude, granularity=args.granularity,
            floor_frac=float(getattr(cfg, "precondition_floor_frac", 1e-6)),
            floor_scope=args.floor_scope, **move_kw,
        ))
    for d, ctrl in sorted(captured.get("dso", {}).items()):
        out.append(_analyse_controller(
            ctrl, f"DSO-{d}", kind="dso", area=str(d),
            lambda_target=args.lambda_dso, tau=args.tau,
            p_star_pu=args.p_star_tso_pu, p_star_mvar=args.p_star_dso_mvar,
            engage_pu=args.engage_dso_pu,
            exclude_classes=exclude, granularity=args.granularity,
            floor_frac=float(getattr(cfg, "precondition_floor_frac", 1e-6)),
            floor_scope=args.floor_scope, **move_kw,
        ))

    # ── Report ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}\n  1. CONTINUOUS CLASSES — curvature rule "
          f"(lambda_tso={args.lambda_tso:g}, lambda_dso={args.lambda_dso:g}, "
          f"tau={args.tau:g})\n{'=' * 78}")
    print("  Only classes the loop actually owns appear.  A class missing for "
          "one loop means\n  that loop has no actuator of that kind — TSO zone "
          "1 owns no PCC interface, so\n  it has no 'pcc' row here or in the "
          "per-area pivot.  'gen' never appears: it is\n  in "
          "precondition_exclude_classes and is designed by the move budget "
          "instead.\n")
    print(f"{'loop':<12}{'class':<10}{'n':>3} {'g_w now':>10} "
          f"{'g_w design':>12} {'[min':>10} {'max]':>10}  {'kappa':>9} {'status':>8}")
    for r in out:
        if r.get("status") != "ok":
            continue
        m = r["continuous_meta"]
        for cls, d in r["continuous"].items():
            print(f"{r['label']:<12}{cls:<10}{d['n']:>3} "
                  f"{d['g_w_current']:>10.4g} {d['g_w_designed_mean']:>12.4g} "
                  f"{d['g_w_designed_min']:>10.4g} {d['g_w_designed_max']:>10.4g}  "
                  f"{m.get('kappa', float('nan')):>9.4g} "
                  f"{m.get('status', ''):>8}")

    # ── Continuous move budget ─────────────────────────────────────────────
    _mm = next((r["move_meta"] for r in out if r.get("status") == "ok"), None)
    print(f"\n{'=' * 104}\n  2. CONTINUOUS CLASSES — per-step move budget "
          f"(exact interior step:  du_i = -(a_i.ytil) / g_w_i)\n{'=' * 104}")
    print(f"  reference error: every controlled bus up to "
          f"{100 * args.dref_pu:.2f} % off reference"
          + (f", other rows {args.dref_q_mvar:g} Mvar" if args.dref_q_mvar else ""))
    print("  move@1bus = only the column's strongest bus is off        "
          "(g_v * max_k|h_ki| * d)")
    print("  move@sys  = every bus off by the same +d                  "
          "(g_v * |sum_k h_ki| * d)")
    print("  move@box  = signs adversarial, |e_k| <= d -- a GUARANTEE  "
          "(g_v * sum_k|h_ki| * d)")
    print("  d@limit   = per-bus deviation at which the shipped g_w first "
          "permits a full budget step")
    print("  move@sys == move@box means that column raises voltage at every "
          "bus it reaches\n  (single-signed h_ki), so the adversarial bound "
          "costs nothing over the systematic one.")
    if _mm is not None and not _mm["exact"]:
        print(f"  WARNING: alpha={_mm['alpha']:g}, max|g_u|={_mm['g_u_max']:g} "
              f"-- (S) is no longer exact for this controller")
    if not args.all_columns:
        print("  Classes with >3 columns are collapsed onto their BINDING "
              "column (largest\n  move@box) — that is the one the design is set "
              "by.  --all-columns lists them all.")
    print(f"\n{'loop':<12}{'class':<10}{'col':>6} {'unit':>5} {'g_w now':>10} "
          f"{'move@1bus':>11} {'move@sys':>11} {'move@box':>11} "
          f"{'d@limit':>9} {'budget':>9} {'g_w(box)':>10}")

    def _move_row(label: str, m: dict[str, Any], col_txt: str,
                  suffix: str = "") -> None:
        g_req = m["g_w_for_move_box"]
        flag = ""
        # Relative tolerance: a config that was *derived* from this rule sits
        # exactly on the boundary, and a 4-significant-digit round-trip through
        # the config file then lands a hair either side of it.
        if g_req == g_req and g_req > m["g_w_current"] * (1.0 + 1e-3):
            flag = "  <-- shipped g_w too LOW for the budget"
        d_txt = (f"{100 * m['d_at_limit_pu']:>8.2f}%"
                 if m["d_at_limit_pu"] == m["d_at_limit_pu"] else f"{'--':>9}")
        print(f"{label:<12}{m['class']:<10}{col_txt:>6} "
              f"{m['unit']:>5} {m['g_w_current']:>10.4g} "
              f"{m['move_1bus']:>11.3e} {m['move_sys']:>11.3e} "
              f"{m['move_box']:>11.3e} {d_txt} "
              f"{m['max_move']:>9.4g} {g_req:>10.4g}{flag}{suffix}")

    for r in out:
        if r.get("status") != "ok":
            continue
        by_cls: dict[str, list[dict[str, Any]]] = {}
        for m in r.get("moves", []):
            by_cls.setdefault(m["class"], []).append(m)
        for cls, ms in by_cls.items():
            live = [m for m in ms
                    if m["proj_per_pu_box"] > 0.0 or m["proj_per_mvar_box"] > 0.0]
            dead = [m for m in ms if m not in live]
            if live:
                if args.all_columns or len(live) <= 3:
                    for m in live:
                        _move_row(r["label"], m, str(m["col"]))
                else:
                    binding = max(live, key=lambda m: m["move_box"])
                    lo = min(m["move_box"] for m in live)
                    _move_row(r["label"], binding,
                              f"{binding['col']}/{len(live)}",
                              f"   binding of {len(live)}; "
                              f"others down to {lo:.3e}")
            for m in dead:
                print(f"{r['label']:<12}{m['class']:<10}{m['col']:>6} "
                      f"{m['unit']:>5} {m['g_w_current']:>10.4g}   "
                      f"zero sensitivity — dead actuator, no budget defined")
    print("\n  Mvar-unit classes show 'nan' under budget/g_w(box) unless "
          "--max-move-q-mvar is\n  given: the curvature rule is their design "
          "authority and this is a diagnostic there.")

    print(f"\n{'=' * 92}\n  3. INTEGER CLASSES — engage deviation at the tap's "
          f"strongest bus\n{'=' * 92}")
    print("  engage@1bus = error concentrated on the tap's strongest bus "
          "(pessimistic)")
    print("  engage@sys  = systematic offset across the whole zone "
          "(design-relevant)")
    print("  engage@Q    = interface-Q error alone; NaN for a TSO "
          "(tso_g_q_pcc = 0)\n")
    print(f"{'loop':<12}{'class':<10}{'col':>4} {'maxbus%':>8} "
          f"{'g_w now':>9} {'eng@1bus':>9} {'eng@sys':>8} {'eng@Q':>10} "
          f"{'target':>7} {'g_w(sys)':>9}")
    for r in out:
        if r.get("status") != "ok":
            continue
        for d in r["integers"]:
            if d.get("status") == "zero_sensitivity":
                print(f"{r['label']:<12}{d['class']:<10}{d['col']:>4}  "
                      f"zero sensitivity — dead actuator, no threshold defined")
                continue
            g_sys = d["g_w_for_engage_uniform"]
            flag = "" if (g_sys == g_sys and g_sys > 0.0) else \
                "  <-- below the overshoot floor"
            print(f"{r['label']:<12}{d['class']:<10}{d['col']:>4} "
                  f"{100 * d['max_bus_step_pu']:>7.3f}% "
                  f"{d['g_w_current']:>9.4g} "
                  f"{100 * d['engage_pu_current']:>8.2f}% "
                  f"{100 * d['engage_pu_uniform_current']:>7.2f}% "
                  f"{d['engage_other_current']:>7.3f}Mvar "
                  f"{100 * d['engage_pu_requested']:>6.2f}% "
                  f"{g_sys:>9.4g}{flag}")

    # ── Paste-ready config block ───────────────────────────────────────────
    # The design rule is per-zone and per-column; ``MultiTSOConfig`` carries one
    # scalar per class.  Compressing the first onto the second is lossy, and the
    # loss is reported rather than hidden.  Geometric mean is the right average
    # for a weight that acts multiplicatively on the curvature.
    def _pack(vals: list[float], how: str) -> tuple[float, float, float, int]:
        """(representative, min, max, n).  ``how='geomean'`` for a weight that
        acts multiplicatively on the curvature; ``how='max'`` for a guarantee,
        where the binding column has to set the scalar."""
        v = np.asarray([x for x in vals if x == x and x > 0.0], float)
        if v.size == 0:
            return float("nan"), float("nan"), float("nan"), 0
        rep = float(v.max()) if how == "max" else float(np.exp(np.mean(np.log(v))))
        return rep, float(v.min()), float(v.max()), int(v.size)

    def _vals(cls_name: str, loops: list[dict[str, Any]] | None = None) -> list[float]:
        """Per-column designed weights for a continuous (curvature) class."""
        vals: list[float] = []
        for r in (out if loops is None else loops):
            if r.get("status") != "ok":
                continue
            d = r["continuous"].get(cls_name)
            if d:
                vals += d["g_w_designed_all"]
        return vals

    def _vals_int(class_suffix: str,
                  loops: list[dict[str, Any]] | None = None) -> list[float]:
        """Per-column designed weights for an integer (commit-threshold) class."""
        return [
            d["g_w_for_engage_uniform"]
            for r in (out if loops is None else loops) if r.get("status") == "ok"
            for d in r["integers"]
            if d.get("class", "").endswith(class_suffix)
            and d.get("g_w_for_engage_uniform", float("nan")) > 0.0
        ]

    def _vals_move(cls_name: str,
                   loops: list[dict[str, Any]] | None = None) -> list[float]:
        """Per-column designed weights for a move-budget class (gen)."""
        return [
            m["g_w_for_move_box"]
            for r in (out if loops is None else loops) if r.get("status") == "ok"
            for m in r.get("moves", [])
            if m.get("class") == cls_name
        ]

    # field -> (value source, aggregation).  ``g_w_gen`` is the move budget's
    # output and is aggregated with ``max``: it is a bound, so the column that
    # needs the most damping fixes the single config scalar.
    FIELDS: list[tuple[str, Any, str]] = [
        ("g_w_der", lambda L: _vals("der", L), "geomean"),
        ("g_w_pcc", lambda L: _vals("pcc", L), "geomean"),
        ("g_w_gen", lambda L: _vals_move("gen", L), "max"),
        ("g_w_dso_der", lambda L: _vals("dso_der", L), "geomean"),
        ("g_w_tso_oltc", lambda L: _vals_int("tso_oltc", L), "geomean"),
        ("g_w_dso_oltc", lambda L: _vals_int("dso_oltc", L), "geomean"),
    ]

    # ── Output weights (g_v / g_q / dso_g_v) ───────────────────────────────
    # These are NOT step sizes and neither rule above sets them.  What can be
    # said about them is stated in three parts, in decreasing strength.
    print(f"\n{'=' * 96}\n  4. OUTPUT WEIGHTS — g_v / g_q / dso_g_v are a gauge "
          f"plus one trade-off ratio\n{'=' * 96}")
    print("  (1) GAUGE.  The interior step is du_i = -(a_i.ytil)/g_w_i and "
          "a_i.ytil is linear\n      in g_y, so (g_y, g_w) -> (c*g_y, c*g_w) "
          "leaves every action identical.  The\n      absolute level of g_v is "
          "therefore not a tuned quantity; only g_y/g_w (the\n      loop gain, "
          "set by the curvature rule) and the ratios BETWEEN output blocks\n"
          "      (the objective trade-off) are observable.  The one legitimate "
          "criterion for\n      the gauge itself is solver conditioning: pick c "
          "so the G_w diagonal straddles 1.")
    _gw_all = np.concatenate([
        np.asarray([m["g_w_current"] for m in r.get("moves", [])] +
                   [d["g_w_current"] for d in r.get("integers", [])
                    if "g_w_current" in d], float)
        for r in out if r.get("status") == "ok"
    ]) if any(r.get("status") == "ok" for r in out) else np.zeros(0)
    _gw_all = _gw_all[_gw_all > 0]
    if _gw_all.size:
        print(f"      shipped G_w diagonal: [{_gw_all.min():.4g}, "
              f"{_gw_all.max():.4g}], spread {_gw_all.max() / _gw_all.min():.3g}x, "
              f"geometric centre {np.exp(np.mean(np.log(_gw_all))):.4g}")

    print("\n  (2) TRADE-OFF.  Within one controller the block ratio is "
          "physical.  Read it as an\n      inverse-square-tolerance (Bryson) "
          "pair, g_block = 1/sigma_block^2: the weighted\n      residual then "
          "counts tolerances, and stating one tolerance fixes the other.")
    print(f"{'loop':<12}{'g_v':>12}{'g_other':>12}{'rows_v':>8}"
          f"{'dV/1Mvar@1bus':>15}{'dV/1Mvar@all':>14}{'Qtol@dV=':>10}")
    for r in out:
        if r.get("status") != "ok":
            continue
        gv, go = r["g_v_typ"], r["g_other_typ"]
        n_v = max(int(r["voltage_rows"]), 1)
        if not (gv == gv and gv > 0 and go == go and go > 0):
            print(f"{r['label']:<12}{gv:>12.4g}{go:>12.4g}{r['voltage_rows']:>8}"
                  f"{'--':>15}{'--':>14}{'--':>10}   (single output block: no "
                  f"trade-off to fix)")
            continue
        # g_v dV^2 = g_o dQ^2 -> the voltage error worth 1 Mvar of the other
        # channel, concentrated on one bus and spread over all n_v buses.
        dv_1 = math.sqrt(go / gv)
        dv_all = math.sqrt(go / (gv * n_v))
        q_tol = args.vtol_pu * math.sqrt(gv / go)
        print(f"{r['label']:<12}{gv:>12.4g}{go:>12.4g}{r['voltage_rows']:>8}"
              f"{100 * dv_1:>14.4f}%{100 * dv_all:>13.4f}%{q_tol:>9.3f}M")
    print(f"      Qtol@dV= is the interface-Q tolerance the SHIPPED weights "
          f"imply once the\n      voltage tolerance is fixed at "
          f"{100 * args.vtol_pu:g} % (--vtol-pu).  If that number is not the "
          f"one\n      you would defend in writing, the block ratio is the "
          f"thing to change, not g_v.")

    print("\n  (3) REALISED BALANCE.  What the loop actually spends its "
          "curvature on:\n      E_block = sum_{k in block} g_y_k ||H[k,:]||^2, "
          "the row analogue of ||a_i||^2.\n      'pre' restricts the columns to "
          "those the curvature rule scales (drops the AVR\n      column, whose "
          "energy would otherwise decide the split by itself).")
    print(f"{'loop':<12}{'block':<10}{'rows':>6}{'g_typ':>12}"
          f"{'E(all cols)':>14}{'share':>8}{'E(pre cols)':>14}{'share':>8}")
    for r in out:
        if r.get("status") != "ok":
            continue
        bl = r.get("blocks", {})
        tot_a = sum(b["energy_all"] for b in bl.values()) or float("nan")
        tot_p = sum(b["energy_preconditioned"] for b in bl.values()) or float("nan")
        for bname, b in bl.items():
            print(f"{r['label']:<12}{bname:<10}{b['n_rows']:>6}"
                  f"{b['g_typ']:>12.4g}{b['energy_all']:>14.4g}"
                  f"{100 * b['energy_all'] / tot_a:>7.1f}%"
                  f"{b['energy_preconditioned']:>14.4g}"
                  f"{100 * b['energy_preconditioned'] / tot_p:>7.1f}%")

    print(f"\n{'=' * 96}\n  5. CONFIG BLOCK — single scalars over the "
          f"per-column design\n{'=' * 96}")
    print(f"{'field':<20}{'designed':>12}{'[min':>12}{'max]':>12}"
          f"{'cols':>6}{'spread':>9}{'agg':>9}   shipped")
    shipped = {
        "g_w_der": float(cfg.g_w_der), "g_w_pcc": float(cfg.g_w_pcc),
        "g_w_gen": float(cfg.g_w_gen),
        "g_w_dso_der": float(cfg.g_w_dso_der),
        "g_w_tso_oltc": float(cfg.g_w_tso_oltc),
        "g_w_dso_oltc": float(cfg.g_w_dso_oltc),
    }
    rows = [(name, _pack(src(None), how), how) for name, src, how in FIELDS]
    for name, (rep, lo, hi, n), how in rows:
        if n == 0:
            continue
        print(f"{name:<20}{rep:>12.4g}{lo:>12.4g}{hi:>12.4g}{n:>6}"
              f"{hi / lo:>8.1f}x{how:>9}   {shipped[name]:g}")
    print("\n  Pinned (gauge / objective trade-off, NOT set by these rules): "
          f"\n  g_v={cfg.g_v:g}  g_q={cfg.g_q:g}  dso_g_v={cfg.dso_g_v:g}  "
          f"tso_g_q_pcc={getattr(cfg, 'tso_g_q_pcc', 0.0):g}  "
          f"shunt_int_g_w={cfg.shunt_int_g_w:g}")
    print("  precondition_g_w = False  <- these are static weights; do NOT "
          "also enable the preconditioner")

    # ── Per-area refinement ────────────────────────────────────────────────
    # One level less compression than the config block: the design aggregated
    # per control area rather than globally.  Reported with the single-factor
    # re-gain that the shipped config can actually apply (``zone_g_w_scale``),
    # and with the residual spread that a single factor per area cannot absorb.
    global_rep = {name: rep for name, (rep, _, _, n), _ in rows if n > 0}
    areas: list[dict[str, Any]] = []
    for r in out:
        if r.get("status") != "ok":
            continue
        entry: dict[str, Any] = {"label": r["label"], "kind": r["kind"],
                                 "area": r["area"], "fields": {}}
        ratios: list[float] = []
        for name, src, how in FIELDS:
            v = [x for x in src([r]) if x == x and x > 0.0]
            if not v:
                continue
            rep, lo, hi, n = _pack(v, how)
            entry["fields"][name] = {"designed": rep, "min": lo, "max": hi,
                                     "n_columns": n,
                                     "global": global_rep.get(name, float("nan"))}
            g = global_rep.get(name)
            if g and g == g and g > 0.0:
                ratios += [x / g for x in v]
        if ratios:
            f = float(np.exp(np.mean(np.log(np.asarray(ratios, float)))))
            res = np.asarray(ratios, float) / f
            entry["factor"] = f
            entry["residual_spread"] = float(res.max() / res.min())
        else:
            entry["factor"] = float("nan")
            entry["residual_spread"] = float("nan")
        areas.append(entry)

    if args.per_area:
        print(f"\n{'=' * 96}\n  6. PER-AREA REFINEMENT — one value per "
              f"(control area, actuator class)\n{'=' * 96}")

        # ── Pivot: areas down, classes across ──────────────────────────────
        # A class an area does not own must read as "no such actuator", not as
        # a gap in the data.  The long format below omitted those rows
        # silently, which is unreadable: TSO zone 1 owns no PCC interface, so
        # its g_w_pcc row simply was not printed.
        _W = 16

        def _pivot(kind: str, title: str) -> list[str]:
            rows_a = [e for e in areas if e["kind"] == kind]
            if not rows_a:
                return []
            cols = [n for n, _, _ in FIELDS
                    if any(n in e["fields"] for e in rows_a)]
            if not cols:
                return []
            rule = "  " + "-" * (12 + _W * len(cols))
            print(f"\n  {title}")
            print("  " + "area".ljust(12)
                  + "".join(c.rjust(_W) for c in cols))
            print(rule)
            for e in rows_a:
                cells = "".join(
                    (f"{e['fields'][c]['designed']:.4g}" if c in e["fields"]
                     else "-").rjust(_W) for c in cols)
                print("  " + e["label"].ljust(12) + cells)
            print(rule)
            print("  " + "global".ljust(12) + "".join(
                (f"{global_rep[c]:.4g}" if c in global_rep else "-").rjust(_W)
                for c in cols) + "   <- geomean/max over all areas")
            print("  " + "shipped".ljust(12) + "".join(
                (f"{shipped[c]:.4g}" if c in shipped else "-").rjust(_W)
                for c in cols) + "   <- the config's global fallback scalar")
            print("  " + "ratio".ljust(12) + "".join(
                ((f"{global_rep[c] / shipped[c]:.2f}x")
                 if c in global_rep and shipped.get(c) else "-").rjust(_W)
                for c in cols) + "   <- designed / shipped")
            return [f"{e['label']}/{c}" for e in rows_a for c in cols
                    if c not in e["fields"]]

        absent = (_pivot("tso", "TSO zones — designed g_w")
                  + _pivot("dso", "DSO areas — designed g_w"))
        if absent:
            print(f"\n  '-' = that area owns no actuator of that class, so "
                  f"there is nothing to\n        design and nothing is "
                  f"missing: {', '.join(absent)}")

        # ── Detail: what the single cell value is aggregated from ──────────
        print(f"\n  Detail — each cell above is a geometric mean over that "
              f"area's columns\n  (max for 'gen', which is a bound), so the "
              f"spread is the loss it hides.")
        print(f"  {'area':<12}{'field':<16}{'designed':>12}{'[min':>12}"
              f"{'max]':>12}{'cols':>6}{'spread':>9}{'vs global':>11}")
        for e in areas:
            for name, d in e["fields"].items():
                g = d["global"]
                print(f"  {e['label']:<12}{name:<16}{d['designed']:>12.4g}"
                      f"{d['min']:>12.4g}{d['max']:>12.4g}{d['n_columns']:>6}"
                      f"{d['max'] / d['min']:>8.1f}x"
                      + (f"{d['designed'] / g:>10.2f}x" if g == g and g > 0
                         else f"{'--':>11}"))

        print("\n  Single-factor re-gain per area  (factor = log-least-squares "
              "optimum over\n  that area's columns; residual = what a single "
              "factor cannot absorb)")
        print(f"  {'area':<12}{'factor':>10}{'residual':>12}")
        for e in areas:
            print(f"  {e['label']:<12}{e['factor']:>10.4g}"
                  f"{e['residual_spread']:>11.1f}x")

        tso_scale = {e["area"]: round(e["factor"], 4) for e in areas
                     if e["kind"] == "tso" and e["factor"] == e["factor"]}
        if tso_scale:
            print("\n  (a) Single factor per zone — re-gains the area without "
                  "disturbing the\n      ratios between its classes, but "
                  "cannot express the residual above:")
            print(f"    zone_g_w_scale={tso_scale}")
        dso_scale = {e["area"]: round(e["factor"], 4) for e in areas
                     if e["kind"] == "dso" and e["factor"] == e["factor"]}
        if dso_scale:
            print("      (no dso_g_w_scale counterpart exists; for reference "
                  f"the DSO factors are {dso_scale})")

        # (b) The full per-area, per-class design.  This is the object the
        # residual column says you need whenever a single factor per area does
        # not fit -- absolute weights, applied by the runner straight onto the
        # class blocks of that controller's params.g_w (and its
        # _g_w_vector_cache, so they reach build_miqp_problem).
        # Config field name -> actuator class name: the runner keys the
        # override by CLASS ("der"), not by config field ("g_w_der").
        def _fmt(name: str, kind: str, keyfmt: Any) -> str:
            lines = [f"    {name}={{"]
            for e in areas:
                if e["kind"] != kind or not e["fields"]:
                    continue
                body = ", ".join(f"{n[len('g_w_'):]!r}: {d['designed']:.4g}"
                                 for n, d in e["fields"].items())
                lines.append(f"        {keyfmt(e['area'])}: {{{body}}},")
            lines.append("    },")
            return "\n".join(lines)

        print("\n  (b) Full per-area, per-class design — absolute weights, no "
              "residual by\n      construction.  Paste into the config; the "
              "runner writes these onto the\n      class blocks of each "
              "controller's g_w vector before the first step:")
        if any(e["kind"] == "tso" for e in areas):
            print(_fmt("zone_g_w_class", "tso", lambda a: int(a)))
        if any(e["kind"] == "dso" for e in areas):
            print(_fmt("dso_g_w_class", "dso", lambda a: repr(str(a))))
        print("      NOTE the global g_w_<class> scalars stay in the config as "
              "the fallback for\n      any area or class not listed here.")

    payload = {
        "config_block": {
            name: {"designed": rep, "min": lo, "max": hi, "n_columns": n,
                   "aggregation": how, "shipped": shipped[name]}
            for name, (rep, lo, hi, n), how in rows if n > 0
        },
        "per_area": areas,
        "zone_g_w_scale": {int(e["area"]): e["factor"] for e in areas
                           if e["kind"] == "tso" and e["factor"] == e["factor"]},
        "baseline": source,
        "operating_point": {
            "scenario": args.scenario if scenario is not None else None,
            "start_time": str(scenario.start_time if scenario is not None
                              else cfg.start_time),
            "network": scenario.scenario if scenario is not None else None,
            "tie_boundary_equivalent": getattr(
                cfg, "tie_boundary_equivalent", None),
        },
        "targets": {
            "lambda_tso": args.lambda_tso,
            "lambda_tso_by_zone": {str(k): v for k, v in lam_by_zone.items()},
            "lambda_dso": args.lambda_dso,
            "tau": args.tau, "p_star_tso_pu": args.p_star_tso_pu,
            "p_star_dso_mvar": args.p_star_dso_mvar,
            "granularity": args.granularity,
            "d_ref_pu": args.dref_pu, "d_ref_q_mvar": args.dref_q_mvar,
            "max_move_gen_pu": args.max_move_gen_pu,
            "max_move_q_mvar": args.max_move_q_mvar,
            "vtol_pu": args.vtol_pu,
        },
        "loops": out,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\n[stage0] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
