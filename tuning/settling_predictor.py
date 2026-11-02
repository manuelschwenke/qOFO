r"""
tuning/settling_predictor.py
============================
Linear prediction of the subordinate (STS) interface-Q settling count.

Purpose
-------
Pre-rank DSO weight sets for the D1 settling screen (handoff 2026-09-28,
``HANDOFF_D_settling_rerun.md``) before the expensive isolated-STS step
protocol (``ch_9_1_ninner_isolated_sts.py``, ~32 CPU-h per 120-step set).

Model (established)
-------------------
The STS-OFO continuous update is the gradient step (``base_controller``,
alpha = 1, factor 2 absorbed by the QP's 1/2)::

    w_k = -G_w^{-1} ( g_q H_Q^T (q_k - q_set) + g_v H_V^T (v_k - v_set) )

so, with a linear plant ``y = H u``, the continuous-actuator iteration is
``u_{k+1} = u_k + w_k`` with iteration matrix ``I - G_w^{-1} H^T W H``.  Its
eigenvalues are ``1 - lambda_i(C)``, ``C = G_w^{-1/2} H^T W H G_w^{-1/2}``,
which is the matrix whose largest eigenvalue the Stage-I screen bounds.

What is predicted
-----------------
A setpoint step of ``step_mvar`` on ONE interface, siblings held (the protocol
of ``ch_9_1_ninner_isolated_sts.py``), from an equilibrium.  The interface-Q
sequence ``q_i(k)`` is iterated for ``n_iter`` subordinate iterations and scored
with the SAME ``n_settle_final`` rule as the simulation.

What is ignored (so this is a ranking proxy, not a measurement)
---------------------------------------------------------------
* integer actuators (OLTC columns dropped -- the E run at ``399e30d3b60b``
  recorded zero DSO tap moves over 120 steps),
* DER capability bounds and voltage-limit constraints (inactive for 10 Mvar
  steps in the E run: zero voltage violations, zero slack),
* measurement noise, sensitivity refresh, plant nonlinearity,
* the one-sample offset between dispatch and record (iteration 0).

Author: Manuel Schwenke / Claude Code, 2026-09-29.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class AreaSensitivity:
    """Continuous-actuator sensitivities of one STS controller at one condition."""

    condition: str
    area: str                 # "DSO_1" ...
    H_Q: np.ndarray           # (n_if, n_der)  interface Q [Mvar] per DER u
    H_V: np.ndarray           # (n_v, n_der)   bus voltage [pu] per DER u
    iface_keys: Tuple[str, ...]   # record keys "DSO_k|trafo_i", controller order


def _capture_condition(cfg, condition: str) -> List[AreaSensitivity]:
    from tuning._sim_loader import get_run_multi_tso_dso

    captured: Dict[str, Any] = {}

    def hook(state: Dict[str, Any]) -> bool:
        captured.update(state)
        return True

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        get_run_multi_tso_dso()(cfg, pre_loop_hook=hook)

    out: List[AreaSensitivity] = []
    for area, ctrl in sorted((captured.get("dso_controllers") or {}).items()):
        c = ctrl.config
        n_if = len(c.interface_trafo_indices)
        n_v = len(c.voltage_bus_indices)
        mapping = getattr(c, "der_mapping", None)
        n_der = mapping.n_der if mapping is not None else len(c.der_indices)
        H = np.asarray(ctrl._expand_H_to_der_level(
            ctrl._build_sensitivity_matrix()), dtype=float)
        # Row layout [Q_interface | V_bus | ...]; DER columns first.
        H_Q = np.ascontiguousarray(H[:n_if, :n_der])
        H_V = np.ascontiguousarray(H[n_if:n_if + n_v, :n_der])
        k = int(str(area).split("_")[-1]) - 1
        keys = tuple(f"{area}|trafo_{k * n_if + i}" for i in range(n_if))
        out.append(AreaSensitivity(condition, str(area), H_Q, H_V, keys))
    return out


def load_or_capture(base_cfg, conditions: Sequence[tuple], cache_dir: Path,
                    *, verbose: bool = True) -> List[AreaSensitivity]:
    """One capture per condition, cached by network key like Stage I."""
    from tuning.stage1_screen import _network_key

    cache_dir.mkdir(parents=True, exist_ok=True)
    net = _network_key(base_cfg)
    out: List[AreaSensitivity] = []
    for name, start, *_ in conditions:
        tag = f"{base_cfg.scenario}_{net}" if net else str(base_cfg.scenario)
        path = cache_dir / f"hsts_{tag}_{name}.pkl"
        if path.exists():
            with path.open("rb") as fh:
                out.extend(pickle.load(fh))
            continue
        cfg = dataclasses.replace(
            base_cfg, start_time=start, n_total_s=60.0, contingencies=[],
            verbose=0, live_plot_controller=False, live_plot_cascade=False,
            live_plot_overview=False, run_stability_analysis=False,
            precondition_g_w=False)
        if verbose:
            print(f"[settle-pred] {name}: capturing STS sensitivities ...",
                  flush=True)
        rows = _capture_condition(cfg, name)
        with path.open("wb") as fh:
            pickle.dump(rows, fh, protocol=pickle.HIGHEST_PROTOCOL)
        out.extend(rows)
    return out


def iterate_step(a: AreaSensitivity, g_q: float, g_v: float, g_w_der: float,
                 iface: int, *, step_mvar: float = 10.0,
                 n_iter: int = 46) -> np.ndarray:
    """``q_iface(k) - q_iface(before)`` for k = 0..n_iter-1 after the step.

    Index 0 is the flow after the first post-step dispatch.
    """
    n_der = a.H_Q.shape[1]
    ginv = 1.0 / float(g_w_der)
    # Gradient-step iteration on the deviation from the pre-step equilibrium:
    # e_q = H_Q du - dq_set, e_v = H_V du.
    dq_set = np.zeros(a.H_Q.shape[0])
    dq_set[iface] = step_mvar
    du = np.zeros(n_der)
    out = np.empty(n_iter)
    A = ginv * (g_q * a.H_Q.T @ a.H_Q + g_v * a.H_V.T @ a.H_V)
    b = ginv * g_q * a.H_Q.T @ dq_set
    for k in range(n_iter):
        du = du - (A @ du - b)
        out[k] = a.H_Q[iface] @ du
    return out


def contraction(a: AreaSensitivity, g_q: float, g_v: float,
                g_w_der: float) -> float:
    """``lambda_max`` of the continuous STS block (DER columns only)."""
    C = (g_q * a.H_Q.T @ a.H_Q + g_v * a.H_V.T @ a.H_V) / float(g_w_der)
    return float(np.linalg.eigvalsh(0.5 * (C + C.T))[-1])


def predict(areas: Sequence[AreaSensitivity], weights: Dict[str, float], *,
            conditions: Optional[Sequence[str]] = None,
            step_mvar: float = 10.0, n_iter: int = 46,
            ) -> List[Dict[str, Any]]:
    """Predicted ``n_settle_final`` per (condition, area, interface)."""
    import sys
    here = Path(__file__).resolve().parents[1] / "experiments" / \
        "ch_09_parameter_selection"
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from ch_9_1_ninner_isolated_sts import n_settle_final

    g_q, g_v, g_w = (float(weights["g_q"]), float(weights["dso_g_v"]),
                     float(weights["g_w_dso_der"]))
    rows: List[Dict[str, Any]] = []
    for a in areas:
        if conditions is not None and a.condition not in conditions:
            continue
        lam = contraction(a, g_q, g_v, g_w)
        for i, key in enumerate(a.iface_keys):
            q = iterate_step(a, g_q, g_v, g_w, i, step_mvar=step_mvar,
                             n_iter=n_iter)
            n, cens = n_settle_final(q)
            rows.append({"condition": a.condition, "dso": a.area,
                         "trafo": key, "n_settle_pred": n,
                         "censored_pred": cens, "lambda_max_cont": lam,
                         "final_mvar": float(q[-1]),
                         "residual_pred_mvar": float(abs(step_mvar - q[-1]))})
    return rows


def summarise(rows: Sequence[Dict[str, Any]], budget: int = 9
              ) -> Dict[str, float]:
    """Pooled and per-DSO p50/p95 of the predicted counts."""
    out: Dict[str, float] = {}
    n = np.array([r["n_settle_pred"] for r in rows], dtype=float)
    out["pred_p50"] = float(np.percentile(n, 50))
    out["pred_p95"] = float(np.percentile(n, 95))
    out["pred_max"] = float(n.max())
    out[f"pred_share_le_{budget}"] = float(np.mean(n <= budget))
    for d in sorted({r["dso"] for r in rows}):
        nd = np.array([r["n_settle_pred"] for r in rows if r["dso"] == d],
                      dtype=float)
        out[f"pred_p50_{d}"] = float(np.percentile(nd, 50))
        out[f"pred_p95_{d}"] = float(np.percentile(nd, 95))
    out["lambda_max_cont"] = float(max(r["lambda_max_cont"] for r in rows))
    out["residual_pred_median"] = float(np.median(
        [r["residual_pred_mvar"] for r in rows]))
    return out
