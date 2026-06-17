#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/003_S_DSO_CIGRE_2026.py
===================================
Single-DSO OFO experiment for the CIGRE 2026 paper (for you, Johannes)

Architecture
------------
* **Plant** -- Full IEEE 39-bus + every active HV sub-network (DSO_1..DSO_4)
  under the ``wind_replace`` scenario.  This is what the plant-side
  ``pp.runpp`` solves every step.
* **TSO layer** -- No OFO.  Every TSO-connected DER (the wind_replace
  STATCOMs) runs a plant-side ``QVLocalLoop`` with q_mode="qv",
  V_ref=1.03 pu, slope=0.06 pu, deadband=+-0.01 pu.  No zone partitioning is used in the
  control sense (the partitioning code still runs but no TSO controller
  is ever stepped).
* **DSO_2** -- One :class:`DSOController` is constructed for DSO_2 only
  (every other HV sub-network has no OFO).  Its actuator block on the
  DER side is ``Q_cor`` (refactor_Qcor_method, ``use_q_cor_actuator=True``):
  the OFO writes a correction term (sgen sign convention) that is
  communicated to each DER and added to the plant-side QVLocalLoop's
  droop response, so the DER's reactive output is
  ``Q_DER = Q_cor + K · (V_ref - V)`` (linear droop, deadband=0.01 in 003).
  The H matrix is therefore the closed-loop ``∂y/∂Q_cor``, obtained by
  post-multiplying the open-loop ``∂y/∂Q_DER`` block by
  ``T' = (I + diag(K) · S_VQ)^{-1}`` (Soleimani §III-B eq. 18).
* **DSO_2 outputs** -- Q at the HV side of the three coupling 3-winding
  transformers (``q_hv_mvar``) and DN voltage magnitudes.
* **DSO_2 actuators** -- ``Q_cor`` on every DSO_2 sgen plus the three
  coupling 3W trafo OLTC tap positions.
* **DSO_2 H matrix** -- Built from the full-grid Jacobian via
  :meth:`JacobianSensitivities.build_sensitivity_matrix_H` with DSO_2-only
  index lists, then post-multiplied by ``T'`` on the DER columns.
  Cross-coupling effects from the rest of the grid are implicit in
  the inverted full Jacobian.

Goal
----
Track ``Q_TSDS = [0, 0, 0]`` Mvar at the three coupling transformers
under the existing wind_replace profile + contingency timeline.  The
exogenous setpoint vector is wired through
``MultiTSOConfig.q_pcc_setpoints_mvar_per_dso``; under
``tso_mode='local'`` the runner synthesises a
:class:`core.message.SetpointMessage` from this dict every step and
delivers it to ``DSO_2``.

Notes for the sensitivity-corrector colleague
----------------------------------------------
This script exposes a read+write interface on the DSO_2 controller's
cached sensitivity matrix ``H`` so a Kalman filter or neural network
can run online: at every DSO step ``k`` the corrector reads ``H(k)``,
predicts ``H(k+1)``, and the prediction is used by the controller in
the next step.  Four pieces do the wiring (see below):

* :func:`get_dso_h_view`        -- read function: labeled, filtered
  view of ``H`` (the colleague's read access).
* :func:`install_h_corrector`   -- step wrapper: replaces
  ``dso_ctrl.step`` with a version that calls a user-supplied
  predictor after every step and writes its result into ``_H_cache``.
* :func:`install_pe_noise`      -- step wrapper: adds gaussian noise
  to the DER (Q_cor) actuator output of the OFO at every step
  (persistent excitation for sensitivity learning).  Toggled by
  ``cfg.dso_pe_noise_enabled`` in :func:`make_config`.
* :func:`_setup_h_predictor`    -- example startup function (passed to
  the runner as ``pre_loop_hook``); primes the cache, prints the view
  once, optionally installs PE noise, and installs the predictor.

Vector layouts (DSO_2 in this script, currents and shunts excluded):

``y`` -- output vector (rows of H), in this order::

    y = [ Q_HV @ coupling_3W_trafo[t]   for t in cfg.interface_trafo_indices ]   # Mvar, generator convention
        [ V[bus]                         for bus in cfg.voltage_bus_indices  ]   # p.u.

``u`` -- actuator vector (cols of H), in this order::

    u = [ Q_cor_DER[d]                   for d in cfg.der_indices            ]   # Mvar, sgen convention (>0 = injection); correction term added to the local Q(V) droop
        [ tap_position[t]                for t in cfg.oltc_trafo_indices     ]   # integer steps relative to neutral

The DER's actual reactive output is ``Q_DER = Q_cor + K · (V_ref - V)``;
``Q_cor`` is what the OFO writes and what the controller's H is
linearized against.

``H = ∂y/∂u`` is the closed-loop linearization of the plant response
around the post-Phase-2 operating point.  Built from the inverse of
the full pandapower Jacobian projected onto DSO_2's actuator /
measurement sets, then post-multiplied on the DER columns by
``T' = (I + diag(K) · S_VQ)^{-1}`` to map the network-level
``∂y/∂Q_DER`` to ``∂y/∂Q_cor`` (see Soleimani §III-B eq. 18).
Cross-coupling from the rest of the grid is implicit in the inverted
full Jacobian.  Layout assumes ``use_q_cor_actuator=True`` and no
grid-forming DER (both hold in 003).

Predictor contract::

    predictor(dso_ctrl) -> Optional[np.ndarray]

is called once per DSO step, *after* ``dso_ctrl.step()`` returns.
* Return a numpy array of shape ``dso_ctrl._H_cache.shape`` (the FULL
  H, not the filtered view) to use it as ``H`` at step ``k+1``.
* Return ``None`` to leave ``_H_cache`` untouched (e.g. warm-up).

Per-step timing::

    step(k)  uses _H_cache (= H(k))
        -> step() returns
            -> predictor reads H(k) and any controller state
                -> predictor returns H(k+1)
                    -> _H_cache := H(k+1)
                        -> step(k+1) uses H(k+1)

For DSO_2 there are no shunts, so ``SensitivityUpdater.update()`` (which
runs inside ``step()`` and rescales shunt columns by V²) is a no-op
pass-through; the predictor's write is preserved through to the next
step.  When generalizing to DSO_1/3/4, beware that shunt columns may
get re-scaled inside ``step()`` and your predictor write would have to
account for that ordering.

Author: Manuel Schwenke, Johannes Ruppert
Date:   2026-05-05
"""
from __future__ import annotations

import importlib
import os
import pickle
import sys
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime
import numpy as np

if TYPE_CHECKING:
    from controller.dso_controller import DSOController

# Type alias for the H-predictor a colleague wires in.  See
# :func:`install_h_corrector` for the contract.
HPredictor = Callable[["DSOController"], Optional[np.ndarray]]

# ── Project root on sys.path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.multi_tso_config import MultiTSOConfig  # noqa: E402
from experiments.helpers.records import MultiTSOIterationRecord  # noqa: E402
from experiments.helpers.plant_io import decouple_trafo3w_hv_with_slack  # noqa: E402
from experiments.runners import run_multi_tso_dso  # noqa: E402
from sensitivity.numerical_h import compute_numerical_h_dso  # noqa: E402

# ``002_M_TSO_M_DSO_COMPARE`` starts with a digit, so the import must go
# through importlib rather than a normal ``from ... import ...``.
_compare = importlib.import_module("experiments.002_M_TSO_M_DSO_COMPARE")
#make_002_base_config = _compare.make_config

# ---------------------------------------------------------------------------
#  Predictor selection
#  Set H_PREDICTOR_MODE to one of:
#    "identity"  -- _unity_multiply_predictor: returns H * 1.0 (wiring smoke-test)
#    "kalman"    -- _kalman_h_predictor:        Kalman-filter random-walk estimator
#    "rls"       -- _rls_h_predictor:           Per-row RLS with forgetting factor
#    "ann"       -- _ANN_h_predictor:           Keras ANN (requires trained model)
#    "numerical" -- _numerical_h_predictor:     ORACLE: true plant secant H,
#                   recomputed each DSO step by closed-loop finite difference on
#                   the live (decoupled) net.  Upper-bound baseline — the OFO
#                   acts with perfect knowledge of the realized sensitivity.
#
#  Set H_PREDICTOR_ROWS to one of:
#    "all"        -- replace the full H matrix with the predictor's output (default)
#    "q_trafo"    -- replace only the Q_trafo rows (∂Q_tr/∂u, the first n_q_trafo
#                    rows); all other rows (voltage, current) are kept from the
#                    current _H_cache unchanged
#    "q_trafo+v"  -- replace the Q_trafo rows AND the voltage rows (∂V/∂u, the
#                    next n_v = len(voltage_bus_indices) rows); the current
#                    (I_line) rows are kept from _H_cache.  Adds voltage-row
#                    online estimation on top of the interface-power rows.
#
#  Set FROZEN_OP_POINT = True to lock the plant at a real historical operating
#  point (FROZEN_OP_TIMESTAMP).  Profiles are enabled so the network is
#  initialised correctly at that timestamp; then cfg.frozen_at keeps the
#  profiles constant for the entire run.  Contingency events are suppressed.
#  PE noise is forced on so estimators remain persistently excited.
# ---------------------------------------------------------------------------
H_PREDICTOR_MODE: str = "identity"
H_PREDICTOR_ROWS: str = "q_trafo+v"   # estimate Q_trafo + voltage rows (I_line rows kept analytical)
# Kalman forgetting factor, applied directly (NOT period-scaled) so the estimator
# behaves identically across DSO periods and is easy to compare run-to-run.
KALMAN_LAM: float = 1
# Step size for the "numerical" oracle's closed-loop finite difference.
# Matched to the deployment per-step OFO excitation (||du_DER|| ~= 3.4 Mvar) so
# the oracle returns the realized SECANT at the OFO scale, not the infinitesimal
# derivative.  The delta-sweep diagnostic confirmed the island is ~linear over
# [0.5, 3.4] Mvar, so the result is insensitive to the exact value in that band.
NUMERICAL_ORACLE_DELTA_MVAR: float = 3.4
FROZEN_OP_POINT:  bool = False
FROZEN_OP_TIMESTAMP: datetime = datetime(2016, 9, 7, 10, 0)  # real op-point to freeze at

# ---------------------------------------------------------------------------
#  Init-H bias injection
#  H_INIT_BIAS_STD > 0 multiplies each entry of the init H by an independent
#  log-normal(mean=1, std=H_INIT_BIAS_STD) factor — a multiplicative ±N%
#  perturbation of the analytical starting point.
#  H_INIT_BIAS_SEED is fixed so identity / kalman / ann runs all start from
#  exactly the same biased H, making comparisons directly meaningful.
#  Set H_INIT_BIAS_STD = 0.0 to disable (default).
# ---------------------------------------------------------------------------
H_INIT_BIAS_STD:  float = 0.1   # e.g. 0.10 for ±10% multiplicative noise; 0.0 = off
H_INIT_BIAS_SEED: int   = 0     # fixed seed → identical b
# ias for every run

# ---------------------------------------------------------------------------
#  DSO_2 interface mode
#  "slack"   -- replace the TN feed at each DSO_2 coupling 3W transformer's HV
#               side with a voltage-holding slack pinned to the operating point.
#               DSO_2 becomes its own island fed by 3 stiff (distributed) slacks,
#               so its online H estimation is NOT perturbed by TSO-side
#               transients (profiles, contingencies, TSO Q(V)).
#  "coupled" -- current behaviour: DSO_2 stays electrically coupled to the TSO
#               (use this to simulate the ANN in the present network state).
# ---------------------------------------------------------------------------
DSO2_INTERFACE_MODE: str = "slack"  # "slack" (decoupled, default) | "coupled"


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

def make_config() -> MultiTSOConfig:
    """Run configuration for the default multi-TSO / multi-DSO run (edit here).

    Single place to change the horizon, objective weights, OFO timing,
    profile and contingency schedule for ``main()``.  ``main_comparison()``
    keeps its own paired config.
    """
    cfg = MultiTSOConfig(
        n_total_s=60.0 * 60 * 5,      # 36-hour (2160-min) simulation
        tso_period_s=60.0 * 3,        # TS-OFO every 3 min
        dso_period_s=10.0,            # DSO-OFO each plant step (dt_s=60 >= 10)
        g_v=3E5,                      # TSO voltage tracking; drives PCC Q dispatch
        g_q=200,                      # DSO Q-tracking
        #tso_g_q_tie=0,
        #tso_g_res_sg=0,
        # ── DSO objective tuning ──
        dso_g_v=15000.0,              # reduced to avoid competing with Q tracking
        dso_g_qi=0,                   # integral Q-tracking (0 = off)
        dso_lambda_qi=0.95,           # leaky integrator decay
        dso_q_integral_max_mvar=200.0,
        dso_gamma_oltc_q=0.0,         # DER-primary, OLTC-backup
        # ── TSO weights (w-shift closed-loop curvature) ──
        g_w_der=100,
        g_w_gen=1e8,
        g_w_pcc=300,
        g_w_tso_oltc=100,
        install_tso_tertiary_shunts=False,
        g_w_tso_shunt=12000,
        # ── DSO weights ──
        g_w_dso_der=1000,
        g_w_dso_oltc=20,
        # ── Local-mode OLTC tap-rate limits (V1/V2 MT+NC, V3 NC) ──
        # max_step=1 (default) + wall-clock cooldown per OLTC type:
        #   MT (machine 2W gen-trafo) -> 1 tap / 180 s = once per TS interval.
        #   NC (coupler 3W interface) -> 1 tap / 60 s  = once per minute.
        # Cooldowns are wall-clock, hence robust to dt_s / dso_period_s.
        local_oltc_max_step_per_dt=1,
        #oltc_cooldown_s_mt=180.0,
        #oltc_cooldown_s_nc=60.0,
        use_fixed_zones=True,         # literature 3-area partition
        run_stability_analysis=False,
        sensitivity_update_interval=1E6,
        verbose=1,
    )

    # ---- DSO layer: only DSO_2 has an OFO controller -------------------
    cfg.dso_mode = "ofo"
    cfg.dso_ids_to_run = ["DSO_2"]

    # ---- Drop line-current rows from H (paper y = [Q_int, V]) ----------
    # With this off the DSO_2 sensitivity matrix reduces to [Q_interface | V]:
    # no ∂I/∂u rows enter the online estimator (Kalman/ANN) and the MIQP
    # enforces no line-current limits.  Regenerate kalman_matrices.npz after
    # toggling this — n_y (hence Q/R dimensions) changes.
    cfg.dso_monitor_currents = False

    # ---- DSO_2 interface: slack-decoupled vs. coupled (DSO2_INTERFACE_MODE)
    cfg.dso2_interface_slack = (DSO2_INTERFACE_MODE == "slack")
    if cfg.dso2_interface_slack:
        # Decoupling islands DSO_2 behind its own boundary slacks.  pandapower
        # cannot run a distributed slack across several islands at once, so the
        # whole run must use a per-island reference slack instead.
        cfg.distributed_slack = False

    # ---- Exogenous Q-setpoint vector at DSO_2's three 3W trafos --------
    # Order matches ``meta.hv_networks[1].coupling_trafo_indices``.
    cfg.q_pcc_setpoints_mvar_per_dso = {"DSO_2": [40.0, 20.0, 80.0]}

    # ---- Run length / timing -------------------------------------------
    cfg.n_total_s = 60.0 * 60 * 2  # 2 h smoke
    cfg.tso_period_s = 180.0  # cosmetic: TSO never steps anyway
    cfg.dso_period_s = 1  # 10
    cfg.warmup_s = 0.0
    cfg.start_time = datetime(2016, 9, 7, 10, 0)
    cfg.use_profiles = True

    # ---- Plant-side Q(V) loop damping ----------------------------------
    # 44 QVLocalLoops (4 TSO STATCOMs + 40 DSO sgens) iterate in parallel
    # inside every ``pp.runpp(run_control=True)``.  STATCOM-class units
    # (S_n ~600 Mvar with slope 0.07) have ~9 GVar/pu open-loop gain;
    # the published-stable damping is 0.1 (see QVLocalLoop docstring).
    # The 002 default of 0.5 is fine for fewer loops but oscillates here.
    # cfg.qv_local_damping = 0.1
    # cfg.qv_local_max_step_frac = None
    # cfg.qv_local_tol_mvar = 0.1

    # ---- Live plots off for diagnostic runs ---------------------------
    cfg.live_plot_controller = False
    cfg.live_plot_cascade = False
    cfg.live_plot_system = False
    cfg.live_plot_tracking = True

    # ---- Persistent excitation (sensitivity learning) -----------------
    # When enabled, gaussian noise N(0, sigma^2) is added to the DSO
    # OFO's continuous (DER Q_cor) actuator output AT EVERY DSO step,
    # AFTER the MIQP solves and BEFORE the command is sent to the plant.
    # The controller's internal u_current is also incremented by the
    # noise so the next OFO step starts from u(k+1)' = u(k+1) + eps and
    # does not "fight" the excitation.  OLTC tap commands are NOT
    # perturbed (integer actuators).  Useful for downstream H estimation
    # (Kalman filter / NN) where the input must be persistently exciting.
    cfg.dso_pe_noise_enabled = True
    cfg.dso_pe_noise_std_mvar = 0.01  # std of N(0, sigma^2) on Q_cor [Mvar] (quartered from 1.0: steady-state ||du|| noise floor ~0.9 Mvar; profiles still excite the varying case)
    cfg.dso_pe_noise_seed = 42  # RNG seed for reproducibility

    # ---- Output directory ---------------------------------------------
    cfg.result_dir = os.path.join("results", "003_cigre_2026")

    # ---- Frozen operating point (estimator convergence mode) ----------
    # Profiles are kept ON so the network initialises at the real historical
    # operating point; cfg.frozen_at then holds that profile constant for the
    # entire run.  This avoids the artificial mean-profile base-case.
    if FROZEN_OP_POINT:
        cfg.use_profiles = True
        cfg.start_time = FROZEN_OP_TIMESTAMP
        cfg.frozen_at = FROZEN_OP_TIMESTAMP
        cfg.contingencies = []
        cfg.dso_pe_noise_enabled = True
        print(f"[003] FROZEN_OP_POINT=True: frozen at {FROZEN_OP_TIMESTAMP:%Y-%m-%d %H:%M}, "
              f"contingencies off, PE noise on")

    return cfg



# ---------------------------------------------------------------------------
#  H predictor interface  (read function + step wrappers + setup function)
# ---------------------------------------------------------------------------
#
# These pieces together let a Kalman filter / NN replace the controller's
# cached H matrix at every DSO step, and (optionally) inject gaussian
# input excitation for sensitivity learning.  See the module docstring
# above for the y / u / H layout and the predictor contract.  Workflow:
#
#       def my_h_predictor(dso_ctrl):
#           H_now = dso_ctrl._H_cache              # full shape, e.g. (24, 13)
#           meas  = dso_ctrl._last_measurement     # latest Measurement (V, Q, ...)
#           view  = get_dso_h_view(dso_ctrl)       # labeled view if you want named rows/cols
#
#           H_next = ...                           # KF / NN: predict H for the next DSO step
#           return H_next                          # MUST be same shape as H_now; or None to skip
#
#       # Install from the startup function passed to run_multi_tso_dso
#       # (see _setup_h_predictor below for an example):
#       install_h_corrector(state["dso_controllers"]["DSO_2"], my_h_predictor)
#
#       # Optional: gaussian persistent-excitation noise on Q_cor output.
#       # Toggle in make_config (cfg.dso_pe_noise_enabled = True/False);
#       # _setup_h_predictor reads the flag and installs the wrapper.


def get_dso_h_view(
    dso_ctrl: "DSOController",
    *,
    exclude_currents: bool = True,
    exclude_shunts: bool = True,
) -> Dict[str, Any]:
    """Return a labeled, optionally-filtered view of the DSO controller's H.

    Default filters drop current rows (``I_line``) and shunt columns
    (``shunt``), leaving voltages + Q at trafos as outputs and DER + OLTC
    as actuators -- the layout the corrector colleague is using.

    Parameters
    ----------
    dso_ctrl : DSOController
        The DSO controller to inspect.  ``dso_ctrl._H_cache`` is primed
        via ``_build_sensitivity_matrix()`` if it has not been built yet.
    exclude_currents : bool, default True
        Drop ``I_line`` rows.
    exclude_shunts : bool, default True
        Drop ``shunt`` columns.

    Returns
    -------
    dict
        Keys:
          ``H``              : np.ndarray, filtered (n_y x n_u) copy of ``_H_cache``.
          ``row_labels``     : list[str], one per row, e.g. ``"V[bus_42]"``.
          ``col_labels``     : list[str], one per column, e.g. ``"OLTC_3W[3]"``.
          ``row_units``      : list[str], physical unit for each row.
          ``col_units``      : list[str], physical unit for each column.
          ``row_kinds``      : list[str], one of ``{"Q_trafo","V_bus","I_line"}``.
          ``col_kinds``      : list[str], one of ``{"Q_cor","OLTC","shunt"}``.
          ``kept_row_mask``  : np.ndarray[bool], length = full-H rows.
          ``kept_col_mask``  : np.ndarray[bool], length = full-H cols.

    Notes
    -----
    The returned ``H`` is a copy; mutating it does not affect the
    controller cache.  To inject a corrected H, use
    :func:`install_h_corrector`.
    """
    if dso_ctrl._H_cache is None:
        dso_ctrl._build_sensitivity_matrix()
    H_full = dso_ctrl._H_cache
    m = dso_ctrl._H_mappings or {}
    cfg = dso_ctrl.config

    # ── Row metadata: order is [Q_trafo2w | Q_trafo3w_HV | V_bus | I_line] ──
    row_kinds: List[str] = []
    row_labels: List[str] = []
    row_units: List[str] = []
    for ti in m.get("trafos", []):
        row_kinds.append("Q_trafo")
        row_labels.append(f"Q_trafo2w[{ti}]")
        row_units.append("Mvar (gen conv)")
    for ti in m.get("trafo3w", []):
        row_kinds.append("Q_trafo")
        row_labels.append(f"Q_trafo3w_HV[{ti}]")
        row_units.append("Mvar (gen conv)")
    for bi in m.get("obs_buses", []):
        row_kinds.append("V_bus")
        row_labels.append(f"V[bus_{bi}]")
        row_units.append("p.u.")
    for li in m.get("lines", []):
        row_kinds.append("I_line")
        row_labels.append(f"I[line_{li}]")
        row_units.append("p.u.")

    # ── Column metadata: order is [Q_cor (per-DER) | OLTC_2W | OLTC_3W | shunt] ──
    # The DER columns of H are the closed-loop ∂y/∂Q_cor (T' applied) under
    # use_q_cor_actuator=True.  Q_cor is the correction term the OFO writes
    # to each DER; the DER's actual Q is Q_cor + K·(V_ref - V).
    col_kinds: List[str] = []
    col_labels: List[str] = []
    col_units: List[str] = []
    for d in cfg.der_indices:
        col_kinds.append("Q_cor")
        col_labels.append(f"Q_cor[sgen_{d}]")
        col_units.append("Mvar (sgen conv, +inj)")
    for ti in m.get("oltc_trafos", []):
        col_kinds.append("OLTC")
        col_labels.append(f"OLTC_2W[{ti}]")
        col_units.append("steps")
    for ti in m.get("oltc_trafo3w", []):
        col_kinds.append("OLTC")
        col_labels.append(f"OLTC_3W[{ti}]")
        col_units.append("steps")
    for bi in m.get("shunt_buses", []):
        col_kinds.append("shunt")
        col_labels.append(f"shunt[bus_{bi}]")
        col_units.append("steps")

    if len(row_kinds) != H_full.shape[0]:
        raise RuntimeError(
            f"H-view row count mismatch: counted {len(row_kinds)} from mappings "
            f"but H has {H_full.shape[0]} rows.  Mappings keys: {list(m.keys())}"
        )
    if len(col_kinds) != H_full.shape[1]:
        raise RuntimeError(
            f"H-view col count mismatch: counted {len(col_kinds)} (n_der="
            f"{len(cfg.der_indices)}) but H has {H_full.shape[1]} cols.  "
            f"Layout assumes w-shift actuator mode and no grid-forming DER; "
            f"V_gf or Q_realized splicing would invalidate this view."
        )

    # ── Filter masks ─────────────────────────────────────────────────────
    row_mask = np.array(
        [not (exclude_currents and k == "I_line") for k in row_kinds],
        dtype=bool,
    )
    col_mask = np.array(
        [not (exclude_shunts and k == "shunt") for k in col_kinds],
        dtype=bool,
    )
    kept_rows = np.flatnonzero(row_mask).tolist()
    kept_cols = np.flatnonzero(col_mask).tolist()

    # Slice once + copy so caller cannot mutate the controller's cache.
    H = H_full[np.ix_(kept_rows, kept_cols)].copy()

    return {
        "H": H,
        "row_labels": [row_labels[i] for i in kept_rows],
        "col_labels": [col_labels[j] for j in kept_cols],
        "row_units":  [row_units[i]  for i in kept_rows],
        "col_units":  [col_units[j]  for j in kept_cols],
        "row_kinds":  [row_kinds[i]  for i in kept_rows],
        "col_kinds":  [col_kinds[j]  for j in kept_cols],
        "kept_row_mask": row_mask,
        "kept_col_mask": col_mask,
    }


def install_h_corrector(
    dso_ctrl: "DSOController",
    predictor: HPredictor,
) -> None:
    """Wrap ``dso_ctrl.step`` so ``predictor(dso_ctrl)`` runs after every step.

    Per step ``k``:
      1. ``dso_ctrl.step(...)`` executes -- uses ``_H_cache`` (= ``H(k)``).
      2. ``predictor(dso_ctrl)`` is called.  It can read any controller
         attribute (``_H_cache``, ``_H_mappings``, ``_last_measurement``,
         ...) and returns either:
           * an ``np.ndarray`` of shape ``_H_cache.shape`` -- written into
             ``_H_cache`` and used as ``H(k+1)`` at the next step.
           * ``None`` -- ``_H_cache`` left untouched (e.g. warm-up window).
      3. Step ``k+1`` runs with the chosen ``H``.

    Calling this twice on the same controller swaps in the new predictor
    without double-wrapping.
    """
    # Detect a previous installation: restore the original step before
    # re-wrapping so we don't pile wrappers on top of each other.
    original_step = getattr(dso_ctrl, "_step_pre_h_corrector", None)
    if original_step is None:
        original_step = dso_ctrl.step
        dso_ctrl._step_pre_h_corrector = original_step  # type: ignore[attr-defined]

    def step_with_corrector(*args, **kwargs):
        result = original_step(*args, **kwargs)
        H_next = predictor(dso_ctrl)
        if H_next is None:
            return result
        if not isinstance(H_next, np.ndarray):
            raise TypeError(
                f"H-predictor must return np.ndarray or None, got "
                f"{type(H_next).__name__}"
            )
        if dso_ctrl._H_cache is None:
            raise RuntimeError(
                "H-predictor returned an array but _H_cache is None -- "
                "did step() fail to build H?"
            )
        if H_next.shape != dso_ctrl._H_cache.shape:
            raise ValueError(
                f"H-predictor returned shape {H_next.shape} but "
                f"_H_cache has shape {dso_ctrl._H_cache.shape}.  Predictor "
                f"must return the FULL-shape H, not the filtered view."
            )
        # Write into the sensitivity updater's base so the in-place reset
        # inside SensitivityUpdater.update() uses the predictor's H, not the
        # frozen init-time H.  Falls back to a plain reference swap when no
        # updater is present (e.g. unit tests).
        if dso_ctrl._sensitivity_updater is not None:
            dso_ctrl._sensitivity_updater.override_base(H_next)
        dso_ctrl._H_cache = H_next
        return result

    dso_ctrl.step = step_with_corrector  # type: ignore[method-assign]


def install_pe_noise(
    dso_ctrl: "DSOController",
    *,
    std_mvar: float,
    rng: np.random.Generator,
) -> None:
    """Wrap ``dso_ctrl.step`` so gaussian noise is added to DER Q_cor commands.

    Per step k:
      1. ``dso_ctrl.step(...)`` runs as normal -- OFO computes
         ``u(k+1) = u(k) + w`` where ``w`` is the MIQP step.
      2. Noise ``eps ~ N(0, std_mvar^2)`` of size ``n_der`` is drawn
         from ``rng``.
      3. The DER (continuous) slice of ``u_new`` and ``u_continuous`` on
         the returned ``ControllerOutput`` is incremented by ``eps``,
         so the runner applies ``u(k+1)' = u(k+1) + eps`` to
         ``net.sgen.q_set_mvar`` (see ``apply_dso_controls`` in
         ``experiments/helpers/plant_io.py``).
      4. ``dso_ctrl._u_current`` is also incremented by ``eps`` so the
         next OFO step starts from ``u(k+1)'`` rather than ``u(k+1)``.
         This prevents the controller from "fighting" the noise -- the
         excitation becomes part of the controller's owned state.
      5. OLTC (integer) commands and any other non-DER columns are NOT
         perturbed.

    Parameters
    ----------
    dso_ctrl : DSOController
        The DSO controller to wrap.
    std_mvar : float
        Standard deviation of the gaussian noise on each DER's Q_cor
        command, in Mvar.  Set to 0 to effectively disable.
    rng : np.random.Generator
        Random source.  Use a seeded ``np.random.default_rng(seed)`` for
        reproducibility.

    Notes
    -----
    Layout assumption: ``u_new = [Q_cor (n_der) | V_gf (n_gf) |
    OLTC (n_oltc) | shunt (n_shunts)]`` with the DER block at indices
    ``[0, n_der)``.  Holds for 003 (no V_gf, no shunts).  If the layout
    changes in the future this slicing must be revisited.

    Δu  =  dso_ctrl._u_current   -  u_prev_stored        # = w(k) + ε(k)
    Δy  =  dso_ctrl._last_measurement  -  y_prev_stored

    Because Δu now contains an independent gaussian component on every DER channel
    (not just the correlated direction the OFO's w would choose), the regression
    Δy ≈ H · Δu is full-rank in expectation — that's what makes H identifiable online.
    Without the noise, w lives on a low-dimensional manifold dictated by the gradient
    direction and the estimator can't recover off-diagonal entries.

    """
    n_der = len(dso_ctrl.config.der_indices)
    original_step = dso_ctrl.step

    def step_with_pe(*args, **kwargs):
        result = original_step(*args, **kwargs)
        if n_der > 0 and std_mvar > 0.0:
            noise = rng.normal(0.0, std_mvar, size=n_der)
            # Mutate what the runner applies to the plant ...
            result.u_new[:n_der] += noise
            result.u_continuous[:n_der] += noise
            # ... and what the controller starts the next step from.
            if dso_ctrl._u_current is not None:
                dso_ctrl._u_current[:n_der] += noise
        return result

    dso_ctrl.step = step_with_pe  # type: ignore[method-assign]


# ─────────────────────────────────────────────────────────────────────────────
#  H-predictor implementations and training utilities
# ─────────────────────────────────────────────────────────────────────────────
#
# Both predictors are callable class instances that satisfy HPredictor exactly
# like plain functions but carry mutable state between consecutive DSO steps.
#
# Workflow
# --------
#   1. collect_training_data()     -- run 003 with PE noise; save (u, y, H)
#   2. generate_kalman_matrices()  -- estimate Q, R from that data
#   3. train_ann_model()           -- fit Keras model; save weights + stats
#   4. In _setup_h_predictor swap the predictor:
#        install_h_corrector(dso_ctrl, _kalman_h_predictor)
#        # or:
#        install_h_corrector(dso_ctrl, _ANN_h_predictor)

_RESULT_DIR_003       = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "003_cigre_2026")
_TRAINING_DATA_PATH   = os.path.join(_RESULT_DIR_003, "training_data.npz")
_KALMAN_MATRICES_PATH = os.path.join(_RESULT_DIR_003, "kalman_matrices.npz")
_ANN_MODEL_PATH       = os.path.join(_RESULT_DIR_003, "ann_h_model.keras")
_ANN_STATS_PATH       = os.path.join(_RESULT_DIR_003, "ann_h_stats.npz")


def get_dso2_features(dso_ctrl: "DSOController") -> Dict[str, np.ndarray]:
    """Return all DSO_2-relevant measurements in a named dict.

    Call this inside any predictor — ``dso_ctrl._last_measurement`` is
    already populated when the predictor runs (it is set at the very start
    of ``dso_ctrl.step()`` and is not cleared afterwards).

    Keys
    ----
    q_trafo_mvar : Q at HV side of the three coupling 3W trafos [Mvar, load conv.]
                   shape (n_interface,) — these are the rows of H that track Q_set
    tap_pos      : OLTC tap positions of the coupling trafos, shape (n_oltc,)
                   integer steps; same ordering as cfg.oltc_trafo_indices
    der_q_mvar   : actual DER reactive output Q_DER = q_set + K·(V_anchor - V) [Mvar]
                   shape (n_der,) — this is what the plant actually produces
    der_p_mw     : DER active power output [MW], shape (n_der,)
    q_set_mvar   : q_set setpoint the OFO last commanded [Mvar], shape (n_der,)
                   = u[:n_der]; this is what H is linearised against
    v_buses_pu   : voltages at cfg.voltage_bus_indices [p.u.], shape (n_v_bus,)
    y            : OFO output vector [q_trafo | v_buses], shape (n_y,)
                   = rows of H; built by _extract_outputs (same as inside step())
    u            : OFO actuator vector [q_set | tap_pos], shape (n_u,)
                   = cols of H; = dso_ctrl._u_current
    """
    meas = dso_ctrl._last_measurement
    cfg  = dso_ctrl.config
    if meas is None or dso_ctrl._u_current is None:
        raise RuntimeError(
            "get_dso2_features() called before the controller has stepped once."
        )

    n_der = len(cfg.der_indices)

    v_buses = np.array(
        [
            float(meas.voltage_magnitudes_pu[np.where(meas.bus_indices == bi)[0][0]])
            for bi in cfg.voltage_bus_indices
        ],
        dtype=np.float64,
    )

    return {
        "q_trafo_mvar": meas.interface_q_hv_side_mvar.copy(),
        "tap_pos":      meas.oltc_tap_positions.astype(np.float64),
        "der_q_mvar":   meas.der_q_mvar.copy(),
        "der_p_mw":     meas.der_p_mw.copy(),
        "q_set_mvar":   dso_ctrl._u_current[:n_der].copy(),
        "v_buses_pu":   v_buses,
        "y":            dso_ctrl._extract_outputs(meas),
        "u":            dso_ctrl._u_current.copy(),
    }


class _KalmanHPredictor:
    """Tracks H as a linear random-walk state with noisy Δy ≈ H·Δu observations.

    State model :  h(k+1) = h(k) + w(k),       w ~ N(0, Q)
    Observation :  Δy(k)  = C(Δu) h(k) + v(k), v ~ N(0, R)
    where h = H.flatten() (row-major) and C(Δu) = I_{n_y} ⊗ Δu^T.

    Q and R are loaded from ``_KALMAN_MATRICES_PATH`` on the first call.
    Falls back to identity defaults if absent; run
    :func:`generate_kalman_matrices` first for data-driven matrices.
    """

    def __init__(self, min_delta_u_norm: float = 0.05, forgetting_factor: float = 0.997,
                 p_min_frac: float = 10.0, p_min_warmup_steps: int = 60) -> None:
        self.min_delta_u_norm = min_delta_u_norm
        # Forgetting factor prevents P from collapsing to ~0 after initial convergence.
        # Without it, K → 0 and the filter can no longer track slow H drift from
        # actuator movements, causing error to re-increase after the initial dip.
        self.lam = forgetting_factor
        # p_min_frac: floor on tr(P) as a fraction of the initial tr(P), active only
        # during the warmup window.  Keeps K elevated long enough to correct a large
        # cold-start bias; after warmup P is free to collapse so noise stops accumulating.
        self.p_min_frac = p_min_frac
        self.p_min_warmup_steps = p_min_warmup_steps
        self._Q: Optional[np.ndarray] = None
        self._R: Optional[np.ndarray] = None
        self._u_scale: Optional[np.ndarray] = None  # per-channel |Δu| RMS for the Σ_q norm
        self._pe_dither_energy: float = 0.0          # expected ‖Δ(PE noise)‖² in the normed step
        self._h: Optional[np.ndarray] = None   # vectorised H estimate
        self._P: Optional[np.ndarray] = None   # estimate covariance
        self._P_max: Optional[float] = None    # trace ceiling for PSD-safe cap
        self._P_min: Optional[float] = None    # trace floor active during warmup
        self._step_count: int = 0
        self._y_prev: Optional[np.ndarray] = None
        self._u_prev: Optional[np.ndarray] = None

    def _load_matrices(self, n_state: int, n_y: int) -> None:
        if os.path.exists(_KALMAN_MATRICES_PATH):
            d = np.load(_KALMAN_MATRICES_PATH)
            self._Q, self._R = d["Q"], d["R"]
            # Per-channel |Δu| RMS used to normalise the Σ_q = ‖Δu‖·Σ_q0 scaling.
            # Missing in legacy files -> ones (raw norm); regenerate for the data-driven scale.
            n_u = int(round(n_state / n_y))
            self._u_scale = d["u_scale"] if "u_scale" in d else np.ones(n_u)
            print(f"[KF] loaded Q {self._Q.shape}, R {self._R.shape}, "
                  f"u_scale {self._u_scale.shape}")
        else:
            self._Q = np.eye(n_state) * 1e-6
            self._R = np.eye(n_y) * 1e-4
            self._u_scale = np.ones(int(round(n_state / n_y)))
            print("[KF] no kalman_matrices.npz found — using I defaults. "
                  "Run generate_kalman_matrices() for data-driven matrices.")

    def reset(self) -> None:
        """Reset KF state; re-seeds from the controller's H on next call."""
        self._h = self._P = self._y_prev = self._u_prev = None
        self._P_max = None
        self._P_min = None
        self._step_count = 0

    def __call__(self, dso_ctrl: "DSOController") -> Optional[np.ndarray]:
        if dso_ctrl._H_cache is None or dso_ctrl._u_current is None:
            return None

        try:
            feats = get_dso2_features(dso_ctrl)
        except RuntimeError:
            return None

        # feats keys available here (replace manual extraction with these):
        #   feats["y"]           -- OFO output vector [q_trafo | v_buses]   (n_y,)
        #   feats["u"]           -- OFO actuator vector [q_cor | tap_pos]   (n_u,)
        #   feats["q_trafo_mvar"]-- Q at coupling 3W trafos                 (n_trafo,)
        #   feats["tap_pos"]     -- OLTC tap positions                       (n_oltc,)
        #   feats["der_q_mvar"]  -- actual DER Q output (Q_cor + K*(Vref-V))(n_der,)
        #   feats["der_p_mw"]    -- DER active power                         (n_der,)
        #   feats["q_set_mvar"]  -- q_set commanded by OFO (= u[:n_der])     (n_der,)
        #   feats["v_buses_pu"]  -- voltages at monitored buses               (n_v,)

        n_y, n_u = dso_ctrl._H_cache.shape
        n_state = n_y * n_u

        if self._Q is None or self._Q.shape[0] != n_state:
            self._load_matrices(n_state, n_y)

        y = feats["y"]
        u = feats["u"]

        # Warm-up: seed h and P from the controller's analytical H
        if self._h is None:
            self._h = dso_ctrl._H_cache.flatten().copy()
            # P_init reflects actual initial uncertainty (bias on H), not Q magnitude.
            # Tying P_init to Q would make K negligibly small when Q is tiny
            # (e.g. when H barely changes step-to-step even with profiles on).
            h_rms = float(np.sqrt(np.mean(self._h ** 2))) or 1e-4
            sigma_init = 0.30 * h_rms   # ~30% per-entry std matches the init bias
            self._P = (sigma_init ** 2) * np.eye(n_state)
            p_init_trace = float(np.trace(self._P))
            self._P_min = self.p_min_frac * p_init_trace
            # P_max must be at least P_min so ceiling never sits below the warmup floor.
            self._P_max = max(p_init_trace, self._P_min)
            self._step_count = 0
            # Expected energy that the PE dither alone injects into the normed step,
            # so the Σ_q scaling reflects natural controller motion, not the excitation
            # noise.  PE noise ~ N(0,σ_u²) i.i.d. on the n_der DER channels, so the
            # differenced dither has variance 2σ_u² per channel; normalise by u_scale.
            cfg = dso_ctrl.config
            sigma_u = float(getattr(cfg, "dso_pe_noise_std_mvar", 0.0) or 0.0)
            if getattr(cfg, "dso_pe_noise_enabled", False) and sigma_u > 0.0 and self._u_scale is not None:
                n_der = len(cfg.der_indices)
                self._pe_dither_energy = float(
                    2.0 * sigma_u ** 2 * np.sum(1.0 / self._u_scale[:n_der] ** 2))
            self._y_prev, self._u_prev = y, u
            return None  # keep analytical H on the first call

        delta_u = u - self._u_prev
        delta_y = y - self._y_prev

        # Predict — Σ_q = ‖Δu‖²·Σ_q0 ties the assumed H drift to actuator motion, so
        # process noise shrinks toward zero at convergence (Δu→0) instead of injecting
        # fictitious diffusion.  The squared scaling is the covariance interpretation:
        # if the per-step H increment scales like Δu, its covariance scales like ‖Δu‖².
        # Δu is per-channel normalised by its RMS scale so DER and OLTC moves contribute
        # comparably; the expected PE-dither energy is removed so s² reflects natural
        # controller motion only (s²→0 once the controller settles, even though the dither
        # keeps moving).  s²≈1 at a typical full step.
        # Forgetting factor keeps P from collapsing; trace cap bounds growth (PSD-safe).
        du_n2 = float(np.sum((delta_u / self._u_scale) ** 2))  # ‖Δu‖² (normalised, dimensionless)
        s2 = max(0.0, du_n2 - self._pe_dither_energy) / len(delta_u)  # ∝ ‖Δu‖² → covariance scaling
        h_p = self._h.copy()
        P_p = self._P / self.lam + s2 * self._Q
        tr = float(np.trace(P_p))
        if tr > self._P_max:
            P_p *= self._P_max / tr

        # Update — skip when ‖Δu‖ is too small (ill-conditioned regression)
        if np.linalg.norm(delta_u) >= self.min_delta_u_norm:
            C = np.kron(np.eye(n_y), delta_u.reshape(1, -1))  # (n_y, n_state)
            S = C @ P_p @ C.T + self._R                        # (n_y, n_y)
            K = np.linalg.solve(S, C @ P_p).T                  # (n_state, n_y)
            self._h = h_p + K @ (delta_y - C @ h_p)
            self._P = (np.eye(n_state) - K @ C) @ P_p
            # P_min floor active only during warmup window: uniform top-up preserves
            # PD and symmetry.  After warmup P is free to collapse so the filter stops
            # injecting noise once the cold-start bias has been corrected.
            if self._step_count < self.p_min_warmup_steps:
                tr_post = float(np.trace(self._P))
                if tr_post < self._P_min:
                    self._P += (self._P_min - tr_post) / n_state * np.eye(n_state)
        else:
            self._h, self._P = h_p, P_p

        self._step_count += 1
        self._y_prev, self._u_prev = y, u
        return self._h.reshape(n_y, n_u)


_kalman_h_predictor = _KalmanHPredictor()


class _PerRowRLSPredictor:
    """Per-row Recursive Least Squares H estimator with forgetting factor.

    Observation model: Δy_i = H[i,:] · Δu  (rows are physically decoupled)
    Each of the n_y rows maintains its own (n_u × n_u) covariance P_i.
    Forgetting factor λ < 1 exponentially discounts old observations.

    Advantages over full Kalman:
    - No Q/R calibration needed; λ is the only tuning knob
    - P stays bounded (λ prevents unbounded growth)
    - OLTC columns remain near their initial values when Δu_tap ≈ 0
    """

    def __init__(
        self,
        forgetting_factor: float = 0.995,
        delta_init: float = 1.0,
        min_delta_u_norm: float = 0.05,
    ):
        self.lam = forgetting_factor
        self.delta = delta_init
        self.min_du = min_delta_u_norm
        self._H: Optional[np.ndarray] = None      # (n_y, n_u)
        self._P: Optional[list] = None            # list of n_y (n_u, n_u) arrays
        self._y_prev: Optional[np.ndarray] = None
        self._u_prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._H = self._P = self._y_prev = self._u_prev = None

    def __call__(self, dso_ctrl: "DSOController") -> Optional[np.ndarray]:
        if dso_ctrl._H_cache is None or dso_ctrl._u_current is None:
            return None
        try:
            feats = get_dso2_features(dso_ctrl)
        except RuntimeError:
            return None

        n_y, n_u = dso_ctrl._H_cache.shape
        y: np.ndarray = feats["y"]
        u: np.ndarray = feats["u"]

        if self._H is None:
            self._H = dso_ctrl._H_cache.copy()
            # Scale the initial covariance to the H magnitude so the first
            # update corrects ~delta*100 % of each entry rather than
            # instantly overshooting. delta=1 → ~1 % correction per step.
            h_rms_sq = float(np.mean(self._H ** 2)) or 1e-8
            delta_scaled = self.delta * h_rms_sq
            self._P = [delta_scaled * np.eye(n_u) for _ in range(n_y)]
            self._y_prev = y.copy()
            self._u_prev = u.copy()
            return None

        delta_u = u - self._u_prev
        delta_y = y - self._y_prev

        if np.linalg.norm(delta_u) >= self.min_du:
            c = delta_u  # shape (n_u,)
            for i in range(n_y):
                P_i = self._P[i]
                Pc = P_i @ c                              # (n_u,)
                denom = self.lam + float(c @ Pc)
                k_i = Pc / denom                          # gain (n_u,)
                e_i = float(delta_y[i]) - float(c @ self._H[i])
                self._H[i] += k_i * e_i
                self._P[i] = (P_i - np.outer(k_i, Pc)) / self.lam

        self._y_prev = y.copy()
        self._u_prev = u.copy()
        return self._H.copy()


_rls_h_predictor = _PerRowRLSPredictor()


class _ANNHPredictor:
    """ANN predictor: estimates H from the current I/O vectors [y, u].

    Feature vector (must match :func:`train_ann_model`): [y(k), u(k)]
    Output: H_analytical(k).flatten() reshaped to (n_y, n_u).

    Loads a Keras model from ``_ANN_MODEL_PATH`` on the first call.
    Returns ``None`` (leaves H untouched) if the model file is absent;
    run :func:`train_ann_model` first.
    """

    def __init__(self) -> None:
        self._model = None
        self._tried_load = False
        self._x_mean: Optional[np.ndarray] = None
        self._x_std:  Optional[np.ndarray] = None
        self._t_mean: Optional[np.ndarray] = None
        self._t_std:  Optional[np.ndarray] = None

    def _try_load(self) -> None:
        self._tried_load = True
        if not os.path.exists(_ANN_MODEL_PATH):
            print("[ANN] no ann_h_model.keras found — returning None. "
                  "Run train_ann_model() first.")
            return
        try:
            from tensorflow import keras
            self._model = keras.models.load_model(_ANN_MODEL_PATH)
            if os.path.exists(_ANN_STATS_PATH):
                st = np.load(_ANN_STATS_PATH)
                self._x_mean, self._x_std = st["x_mean"], st["x_std"]
                self._t_mean, self._t_std = st["t_mean"], st["t_std"]
            print(f"[ANN] loaded model from {_ANN_MODEL_PATH}")
        except Exception as exc:
            print(f"[ANN] failed to load model: {exc}")

    def __call__(self, dso_ctrl: "DSOController") -> Optional[np.ndarray]:
        if not self._tried_load:
            self._try_load()
        if self._model is None:
            return None
        if dso_ctrl._H_cache is None or dso_ctrl._u_current is None:
            return None

        try:
            feats = get_dso2_features(dso_ctrl)
        except RuntimeError:
            return None

        # feats keys available for building the input feature vector:
        #   feats["y"]           -- OFO output vector [q_trafo | v_buses]   (n_y,)
        #   feats["u"]           -- OFO actuator vector [q_cor | tap_pos]   (n_u,)
        #   feats["q_trafo_mvar"]-- Q at coupling 3W trafos                 (n_trafo,)
        #   feats["tap_pos"]     -- OLTC tap positions                       (n_oltc,)
        #   feats["der_q_mvar"]  -- actual DER Q output (Q_cor + K*(Vref-V))(n_der,)
        #   feats["der_p_mw"]    -- DER active power                         (n_der,)
        #   feats["q_set_mvar"]  -- q_set commanded by OFO (= u[:n_der])     (n_der,)
        #   feats["v_buses_pu"]  -- voltages at monitored buses               (n_v,)

        n_y, n_u = dso_ctrl._H_cache.shape
        # Feature vector: [y(k), u(k)] — must match train_ann_model.
        x = np.concatenate([feats["y"], feats["u"]]).reshape(1, -1).astype(np.float32)

        if self._x_mean is not None:
            x = (x - self._x_mean) / (self._x_std + 1e-8)
        h_next = self._model.predict(x, verbose=0)[0]
        if self._t_mean is not None:
            h_next = h_next * (self._t_std + 1e-8) + self._t_mean

        return h_next.reshape(n_y, n_u)


_ANN_h_predictor = _ANNHPredictor()


# ── Training data collection ──────────────────────────────────────────────────

def collect_training_data(
    n_total_s: float = 7200.0,
    pe_noise_std_mvar: float = 0.001,
    output_path: str = _TRAINING_DATA_PATH,
) -> str:
    """Run the 003 simulation and record measurements at every DSO_2 step.

    PE noise is enabled so that Δu is persistently exciting and off-diagonal
    H entries are identifiable from the data.

    Saved .npz arrays  (N = number of DSO steps recorded)
    -----------------
    u            : (N, n_u)       OFO actuator vector [q_cor | tap_pos]
    y            : (N, n_y)       OFO output vector   [q_trafo | v_buses]
    H            : (N, n_y, n_u)  analytical sensitivity matrix
    q_trafo_mvar : (N, n_trafo)   Q at HV side of coupling 3W trafos [Mvar, load conv.]
    tap_pos      : (N, n_oltc)    OLTC tap positions
    der_q_mvar   : (N, n_der)     actual DER Q output  Q_DER = Q_cor + K·(Vref−V)
    der_p_mw     : (N, n_der)     DER active power [MW]
    q_set_mvar   : (N, n_der)     q_set commanded by OFO  (= u[:n_der])
    v_buses_pu   : (N, n_v_bus)   voltages at monitored buses [p.u.]

    Returns the absolute path of the saved file.
    """
    cfg = make_config()
    cfg.n_total_s = n_total_s
    cfg.dso_pe_noise_enabled = True
    cfg.dso_pe_noise_std_mvar = pe_noise_std_mvar
    cfg.live_plot_controller = False
    cfg.live_plot_cascade = False

    cfg.local_sensitivities_tso=True
    cfg.local_sensitivities_dso=True

    records: List[Dict[str, np.ndarray]] = []
    _net_ref: List[Any] = []

    def _collector(dso_ctrl: "DSOController") -> Optional[np.ndarray]:
        if dso_ctrl._H_cache is None or dso_ctrl._u_current is None:
            return None
        try:
            feats = get_dso2_features(dso_ctrl)
        except RuntimeError:
            return None
        feats["H"] = dso_ctrl._H_cache.copy()
        # Record the true Jacobian H at the current operating point as training target.
        if _net_ref:
            h_anal = dso_ctrl.compute_h_analytical(_net_ref[0])
            if h_anal is not None:
                feats["H_analytical"] = h_anal
        records.append(feats)
        return None  # keep the cached H unchanged; only observe

    def _pre_loop(state: dict) -> None:
        dso_ctrl = state["dso_controllers"].get("DSO_2")
        if dso_ctrl is None:
            return
        net = state.get("net")
        if net is not None:
            _net_ref.append(net)
        rng = np.random.default_rng(cfg.dso_pe_noise_seed)
        install_pe_noise(dso_ctrl, std_mvar=pe_noise_std_mvar, rng=rng)
        install_h_corrector(dso_ctrl, _collector)
        print(f"[collect] DSO_2 ready; PE std={pe_noise_std_mvar} Mvar, "
              f"duration={n_total_s / 3600:.1f} h")

    try:
        run_multi_tso_dso(cfg, pre_loop_hook=_pre_loop)
    except Exception as exc:
        if not records:
            raise RuntimeError(
                f"Simulation failed before any DSO steps were recorded: {exc}"
            ) from exc
        print(f"[collect] simulation ended early ({type(exc).__name__}: {exc}); "
              f"saving {len(records)} steps collected so far.")

    if not records:
        raise RuntimeError("No DSO steps recorded — did the simulation succeed?")

    # Only keep keys present in every record (H_analytical may be absent on warm-up).
    common_keys = set(records[0].keys())
    for r in records[1:]:
        common_keys &= set(r.keys())
    arrays = {k: np.stack([r[k] for r in records]) for k in common_keys}
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(output_path, **arrays)
    abs_out = os.path.abspath(output_path)
    print(f"[collect] {len(records)} steps -> {abs_out}")
    for k, v in arrays.items():
        print(f"[collect]   {k}: {v.shape}")
    return abs_out


# ── Kalman Q/R matrix estimation ─────────────────────────────────────────────

def generate_kalman_matrices(
    data_path: str = _TRAINING_DATA_PATH,
    output_path: str = _KALMAN_MATRICES_PATH,
    min_delta_u_norm: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Estimate Kalman process noise Q and measurement noise R from training data.

    Q -- diagonal covariance of ΔH(k) = H(k+1) - H(k) over all steps.
         Controls how fast the filter allows H to drift between steps.
    R -- empirical covariance of residuals Δy(k) - H(k-1) @ Δu(k).
         Captures linearisation error and true measurement noise.

    Steps with ‖Δu‖ < ``min_delta_u_norm`` are excluded from R estimation
    to avoid inflating residuals from near-zero actuator changes.

    Saves a .npz with keys ``"Q"`` (n_state × n_state) and ``"R"`` (n_y × n_y).
    Returns ``{"Q": ..., "R": ...}``.
    """
    d = np.load(data_path)
    U: np.ndarray = d["u"]            # (N, n_u)  OFO actuator vector
    Y: np.ndarray = d["y"]            # (N, n_y)  OFO output vector
    # Prefer H_analytical (true Jacobian) over the cached H when available.
    H: np.ndarray = d["H_analytical"] if "H_analytical" in d else d["H"]
    print(f"[kalman] using {'H_analytical' if 'H_analytical' in d else 'H (cached)'} "
          f"from training data")
    N, n_y, n_u = H.shape
    n_state = n_y * n_u

    # Process noise Q: diagonal variance of flattened H step-differences
    dH = np.diff(H, axis=0).reshape(N - 1, n_state)  # (N-1, n_state)
    Q  = np.diag(np.var(dH, axis=0))                  # (n_state, n_state)

    # Per-channel actuator-step scale: RMS of Δu per input channel.  Used to
    # normalise the runtime Σ_q = ‖Δu‖·Σ_q0 scaling so DER (Mvar) and OLTC (taps)
    # moves contribute comparably and s≈1 at a typical full step.
    dU = np.diff(U, axis=0)                            # (N-1, n_u)
    u_scale = np.sqrt(np.mean(dU ** 2, axis=0))        # (n_u,)
    u_scale = np.where(u_scale > 1e-9, u_scale, 1.0)   # guard zero-motion channels

    # Measurement noise R: covariance of Δy - H(k-1)@Δu(k) residuals
    residuals = []
    for k in range(1, N):
        du = U[k] - U[k - 1]
        if np.linalg.norm(du) < min_delta_u_norm:
            continue
        residuals.append((Y[k] - Y[k - 1]) - H[k - 1] @ du)

    if len(residuals) < 2:
        R = np.eye(n_y) * 1e-4
        print("[kalman] too few excited steps for R estimation — using I*1e-4")
    else:
        res = np.stack(residuals)                                   # (M, n_y)
        R   = np.cov(res.T) if n_y > 1 else np.atleast_2d(np.var(res))

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(output_path, Q=Q, R=R, u_scale=u_scale)
    q_d, r_d = np.diag(Q), np.diag(R)
    print(f"[kalman] Q {Q.shape}, R {R.shape}, u_scale {u_scale.shape} "
          f"-> {os.path.abspath(output_path)}")
    print(f"[kalman]   Q diag: [{q_d.min():.2e}, {q_d.max():.2e}]")
    print(f"[kalman]   R diag: [{r_d.min():.2e}, {r_d.max():.2e}]")
    print(f"[kalman]   u_scale: [{u_scale.min():.3f}, {u_scale.max():.3f}]")
    return {"Q": Q, "R": R, "u_scale": u_scale}


# ── ANN training ─────────────────────────────────────────────────────────────

def train_ann_model(
    data_path: str = _TRAINING_DATA_PATH,
    model_path: str = _ANN_MODEL_PATH,
    stats_path: str = _ANN_STATS_PATH,
    epochs: int = 200,
    batch_size: int = 32,
    validation_split: float = 0.15,
    hidden_units: tuple = (256, 128, 64),
) -> None:
    """Fit a dense ANN to predict H(k+1) from DSO_2 measurements at step k.

    Features : x = [y(k), u(k)]
    Targets  : t = H_analytical(k).flatten()

    Input/output pairs are (y(k), u(k)) → H_analytical(k) — the ANN
    estimates the true Jacobian at the current operating point directly
    from the I/O vectors.  ``H_analytical`` is used when present in the
    training file (generated by :func:`collect_training_data`); falls
    back to ``H`` (cached sensitivity) for compatibility with older data.

    Both x and t are z-score normalised; statistics are saved alongside
    the model in ``stats_path`` and applied by :class:`_ANNHPredictor`.

    Architecture : Dense(hidden[0], relu) → … → Dense(n_state, linear)
    Loss         : MSE on normalised targets, early stopping (patience=20)
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required: pip install tensorflow"
        ) from exc

    d = np.load(data_path)
    U: np.ndarray = d["u"]   # (N, n_u)
    Y: np.ndarray = d["y"]   # (N, n_y)
    # Prefer H_analytical (true Jacobian) as target when available.
    H_target: np.ndarray = (
        d["H_analytical"] if "H_analytical" in d else d["H"]
    )
    print(f"[ANN] training target: "
          f"{'H_analytical' if 'H_analytical' in d else 'H (cached)'}")
    N = len(U)
    if N < 1:
        raise ValueError(f"Need at least 1 training step, got {N}.")

    n_y, n_u = H_target.shape[1], H_target.shape[2]
    n_state = n_y * n_u

    # Samples: x[k] = [y(k), u(k)]  →  t[k] = H_analytical(k).flatten()
    X = np.concatenate([Y, U], axis=1).astype(np.float32)   # (N, n_y+n_u)
    T = H_target.reshape(N, n_state).astype(np.float32)     # (N, n_state)

    x_mean, x_std = X.mean(0), X.std(0)
    t_mean, t_std = T.mean(0), T.std(0)
    X_n = (X - x_mean) / (x_std + 1e-8)
    T_n = (T - t_mean) / (t_std + 1e-8)

    inputs = keras.Input(shape=(X_n.shape[1],))
    z = inputs
    for units in hidden_units:
        z = keras.layers.Dense(units, activation="relu")(z)
    outputs = keras.layers.Dense(n_state)(z)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    model.fit(
        X_n, T_n,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=20, restore_best_weights=True, verbose=1
            ),
        ],
        verbose=1,
    )

    out_dir = os.path.dirname(os.path.abspath(model_path))
    os.makedirs(out_dir, exist_ok=True)
    model.save(model_path)
    np.savez(stats_path, x_mean=x_mean, x_std=x_std, t_mean=t_mean, t_std=t_std)
    print(f"[ANN] model -> {os.path.abspath(model_path)}")
    print(f"[ANN] stats -> {os.path.abspath(stats_path)}")


class _NumericalHPredictor:
    """ORACLE predictor: returns the TRUE plant sensitivity at the current
    operating point, by closed-loop finite difference on the live combined net.

    This is the upper-bound baseline for the H-estimation comparison: the OFO
    acts with perfect knowledge of the realized secant at the OFO step scale
    (``delta_q_mvar`` = :data:`NUMERICAL_ORACLE_DELTA_MVAR`).  Recomputed every
    DSO step so it tracks operating-point drift.  Cost ~``2*(n_DER_bus +
    n_OLTC)`` plant power flows per step.

    ``combined_net`` is injected by :func:`_setup_h_predictor` (the same live
    net object the runner mutates each step).  ``compute_numerical_h_dso``
    deep-copies it internally, so the live converged state is never disturbed.
    """

    def __init__(self) -> None:
        self.combined_net: Optional[Any] = None
        self.delta_q_mvar: float = NUMERICAL_ORACLE_DELTA_MVAR

    def __call__(self, dso_ctrl: "DSOController") -> Optional[np.ndarray]:
        if self.combined_net is None or dso_ctrl._H_cache is None:
            return None
        try:
            H = compute_numerical_h_dso(
                self.combined_net, dso_ctrl,
                closed_loop=True, delta_q_mvar=self.delta_q_mvar,
            )
        except Exception as exc:  # noqa: BLE001 — oracle must never crash the run
            print(f"[numerical] H recompute failed: {type(exc).__name__}: {exc}")
            return None
        if H is None or H.shape != dso_ctrl._H_cache.shape:
            print(f"[numerical] shape mismatch / None "
                  f"(got {None if H is None else H.shape}, "
                  f"want {dso_ctrl._H_cache.shape}); keeping cache")
            return None
        return H


_numerical_h_predictor = _NumericalHPredictor()


def _unity_multiply_predictor(
    dso_ctrl: "DSOController",
) -> Optional[np.ndarray]:
    """Workflow-test predictor: returns ``H * 1.0`` (a fresh array).

    Functionally equivalent to ``_identity_predictor`` -- the H written
    back is identical to the H that came in -- but the array IS
    routed through the wrapper's full write path: type check, shape
    check, ``_H_cache`` reassignment, and use of the new object at step
    ``k+1``.  Use this to verify before plugging in a real KF / NN.

    Replace with the real predictor once the wiring is validated.
    """
    return dso_ctrl._H_cache * 1


# ---------------------------------------------------------------------------
#  Per-step observer (sidecar capture)
# ---------------------------------------------------------------------------

class _DSO2Observer:
    """Wraps a predictor and records H_used / H_predicted / H_analytical / y / u per step.

    ``h_used``       -- ``_H_cache`` just before the predictor fires; this is
                        the H matrix that was used in the MIQP at this step.
    ``h_predicted``  -- what the predictor returns (same as ``h_used`` when
                        the predictor returns ``None``, e.g. during warm-up).
    ``h_analytical`` -- H freshly recomputed from the current operating point
                        of the combined network (true Jacobian linearisation).
                        Requires *combined_net* to be passed at construction;
                        ``None`` entries are dropped at save time.
    ``y``            -- OFO output vector extracted from the measurement.
    ``u``            -- OFO actuator vector at this step.

    Call :meth:`save` after the simulation to dump all arrays to a ``.npz``
    sidecar file.
    """

    def __init__(
        self,
        predictor: HPredictor,
        combined_net: Optional[Any] = None,
    ) -> None:
        self._predictor    = predictor
        self._combined_net = combined_net
        self._h_used:       List[np.ndarray] = []
        self._h_predicted:  List[np.ndarray] = []
        self._h_analytical: List[np.ndarray] = []
        self._y:            List[np.ndarray] = []
        self._u:            List[np.ndarray] = []
        self._n_q_trafo:    int = 0   # set on first call from _H_mappings

    def __call__(self, dso_ctrl: "DSOController") -> Optional[np.ndarray]:
        # Capture H_used = the H the MIQP just used (before prediction).
        h_used = dso_ctrl._H_cache.copy() if dso_ctrl._H_cache is not None else None

        # Latch Q_trafo row count from mappings (constant after init).
        if self._n_q_trafo == 0 and dso_ctrl._H_mappings is not None:
            m = dso_ctrl._H_mappings
            self._n_q_trafo = len(m.get("trafos", [])) + len(m.get("trafo3w", []))

        # Run the actual predictor.
        h_predicted = self._predictor(dso_ctrl)

        # If the predictor returned None (warm-up / no-op), treat as identity.
        h_for_next = h_predicted if h_predicted is not None else (
            h_used.copy() if h_used is not None else None
        )

        # Compute the true analytical H at the current operating point.
        h_analytical: Optional[np.ndarray] = None
        if self._combined_net is not None and h_used is not None:
            h_analytical = dso_ctrl.compute_h_analytical(self._combined_net)

        # Capture y and u from the controller's last measurement.
        try:
            feats = get_dso2_features(dso_ctrl)
            y, u  = feats["y"], feats["u"]
        except RuntimeError:
            y = u = None

        if h_used is not None and y is not None:
            self._h_used.append(h_used)
            self._h_predicted.append(h_for_next if h_for_next is not None else h_used.copy())
            if h_analytical is not None:
                self._h_analytical.append(h_analytical)
            self._y.append(y)
            self._u.append(u)

        return h_predicted  # None keeps _H_cache unchanged (observer is transparent)

    def save(self, path: str) -> None:
        """Save recorded arrays to a compressed ``.npz`` file."""
        if not self._h_used:
            print("[observer] no steps recorded; sidecar not saved.")
            return
        out_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        arrays: Dict[str, np.ndarray] = dict(
            h_used      = np.stack(self._h_used),
            h_predicted = np.stack(self._h_predicted),
            y           = np.stack(self._y),
            u           = np.stack(self._u),
        )
        if self._h_analytical:
            if len(self._h_analytical) == len(self._h_used):
                arrays["h_analytical"] = np.stack(self._h_analytical)
            else:
                print(f"[observer] h_analytical has {len(self._h_analytical)} entries "
                      f"vs {len(self._h_used)} steps — skipping (shape mismatch).")
        if self._n_q_trafo > 0:
            arrays["n_q_trafo"] = np.array(self._n_q_trafo)
        np.savez_compressed(path, **arrays)
        keys_str = ", ".join(arrays.keys())
        print(f"[observer] {len(self._h_used)} steps [{keys_str}] -> {os.path.abspath(path)}")


def _make_row_masked_predictor(
    predictor: HPredictor,
    rows_mode: str,
) -> HPredictor:
    """Wrap *predictor* so only selected rows of H are written back to ``_H_cache``.

    rows_mode
    ---------
    ``"all"``       -- full H is written (pass-through, no mask applied).
    ``"q_trafo"``   -- only the Q_trafo rows (``H[:n_q_trafo, :]``) are taken
                       from the predictor's output; all remaining rows are kept
                       from the current ``_H_cache`` unchanged.  The number of
                       Q_trafo rows is read at call-time from
                       ``dso_ctrl._H_mappings`` so it adapts automatically to
                       the network topology.
    ``"q_trafo+v"`` -- the Q_trafo rows AND the following ``n_v`` voltage rows
                       (``H[:n_q_trafo + n_v, :]``, ``n_v = len(voltage_bus_indices)``)
                       are taken from the predictor; the remaining (I_line) rows
                       are kept from ``_H_cache``.

    The wrapped predictor always returns a full-shape array, so the shape
    check in :func:`install_h_corrector` continues to pass unchanged.
    """
    if rows_mode == "all":
        return predictor

    def _masked(dso_ctrl: "DSOController") -> Optional[np.ndarray]:
        H_next = predictor(dso_ctrl)
        if H_next is None:
            return None
        if dso_ctrl._H_cache is None:
            return H_next
        m = dso_ctrl._H_mappings or {}
        n_q_trafo = len(m.get("trafos", [])) + len(m.get("trafo3w", []))
        if n_q_trafo == 0:
            return H_next
        n_rows = n_q_trafo
        if rows_mode == "q_trafo+v":
            n_rows += len(dso_ctrl.config.voltage_bus_indices)   # add the voltage rows
        H_out = dso_ctrl._H_cache.copy()
        H_out[:n_rows] = H_next[:n_rows]
        return H_out

    return _masked


def _setup_h_predictor(cfg: MultiTSOConfig, state: Dict[str, Any]) -> Optional["_DSO2Observer"]:
    """Startup function: prime DSO_2's H, print view, install predictor (and PE noise).

    Returns the :class:`_DSO2Observer` so the caller (:func:`run`) can save
    the sidecar ``.npz`` after the simulation completes.

    Passed to the runner as ``pre_loop_hook`` (wrapped in a lambda that
    binds ``cfg``); runs once after Phase-2 init and before the main
    time loop.  See ``run()`` below for how it is wired in.

    If ``cfg.dso_pe_noise_enabled`` is True, gaussian persistent-
    excitation noise is installed on the DER actuator output
    (innermost wrapper, runs first); the H predictor wrapper is
    installed on top of it.  Wrapper call order at every DSO step
    therefore is:

        runner --> dso_ctrl.step (= H-corrector wrapper)
                     --> PE-noise wrapper
                       --> original step (OFO MIQP)
                       <-- ControllerOutput
                     <-- ControllerOutput, with noise added
                   <-- predictor called, _H_cache rewritten
                  --> result returned to runner --> applied to plant
    """
    dso_ctrl = state["dso_controllers"].get("DSO_2")
    if dso_ctrl is None:
        print("[H] DSO_2 not found in controllers; H predictor not installed.")
        return None

    view = get_dso_h_view(dso_ctrl)
    n_rows_full = len(view["kept_row_mask"])
    n_cols_full = len(view["kept_col_mask"])
    n_q = sum(k == "Q_trafo" for k in view["row_kinds"])
    n_v = sum(k == "V_bus"   for k in view["row_kinds"])
    n_der = sum(k == "Q_cor" for k in view["col_kinds"])
    n_oltc = sum(k == "OLTC" for k in view["col_kinds"])
    print(f"[H] DSO_2 view {view['H'].shape}  "
          f"(full was ({n_rows_full}, {n_cols_full}))")
    print(f"[H]   rows: {n_q} Q_trafo + {n_v} V_bus")
    print(f"[H]   cols: {n_der} Q_cor (DER) + {n_oltc} OLTC")

    # ── Init-H bias injection ─────────────────────────────────────────────
    # Apply a multiplicative perturbation to the init H so that all predictor
    # modes (identity / kalman / ann) start from exactly the same wrong H and
    # we can observe which one recovers faster.  Fixed seed guarantees that
    # every run with the same H_INIT_BIAS_SEED gets an identical bias matrix.
    if H_INIT_BIAS_STD > 0.0:
        combined_net = state.get("net")
        H_base = dso_ctrl.compute_h_analytical(combined_net) if combined_net is not None else None
        if H_base is None:
            H_base = dso_ctrl._H_cache
            print("[H] [warn] compute_h_analytical failed — falling back to split-network H_cache")
        rng_bias = np.random.default_rng(H_INIT_BIAS_SEED)
        noise = rng_bias.normal(1.0, H_INIT_BIAS_STD, size=H_base.shape)
        biased = H_base * noise
        if dso_ctrl._sensitivity_updater is not None:
            dso_ctrl._sensitivity_updater.override_base(biased)
        dso_ctrl._H_cache = biased
        print(f"[H] init bias applied: std={H_INIT_BIAS_STD:.0%}, "
              f"seed={H_INIT_BIAS_SEED} -> same bias for all predictor modes")

    # Install PE noise FIRST (innermost wrapper) so the H corrector
    # installed below sits on the outside.
    if getattr(cfg, "dso_pe_noise_enabled", False):
        rng = np.random.default_rng(cfg.dso_pe_noise_seed)
        install_pe_noise(
            dso_ctrl,
            std_mvar=cfg.dso_pe_noise_std_mvar,
            rng=rng,
        )
        print(f"[PE] installed gaussian noise std={cfg.dso_pe_noise_std_mvar} Mvar "
              f"on {len(dso_ctrl.config.der_indices)} DER Q_cor channels "
              f"(seed={cfg.dso_pe_noise_seed})")

    # Scale Kalman parameters to the actual DSO period.
    # All three defaults were tuned for 10 s steps; rescale proportionally so
    # wall-clock behaviour is invariant to the chosen period.
    _BASE_PERIOD_S = 10.0
    _period_ratio  = float(getattr(cfg, "dso_period_s", _BASE_PERIOD_S)) / _BASE_PERIOD_S
    _kalman_h_predictor.min_delta_u_norm   = 1.0                            # physical Mvar gate just above the ~0.9 noise floor (noise std 0.25): KF learns on real moves, skips noise-only steps so it freezes post-convergence
    _kalman_h_predictor.p_min_warmup_steps = 0                              # p_min floor DISABLED: the cold-start gain it held high was injecting noise into the over-actuated null space (validated: drift 0.42->0.14 frozen)
    _kalman_h_predictor.lam                = KALMAN_LAM                     # fixed (not period-scaled) for run-to-run comparability
    print(f"[kalman] period_ratio={_period_ratio:.2f}  "
          f"min_du={_kalman_h_predictor.min_delta_u_norm:.4f}  "
          f"warmup={_kalman_h_predictor.p_min_warmup_steps} steps  "
          f"lam={_kalman_h_predictor.lam:.5f} (fixed KALMAN_LAM)")

    # Inject the live combined net into the numerical oracle so it can
    # finite-difference the true plant sensitivity at each step.
    _numerical_h_predictor.combined_net = state.get("net")
    _numerical_h_predictor.delta_q_mvar = NUMERICAL_ORACLE_DELTA_MVAR

    _PREDICTORS = {
        "identity":  _unity_multiply_predictor,
        "kalman":    _kalman_h_predictor,
        "rls":       _rls_h_predictor,
        "ann":       _ANN_h_predictor,
        "numerical": _numerical_h_predictor,
    }
    if H_PREDICTOR_MODE not in _PREDICTORS:
        raise ValueError(
            f"Unknown H_PREDICTOR_MODE {H_PREDICTOR_MODE!r}. "
            f"Choose one of: {list(_PREDICTORS)}"
        )
    predictor = _PREDICTORS[H_PREDICTOR_MODE]
    if H_PREDICTOR_ROWS != "all":
        predictor = _make_row_masked_predictor(predictor, H_PREDICTOR_ROWS)
        _rows_desc = {"q_trafo": "Q_trafo rows",
                      "q_trafo+v": "Q_trafo + voltage rows"}.get(H_PREDICTOR_ROWS, H_PREDICTOR_ROWS)
        print(f"[H] row mask: {H_PREDICTOR_ROWS!r} — predictor updates {_rows_desc}; "
              f"remaining rows kept from cache")
    combined_net = state.get("net")
    observer  = _DSO2Observer(predictor, combined_net=combined_net)
    install_h_corrector(dso_ctrl, observer)
    print(f"[H] installed predictor: {H_PREDICTOR_MODE!r} ({type(predictor).__name__})"
          f" wrapped in _DSO2Observer")
    return observer


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------

def run() -> List[MultiTSOIterationRecord]:
    """Run the 003 experiment end-to-end and pickle the log."""
    cfg = make_config()
    out_dir = cfg.result_dir
    os.makedirs(out_dir, exist_ok=True)

    print()
    print("=" * 72)
    print("  RUNNING EXPERIMENT 003_CIGRE_2026 (single-DSO OFO on DSO_2)")
    print("=" * 72)
    print(f"  scenario={cfg.scenario}")
    print(f"  tso_mode={cfg.tso_mode}, tso_local_mode={cfg.tso_local_mode}")
    print(f"  dso_mode={cfg.dso_mode}, dso_ids_to_run={cfg.dso_ids_to_run}")
    print(f"  q_pcc_setpoints={cfg.q_pcc_setpoints_mvar_per_dso}")
    print(f"  n_total_s={cfg.n_total_s:.0f}, dso_period_s={cfg.dso_period_s:.0f}")
    print(f"  H_PREDICTOR_MODE={H_PREDICTOR_MODE!r}  H_PREDICTOR_ROWS={H_PREDICTOR_ROWS!r}")
    print(f"  DSO2_INTERFACE_MODE={DSO2_INTERFACE_MODE!r}  "
          f"(dso2_interface_slack={cfg.dso2_interface_slack})")
    if FROZEN_OP_POINT:
        print(f"  FROZEN_OP_POINT=True  frozen_at={FROZEN_OP_TIMESTAMP:%Y-%m-%d %H:%M}, contingencies=off")
    else:
        n_ev = len(cfg.contingencies)
        print(f"  FROZEN_OP_POINT=False  (profiles on, contingencies={n_ev})")
    if H_INIT_BIAS_STD > 0.0:
        print(f"  H_INIT_BIAS_STD={int(H_INIT_BIAS_STD*100)}%  H_INIT_BIAS_SEED={H_INIT_BIAS_SEED} "
              f"(same bias for all predictor modes)")
    print(f"  result_dir={out_dir}")

    _observer_ref: List[Optional[_DSO2Observer]] = [None]

    def _pre_loop(state: dict) -> None:
        if cfg.dso2_interface_slack:
            dso_ctrl = state["dso_controllers"].get("DSO_2")
            if dso_ctrl is None:
                print("[iface] DSO_2 not found; interface slack NOT installed.")
            else:
                ifaces = list(dso_ctrl.config.interface_trafo_indices)
                created = decouple_trafo3w_hv_with_slack(state["net"], ifaces)
                print(f"[iface] DSO_2 decoupled from TSO: replaced HV feed of "
                      f"3W trafos {ifaces} with pinned slacks (ext_grid {created}).")
        obs = _setup_h_predictor(cfg, state)
        _observer_ref[0] = obs

    try:
        log = run_multi_tso_dso(cfg, pre_loop_hook=_pre_loop)
    except Exception as exc:
        print(f"  [003] FAILED: {type(exc).__name__}: {exc}")
        log = []

    now = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    _suffix = "_" + H_PREDICTOR_MODE
    if FROZEN_OP_POINT:
        _suffix += "_frozen"
    if H_INIT_BIAS_STD > 0.0:
        _suffix += f"_biased{int(H_INIT_BIAS_STD * 100)}pct"
    pkl_path = os.path.join(out_dir, now + _suffix + ".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(log, f)
    print(f"  [003] wrote {len(log)} records -> {pkl_path}")

    sidecar_path: Optional[str] = None
    observer = _observer_ref[0]
    if observer is not None:
        sidecar_path = os.path.join(out_dir, now + "_dso2_ctrl.npz")
        observer.save(sidecar_path)

    # ── Auto-analysis: show the analysis plot for this run immediately ──
    _analysis = importlib.import_module("experiments.003_analysis")
    _analysis.plot_comparison(pkl_path, sidecar_path)

    return log


def main() -> None:
    run()


if __name__ == "__main__":
    main()
