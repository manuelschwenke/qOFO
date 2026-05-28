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
  DER side is the w-shift ``q_set`` (sgen sign convention) commanded
  to each DER alongside a reanchored ``qv_vref_anchor_pu``.  The
  plant-side QVLocalLoop then computes
  ``Q_DER = q_set - K · (V - V_anchor)`` (linear droop, deadband=0.01
  in 003).  The H matrix is the closed-loop ``∂y/∂q_set``, obtained
  by post-multiplying the open-loop ``∂y/∂Q_DER`` block by
  ``T' = (I + diag(K) · S_VQ)^{-1}`` (same matrix as the earlier
  q_set formulation, see :func:`controller.der_qv_local_loop.
  compute_w_shift_h_transform`).
* **DSO_2 outputs** -- Q at the HV side of the three coupling 3-winding
  transformers (``q_hv_mvar``) and DN voltage magnitudes.
* **DSO_2 actuators** -- ``q_set`` on every DSO_2 sgen plus the three
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
  to the DER (q_set) actuator output of the OFO at every step
  (persistent excitation for sensitivity learning).  Toggled by
  ``cfg.dso_pe_noise_enabled`` in :func:`make_base_config`.
* :func:`_setup_h_predictor`    -- example startup function (passed to
  the runner as ``pre_loop_hook``); primes the cache, prints the view
  once, optionally installs PE noise, and installs the predictor.

Vector layouts (DSO_2 in this script, currents and shunts excluded):

``y`` -- output vector (rows of H), in this order::

    y = [ Q_HV @ coupling_3W_trafo[t]   for t in cfg.interface_trafo_indices ]   # Mvar, generator convention
        [ V[bus]                         for bus in cfg.voltage_bus_indices  ]   # p.u.

``u`` -- actuator vector (cols of H), in this order::

    u = [ q_set_DER[d]                   for d in cfg.der_indices            ]   # Mvar, sgen convention (>0 = injection); commanded at the reanchored V_ref
        [ tap_position[t]                for t in cfg.oltc_trafo_indices     ]   # integer steps relative to neutral

The DER's actual reactive output is
``Q_DER = q_set - K · (V - V_anchor)``; ``q_set`` is what the OFO
writes and what the controller's H is linearized against.

``H = ∂y/∂u`` is the closed-loop linearization of the plant response
around the post-Phase-2 operating point.  Built from the inverse of
the full pandapower Jacobian projected onto DSO_2's actuator /
measurement sets, then post-multiplied on the DER columns by
``T' = (I + diag(K) · S_VQ)^{-1}`` to map the network-level
``∂y/∂Q_DER`` to ``∂y/∂q_set`` (same matrix as the earlier q_set
formulation).  Cross-coupling from the rest of the grid is implicit
in the inverted full Jacobian.  Layout assumes the w-shift actuator
and no grid-forming DER (both hold in 003).

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
from experiments.runners import run_multi_tso_dso  # noqa: E402

# ``002_M_TSO_M_DSO_COMPARE`` starts with a digit, so the import must go
# through importlib rather than a normal ``from ... import ...``.
_compare = importlib.import_module("experiments.002_M_TSO_M_DSO_COMPARE")
make_002_base_config = _compare.make_base_config


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


def make_base_config() -> MultiTSOConfig:
    """Configuration for the 003 single-DSO experiment.

    Inherits the validated wind_replace + weight tuning + contingency
    timeline from ``002_M_TSO_M_DSO_COMPARE.make_base_config()`` and
    overrides only the flags that switch the run into the
    "TSO local Q(V), DSO_2 OFO with q_set" mode.
    """
    cfg = make_002_base_config()

    # ---- Network / scenario --------------------------------------------
    cfg.scenario = "wind_replace"

    # ---- TSO layer: no OFO; every TSO DER runs a plant-side Q(V) loop --
    cfg.tso_mode = "local"
    cfg.tso_local_mode = "qv"

    # ---- refactor_Qcor_method q_mode plant model ----------------------
    cfg.tso_q_mode = "qv"
    cfg.dso_q_mode = "qv"
    cfg.tso_qv_vref_pu = 1.03
    cfg.dso_qv_vref_pu = 1.03
    cfg.tso_qv_slope_pu = 0.06
    cfg.dso_qv_slope_pu = 0.06
    # No deadband (linear droop through V_ref).
    cfg.tso_qv_deadband_pu = 0.01
    cfg.dso_qv_deadband_pu = 0.01

    cfg.g_q = 250
    cfg.dso_g_v = 20000
    cfg.g_w_dso_der = 1000

    # ---- DSO layer: only DSO_2 has an OFO controller -------------------
    cfg.dso_mode = "ofo"
    cfg.dso_ids_to_run = ["DSO_2"]

    # ---- Exogenous Q-setpoint vector at DSO_2's three 3W trafos --------
    # Order matches ``meta.hv_networks[1].coupling_trafo_indices``.
    cfg.q_pcc_setpoints_mvar_per_dso = {"DSO_2": [0.0, -20.0, 20.0]}

    # ---- Run length / timing -------------------------------------------
    cfg.n_total_s = 60.0 * 60 * 2     # 2 h smoke
    cfg.tso_period_s = 180.0          # cosmetic: TSO never steps anyway
    cfg.dso_period_s = 20.0
    cfg.warmup_s = 0.0
    start_time = datetime(2016, 9, 7, 8, 0)
    use_profiles = True

    # ---- Plant-side Q(V) loop damping ----------------------------------
    # 44 QVLocalLoops (4 TSO STATCOMs + 40 DSO sgens) iterate in parallel
    # inside every ``pp.runpp(run_control=True)``.  STATCOM-class units
    # (S_n ~600 Mvar with slope 0.07) have ~9 GVar/pu open-loop gain;
    # the published-stable damping is 0.1 (see QVLocalLoop docstring).
    # The 002 default of 0.5 is fine for fewer loops but oscillates here.
    cfg.qv_local_damping = 0.1
    cfg.qv_local_max_step_frac = None
    cfg.qv_local_tol_mvar = 0.1

    # ---- Live plots off for diagnostic runs ---------------------------
    cfg.live_plot_controller = True
    cfg.live_plot_cascade = True    # This is the live plotter for the DSO
    cfg.live_plot_system = False

    # ---- Persistent excitation (sensitivity learning) -----------------
    # When enabled, gaussian noise N(0, sigma^2) is added to the DSO
    # OFO's continuous (DER q_set) actuator output AT EVERY DSO step,
    # AFTER the MIQP solves and BEFORE the command is sent to the plant.
    # The controller's internal u_current is also incremented by the
    # noise so the next OFO step starts from u(k+1)' = u(k+1) + eps and
    # does not "fight" the excitation.  OLTC tap commands are NOT
    # perturbed (integer actuators).  Useful for downstream H estimation
    # (Kalman filter / NN) where the input must be persistently exciting.
    cfg.dso_pe_noise_enabled = False
    cfg.dso_pe_noise_std_mvar = 1.0   # std of N(0, sigma^2) on q_set [Mvar]
    cfg.dso_pe_noise_seed = 42        # RNG seed for reproducibility

    # ---- Output directory ---------------------------------------------
    cfg.result_dir = os.path.join("results", "003_cigre_2026")

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
#       # Optional: gaussian persistent-excitation noise on q_set output.
#       # Toggle in make_base_config (cfg.dso_pe_noise_enabled = True/False);
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
          ``col_kinds``      : list[str], one of ``{"q_set","OLTC","shunt"}``.
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

    # ── Column metadata: order is [q_set (per-DER) | OLTC_2W | OLTC_3W | shunt] ──
    # The DER columns of H are the closed-loop ∂y/∂q_set (T' applied).
    # q_set is the OFO command at the reanchored V_ref; the DER's
    # actual Q is q_set - K·(V - V_anchor).
    col_kinds: List[str] = []
    col_labels: List[str] = []
    col_units: List[str] = []
    for d in cfg.der_indices:
        col_kinds.append("q_set")
        col_labels.append(f"q_set[sgen_{d}]")
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
            f"Layout assumes the w-shift actuator and no grid-forming DER; "
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
        dso_ctrl._H_cache = H_next
        return result

    dso_ctrl.step = step_with_corrector  # type: ignore[method-assign]


def install_pe_noise(
    dso_ctrl: "DSOController",
    *,
    std_mvar: float,
    rng: np.random.Generator,
) -> None:
    """Wrap ``dso_ctrl.step`` so gaussian noise is added to DER q_set commands.

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
        Standard deviation of the gaussian noise on each DER's q_set
        command, in Mvar.  Set to 0 to effectively disable.
    rng : np.random.Generator
        Random source.  Use a seeded ``np.random.default_rng(seed)`` for
        reproducibility.

    Notes
    -----
    Layout assumption: ``u_new = [q_set (n_der) | V_gf (n_gf) |
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


def _johannes_kalman_predictor(dso_ctrl: "DSOController") -> Optional[np.ndarray]:
    """Skip-write stub: returns ``None`` so ``_H_cache`` is left untouched.

    Use this as the gate during a warm-up window or whenever you do NOT
    want a write at the current step.  The wrapper short-circuits on
    ``None`` and never assigns to ``_H_cache``.
    """
    return None


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
    return dso_ctrl._H_cache * 0.9


def _setup_h_predictor(cfg: MultiTSOConfig, state: Dict[str, Any]) -> None:
    """Startup function: prime DSO_2's H, print view, install predictor (and PE noise).

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
    n_der = sum(k == "q_set" for k in view["col_kinds"])
    n_oltc = sum(k == "OLTC" for k in view["col_kinds"])
    print(f"[H] DSO_2 view {view['H'].shape}  "
          f"(full was ({n_rows_full}, {n_cols_full}))")
    print(f"[H]   rows: {n_q} Q_trafo + {n_v} V_bus")
    print(f"[H]   cols: {n_der} q_set (DER) + {n_oltc} OLTC")

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
              f"on {len(dso_ctrl.config.der_indices)} DER q_set channels "
              f"(seed={cfg.dso_pe_noise_seed})")

    install_h_corrector(dso_ctrl, _unity_multiply_predictor) # Insert your predictor here
    print("[H] installed _unity_multiply_predictor "
          "(returns H * 1.0; replace with KF / NN)")
    return None  # continue main loop


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------

def run() -> List[MultiTSOIterationRecord]:
    """Run the 003 experiment end-to-end and pickle the log."""
    cfg = make_base_config()
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
    print(f"  result_dir={out_dir}")

    try:
        log = run_multi_tso_dso(
            cfg,
            pre_loop_hook=lambda state: _setup_h_predictor(cfg, state),
        )
    except Exception as exc:
        print(f"  [003] FAILED: {type(exc).__name__}: {exc}")
        log = []

    pkl_path = os.path.join(out_dir, "log.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(log, f)
    print(f"  [003] wrote {len(log)} records -> {pkl_path}")
    return log


def main() -> None:
    run()


if __name__ == "__main__":
    main()
