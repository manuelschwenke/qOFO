#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/runners/multi_tso_dso.py
=====================================
Multi-TSO / multi-DSO OFO simulation entry point.

This module hosts :func:`run_multi_tso_dso`, the main loop that builds the
IEEE 39-bus network, instantiates the multi-zone TSO and DSO OFO
controllers, and steps the cascaded outer/inner loop in time.  It was
extracted from ``experiments/000_M_TSO_M_DSO.py`` so that experiment
scripts (000, 002, 003) can share a single canonical implementation.

Architecture (matches the multi-TSO theory in Schwenke / CIGRE 2026)
---------------------------------------------------------------------

    ┌──────────────────────────────────────────────────────────┐
    │              IEEE 39-bus network (plant)                 │
    │  Zone 1        │  Zone 2 (w/ DSOs) │  Zone 3             │
    │  TSOCtrl_1     │  TSOCtrl_2        │  TSOCtrl_3          │
    │  (4 gen incl.  │  ├── DSOCtrl_2_0  │  (4 gen)            │
    │   slack)       │  └── DSOCtrl_2_1  │                     │
    └──────────────────────────────────────────────────────────┘

Step sequence (each simulation step dt_s):
    1.  Apply time-series profiles to plant network.
    2.  Run power flow on plant network.
    3.  If TSO step: call coordinator.step(measurements_per_zone).
        * Each TSOController.step() solves its local MIQP independently.
        * Coordinator optionally recomputes H_ij and checks contraction.
    4.  If DSO step: call DSOController.step() for each HV sub-network DSO.
    5.  Apply all new setpoints to plant network.
    6.  Run power flow, record results.

Sensitivity matrices
--------------------
* H_ii (local, zone i): computed by TSOController._build_sensitivity_matrix()
  using generator terminal buses + DER sgens as column inputs and zone buses
  as row outputs.
* H_ij (cross-zone, i≠j): computed by MultiTSOCoordinator.compute_cross_sensitivities()
  using zone_j's inputs and zone_i's observed outputs.
* H_DSO (per HV sub-network): computed by DSOController._build_sensitivity_matrix()
  using DSO DER Q + 3 coupling OLTC as inputs, interface Q + HV voltages + line
  currents as outputs.

Author: Manuel Schwenke / Claude Code
"""

from __future__ import annotations

import copy
import os
import sys
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import dataclasses

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged
from numpy.typing import NDArray

# ── Ensure project root is on sys.path when imported as a package module ─────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.reachability import ReachabilityMonitor, ReachabilityViolation
from controller.base_controller import OFOParameters
from controller.central_controller import CentralControllerConfig, CentralOFOController
from controller.dso_controller import DSOController, DSOControllerConfig
from controller.multi_tso_coordinator import MultiTSOCoordinator, ZoneDefinition
from controller.tso_controller import TSOController, TSOControllerConfig
from controller.shunt_integrator import ShuntBankConfig, ShuntIntegrator
from core.actuator_bounds import (
    ActuatorBounds,
    GeneratorParameters,
    generator_parameters_for_type,
    set_der_q_capability_override,
    set_der_qv_deadband_by_sgen,
    set_der_qv_slope_by_sgen,
    set_der_qv_deadband_override,
)
from core.measurement import (
    Measurement,
    measure_zone_tso,
    measure_zone_dso,
    measure_central,
)
from core.measurement_noise import MeasurementNoiseModel
from core.reporting import load_and_apply_tuned_params
from core.profiles import (
    DEFAULT_PROFILES_CSV,
    apply_profiles,
    load_profiles,
    snapshot_base_values,
)
from network.ieee39.dso_overrides import apply_dso_overrides
from network.ieee39.load_model import apply_zip_load_model
from network.ieee39.zonal_balancing import apply_gen_dispatch, compute_zonal_gen_dispatch
from network.ieee39 import (
    add_hv_networks,
    build_ieee39_net,
    HVNetworkInfo,
    tag_der_q_modes,
)
from network.zone_partition import (
    fixed_zone_partition_ieee39,
    spectral_zone_partition,
    relabel_zones_by_generator_count,
    get_zone_lines,
    get_tie_lines,
    get_zone_tie_lines,
)
from configs.config import MultiTSOConfig
from experiments.helpers import (
    MultiTSOIterationRecord,
    _apply_contingency,
    _network_state,
    prepare_load_contingencies,
)
from core.plant import (
    ActuatorWrites,
    PandapowerStaticPlant,
    Plant,
    shunt_steps_for_buses,
    writes_from_central,
    writes_from_dso,
    writes_from_zone_tso,
)
from sensitivity.jacobian import JacobianSensitivities
from sensitivity.network_reduction import (
    build_dso_local_net,
    build_tso_local_net,
)
from sensitivity.numerical_h import (
    compute_numerical_h_dso,
    compute_numerical_h_tso,
)
from core.message import SetpointMessage, ShuntDisturbanceMessage
from controller.der_qv_local_loop import (
    CosPhiConstLoop,
    QVLocalLoop,
    clear_seed_lu_cache,
    install_der_q_loops,
    seed_qv_equilibrium,
)
from controller.gw_precondition import curvature_spectrum, precondition_g_w

# Helpers live in a sibling module; import at the top so callers (e.g.
# ``tuning.ceilings``) can monkey-patch them on this module's namespace.
from experiments.runners._multi_tso_helpers import (
    _OLTCRateLimiter,
    _clamp_oltc_taps,
    _collect_contingency_watch_buses,
    _dump_contingency_diagnostics,
    _record_dso_group_and_transformer_data,
    _record_dso_measurement_snapshot,
    _record_tso_measurement_snapshot,
    _record_hv_group_observables,
    _record_local_dso_trafo_data,
    _record_zone_live_plot_observables,
    _run_delayed_stability_analysis,
    _snapshot_oltc_taps,
)


# =============================================================================
#  V5 (central) closed-loop curvature probe
# =============================================================================

def _dump_central_curvature(central_controller, central_cfg) -> None:
    """Print the closed-loop curvature spectrum of the central V5 controller.

    One OFO tick (unconstrained, slack/usage dropped) is
    ``sigma* = -G_w^{-1} H_V^T diag(g_v) (V - V*)``, so the per-tick
    voltage-error map is ``e_{k+1} = (I - M) e_k`` with
    ``M = H_V G_w^{-1} H_V^T diag(g_v)``.  OFO is stable iff
    ``eig(M) ⊂ (0, 2)`` and well-damped for ``lambda_max(M) ≲ 1``.

    Eigenvalues are computed from the symmetric similarity form
    ``D_v^{1/2} H_V G_w^{-1} H_V^T D_v^{1/2}`` (same spectrum as M, but
    guaranteed real / PSD).  Uses exactly the expanded H and the
    per-variable g_w vector the MIQP itself sees, so the reported
    ``lambda_max`` is the one that governs the actual closed loop.

    Read-only: touches only the cached H, g_v and g_w on the controller.
    The suggested ``kappa`` is the global factor to multiply the whole
    ``g_w`` block by so that ``lambda_max(M) -> 1`` (since ``M ∝ G_w^{-1}``,
    scaling g_w by kappa scales lambda_max by 1/kappa).
    """
    H = central_controller._expand_H_to_der_level(
        central_controller._build_sensitivity_matrix()
    )
    n_v = len(central_cfg.voltage_bus_indices)
    H_v = np.asarray(H[:n_v, :], dtype=np.float64)

    g_w_vec, _ = central_controller._get_per_variable_weights()
    if g_w_vec is None:
        g_w_vec = np.broadcast_to(
            np.asarray(central_controller.params.g_w, dtype=np.float64),
            (H_v.shape[1],),
        )
    g_w_vec = np.asarray(g_w_vec, dtype=np.float64)
    g_v_vec = np.asarray(central_controller.g_v_per_bus, dtype=np.float64)

    # Shared implementation with the Tier-2 preconditioner so the probe and
    # the auto-kappa rule read exactly the same spectrum.
    spec = curvature_spectrum(H_v, g_v_vec, g_w_vec)
    kappa = spec.lambda_max  # to reach lambda_max = 1

    print("  [V5-curvature] M = H_V G_w^-1 H_V^T diag(g_v)  "
          f"(n_v={n_v}, n_u={H_v.shape[1]})")
    print(f"  [V5-curvature]   lambda_max(M)  = {spec.lambda_max:.4g}   "
          f"(OFO stable iff < 2; well-damped if <~ 1)")
    print(f"  [V5-curvature]   lambda_min+(M) = {spec.lambda_min_pos:.4g}   "
          f"cond+ = {spec.cond:.4g}")
    print(f"  [V5-curvature]   suggested kappa (x whole g_w block) "
          f"to reach lambda_max=1 : {kappa:.4g}")


def _apply_gw_preconditioning(
    controller, config, label: str, verbose: int,
):
    """Tier-2 curvature-based ``g_w`` preconditioning for one controller.

    Reads the controller's cached voltage sensitivities and replaces the
    *continuous* actuator classes' ``g_w`` with a column-norm-preconditioned,
    auto-``kappa`` vector that targets ``config.precondition_lambda_target``
    (see :mod:`controller.gw_precondition`).  Read-only w.r.t. the plant.
    Integer classes (OLTC/shunt) are detected via the controller's integer
    index set and left at their config/BO values.

    Returns the :class:`PreconditionResult` or ``None`` when skipped (no
    voltage curvature, no actuator classes, or no continuous class).
    """
    # Full weighted objective block, not the voltage rows alone.  For a
    # DSO the voltage block carries under 1 % of the objective (interface-Q
    # dominates ~500x in priority terms), so preconditioning against it either
    # mis-scales g_w or -- as was the case -- declines to act at all.  Falls
    # back to voltage_curvature_inputs() for controllers that do not override.
    vci = controller.objective_curvature_inputs()
    if vci is None:
        if verbose >= 1:
            print(f"  [precond:{label}] skipped (no voltage curvature)")
        return None
    H_v, g_v_vec = vci

    class_map = controller._actuator_class_indices()
    if not class_map:
        if verbose >= 1:
            print(f"  [precond:{label}] skipped (no actuator classes)")
        return None

    g_w_cur, _ = controller._get_per_variable_weights()
    if g_w_cur is None:
        g_w_cur = np.broadcast_to(
            np.asarray(controller.params.g_w, dtype=np.float64),
            (H_v.shape[1],),
        )
    g_w_cur = np.asarray(g_w_cur, dtype=np.float64)

    # Continuous classes = those whose indices are not integer variables,
    # minus the operator's explicit exclusions (default: 'gen' AVR setpoint).
    int_set = {int(i) for i in controller._integer_indices}
    excluded = set(getattr(config, "precondition_exclude_classes", ()) or ())
    cont_classes = [
        c for c, idx in class_map.items()
        if c not in excluded
        and not ({int(i) for i in np.asarray(idx).tolist()} & int_set)
    ]
    if not cont_classes:
        if verbose >= 1:
            print(f"  [precond:{label}] skipped (no continuous classes)")
        return None

    # Per-layer lambda target, falling back to the shared one.  The layers sit
    # in different regimes (TSO zone 2 continuous-lambda 1.775 vs zone 1 0.021),
    # so a single shared target cannot be meaningful for both.
    _is_dso = label.startswith("DSO")
    _lam_override = (
        config.precondition_lambda_target_dso if _is_dso
        else config.precondition_lambda_target_tso
    )
    lambda_target = float(
        _lam_override if _lam_override is not None
        else config.precondition_lambda_target
    )

    res = precondition_g_w(
        H_v=H_v,
        g_v=g_v_vec,
        g_w_current=g_w_cur,
        class_index_map=class_map,
        preconditionable_classes=cont_classes,
        lambda_target=lambda_target,
        granularity=str(config.precondition_granularity),
        floor_frac=float(config.precondition_floor_frac),
        mode=str(getattr(config, "precondition_mode", "cap")),
        class_scale_overrides=(
            getattr(config, "precondition_class_scales", None) or None
        ),
        lambda_scope=str(getattr(config, "precondition_lambda_scope", "all")),
    )
    controller.apply_preconditioned_g_w(res.g_w_new)

    if verbose >= 1:
        tag = {
            "reduced":           "REDUCED",
            "raised":            "RAISED (mode='set')",
            "within_margin":     "within margin (no-op)",
            "integer_dominated": "INTEGER-DOMINATED (no-op; OLTC binds - tune cadence)",
            "no_class":          "no continuous class (no-op)",
        }.get(res.status, res.status)
        print(
            f"  [precond:{label}] {tag}: lambda_max {res.lambda_max_before:.3g} "
            f"-> {res.lambda_max_after:.3g} (target "
            f"{lambda_target:g}, floor={res.lambda_floor:.3g}), "
            f"kappa={res.kappa:.3g}"
        )
        if res.applied:
            scales = "  ".join(
                f"{c}={v:.3g}" for c, v in res.class_scales.items()
            )
            print(
                f"  [precond:{label}]   g_w[{config.precondition_granularity}]: "
                f"{scales}  (cond+ {res.spectrum_before.cond:.3g} -> "
                f"{res.spectrum_after.cond:.3g})"
            )
    return res


# =============================================================================
#  Main simulation function
# =============================================================================

def run_multi_tso_dso(
    config: MultiTSOConfig,
    pre_loop_hook: Optional[Callable[[Dict[str, Any]], Any]] = None,
    plant_factory: Optional[Callable[..., Plant]] = None,
) -> List[MultiTSOIterationRecord]:
    """
    Execute the multi-TSO / multi-DSO OFO simulation.

    Parameters
    ----------
    config : MultiTSOConfig
        All simulation parameters.  See dataclass docstring for details.
    pre_loop_hook : optional
        If provided, called once with the post-Phase-2 state dict
        (``net``, ``meta``, ``tso_controllers``, ``dso_controllers``,
        ``shared_jac``, ``dso_to_tso_id``, ``zone_defs``,
        ``coordinator``) right before the main time loop. If the hook
        returns a truthy value, the runner returns immediately with an
        empty log -- used by diagnostic scripts that only need the
        post-init state.
    plant_factory : optional
        ``(net, *, meta, zone_map) -> Plant`` constructor for the plant
        the main loop acts on (RMS build plan, Phase 6); ``meta`` /
        ``zone_map`` are passed as keywords so a co-simulation factory
        can snapshot-sync its external model.  Default ``None`` = the
        quasi-static
        :class:`core.plant.PandapowerStaticPlant` on the runner-built
        ``net`` -- bit-for-bit the legacy behaviour
        (tests/runner_refactor_regression.py).  A non-static plant (e.g.
        ``pf.plant.PowerFactoryPlant``) must keep ``net`` as its mirror /
        measurement image; features that still act on the net directly
        (contingencies, OLTC tap-rate limiter, time-series profiles) are
        rejected up front for such plants.

    Returns
    -------
    log : List[MultiTSOIterationRecord]
        One record per simulation step.
    """
    v_set = config.v_setpoint_pu
    verbose = config.verbose

    # DER capability override, applied before anything reads a capability:
    # the controller's ActuatorBounds and the plant's Q(V) clip both consult
    # the module-level value, so it must be set once, up front, for the two
    # to stay in lockstep.  Always announced -- a run under an artificial
    # capability must never be mistaken for a physical one.
    set_der_q_capability_override(config.der_q_capability_override_pu)
    if config.der_q_capability_override_pu is not None:
        print(f"  [!] DER capability OVERRIDDEN to "
              f"+-{config.der_q_capability_override_pu:g} pu of S_n "
              f"(P-independent, ignores op_diagram). "
              f"TEMPORARY DIAGNOSTIC -- not a physical model.")
    # DER Q(V) deadband override -- read by the RMS QVPRE anchor pass; the
    # static side is driven through tso_/dso_qv_deadband_pu into net.sgen by
    # tag_der_q_modes below.  Both must share one value or the plants settle in
    # different droop basins at the deadband edge (2026-07-24).
    # Clear any map left behind by a previous run in this process; step [4]
    # republishes it once net.sgen has been tagged.
    set_der_qv_deadband_by_sgen(None)
    set_der_qv_slope_by_sgen(None)
    set_der_qv_deadband_override(config.der_qv_deadband_override_pu)
    if config.der_qv_deadband_override_pu is not None:
        print(f"  [!] DER Q(V) deadband OVERRIDDEN to "
              f"{config.der_qv_deadband_override_pu:g} pu on both plants "
              f"(removes the deadband-edge multi-equilibrium).")

    measurement_noise_model = MeasurementNoiseModel(config.measurement_noise)

    def _feedback_measurement(
        measurement: Measurement,
        *,
        sample_id,
        initialisation: bool = False,
    ) -> Measurement:
        """Apply configured sensor noise to one controller-facing packet."""
        return measurement_noise_model.apply(
            measurement,
            net,
            sample_id=sample_id,
            initialisation=initialisation,
        )

    def _zone_scalar(zone_dict: Optional[Dict[int, float]], zone_id: int,
                      fallback: float) -> float:
        """Per-zone override lookup mirroring the ``zone_v_setpoints_pu``
        idiom used throughout this function: ``zone_dict[zone_id]`` if
        present, else ``fallback``."""
        return (float(zone_dict.get(zone_id, fallback))
                if zone_dict is not None else float(fallback))

    # Single centralized controller path (CIGRE V5 best-case reference).
    # When True, one CentralOFOController owns every actuator and observes
    # every measurement; the 3-zone partition + per-HV metadata are retained
    # only as a recording lens, and the distributed per-zone TSO / per-DSO
    # controllers are constructed for that recording but never stepped.
    _central = (config.control_scope == "central")
    # Firing period of the single central controller.  Default (None) ->
    # every step (dt_s): V5 replaces the fast STS-OFO layer, so it must fire
    # as often to be a true best-case upper bound.
    _central_period_s = (
        config.central_period_s if config.central_period_s is not None
        else config.dt_s
    )

    if verbose >= 1:
        print("=" * 72)
        zone_method = "fixed 3-area" if config.use_fixed_zones else "spectral"
        print("  MULTI-TSO / MULTI-DSO OFO -- IEEE 39-bus New England")
        print(f"  V_set = {v_set:.3f} p.u.  |  N_zones = 3")
        print(f"  Zone partition: {zone_method}  |  4 HV sub-networks (DSO_1..DSO_4)")
        if config.measurement_noise.enabled:
            _eq = config.measurement_noise.equivalent_bounds()
            _comp = config.measurement_noise.profile_components()
            _persistent = 100.0 * (
                1.0 - config.measurement_noise.sample_noise_fraction
            )
            print(
                "  Measurement chain: "
                f"{config.measurement_noise.profile} "
                f"(equiv. rectangular: "
                f"V_EHV={100 * _eq['voltage_ehv']:.2f}%, "
                f"V_HV={100 * _eq['voltage_hv']:.2f}%, "
                f"I={100 * _eq['current']:.2f}%, "
                f"P_EHV={100 * _eq['active_power_ehv']:.2f}%; "
                f"phase={_comp['power_phase_angle_deg']:.3f} deg; "
                f"{_persistent:.0f}% persistent; "
                f"seed={config.measurement_noise.seed})"
            )
        print("=" * 72)

    # =========================================================================
    # STEP 1: Build the base IEEE 39-bus network (no DSO feeders yet)
    # =========================================================================
    if verbose >= 1:
        print("[1] Building IEEE 39-bus network ...")

    net, meta = build_ieee39_net(
        ext_grid_vm_pu=1.03,
        scenario=config.scenario,
        verbose=(verbose >= 1),
    )

    # =========================================================================
    # STEP 2: Zone partitioning
    # =========================================================================
    if config.use_fixed_zones:
        if verbose >= 1:
            print()
            print("[2] Fixed 3-area zone partition (literature) ...")
        zone_map, bus_zone = fixed_zone_partition_ieee39(
            net, verbose=(verbose >= 2)
        )
    else:
        if verbose >= 1:
            print("[2] Spectral zone partitioning (N=3) ...")
        zone_map, bus_zone = spectral_zone_partition(
            net, n_zones=3, verbose=(verbose >= 2)
        )

    # CONSOLE OUTPUT ##########################################################
    if verbose >= 1:
        _gen_grid_set = set(meta.gen_grid_bus_indices or meta.gen_bus_indices)
        for z in sorted(zone_map.keys()):
            n_gen_z = sum(1 for b in zone_map[z] if b in _gen_grid_set)
            n_load_z = sum(
                1 for li in net.load.index if int(net.load.at[li, "bus"]) in zone_map[z]
            )
            print(f"  Zone {z}: {len(zone_map[z])} buses, "
                  f"{n_gen_z} generators, {n_load_z} loads")
    ###########################################################################

    # =========================================================================
    # STEP 3: Attach 3 HV sub-networks (110 kV, TUDA topology)
    # =========================================================================
    if verbose >= 1:
        print()
        print("[3] Attaching 3 HV sub-networks (DSO_1..DSO_3) ...")

    # Resolve the effective switched-shunt dispatch mode.  ``shunt_dispatch`` is
    # the master switch; for backward compatibility a legacy config that only
    # sets ``install_tso_tertiary_shunts=True`` (leaving shunt_dispatch at its
    # 'off' default) is treated as the legacy MIQP-dispatched bipolar bank.
    config.validate_integrator_mode()
    _shunt_mode = config.shunt_dispatch
    if _shunt_mode == "off" and config.install_tso_tertiary_shunts:
        _shunt_mode = "miqp"

    if _shunt_mode == "integrator":
        meta = add_hv_networks(
            net, meta,
            install_tso_tertiary_shunts=True,
            tso_shunt_kind="msc_msr",
            msc_n_levels=config.tso_shunt_msc_n_levels,
            msr_n_levels=config.tso_shunt_msr_n_levels,
            msc_q_step_mvar=config.tso_shunt_msc_q_step_mvar,
            msr_q_step_mvar=config.tso_shunt_msr_q_step_mvar,
            verbose=(verbose >= 2),
        )
    else:
        meta = add_hv_networks(
            net, meta,
            install_tso_tertiary_shunts=config.install_tso_tertiary_shunts,
            tso_tertiary_shunt_q_mvar=config.tso_tertiary_shunt_q_mvar,
            verbose=(verbose >= 2),
        )

    # ── Per-DSO scenario multipliers ─────────────────────────────────────
    # Applied here, before ANY power flow / load model / droop tagging /
    # operating-point init, because they rewrite p_mw, base_p_mw, sn_mva and
    # the reactive-load base that all of those read.  Scenario multipliers,
    # not builder state: constants.py keeps the symmetric networks and this
    # scales one underlay on top.  The RMS snapshot is taken downstream of
    # this, so pf_sync carries the scaled ratings into PowerFactory.
    if (config.dso_der_scale or config.dso_load_p_scale
            or config.dso_load_q_profile_base_mvar
            or config.dso_line_std_type):
        apply_dso_overrides(
            net, meta.hv_networks,
            dso_der_scale=config.dso_der_scale or None,
            dso_load_p_scale=config.dso_load_p_scale or None,
            dso_load_q_profile_base_mvar=(
                config.dso_load_q_profile_base_mvar or None),
            dso_line_std_type=config.dso_line_std_type or None,
        )
        if verbose >= 1:
            _applied = net["dso_overrides"]
            print(f"  [dso-override] applied {_applied} -- scenario "
                  f"multiplier on top of scenario={config.scenario!r}; "
                  f"results are NOT comparable with an unscaled run")

    # add_hv_networks() may remove buses (e.g. bus 11/0-idx = IEEE bus 12).
    # Purge any removed buses from zone_map so downstream logic stays consistent.
    existing_buses = set(net.bus.index)
    for z in zone_map:
        zone_map[z] = [b for b in zone_map[z] if b in existing_buses]

    # =========================================================================
    # STEP 3b: Plant load model (constant-PQ vs anchored ZIP)
    # =========================================================================
    # Decision 2026-07-17 (RMS co-simulation): default "zip" gives every load
    # P ~ V^1 / Q ~ V^2 anchored at load_zip_anchor_vm_pu, matching the
    # PowerFactory load model exactly.  Contingency stress loads (created in
    # step [9]) deliberately stay constant-PQ.
    if config.load_model == "zip":
        if verbose >= 1:
            print()
            print("[3b] Applying anchored ZIP load model ...")
        apply_zip_load_model(
            net,
            anchor_vm_pu=config.load_zip_anchor_vm_pu,
            verbose=(verbose >= 1),
        )
    elif config.load_model != "const_pq":
        raise ValueError(
            f"config.load_model must be 'zip' or 'const_pq', "
            f"got {config.load_model!r}"
        )

    # =========================================================================
    # STEP 4: tag every DER with its q_mode (Soleimani §III-B Q_cor path)
    # =========================================================================
    # All DERs stay as pp.sgen.  Tag every TSO and DSO sgen with its
    # q_mode and parameters from the runner-level MultiTSOConfig
    # hierarchy.  The plant-side controllers (QVLocalLoop /
    # CosPhiConstLoop) read these columns each PF iteration.
    # ToDo: Cleanup
    if verbose >= 1:
        print()
        print("[4] Tagging DER q_modes ...")
    meta = tag_der_q_modes(
        net, meta,
        tso_q_mode=config.tso_q_mode,
        dso_q_mode=config.dso_q_mode,
        q_mode_overrides=config.der_q_mode_overrides,
        tso_qv_slope_pu=config.tso_qv_slope_pu,
        dso_qv_slope_pu=config.dso_qv_slope_pu,
        qv_slope_pu_overrides=config.der_qv_slope_pu_overrides,
        tso_qv_vref_pu=config.tso_qv_vref_pu,
        dso_qv_vref_pu=config.dso_qv_vref_pu,
        qv_vref_pu_overrides=config.der_qv_vref_pu_overrides,
        tso_qv_deadband_pu=config.tso_qv_deadband_pu,
        dso_qv_deadband_pu=config.dso_qv_deadband_pu,
        qv_deadband_pu_overrides=config.der_qv_deadband_pu_overrides,
        tso_cosphi=config.tso_cosphi,
        dso_cosphi=config.dso_cosphi,
        cosphi_overrides=config.der_cosphi_overrides,
        tso_cosphi_sign=config.tso_cosphi_sign,
        dso_cosphi_sign=config.dso_cosphi_sign,
        cosphi_sign_overrides=config.der_cosphi_sign_overrides,
        verbose=(verbose >= 1),
    )

    # Publish the tagged per-park deadbands for the RMS plant.  Its QVPRE
    # anchor pass otherwise reads the exported snapshot, which carries the
    # MultiTSOConfig() DEFAULT deadband rather than this run's -- so without
    # this the two plants disagree whenever tso_/dso_qv_deadband_pu is not the
    # default, and delta_TS != delta_DS cannot be represented at all.
    if "qv_deadband_pu" in net.sgen.columns:
        _db_by_sgen = {
            int(s): float(net.sgen.at[s, "qv_deadband_pu"])
            for s in net.sgen.index
            if not pd.isna(net.sgen.at[s, "qv_deadband_pu"])
        }
        set_der_qv_deadband_by_sgen(_db_by_sgen)
        if verbose >= 1:
            _ts = sorted({round(float(net.sgen.at[s, "qv_deadband_pu"]), 6)
                          for s in meta.tso_der_indices})
            _ds = sorted({round(float(net.sgen.at[s, "qv_deadband_pu"]), 6)
                          for s in meta.dso_der_indices})
            print(f"[4a] Q(V) deadband published to both plants: "
                  f"TS={_ts} DS={_ds} pu ({len(_db_by_sgen)} parks)")

    # Same for the DROOP.  Without this --der-slope moves only the static
    # plant: the RMS anchor pass reads qv_slope_pu from the exported snapshot,
    # which carries MultiTSOConfig() defaults, so a droop sweep would have
    # compared a static plant at the swept value against an RMS plant pinned
    # at 0.06.
    if "qv_slope_pu" in net.sgen.columns:
        _sl_by_sgen = {
            int(s): float(net.sgen.at[s, "qv_slope_pu"])
            for s in net.sgen.index
            if not pd.isna(net.sgen.at[s, "qv_slope_pu"])
        }
        set_der_qv_slope_by_sgen(_sl_by_sgen)
        if verbose >= 1:
            _ts = sorted({round(float(net.sgen.at[s, "qv_slope_pu"]), 6)
                          for s in meta.tso_der_indices})
            _ds = sorted({round(float(net.sgen.at[s, "qv_slope_pu"]), 6)
                          for s in meta.dso_der_indices})
            print(f"[4a] Q(V) droop published to both plants: "
                  f"TS={_ts} DS={_ds} pu ({len(_sl_by_sgen)} parks)")

    # NB: install_der_q_loops is deferred to step [10.3], AFTER the
    # Phase 1/2 OLTC init.  If we install now, the plant-side QVLocalLoops
    # (one per TSO + DSO DER) would run during Phase 1 alongside the
    # temp PV gens used to seed STATCOM Q, and the inner-loop dynamics
    # destabilise the init.
    if verbose >= 1:
        print(
            f"[4b] Deferred plant-side loop install for "
            f"{len(meta.tso_der_indices)} TSO + "
            f"{len(meta.dso_der_indices)} DSO DERs until after "
            f"step [10] OLTC init."
        )

    # =========================================================================
    # STEP 5: Build ZoneDefinitions and TSOControllerConfigs
    # =========================================================================
    if verbose >= 1:
        print()
        print("[5] Building zone definitions and controller configs ...")

    # ── Partition generator indices per zone ──────────────────────────────────
    # Use gen_grid_bus_indices (original 345 kV bus) for zone assignment,
    # but store gen_bus_indices (terminal bus, possibly 10.5 kV) for
    # sensitivity computation.
    zone_gen_indices: Dict[int, List[int]] = {z: [] for z in zone_map}
    zone_gen_buses:   Dict[int, List[int]] = {z: [] for z in zone_map}
    gen_grid_buses = meta.gen_grid_bus_indices if meta.gen_grid_bus_indices else meta.gen_bus_indices
    for g, g_bus, g_grid_bus in zip(meta.gen_indices, meta.gen_bus_indices, gen_grid_buses):
        for z, buses in zone_map.items():
            if g_grid_bus in set(buses):
                zone_gen_indices[z].append(g)
                zone_gen_buses[z].append(g_bus)  # terminal bus for sensitivity
                break

    # ── Generators the OFO must not dispatch ──────────────────────────────────
    # The slack machine is the network equivalent for the rest of the
    # interconnection (IEEE 39-bus: G 01, 10 GVA at bus 40).  A TSO does not
    # set a neighbouring system's voltage, and the PowerFactory RMS model has
    # no AVR block for it, so dispatching it made the static plant strictly
    # more capable than the RMS one.  Removed on both sides (2026-07-21);
    # the machine stays observed -- only its V-ref actuator is withdrawn.
    if config.dispatch_slack_gen_v_ref:
        _non_dispatchable_gens: Set[int] = set()
    else:
        _non_dispatchable_gens = {
            int(g) for g in net.gen.index
            if bool(net.gen.get("slack", pd.Series(dtype=bool)).get(g, False))
        }
    if _non_dispatchable_gens:
        print(f"  [actuators] AVR V-ref withheld from slack/equivalent "
              f"gen(s): {sorted(_non_dispatchable_gens)}")

    # ── Partition TSO DER indices per zone ────────────────────────────────────
    zone_der_indices: Dict[int, List[int]] = {z: [] for z in zone_map}
    zone_der_buses:   Dict[int, List[int]] = {z: [] for z in zone_map}
    for s_idx, s_bus in zip(meta.tso_der_indices, meta.tso_der_buses):
        for z, buses in zone_map.items():
            if s_bus in set(buses):
                zone_der_indices[z].append(s_idx)
                zone_der_buses[z].append(s_bus)
                break

    # ── Propagate the per-zone voltage schedule to the autonomous V layer ──────
    # zone_v_setpoints_pu sets each zone's OFO *tracking reference*.  For that
    # reference to be physically realisable (not fought by the plant), the
    # autonomous voltage layer must be anchored at the same schedule:
    #   (1) synchronous-machine AVR setpoints (gen.vm_pu) — g_w_gen (~5e9) freezes
    #       these at the uniform build value (build.py sets all to ext_grid_vm_pu),
    #       so the OFO can never reach a divergent schedule; they must START there.
    #   (2) TSO-DER Q(V) droop nominal centre (qv_vref_pu) — the loops re-anchor on
    #       measured V at runtime (plant_io), so they follow once the AVRs hold the
    #       zone; the nominal only fixes the cold-start / equilibrium seed.
    # Gated on an explicit schedule; uniform/None leaves the build values intact.
    if config.zone_v_setpoints_pu is not None:
        _sched = config.zone_v_setpoints_pu
        for z, gs in zone_gen_indices.items():
            if z in _sched:
                for g in gs:
                    net.gen.at[g, "vm_pu"] = float(_sched[z])
        for eg in net.ext_grid.index:
            eb = int(net.ext_grid.at[eg, "bus"])
            ez = next((zz for zz, bs in zone_map.items() if eb in set(bs)), None)
            if ez in _sched:
                net.ext_grid.at[eg, "vm_pu"] = float(_sched[ez])
        if "qv_vref_pu" in net.sgen.columns:
            for z, ss in zone_der_indices.items():
                if z in _sched:
                    for s in ss:
                        net.sgen.at[s, "qv_vref_pu"] = float(_sched[z])

    # ── Partition machine-transformer OLTCs per zone ───────────────────────────
    # Exclude the slack gen's machine trafo from the controllable OLTC set.
    # Its LV bus is the PYPOWER angle reference, so
    # :func:`sensitivity.jacobian.JacobianSensitivities.compute_dV_ds_2w`
    # cannot produce a sensitivity column for it (the reference bus is not
    # in the Jacobian).  The slack gen's ``vm_pu`` setpoint already gives
    # the TSO a direct voltage control at that terminal, so losing the
    # redundant OLTC degree of freedom is acceptable.
    slack_gen_term_buses: Set[int] = set()
    if "slack" in net.gen.columns:
        for g in net.gen.index[net.gen["slack"].astype(bool)]:
            slack_gen_term_buses.add(int(net.gen.at[g, "bus"]))

    zone_oltc_trafos: Dict[int, List[int]] = {z: [] for z in zone_map}
    for t_idx, g_idx in zip(meta.machine_trafo_indices, meta.machine_trafo_gen_map):
        lv_bus = int(net.trafo.at[t_idx, "lv_bus"])
        if lv_bus in slack_gen_term_buses:
            continue  # slack-gen OLTC excluded (see comment above)
        # Machine trafo's grid bus = hv_bus of the 2W transformer
        grid_bus = int(net.trafo.at[t_idx, "hv_bus"])
        for z, buses in zone_map.items():
            if grid_bus in set(buses):
                zone_oltc_trafos[z].append(t_idx)
                break

    # ── Build gen→trafo map for contingency handling ────────────────────────
    # Maps net.gen index → net.trafo index of the associated machine trafo.
    gen_trafo_map: Dict[int, int] = {
        g_idx: t_idx
        for t_idx, g_idx in zip(meta.machine_trafo_indices,
                                meta.machine_trafo_gen_map)
        if g_idx >= 0  # skip non-machine OLTCs (marked -1)
    }

    # ── Group HV sub-networks by zone ────────────────────────────────────────
    # HVNetworkInfo carries its home zone from the 3-area constants.
    def _hv_zone(hv) -> int:
        return int(hv.zone)

    zone_hv_networks: Dict[int, List[HVNetworkInfo]] = {z: [] for z in zone_map}
    for hv in meta.hv_networks:
        zone_hv_networks[_hv_zone(hv)].append(hv)

    # All DSO IDs (one per HV sub-network)
    dso_ids: List[str] = [hv.net_id for hv in meta.hv_networks]

    # Per-zone PCC trafos and DSO IDs (parallel lists)
    zone_pcc_trafos: Dict[int, List[int]] = {z: [] for z in zone_map}
    zone_pcc_dso_ids: Dict[int, List[str]] = {z: [] for z in zone_map}
    for hv in meta.hv_networks:
        for trafo_idx in hv.coupling_trafo_indices:
            zone_pcc_trafos[_hv_zone(hv)].append(trafo_idx)
            zone_pcc_dso_ids[_hv_zone(hv)].append(hv.net_id)

    # Save original TN-only zone map for TSO monitoring (before HV extension).
    # The TSO monitors TN-level voltages and line currents only; HV elements
    # are the DSO's domain.
    tn_zone_map: Dict[int, List[int]] = {
        z: [b for b in buses if b in net.bus.index]
        for z, buses in zone_map.items()
    }

    def _extend_zone_map_for_dispatch(zmap, hv_zone_of) -> None:
        """Extend a zone map in place with HV sub-network buses and
        machine-transformer LV terminal buses (dispatch / ownership)."""
        for hv in meta.hv_networks:
            _z_hv = hv_zone_of(hv)
            zmap[_z_hv] = sorted(set(zmap[_z_hv]) | set(hv.bus_indices))
        aux_lengths = {
            len(meta.internal_aux_bus_indices),
            len(meta.internal_aux_parent_buses),
            len(meta.internal_aux_line_indices),
        }
        if len(aux_lengths) != 1:
            raise ValueError(
                "Internal auxiliary metadata lists have different lengths"
            )
        for aux_bus, parent_bus in zip(
            meta.internal_aux_bus_indices, meta.internal_aux_parent_buses,
        ):
            owners = [
                z for z, buses in zmap.items() if parent_bus in set(buses)
            ]
            if len(owners) != 1:
                raise ValueError(
                    f"Auxiliary bus {aux_bus} parent {parent_bus} belongs to "
                    f"{len(owners)} zones; expected exactly one"
                )
            zmap[owners[0]] = sorted(
                set(zmap[owners[0]]) | {int(aux_bus)}
            )
        for tidx, gidx in zip(meta.machine_trafo_indices,
                              meta.machine_trafo_gen_map):
            if gidx < 0:
                continue
            lv_bus = int(net.trafo.at[tidx, "lv_bus"])
            hv_bus = int(net.trafo.at[tidx, "hv_bus"])
            for z, buses in zmap.items():
                if hv_bus in set(buses):
                    if lv_bus not in set(buses):
                        zmap[z] = sorted(set(zmap[z]) | {lv_bus})
                    break

    # Extend zone bus indices with HV sub-network buses and machine LV
    # terminal buses (for dispatch / ownership).
    _extend_zone_map_for_dispatch(zone_map, _hv_zone)

    # The dispatch partition is identical to the control partition.
    dispatch_zone_map = zone_map

    # HV-network lookup for DSO controller init
    hv_info_map: Dict[str, HVNetworkInfo] = {hv.net_id: hv for hv in meta.hv_networks}

    # Map each TSO-owned tertiary shunt bus to its parent DSO id.  Used at
    # run-time to dispatch ``ShuntDisturbanceMessage`` to the affected DSO
    # whenever the TSO MIQP switches a shunt.  The shunt sits at the
    # tertiary of the first coupling 3-winding transformer (see add_hv_networks).
    shunt_bus_to_dso_id: Dict[int, str] = {}
    for hv in meta.hv_networks:
        if hv.coupling_lv_bus_indices:
            shunt_bus_to_dso_id[int(hv.coupling_lv_bus_indices[0])] = hv.net_id

    # ── Build ZoneDefinition for each zone ────────────────────────────────────
    # TSO monitoring uses TN-only buses and lines (tn_zone_map).
    # Line filtering: TSOController's sensitivity builder (build_sensitivity_matrix_H)
    # only computes ∂I/∂u for lines where BOTH endpoints are PQ buses.  Lines
    # touching a PV generator bus are excluded from the I-rows of H_physical.
    # To avoid a shape mismatch we pre-filter zone lines to PQ-bus endpoints only.
    pv_and_slack_buses_run = (
        set(int(net.gen.at[g, "bus"]) for g in net.gen.index) |
        set(int(net.ext_grid.at[e, "bus"]) for e in net.ext_grid.index)
    )

    # ── Partition TSO-owned tertiary shunts per zone ──────────────────────────
    # Each shunt is owned by the TSO zone hosting the parent DSO sub-network
    # (see meta.tso_tertiary_shunt_zones, populated by add_hv_networks).  The
    # DSO controllers are blind to these shunts; the bus indices flow into
    # ZoneDefinition.shunt_bus_indices and from there into TSOControllerConfig.
    # Only the legacy MIQP path puts shunt buses into the TSO control vector.
    # In 'integrator' mode the banks are driven by the separate ShuntIntegrator
    # (built below) and must NOT appear as MIQP integers, so the per-zone shunt
    # lists are left empty (→ TSOControllerConfig.shunt_bus_indices == []).
    zone_shunt_buses:  Dict[int, List[int]]   = {z: [] for z in zone_map}
    zone_shunt_qsteps: Dict[int, List[float]] = {z: [] for z in zone_map}
    if _shunt_mode == "miqp":
        for sb, q_step, sz in zip(
            meta.tso_tertiary_shunt_buses,
            meta.tso_tertiary_shunt_q_steps_mvar,
            meta.tso_tertiary_shunt_zones,
        ):
            if sz in zone_shunt_buses:
                zone_shunt_buses[sz].append(int(sb))
                zone_shunt_qsteps[sz].append(float(q_step))

    zone_defs: Dict[int, ZoneDefinition] = {}
    for z in sorted(zone_map.keys()):
        # TSO monitors TN-level elements only
        tn_bus_set = set(tn_zone_map[z])
        all_z_lines = get_zone_lines(net, tn_bus_set)
        # Keep only lines with both endpoints at PQ buses (Jacobian builder requirement)
        z_lines = [
            li for li in all_z_lines
            if int(net.line.at[li, "from_bus"]) not in pv_and_slack_buses_run
            and int(net.line.at[li, "to_bus"]) not in pv_and_slack_buses_run
        ]
        z_line_max_i_ka = [
            float(net.line.at[li, "max_i_ka"]) for li in z_lines
        ]

        # Voltage observation buses: only TN PQ buses (not PV/slack, not HV).
        v_bus_indices_z = [
            b for b in tn_zone_map[z] if b not in pv_and_slack_buses_run
        ]

        zone_defs[z] = ZoneDefinition(
            zone_id=z,
            bus_indices=zone_map[z],
            gen_indices=zone_gen_indices[z],
            gen_bus_indices=zone_gen_buses[z],
            tso_der_indices=zone_der_indices[z],
            tso_der_buses=zone_der_buses[z],
            v_bus_indices=v_bus_indices_z,  # PQ buses only (V-observable)
            line_indices=z_lines,
            line_max_i_ka=z_line_max_i_ka,
            pcc_trafo_indices=zone_pcc_trafos[z],
            pcc_dso_ids=zone_pcc_dso_ids[z],
            oltc_trafo_indices=zone_oltc_trafos[z],
            shunt_bus_indices=zone_shunt_buses[z],
            shunt_q_steps_mvar=zone_shunt_qsteps[z],
            v_setpoint_pu=(
                float(config.zone_v_setpoints_pu.get(z, v_set))
                if config.zone_v_setpoints_pu is not None else v_set
            ),
            # alpha removed (absorbed into g_w)
            g_v=_zone_scalar(config.zone_g_v, z, config.g_v),
            g_w_der=config.g_w_der,
            g_w_gen=config.g_w_gen,
            g_w_pcc=config.g_w_pcc,
            g_w_oltc=config.g_w_tso_oltc,
            g_w_shunt=config.g_w_tso_shunt,
            g_q_tso=config.tso_g_q_pcc,
        )

    # Populate per-zone tie-line sets (Phase A: monitoring only).
    # A tie line is one whose two endpoints sit in two different TSO zones.
    # For each zone we record the tie lines touching its bus set together
    # with the IN-ZONE endpoint bus (sign anchor for Q_tie measurement
    # and sensitivity).  Both zones touching the same line own it, each
    # at its own end — symmetric decentralised monitoring.
    _tn_zone_buses_set = {z: set(tn_zone_map[z]) for z in zone_defs}
    for z, zd in zone_defs.items():
        other_lists = [
            _tn_zone_buses_set[zj] for zj in zone_defs if zj != z
        ]
        pairs = get_zone_tie_lines(net, _tn_zone_buses_set[z], other_lists)
        zd.tie_line_indices = [li for li, _ in pairs]
        zd.tie_line_endpoint_buses = [endp for _, endp in pairs]
    if verbose >= 1:
        for z, zd in zone_defs.items():
            hv_names = [hv.net_id for hv in zone_hv_networks.get(z, [])]
            print(f"  Zone {z}: {len(zd.gen_indices)} gen, {len(zd.tso_der_indices)} DER, "
                  f"{len(zd.oltc_trafo_indices)} OLTC, "
                  f"{len(zd.shunt_bus_indices)} shunt, "
                  f"{len(zd.line_indices)} lines, {len(zd.pcc_trafo_indices)} PCC trafos, "
                  f"{len(zd.tie_line_indices)} tie lines  "
                  f"DSOs: {hv_names}")

    # ── Live-plot statics (tie-line map, gen P/Q limits) ─────────────────────
    # The inter-zone tie-line map feeds the TSO-CONTROLLER tie-line Q tile.
    # Generator P/Q limits feed the SYSTEM-POWER-FLOW generator tiles.
    tie_line_map: Dict[Tuple[int, int], List[int]] = {}
    zone_ids_sorted = sorted(zone_defs.keys())
    for i, z_i in enumerate(zone_ids_sorted):
        for z_j in zone_ids_sorted[i + 1:]:
            ties = get_tie_lines(
                net, set(tn_zone_map[z_i]), set(tn_zone_map[z_j]),
            )
            if ties:
                tie_line_map[(z_i, z_j)] = list(ties)

    gen_limits_static: Dict[int, Dict[str, float]] = {} # ToDo: what are these limits? we want to use op-diagram!
    for g_idx in net.gen.index:
        limits: Dict[str, float] = {}
        for key in ("min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"):
            limits[key] = (
                float(net.gen.at[g_idx, key])
                if key in net.gen.columns else float("nan")
            )
        gen_limits_static[int(g_idx)] = limits

    # =========================================================================
    # STEP 6: Initialise TSOControllers (one per zone)
    # =========================================================================
    if verbose >= 1:
        print()
        print("[6] Initialising TSOControllers ...")

    ns0 = _network_state(net)  # initial network state snapshot

    # Build one full-network Jacobian at the current (pre-profile) operating
    # point and share it across every TSO and DSO controller, plus the
    # coordinator.  This snapshot is replaced by a fresh post-Phase-2 one
    # below (see "Rebuild shared Jacobian"), so all controllers eventually
    # operate on the same post-init cached plant model.  Avoids 8 redundant
    # deep-copy + pp.runpp + dense-inversion calls inside the construction
    # loops.
    _t_jac_initial = perf_counter()
    try:
        shared_jac = JacobianSensitivities(net)
    except LoadflowNotConverged:
        # The un-profiled build state is only a bootstrap: this object is
        # REPLACED by the post-Phase-2 rebuild below before any control is
        # computed, and every controller's H cache is invalidated with it.
        # Letting a divergence here abort the whole run therefore kills a
        # scenario for the sake of a value that is thrown away.
        #
        # Reachable whenever the nameplate (un-profiled) operating point is
        # heavier than the profiled one the run actually starts from -- e.g.
        # a per-DSO DER multiplier: DSO_3 x2 diverges here at 5698 MW of sgen
        # while the 2016-01-05 08:00 profiled point (2325 MW) converges under
        # both slack conventions (measured 2026-07-30).
        #
        # Fall back to the profiled operating point on a COPY, so ``net``
        # keeps its documented pre-profile state for steps [6]-[8] and every
        # run that converges today is bit-for-bit unaffected.
        if not config.use_profiles:
            raise
        if verbose >= 1:
            print("  [6] bootstrap Jacobian diverged at the un-profiled build "
                  "point; retrying at the profiled start instant (this object "
                  "is replaced post-Phase-2 and never used for control)")
        _probe_net = copy.deepcopy(net)
        _probe_profiles = load_profiles(
            config.profiles_csv or DEFAULT_PROFILES_CSV,
            timestep_s=config.dt_s,
        )
        snapshot_base_values(_probe_net)
        apply_profiles(_probe_net, _probe_profiles, config.start_time)
        shared_jac = JacobianSensitivities(_probe_net)
        del _probe_net, _probe_profiles
    if verbose >= 1:
        print(f"  [T] initial shared JacobianSensitivities: {perf_counter() - _t_jac_initial:.2f} s")

    _t_step5 = perf_counter()
    tso_controllers: Dict[int, TSOController] = {}
    for z, zd in zone_defs.items():

        # ── G_w diagonal for this zone's u vector ────────────────────────────
        gw_diag = zd.gw_diagonal()
        # Output-slack weights in g_z (output-vector) order.
        _g_z_voltage_z = _zone_scalar(config.zone_g_z_voltage, z, config.g_z_voltage)
        gz_diag_target = np.concatenate([
            np.full(len(zd.v_bus_indices),     _g_z_voltage_z),      # V slacks
            np.full(len(zd.pcc_trafo_indices), config.g_z_q_pcc),    # Q_PCC slacks
            np.full(len(zd.line_indices),      config.g_z_current),  # current slacks
            np.full(len(zd.gen_indices),       config.g_z_q_gen),    # Q_gen slacks
        ])
        # During warmup use a tiny g_z; after warmup switch to gz_diag_target
        if config.g_z_warmup_s > 0:
            gz_diag = np.where(gz_diag_target > 0, config.g_z_warmup_value, 0.0)
        else:
            gz_diag = gz_diag_target

        ofo_params = OFOParameters(
            g_w=gw_diag,
            g_z=gz_diag,
            g_u=np.zeros_like(gw_diag),
            alpha=1.0,  # Q_cor mode does not use command relaxation
            int_max_step=config.int_max_step,
            int_cooldown=config.int_cooldown,
            int_cooldown_s=config.oltc_cooldown_s,
            adapt_g_w_classes=config.tso_adapt_g_w_classes(),
            g_w_adapt_meta=config.make_g_w_adapt_meta(),
        )

        # Build gen→OLTC position mapping for capability-based OLTC blocking.
        # gen_trafo_map: net.gen index → net.trafo index (machine trafo).
        # We need: position in gen_indices → position in oltc_trafo_indices.
        _oltc_pos = {t: k for k, t in enumerate(zd.oltc_trafo_indices)}
        _gen_oltc_map: Dict[int, int] = {}
        for gen_pos, g_idx in enumerate(zd.gen_indices):
            t_idx = gen_trafo_map.get(g_idx)
            if t_idx is not None and t_idx in _oltc_pos:
                _gen_oltc_map[gen_pos] = _oltc_pos[t_idx]

        # TSOControllerConfig: pass zone-specific index sets
        tso_cfg = TSOControllerConfig(
            der_indices=zd.tso_der_indices,
            pcc_trafo_indices=zd.pcc_trafo_indices,
            pcc_dso_controller_ids=zd.pcc_dso_ids,
            oltc_trafo_indices=zd.oltc_trafo_indices,
            shunt_bus_indices=zd.shunt_bus_indices,
            shunt_q_steps_mvar=zd.shunt_q_steps_mvar,
            voltage_bus_indices=zd.v_bus_indices,
            current_line_indices=zd.line_indices,
            current_line_max_i_ka=zd.line_max_i_ka if zd.line_indices else None,
            v_min_pu=_zone_scalar(config.zone_v_min_pu, z, config.v_min_pu),
            v_max_pu=_zone_scalar(config.zone_v_max_pu, z, config.v_max_pu),
            v_setpoints_pu=np.full(len(zd.v_bus_indices), zd.v_setpoint_pu),
            g_v=zd.g_v,
            gen_indices=zd.gen_indices,
            gen_bus_indices=zd.gen_bus_indices,
            non_dispatchable_gen_indices=_non_dispatchable_gens,
            gen_oltc_map=_gen_oltc_map,
            enable_saturation_mode=config.enable_avr_saturation_mode,
            g_q_tso=config.tso_g_q_pcc,
            pcc_capability_on_output=config.tso_pcc_capability_on_output,
            g_res_sg=_zone_scalar(config.zone_tso_g_res_sg, z, config.tso_g_res_sg),
            g_res_der=_zone_scalar(config.zone_tso_g_res_der, z, config.tso_g_res_der),
            qv_slope_pu=config.tso_qv_slope_pu,
            g_loss=_zone_scalar(config.zone_tso_g_loss, z, config.tso_g_loss),
        )

        if verbose >= 1:
            print(f"  [zone {z}] g_v={tso_cfg.g_v:g}  "
                  f"v_min/max_pu=[{tso_cfg.v_min_pu:.3f},{tso_cfg.v_max_pu:.3f}]  "
                  f"g_res_sg={tso_cfg.g_res_sg:g}  g_res_der={tso_cfg.g_res_der:g}  "
                  f"g_loss={tso_cfg.g_loss:g}  g_z_voltage={_g_z_voltage_z:g}")

        # ActuatorBounds for DERs in this zone
        if zd.tso_der_indices:
            s_rated = np.array(
                [float(net.sgen.at[s, "sn_mva"]) for s in zd.tso_der_indices],
                dtype=np.float64,
            )
            p_max = np.array(
                [float(net.sgen.at[s, "p_mw"]) for s in zd.tso_der_indices],
                dtype=np.float64,
            )
        else:
            s_rated = np.array([], dtype=np.float64)
            p_max   = np.array([], dtype=np.float64)

        # Generator capability parameters for this zone.
        # Nameplate is set unconditionally in build_ieee39_net (see
        # network/ieee39/constants.NAMEPLATE_FACTOR).
        gen_params = []
        for g in zd.gen_indices:
            sn       = float(net.gen.at[g, "sn_mva"])
            p_max_mw = float(net.gen.at[g, "max_p_mw"])
            # Machine type comes from GEN_NAMEPLATE via network.ieee39.build
            # (net.gen["type"]): "Hydro" -> salient pole, everything else
            # (Nuclear / Coal / the aggregated "Equivalent" anchor) -> round
            # rotor.  Missing column falls back to round rotor.
            gen_type = (
                net.gen.at[g, "type"] if "type" in net.gen.columns else ""
            )
            gen_params.append(
                generator_parameters_for_type(gen_type, sn, p_max_mw)
            )

        # Read per-DER operating diagram type (STATCOM vs VDE-AR-N-4120-v2)
        der_op_diagrams = []
        for s in zd.tso_der_indices:
            od = net.sgen.at[s, "op_diagram"] if "op_diagram" in net.sgen.columns else None
            der_op_diagrams.append(str(od) if od and str(od) != "nan" else "VDE-AR-N-4120-v2")

        bounds = ActuatorBounds(
            der_indices=np.array(zd.tso_der_indices, dtype=np.int64),
            der_s_rated_mva=s_rated,
            der_p_max_mw=p_max,
            oltc_indices=np.array(zd.oltc_trafo_indices, dtype=np.int64),
            oltc_tap_min=np.array(
                [int(net.trafo.at[t, "tap_min"]) for t in zd.oltc_trafo_indices],
                dtype=np.int64,
            ),
            oltc_tap_max=np.array(
                [int(net.trafo.at[t, "tap_max"]) for t in zd.oltc_trafo_indices],
                dtype=np.int64,
            ),
            shunt_indices=np.array(zd.shunt_bus_indices, dtype=np.int64),
            shunt_q_mvar=np.array(zd.shunt_q_steps_mvar, dtype=np.float64),
            gen_params=gen_params,
            der_op_diagrams=der_op_diagrams,
        )

        ctrl = TSOController(
            controller_id=f"tso_zone_{z}",
            params=ofo_params,
            config=tso_cfg,
            network_state=ns0,
            actuator_bounds=bounds,
            sensitivities=shared_jac,
        )
        # _u_current is initialised later (step 7e), after profiles and
        # OLTC/STATCOM init have settled the operating point.
        tso_controllers[z] = ctrl

    if verbose >= 1:
        print(f"  [T] step [5] TSO controller construction: {perf_counter() - _t_step5:.2f} s")

    # Package names are accepted as aliases; internal mode strings remain
    # stable for existing configs and pickles.
    if config.coordination_mode == "sbx_h":
        config.coordination_mode = "sbx"
    elif config.coordination_mode == "sbx_v":
        config.coordination_mode = "sbxv"
    if config.coordination_mode not in ("none", "sbx", "sbxv"):
        raise ValueError(
            f"unknown coordination_mode '{config.coordination_mode}' "
            "(expected 'none', 'sbx_h'/'sbx', or 'sbx_v'/'sbxv')."
        )

    # ── SBX (Scheduled Boundary Exchange) — coordination_mode="sbx" ─────────
    # Horizontal corridor scheduling at fixed contract prices
    # (STATUS_SBX.md, plan v2/v2.2).  Adapter construction is deferred to
    # the first TSO tick at/after sbx_warmup_s (contract defaults from
    # the SETTLED closed-loop state, A7 as revised in STATUS_SBX.md
    # Phase 5); the checks below fail fast at configuration time.
    sbx_runtime: Dict[str, Any] = {"adapter": None, "config": None}
    if config.coordination_mode == "sbx":
        # v6 (2026-07-12): the adapter reads measurements and controller
        # CONFIG only — no cached-Jacobian access remains, so the former
        # numerical_h / sensitivities restrictions are gone with the
        # deal layer.
        if not (0.0 <= config.sbx_warmup_s < config.n_total_s):
            raise ValueError(
                f"sbx_warmup_s ({config.sbx_warmup_s}) must lie inside "
                f"the simulation horizon [0, {config.n_total_s})."
            )
        from sbx_h.config import SBXConfig as _SBXConfig
        _sbx_cfg = config.sbx_config
        if _sbx_cfg is None:
            _sbx_cfg = _SBXConfig(tso_period_s=float(config.tso_period_s))
        if not isinstance(_sbx_cfg, _SBXConfig):
            raise ValueError(
                "MultiTSOConfig.sbx_config must be an sbx_h.config.SBXConfig "
                f"instance (got {type(_sbx_cfg).__name__})."
            )
        if abs(_sbx_cfg.tso_period_s - float(config.tso_period_s)) > 1e-9:
            raise ValueError(
                f"SBXConfig.tso_period_s ({_sbx_cfg.tso_period_s}) must "
                f"match MultiTSOConfig.tso_period_s ({config.tso_period_s}) "
                "— the cycle length k_sched is defined in TSO iterations."
            )
        sbx_runtime["config"] = _sbx_cfg

    # ── SBX-V (vertical band-and-request, TSO–DSO) — "sbxv" ─────────────
    # STATUS_SBXV.md; plan §9 Phase 5 wiring.  The adapter installs
    # PricingSolver proxies on the zone TSO controllers (Phase-1 seam)
    # and is constructed BEFORE the main loop (metering starts at t=0);
    # everything below is inert under every other coordination_mode.
    sbxv_runtime: Dict[str, Any] = {"adapter": None, "config": None}
    if config.coordination_mode == "sbxv":
        from sbx_v.config import SBXVConfig as _SBXVConfig
        _sbxv_cfg = config.sbxv_config
        if _sbxv_cfg is None:
            _sbxv_cfg = _SBXVConfig(
                tso_period_s=float(config.tso_period_s))
        if not isinstance(_sbxv_cfg, _SBXVConfig):
            raise ValueError(
                "MultiTSOConfig.sbxv_config must be an "
                "sbx_v.config.SBXVConfig instance "
                f"(got {type(_sbxv_cfg).__name__})."
            )
        if abs(_sbxv_cfg.tso_period_s
               - float(config.tso_period_s)) > 1e-9:
            raise ValueError(
                f"SBXVConfig.tso_period_s ({_sbxv_cfg.tso_period_s}) "
                f"must match MultiTSOConfig.tso_period_s "
                f"({config.tso_period_s}) — the SBX-V window counter is "
                "defined in TSO iterations (STATUS_SBXV.md §0.3)."
            )
        sbxv_runtime["config"] = _sbxv_cfg

    # =========================================================================
    # STEP 7: Initialise DSO controllers (one per HV sub-network, all zones)
    #         (skipped when dso_mode='local'; see local-mode print branch)
    # =========================================================================
    dso_controllers: Dict[str, DSOController] = {}

    if config.dso_mode == "local":
        if verbose >= 1:
            print()
            print("[7] DSO mode = 'local' — skipping OFO DSO controllers.")
            print("    Coupler OLTCs: pandapower DiscreteTapControl (AVR)")
            n_der_total = sum(len(hv.sgen_indices) for hv in meta.hv_networks)
            if config.dso_q_mode == "qv":
                print(f"    DER Q control: Q(V) linear droop, "
                      f"V_set={config.dso_qv_vref_pu:.3f} p.u., "
                      f"slope={config.dso_qv_slope_pu:.3f}, "
                      f"deadband={config.dso_qv_deadband_pu:.3f}  "
                      f"({n_der_total} DER)")
            else:
                print(f"    DER Q control: cos phi = {config.dso_cosphi:.2f} "
                      f"({n_der_total} DER)")

    if config.dso_mode == "local":
        # No OFO DSO controllers — skip to step 7.
        pass
    else:
        if verbose >= 1:
            print()
            print("[7] Initialising DSO controllers (5 HV sub-networks) ...")

    _t_step6 = perf_counter()
    for hv in meta.hv_networks if config.dso_mode != "local" else []:
        # Allow-list filter: when ``config.dso_ids_to_run`` is non-empty the
        # runner constructs OFO controllers only for the listed DSOs.  The
        # remaining HV sub-networks still exist in the plant network and
        # exchange power through their coupling 3W transformers, but they
        # have no OFO controller — their DERs run only the plant-side
        # Q(V) / cos(phi) loop and their OLTC taps stay at the value
        # computed during the OLTC initialisation phase.  Used by
        # ``003_M_DSO_CIGRE_2026.py`` to focus the optimisation on DSO_2.
        if config.dso_ids_to_run and hv.net_id not in config.dso_ids_to_run:
            if verbose >= 1:
                print(f"  [7] {hv.net_id}: skipped (not in dso_ids_to_run)")
            continue
        dso_id = hv.net_id  # e.g. "DSO_1"
        interface_trafos = list(hv.coupling_trafo_indices)
        # Every DSO DER stays as pp.sgen under Q_cor mode (no promotion).
        der_indices = list(hv.sgen_indices)
        v_buses = list(hv.bus_indices)

        # HV lines — filter to PQ-bus endpoints only (same as TN lines)
        hv_lines = [
            li for li in hv.line_indices
            if int(net.line.at[li, "from_bus"]) not in pv_and_slack_buses_run
            and int(net.line.at[li, "to_bus"]) not in pv_and_slack_buses_run
        ]
        hv_line_max = [float(net.line.at[li, "max_i_ka"]) for li in hv_lines]

        # G_w diagonal: [Q_cor_DER | OLTC_tap].  Q_cor units are Mvar,
        # so g_w_dso_der (1/Mvar²) is the right knob.
        dso_gw_diag = np.concatenate([
            np.full(len(der_indices), config.g_w_dso_der),
            np.full(len(interface_trafos), config.g_w_dso_oltc),
        ])

        dso_cfg = DSOControllerConfig(
            der_indices=der_indices,
            oltc_trafo_indices=interface_trafos,
            shunt_bus_indices=[],
            shunt_q_steps_mvar=[],
            interface_trafo_indices=interface_trafos,
            voltage_bus_indices=v_buses,
            current_line_indices=hv_lines,
            current_line_max_i_ka=hv_line_max if hv_lines else None,
            g_q=config.g_q,
            v_setpoints_pu=np.full(len(v_buses), v_set),
            g_v=config.dso_g_v,
            gamma_oltc_q=config.dso_gamma_oltc_q,
            qv_slope_pu=config.dso_qv_slope_pu,
        )

        dso_s_rated = np.array(
            [float(net.sgen.at[s, "sn_mva"]) for s in der_indices],
            dtype=np.float64,
        )
        dso_p_max = np.array(
            [float(net.sgen.at[s, "p_mw"]) for s in der_indices],
            dtype=np.float64,
        )

        dso_der_op_diagrams = []
        for s in der_indices:
            od = net.sgen.at[s, "op_diagram"] if "op_diagram" in net.sgen.columns else None
            dso_der_op_diagrams.append(str(od) if od and str(od) != "nan" else "VDE-AR-N-4120-v2")

        dso_bounds = ActuatorBounds(
            der_indices=np.array(der_indices, dtype=np.int64),
            der_s_rated_mva=dso_s_rated,
            der_p_max_mw=dso_p_max,
            oltc_indices=np.array(interface_trafos, dtype=np.int64),
            oltc_tap_min=np.array(
                [int(net.trafo3w.at[t, "tap_min"]) for t in interface_trafos],
                dtype=np.int64,
            ),
            oltc_tap_max=np.array(
                [int(net.trafo3w.at[t, "tap_max"]) for t in interface_trafos],
                dtype=np.int64,
            ),
            shunt_indices=np.array([], dtype=np.int64),
            shunt_q_mvar=np.array([], dtype=np.float64),
            der_op_diagrams=dso_der_op_diagrams,
        )

        n_iface = len(interface_trafos)
        n_v = len(v_buses)
        n_i = len(hv_lines)
        # Q_cor mode: no Q_realized soft rows.
        dso_gz_target = np.concatenate([
            np.full(n_iface, config.g_z_interface),  # interface-Q slacks
            np.full(n_v,     config.g_z_voltage),    # voltage slacks
            np.full(n_i,     config.g_z_current),    # current slacks
        ])
        if config.g_z_warmup_s > 0:
            dso_gz = np.where(dso_gz_target > 0, config.g_z_warmup_value, 0.0)
        else:
            dso_gz = dso_gz_target
        dso_ofo = OFOParameters(
            g_w=dso_gw_diag,
            g_z=dso_gz,
            g_u=np.zeros_like(dso_gw_diag),
            alpha=1.0,
            int_max_step=config.int_max_step,
            int_cooldown=config.int_cooldown,
            int_cooldown_s=config.oltc_cooldown_s,
            adapt_g_w_classes=config.dso_adapt_g_w_classes(),
            g_w_adapt_meta=config.make_g_w_adapt_meta(),
        )

        dso_ctrl = DSOController(
            controller_id=dso_id,
            params=dso_ofo,
            config=dso_cfg,
            network_state=ns0,
            actuator_bounds=dso_bounds,
            sensitivities=shared_jac,
        )
        # _u_current is initialised later (step 7e), after profiles and
        # OLTC/STATCOM init have settled the operating point.
        dso_controllers[dso_id] = dso_ctrl

        if verbose >= 1:
            print(f"  {dso_id} (zone {hv.zone}): {len(der_indices)} DER, "
                  f"{n_iface} PCC trafos, {n_v} V-buses, {n_i} lines")

    if verbose >= 1 and config.dso_mode != "local":
        print(f"  [T] step [7] DSO controller construction: {perf_counter() - _t_step6:.2f} s")

    # Map each DSO controller ID to the ID of its supervising TSO controller.
    # TSO controller IDs follow the pattern "tso_zone_{z}" (see TSOController init above).
    dso_to_tso_id: Dict[str, str] = {
        hv.net_id: f"tso_zone_{_hv_zone(hv)}"
        for hv in meta.hv_networks
    }

    # DSO group map (trivial: each DSO = its own group)
    dso_group_map: Dict[str, str] = {hv.net_id: hv.net_id for hv in meta.hv_networks}
    last_dso_q_set_mvar: Dict[str, Optional[NDArray]] = {
        dso_id: None for dso_id in dso_ids
    }
    # Latest TSO-dispatched interface Q setpoint per coupling-trafo index,
    # persisted across steps.  Lets the local-DSO recording path (one-sided-OFO
    # variants: TSO OFO + local DSO, e.g. comparison scenario "T1") expose the
    # same per-trafo interface setpoint field as the cascaded OFO path so the
    # CIGRE interface-tracking-error figure can overlay both uniformly.
    last_pcc_set_per_trafo: Dict[int, float] = {}

    # =========================================================================
    # STEP 8: Initialise MultiTSOCoordinator
    # =========================================================================
    if verbose >= 1:
        print()
        print("[8] Initialising MultiTSOCoordinator ...")

    coordinator = MultiTSOCoordinator(
        zones=list(zone_defs.values()),
        net=net,
        verbose=verbose,
    )
    for z, ctrl in tso_controllers.items():
        coordinator.register_tso_controller(z, ctrl)

    # =========================================================================
    # STEP 8b: Build the single centralized controller (control_scope='central')
    # =========================================================================
    # The CentralOFOController owns the UNION of every zone's + every HV
    # sub-network's actuators and observations: all TSO+DSO DER (w-shift Q),
    # all synchronous-machine AVR setpoints, all 2W machine OLTCs, all TSO
    # shunts, and all 3W coupler OLTCs.  It observes every TN+HV PQ bus
    # voltage and line current, with a generator-Q capability soft band.  No
    # interface-Q / tie-Q tracking — the sole objective is voltage tracking,
    # with g_v on TN buses and central_dso_g_v on HV buses.  It always uses the
    # FULL-network ``shared_jac`` (re-assigned after the Phase-2 rebuild),
    # independent of the local-sensitivity flags that the per-zone controllers
    # honour.  The per-zone TSO / DSO controllers above are kept solely as the
    # recording lens (their ``actuator_bounds`` feed the per-zone figures).
    central_controller: Optional[CentralOFOController] = None
    central_cfg: Optional[CentralControllerConfig] = None
    if _central:
        if verbose >= 1:
            print()
            print("[8b] Building single centralized controller (V5 reference) ...")

        # ── Actuator union (deterministic order: zone 0,1,2 then DSO_1..) ──
        tso_der_set = set(int(s) for s in meta.tso_der_indices)
        c_der_indices = (
            [int(s) for s in meta.tso_der_indices if int(s) in net.sgen.index]
            + [int(s) for s in meta.dso_der_indices if int(s) in net.sgen.index]
        )
        c_gen_indices: List[int] = []
        c_gen_buses: List[int] = []
        c_oltc2w: List[int] = []
        c_shunt_buses: List[int] = []
        c_shunt_qsteps: List[float] = []
        for z in sorted(zone_defs.keys()):
            zd = zone_defs[z]
            c_gen_indices += [int(g) for g in zd.gen_indices]
            c_gen_buses += [int(b) for b in zd.gen_bus_indices]
            c_oltc2w += [int(t) for t in zd.oltc_trafo_indices]
            c_shunt_buses += [int(b) for b in zd.shunt_bus_indices]
            c_shunt_qsteps += [float(q) for q in zd.shunt_q_steps_mvar]
        c_oltc3w = [int(t) for hv in meta.hv_networks for t in hv.coupling_trafo_indices]

        # ── Observation union: TN PQ buses + HV buses (dedup, order-stable) ──
        hv_bus_set: Set[int] = {int(b) for hv in meta.hv_networks for b in hv.bus_indices}
        c_v_buses: List[int] = []
        _seen_v: Set[int] = set()
        for z in sorted(zone_defs.keys()):
            for b in zone_defs[z].v_bus_indices:        # TN PQ buses
                b = int(b)
                if b not in _seen_v:
                    _seen_v.add(b); c_v_buses.append(b)
        for b in sorted(hv_bus_set):                    # HV (STS) buses
            if b in pv_and_slack_buses_run or b not in net.bus.index:
                continue
            if b not in _seen_v:
                _seen_v.add(b); c_v_buses.append(b)

        # Per-bus voltage weight: g_v on TN buses, central_dso_g_v on HV buses.
        c_g_v_per_bus = np.array(
            [config.central_dso_g_v if b in hv_bus_set else config.g_v
             for b in c_v_buses],
            dtype=np.float64,
        )

        # ── Line union: TN zone lines + HV lines (PQ-endpoint only) ──
        c_lines: List[int] = []
        c_lines_max: List[float] = []
        _seen_l: Set[int] = set()
        for z in sorted(zone_defs.keys()):
            zd = zone_defs[z]
            for li, imax in zip(zd.line_indices, zd.line_max_i_ka):
                li = int(li)
                if li not in _seen_l:
                    _seen_l.add(li); c_lines.append(li); c_lines_max.append(float(imax))
        for hv in meta.hv_networks:
            for li in hv.line_indices:
                li = int(li)
                if li in _seen_l:
                    continue
                if (int(net.line.at[li, "from_bus"]) in pv_and_slack_buses_run
                        or int(net.line.at[li, "to_bus"]) in pv_and_slack_buses_run):
                    continue
                _seen_l.add(li)
                c_lines.append(li)
                c_lines_max.append(float(net.line.at[li, "max_i_ka"]))

        n_v_c = len(c_v_buses)

        central_cfg = CentralControllerConfig(
            der_indices=c_der_indices,
            pcc_trafo_indices=[],
            pcc_dso_controller_ids=[],
            oltc_trafo_indices=c_oltc2w,
            shunt_bus_indices=c_shunt_buses,
            shunt_q_steps_mvar=c_shunt_qsteps,
            voltage_bus_indices=c_v_buses,
            current_line_indices=c_lines,
            current_line_max_i_ka=c_lines_max if c_lines else None,
            v_setpoints_pu=np.full(n_v_c, v_set),
            g_v=config.g_v,
            gen_indices=c_gen_indices,
            gen_bus_indices=c_gen_buses,
            g_q_tso=0.0,
            qv_slope_pu=config.tso_qv_slope_pu,
            oltc_trafo3w_indices=c_oltc3w,
            g_v_per_bus=c_g_v_per_bus,
        )

        # ── ActuatorBounds spanning every actuator ──
        c_der_s_rated = np.array(
            [float(net.sgen.at[s, "sn_mva"]) for s in c_der_indices], dtype=np.float64,
        )
        c_der_p_max = np.array(
            [float(net.sgen.at[s, "p_mw"]) for s in c_der_indices], dtype=np.float64,
        )
        c_der_op_diagrams = []
        for s in c_der_indices:
            od = net.sgen.at[s, "op_diagram"] if "op_diagram" in net.sgen.columns else None
            c_der_op_diagrams.append(str(od) if od and str(od) != "nan" else "VDE-AR-N-4120-v2")
        c_gen_params = []
        for g in c_gen_indices:
            c_gen_params.append(
                generator_parameters_for_type(
                    net.gen.at[g, "type"] if "type" in net.gen.columns else "",
                    float(net.gen.at[g, "sn_mva"]),
                    float(net.gen.at[g, "max_p_mw"]),
                )
            )
        central_bounds = ActuatorBounds(
            der_indices=np.array(c_der_indices, dtype=np.int64),
            der_s_rated_mva=c_der_s_rated,
            der_p_max_mw=c_der_p_max,
            oltc_indices=np.array(c_oltc2w, dtype=np.int64),
            oltc_tap_min=np.array(
                [int(net.trafo.at[t, "tap_min"]) for t in c_oltc2w], dtype=np.int64,
            ),
            oltc_tap_max=np.array(
                [int(net.trafo.at[t, "tap_max"]) for t in c_oltc2w], dtype=np.int64,
            ),
            shunt_indices=np.array(c_shunt_buses, dtype=np.int64),
            shunt_q_mvar=np.array(c_shunt_qsteps, dtype=np.float64),
            gen_params=c_gen_params,
            der_op_diagrams=c_der_op_diagrams,
        )

        # ── OFO weights.  g_w order = control-vector order
        #    [ Q_DER | V_gen | s_OLTC2w | s_shunt | s_OLTC3w ] (PCC empty).
        c_gw_der = np.array(
            [config.g_w_der if s in tso_der_set else config.g_w_dso_der
             for s in c_der_indices],
            dtype=np.float64,
        )
        c_gw = np.concatenate([
            c_gw_der,
            np.full(len(c_gen_indices), config.g_w_gen),
            np.full(len(c_oltc2w), config.g_w_tso_oltc),
            np.full(len(c_shunt_buses), config.g_w_tso_shunt),
            np.full(len(c_oltc3w), config.g_w_dso_oltc),
        ])
        # g_z order = output-vector order [ V | I | Q_gen ] (Q_PCC/Q_tie empty).
        c_gz = np.concatenate([
            np.full(n_v_c, config.g_z_voltage),
            np.full(len(c_lines), config.g_z_current),
            np.full(len(c_gen_indices), config.g_z_q_gen),
        ])
        central_params = OFOParameters(
            g_w=c_gw,
            g_z=c_gz,
            g_u=np.zeros_like(c_gw),
            alpha=1.0,
            int_max_step=config.int_max_step,
            int_cooldown=config.int_cooldown,
            int_cooldown_s=config.oltc_cooldown_s,
        )

        central_controller = CentralOFOController(
            controller_id="central",
            params=central_params,
            config=central_cfg,
            network_state=ns0,
            actuator_bounds=central_bounds,
            sensitivities=shared_jac,
        )
        if verbose >= 1:
            print(f"  central controller: {len(c_der_indices)} DER, "
                  f"{len(c_gen_indices)} gen AVR, {len(c_oltc2w)} 2W OLTC, "
                  f"{len(c_oltc3w)} 3W OLTC, {len(c_shunt_buses)} shunt; "
                  f"{n_v_c} V-buses ({len(hv_bus_set & set(c_v_buses))} HV), "
                  f"{len(c_lines)} lines")

    # =========================================================================
    # STEP 9: Load profiles and compute zonal generator dispatch
    # =========================================================================
    use_profiles = config.use_profiles
    start_time = config.start_time
    # Deep copy, not ``list(...)``: ``prepare_load_contingencies`` *resolves*
    # events by writing ``ev.element_index`` back into them (contingency.py:161).
    # A shallow list copy shares the ``ContingencyEvent`` objects with
    # ``config.contingencies``, so that resolution leaks into the caller's
    # config -- and any caller that reuses one config for several runs (every
    # tuning driver: it builds the scenario set once and runs it repeatedly)
    # then re-enters with ``action='connect'`` *and* an explicit index, which
    # the contradiction guard at contingency.py:70 correctly rejects. The run
    # died with an empty log from the second use onward, and only for scenarios
    # using mode-3 ``connect`` load events.
    contingencies = (
        [copy.deepcopy(ev) for ev in config.contingencies]
        if config.contingencies else []
    )

    profiles = None
    gen_dispatch = None

    # Start the init-timing clock before the profile branch: the total is
    # reported unconditionally further down, but profiles are optional
    # (a Gate E replay runs with use_profiles=False).
    _t_init_total = perf_counter()

    if use_profiles:
        profiles_csv = config.profiles_csv or DEFAULT_PROFILES_CSV
        if verbose >= 1:
            print()
            print(f"[9] Loading profiles from {profiles_csv}")
            print(f"     start_time = {start_time:%d.%m.%Y %H:%M}")

        _t = perf_counter()
        profiles = load_profiles(profiles_csv, timestep_s=config.dt_s)
        snapshot_base_values(net)
        if verbose >= 1:
            print(f"  [T] load_profiles + snapshot_base_values: {perf_counter() - _t:.2f} s")

        # ── Exogenous load step (disturbance-rejection studies) ───────────
        # Applied HERE, to the already-interpolated DataFrame, so the step is
        # exact at dt_s resolution.  Stepping the source CSV instead would be
        # smeared into a 15-minute ramp by load_profiles' linear interpolation.
        # Both plants read this same frame -- static via apply_profiles, RMS
        # via Plant.apply_exogenous (EvtLod) -- so the disturbance reaches both
        # legs identically and through supported paths only.  Inert unless
        # load_step_time_s is set.
        _step_t = getattr(config, "load_step_time_s", None)
        _step_bus = getattr(config, "load_step_bus", None)
        if _step_t is not None and _step_bus is not None:
            # LOCALISED additive step: give the target load its own synthetic
            # profile column equal to its original profile plus delta/base_p
            # from the step instant. Both plants consume profile columns
            # per-load, so this reaches the RMS plant through the same EvtLod
            # path as any other profile change -- no new plumbing.
            _dp = float(getattr(config, "load_step_delta_mw", 0.0))
            _cand = [int(i) for i in net.load.index
                     if int(net.load.at[i, "bus"]) == int(_step_bus)
                     and pd.notna(net.load.at[i, "profile_p"])]
            if not _cand:
                raise ValueError(
                    f"load_step_bus={_step_bus} has no profiled load row")
            # largest load at the bus, so the step lands on the dominant row
            _row = max(_cand, key=lambda i: float(net.load.at[i, "base_p_mw"]))
            _base = float(net.load.at[_row, "base_p_mw"])
            if abs(_base) < 1e-9:
                raise ValueError(
                    f"load row {_row} at bus {_step_bus} has base_p_mw = 0, "
                    "so an additive step cannot be expressed as a profile "
                    "multiplier")
            _src = str(net.load.at[_row, "profile_p"])
            if _src not in profiles.columns:
                raise ValueError(f"profile column {_src!r} not in the data")
            _t_step = start_time + timedelta(seconds=float(_step_t))
            _mask = profiles.index >= _t_step
            if not _mask.any():
                raise ValueError(
                    f"load step at t={_step_t:g}s ({_t_step}) lies beyond the "
                    f"profile horizon (ends {profiles.index.max()})")
            _newcol = f"STEP_BUS_{int(_step_bus)}"
            profiles = profiles.copy()
            profiles[_newcol] = profiles[_src]
            profiles.loc[_mask, _newcol] = (
                profiles.loc[_mask, _src] + _dp / _base)
            net.load.at[_row, "profile_p"] = _newcol
            if verbose >= 1:
                print(f"  [load-step] LOCALISED {_dp:+g} MW at bus "
                      f"{int(_step_bus)} (load row {_row}, base "
                      f"{_base:.2f} MW, profile {_src} -> {_newcol}) from "
                      f"t={_step_t:g}s ({_t_step:%Y-%m-%d %H:%M:%S})")
        elif _step_t is not None:
            _f = float(getattr(config, "load_step_factor", 1.0))
            _cols = [c for c in getattr(config, "load_step_columns", ())
                     if c in profiles.columns]
            if not _cols:
                raise ValueError(
                    "load_step_time_s is set but none of load_step_columns "
                    f"{tuple(getattr(config, 'load_step_columns', ()))} exist "
                    f"in the profile data (have: {list(profiles.columns)})")
            _t_step = start_time + timedelta(seconds=float(_step_t))
            _mask = profiles.index >= _t_step
            if not _mask.any():
                raise ValueError(
                    f"load step at t={_step_t:g}s ({_t_step}) lies beyond the "
                    f"profile horizon (ends {profiles.index.max()})")
            profiles = profiles.copy()
            profiles.loc[_mask, _cols] = profiles.loc[_mask, _cols] * _f
            if verbose >= 1:
                print(f"  [load-step] x{_f:g} on {_cols} from t={_step_t:g}s "
                      f"({_t_step:%Y-%m-%d %H:%M:%S}); "
                      f"{int(_mask.sum())} of {len(profiles)} samples affected")

        # Pre-create dormant loads for load-contingency events (must be
        # after snapshot_base_values so base columns exist).
        if contingencies:
            prepare_load_contingencies(net, contingencies, verbose=verbose)

        # Clip profile DataFrame to the simulation window only.
        # Without this, compute_zonal_gen_dispatch iterates the full profile
        # horizon (up to 525 600 rows at 60 s resolution) unnecessarily.
        # Note: must clip BOTH start and end — load_profiles returns the
        # full year (e.g. 2016-01-01 .. 2016-12-31).  ``profiles.loc[:t_end]``
        # alone would still iterate every row from the CSV start through
        # ``start_time``, which for an April start_time is ~3.5 months of
        # rows that compute_zonal_gen_dispatch would scan in vain.
        t_end = start_time + timedelta(seconds=config.n_total_s)
        profiles = profiles.loc[start_time:t_end]

        # Apply initial profiles
        _t = perf_counter()
        apply_profiles(net, profiles, start_time)
        if verbose >= 1:
            print(f"  [T] apply_profiles: {perf_counter() - _t:.2f} s")

        if config.use_zonal_gen_dispatch:
            # Per-generator P_min: 20% of P_max (consistent with
            # GeneratorParameters.p_min_mw construction above).
            _gen_p_min_dict: Dict[int, float] = {
                int(g): float(net.gen.at[g, "p_mw"]) * 0.0
                for g in net.gen.index
            }
            _t = perf_counter()
            gen_dispatch = compute_zonal_gen_dispatch(
                net, profiles, dispatch_zone_map,
                gen_p_min_mw=_gen_p_min_dict,
            )
            apply_gen_dispatch(net, gen_dispatch, start_time)
            if verbose >= 1:
                print(f"  [T] compute+apply zonal gen dispatch: {perf_counter() - _t:.2f} s")

        # Re-converge after profile application
        _t = perf_counter()
        pp.runpp(net, max_iteration=50, run_control=False, calculate_voltage_angles=True, init='auto',
                 distributed_slack=config.distributed_slack,
                 enforce_q_lims=config.enforce_q_lims_plant)
        if verbose >= 1:
            print(f"  [T] post-profile pp.runpp: {perf_counter() - _t:.2f} s")

    # ── STEP 10: Combined operating-point init (three phases) ────────────
    # After profiles, bring STATCOM Q, OLTC taps, and plant-side Q(V)
    # loops to a self-consistent state at the profile-scaled operating
    # point.  Done in three phases so the TN settles *before* the coupler
    # 3W OLTCs adjust, and the plant-side q_mode loops install AFTER both:
    #
    #   Phase 1 (TSO):  STATCOM Q (temp-PV-gen trick) + machine 2W OLTC
    #                   → one run_control PF at v_setpoint_pu.
    #   Phase 2 (DSO):  coupler 3W OLTC
    #                   → one run_control PF at oltc_init_v_target_pu.
    #   Phase 3 (DER):  install QVLocalLoop / CosPhiConstLoop per DER
    #                   (deferred from step [4]) and seed q_mvar with
    #                   the analytical closed-loop equilibrium.
    #
    # In "cascade" DSO mode the coupler controllers are dropped after
    # Phase 2 (OFO takes over).  In "local" DSO mode they stay active
    # as local AVR for the rest of the simulation.
    from pandapower.control import DiscreteTapControl

    v_init_mt  = v_set                         # machine trafos → v_setpoint
    v_init_dso = config.oltc_init_v_target_pu  # coupler MV-side → 1.03
    tol_pu     = config.dso_oltc_init_tol_pu
    _local_dso = config.dso_mode == "local"
    _local_tso = config.tso_mode == "local"
    # Plant-side Q(V) loops (QVLocalLoop / CosPhiConstLoop) plus any
    # DiscreteTapControl on couplers / machine trafos must be iterated
    # by pp.runpp(run_control=True) every step.  Always True under the
    # Q_cor plant model.
    _run_control = True

    # -- Phase 1: STATCOM Q + machine 2W OLTC -----------------------------
    # TSO-side only: HV-side (subnet=="DN") STATCOMs stay at q_mvar=0
    # here; the DSO controller dispatches their Q at run time.
    _statcom_mask = (
        net.sgen["name"].astype(str).str.contains("STATCOM")
        & (net.sgen["subnet"].astype(str) != "DN")
    )
    _statcom_idxs = net.sgen.index[_statcom_mask].tolist()

    _tmp_map: Dict[int, int] = {}
    for si in _statcom_idxs:
        bus = int(net.sgen.at[si, "bus"])
        p = float(net.sgen.at[si, "p_mw"])
        sn = float(net.sgen.at[si, "sn_mva"])
        net.sgen.at[si, "in_service"] = False
        gi = pp.create_gen(
            net, bus=bus, p_mw=p, vm_pu=v_set, sn_mva=sn,
            max_q_mvar=sn, min_q_mvar=-sn,
            in_service=True, name="_TEMP_INIT",
        )
        _tmp_map[int(gi)] = si

    for tidx in meta.machine_trafo_indices:
        DiscreteTapControl(
            net, element_index=tidx,
            vm_lower_pu=v_init_mt - tol_pu,
            vm_upper_pu=v_init_mt + tol_pu,
            side="hv", element="trafo",
        )

    if verbose >= 1:
        print(f"[10.1] Phase 1 (TSO): {len(_tmp_map)} STATCOM Q via temp-PV-gens + "
              f"{len(meta.machine_trafo_indices)} machine OLTC "
              f"-> target {v_init_mt:.3f} +-{tol_pu:.3f} p.u.")

    _t = perf_counter()
    pp.runpp(net, run_control=True, calculate_voltage_angles=True,
             max_iteration=50, distributed_slack=config.distributed_slack,
             enforce_q_lims=config.enforce_q_lims_plant)
    if verbose >= 1:
        print(f"  [T] Phase 1 pp.runpp(run_control=True): {perf_counter() - _t:.2f} s")

    # Transfer Q from temp-PV-gens to STATCOM sgens, then drop temp gens.
    for gi, si in _tmp_map.items():
        net.sgen.at[si, "q_mvar"] = float(net.res_gen.at[gi, "q_mvar"])
        net.sgen.at[si, "in_service"] = True
    if _tmp_map:
        net.gen.drop(index=list(_tmp_map.keys()), inplace=True)

    # Drop machine-trafo controllers — but preserve any plant-side
    # QVLocalLoops if they were already installed (defensive: under the
    # current flow they install at [10.3], after this drop, so this is a
    # no-op unless an external pre_loop_hook seeded them earlier).
    if hasattr(net, "controller") and len(net.controller) > 0:
        drop_idx = [
            idx for idx, row in net.controller.iterrows()
            if not isinstance(row["object"], QVLocalLoop)
        ]
        if drop_idx:
            net.controller.drop(index=drop_idx, inplace=True)

    if verbose >= 2:
        print("[10.1] Phase 1 result (machine 2W OLTC):")
        for tidx, gidx in zip(meta.machine_trafo_indices, meta.machine_trafo_gen_map):
            tap = int(net.trafo.at[tidx, "tap_pos"])
            hv_bus = int(net.trafo.at[tidx, "hv_bus"])
            vm = float(net.res_bus.at[hv_bus, "vm_pu"])
            print(f"    trafo {tidx} (gen {gidx}): tap_pos={tap:+d}, "
                  f"V_hv={vm:.4f} p.u.")

    # -- Phase 2: coupler 3W OLTC -----------------------------------------
    for hv in meta.hv_networks:
        for t3w in hv.coupling_trafo_indices:
            DiscreteTapControl(
                net, element_index=t3w,
                vm_lower_pu=v_init_dso - tol_pu,
                vm_upper_pu=v_init_dso + tol_pu,
                side="mv", element="trafo3w",
            )

    if verbose >= 1:
        n_coup = sum(len(hv.coupling_trafo_indices) for hv in meta.hv_networks)
        print(f"[10.2] Phase 2 (DSO): {n_coup} coupler 3W OLTC "
              f"-> target {v_init_dso:.3f} +-{tol_pu:.3f} p.u.")

    _t = perf_counter()
    pp.runpp(net, run_control=True, calculate_voltage_angles=True,
             max_iteration=100, distributed_slack=config.distributed_slack,
             enforce_q_lims=config.enforce_q_lims_plant)
    if verbose >= 1:
        print(f"  [T] Phase 2 pp.runpp(run_control=True): {perf_counter() - _t:.2f} s")

    if verbose >= 2:
        for hv in meta.hv_networks:
            for t3w in hv.coupling_trafo_indices:
                tap = int(net.trafo3w.at[t3w, "tap_pos"])
                mv_bus = int(net.trafo3w.at[t3w, "mv_bus"])
                vm = float(net.res_bus.at[mv_bus, "vm_pu"])
                print(f"  {hv.net_id} trafo3w {t3w}: tap_pos={tap:+d}, "
                      f"V_mv={vm:.4f} p.u.")

    # In "cascade" DSO mode, drop coupler controllers (OFO takes over).
    # In "local" DSO mode, keep them active as local AVR.  Either way
    # preserve the plant-side QVLocalLoops if any are present (defensive;
    # under the current flow they install at [10.3] AFTER this drop).
    # Central mode (V5) always drops them: the single OFO controller owns
    # the coupler 3W OLTC taps, so a co-active DiscreteTapControl would
    # fight it for the same taps every run_control PF.
    if not _local_dso or _central:
        if hasattr(net, "controller") and len(net.controller) > 0:
            drop_idx = [
                idx for idx, row in net.controller.iterrows()
                if not isinstance(row["object"], (QVLocalLoop, CosPhiConstLoop))
            ]
            if drop_idx:
                net.controller.drop(index=drop_idx, inplace=True)
    elif verbose >= 1:
        print(f"  [local DSO] Kept {len(net.controller)} coupler OLTC "
              f"DiscreteTapControl(s) active for simulation.")

    # ── Phase 3 (DER): install plant-side q_mode loops ──
    # Phase 1/2 init has settled; now install QVLocalLoop / CosPhiConstLoop
    # on every tagged DER so the local Q(V) feedback runs through the
    # main loop alongside the OFO.  TSO and DSO DERs install separately
    # so each gets its level-appropriate convergence tolerance
    # (TSO: 0.1 Mvar, DSO: 0.01 Mvar by default).
    #
    # Seed each DER's q_mvar with the *exact linear closed-loop
    # equilibrium* (Soleimani §IV-B eq. 18) computed from the post-
    # Phase-2 V via ``seed_qv_equilibrium``.  This bypasses the
    # multi-DER Gauss-Jacobi instability that broke the 24-hour main()
    # run with 500-iteration controller loops: the controllers install
    # already at their attractor and only need to refine residual
    # nonlinearity over a handful of iterations.
    tso_sgens = [int(s) for s in meta.tso_der_indices
                 if int(s) in net.sgen.index]
    dso_sgens = [int(s) for s in meta.dso_der_indices
                 if int(s) in net.sgen.index]
    # Reset the w-shift actuator state (the OFO has not commanded yet)
    # before the analytical seed.  ``q_set_mvar`` starts at 0.  By default
    # ``qv_vref_anchor_pu`` is left as the apply-step's responsibility
    # — until the first apply, the local QVLocalLoop falls back to the
    # nominal ``qv_vref_pu`` (1.03).  The RMS plant, however, anchors to the
    # local voltage ``v_lf`` at init, so the two plants droop about different
    # anchors on the FIRST profiled re-solve.  ``seed_der_anchor_to_local_v``
    # closes that one-interval mismatch by anchoring the static side to the
    # same local operating-point voltage at init (2026-07-24 anchor-seed test).
    _n_anchor_seeded = 0
    for s in tso_sgens + dso_sgens:
        if "q_set_mvar" in net.sgen.columns:
            net.sgen.at[s, "q_set_mvar"] = 0.0
        if (config.seed_der_anchor_to_local_v
                and "qv_vref_anchor_pu" in net.sgen.columns
                and hasattr(net, "res_bus") and net.res_bus is not None):
            bus = int(net.sgen.at[s, "bus"])
            if bus in net.res_bus.index:
                net.sgen.at[s, "qv_vref_anchor_pu"] = float(
                    net.res_bus.at[bus, "vm_pu"])
                _n_anchor_seeded += 1
    if config.seed_der_anchor_to_local_v and verbose >= 1:
        print(f"  [anchor-seed] initialised {_n_anchor_seeded} DER Q(V) anchors "
              f"to local res_bus.vm_pu (matches RMS v_lf at init)")
    # Seed using a fresh Jacobian at the post-Phase-2 operating
    # point so S_VQ matches the network state we'll iterate from.
    if not config.disable_qv_seed:
        _seed_jac = JacobianSensitivities(net)
        seed_qv_equilibrium(
            net, tso_sgens + dso_sgens, _seed_jac,
            verbose=(verbose >= 1),
        )
    # TSO STATCOMs have R*S_VQ ~ 8 vs DSO ~ 0.7, so cap TSO damping
    # to 0.03 to keep the per-DER contraction stable under multi-DER
    # coupling (commit e2746fe rationale; previously documented in
    # qv_local_damping docstring but not enforced in code).
    _tso_damp = min(float(config.qv_local_damping), 0.03)
    n_tso = len(install_der_q_loops(
        net, tso_sgens,
        qv_damping=_tso_damp,
        qv_max_step_frac=config.qv_local_max_step_frac,
        qv_tol_mvar=config.tso_qv_tol_mvar,
    )) if tso_sgens else 0
    n_dso = len(install_der_q_loops(
        net, dso_sgens,
        qv_damping=config.qv_local_damping,
        qv_max_step_frac=config.qv_local_max_step_frac,
        qv_tol_mvar=config.dso_qv_tol_mvar,
    )) if dso_sgens else 0
    if verbose >= 1:
        print(
            f"[10.3] Phase 3 (DER): installed {n_tso + n_dso} DER q_mode loops "
            f"({n_tso} TSO @ tol={config.tso_qv_tol_mvar} Mvar damp={_tso_damp:.3f} + "
            f"{n_dso} DSO @ tol={config.dso_qv_tol_mvar} Mvar damp={float(config.qv_local_damping):.3f}) "
            f"post Phase 2; seeded q_mvar with closed-loop equilibrium."
        )

    # Re-converge with final Q and tap positions.  Run *without* the
    # plant-side controllers here: Phase 2 left the network at a
    # converged operating point and the QVLocalLoops install at a damped
    # droop seed (above), then iterate inside the first main-loop runpp.
    # Running run_control=True here would force NR to digest a 44-DER Q
    # step in one call, making inner NR ill-conditioned on large profiles
    # (long sims with contingencies).
    _t = perf_counter()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True,
             max_iteration=50,
             distributed_slack=config.distributed_slack,
             enforce_q_lims=config.enforce_q_lims_plant)
    if verbose >= 1:
        print(f"  [T] final re-converge pp.runpp: {perf_counter() - _t:.2f} s")

    # ── Converge the plant-side Q(V) loops before the seam ────────────────
    # The solve above deliberately runs run_control=False, so the state it
    # leaves is NOT a fixed point of the plant's own DER Q(V) law: the
    # QVLocalLoops were installed at a single-pass linear seed and still
    # carry a pending correction.  Historically they absorbed it inside the
    # first main-loop runpp, which is harmless for a static plant but not
    # for a substituted one:
    #
    #   * PandapowerStaticPlant removes the residual on its first advance();
    #   * PowerFactoryPlant instead ANCHORS its QVPRE blocks to this state
    #     (qset = q_lf, Vanchor = v_lf in pf.plant._anchor_qv_precontrollers),
    #     freezing the residual in permanently.
    #
    # The two plants then hold Q(V) characteristics offset by exactly that
    # residual for the whole run.  Measured on rural_700 (2026-07-30): the
    # first advance moved DER Q by +17.7 Mvar and the bus voltages by 2.3 mpu
    # mean / 27 mpu worst, which reappeared as a flat -1.8 mpu TS zone-voltage
    # offset held until the TSO's next dispatch 180 s later.
    #
    # Converging here makes the seam a genuine fixed point, so both plants
    # start on the same characteristic.  Applied unconditionally, not only
    # for a substituted plant: the seam state should satisfy the plant law
    # regardless of which plant reads it, and a quasi-static run should not
    # spend its first interval relaxing an initialisation artefact either.
    #
    # ``max_iter`` is the run_control iteration cap and MUST be raised: the
    # 44 coupled DER loops do not converge within pandapower's default 30
    # (ControllerNotConverged after 31 calls on rural_700).  300 is what
    # PandapowerStaticPlant.advance already uses, so this is the same
    # iteration budget the first main-loop solve would have had.
    _t = perf_counter()
    pp.runpp(net, run_control=True, calculate_voltage_angles=True,
             max_iteration=50, max_iter=300,
             distributed_slack=config.distributed_slack,
             enforce_q_lims=config.enforce_q_lims_plant)
    if verbose >= 1:
        print(f"  [T] Q(V) fixed-point solve at the plant seam: "
              f"{perf_counter() - _t:.2f} s")

    # ── Plant seam (RMS build plan, Phase 6) ──────────────────────────────
    # From here on every standard-loop plant interaction -- actuator
    # writes, plant response, measurement-image refresh -- goes through
    # the Plant interface.  The static default reproduces the legacy
    # behaviour bit-for-bit; a substituted RMS plant keeps ``net`` as its
    # mirror, so the direct ``net`` *reads* throughout this function
    # remain valid for both.
    if plant_factory is not None:
        configure_profiles = getattr(
            plant_factory, "configure_exogenous_profiles", None)
        if (use_profiles and profiles is not None
                and callable(configure_profiles)):
            configure_profiles(
                profiles,
                start_time=start_time,
                dt_s=config.dt_s,
                duration_s=config.n_total_s,
            )
        plant = plant_factory(net, meta=meta, zone_map=zone_map)
    else:
        plant = PandapowerStaticPlant(
            net,
            distributed_slack=config.distributed_slack,
            enforce_q_lims=config.enforce_q_lims_plant,
        )

    # ── Slack bus diagnostic after OLTC init ──────────────────────────────
    if verbose >= 1:
        # Prefer the slack-gen form (IEEE 39 distributed slack); fall back
        # to the legacy ext_grid form (TUDA benchmark, other networks).
        _slack_p, _slack_q = float("nan"), float("nan")
        if "slack" in net.gen.columns and len(net.gen) > 0:
            _slack_gens = net.gen.index[net.gen["slack"].astype(bool)].tolist()
            if _slack_gens:
                _sg = _slack_gens[0]
                _slack_p = float(net.res_gen.at[_sg, "p_mw"])
                _slack_q = float(net.res_gen.at[_sg, "q_mvar"])
        if (not np.isfinite(_slack_p)) and not net.ext_grid.empty:
            _sg = net.ext_grid.index[0]
            _slack_p = float(net.res_ext_grid.at[_sg, "p_mw"])
            _slack_q = float(net.res_ext_grid.at[_sg, "q_mvar"])
        print(f"  Slack bus: P = {_slack_p:.1f} MW, Q = {_slack_q:.1f} Mvar")
        # Warn on extreme machine trafo taps
        for tidx in meta.machine_trafo_indices:
            tap = int(net.trafo.at[tidx, "tap_pos"])
            if abs(tap) >= 7:
                hv_bus = int(net.trafo.at[tidx, "hv_bus"])
                print(f"  WARNING: Machine trafo {tidx} at tap {tap:+d} (bus {hv_bus})")

    # ── Voltage feasibility check after OLTC init ──────────────────────
    _v_violations_found = False
    if verbose >= 1:
        print(f"  {'Network':<12s} {'V_min':>7s} {'V_max':>7s} {'Headroom':>10s}")
        _items: list = []
        for hv in meta.hv_networks:
            v_buses = list(hv.bus_indices)
            vm_pu = net.res_bus.loc[v_buses, "vm_pu"].to_numpy(dtype=float)
            _items.append((hv.net_id, vm_pu))
        for z, zd in zone_defs.items():
            vm_pu = net.res_bus.loc[zd.v_bus_indices, "vm_pu"].to_numpy(dtype=float)
            _items.append((f"TSO Zone {z}", vm_pu))
        for label, vm_pu in _items:
            vm_min = float(np.min(vm_pu))
            vm_max = float(np.max(vm_pu))
            headroom = min(1.1 - vm_max, vm_min - 0.9)
            n_viol = int(np.sum(vm_pu < 0.9) + np.sum(vm_pu > 1.1))
            flag = "⚠" if n_viol > 0 or headroom < 0.02 else " "
            if n_viol > 0:
                _v_violations_found = True
            print(f"  {flag}{label:<11s} {vm_min:>7.4f} {vm_max:>7.4f} "
                  f"{headroom:>+9.4f} p.u.")
    if _v_violations_found:
        print("  ⚠ Voltage violations — MIQP may be infeasible with hard constraints.")

    # Rebuild shared Jacobian at the post-Phase-2 operating point and replace
    # the pre-profile snapshot held by every controller.  The H matrices
    # cached in each controller's _H_cache (and _sensitivity_updater) were
    # built from the stale Jacobian, so invalidate them — the next call to
    # _build_sensitivity_matrix will rebuild from the fresh shared_jac.
    _t = perf_counter()
    shared_jac = JacobianSensitivities(net)
    # Invalidate the seed_qv_equilibrium LU cache: a new sensitivities
    # object was constructed, so its id() changes and S_VQ_der may move.
    clear_seed_lu_cache()

    # ── Optional: per-controller local-net Jacobians (Ward-style boundary) ─
    # When ``config.local_sensitivities_tso`` is True, each TSO controller
    # gets a Jacobian built from its own reduced zone net (tie-line far-
    # end + 3W primary boundaries as PQ loads; synthetic shunts at the
    # 3W primary).  Same idea for DSOs when ``local_sensitivities_dso``
    # is True (HV sub-network only, 3W primary as virtual slack).  When
    # both are False, every controller shares ``shared_jac`` (the
    # historical behaviour).  See :mod:`sensitivity.network_reduction`.
    _per_zone_local_jac: Dict[int, JacobianSensitivities] = {}
    _per_zone_synth_shunt_map: Dict[int, Dict[int, int]] = {}
    _per_dso_local_jac: Dict[str, JacobianSensitivities] = {}
    _zone_promoted_oltcs: Dict[int, Tuple[int, ...]] = {}

    # Build a per-zone "all 2W trafos to keep in the reduced TSO net" list
    # once (used by both the initial build and any later rebuild on shunt
    # switching).  Includes:
    #   * every machine 2W trafo whose gen is in the zone (for the gen's
    #     terminal bus to survive the selective drop, INCLUDING the slack-
    #     gen's trafo which ``zone_defs[z].oltc_trafo_indices`` excludes);
    #   * every 2W trafo the TSO MIQP can act on
    #     (``zone_defs[z].oltc_trafo_indices``) — covers ``gen_idx=-1``
    #     tap-changing interconnect trafos whose LV bus would otherwise be
    #     dropped, leaving the controller's OLTC column count out of sync
    #     with the reduced Jacobian's columns.
    _zone_all_machine_trafos: Dict[int, List[int]] = {z: [] for z in zone_defs}
    _zone_gen_set: Dict[int, set] = {
        z: set(int(g) for g in zd.gen_indices) for z, zd in zone_defs.items()
    }
    for t_idx, g_idx in zip(meta.machine_trafo_indices, meta.machine_trafo_gen_map):
        if int(g_idx) < 0:
            continue
        for z in zone_defs:
            if int(g_idx) in _zone_gen_set[z]:
                _zone_all_machine_trafos[z].append(int(t_idx))
                break
    for z, zd in zone_defs.items():
        for t_idx in zd.oltc_trafo_indices:
            if int(t_idx) not in _zone_all_machine_trafos[z]:
                _zone_all_machine_trafos[z].append(int(t_idx))

    def _build_tso_local_jac(z: int):
        """Build per-zone reduced net + JacobianSensitivities.

        Returns
        -------
        (JacobianSensitivities, Dict[int,int], Tuple[int,...])
            Jacobian, {tertiary_bus -> primary_bus} synthetic shunt remap,
            and the tuple of OLTC trafo indices that need to be flagged
            OOS on the controller (because a promoted slack-gen sits on
            their bus, killing the dV/ds sensitivity column).
        """
        zd = zone_defs[z]
        result = build_tso_local_net(
            net=net,
            zone_bus_indices=tn_zone_map[z],
            gen_indices_in_zone=zd.gen_indices,
            machine_trafo_indices_in_zone=_zone_all_machine_trafos[z],
            tie_line_indices=zd.tie_line_indices,
            tie_line_endpoint_buses=zd.tie_line_endpoint_buses,
            hv_networks_in_zone=zone_hv_networks.get(z, []),
            tso_shunt_buses_in_zone=zd.shunt_bus_indices,
            tso_shunt_q_steps_mvar_in_zone=zd.shunt_q_steps_mvar,
            tie_boundary=getattr(config, "tie_boundary_equivalent", "pq"),
            tie_thevenin_k=getattr(config, "tie_thevenin_k", 1.0),
            verbose=verbose,
        )
        return (
            JacobianSensitivities(result.net),
            dict(result.synthetic_shunt_map),
            tuple(result.promoted_slack_oltc_indices),
        )

    def _apply_promoted_oos_oltc(
        ctrl: TSOController, promoted_oltcs: Tuple[int, ...]
    ) -> None:
        """Flag the promoted slack-gen's machine OLTC(s) as out-of-service
        on the controller so its sensitivity-matrix assembly skips the
        corresponding column.  Idempotent."""
        if not promoted_oltcs:
            return
        promoted_set = {int(t) for t in promoted_oltcs}
        for k, t in enumerate(ctrl.config.oltc_trafo_indices):
            if int(t) in promoted_set:
                ctrl._oos_oltc_mask[k] = True

    if config.local_sensitivities_tso:
        if verbose >= 1:
            print("  [local_sensitivities_tso=True] building per-zone reduced "
                  "TSO Jacobians ...")
        for z, ctrl in tso_controllers.items():
            _jz, _syn_map, _promoted_oltcs = _build_tso_local_jac(z)
            _per_zone_local_jac[z] = _jz
            _per_zone_synth_shunt_map[z] = _syn_map
            _zone_promoted_oltcs[z] = _promoted_oltcs
            ctrl.sensitivities = _jz
            ctrl.invalidate_sensitivity_cache()
            _apply_promoted_oos_oltc(ctrl, _promoted_oltcs)
            # Plumb the synthetic-shunt remapping into the controller
            # config so build_sensitivity_matrix_H reads shunt columns at
            # the synthetic primary bus rather than the (dropped) tertiary.
            if _syn_map:
                ctrl.config.shunt_sensitivity_bus_indices = [
                    _syn_map.get(int(b), int(b))
                    for b in ctrl.config.shunt_bus_indices
                ]
            else:
                ctrl.config.shunt_sensitivity_bus_indices = None
            if verbose >= 2:
                print(f"    Zone {z}: reduced net has "
                      f"{len(_jz.net.bus)} buses, "
                      f"{len(_jz.net.gen)} gens, "
                      f"{len(_jz.net.line)} lines, "
                      f"{len(_jz.net.shunt)} shunts; "
                      f"synthetic shunts: {len(_syn_map)}; "
                      f"promoted-slack OOS OLTCs: {_promoted_oltcs}")
    else:
        for ctrl in tso_controllers.values():
            ctrl.sensitivities = shared_jac
            ctrl.invalidate_sensitivity_cache()
            ctrl.config.shunt_sensitivity_bus_indices = None

    if config.local_sensitivities_dso:
        if verbose >= 1:
            print("  [local_sensitivities_dso=True] building per-DSO reduced "
                  "Jacobians ...")
        for dso_id, dso_ctrl in dso_controllers.items():
            hv = hv_info_map[dso_id]
            result = build_dso_local_net(net, hv, verbose=verbose)
            _jd = JacobianSensitivities(result.net)
            _per_dso_local_jac[dso_id] = _jd
            dso_ctrl.sensitivities = _jd
            dso_ctrl.invalidate_sensitivity_cache()
            if verbose >= 2:
                print(f"    {dso_id}: reduced net has "
                      f"{len(_jd.net.bus)} buses, "
                      f"{len(_jd.net.gen)} gens, "
                      f"{len(_jd.net.line)} lines, "
                      f"{len(_jd.net.shunt)} shunts")
    else:
        for dso_ctrl in dso_controllers.values():
            dso_ctrl.sensitivities = shared_jac
            dso_ctrl.invalidate_sensitivity_cache()

    if verbose >= 1:
        print(f"  [T] post-Phase-2 shared JacobianSensitivities rebuild + reassign: "
              f"{perf_counter() - _t:.2f} s")

    # Re-initialise all controllers so _u_current reflects the updated
    # operating point (profiles + correct tap positions).
    _t = perf_counter()
    for z, ctrl in tso_controllers.items():
        ctrl.initialise(_feedback_measurement(
            measure_zone_tso(net, zone_defs[z], 0),
            sample_id=("initialisation", 0),
            initialisation=True,
        ))
    for dso_id, dso_ctrl in dso_controllers.items():
        dso_ctrl.initialise(_feedback_measurement(
            measure_zone_dso(net, dso_ctrl.config, 0),
            sample_id=("initialisation", 0),
            initialisation=True,
        ))
    if _central and central_controller is not None:
        # The centralized controller always uses the FULL-network Jacobian
        # rebuilt at the post-Phase-2 operating point (overriding any
        # per-controller local-sensitivity reduction the loop above applied
        # to the recording-only TSO/DSO controllers).
        central_controller.sensitivities = shared_jac
        central_controller.invalidate_sensitivity_cache()
        central_controller.initialise(_feedback_measurement(
            measure_central(net, central_cfg, 0),
            sample_id=("initialisation", 0),
            initialisation=True,
        ))
        if getattr(config, "debug_central_curvature", False):
            _dump_central_curvature(central_controller, central_cfg)
    if verbose >= 1:
        print(f"  [T] controller .initialise() loop: {perf_counter() - _t:.2f} s")

    # ── Optional: pin a numerical (finite-difference) H matrix ─────────────
    # When ``config.numerical_h`` is True and local-sensitivity mode is
    # off, replace each controller's analytical H by a numerical one
    # computed by perturbing the plant net and reading the closed-loop
    # response.  Disable invalidation so the pinned H survives every
    # subsequent step (frozen baseline).  ``refresh_shared_jac_on_tso``
    # additionally re-pins the numerical H on every TSO tick.
    def _pin_numerical_h() -> None:
        for z, ctrl in tso_controllers.items():
            H_num = compute_numerical_h_tso(
                net, ctrl,
                closed_loop=config.numerical_h_closed_loop,
                verbose=verbose,
            )
            ctrl._H_cache = H_num
            ctrl._H_mappings = {}
        for dso_id, dso_ctrl in dso_controllers.items():
            H_num = compute_numerical_h_dso(
                net, dso_ctrl,
                closed_loop=config.numerical_h_closed_loop,
                verbose=verbose,
            )
            dso_ctrl._H_cache = H_num
            dso_ctrl._H_mappings = {}

    if (
        config.numerical_h
        and not config.local_sensitivities_tso
        and not config.local_sensitivities_dso
    ):
        _t = perf_counter()
        _pin_numerical_h()
        # Monkey-patch invalidation so subsequent code paths (contingency,
        # shunt-switch) cannot blow away the pinned cache.  Under
        # ``refresh_shared_jac_on_tso=True`` the main loop re-pins
        # explicitly via the same helper.
        def _noop_invalidate(self):
            return
        for ctrl in tso_controllers.values():
            ctrl.invalidate_sensitivity_cache = _noop_invalidate.__get__(ctrl)
        for dso_ctrl in dso_controllers.values():
            dso_ctrl.invalidate_sensitivity_cache = _noop_invalidate.__get__(dso_ctrl)
        if verbose >= 1:
            print(f"  [T] numerical H pin (frozen): {perf_counter() - _t:.2f} s")

    # ── Tier-2: curvature-based g_w preconditioning (optional) ───────────
    # Replace the continuous-class g_w with a column-norm-preconditioned,
    # auto-kappa vector targeting config.precondition_lambda_target, derived
    # from each controller's cached H.  Runs after init / numerical-H pin so
    # it preconditions against the H the MIQP will actually use.  No-op
    # unless config.precondition_g_w; leaves the BO/config path intact.
    if getattr(config, "precondition_g_w", False):
        _t = perf_counter()
        for z, ctrl in tso_controllers.items():
            _apply_gw_preconditioning(ctrl, config, f"TSO-z{z}", verbose)
        for dso_id, dso_ctrl in dso_controllers.items():
            _apply_gw_preconditioning(dso_ctrl, config, f"DSO-{dso_id}", verbose)
        if _central and central_controller is not None:
            _apply_gw_preconditioning(
                central_controller, config, "central", verbose,
            )
        if verbose >= 1:
            print(f"  [T] g_w preconditioning: {perf_counter() - _t:.2f} s")

    # ── Send initial DSO capability messages to TSO controllers ──────────
    # Without this, PCC capability bounds stay at the default ±1e-6 Mvar
    # until the first DSO step inside the loop.  The first TSO step then
    # sees near-zero capability and locks q_pcc; the second TSO step
    # (with real bounds) produces a large corrective jump.
    _t = perf_counter()
    for dso_id, dso_ctrl in dso_controllers.items():
        meas_init_dso = _feedback_measurement(
            measure_zone_dso(net, dso_ctrl.config, 0),
            sample_id=("initialisation", 0),
            initialisation=True,
        )
        tso_id = dso_to_tso_id[dso_id]
        cap_msg = dso_ctrl.generate_capability_message(
            target_controller_id=tso_id,
            measurement=meas_init_dso,
        )
        target_tso = next(
            ctrl for ctrl in tso_controllers.values()
            if ctrl.controller_id == tso_id
        )
        target_tso.receive_capability(cap_msg)
    if verbose >= 1:
        print(f"  [T] DSO capability messages loop: {perf_counter() - _t:.2f} s")
        for z, ctrl in tso_controllers.items():
            n_pcc = len(zone_defs[z].pcc_trafo_indices)
            if n_pcc > 0:
                print(f"  Zone {z}: initial PCC capability "
                      f"[{ctrl.pcc_capability_min_mvar[0]:.1f}, "
                      f"{ctrl.pcc_capability_max_mvar[0]:.1f}] Mvar")

    # ── Cross-sensitivity computation (needed by stability analysis) ──────
    # Reuse the same shared Jacobian to avoid yet another deep-copy + PF +
    # dense inversion inside the coordinator.  Under local TSO mode the
    # off-diagonal H_ij blocks are zeroed to stay consistent with each
    # TSO controller's restricted reduced-net view (no cross-zone
    # coupling assumed).
    _t = perf_counter()
    coordinator.compute_cross_sensitivities(
        jac=shared_jac,
        zero_offdiag=bool(config.local_sensitivities_tso),
    )
    if verbose >= 1:
        print(f"  [T] coordinator.compute_cross_sensitivities: {perf_counter() - _t:.2f} s")
    _t = perf_counter()
    coordinator.compute_M_blocks()
    if verbose >= 1:
        print(f"  [T] coordinator.compute_M_blocks: {perf_counter() - _t:.2f} s")

    _t = perf_counter()
    coordinator.check_contraction()
    if verbose >= 1:
        print(f"  [T] coordinator.check_contraction: {perf_counter() - _t:.2f} s")
        print(f"  [T] TOTAL init after [9]: {perf_counter() - _t_init_total:.2f} s")

    # Stability analysis is deferred until ``config.stability_analysis_at_s``
    # simulated seconds.  Running it at t=0 with an uncontrolled initial
    # operating point produces misleading curvature matrices.
    _stability_analysis_done = False

    # ── Optionally load tuned params from a previous run ────────────────
    # If ``config.load_tuned_params_path`` is set and points to a valid
    # JSON snapshot, apply those g_w / alpha values directly to the
    # controllers.  The delayed stability analysis still runs for
    # documentation.
    _tuned_params_loaded = False
    if config.load_tuned_params_path:
        if verbose >= 1:
            print(f"[11] Loading tuned params from "
                  f"{config.load_tuned_params_path} ...")
        try:
            _tuned_params_loaded = load_and_apply_tuned_params(
                config.load_tuned_params_path,
                zone_defs=zone_defs,
                tso_controllers=tso_controllers,
                dso_controllers=dso_controllers,
                verbose=verbose,
            )
        except Exception as _exc:
            if verbose >= 1:
                print(f"  ERROR loading tuned params: {_exc}")
            _tuned_params_loaded = False

    if contingencies and verbose >= 1:
        print(f"  Scheduled contingencies ({len(contingencies)}):")
        for ev in contingencies:
            t_label = f"t={ev.effective_time_s:.0f}s" if ev.time_s is not None else f"min {ev.minute}"
            print(f"    {t_label}: {ev.action} {ev.element_type}[{ev.element_index}]")


    # =========================================================================
    # STEP 12: Q-tracking capacity diagnostic
    # =========================================================================
    if verbose >= 2:
        print()
        print("[12] Q-tracking capacity diagnostic")
        # DER Q capacity (VDE-AR-N 4120 v2)
        for did, hv in hv_info_map.items():
            tot_qmin, tot_qmax, tot_qact = 0.0, 0.0, 0.0
            for idx in hv.sgen_indices:
                sn = net.sgen.at[idx, "sn_mva"]
                p_act = abs(net.res_sgen.at[idx, "p_mw"])
                p_ratio = p_act / sn if sn > 0 else 0
                if p_ratio < 0.1:
                    qmin, qmax = 0.0, 0.0
                elif p_ratio < 0.2:
                    t = (p_ratio - 0.1) / 0.1
                    qmin = (-0.10 + t * (-0.23)) * sn
                    qmax = ( 0.10 + t * ( 0.31)) * sn
                else:
                    qmin, qmax = -0.33 * sn, 0.41 * sn
                tot_qmin += qmin; tot_qmax += qmax
                tot_qact += net.res_sgen.at[idx, "q_mvar"]
            # Load Q
            q_load = net.res_load.loc[list(hv.load_indices), "q_mvar"].sum()
            p_load = net.res_load.loc[list(hv.load_indices), "p_mw"].sum()
            # Line Q losses
            q_line_loss = net.res_line.loc[list(hv.line_indices), "ql_mvar"].sum()
            # Interface Q
            q_iface = sum(net.res_trafo3w.at[t, "q_hv_mvar"]
                          for t in hv.coupling_trafo_indices)
            print(f"  {did}:")
            print(f"    DER Q capacity:  [{tot_qmin:+.0f}, {tot_qmax:+.0f}] Mvar  "
                  f"(actual: {tot_qact:+.1f} Mvar)")
            print(f"    Load Q:          {q_load:+.0f} Mvar  "
                  f"(P={p_load:.0f} MW)")
            print(f"    Line Q losses:   {q_line_loss:+.1f} Mvar")
            print(f"    Interface Q(HV): {q_iface:+.1f} Mvar")
            print(f"    Required DER Q for Q_iface=0: "
                  f"{q_load + q_line_loss:.0f} Mvar (to compensate loads+losses)")

        # IEEE 39-bus line lengths between coupling buses
        print()
        print("  Transmission line lengths between coupling buses:")
        for did, hv in hv_info_map.items():
            cbs = list(hv.coupling_ieee_buses)
            for i, b1 in enumerate(cbs):
                for b2 in cbs[i+1:]:
                    mask = ((net.line.from_bus == b1) & (net.line.to_bus == b2)) | \
                           ((net.line.from_bus == b2) & (net.line.to_bus == b1))
                    if mask.any():
                        for lidx in net.line.index[mask]:
                            L = net.line.at[lidx, "length_km"]
                            print(f"    {did}: TN line {b1}-{b2}: {L:.1f} km (345 kV)")
        print("  HV sub-network line lengths:")
        for did, hv in hv_info_map.items():
            lines = net.line.loc[list(hv.line_indices)]
            print(f"    {did} (scale={hv.line_length_scale}): "
                  f"range [{lines.length_km.min():.1f}, {lines.length_km.max():.1f}] km, "
                  f"total={lines.length_km.sum():.0f} km (110 kV)")


    # =========================================================================
    # STEP 13: Main simulation loop
    # =========================================================================
    if verbose >= 1:
        n_steps = int(config.n_total_s / config.dt_s)
        dur_str = f"start={start_time:%d.%m.%Y %H:%M}  " if use_profiles else ""
        print()
        warmup_str = f", warmup={config.warmup_s:.0f}s" if config.warmup_s > 0 else ""
        print(f"[13] Starting simulation: {n_steps} steps  "
              f"({dur_str}dt={config.dt_s:.0f}s, TSO/{config.tso_period_s/60:.0f}min, "
              f"DSO/{config.dso_period_s/60:.0f}min{warmup_str})")
        print()

    log: List[MultiTSOIterationRecord] = []

    # ── Voltage-stability / nose-curve reachability guard ────────────────────
    # At every step the converged plant equilibrium is checked against the
    # modal Q-V criterion (analysis.reachability).  The margin is recorded into
    # each MultiTSOIterationRecord so the full trajectory is available, and the
    # run aborts (ReachabilityViolation) at the first equilibrium that is not on
    # the stable upper voltage branch.  The monitor canonicalises the
    # distributed-slack Jacobian internally (single-slack re-converge on a deep
    # copy), so net is left untouched.
    _reach_monitor: Optional[ReachabilityMonitor] = (
        ReachabilityMonitor(
            tau_sigma=config.reach_tau_sigma,
            tau_eig=config.reach_tau_eig,
        )
        if config.enable_reachability_guard else None
    )
    if verbose >= 1 and _reach_monitor is not None:
        print(f"  [13] Reachability guard ON "
              f"(tau_sigma={config.reach_tau_sigma:.1e}, "
              f"tau_eig={config.reach_tau_eig:.1e})")

    # ── Optionally create live plot windows (three figures, 1/3 screen each) ─
    _plotter_tso = None
    _plotter_dso = None
    _plotter_sys = None
    _plotter_track = None

    if config.live_plot_controller:
        from visualisation.plot_tso_controller import TSOControllerLivePlotter
        _plotter_tso = TSOControllerLivePlotter(
            zone_ids=zone_ids_sorted,
            tie_line_pairs=sorted(tie_line_map.keys()),
            n_oltc_per_zone={z: len(zd.oltc_trafo_indices) for z, zd in zone_defs.items()},
            n_shunt_per_zone={
                # In integrator mode the banks are dispatched outside the MIQP,
                # so zd.shunt_bus_indices is empty; count the integrator banks
                # owned by the zone from meta instead.
                z: (
                    sum(1 for sz in meta.tso_tertiary_shunt_zones if sz == z)
                    if _shunt_mode == "integrator"
                    else len(getattr(zd, "shunt_bus_indices", []) or [])
                )
                for z, zd in zone_defs.items()
            },
            v_setpoint_pu=config.v_setpoint_pu,
            v_min_pu=0.9, v_max_pu=1.1,
            sub_minute=False, update_every=1, slot_idx=0,
            layout=config.live_plot_layout,
            show_line_currents=config.live_plot_show_line_currents,
            show_reserves=config.live_plot_show_reserves,
            show_tie_flows=config.live_plot_show_tie_flows,
            use_tex=config.live_plot_use_tex,
        )

    if config.live_plot_cascade and dso_ids:
        from visualisation.plot_cascade_dso import CascadeDSOLivePlotter
        _plotter_dso = CascadeDSOLivePlotter(
            dso_ids=dso_ids,
            v_setpoint_pu=config.v_setpoint_pu,
            v_min_pu=0.9, v_max_pu=1.1,
            sub_minute=False, update_every=1, slot_idx=1,
            layout=config.live_plot_layout,
            show_line_currents=config.live_plot_show_line_currents,
            use_tex=config.live_plot_use_tex,
        )

    if config.live_plot_system:
        from visualisation.plot_system_power_flow import SystemPowerFlowLivePlotter
        # Interface trafo IDs mirror the record's trafo_key convention.
        # In OFO mode keys are "{dso_id}|trafo_{idx}"; in local mode they fall
        # back to "{group_id}|trafo_{idx}".  Build the OFO form when DSO
        # controllers are present, else the local form.
        if dso_controllers:
            _interface_trafo_ids = [
                f"{did}|trafo_{t}"
                for did, ctrl in dso_controllers.items()
                for t in ctrl.config.interface_trafo_indices
            ]
        else:
            _interface_trafo_ids = [
                f"{hv.net_id}|trafo_{t}"
                for hv in meta.hv_networks
                for t in hv.coupling_trafo_indices
            ]
        _plotter_sys = SystemPowerFlowLivePlotter(
            zone_ids=zone_ids_sorted,
            dso_ids=dso_ids,
            interface_trafo_ids=_interface_trafo_ids,
            zone_gen_indices={z: list(zd.gen_indices) for z, zd in zone_defs.items()},
            gen_limits_static=gen_limits_static,
            sub_minute=False, update_every=1, slot_idx=2,
            layout=config.live_plot_layout,
            use_tex=config.live_plot_use_tex,
        )

    if config.live_plot_tracking:
        from visualisation.plot_tracking import TrackingLivePlotter
        # Per-DSO interface-trafo key map, matching the record's trafo_key
        # convention "{dso_id}|trafo_{idx}" (OFO controllers when present,
        # else the local-DSO HV-network fallback — see SystemPowerFlow above).
        if dso_controllers:
            _track_trafo_keys = {
                did: [f"{did}|trafo_{t}" for t in ctrl.config.interface_trafo_indices]
                for did, ctrl in dso_controllers.items()
            }
        else:
            _track_trafo_keys = {
                hv.net_id: [f"{hv.net_id}|trafo_{t}" for t in hv.coupling_trafo_indices]
                for hv in meta.hv_networks
            }
        # Tie-Q monitoring per zone-pair: the tracking plot shows the measured
        # inter-zone tie flow against a fixed 0 Mvar reference — tie-Q is
        # observed, not controlled (no commanded tie-flow setpoint exists).
        _plotter_track = TrackingLivePlotter(
            zone_ids=zone_ids_sorted,
            n_v_bus_per_zone={z: len(zd.v_bus_indices) for z, zd in zone_defs.items()},
            dso_ids=dso_ids,
            dso_trafo_keys=_track_trafo_keys,
            tie_line_pairs=sorted(tie_line_map.keys()),
            sub_minute=False, update_every=1, slot_idx=2,
            layout=config.live_plot_layout,
            use_tex=config.live_plot_use_tex,
        )

    # Figure 6 — SBX MECHANISM (gated; None when off).  Corridor keys are
    # discovered from the first record; the adapter handle is passed per
    # step (None until the contract-freeze tick).
    _plotter_sbx = None
    if config.live_plot_sbx:
        if config.coordination_mode != "sbx":
            raise ValueError(
                "live_plot_sbx=True requires coordination_mode='sbx_h' — "
                "the figure draws scheduled-voltage hold/sag support state."
            )
        from visualisation.plot_sbx import SBXMechanismLivePlotter
        _plotter_sbx = SBXMechanismLivePlotter(
            layout=config.live_plot_layout,
            use_tex=config.live_plot_use_tex,
        )
        # Handle for experiments (e.g. saving the final figure) — read
        # from the pre_loop_hook state alongside the adapter.
        sbx_runtime["live_plotter"] = _plotter_sbx

    # Figure 7 — SBX-V MECHANISM (gated; None when off).  The plotter
    # reads only adapter-side market/metering state and recorded plant
    # outputs; it never supplies information to either controller.
    _plotter_sbxv = None
    if config.live_plot_sbxv:
        if config.coordination_mode != "sbxv":
            raise ValueError(
                "live_plot_sbxv=True requires coordination_mode='sbx_v' — "
                "the figure draws vertical bands, grants and settlement."
            )
        from visualisation.plot_sbxv import SBXVMechanismLivePlotter
        _plotter_sbxv = SBXVMechanismLivePlotter(
            layout=config.live_plot_layout,
            use_tex=config.live_plot_use_tex,
        )
        sbxv_runtime["live_plotter"] = _plotter_sbxv

    def _is_period_hit(time_s: float, period_s: float) -> bool:
        """True if time_s is a multiple of period_s (within 1 s tolerance)."""
        rem = time_s % period_s
        return rem < 1.0 or abs(rem - period_s) < 1.0

    tso_step_count = 0  # count TSO steps for sensitivity refresh logic

    # ── g_z warmup: build target g_z vectors for the switch ──────────────
    _gz_warmup_done = (config.g_z_warmup_s <= 0)
    _gz_targets_tso: Dict[int, NDArray[np.float64]] = {}
    _gz_targets_dso: Dict[str, NDArray[np.float64]] = {}
    if not _gz_warmup_done:
        for z, zd in zone_defs.items():
            _gz_targets_tso[z] = np.concatenate([
                np.full(len(zd.v_bus_indices),     config.g_z_voltage),
                np.full(len(zd.pcc_trafo_indices), config.g_z_q_pcc),
                np.full(len(zd.line_indices),      config.g_z_current),
                np.full(len(zd.gen_indices),       config.g_z_q_gen),
            ])
        for dso_id_tmp, dso_ctrl_tmp in dso_controllers.items():
            cfg_tmp = dso_ctrl_tmp.config
            n_iface_tmp = len(cfg_tmp.interface_trafo_indices)
            n_v_tmp = len(cfg_tmp.voltage_bus_indices)
            n_i_tmp = len(cfg_tmp.current_line_indices)
            _gz_targets_dso[dso_id_tmp] = np.concatenate([
                np.full(n_iface_tmp, config.g_z_interface),
                np.full(n_v_tmp,     config.g_z_voltage),
                np.full(n_i_tmp,     config.g_z_current),
            ])

    n_steps = int(config.n_total_s / config.dt_s)

    # ── TSO local-mode setup (one-shot, before main loop) ─────────────────
    # When the OFO TSO controller is skipped, two pieces keep TSO-side
    # primary voltage control alive (the windpark sgens already have
    # QVLocalLoop / CosPhiConstLoop installed by ``install_der_q_loops``
    # in step [10.3]):
    #
    #   (1) Generator AVR setpoints pinned to ``config.v_setpoint_pu``
    #       (1.03 pu by default).  Without OFO, nothing else writes
    #       net.gen.vm_pu, but we re-pin defensively in case a profile
    #       update touches it.
    #   (2) DiscreteTapControl on every machine 2W trafo, V_target =
    #       v_setpoint_pu, controlling the HV (grid) side.  These are
    #       the same controllers used in the Phase 1 OLTC init at
    #       lines ~1323; they were dropped after that init phase but
    #       must be re-installed to stay active for the simulation.
    _tso_der_idx_list: List[int] = [int(s) for s in meta.tso_der_indices]
    if _local_tso:
        # (1) Pin generator AVR setpoints
        net.gen.loc[:, "vm_pu"] = float(config.v_setpoint_pu)
        if verbose >= 1:
            print(f"  [local TSO] Pinned net.gen.vm_pu = {config.v_setpoint_pu:.3f} "
                  f"on {len(net.gen)} synchronous machines")

        # (2) Machine 2W OLTC DiscreteTapControl, HV side -> v_setpoint_pu
        _mt_tol_pu = config.dso_oltc_init_tol_pu
        for _tidx in meta.machine_trafo_indices:
            DiscreteTapControl(
                net, element_index=int(_tidx),
                vm_lower_pu=config.v_setpoint_pu - _mt_tol_pu,
                vm_upper_pu=config.v_setpoint_pu + _mt_tol_pu,
                side="hv", element="trafo",
            )
        if verbose >= 1:
            print(f"  [local TSO] Re-installed DiscreteTapControl on "
                  f"{len(meta.machine_trafo_indices)} machine 2W trafos "
                  f"(target HV side = {config.v_setpoint_pu:.3f} +/- "
                  f"{_mt_tol_pu:.3f} p.u.)")

    # ── Persistent OLTC rate-limiter for local-control mode ──────────────
    # Wraps the existing per-step ±max_step clamp with a wall-clock
    # cooldown (config.oltc_cooldown_s) so each DiscreteTapControl-managed
    # OLTC is locked for that many seconds after every actual tap
    # movement.  Only active when at least one local-mode tap controller
    # is present; the OFO MIQP path enforces the same cooldown via
    # OFOParameters.int_cooldown_s on its own integer block.
    # Central mode owns every OLTC through the single MIQP (wall-clock
    # cooldown via OFOParameters.int_cooldown_s), so the local-mode
    # DiscreteTapControl rate-limiter is disabled there.
    _oltc_local_active = (_local_dso or _local_tso) and not _central
    _oltc_limiter = _OLTCRateLimiter(
        max_step=config.local_oltc_max_step_per_dt,
        cooldown_s=(config.oltc_cooldown_s if _oltc_local_active else 0.0),
        cooldown_by_table=(
            {
                # MT = machine 2W gen-transformer OLTCs (net.trafo);
                # NC = coupler 3W OLTCs (net.trafo3w) at the TS--STS interface.
                "trafo": (config.oltc_cooldown_s_mt
                          if config.oltc_cooldown_s_mt is not None
                          else config.oltc_cooldown_s),
                "trafo3w": (config.oltc_cooldown_s_nc
                            if config.oltc_cooldown_s_nc is not None
                            else config.oltc_cooldown_s),
            }
            if _oltc_local_active else {}
        ),
    )
    _oltc_limiter_active = _oltc_local_active and _oltc_limiter.active

    # ── SBX-V adapter construction (before the loop: metering and the
    # PricingSolver proxies must be in place from t = 0) ─────────────────
    if config.coordination_mode == "sbxv":
        from sbx_v.adapter import SBXVRunnerAdapter
        sbxv_runtime["adapter"] = SBXVRunnerAdapter(
            sbxv_runtime["config"], tso_controllers,
            tso_period_s=float(config.tso_period_s),
            net=net,
        )
        if verbose >= 1:
            _sv = sbxv_runtime["config"]
            _ad = sbxv_runtime["adapter"]
            print(f"  [sbxv] adapter armed: {len(_ad.areas)} aggregation "
                  f"area(s), preset '{_sv.band_preset}', window "
                  f"{_sv.window_s:.0f} s = {_sv.k_window} TSO "
                  f"iterations, quantum {_sv.dq_grant_mvar:.0f} Mvar")
            for _a in sorted(_ad.bands):
                _b = _ad.bands[_a]
                print(f"  [sbxv]   {_a}: Normalbereich raising "
                      f"{_b.q_raise_mvar:.1f} / lowering "
                      f"{_b.q_lower_mvar:.1f} Mvar")

    # ── Per-area, per-class g_w overrides ─────────────────────────────────────
    # Absolute weights, one level finer than ``zone_g_w_scale``: that knob has a
    # single number per area and cannot express an area whose classes want to
    # move in opposite directions, which is what the analytic per-column design
    # actually asks for (``tuning_mc.stage_0_preconditioning --per-area``).
    #
    # Applied BEFORE ``zone_g_w_scale`` so the scale, if also set, multiplies
    # these rather than being overwritten by them.
    #
    # Both ``params.g_w`` and ``_g_w_vector_cache`` are written.  This is not
    # belt-and-braces: ``build_miqp_problem`` takes ``g_w_vector`` in preference
    # to ``g_w`` whenever the cache is non-None, so writing only ``params.g_w``
    # (which is all ``zone_g_w_scale`` below does) is silently ignored on any
    # controller that has a DER mapping.  Mirrors
    # ``BaseOFOController.apply_preconditioned_g_w``.
    def _apply_class_g_w(_ctrl, _spec: dict, _tag: str) -> None:
        _gw = getattr(getattr(_ctrl, "params", None), "g_w", None)
        if _gw is None or not _spec:
            return
        _vec = np.broadcast_to(np.asarray(_gw, dtype=float),
                               (_ctrl.n_controls,)).copy()
        _cls_map = _ctrl._actuator_class_indices()
        _hit, _miss = [], []
        for _cls, _val in _spec.items():
            _idx = _cls_map.get(str(_cls))
            if _idx is None or np.size(_idx) == 0:
                _miss.append(str(_cls))
                continue
            _v = float(_val)
            if not (_v > 0.0):
                raise ValueError(
                    f"g_w must be strictly positive; {_tag} class "
                    f"{_cls!r} got {_v!r}"
                )
            _vec[np.asarray(_idx, dtype=int)] = _v
            _hit.append(f"{_cls}={_v:g}")
        _ctrl.params = dataclasses.replace(_ctrl.params, g_w=_vec)
        if getattr(_ctrl, "_g_w_vector_cache", None) is not None:
            _ctrl._g_w_vector_cache = _vec.copy()
        if verbose >= 1 and _hit:
            print(f"  [g_w_class] {_tag}: {'  '.join(_hit)}")
        if _miss:
            print(f"  [g_w_class] WARNING {_tag}: no such actuator class "
                  f"{_miss} -- override ignored (known: "
                  f"{sorted(_cls_map)})")

    if getattr(config, "zone_g_w_class", None):
        for _z, _ctrl in tso_controllers.items():
            _spec = config.zone_g_w_class.get(int(_z))
            if _spec:
                _apply_class_g_w(_ctrl, _spec, f"zone {_z}")
    if getattr(config, "dso_g_w_class", None):
        for _d, _ctrl in dso_controllers.items():
            _spec = config.dso_g_w_class.get(str(_d))
            if _spec:
                _apply_class_g_w(_ctrl, _spec, f"{_d}")

    # ── Per-DSO voltage-tracking weight ──────────────────────────────────────
    # Applied here, next to dso_g_w_class, because the two are a pair: with
    # dso_gamma_oltc_q = 0 the DSO OLTC sees only the voltage gradient, so
    # dso_g_v / g_w_dso_oltc is that area's OLTC loop gain and the two must be
    # moved together (see MultiTSOConfig.dso_g_v_per_area).  Warn if a caller
    # raises g_v without matching the OLTC weight -- that is the configuration
    # that limit-cycles the tap.
    if getattr(config, "dso_g_v_per_area", None):
        for _d, _ctrl in dso_controllers.items():
            _gv = config.dso_g_v_per_area.get(str(_d))
            if _gv is None:
                continue
            _gv = float(_gv)
            if not (_gv >= 0.0):
                raise ValueError(
                    f"dso_g_v_per_area[{_d!r}] must be >= 0, got {_gv!r}"
                )
            _gv_old = float(_ctrl.config.g_v)
            _ctrl.config.g_v = _gv
            # Invalidate any cached objective/curvature state derived from g_v.
            _ctrl._H_cache = getattr(_ctrl, "_H_cache", None)
            if verbose >= 1:
                print(f"  [dso_g_v] {_d}: {_gv_old:g} -> {_gv:g}")
            # Loop-gain check against this area's OLTC weight.
            _ratio_gv = _gv / _gv_old if _gv_old > 0 else float("inf")
            _spec = (config.dso_g_w_class or {}).get(str(_d)) or {}
            _oltc_new = _spec.get("dso_oltc")
            _oltc_old = float(config.g_w_dso_oltc)
            _ratio_gw = (float(_oltc_new) / _oltc_old
                         if _oltc_new and _oltc_old > 0 else 1.0)
            if _ratio_gv > 1.0 and not (0.8 <= _ratio_gw / _ratio_gv <= 1.25):
                print(
                    f"  [dso_g_v] WARNING {_d}: g_v raised x{_ratio_gv:.2f} but "
                    f"g_w_dso_oltc only x{_ratio_gw:.2f}.  With "
                    f"dso_gamma_oltc_q={config.dso_gamma_oltc_q:g} the OLTC is "
                    f"voltage-driven only, so this raises its loop gain "
                    f"x{_ratio_gv / _ratio_gw:.2f} and the integer tap may "
                    f"limit-cycle.  Set dso_g_w_class[{_d!r}]['dso_oltc'] = "
                    f"{_oltc_old * _ratio_gv:.0f} to hold it."
                )

    # ── Per-DSO interface-Q weight ───────────────────────────────────────────
    # Third leg of the relief, written by ``apply_dso_v_relief(scale_q=True)``.
    # Holding dso_g_v / g_w_dso_oltc preserves the OLTC's VOLTAGE threshold; the
    # factor on g_w_dso_oltc is uncompensated in its INTERFACE-Q threshold, so
    # without this a relieved area is Q-inert whenever dso_gamma_oltc_q > 0
    # (measured 2026-08-20: 108-244 Mvar to commit, against ~6 Mvar of RMSE).
    # Applied in the same place and the same way as dso_g_v_per_area so the two
    # cannot drift apart.
    if getattr(config, "dso_g_q_per_area", None):
        if float(getattr(config, "dso_gamma_oltc_q", 0.0)) <= 0.0 and verbose >= 1:
            print(
                "  [dso_g_q] NOTE: dso_g_q_per_area is set but "
                "dso_gamma_oltc_q = 0, so the DSO OLTC carries no Q gradient "
                "and this changes the tap not at all.  It still moves the "
                "continuous DSO-DER block."
            )
        for _d, _ctrl in dso_controllers.items():
            _gq = config.dso_g_q_per_area.get(str(_d))
            if _gq is None:
                continue
            _gq = float(_gq)
            if not (_gq >= 0.0):
                raise ValueError(
                    f"dso_g_q_per_area[{_d!r}] must be >= 0, got {_gq!r}"
                )
            _gq_old = float(_ctrl.config.g_q)
            _ctrl.config.g_q = _gq
            # Same cache invalidation as the g_v leg: the interface-Q curvature
            # H_Q G_w^-1 H_Q^T diag(g_q) depends on it.
            _ctrl._H_cache = getattr(_ctrl, "_H_cache", None)
            if verbose >= 1:
                print(f"  [dso_g_q] {_d}: {_gq_old:g} -> {_gq:g}")

    # ── Per-DSO OLTC Q-gain (gamma) ──────────────────────────────────────────
    # The role-targeted counterpart of the two blocks above: gamma multiplies
    # ONLY the OLTC columns of dQ/du, so it changes the tap's interface-Q
    # commit threshold without touching the continuous DER block.  That is why
    # it is preferred over dso_g_q_per_area for tap behaviour -- see
    # MultiTSOConfig.dso_gamma_oltc_q_per_area.
    #
    # Applied last of the three, and still before the first step, so a config
    # that sets several of them gets one consistent controller.
    if getattr(config, "dso_gamma_oltc_q_per_area", None):
        from controller.dso_controller import GAMMA_OLTC_Q_MAX
        for _d, _ctrl in dso_controllers.items():
            _gam = config.dso_gamma_oltc_q_per_area.get(str(_d))
            if _gam is None:
                continue
            _gam = float(_gam)
            if not (0.0 <= _gam <= GAMMA_OLTC_Q_MAX):
                raise ValueError(
                    f"dso_gamma_oltc_q_per_area[{_d!r}] must be in "
                    f"[0, {GAMMA_OLTC_Q_MAX:g}], got {_gam!r}"
                )
            _gam_old = float(_ctrl.config.gamma_oltc_q)
            _ctrl.config.gamma_oltc_q = _gam
            # gamma enters the objective curvature through the OLTC columns of
            # H, so any cached curvature derived from it is stale.
            _ctrl._H_cache = getattr(_ctrl, "_H_cache", None)
            if verbose >= 1:
                print(f"  [dso_gamma] {_d}: {_gam_old:g} -> {_gam:g}")

    # ── Per-zone loop-gain scaling ────────────────────────────────────────────
    # Applied here, after every controller exists and before the first step, so
    # the whole run sees one consistent gain.  ``OFOParameters`` is frozen, so
    # the vector is rebuilt rather than mutated.  A change of boundary
    # equivalent rescales H by a different factor in each area, and this is the
    # knob that lets each area be re-gained to match.
    if getattr(config, "zone_g_w_scale", None):
        for _z, _ctrl in tso_controllers.items():
            _s = float(config.zone_g_w_scale.get(int(_z), 1.0))
            _gw = getattr(getattr(_ctrl, "params", None), "g_w", None)
            if _s == 1.0 or _gw is None:
                continue
            _scaled = (np.asarray(_gw, dtype=float) * _s
                       if np.ndim(_gw) else float(_gw) * _s)
            _ctrl.params = dataclasses.replace(_ctrl.params, g_w=_scaled)
            # Same reason as _apply_class_g_w above: a non-None cache shadows
            # params.g_w inside build_miqp_problem, so the scale would be a
            # silent no-op on a controller with a DER mapping.
            if getattr(_ctrl, "_g_w_vector_cache", None) is not None:
                _ctrl._g_w_vector_cache = np.broadcast_to(
                    np.asarray(_scaled, dtype=float),
                    (_ctrl.n_controls,)).copy()
            if verbose >= 1:
                print(f"  [zone_g_w_scale] zone {_z}: g_w x {_s:g}")

    if pre_loop_hook is not None:
        _hook_state = {
            "net": net,
            "meta": meta,
            "tso_controllers": tso_controllers,
            "dso_controllers": dso_controllers,
            "shared_jac": shared_jac,
            "dso_to_tso_id": dso_to_tso_id,
            "zone_defs": zone_defs,
            "coordinator": coordinator,
            "config": config,
            # SBX internals ({"adapter": None, ...} unless
            # coordination_mode="sbx"; the adapter is constructed at the
            # first TSO tick after sbx_warmup_s and filled in here).
            "sbx_runtime": sbx_runtime,
            # SBX-V internals ({"adapter": None, ...} unless
            # coordination_mode="sbxv"; constructed above, before the
            # loop — call adapter.finalise() after the run for the
            # settlement plane).
            "sbxv_runtime": sbxv_runtime,
        }
        _hook_result = pre_loop_hook(_hook_state)
        if _hook_result:
            if verbose >= 1:
                print("[pre_loop_hook] returned truthy -- skipping main loop.")
            return []

    # ── Build per-zone switched-shunt integrators (integrator mode only) ──────
    # Each MSC / MSR bank is dispatched by the integrating mechanism OUTSIDE the
    # MIQP.  ``q_itf_sh_offset`` is the persistent per-interface feedforward
    # offset [Mvar] added to the Q_PCC setpoints sent to the DSO, so the DSO
    # does not counteract a committed switch (it then rejects only the residual).
    zone_integrators: Dict[int, ShuntIntegrator] = {}
    q_itf_sh_offset: Dict[int, float] = {}
    if _shunt_mode == "integrator":
        _banks_by_zone: Dict[int, List[ShuntBankConfig]] = {z: [] for z in zone_defs}
        for sh_idx, sb, q_step, sz, kind, nlev in zip(
            meta.tso_tertiary_shunt_indices,
            meta.tso_tertiary_shunt_buses,
            meta.tso_tertiary_shunt_q_steps_mvar,
            meta.tso_tertiary_shunt_zones,
            meta.tso_tertiary_shunt_kinds,
            meta.tso_tertiary_shunt_n_levels,
        ):
            dso_id_b = shunt_bus_to_dso_id.get(int(sb))
            hv_b = hv_info_map.get(dso_id_b) if dso_id_b is not None else None
            if hv_b is None or not hv_b.coupling_trafo_indices:
                raise ValueError(
                    f"Cannot resolve interface 3W transformer / DSO for tertiary "
                    f"shunt at bus {sb} (dso_id={dso_id_b})"
                )
            interface_t3w = int(hv_b.coupling_trafo_indices[0])
            if sz not in _banks_by_zone:
                continue
            _banks_by_zone[sz].append(
                ShuntBankConfig(
                    shunt_idx=int(sh_idx),
                    bus_idx=int(sb),
                    interface_trafo3w_idx=interface_t3w,
                    dso_id=str(dso_id_b),
                    kind=str(kind),
                    q_step_mvar=float(q_step),
                    n_levels=int(nlev),
                    g_w=float(config.shunt_int_g_w),
                    delta=float(config.shunt_int_delta_mvar),
                    t_dwell_s=float(config.shunt_int_t_dwell_s),
                    daily_switch_budget=int(config.shunt_int_daily_budget),
                    y_h_min=float(config.shunt_int_v_min_pu),
                    y_h_max=float(config.shunt_int_v_max_pu),
                )
            )
            q_itf_sh_offset.setdefault(interface_t3w, 0.0)
        for z_b, bclist in _banks_by_zone.items():
            if bclist:
                zone_integrators[z_b] = ShuntIntegrator.from_configs(bclist)
        if verbose >= 1:
            _n_banks = sum(len(i.banks) for i in zone_integrators.values())
            print(f"  [shunt-integrator] built {_n_banks} MSC/MSR banks "
                  f"across {len(zone_integrators)} zone(s).")

    # Features below that still act on ``net`` directly (recovery ladders
    # included) are only correct for the quasi-static plant; fail fast for
    # any other plant rather than run a silently inconsistent co-simulation.
    if not isinstance(plant, PandapowerStaticPlant):
        # Contingencies are no longer wholesale unsupported: a non-static plant
        # that can translate an event into a simulation event may run it.  Only
        # the event TYPES the plant cannot deliver are refused, so an N-1
        # machine trip works while a line trip or a restore -- which would put
        # the mirror and the simulator on different topologies -- still fails
        # fast.
        _supports_ctg = getattr(plant, "supports_contingency", None)
        _bad_ctg = sorted({
            f"{ev.element_type}/{ev.action}" for ev in contingencies
            if _supports_ctg is None or not _supports_ctg(ev)
        })
        _unsupported = [
            label for cond, label in (
                (bool(_bad_ctg),
                 "contingency events of type " + ", ".join(_bad_ctg)),
                (_oltc_limiter_active, "local OLTC tap-rate limiter"),
                # Profiles are supported since 2026-07-21 via
                # Plant.apply_exogenous (EvtLod for loads, EvtParam on the
                # WECC Pref_in for DER P), so they are no longer listed here.
                (gen_dispatch is not None,
                 "zonal generator dispatch schedule (still writes net "
                 "directly)"),
            ) if cond
        ]
        if _unsupported:
            raise NotImplementedError(
                "non-static plant does not support: " + ", ".join(_unsupported)
            )

    for step in range(1, n_steps + 1):
        _t_step = perf_counter()
        time_s  = step * config.dt_s
        # Seconds already advanced this interval by the RMS profile pre-settle
        # (see the profile branch); subtracted from the end-of-step advance so
        # the total stays dt_s.  0 unless profiles + a non-static plant.
        _profile_settle_s = 0.0
        run_tso = (step == 1) or _is_period_hit(time_s, config.tso_period_s)
        run_dso = _is_period_hit(time_s, config.dso_period_s)
        run_central = _central and (
            (step == 1) or _is_period_hit(time_s, _central_period_s)
        )
        _in_warmup = time_s <= config.warmup_s
        # Track whether anything wrote new actuator commands this step.
        # Used to decide whether the end-of-step PF is needed: if no
        # MIQP fired and no contingency was applied, the post-profile
        # PF already reflects the final state.
        _contingency_fired_this_step = False

        # ── g_z warmup → activate output constraints ─────────────────────
        if not _gz_warmup_done and time_s >= config.g_z_warmup_s:
            _gz_warmup_done = True
            for z, ctrl in tso_controllers.items():
                ctrl.update_g_z(_gz_targets_tso[z])
            for did, dctrl in dso_controllers.items():
                dctrl.update_g_z(_gz_targets_dso[did])
            if verbose >= 1:
                print(f"  -- g_z warmup complete at t={time_s:.0f}s: "
                      f"output constraints activated "
                      f"(g_z_voltage={config.g_z_voltage:.0e}) --")

        rec = MultiTSOIterationRecord(
            step=step, time_s=time_s, tso_active=run_tso, dso_active=run_dso
        )
        # Preserve the exact noisy packets presented to each controller.
        # They are recorded separately from the post-control plant truth.
        tso_plot_measurements: Dict[int, Measurement] = {}
        dso_plot_measurements: Dict[str, Measurement] = {}


        # ── Local-mode OLTC rate-limit snapshot ──────────────────────────────
        # Snapshot every DiscreteTapControl tap_pos at the start of the
        # step so plant PFs in the step (post-profile, post-contingency,
        # end-of-step) can be clamped to ±config.local_oltc_max_step_per_dt
        # of these values AND blocked from moving twice within
        # config.oltc_cooldown_s seconds.  Only relevant when local-mode
        # tap controllers are present (cascade-DSO local mode and/or TSO
        # local mode); the OFO MIQP path manages its OLTCs via
        # int_max_step / int_cooldown / int_cooldown_s in the controller.
        if _oltc_limiter_active:
            _oltc_limiter.snapshot(net)

        # ── Apply time-series profiles ────────────────────────────────────────
        if use_profiles and profiles is not None:
            t_now = start_time + timedelta(seconds=time_s)
            # Through the plant, not the net: a non-static plant must turn
            # the profile step into simulation events (PF reads element
            # input data only at initialisation).  The static plant's
            # implementation is exactly the previous apply_profiles(net,...).
            plant.apply_exogenous(profiles, t_now)
            if gen_dispatch is not None:
                apply_gen_dispatch(net, gen_dispatch, t_now)
            # Converge PF so that measurements (q_pcc, voltages) reflect the
            # new profiles/dispatch BEFORE controllers read them.
            # Warm-start the QVLocalLoops with the linear closed-loop
            # equilibrium so the run_control iteration only has to refine
            # the nonlinear residual.  Bypasses the multi-DER Gauss-Jacobi
            # coupling instability.
            # ToDo: pre seed currently disabled
            # seed_qv_equilibrium(
            #     net,
            #     list(meta.tso_der_indices) + list(meta.dso_der_indices),
            #     shared_jac,
            # )
            # Make the profile visible to the controllers before they read.
            # Static plant: advance() is an instant re-solve, so this settles
            # to the profiled steady state.  RMS plant: advance() is real
            # time, and apply_exogenous scheduled the profile as events at
            # t+eps -- advancing ``rms_profile_settle_s`` here fires them and
            # lets the state settle so the controllers read POST-profile;
            # the end-of-step advance below then runs the remaining
            # ``dt_s - settle`` (total = dt_s, clock unchanged).  At settle=0
            # the RMS controllers read the pre-profile state (the one-interval
            # lag that seeded the DSO_4 runaway, 2026-07-22).  Running the
            # full dt_s here would advance the RMS plant TWICE (2x clock).
            if isinstance(plant, PandapowerStaticPlant):
                plant.advance(config.dt_s)   # re-solve to reflect the profile
                plant.read_y()               # refresh the measurement image
            elif config.rms_profile_settle_s > 0.0:
                _profile_settle_s = min(config.rms_profile_settle_s,
                                        config.dt_s - config.dt_s * 0.1)
                plant.advance(_profile_settle_s)
                plant.read_y()
            if _oltc_limiter_active:
                _moved = _oltc_limiter.clamp(net, time_s)
                if _moved:
                    if verbose >= 1:
                        _pretty = ", ".join(
                            f"{tab}#{tid} {prev:+d}->{new:+d}"
                            for tab, tid, prev, new in _moved
                        )
                        print(
                            f"  [Step {step}] post-profile OLTC tap-rate limit "
                            f"({len(_moved)}): {_pretty}; re-running PF "
                            f"with run_control=False..."
                        )
                    pp.runpp(
                        net, run_control=False, calculate_voltage_angles=True,
                        max_iteration=100,
                        distributed_slack=config.distributed_slack,
                        enforce_q_lims=config.enforce_q_lims_plant,
                    )

        # ── Apply contingency events ──────────────────────────────────────────
        if contingencies:
            fired = [
                ev for ev in contingencies
                if abs(ev.effective_time_s - time_s) < 1e-9
            ]
            if fired:
                # Collect grid-side neighbourhood of any gen-trip event, BEFORE
                # applying the trip (after the trip, gen_trafo_map's trafo may
                # be OOS but still has hv_bus / lv_bus — still usable).
                watch_buses = _collect_contingency_watch_buses(
                    net, fired, gen_trafo_map
                )

                if verbose > 1:
                    _dump_contingency_diagnostics(
                        net, label=f"PRE-TRIP t={time_s:.0f}s",
                        watch_bus_0idx=watch_buses,
                    )

                _contingency_fired_this_step = True
                for ev in fired:
                    _apply_contingency(net, ev, verbose,
                                       gen_trafo_map=gen_trafo_map)
                    # Mirror updated above; now the plant itself.  For the
                    # quasi-static plant ``net`` IS the plant and this is a
                    # no-op; a non-static plant must translate the event,
                    # because PowerFactory reads element input attributes only
                    # at initialisation -- without this the simulator would
                    # stay on the pre-contingency topology while the mirror
                    # showed the new one.
                    plant.apply_contingency(ev, gen_trafo_map=gen_trafo_map)

                # Re-converge PF with new topology so measurements
                # reflect the post-contingency operating point.
                # Re-seed the QVLocalLoops with the analytical closed-loop
                # equilibrium at the post-contingency operating point.
                # Without this the per-DER damped iteration starts from
                # the pre-contingency Q values and can hit the 501-
                # iteration controller cap on severe topology changes
                # (gen+trafo trips).  Mirrors the post-profile seed at the
                # top of the step.
                # ToDo: pre seed currently disabled
                # seed_qv_equilibrium(
                #     net,
                #     list(meta.tso_der_indices) + list(meta.dso_der_indices),
                #     shared_jac,
                # )
                try:
                    # First attempt: no Q-limit enforcement so that gens
                    # transiently producing Q outside their static box
                    # immediately after a topology change (gen trip) can
                    # converge in PV mode rather than cascading into
                    # PV→PQ flips that stall Newton-Raphson.  The
                    # subsequent end-of-step PF (with
                    # enforce_q_lims=config.enforce_q_lims_plant) clamps
                    # any out-of-box Q on the same step.  Keeping the
                    # asymmetry preserves the legacy retry path below
                    # (which adds enforce_q_lims=True as a recovery
                    # action when this unclipped attempt diverges).
                    # Only the quasi-static plant may be re-solved here.  For a
                    # non-static plant ``net`` is the measurement MIRROR: a
                    # pandapower solve would overwrite the PowerFactory
                    # measurements with a static solution, and every reading
                    # afterwards would silently be from the wrong plant.  The
                    # RMS plant has been handed an EvtOutage instead, and the
                    # next advance()/read_y() produces the post-contingency
                    # measurements from the simulator itself.
                    if isinstance(plant, PandapowerStaticPlant):
                        pp.runpp(net, run_control=_run_control,
                                 calculate_voltage_angles=True,
                                 max_iteration=50,
                                 max_iter=300,
                                 distributed_slack=config.distributed_slack)
                    pf_converged = True
                except LoadflowNotConverged:
                    pf_converged = False
                    print("\n  *** Post-contingency PF did NOT converge "
                          "with default settings. ***")
                    print("  *** Running pp.diagnostic() to identify "
                          "topology / balance issues ***\n")
                    try:
                        pp.diagnostic(net, report_style="compact")
                    except Exception as exc:
                        print(f"  pp.diagnostic failed: {exc}")

                    print("\n  *** Retrying PF with enforce_q_lims=True, "
                          "init='flat', max_iteration=100 ***\n")
                    try:
                        pp.runpp(net, run_control=_run_control,
                                 calculate_voltage_angles=True,
                                 max_iteration=100,
                                 max_iter=300,
                                 distributed_slack=config.distributed_slack,
                                 enforce_q_lims=True,
                                 init="flat")
                        pf_converged = True
                        print("  *** Retry converged "
                              "→ original failure is Q-limit / warm-start "
                              "related. ***\n")
                    except LoadflowNotConverged:
                        print("  *** Retry with enforce_q_lims + flat start "
                              "ALSO diverged — structural issue. ***\n")
                        if verbose > 1:
                        # Dump what diagnostics we can without res_* tables
                            _dump_contingency_diagnostics(
                                net, label=f"POST-TRIP FAILED t={time_s:.0f}s",
                                watch_bus_0idx=watch_buses,
                            )
                        raise
                if verbose > 1:
                    _dump_contingency_diagnostics(
                        net, label=f"POST-TRIP t={time_s:.0f}s",
                        watch_bus_0idx=watch_buses,
                    )

                if _oltc_limiter_active:
                    _moved = _oltc_limiter.clamp(net, time_s)
                    if _moved:
                        if verbose >= 1:
                            _pretty = ", ".join(
                                f"{tab}#{tid} {prev:+d}->{new:+d}"
                                for tab, tid, prev, new in _moved
                            )
                            print(
                                f"  [Step {step}] post-contingency OLTC "
                                f"tap-rate limit ({len(_moved)}): {_pretty}; "
                                f"re-running PF with run_control=False..."
                            )
                        # Match the post-contingency PF's enforce_q_lims=False
                        # behaviour to keep convergence robust on the
                        # transient post-trip operating point.
                        pp.runpp(
                            net, run_control=False,
                            calculate_voltage_angles=True,
                            max_iteration=200,
                            distributed_slack=config.distributed_slack,
                            enforce_q_lims=False,
                        )

                # Notify controllers: freeze OOS actuator bounds, zero
                # their H columns, and invalidate sensitivity caches.
                coordinator.update_outage_masks(net)
                coordinator.invalidate_sensitivity_cache()

                # The centralized controller is not registered with the
                # coordinator, so mask its OOS generators / 2W machine OLTCs
                # directly (mirrors coordinator.update_outage_masks).  Its
                # cached H over the frozen shared_jac is otherwise unchanged
                # — the same "controllers know the plant only through cached
                # sensitivities" assumption as the distributed variants.
                if _central and central_controller is not None:
                    _oos_gen = {
                        int(g) for g in central_cfg.gen_indices
                        if not bool(net.gen.at[g, "in_service"])
                    }
                    _oos_oltc = {
                        int(t) for t in central_cfg.oltc_trafo_indices
                        if not bool(net.trafo.at[t, "in_service"])
                    }
                    central_controller.update_outage_mask(_oos_gen, _oos_oltc)
                    central_controller.invalidate_sensitivity_cache()

                # Under local-net sensitivity mode the controllers'
                # reduced Jacobians are intentionally frozen at the
                # pre-contingency cached operating point — that's the
                # decentralised assumption ("controllers know the plant
                # only through cached sensitivities").  We do NOT
                # rebuild the local nets here; we only re-apply the
                # promoted-slack OOS OLTC marks because
                # ``update_outage_masks`` above just reset them from
                # plant ``in_service``.
                if config.local_sensitivities_tso:
                    for z, ctrl in tso_controllers.items():
                        _apply_promoted_oos_oltc(
                            ctrl,
                            _zone_promoted_oltcs.get(z, ()),
                        )

        # ── Central single-controller step (control_scope='central', V5) ──────
        # One MIQP over every actuator / measurement, fired every
        # _central_period_s (default: every step).  Replaces the per-zone TSO
        # step and the per-DSO step; the 3-zone scaffolding below it is used
        # only for recording.
        if run_central and central_controller is not None:
            meas_central = _feedback_measurement(
                measure_central(plant.read_y(), central_cfg, step),
                sample_id=("control", step),
            )
            tso_plot_measurements = {z: meas_central for z in zone_defs}
            # w-shift reanchoring: reset the DER block of _u_current to the
            # measured Q so the OFO update u_new = u_old + sigma yields
            # q_set = Q_meas + sigma (the per-step "increment" interpretation
            # of the w-shift actuator).  The cascaded path does this for every
            # TSO controller (MultiTSOCoordinator.step) and every DSO controller
            # (DSO step below); the central controller bypasses both, so without
            # this reset its commanded q_set teleports from the stale last
            # command rather than incrementing from the realised Q — producing
            # the discrete DER jumps and poor tracking otherwise observed.
            central_controller.apply_qw_reset(meas_central)
            central_out = central_controller.step(meas_central, sim_time_s=time_s)
            plant.apply_u(writes_from_central(
                plant.read_y(), central_cfg, central_out.u_new,
            ))
            if verbose >= 1:
                _n_int = len(central_out.u_integer)
                print(f"  [central t={int(time_s/60):3d} min] "
                      f"obj={central_out.objective_value:.4e}  "
                      f"status={central_out.solver_status}  "
                      f"solve={central_out.solve_time_s:.2f}s  "
                      f"({_n_int} integer actuators)")

        # ── TSO step ──────────────────────────────────────────────────────────
        # Skipped entirely in TSO local mode: the CharacteristicControllers
        # (Q(V)) or the static cos phi=1 setting (Q=0) take over from the OFO
        # coordinator.  The DSO loop below is also disabled for L0/L1/L2 because
        # those scenarios use dso_mode='local'.  Also skipped in central mode
        # (control_scope='central'): the single controller above owns the TSO
        # actuators.
        if run_tso and not _local_tso and not _central:
            tso_step_count += 1
            # Decide whether to refresh cross-sensitivities this step
            refresh_H = (config.sensitivity_update_interval > 0
                         and tso_step_count % config.sensitivity_update_interval == 0)

            # Optional: rebuild the full-network ``shared_jac`` on every
            # TSO tick and reassign it to every controller so the next
            # TSO step's sensitivity matrix reflects the current
            # operating point.  Gated on ``refresh_shared_jac_on_tso``
            # and only meaningful in full-Jacobian mode (no-op when
            # ``local_sensitivities_tso`` or ``local_sensitivities_dso``
            # is True — those controllers hold their own reduced
            # Jacobians, which the runner intentionally freezes).
            if (
                config.refresh_shared_jac_on_tso
                and not config.local_sensitivities_tso
                and not config.local_sensitivities_dso
            ):
                if config.numerical_h:
                    # Re-pin the numerical H at the current operating
                    # point.  Skip the analytical shared_jac rebuild —
                    # the numerical H bypasses it entirely.
                    for z, ctrl in tso_controllers.items():
                        ctrl._H_cache = compute_numerical_h_tso(
                            net, ctrl,
                            closed_loop=config.numerical_h_closed_loop,
                            verbose=verbose,
                        )
                    for dso_id, dso_ctrl in dso_controllers.items():
                        dso_ctrl._H_cache = compute_numerical_h_dso(
                            net, dso_ctrl,
                            closed_loop=config.numerical_h_closed_loop,
                            verbose=verbose,
                        )
                else:
                    shared_jac = JacobianSensitivities(net)
                    clear_seed_lu_cache()
                    for ctrl in tso_controllers.values():
                        ctrl.sensitivities = shared_jac
                        ctrl.invalidate_sensitivity_cache()
                    for dso_ctrl in dso_controllers.values():
                        dso_ctrl.sensitivities = shared_jac
                        dso_ctrl.invalidate_sensitivity_cache()
                    coordinator.invalidate_sensitivity_cache()

            # Build per-zone measurements from plant network
            measurements: Dict[int, Measurement] = {
                z: _feedback_measurement(
                    measure_zone_tso(plant.read_y(), zd, step),
                    sample_id=("control", step),
                )
                for z, zd in zone_defs.items()
            }
            tso_plot_measurements = measurements

            # ── SBX horizontal round (BEFORE the zones solve): feed the
            # scheduler every TSO tick. At cycle boundaries the elapsed
            # window is settled, the active terminal-voltage schedule is
            # applied, and the A4 re-planning indicator is updated.
            # Between boundaries the references remain frozen.
            # The adapter is constructed at the FIRST TSO tick at/after
            # sbx_warmup_s: controller-intent schedules start after
            # the optional activation delay; an explicit planning
            # schedule may override them.
            if config.coordination_mode == "sbx":
                if (sbx_runtime["adapter"] is None
                        and time_s >= config.sbx_warmup_s):
                    from sbx_h.adapter import SBXRunnerAdapter
                    # v3: optional planning-anchored contract-voltage
                    # schedule (JSON from experiments/017_SBX_PLANNING).
                    _sbx_schedules = None
                    _sbx_bands = None
                    if config.sbx_v_std_schedule_path is not None:
                        import json as _json
                        with open(config.sbx_v_std_schedule_path,
                                  encoding="utf-8") as _fh:
                            _raw = _json.load(_fh)
                        # Entries: [t, v_a, v_b] or (with the planning-
                        # derived band) [t, v_a, v_b, band].
                        _sbx_schedules = {}
                        _sbx_bands = {}
                        for _k, _entries in _raw.items():
                            _key = tuple(int(x) for x in _k.split("-"))
                            _sbx_schedules[_key] = [
                                (float(e[0]), tuple(e[1]), tuple(e[2]))
                                for e in _entries
                            ]
                            if all(len(e) >= 4 for e in _entries):
                                _sbx_bands[_key] = [
                                    (float(e[0]), float(e[3]))
                                    for e in _entries
                                ]
                        if not _sbx_bands:
                            _sbx_bands = None
                    sbx_runtime["adapter"] = SBXRunnerAdapter(
                        net, {z: list(b) for z, b in tn_zone_map.items()},
                        tso_controllers, sbx_runtime["config"],
                        v_std_schedules=_sbx_schedules,
                        q_band_schedules=_sbx_bands,
                        support_intervals=config.sbx_support_intervals,
                        freeze_time_s=float(time_s),
                    )
                    if verbose >= 1:
                        _sc = sbx_runtime["config"]
                        _ad = sbx_runtime["adapter"]
                        _n_sup = (0 if config.sbx_support_intervals is None
                                  else sum(len(v) for v in
                                           config.sbx_support_intervals
                                           .values()))
                        print(f"  [sbx] v6 contracts initialized at t="
                              f"{time_s / 60.0:.0f} min: "
                              f"{len(_ad.registry)} corridor(s), "
                              f"source={_ad.schedule_source}, "
                              f"k_sched={_sc.k_sched} TSO iterations "
                              f"({_sc.t_cycle_min:.0f} min), "
                              f"{_n_sup} planned-support interval(s)")
                        _diag = _ad.initial_schedule_diagnostics
                        _below = [
                            row for row in _diag
                            if not row["initially_holds"]
                        ]
                        _worst = min(
                            (float(row["hold_margin_pu"])
                             for row in _diag),
                            default=float("nan"),
                        )
                        print(
                            f"  [sbx] initial hold pre-check: "
                            f"{len(_diag) - len(_below)}/{len(_diag)} "
                            f"terminals inside tolerance; "
                            f"worst margin={1e3 * _worst:+.2f} mpu"
                        )
                        for hit in _ad.border_actuators:
                            print(f"  [sbx] border actuator: "
                                  f"{hit['element']} {hit['index']} at "
                                  f"bus {hit['bus']} — hop {hit['hop']} "
                                  f"from terminal {hit['terminal_bus']} "
                                  f"of corridor {hit['corridor']} "
                                  f"(area {hit['area']})")
                if sbx_runtime["adapter"] is not None:
                    sbx_runtime["adapter"].on_tso_step(
                        tso_step_count - 1, measurements, tso_controllers,
                    )

            # ── SBX-V vertical round (BEFORE the zones solve): feed the
            # need trackers from the zone measurements and run the
            # request pipeline for the next window on a set flag; arm
            # the PricingSolver spec provider for this iteration.  The
            # priced segment structure changes only at commit instants
            # (R3 — sbxv/commit.py).
            if config.coordination_mode == "sbxv":
                sbxv_runtime["adapter"].before_solve(
                    tso_step_count - 1, measurements, tso_controllers,
                )

            # Snapshot u_current[pcc_slice] per zone BEFORE coordinator.step
            # so the diagnostic below can show u_new - u_old per Q_PCC,set.
            _prev_pcc_u: Dict[int, NDArray[np.float64]] = {}
            if verbose >= 1:
                for z, ctrl in tso_controllers.items():
                    if ctrl._u_current is None:
                        continue
                    n_der_z = len(zone_defs[z].tso_der_indices)
                    n_pcc_z = len(zone_defs[z].pcc_trafo_indices)
                    if n_pcc_z > 0:
                        _prev_pcc_u[z] = ctrl._u_current[
                            n_der_z:n_der_z + n_pcc_z
                        ].copy()

            # Run decentralised TSO step for all zones
            tso_outputs = coordinator.step(
                measurements,
                step,
                recompute_cross_sensitivities=refresh_H,
                sim_time_s=time_s,
            )

            # ── SBX-V: capture the dispatched netted PCC-Q reference per
            # AggregationArea (the logged Abruf) for the settlement plane.
            if config.coordination_mode == "sbxv":
                sbxv_runtime["adapter"].after_solve(
                    tso_step_count - 1, tso_controllers,
                )

            # ── Q_PCC,set command diagnostic ───────────────────────────
            # For each PCC trafo: print previous command, new command,
            # delta sigma, the input-bound the MIQP was given, and the
            # distance from the new u to the nearest bound.  If sigma is
            # essentially zero AND the new u is far from both bounds,
            # the OFO sees no gradient on this column (most likely
            # cause: DER absorbed the V-tracking signal).  If sigma is
            # zero AND the new u is at a bound, the bound is clamping.
            if verbose >= 3:
                for z, tso_out in tso_outputs.items():
                    n_der_z = len(zone_defs[z].tso_der_indices)
                    n_pcc_z = len(zone_defs[z].pcc_trafo_indices)
                    if n_pcc_z == 0:
                        continue
                    ctrl_z = tso_controllers[z]
                    u_new_pcc = tso_out.u_new[n_der_z:n_der_z + n_pcc_z]
                    u_old_pcc = _prev_pcc_u.get(
                        z, np.zeros(n_pcc_z, dtype=np.float64)
                    )
                    # Recompute the input bound the MIQP saw — same code
                    # as TSOController._compute_input_bounds for PCC.
                    if not ctrl_z.config.pcc_capability_on_output:
                        q_iface_now_z = []
                        for t in zone_defs[z].pcc_trafo_indices:
                            if t in net.res_trafo3w.index:
                                q_iface_now_z.append(
                                    float(net.res_trafo3w.at[t, "q_hv_mvar"])
                                )
                            elif t in net.res_trafo.index:
                                q_iface_now_z.append(
                                    float(net.res_trafo.at[t, "q_hv_mvar"])
                                )
                            else:
                                q_iface_now_z.append(0.0)
                        q_iface_now_arr = np.asarray(q_iface_now_z, dtype=np.float64)
                        lb = q_iface_now_arr + ctrl_z.pcc_capability_min_mvar
                        ub = q_iface_now_arr + ctrl_z.pcc_capability_max_mvar
                    else:
                        lb = np.full(n_pcc_z, -1e4)
                        ub = np.full(n_pcc_z, +1e4)
                    for k, t in enumerate(zone_defs[z].pcc_trafo_indices):
                        u_o = float(u_old_pcc[k]) if k < len(u_old_pcc) else 0.0
                        u_n = float(u_new_pcc[k])
                        sigma = u_n - u_o
                        slack_lo = u_n - lb[k]
                        slack_hi = ub[k] - u_n
                        # Tag: 'AT_LB' if at lower bound, 'AT_UB' if at upper,
                        # 'BOUND_TIGHT' if width < 1 Mvar, else 'FREE'
                        if (ub[k] - lb[k]) < 1.0:
                            tag = "BOUND_TIGHT"
                        elif abs(slack_lo) < 0.5:
                            tag = "AT_LB"
                        elif abs(slack_hi) < 0.5:
                            tag = "AT_UB"
                        else:
                            tag = "FREE"
                        print(
                            f"  [pcc-set z{z} t={t}] u_old={u_o:+7.2f} -> "
                            f"u_new={u_n:+7.2f}  Δ={sigma:+6.3f}  "
                            f"bound=[{lb[k]:+7.2f}, {ub[k]:+7.2f}]  "
                            f"[{tag}]"
                        )

            # Apply TSO controls to the plant.  The pre-write shunt steps
            # are captured first (apply_zone_tso_controls' legacy return
            # contract) so shunt switches can be detected below.
            for z, tso_out in tso_outputs.items():
                prev_shunt_steps = shunt_steps_for_buses(
                    plant.read_y(), zone_defs[z].shunt_bus_indices,
                )
                plant.apply_u(writes_from_zone_tso(
                    plant.read_y(), zone_defs[z], tso_out.u_new,
                ))

                # Record per-zone results.
                # Q_cor mode actuator order on u (TSOControllerConfig._continuous_block):
                #   [ Q_cor_DER | Q_PCC_set | V_gen_set | s_OLTC | s_shunt ]
                u = tso_out.u_new
                n_der = len(zone_defs[z].tso_der_indices)
                n_pcc = len(zone_defs[z].pcc_trafo_indices)
                n_gen = len(zone_defs[z].gen_indices)
                n_oltc = len(zone_defs[z].oltc_trafo_indices)
                n_shunt = len(zone_defs[z].shunt_bus_indices)
                off = 0
                rec.zone_q_der[z]         = u[off:off+n_der].copy(); off += n_der
                rec.zone_q_pcc_set[z]     = u[off:off+n_pcc].copy(); off += n_pcc
                # Persist per-trafo interface setpoint (parallel to
                # pcc_trafo_indices) for the local-DSO recording path below.
                for _k, _t in enumerate(zone_defs[z].pcc_trafo_indices):
                    last_pcc_set_per_trafo[int(_t)] = float(rec.zone_q_pcc_set[z][_k])
                rec.zone_v_gen[z]         = u[off:off+n_gen].copy(); off += n_gen
                rec.zone_oltc_taps[z]     = u[off:off+n_oltc].copy(); off += n_oltc
                rec.zone_tso_objective[z] = tso_out.objective_value
                rec.zone_tso_status[z]    = tso_out.solver_status
                rec.zone_tso_solve_s[z]   = tso_out.solve_time_s

                # Record contraction diagnostic
                diag = coordinator.last_coupling_diagnostics.get(z, {})
                rec.zone_contraction_lhs[z] = diag.get("contraction_lhs", float("nan"))

                # ── TSO-owned shunt switching: detect and propagate ───────
                # When the MIQP switches a shunt step, apply a rank-1 SMW
                # update to the TSO's own cached J⁻¹ (no pp.runpp), drop
                # the H caches so the next TSO step rebuilds H from the
                # updated dV_dQ_reduced, then dispatch a
                # ``ShuntDisturbanceMessage`` to the DSO whose tertiary
                # hosts the shunt so the DSO can refresh its own model.
                if n_shunt > 0:
                    # Shunt block sits after [Q_cor_DER | Q_PCC | V_gen | OLTC].
                    shunt_offset = n_der + n_pcc + n_gen + n_oltc
                    new_shunt_steps = [
                        int(round(float(u[shunt_offset + k])))
                        for k in range(n_shunt)
                    ]
                    changed = [
                        (int(zone_defs[z].shunt_bus_indices[k]), s_new, k)
                        for k, (s_new, s_prev) in enumerate(
                            zip(new_shunt_steps, prev_shunt_steps)
                        )
                        if s_new != s_prev
                    ]
                    if changed:
                        tso_ctrl_z = tso_controllers[z]

                        # ── Diagnostic: which gradient component drove the
                        # switch?  Decompose the shunt column of H into V,
                        # I, Q_gen contributions and weight by the matching
                        # tracking error / soft-slack error.  Helpful for
                        # debugging "why did the shunt switch from -1 to 0?"
                        # — if the V-tracking term has the same sign as
                        # the actual move, the optimiser was tracking V;
                        # otherwise another term dominated.
                        H_z = getattr(tso_ctrl_z, "_H_cache", None)
                        last_meas = getattr(tso_ctrl_z, "_last_measurement", None)
                        if H_z is not None and last_meas is not None:
                            n_v_z = len(zone_defs[z].v_bus_indices)
                            n_pcc_z = len(zone_defs[z].pcc_trafo_indices)
                            n_i_z = len(zone_defs[z].line_indices)
                            n_gen_z = len(zone_defs[z].gen_indices)
                            v_set = float(config.v_setpoint_pu)
                            v_pu = np.asarray(
                                [float(last_meas.voltage_magnitudes_pu[
                                    np.where(last_meas.bus_indices == b)[0][0]
                                ]) for b in zone_defs[z].v_bus_indices],
                                dtype=np.float64,
                            )
                            v_err = v_pu - v_set
                            for sb, s_new, k_sh in changed:
                                col = shunt_offset + k_sh
                                s_prev_k = prev_shunt_steps[k_sh]
                                ds = s_new - s_prev_k
                                col_v_part = H_z[:n_v_z, col]
                                grad_v = float(v_err @ col_v_part)
                                # Q_gen contribution
                                # New row layout: [V | Q_PCC | I | Q_gen]
                                q_row_start = n_v_z + n_pcc_z + n_i_z
                                col_qg_part = H_z[q_row_start:q_row_start + n_gen_z, col]
                                # The gradient contribution from V tracks (V-V_set);
                                # for Q_gen it tracks (Q_gen - Q_gen_target) — but
                                # we don't have Q_target handy here; report magnitude
                                # so user can see if Q_gen column is non-trivial.
                                qg_norm = float(np.linalg.norm(col_qg_part))
                                v_min = float(v_pu.min())
                                v_max = float(v_pu.max())
                                # Expected V-driven move direction:
                                #   v_err·col_v < 0 → MIQP wants Δs > 0 (s up)
                                #   v_err·col_v > 0 → MIQP wants Δs < 0 (s down)
                                v_wants_up = grad_v < 0.0
                                actual_up = ds > 0
                                consistent = (v_wants_up == actual_up)
                                tag = "OK" if consistent else "INCONSISTENT-with-V"
                                print(
                                    f"  [shunt-switch z{z}] bus={sb}: "
                                    f"s {s_prev_k:+d}→{s_new:+d} (Δ={ds:+d})  "
                                    f"V[{v_min:.4f},{v_max:.4f}]  "
                                    f"V-grad={grad_v:+.3e}  "
                                    f"|Q_gen-col|={qg_norm:.3e}  "
                                    f"[{tag}]"
                                )

                        if config.local_sensitivities_tso:
                            # Under local-net mode the TSO's Jacobian has a
                            # *synthetic* shunt at the 3W primary bus, not
                            # the (dropped) tertiary; the SMW lookup would
                            # miss it.  Rebuild the reduced Jacobian from
                            # scratch instead — the cached step on the
                            # synthetic shunt then matches the plant.
                            _jz_new, _syn_map_new, _promoted_oltcs_new = (
                                _build_tso_local_jac(z)
                            )
                            _per_zone_local_jac[z] = _jz_new
                            _per_zone_synth_shunt_map[z] = _syn_map_new
                            _zone_promoted_oltcs[z] = _promoted_oltcs_new
                            tso_ctrl_z.sensitivities = _jz_new
                            if _syn_map_new:
                                tso_ctrl_z.config.shunt_sensitivity_bus_indices = [
                                    _syn_map_new.get(int(b), int(b))
                                    for b in tso_ctrl_z.config.shunt_bus_indices
                                ]
                            _apply_promoted_oos_oltc(
                                tso_ctrl_z, _promoted_oltcs_new
                            )
                            tso_ctrl_z.invalidate_sensitivity_cache()
                        else:
                            any_smw = False
                            for sb, s_new, _ in changed:
                                applied = tso_ctrl_z.sensitivities.apply_shunt_step_change_smw(
                                    sb, s_new,
                                )
                                any_smw = any_smw or applied
                            if any_smw:
                                tso_ctrl_z.invalidate_sensitivity_cache()

                        # Dispatch a per-DSO ShuntDisturbanceMessage so each
                        # affected DSO updates its own cached J⁻¹.
                        per_dso: Dict[str, List[Tuple[int, int, float]]] = {}
                        for sb, s_new, k_idx in changed:
                            dso_id_aff = shunt_bus_to_dso_id.get(sb)
                            if dso_id_aff is None:
                                continue
                            q_step = float(zone_defs[z].shunt_q_steps_mvar[k_idx])
                            per_dso.setdefault(dso_id_aff, []).append(
                                (sb, s_new, q_step)
                            )
                        for dso_id_aff, items in per_dso.items():
                            dso_ctrl_aff = dso_controllers.get(dso_id_aff)
                            if dso_ctrl_aff is None:
                                continue
                            msg = ShuntDisturbanceMessage(
                                source_controller_id=tso_ctrl_z.controller_id,
                                target_controller_id=dso_id_aff,
                                iteration=step,
                                shunt_bus_indices=np.array(
                                    [it[0] for it in items], dtype=np.int64,
                                ),
                                shunt_steps=np.array(
                                    [it[1] for it in items], dtype=np.int64,
                                ),
                                shunt_q_steps_mvar=np.array(
                                    [it[2] for it in items], dtype=np.float64,
                                ),
                            )
                            dso_ctrl_aff.receive_disturbance_message(msg)

            # ── Switched-shunt integrator dispatch (integrator mode) ──────────
            # Runs on the SAME measurement / cached operating point the MIQP used
            # this TSO instant, OUTSIDE the MIQP.  Each commit is applied
            # atomically in this instant: physical toggle (plant) + DSO interface
            # feedforward + rank-1 SMW refresh of the TSO and DSO cached Jacobians
            # (NO power flow).  The three nets (plant, TSO cache, DSO cache) are
            # independent deep copies, so the toggle and the SMW reads do not
            # interfere.
            if _shunt_mode == "integrator":
                for z_i, integ in zone_integrators.items():
                    tso_ctrl_i = tso_controllers[z_i]
                    meas_i = getattr(tso_ctrl_i, "_last_measurement", None)
                    if meas_i is None:
                        continue
                    zd_i = zone_defs[z_i]
                    n_v_i = len(zd_i.v_bus_indices)
                    vb_i = list(zd_i.v_bus_indices)
                    pcc_i = list(zd_i.pcc_trafo_indices)
                    grad_y_i = tso_ctrl_i._compute_output_gradient(meas_i)
                    sens_i = tso_ctrl_i.sensitivities
                    n_pcc_i = len(pcc_i)
                    n_i_i = len(zd_i.line_indices)
                    n_gen_i = len(zd_i.gen_indices)
                    gen_buses_i = list(zd_i.gen_bus_indices)
                    use_reserve = (config.tso_g_res_sg != 0.0 and n_gen_i > 0)

                    grad_g_list: List[float] = []
                    v_meas_list: List[NDArray[np.float64]] = []
                    h_v_list: List[NDArray[np.float64]] = []
                    for bank in integ.banks:
                        bc = bank.config
                        sign_b = -1.0 if bc.kind == "MSC" else +1.0
                        # ── Voltage term: ∂V/∂Q_eq at the zone's EHV obs buses,
                        # in physical [pu/Mvar] (compute_dV_dQ_shunt now divides
                        # by the system base at source).
                        h_col, obs_map = sens_i.compute_dV_dQ_shunt(
                            shunt_bus_idx=bc.bus_idx,
                            observation_bus_indices=vb_i,
                            q_step_mvar=sign_b,
                        )
                        h_col = np.asarray(h_col, dtype=np.float64)
                        if h_col.size == 0:
                            # No voltage coupling from any zone PQ bus this instant.
                            grad_g_list.append(0.0)
                            v_meas_list.append(np.array([1.0]))
                            h_v_list.append(np.array([0.0]))
                            continue
                        v_meas_b = np.array([
                            float(meas_i.voltage_magnitudes_pu[
                                np.where(meas_i.bus_indices == b)[0][0]
                            ]) for b in obs_map
                        ], dtype=np.float64)
                        grad_v_obs = np.array([
                            float(grad_y_i[vb_i.index(b)]) for b in obs_map
                        ], dtype=np.float64)
                        g_h = float(grad_v_obs @ h_col)
                        # ── Q_PCC term (dimensionless Mvar/Mvar interface ratio —
                        # base-independent, no s_base scaling).  Active only when
                        # the bank's interface is a monitored PCC.
                        t_if = bc.interface_trafo3w_idx
                        if pcc_i and t_if in pcc_i:
                            h_qpcc = float(sens_i.compute_dQtrafo3w_hv_dQ_shunt(
                                trafo3w_idx=t_if, shunt_bus_idx=bc.bus_idx,
                                q_step_mvar=sign_b,
                            ))
                            g_h += float(grad_y_i[n_v_i + pcc_i.index(t_if)]) * h_qpcc
                        # ── SG reactive-reserve term: ∂Q_gen/∂Q_eq (dimensionless
                        # Mvar/Mvar ratio — base-independent).  Lets the reserve
                        # objective (tso_g_res_sg) drive the bulk shunt to offload
                        # generator reactive loading on sustained need.  Opt-in via
                        # tso_g_res_sg != 0.
                        if use_reserve:
                            Hg, _, _ = sens_i.compute_dQgen_dQ_shunt_matrix(
                                gen_buses_i, [int(bc.bus_idx)], [sign_b],
                            )
                            qg0 = n_v_i + n_pcc_i + n_i_i
                            g_h += float(grad_y_i[qg0:qg0 + n_gen_i] @ Hg[:, 0])
                        grad_g_list.append(g_h)
                        v_meas_list.append(v_meas_b)
                        h_v_list.append(h_col)

                    commits = integ.update(
                        grad_g_list, v_meas_list, h_v_list, t_now=time_s,
                    )
                    for commit in commits:
                        # (i) Toggle the physical bank (ground truth, plant net).
                        plant.apply_u(ActuatorWrites(shunt_step={
                            int(commit.shunt_idx): int(commit.pp_step_new),
                        }))
                        # (ii) Step the DSO interface feedforward from the DSO's
                        #      OWN 3W model at the current (pre-switch) operating
                        #      point — atomically with the toggle.
                        dso_ctrl_c = dso_controllers.get(commit.dso_id)
                        sign_c = -1.0 if commit.kind == "MSC" else +1.0
                        if dso_ctrl_c is not None:
                            dq_itf = float(
                                dso_ctrl_c.sensitivities.compute_dQtrafo3w_hv_dQ_shunt(
                                    trafo3w_idx=commit.interface_trafo3w_idx,
                                    shunt_bus_idx=commit.bus_idx,
                                    q_step_mvar=sign_c * commit.q_step_mvar,
                                )
                            ) * float(commit.direction)
                        else:
                            dq_itf = 0.0
                        if not np.isfinite(dq_itf):
                            raise ValueError(
                                f"dQ_itf_sh not finite for {commit.kind} commit at "
                                f"bus {commit.bus_idx} (interface "
                                f"{commit.interface_trafo3w_idx})"
                            )
                        q_itf_sh_offset[int(commit.interface_trafo3w_idx)] = (
                            q_itf_sh_offset.get(int(commit.interface_trafo3w_idx), 0.0)
                            + dq_itf
                        )
                        # (iii) Refresh the TSO's cached Jacobian (rank-1 SMW, no
                        #       power flow), then drop its H cache.
                        if sens_i.apply_shunt_step_change_smw(
                            commit.bus_idx, commit.pp_step_new,
                            shunt_idx=commit.shunt_idx,
                        ):
                            tso_ctrl_i.invalidate_sensitivity_cache()
                        # (iv) Tell the DSO (SMW-only refresh, no power flow).
                        if dso_ctrl_c is not None:
                            dso_ctrl_c.receive_disturbance_message(
                                ShuntDisturbanceMessage(
                                    source_controller_id=tso_ctrl_i.controller_id,
                                    target_controller_id=commit.dso_id,
                                    iteration=step,
                                    shunt_bus_indices=np.array(
                                        [commit.bus_idx], dtype=np.int64),
                                    shunt_steps=np.array(
                                        [commit.pp_step_new], dtype=np.int64),
                                    shunt_q_steps_mvar=np.array(
                                        [commit.q_step_mvar], dtype=np.float64),
                                    shunt_indices=np.array(
                                        [commit.shunt_idx], dtype=np.int64),
                                )
                            )
                        if verbose >= 1:
                            print(
                                f"  [shunt-commit z{z_i}] {commit.kind} "
                                f"bus={commit.bus_idx} ℓ {commit.old_level}→"
                                f"{commit.new_level} (Δ={commit.direction:+d})  "
                                f"dQ_itf={dq_itf:+.2f} Mvar"
                            )

            # TSO sends Q setpoints to DSOs via grouped setpoint messages
            for z, ctrl in tso_controllers.items():
                for msg in ctrl.generate_setpoint_messages():
                    if msg.target_controller_id in dso_controllers:
                        # Add the persistent switched-shunt feedforward offset to
                        # the interface setpoints so the DSO does not counteract a
                        # committed switch (it rejects only the residual).
                        #
                        # PAIRED WITH the capability-band shift in the DSO step
                        # above: this moves the setpoint, that moves the bound
                        # the setpoint has to satisfy.  Changing one without the
                        # other puts the dispatched value outside the DSO's
                        # reported capability by exactly the offset — the
                        # 2026-08-13 fault.
                        if _shunt_mode == "integrator" and q_itf_sh_offset:
                            q_adj = msg.q_setpoints_mvar.copy()
                            for _ii, _t in enumerate(
                                msg.interface_transformer_indices
                            ):
                                q_adj[_ii] += q_itf_sh_offset.get(int(_t), 0.0)
                            msg.q_setpoints_mvar = q_adj
                        dso_controllers[msg.target_controller_id].receive_setpoint(msg)
                        # Record total Q setpoint (sum over interface trafos)
                        rec.dso_q_set_mvar[msg.target_controller_id] = float(
                            msg.q_setpoints_mvar.sum()
                        )
                        last_dso_q_set_mvar[msg.target_controller_id] = (
                            msg.q_setpoints_mvar.copy()
                        )

        # Exogenous Q-setpoint injection for the no-TSO-OFO branch.  Used
        # by 003_M_DSO_CIGRE_2026 to push externally-defined Q_PCC setpoints
        # to one or more DSOs while ``tso_mode='local'`` keeps the TSO
        # layer purely under local Q(V) control.  Only runs when the TSO
        # OFO setpoint dispatch above is inactive (``_local_tso=True``)
        # AND the runner config supplies a setpoint dictionary.
        if (
            run_tso
            and _local_tso
            and config.q_pcc_setpoints_mvar_per_dso
        ):
            for dso_id, q_vec in config.q_pcc_setpoints_mvar_per_dso.items():
                if dso_id not in dso_controllers:
                    continue
                dso_ctrl_t = dso_controllers[dso_id]
                msg = SetpointMessage(
                    source_controller_id="exogenous",
                    target_controller_id=dso_id,
                    iteration=step,
                    interface_transformer_indices=np.array(
                        dso_ctrl_t.config.interface_trafo_indices,
                        dtype=np.int64,
                    ),
                    q_setpoints_mvar=np.asarray(q_vec, dtype=np.float64),
                )
                dso_ctrl_t.receive_setpoint(msg)
                rec.dso_q_set_mvar[dso_id] = float(msg.q_setpoints_mvar.sum())
                last_dso_q_set_mvar[dso_id] = msg.q_setpoints_mvar.copy()

        # Exogenous Q_PCC injection with a LIVE (but frozen) OFO parent.
        #
        # Deliberately a separate block from the ``_local_tso`` one above, and
        # deliberately not gated on ``run_tso``: with a frozen parent
        # (``tso_period_s > n_total_s``) ``run_tso`` is true only at step 1, so
        # folding this into that condition would deliver the schedule once and
        # never again.  Setpoints persist in the subordinate controller until
        # replaced, so re-delivering the value in force every step is a no-op
        # except at a schedule boundary -- which is what makes the boundary
        # exact rather than dependent on the dispatch grid.
        #
        # Used by the Sec. 9.1 isolated-STS measurement of ``N_inner``
        # (eq. 9.2).  Default-off, so no existing configuration is affected.
        if config.q_pcc_injection_with_ofo_parent:
            _sched = config.q_pcc_setpoint_schedule_per_dso or {}
            _const = config.q_pcc_setpoints_mvar_per_dso or {}
            for dso_id in set(_sched) | set(_const):
                if dso_id not in dso_controllers:
                    continue
                if dso_id in _sched:
                    # Last entry whose start time has been reached.  Entries
                    # need not be sorted; ``max`` over the eligible set is
                    # order-independent and cheap at these sizes.
                    eligible = [e for e in _sched[dso_id]
                                if float(e["t_s"]) <= time_s]
                    if not eligible:
                        continue
                    q_vec = max(eligible, key=lambda e: float(e["t_s"]))["q_mvar"]
                else:
                    q_vec = _const[dso_id]
                dso_ctrl_t = dso_controllers[dso_id]
                msg = SetpointMessage(
                    source_controller_id="exogenous",
                    target_controller_id=dso_id,
                    iteration=step,
                    interface_transformer_indices=np.array(
                        dso_ctrl_t.config.interface_trafo_indices,
                        dtype=np.int64,
                    ),
                    q_setpoints_mvar=np.asarray(q_vec, dtype=np.float64),
                )
                dso_ctrl_t.receive_setpoint(msg)
                rec.dso_q_set_mvar[dso_id] = float(msg.q_setpoints_mvar.sum())
                last_dso_q_set_mvar[dso_id] = msg.q_setpoints_mvar.copy()

        # ── DSO step (all zones) ──────────────────────────────────────────────
        if run_dso and not _local_dso:
            for dso_id, dso_ctrl in dso_controllers.items():
                # meas_dso reflects the current operating point BEFORE this DSO step.
                # This is the correct basis for the capability message: it tells the TSO
                # what the DSO can still do from its present dispatch, not what it just did.
                meas_dso = _feedback_measurement(
                    measure_zone_dso(plant.read_y(), dso_ctrl.config, step),
                    sample_id=("control", step),
                )
                dso_plot_measurements[dso_id] = meas_dso

                # --- Capability message: DSO → TSO (must precede TSO solve) ----------
                tso_id = dso_to_tso_id[dso_id]
                cap_msg = dso_ctrl.generate_capability_message(
                    target_controller_id=tso_id,
                    measurement=meas_dso,
                )
                # ── Switched-shunt feedforward: shift the capability band ─────
                # PAIRED WITH the ``q_adj`` block in the TSO setpoint dispatch
                # below — keep the two together, they must move as one.
                #
                # The DSO is dispatched ``u_pcc + q_itf_sh_offset``, but the
                # MIQP bounds ``u_pcc`` itself against this band.  Without the
                # matching shift the quantity actually REQUESTED escapes the
                # band by exactly the offset.  Requiring the emitted setpoint
                # to be reachable:
                #
                #   u_pcc + off ∈ [q_now + dmin,       q_now + dmax]
                #  ⟺ u_pcc      ∈ [q_now + dmin - off, q_now + dmax - off]
                #
                # so the offset comes off the reported deltas here.
                #
                # Measured 2026-08-13 without this (6 h, IEEE 39 + 4 HV nets):
                # DSO_2 settled at ``rail + offset`` — setpoint -137.46 against
                # a reported rail of -89.5 with a cumulative offset of -48.23 —
                # leaving a -57.7 Mvar interface gap that no actuator could
                # close and that was insensitive to g_q, g_z_q_pcc and the
                # sensitivity model.  With the shift: -57.7 -> +1.7 Mvar, and
                # the setpoint tracks the reported rail to ~0.1 Mvar.
                # See 00_daily_log/2026-08-13_capability_open_loop.md.
                if _shunt_mode == "integrator" and q_itf_sh_offset:
                    _cap_min = np.asarray(cap_msg.q_min_mvar, dtype=np.float64).copy()
                    _cap_max = np.asarray(cap_msg.q_max_mvar, dtype=np.float64).copy()
                    for _ci, _ct in enumerate(
                        cap_msg.interface_transformer_indices
                    ):
                        _off = q_itf_sh_offset.get(int(_ct), 0.0)
                        _cap_min[_ci] -= _off
                        _cap_max[_ci] -= _off
                    cap_msg.q_min_mvar = _cap_min
                    cap_msg.q_max_mvar = _cap_max
                # Deliver to the responsible TSO controller.
                target_tso = next(
                    ctrl for ctrl in tso_controllers.values()
                    if ctrl.controller_id == tso_id
                )
                target_tso.receive_capability(cap_msg)

                # ── PCC capability diagnostic ───────────────────────────
                # Print, on each TSO tick, the per-coupling-trafo bounds
                # the TSO will use for Q_PCC,set this step:
                #
                #   q_now  = current Q at the HV port (load convention)
                #   Δmin, Δmax = capability deltas reported by the DSO
                #   bound  = [q_now + Δmin, q_now + Δmax]  (input bound on
                #            Q_PCC,set under ``tso_pcc_capability_on_output=False``)
                #   width  = Δmax − Δmin  (PCC freedom this step in Mvar)
                #
                # Also reports DER-rail headroom: per-DER (q_max − q_now)
                # and (q_now − q_min) aggregated across the DSO.  When
                # both totals are small the DSO is fully saturated and
                # the TSO has no PCC freedom regardless of g_w_pcc.
                if verbose >= 3 and run_tso:
                    try:
                        ifaces = list(dso_ctrl.config.interface_trafo_indices)
                        q_now_per = []
                        for t in ifaces:
                            if t in net.res_trafo3w.index:
                                q_now_per.append(
                                    float(net.res_trafo3w.at[t, "q_hv_mvar"])
                                )
                            elif t in net.res_trafo.index:
                                q_now_per.append(
                                    float(net.res_trafo.at[t, "q_hv_mvar"])
                                )
                            else:
                                q_now_per.append(float("nan"))
                        d_min = list(cap_msg.q_min_mvar)
                        d_max = list(cap_msg.q_max_mvar)
                        # Per-DER headroom totals
                        der_p_now = dso_ctrl._extract_der_active_power(meas_dso)
                        q_der_min, q_der_max = dso_ctrl.actuator_bounds.compute_der_q_bounds(
                            der_p_now
                        )
                        q_der_now = meas_dso.der_q_mvar.copy()
                        head_up_total = float(np.sum(
                            np.maximum(q_der_max - q_der_now, 0.0)
                        ))
                        head_dn_total = float(np.sum(
                            np.maximum(q_der_now - q_der_min, 0.0)
                        ))
                        n_sat_up = int(np.sum(
                            (q_der_max - q_der_now) < 0.5
                        ))
                        n_sat_dn = int(np.sum(
                            (q_der_now - q_der_min) < 0.5
                        ))
                        for k, t in enumerate(ifaces):
                            qn = q_now_per[k]
                            dl = float(d_min[k]) if k < len(d_min) else float("nan")
                            du = float(d_max[k]) if k < len(d_max) else float("nan")
                            print(
                                f"  [cap {dso_id} t={t}] q_now={qn:+7.2f}  "
                                f"Δ=[{dl:+7.2f}, {du:+7.2f}]  "
                                f"bound=[{qn + dl:+7.2f}, {qn + du:+7.2f}]  "
                                f"width={du - dl:6.2f} Mvar"
                            )
                        print(
                            f"  [cap {dso_id}] DER headroom: up={head_up_total:6.1f} Mvar  "
                            f"down={head_dn_total:6.1f} Mvar  "
                            f"saturated DERs: up={n_sat_up}/{len(q_der_now)}  "
                            f"down={n_sat_dn}/{len(q_der_now)}"
                        )
                    except Exception as _exc:
                        print(f"  [cap {dso_id}] diagnostic failed: {_exc}")

                # --- DSO optimisation step -------------------------------------------
                # w-shift reanchoring: reset the DER block of _u_current
                # to the measured Q so the OFO's update u_new = u_old +
                # sigma yields q_set = Q_meas + sigma.  See
                # :meth:`DSOController.apply_qw_reset`.
                if hasattr(dso_ctrl, "apply_qw_reset"):
                    dso_ctrl.apply_qw_reset(meas_dso)
                dso_out = dso_ctrl.step(meas_dso, sim_time_s=time_s)
                # Record the MIQP diagnostics.  ``dso_objective``/
                # ``dso_status`` existed on the record but were never
                # assigned, so a DSO that declines to act was
                # indistinguishable from one that solved and chose a zero
                # step (Gate E, 2026-07-21: the RMS DSO answered a 31 Mvar
                # setpoint change with ~1e-4 Mvar steps for 6 iterations and
                # there was no logged quantity that could explain it).
                rec.dso_objective[dso_id] = (
                    float(dso_out.objective_value)
                    if dso_out.objective_value is not None else None
                )
                rec.dso_status[dso_id] = str(dso_out.solver_status)
                rec.dso_sigma_norm[dso_id] = float(
                    np.linalg.norm(np.asarray(dso_out.sigma, dtype=float))
                )
                rec.dso_z_slack_max[dso_id] = (
                    float(np.max(np.abs(np.asarray(dso_out.z_slack,
                                                   dtype=float))))
                    if dso_out.z_slack is not None
                    and np.size(dso_out.z_slack) else 0.0
                )
                plant.apply_u(writes_from_dso(dso_ctrl.config, dso_out.u_new))

        # ── End-of-step Power flow ─────────────────────────────────────
        # Skip when nothing wrote new actuator commands this step: the
        # post-profile (and post-contingency, if any) PFs already left
        # the network at its final operating point.  This is the typical
        # case for L0/L1 (no MIQP at all) and the off-cycle steps of
        # T0/T1 where DSO is local.  An MIQP fired ⇒ end-of-step PF is
        # required to propagate the new q_cor / V_set / OLTC commands.
        _miqp_acted = (
            (run_tso and not _local_tso and not _central)
            or (run_dso and not _local_dso)
            or run_central
        )
        _needs_end_pf = _miqp_acted or _contingency_fired_this_step
        if (_needs_end_pf and isinstance(plant, PandapowerStaticPlant)
                and not config.disable_qv_seed):
            # Warm-start QVLocalLoops from the linear closed-loop
            # equilibrium so run_control only refines residuals.  This
            # writes the quasi-static DER ``q_mvar`` state directly and
            # must therefore never run against an external RMS plant.
            seed_qv_equilibrium(
                net,
                list(meta.tso_der_indices) + list(meta.dso_der_indices),
                shared_jac,
            )
        try:
            if _needs_end_pf:
                try:
                    # Propagate the new commands over the rest of this
                    # dispatch interval.  The static plant ignores the
                    # duration (one re-solve, legacy behaviour); the RMS
                    # plant advances real time -- ``dt_s`` minus whatever the
                    # profile pre-settle already consumed, so the interval
                    # totals dt_s and the clock stays 1x.
                    plant.advance(config.dt_s - _profile_settle_s)
                    plant.read_y()
                except LoadflowNotConverged:
                    # End-of-step PF with enforce_q_lims=True can fail to
                    # converge on stressful events (e.g., a heavy load
                    # connect) when run_control=True is also active: the
                    # local Q(V) droop and the PV→PQ flips from
                    # enforce_q_lims interact and oscillate within the 50-NR
                    # iteration budget.  Fall back to a flat-start retry
                    # without Q-limit enforcement so the timestep still
                    # records a converged state; the next step's PFs (post-
                    # profile, post-contingency, or the next end-of-step PF
                    # itself) re-attempt with the clamp on a more relaxed
                    # operating point.
                    if not isinstance(plant, PandapowerStaticPlant):
                        raise
                    if verbose >= 1:
                        print(f"  [Step {step}] end-of-step PF with "
                              f"enforce_q_lims={config.enforce_q_lims_plant} "
                              f"diverged; retrying flat-start unclipped...")
                    pp.runpp(net, run_control=_run_control,
                             calculate_voltage_angles=True,
                             max_iteration=100,
                             max_iter=300,
                             distributed_slack=config.distributed_slack,
                             enforce_q_lims=False,
                             init='flat')
            elif not isinstance(plant, PandapowerStaticPlant):
                # No MIQP acted this step, but RMS time must still flow
                # (the static plant's operating point is already final).
                plant.advance(config.dt_s - _profile_settle_s)
                plant.read_y()
            if _oltc_limiter_active and _needs_end_pf:
                _moved = _oltc_limiter.clamp(net, time_s)
                if _moved:
                    if verbose >= 3:
                        _pretty = ", ".join(
                            f"{tab}#{tid} {prev:+d}->{new:+d}"
                            for tab, tid, prev, new in _moved
                        )
                        print(
                            f"  [Step {step}] end-of-step OLTC tap-rate "
                            f"limit ({len(_moved)}): {_pretty}; re-running "
                            f"PF with run_control=False..."
                        )
                    pp.runpp(
                        net, run_control=False, calculate_voltage_angles=True,
                        max_iteration=100,
                        distributed_slack=config.distributed_slack,
                        enforce_q_lims=config.enforce_q_lims_plant,
                    )
        except Exception as e:
            if not isinstance(plant, PandapowerStaticPlant):
                raise
            print(f"  [Step {step}] Power flow failed: {e}")
            log.append(rec)
            continue

        # ── Voltage-stability / nose-curve reachability guard ────────────────
        # The network is converged here (PF failures above did ``continue``).
        # Verify the equilibrium lies on the stable upper voltage branch via
        # the modal Q-V criterion, record the margin into this step's record,
        # and abort at the first lower-branch point.  Placed BEFORE any metric
        # logging and OUTSIDE the PF try/except so the ReachabilityViolation
        # propagates out of the runner (Fail-Fast) instead of being swallowed.
        if _reach_monitor is not None:
            try:
                _reach = _reach_monitor.check_step(
                    net, step_index=step, time_s=time_s
                )
            except ReachabilityViolation as _rv:
                # Hand the records accumulated before the violation to the
                # caller (experiment drivers persist them); then re-raise so the
                # run still aborts (Fail-Fast).
                _rv.partial_log = log
                raise
            rec.reach_sigma_min_J = _reach.sigma_min_J
            rec.reach_lambda_min_JR = _reach.lambda_min_JR
            rec.reach_critical_bus = _reach.critical_bus

        # ── Record post-PF observables (require converged res_* tables) ──────
        if run_dso and not _local_dso:
            for dso_id, dso_ctrl in dso_controllers.items():
                # Actual Q at PCC (sum over all 3W interface trafos)
                q_actual_sum = sum(
                    float(net.res_trafo3w.at[t, "q_hv_mvar"])
                    for t in dso_ctrl.config.interface_trafo_indices
                    if t in net.res_trafo3w.index
                )
                rec.dso_q_actual_mvar[dso_id] = q_actual_sum

            # Iterate only over DSOs with constructed OFO controllers
            # (``config.dso_ids_to_run`` may have restricted the set).
            _record_dso_group_and_transformer_data(
                rec=rec,
                net=net,
                dso_ids=list(dso_controllers.keys()),
                dsocontrollers=dso_controllers,
                dso_group_map=dso_group_map,
                last_dso_q_set_mvar=last_dso_q_set_mvar,
                hv_info_map=hv_info_map,
            )

        if _local_dso:
            # Record PCC Q actuals and HV group voltage stats even without OFO
            # DSO controllers so that comparison plots have the same dso_group_ids
            # as the coordinated scenario.
            for hv in meta.hv_networks:
                q_actual_sum = sum(
                    float(net.res_trafo3w.at[t, "q_hv_mvar"])
                    for t in hv.coupling_trafo_indices
                    if t in net.res_trafo3w.index
                )
                rec.dso_q_actual_mvar[hv.net_id] = q_actual_sum

                vm_hv = np.array(
                    [float(net.res_bus.at[b, "vm_pu"]) for b in hv.bus_indices
                     if b in net.res_bus.index],
                    dtype=np.float64,
                )
                if vm_hv.size:
                    rec.dso_group_v_min_pu[hv.net_id]  = float(vm_hv.min())
                    rec.dso_group_v_max_pu[hv.net_id]  = float(vm_hv.max())
                    rec.dso_group_v_mean_pu[hv.net_id] = float(vm_hv.mean())
                    # Per-bus DSO voltages for the Gate-E comparison.
                    for b in hv.bus_indices:
                        if b in net.res_bus.index:
                            rec.bus_vm_pu[int(b)] = float(
                                net.res_bus.at[b, "vm_pu"])
                rec.dso_controller_group[hv.net_id] = hv.net_id

                # DER reactive power per HV group (post-PF measurement of
                # local-mode Q(V) droop or cos phi=1 dispatch).  Without
                # this the cascade-DSO live plot's "DER Q per HV group"
                # tile shows "no DSO DER dispatch available" in any
                # scenario with dso_mode='local' (L0/L1/L2 and T-OFO).
                # The ±sn_mva headroom band is a static box approximation
                # to the VDE-AR-N 4120 capability used by the OFO path
                # (compute_der_q_bounds); it is sufficient for live
                # visualisation in local mode where no controller drives
                # the DER toward the precise envelope.
                valid_der = [
                    s for s in hv.sgen_indices if s in net.res_sgen.index
                ]
                if valid_der:
                    rec.dso_group_q_der_mvar[hv.net_id] = float(
                        net.res_sgen.loc[valid_der, "q_mvar"].sum()
                    )
                    sn_total = float(
                        net.sgen.loc[valid_der, "sn_mva"].sum()
                    )
                    rec.dso_group_q_der_min_mvar[hv.net_id] = -sn_total
                    rec.dso_group_q_der_max_mvar[hv.net_id] = +sn_total
            _record_local_dso_trafo_data(rec, net, hv_info_map)
            _record_hv_group_observables(rec, net, hv_info_map)

            # Expose the TSO-dispatched interface Q setpoint per coupling trafo
            # so one-sided-OFO variants (TSO OFO + local DSO) report the same
            # interface tracking-error field as the cascaded path.  Keyed
            # identically to _record_local_dso_trafo_data ("{net_id}|trafo_{t}").
            # Empty when the TSO is also local (no setpoint dispatched).
            for hv in meta.hv_networks:
                for _t in hv.coupling_trafo_indices:
                    _t = int(_t)
                    if _t in last_pcc_set_per_trafo:
                        rec.dso_trafo_q_set_mvar[f"{hv.net_id}|trafo_{_t}"] = (
                            last_pcc_set_per_trafo[_t]
                        )

        # ── Record plant voltages per zone ────────────────────────────────────
        for z, zd in zone_defs.items():
            vm_bus_ids = [
                int(b) for b in zd.v_bus_indices if b in net.res_bus.index
            ]
            vm_zone = np.array(
                [float(net.res_bus.at[b, "vm_pu"]) for b in vm_bus_ids],
                dtype=np.float64,
            )
            if vm_zone.size > 0:
                rec.zone_v_min[z]  = float(vm_zone.min())
                rec.zone_v_max[z]  = float(vm_zone.max())
                rec.zone_v_mean[z] = float(vm_zone.mean())
                # Per-bus values too: the envelope alone cannot support a
                # bus-by-bus static-vs-RMS comparison (Gate E, 2026-07-21).
                for b, v in zip(vm_bus_ids, vm_zone):
                    rec.bus_vm_pu[int(b)] = float(v)
                # Spatial RMS of the voltage error to the setpoint across the
                # zone's observed EHV buses (CIGRE per-zone tracking figure).
                cfg_v = tso_controllers[z].config
                if cfg_v.v_setpoints_pu is None:
                    v_ref_zone = np.full(vm_zone.shape, v_set, dtype=float)
                else:
                    v_ref_by_bus = {
                        int(bus): float(ref)
                        for bus, ref in zip(
                            cfg_v.voltage_bus_indices, cfg_v.v_setpoints_pu
                        )
                    }
                    v_ref_zone = np.array(
                        [v_ref_by_bus.get(bus, v_set) for bus in vm_bus_ids],
                        dtype=float,
                    )
                rec.zone_v_rms_err_pu[z] = float(
                    np.sqrt(np.mean((vm_zone - v_ref_zone) ** 2))
                )

            # Generator P, Q from converged power flow (every step).
            # Live plots consume these each update, so they cannot be gated
            # on run_tso.
            if zd.gen_indices:
                rec.zone_q_gen[z] = np.array(
                    [net.res_gen.at[idx, "q_mvar"] for idx in zd.gen_indices],
                    dtype=np.float64,
                )
                rec.zone_p_gen[z] = np.array(
                    [net.res_gen.at[idx, "p_mw"] for idx in zd.gen_indices],
                    dtype=np.float64,
                )
                # Synchronous-machine Q headroom from the Milano §12.2.1
                # capability curve — matches the bound that the TSO MIQP
                # actually enforces (see TSOController._build_constraint_bounds
                # at controller/tso_controller.py:1000).  At each step the
                # bound is recomputed from the current P and terminal V via
                # ActuatorBounds.compute_gen_q_bounds.
                # Headroom = signed min margin: min(q_max - q, q - q_min).
                # Positive = inside envelope; negative = capability violated.
                gen_vm = np.array(
                    [float(net.gen.at[g, "vm_pu"]) for g in zd.gen_indices],
                    dtype=np.float64,
                )
                q_min_cap, q_max_cap = tso_controllers[z].actuator_bounds.compute_gen_q_bounds(
                    rec.zone_p_gen[z], gen_vm,
                )
                q_act = rec.zone_q_gen[z]
                rec.gen_q_headroom_mvar[z] = np.minimum(
                    q_max_cap - q_act, q_act - q_min_cap,
                )
                # Normalised reserve r_Q = min(q_max-q, q-q_min)/(q_max-q_min)
                # for the TRACKING ERRORS & RESERVES live plot.  NaN where the
                # capability band collapses (q_max == q_min).
                _gen_rng = q_max_cap - q_min_cap
                rec.gen_q_reserve[z] = np.where(
                    _gen_rng > 1e-9,
                    np.minimum(q_max_cap - q_act, q_act - q_min_cap) / _gen_rng,
                    np.nan,
                )

            # ── Live-plot ACTUATORS tiles (TSO controller live plot) ──────
            # Populated from net state every step in BOTH OFO and local
            # modes:
            #   - zone_q_der:     net.sgen.q_mvar at each TSO DER index
            #   - zone_v_gen:     net.gen.vm_pu (AVR setpoint, constant in
            #                     local mode; OFO writes it from u_new)
            #   - zone_oltc_taps: net.trafo.tap_pos at machine 2W indices
            # The OFO TSO step also writes these from u_new on TSO ticks
            # (every 3 min); reading from net state every step gives smooth
            # time series across both modes.  Values reflect the converged
            # PF (PF does not modify sgen.q_mvar / gen.vm_pu / trafo.tap_pos
            # so for OFO they equal the commanded values).
            # zone_q_der: realised Q of each TSO-connected DER sgen in the
            # zone (the live plot's TSO DER reactive infeed).
            der_q_following = (
                np.array(
                    [float(net.res_sgen.at[idx, "q_mvar"])
                     for idx in zd.tso_der_indices],
                    dtype=np.float64,
                )
                if zd.tso_der_indices
                else np.array([], dtype=np.float64)
            )
            if der_q_following.size:
                rec.zone_q_der[z] = der_q_following
                # Normalised TSO-DER reserve r_Q from the VDE/STATCOM
                # capability band (TRACKING ERRORS & RESERVES live plot).
                # Order matches zd.tso_der_indices == actuator_bounds.der_indices.
                der_p = np.array(
                    [float(net.res_sgen.at[idx, "p_mw"])
                     for idx in zd.tso_der_indices],
                    dtype=np.float64,
                )
                q_min_der, q_max_der = (
                    tso_controllers[z].actuator_bounds.compute_der_q_bounds(der_p)
                )
                _der_rng = q_max_der - q_min_der
                rec.tso_der_q_reserve[z] = np.where(
                    _der_rng > 1e-9,
                    np.minimum(q_max_der - der_q_following,
                               der_q_following - q_min_der) / _der_rng,
                    np.nan,
                )
            if zd.gen_indices:
                rec.zone_v_gen[z] = np.array(
                    [float(net.gen.at[idx, "vm_pu"]) for idx in zd.gen_indices],
                    dtype=np.float64,
                )
            if zd.oltc_trafo_indices:
                rec.zone_oltc_taps[z] = np.array(
                    [int(net.trafo.at[idx, "tap_pos"]) for idx in zd.oltc_trafo_indices],
                    dtype=np.int64,
                )

            # Zone's own EHV line loss [MW], ground truth over exactly the
            # lines its TSO loss objective (g_loss, current_line_indices)
            # targets -- see MultiTSOIterationRecord.zone_losses_mw docstring.
            rec.zone_losses_mw[z] = (
                float(net.res_line.loc[zd.line_indices, "pl_mw"].sum())
                if zd.line_indices else 0.0
            )

        # ── Total network losses (single scalar per record) ──────────────────
        rec.total_losses_mw = (
            float(net.res_line["pl_mw"].sum())
            + float(net.res_trafo["pl_mw"].sum())
            + float(net.res_trafo3w["pl_mw"].sum())
        )

        # ── Slack saturation diagnostic (added 2026-05-02) ───────────────────
        # Records the slack's P/Q every step plus a flag for whether |Q| is
        # within 1 % of max_q_mvar (saturation indicator).  Helps post-hoc
        # diagnosis of L0 / cos-phi-1 divergence: if the slack pegs at its
        # capability limit before NR fails, that is the proximate cause.
        if "slack" in net.gen.columns and len(net.gen) > 0:
            _slack_idxs = net.gen.index[net.gen["slack"].astype(bool)].tolist()
            if _slack_idxs:
                _sg = _slack_idxs[0]
                rec.slack_p_mw   = float(net.res_gen.at[_sg, "p_mw"])
                rec.slack_q_mvar = float(net.res_gen.at[_sg, "q_mvar"])
                _qmax = float(net.gen.at[_sg, "max_q_mvar"])
                _qmin = float(net.gen.at[_sg, "min_q_mvar"])
                _qabs_lim = max(abs(_qmax), abs(_qmin), 1.0)
                rec.slack_q_at_limit = bool(
                    abs(rec.slack_q_mvar) >= 0.99 * _qabs_lim
                )
        elif not net.ext_grid.empty:
            _sg = net.ext_grid.index[0]
            rec.slack_p_mw   = float(net.res_ext_grid.at[_sg, "p_mw"])
            rec.slack_q_mvar = float(net.res_ext_grid.at[_sg, "q_mvar"])
            rec.slack_q_at_limit = False  # ext_grid is unbounded

        # ── Record per-zone live-plot observables (loadings, balances,
        #    tie-line Q, shunt states) every step.
        _record_zone_live_plot_observables(
            rec=rec, net=net,
            zone_defs=zone_defs, tn_zone_map=tn_zone_map,
            tie_line_map=tie_line_map,
        )

        # Controller-facing analogue snapshots: these are the same noisy
        # pre-control packets consumed by the OFO controllers.  Exact plant
        # observables above remain available to the system/tracking views.
        if tso_plot_measurements:
            _record_tso_measurement_snapshot(
                rec, tso_plot_measurements, zone_defs, tie_line_map,
                tso_controllers, default_v_setpoint_pu=v_set,
            )
        if dso_plot_measurements:
            _record_dso_measurement_snapshot(
                rec,
                dso_plot_measurements,
                dso_controllers,
                dso_group_map,
            )
        # Integrator mode: the recorder reads zd.shunt_bus_indices (empty here),
        # so populate the live-plot shunt states from the integrator banks'
        # committed pandapower steps (read by explicit shunt index — a tertiary
        # hosts both an MSC and an MSR).  Order matches the bank ordering.
        if _shunt_mode == "integrator":
            for z_s, integ_s in zone_integrators.items():
                # Sign the committed step by device class for the live plot,
                # matching the legacy bipolar convention (+ = inductive/reactor,
                # - = capacitive): MSR positive, MSC negative.
                rec.zone_tso_shunt_states[z_s] = np.asarray(
                    [
                        int(net.shunt.at[b.config.shunt_idx, "step"])
                        * (-1 if b.config.kind == "MSC" else 1)
                        for b in integ_s.banks
                    ],
                    dtype=np.int64,
                )

        # ── Print progress ────────────────────────────────────────────────────
        if verbose >= 1 and run_tso:
            min_num = int(time_s / 60)
            v_info = "  ".join(
                f"Z{z}: [{rec.zone_v_min.get(z, float('nan')):.3f}, "
                f"{rec.zone_v_max.get(z, float('nan')):.3f}] p.u."
                for z in sorted(zone_defs.keys())
            )
            print(f"  t={min_num:3d} min | {v_info}")
            # ── Adaptive g_w live state per controller ─────────────────────
            # For each adapted class, show ``min..max`` of g_w_live across
            # the variables in that class.  Single-actuator classes (or
            # classes whose entries are all equal) collapse to one number.
            # Lines are skipped entirely when no controller has an active
            # adapter so the baseline log stays clean.
            def _fmt_class(arr):
                vmin = float(arr.min())
                vmax = float(arr.max())
                if arr.size == 1 or vmin == vmax:
                    return f"{vmin:.3g}"
                return f"{vmin:.3g}..{vmax:.3g}"

            tso_parts = []
            for z in sorted(tso_controllers.keys()):
                s = tso_controllers[z].adapter_summary()
                if s:
                    kv = " ".join(f"{k}={_fmt_class(v)}" for k, v in s.items())
                    tso_parts.append(f"Z{z}[{kv}]")
            if tso_parts:
                print(f"  t={min_num:3d} min | g_w TSO  {' '.join(tso_parts)}")
            dso_parts = []
            for did in sorted(dso_controllers.keys()):
                s = dso_controllers[did].adapter_summary()
                if s:
                    kv = " ".join(f"{k}={_fmt_class(v)}" for k, v in s.items())
                    dso_parts.append(f"{did}[{kv}]")
            if dso_parts:
                print(f"  t={min_num:3d} min | g_w DSO  {' '.join(dso_parts)}")
            if verbose >= 2:
                for z in sorted(zone_defs.keys()):
                    lhs = rec.zone_contraction_lhs.get(z, float("nan"))
                    print(f"    Zone {z}: contraction_lhs={lhs:.3f}  "
                          f"obj={rec.zone_tso_objective.get(z, float('nan')):.4e}")

        # ── Collect load-balance aggregates ──────────────────────────────
        _non_bound = ~net.sgen["name"].astype(str).str.startswith("BOUND_")
        rec.total_load_p_mw    = float(net.load["p_mw"].sum())
        rec.total_load_q_mvar  = float(net.load["q_mvar"].sum())
        rec.total_sgen_p_mw    = float(net.sgen.loc[_non_bound, "p_mw"].sum())
        rec.total_gen_p_mw     = float(net.res_gen["p_mw"].sum()) + float(net.res_ext_grid["p_mw"].sum())
        rec.total_gen_q_mvar   = float(net.res_gen["q_mvar"].sum()) + float(net.res_ext_grid["q_mvar"].sum())
        rec.residual_load_p_mw = rec.total_load_p_mw - rec.total_sgen_p_mw

        if _plotter_tso is not None:
            _plotter_tso.update(rec)
        if _plotter_dso is not None:
            _plotter_dso.update(rec)
        if _plotter_sys is not None:
            _plotter_sys.update(rec)
        if _plotter_track is not None:
            _plotter_track.update(rec)
        if _plotter_sbx is not None:
            # True reference-end corridor flow: q measured AT each tie's
            # bus_a endpoint (q_from or q_to per line orientation).
            # rec.zone_tie_q_mvar is NOT usable here: it negates q_from
            # for lines oriented from the higher zone, which misstates
            # heavily charged ties (line 14: ~107 Mvar of charging)
            # relative to the SBX schedule convention.
            _sbx_ad = sbx_runtime["adapter"]
            _sbx_q = None
            if _sbx_ad is not None:
                _sbx_q = {}
                for _ck, _corr in _sbx_ad.registry.items():
                    _tot = 0.0
                    for _ln in _corr.lines:
                        if int(net.line.at[_ln.line_idx, "from_bus"]) \
                                == _ln.bus_a:
                            _tot += float(net.res_line.at[
                                _ln.line_idx, "q_from_mvar"])
                        else:
                            _tot += float(net.res_line.at[
                                _ln.line_idx, "q_to_mvar"])
                    _sbx_q[_ck] = _tot
            _plotter_sbx.update(rec, _sbx_ad, _sbx_q)

        # ── SBX-V four-quadrant metering: record the elapsed plant
        # interval [t − dt, t) with this step's post-power-flow boundary
        # Q per NVP (sbxv/metering.py; [LF §5.5, §8.3]).
        if config.coordination_mode == "sbxv" \
                and sbxv_runtime["adapter"] is not None:
            sbxv_runtime["adapter"].on_plant_step(
                time_s, config.dt_s, net,
            )
            if _plotter_sbxv is not None:
                _plotter_sbxv.update(
                    rec, sbxv_runtime["adapter"],
                )

        if not _in_warmup:
            log.append(rec)

        if verbose >= 1:
            _dt_step = perf_counter() - _t_step
            _flags = []
            if run_tso:
                _flags.append("T")
            if run_dso:
                _flags.append("D")
            if _contingency_fired_this_step:
                _flags.append("X")
            if not _needs_end_pf:
                _flags.append("skip-endPF")
            _flag_str = ",".join(_flags) if _flags else "-"
            print(f"  [T] step {step:4d} t={time_s/60.0:6.1f} min  "
                  f"wall={_dt_step:5.2f} s  [{_flag_str}]")

        # ── Delayed auto-tune + stability analysis ──────────────────────
        # Triggered once when the simulated time crosses
        # ``config.stability_analysis_at_s``.  By default this is t=60
        # min, giving the controller time to equilibrate before we
        # auto-tune and analyse the operating point.  Running either at
        # t=0 would produce misleading results because the uncontrolled
        # initial state still has large tracking gradients.
        #
        # Sequence:
        #   1. (if config.run_stability_analysis) run the multi-zone
        #      stability report, print the compact summary, and write a
        #      markdown report in ``config.result_dir``.
        if (not _stability_analysis_done
                and time_s >= config.stability_analysis_at_s):
            _stability_analysis_done = True
            # NOTE: g_w tuning now runs at t=0 (before the main loop),
            # not here.  Only the delayed stability report remains.
            # Skip stability analysis entirely in TSO local mode: the
            # multi-zone OFO controllers are bypassed, so the spectral-gap
            # analysis is not meaningful (and would dereference state that
            # the local-mode runner never populates).
            if config.run_stability_analysis and not _local_tso:
                try:
                    _run_delayed_stability_analysis(
                        config=config,
                        time_s=time_s,
                        net=net,
                        coordinator=coordinator,
                        zone_defs=zone_defs,
                        tso_controllers=tso_controllers,
                        dso_controllers=dso_controllers,
                        hv_info_map=hv_info_map,
                        verbose=verbose,
                    )
                except Exception as _exc:
                    if verbose >= 1:
                        print(f"  WARNING: delayed stability analysis failed: {_exc}")

    # =========================================================================
    # STEP 14: Print final summary
    # =========================================================================
    if verbose >= 1:
        print()
        print("=" * 72)
        print("  FINAL SUMMARY")
        print("=" * 72)
        for z, zd in zone_defs.items():
            last_rec = next((r for r in reversed(log) if z in r.zone_v_mean), None)
            if last_rec is None:
                continue
            v_mean = last_rec.zone_v_mean.get(z, float("nan"))
            v_err  = abs(v_mean - v_set)
            print(f"  Zone {z}: V_mean={v_mean:.4f} p.u.  |V - V_set|={v_err:.4f}")
        print("=" * 72)

        # ── DSO tracking quality ─────────────────────────────────────────────
        print()
        print("=" * 72)
        print("  DSO Q-TRACKING QUALITY")
        print("=" * 72)
        for dso_id in sorted(set().union(*(r.dso_q_set_mvar.keys() for r in log))):
            q_sets = []
            q_acts = []
            for r in log:
                qs = r.dso_q_set_mvar.get(dso_id)
                qa = r.dso_q_actual_mvar.get(dso_id)
                if qs is not None and qa is not None:
                    q_sets.append(qs)
                    q_acts.append(qa)
            if q_sets:
                errors = [abs(s - a) for s, a in zip(q_sets, q_acts)]
                print(f"  {dso_id}: Q_set={q_sets[-1]:+8.2f} Mvar, "
                      f"Q_act={q_acts[-1]:+8.2f} Mvar, "
                      f"|err|={errors[-1]:.2f} Mvar, "
                      f"mean|err|={np.mean(errors):.2f} Mvar, "
                      f"max|err|={max(errors):.2f} Mvar")
        print("=" * 72)

    return log
