"""
sensitivity/numerical_h.py
==========================
Build the TSO / DSO controller H matrix by *finite-difference perturbation*
of the plant net, instead of the analytical Jacobian-based formulas in
:mod:`sensitivity.jacobian`.

Purpose
-------
Used by ``experiments/004b_REFRESH_PROOF.py`` (and friends) to test
whether the FULL-mode steady-state Q-tracking gap vs LOCAL comes from
the analytical sensitivity *computation* (formula bias) rather than the
boundary-modeling choice.  If numerical H tracks the analytical H, the
gap is purely structural (boundary).  If numerical H differs and tracks
the plant better, there is room to improve the analytical implementation.

Method
------
For each input column ``k`` of the controller's H matrix the function:

1.  Records the controller's outputs ``y_0`` at the current operating
    point.
2.  Perturbs the corresponding plant-side actuator by ``+δ`` and runs
    ``pp.runpp(run_control=True)`` so the plant-side Q(V) loops respond
    to convergence.
3.  Records ``y_plus``.
4.  Repeats with ``-δ`` to get ``y_minus``.
5.  Restores the original actuator value and runs PF again so the next
    perturbation starts from the cached OP.
6.  ``H[:, k] = (y_plus - y_minus) / (2 δ)``.

The output vector ``y`` matches the row layout used by the analytical
builders:

* TSO: ``[V_bus | Q_PCC | I_line | Q_gen | Q_gf | Q_tie]``
* DSO: ``[Q_iface | V_bus | I_line]``

Perturbation magnitudes (configurable via constructor args):

* DER Q (sgen ``q_set_mvar``): ±1 Mvar — closed-loop ``q_set`` actuator,
  the QVLocalLoop sees the new setpoint and the realised ``q_mvar``
  settles via the local droop.
* PCC Q-setpoint (modelled as Q-load at the trafo HV bus): ±1 Mvar.
* V_gen (``net.gen.vm_pu``): ±0.001 p.u. — chosen small enough that
  AVR-stiff response stays linear, large enough to escape PF
  convergence noise.
* OLTC tap: ±1 step.
* Shunt step: ±1 step.

Costs roughly ``2 · n_inputs`` plant power-flow calls per H build —
about 1-3 s per controller on IEEE 39 with HV sub-networks.

Author: Manuel Schwenke / Claude Code
Date: 2026-05-27
"""
from __future__ import annotations

import copy
from typing import Iterable, List, Optional

import numpy as np
from numpy.typing import NDArray
import pandapower as pp


# ---------------------------------------------------------------------------
#  Output reader (shared between TSO and DSO)
# ---------------------------------------------------------------------------

def _read_tso_outputs(net, cfg) -> NDArray[np.float64]:
    """Read the TSO output vector ``[V | Q_PCC | I | Q_gen | Q_gf | Q_tie]``.

    Q_PCC follows the analytical builder's load convention (positive =
    Q INTO the trafo from the HV bus), which is the same sign as
    ``net.res_trafo3w.q_hv_mvar`` in pandapower.

    Q_tie sign convention: positive = Q leaving the zone through the
    in-zone endpoint, i.e. ``q_from_mvar`` when the in-zone endpoint is
    the from-bus, else ``q_to_mvar``.  Matches the analytical builder.
    """
    y: List[float] = []

    # V_bus
    for b in cfg.voltage_bus_indices:
        y.append(float(net.res_bus.at[b, "vm_pu"]) if b in net.res_bus.index else 0.0)

    # Q_PCC (load convention; 3W coupler q_hv_mvar)
    for t in cfg.pcc_trafo_indices:
        if hasattr(net, "trafo3w") and t in net.res_trafo3w.index:
            y.append(float(net.res_trafo3w.at[t, "q_hv_mvar"]))
        elif t in net.res_trafo.index:
            y.append(float(net.res_trafo.at[t, "q_hv_mvar"]))
        else:
            y.append(0.0)

    # I_line
    for li in cfg.current_line_indices:
        y.append(float(net.res_line.at[li, "i_ka"]) if li in net.res_line.index else 0.0)

    # Q_gen
    for g in cfg.gen_indices:
        y.append(float(net.res_gen.at[g, "q_mvar"]) if g in net.res_gen.index else 0.0)

    # Q_gf (grid-forming gens)
    gf_indices = getattr(cfg, "gridforming_gen_indices", []) or []
    for g in gf_indices:
        y.append(float(net.res_gen.at[g, "q_mvar"]) if g in net.res_gen.index else 0.0)

    # Q_tie
    tie_indices = getattr(cfg, "tie_line_indices", []) or []
    tie_endpoints = getattr(cfg, "tie_line_endpoint_buses", []) or []
    for li, in_bus in zip(tie_indices, tie_endpoints):
        if li in net.res_line.index:
            from_bus = int(net.line.at[li, "from_bus"])
            if in_bus == from_bus:
                y.append(float(net.res_line.at[li, "q_from_mvar"]))
            else:
                y.append(float(net.res_line.at[li, "q_to_mvar"]))
        else:
            y.append(0.0)

    return np.asarray(y, dtype=np.float64)


def _read_dso_outputs(net, cfg) -> NDArray[np.float64]:
    """Read the DSO output vector ``[Q_iface | V_bus | I_line]``.

    ``Q_iface`` follows the same load convention as TSO ``Q_PCC``.
    """
    y: List[float] = []

    # Q_iface (HV-side Q at the interface 3W coupler)
    for t in cfg.interface_trafo_indices:
        if hasattr(net, "trafo3w") and t in net.res_trafo3w.index:
            y.append(float(net.res_trafo3w.at[t, "q_hv_mvar"]))
        elif t in net.res_trafo.index:
            y.append(float(net.res_trafo.at[t, "q_hv_mvar"]))
        else:
            y.append(0.0)

    # V_bus
    for b in cfg.voltage_bus_indices:
        y.append(float(net.res_bus.at[b, "vm_pu"]) if b in net.res_bus.index else 0.0)

    # I_line
    for li in cfg.current_line_indices:
        y.append(float(net.res_line.at[li, "i_ka"]) if li in net.res_line.index else 0.0)

    return np.asarray(y, dtype=np.float64)


# ---------------------------------------------------------------------------
#  Common PF helper
# ---------------------------------------------------------------------------

def _runpp(net, *, run_control: bool = True, max_iter: int = 100) -> None:
    """Single point of contact for ``pp.runpp`` with the standard
    closed-loop settings used by the runner."""
    pp.runpp(
        net,
        run_control=run_control,
        calculate_voltage_angles=True,
        max_iteration=max_iter,
        max_iter=300,
        distributed_slack=False,
        enforce_q_lims=False,
    )


# ---------------------------------------------------------------------------
#  TSO numerical H
# ---------------------------------------------------------------------------

def compute_numerical_h_tso(
    net,
    ctrl,
    *,
    delta_q_mvar: float = 1.0,
    delta_v_pu: float = 0.001,
    closed_loop: bool = True,
    verbose: int = 0,
) -> NDArray[np.float64]:
    """Build the TSO controller's H matrix via plant-level finite difference.

    Parameters
    ----------
    net : pp.pandapowerNet
        Plant network at the cached operating point.  A deep copy is
        used internally; the original is not modified.
    ctrl : TSOController
        Source of the index sets (``ctrl.config``) and OOS masks
        (``ctrl._oos_oltc_mask``, ``ctrl._oos_gen_mask``).
    delta_q_mvar : float
        ±Mvar perturbation magnitude for DER and PCC_set columns.
    delta_v_pu : float
        ±p.u. perturbation magnitude for V_gen and V_gf columns.
    closed_loop : bool, default True
        If True, perturbations run ``pp.runpp(run_control=True)`` so the
        plant-side Q(V) loops respond during the finite-difference (the
        DER column then represents ``∂y/∂q_set`` directly; V_gen / OLTC
        / shunt columns include any QV-loop reaction to those moves
        too).  If False, perturbations run ``run_control=False`` (pure
        algebraic plant response, no QV-loop reaction) and the
        analytical ``T_prime = (I + diag(K)·S_VQ)^{-1}`` transform is
        applied to the DER columns post-hoc, mirroring the analytical
        builder's structure.  The latter is the right choice for an
        apples-to-apples test of the analytical computation: only the
        base ``∂y/∂Q_DER`` block differs between numerical-open-loop
        and analytical, while T_prime, V_gen / OLTC / shunt columns
        share the same semantics across both methods.

    Returns
    -------
    H : NDArray[np.float64]
        Sensitivity matrix in the controller's layout:

        * rows: ``[V_bus | Q_PCC | I_line | Q_gen | Q_gf | Q_tie]``
        * cols: ``[Q_DER | Q_PCC_set | V_gen | V_gf | OLTC | shunt]``
    """
    cfg = ctrl.config

    n_der = len(cfg.der_indices)
    n_pcc = len(cfg.pcc_trafo_indices)
    n_gen = len(cfg.gen_indices)
    n_gf = len(getattr(cfg, "gridforming_gen_indices", []) or [])
    n_oltc = len(cfg.oltc_trafo_indices)
    n_shunt = len(cfg.shunt_bus_indices)
    n_v = len(cfg.voltage_bus_indices)
    n_i = len(cfg.current_line_indices)
    n_tie = len(getattr(cfg, "tie_line_indices", []) or [])

    n_inputs = n_der + n_pcc + n_gen + n_gf + n_oltc + n_shunt
    n_outputs = n_v + n_pcc + n_i + n_gen + n_gf + n_tie
    H = np.zeros((n_outputs, n_inputs), dtype=np.float64)

    if n_inputs == 0 or n_outputs == 0:
        return H

    work = copy.deepcopy(net)
    # Establish baseline PF with run_control so the plant-side QV loops
    # are at convergence before the first perturbation.
    _runpp(work, run_control=True)

    # ── PCC HV buses (used for Q_PCC_set column injection) ──
    pcc_hv_buses: List[int] = []
    pcc_in_trafo3w = (
        hasattr(work, "trafo3w") and not work.trafo3w.empty
        and all(t in work.trafo3w.index for t in cfg.pcc_trafo_indices)
    )
    for t in cfg.pcc_trafo_indices:
        if pcc_in_trafo3w:
            pcc_hv_buses.append(int(work.trafo3w.at[t, "hv_bus"]))
        elif t in work.trafo.index:
            pcc_hv_buses.append(int(work.trafo.at[t, "hv_bus"]))
        else:
            pcc_hv_buses.append(-1)

    col = 0

    # ── DER columns (Q_set actuator; closed-loop via QVLocalLoop) ──
    for k, s_idx in enumerate(cfg.der_indices):
        if s_idx not in work.sgen.index:
            col += 1
            continue
        col_idx = "q_set_mvar" if "q_set_mvar" in work.sgen.columns else "q_mvar"
        q0 = float(work.sgen.at[s_idx, col_idx])
        work.sgen.at[s_idx, col_idx] = q0 + delta_q_mvar
        _runpp(work, run_control=closed_loop)
        y_plus = _read_tso_outputs(work, cfg)
        work.sgen.at[s_idx, col_idx] = q0 - delta_q_mvar
        _runpp(work, run_control=closed_loop)
        y_minus = _read_tso_outputs(work, cfg)
        H[:, col] = (y_plus - y_minus) / (2.0 * delta_q_mvar)
        work.sgen.at[s_idx, col_idx] = q0
        col += 1
    # Restore baseline PF so next-block perturbations start clean
    _runpp(work, run_control=True)

    # ── Q_PCC_set columns (modelled as Q-load at the HV bus) ──
    # Add a load at the HV bus, perturb its q_mvar.  Then set the
    # diagonal Q_PCC row to 1.0 to match the controller's identity
    # convention (the DSO is assumed to track its Q_PCC,set perfectly).
    for k, (hv_bus, t_idx) in enumerate(zip(pcc_hv_buses, cfg.pcc_trafo_indices)):
        if hv_bus < 0 or hv_bus not in work.bus.index:
            col += 1
            continue
        ld_idx = pp.create_load(
            work, bus=int(hv_bus), q_mvar=delta_q_mvar, p_mw=0.0,
            name="_NUMH_PCC_SET",
        )
        _runpp(work, run_control=closed_loop)
        y_plus = _read_tso_outputs(work, cfg)
        work.load.at[ld_idx, "q_mvar"] = -delta_q_mvar
        _runpp(work, run_control=closed_loop)
        y_minus = _read_tso_outputs(work, cfg)
        H[:, col] = (y_plus - y_minus) / (2.0 * delta_q_mvar)
        # Identity convention on the Q_PCC diagonal
        H[n_v + k, col] = 1.0
        work.load.drop(index=ld_idx, inplace=True)
        col += 1
    _runpp(work, run_control=True)

    # ── V_gen columns (synchronous-machine AVR) ──
    oos_gen = getattr(ctrl, "_oos_gen_mask", np.zeros(n_gen, dtype=bool))
    for k, g_idx in enumerate(cfg.gen_indices):
        if k < len(oos_gen) and bool(oos_gen[k]):
            col += 1  # OOS gen → zero column
            continue
        if g_idx not in work.gen.index:
            col += 1
            continue
        v0 = float(work.gen.at[g_idx, "vm_pu"])
        work.gen.at[g_idx, "vm_pu"] = v0 + delta_v_pu
        _runpp(work, run_control=closed_loop)
        y_plus = _read_tso_outputs(work, cfg)
        work.gen.at[g_idx, "vm_pu"] = v0 - delta_v_pu
        _runpp(work, run_control=closed_loop)
        y_minus = _read_tso_outputs(work, cfg)
        H[:, col] = (y_plus - y_minus) / (2.0 * delta_v_pu)
        work.gen.at[g_idx, "vm_pu"] = v0
        col += 1
    _runpp(work, run_control=True)

    # ── V_gf columns (grid-forming gens) ──
    gf_indices = list(getattr(cfg, "gridforming_gen_indices", []) or [])
    for k, g_idx in enumerate(gf_indices):
        if g_idx not in work.gen.index:
            col += 1
            continue
        v0 = float(work.gen.at[g_idx, "vm_pu"])
        work.gen.at[g_idx, "vm_pu"] = v0 + delta_v_pu
        _runpp(work, run_control=closed_loop)
        y_plus = _read_tso_outputs(work, cfg)
        work.gen.at[g_idx, "vm_pu"] = v0 - delta_v_pu
        _runpp(work, run_control=closed_loop)
        y_minus = _read_tso_outputs(work, cfg)
        H[:, col] = (y_plus - y_minus) / (2.0 * delta_v_pu)
        work.gen.at[g_idx, "vm_pu"] = v0
        col += 1
    _runpp(work, run_control=True)

    # ── OLTC columns (machine 2W tap_pos, ±1 step) ──
    oos_oltc = getattr(ctrl, "_oos_oltc_mask", np.zeros(n_oltc, dtype=bool))
    for k, t_idx in enumerate(cfg.oltc_trafo_indices):
        if k < len(oos_oltc) and bool(oos_oltc[k]):
            col += 1
            continue
        if t_idx not in work.trafo.index:
            col += 1
            continue
        t0 = int(work.trafo.at[t_idx, "tap_pos"])
        # Respect tap bounds for the perturbation step.
        t_min = int(work.trafo.at[t_idx, "tap_min"]) if "tap_min" in work.trafo.columns else -10
        t_max = int(work.trafo.at[t_idx, "tap_max"]) if "tap_max" in work.trafo.columns else  10
        t_plus  = min(t0 + 1, t_max)
        t_minus = max(t0 - 1, t_min)
        # Central diff denominator may shrink at the bound — use the
        # actually-applied delta.
        d_plus  = t_plus  - t0
        d_minus = t0 - t_minus
        if d_plus + d_minus == 0:
            col += 1
            continue
        work.trafo.at[t_idx, "tap_pos"] = t_plus
        _runpp(work, run_control=closed_loop)
        y_plus = _read_tso_outputs(work, cfg)
        work.trafo.at[t_idx, "tap_pos"] = t_minus
        _runpp(work, run_control=closed_loop)
        y_minus = _read_tso_outputs(work, cfg)
        H[:, col] = (y_plus - y_minus) / float(d_plus + d_minus)
        work.trafo.at[t_idx, "tap_pos"] = t0
        col += 1
    _runpp(work, run_control=True)

    # ── Shunt columns (step ±1) ──
    for k, sb in enumerate(cfg.shunt_bus_indices):
        if work.shunt.empty:
            col += 1
            continue
        mask = work.shunt["bus"] == int(sb)
        if not mask.any():
            col += 1
            continue
        sh_idx = int(work.shunt.index[mask][0])
        s0 = int(work.shunt.at[sh_idx, "step"])
        s_min = int(work.shunt.at[sh_idx, "min_step"]) if "min_step" in work.shunt.columns else -10
        s_max = int(work.shunt.at[sh_idx, "max_step"]) if "max_step" in work.shunt.columns else  10
        s_plus  = min(s0 + 1, s_max)
        s_minus = max(s0 - 1, s_min)
        d_plus  = s_plus  - s0
        d_minus = s0 - s_minus
        if d_plus + d_minus == 0:
            col += 1
            continue
        work.shunt.at[sh_idx, "step"] = s_plus
        _runpp(work, run_control=closed_loop)
        y_plus = _read_tso_outputs(work, cfg)
        work.shunt.at[sh_idx, "step"] = s_minus
        _runpp(work, run_control=closed_loop)
        y_minus = _read_tso_outputs(work, cfg)
        H[:, col] = (y_plus - y_minus) / float(d_plus + d_minus)
        work.shunt.at[sh_idx, "step"] = s0
        col += 1

    # ── Apply analytical T_prime to the DER columns when in open-loop ──
    # mode.  In closed-loop mode the QV-droop response is already baked
    # into the perturbation outputs; T_prime would double-count it.
    # Gated by ``apply_qv_h_transform`` (default False => bare open-loop
    # H = dy/dQ_DER, matching the dissertation's reference-anchored model).
    if not closed_loop and n_der > 0 and ctrl.config.apply_qv_h_transform:
        try:
            der_bus_indices_tso = [
                int(net.sgen.at[s, "bus"]) for s in ctrl.config.der_indices
                if s in net.sgen.index
            ]
            if len(der_bus_indices_tso) == n_der:
                T_prime = ctrl._compute_w_shift_transform_T_prime(der_bus_indices_tso)
                if T_prime is not None:
                    H[:, :n_der] = H[:, :n_der] @ T_prime
                    if verbose >= 1:
                        print(f"    [numerical_h_tso {ctrl.controller_id}] "
                              f"applied analytical T_prime to DER columns")
        except Exception as _exc:
            if verbose >= 1:
                print(f"    [numerical_h_tso {ctrl.controller_id}] "
                      f"T_prime application skipped: {_exc}")

    if verbose >= 1:
        print(f"    [numerical_h_tso {ctrl.controller_id}] "
              f"shape={H.shape}  ||H||_F={float(np.linalg.norm(H)):.3e}  "
              f"(closed_loop={closed_loop})")

    return H


# ---------------------------------------------------------------------------
#  DSO numerical H
# ---------------------------------------------------------------------------

def compute_numerical_h_dso(
    net,
    ctrl,
    *,
    delta_q_mvar: float = 1.0,
    delta_v_pu: float = 0.001,
    closed_loop: bool = True,
    verbose: int = 0,
) -> NDArray[np.float64]:
    """Build the DSO controller's H matrix via plant-level finite difference.

    DSO layout (mirrors :meth:`DSOController._build_sensitivity_matrix`):

    * rows: ``[Q_iface | V_bus | I_line]``
    * cols: ``[DER (unique buses) | OLTC | shunt]``

    DER columns are at the *unique-bus* level (multiple DERs sharing a
    bus collapse to a single column), mirroring the controller's
    ``der_bus_indices`` deduplication.  Per-DER expansion is done by the
    base class via the ``E`` matrix; that step is unchanged.
    """
    cfg = ctrl.config

    der_buses_full = [int(net.sgen.at[s, "bus"]) for s in cfg.der_indices]
    unique_buses: List[int] = []
    bus_to_sgen: dict = {}  # one representative sgen per unique bus
    der_to_unique: List[int] = []
    for s, b in zip(cfg.der_indices, der_buses_full):
        if int(b) not in unique_buses:
            unique_buses.append(int(b))
            bus_to_sgen[int(b)] = int(s)
        der_to_unique.append(unique_buses.index(int(b)))
    n_der_bus = len(unique_buses)
    n_der_per = len(cfg.der_indices)
    n_oltc = len(cfg.oltc_trafo_indices)
    n_shunt = len(cfg.shunt_bus_indices)
    n_iface = len(cfg.interface_trafo_indices)
    n_v = len(cfg.voltage_bus_indices)
    n_i = len(cfg.current_line_indices)

    n_inputs_bus = n_der_bus + n_oltc + n_shunt
    n_outputs = n_iface + n_v + n_i
    H_bus = np.zeros((n_outputs, n_inputs_bus), dtype=np.float64)

    if n_inputs_bus == 0 or n_outputs == 0:
        # Return the per-DER-expanded shape for layout consistency
        return np.zeros((n_outputs, n_der_per + n_oltc + n_shunt), dtype=np.float64)

    work = copy.deepcopy(net)
    _runpp(work, run_control=True)

    col = 0

    # ── DER (unique-bus) columns ──
    # Perturb the representative sgen at each unique DER bus.
    for k, b in enumerate(unique_buses):
        s_idx = bus_to_sgen[b]
        if s_idx not in work.sgen.index:
            col += 1
            continue
        col_idx = "q_set_mvar" if "q_set_mvar" in work.sgen.columns else "q_mvar"
        q0 = float(work.sgen.at[s_idx, col_idx])
        work.sgen.at[s_idx, col_idx] = q0 + delta_q_mvar
        _runpp(work, run_control=closed_loop)
        y_plus = _read_dso_outputs(work, cfg)
        work.sgen.at[s_idx, col_idx] = q0 - delta_q_mvar
        _runpp(work, run_control=closed_loop)
        y_minus = _read_dso_outputs(work, cfg)
        H_bus[:, col] = (y_plus - y_minus) / (2.0 * delta_q_mvar)
        work.sgen.at[s_idx, col_idx] = q0
        col += 1
    _runpp(work, run_control=True)

    # ── OLTC columns (3W coupler tap, ±1) ──
    is_3w = (
        hasattr(work, "trafo3w") and not work.trafo3w.empty
        and all(t in work.trafo3w.index for t in cfg.oltc_trafo_indices)
    )
    oltc_table = work.trafo3w if is_3w else work.trafo
    for k, t_idx in enumerate(cfg.oltc_trafo_indices):
        if t_idx not in oltc_table.index:
            col += 1
            continue
        t0 = int(oltc_table.at[t_idx, "tap_pos"])
        t_min = int(oltc_table.at[t_idx, "tap_min"]) if "tap_min" in oltc_table.columns else -10
        t_max = int(oltc_table.at[t_idx, "tap_max"]) if "tap_max" in oltc_table.columns else  10
        t_plus  = min(t0 + 1, t_max)
        t_minus = max(t0 - 1, t_min)
        d_plus  = t_plus  - t0
        d_minus = t0 - t_minus
        if d_plus + d_minus == 0:
            col += 1
            continue
        oltc_table.at[t_idx, "tap_pos"] = t_plus
        _runpp(work, run_control=closed_loop)
        y_plus = _read_dso_outputs(work, cfg)
        oltc_table.at[t_idx, "tap_pos"] = t_minus
        _runpp(work, run_control=closed_loop)
        y_minus = _read_dso_outputs(work, cfg)
        H_bus[:, col] = (y_plus - y_minus) / float(d_plus + d_minus)
        oltc_table.at[t_idx, "tap_pos"] = t0
        col += 1
    _runpp(work, run_control=True)

    # ── Shunt columns ──  (DSO config has none in IEEE 39 setup, but defensive)
    for k, sb in enumerate(cfg.shunt_bus_indices):
        if work.shunt.empty:
            col += 1
            continue
        mask = work.shunt["bus"] == int(sb)
        if not mask.any():
            col += 1
            continue
        sh_idx = int(work.shunt.index[mask][0])
        s0 = int(work.shunt.at[sh_idx, "step"])
        work.shunt.at[sh_idx, "step"] = s0 + 1
        _runpp(work, run_control=closed_loop)
        y_plus = _read_dso_outputs(work, cfg)
        work.shunt.at[sh_idx, "step"] = s0 - 1
        _runpp(work, run_control=closed_loop)
        y_minus = _read_dso_outputs(work, cfg)
        H_bus[:, col] = (y_plus - y_minus) / 2.0
        work.shunt.at[sh_idx, "step"] = s0
        col += 1

    # ── Apply analytical T_prime to bus-level DER columns when in ─────
    # open-loop mode.  In closed-loop mode the QV-droop response is
    # already in the perturbation outputs.
    # Gated by ``apply_qv_h_transform`` (default False => bare open-loop
    # H = dy/dQ_DER, matching the dissertation's reference-anchored model).
    if not closed_loop and n_der_bus > 0 and ctrl.config.apply_qv_h_transform:
        try:
            T_prime = ctrl._compute_w_shift_transform_T_prime(
                unique_buses, der_buses_full,
            )
            if T_prime is not None:
                H_bus[:, :n_der_bus] = H_bus[:, :n_der_bus] @ T_prime
                if verbose >= 1:
                    print(f"    [numerical_h_dso {ctrl.controller_id}] "
                          f"applied analytical T_prime to DER columns")
        except Exception as _exc:
            if verbose >= 1:
                print(f"    [numerical_h_dso {ctrl.controller_id}] "
                      f"T_prime application skipped: {_exc}")

    # ── Expand unique-bus DER columns to per-DER (matching the
    # DSOController._build_sensitivity_matrix legacy expansion path) ──
    H_der_per = H_bus[:, :n_der_bus][:, der_to_unique]   # shape (n_outputs, n_der_per)
    H_rest = H_bus[:, n_der_bus:]                         # OLTC + shunt
    H = np.hstack([H_der_per, H_rest])

    if verbose >= 1:
        print(f"    [numerical_h_dso {ctrl.controller_id}] "
              f"bus-level shape={H_bus.shape}, per-DER shape={H.shape}, "
              f"||H||_F={float(np.linalg.norm(H)):.3e}  "
              f"(closed_loop={closed_loop})")

    return H


# ---------------------------------------------------------------------------
#  Per-DER column expansion (matches DSOController base behaviour)
# ---------------------------------------------------------------------------

def expand_dso_h_to_per_der(
    H_bus_level: NDArray[np.float64],
    cfg,
) -> NDArray[np.float64]:
    """Expand the bus-level DSO H (one column per unique DER bus) to
    per-DER (one column per ``cfg.der_indices``).  All DERs sharing a
    bus get the same bus column.

    Mirrors the legacy per-DER expansion that ``DSOController._build_sensitivity_matrix``
    applies after calling its bus-level builder.
    """
    n_der = len(cfg.der_indices)
    n_oltc = len(cfg.oltc_trafo_indices)
    n_shunt = len(cfg.shunt_bus_indices)
    n_outputs = H_bus_level.shape[0]

    # Map each DER to its unique-bus column index.
    der_buses_full = list(map(int, cfg.der_indices))  # placeholder; resolved below
    # Caller must supply the live net; instead we reconstruct from the
    # bus order in H_bus_level's input layout.  The DSO controller
    # itself rebuilds der_to_unique from the *current* net.sgen.bus, so
    # this helper is only safe to call when the bus layout is stable.
    raise NotImplementedError(
        "expand_dso_h_to_per_der requires the live net for bus lookups; "
        "expand inside the caller by reading ``net.sgen.at[s,'bus']``."
    )
