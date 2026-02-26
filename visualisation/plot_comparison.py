"""
Reference vs. Cascade Comparison Plots
=======================================

All plotting functions for comparing the MINLP AC-OPF reference against
the cascaded TSO-DSO OFO controller.  Extracted from ``run_reference.py``
so that the runner module stays focused on simulation logic.

Each function takes pre-computed metric arrays (produced by
``compare_results()`` in ``run_reference.py``) and generates one
matplotlib figure.

Author: Claude (generated)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from run_reference import ReferenceResult


# =============================================================================
#  Helper
# =============================================================================

def _forward_fill(
        common_mins: list[int],
        by_min: dict,
        extractor,
        n_cols: int,
        *,
        dtype=float,
) -> np.ndarray:
    """
    Build a (T × n_cols) array from a cascade log by forward-filling the last
    known controller output across timesteps where the controller did not fire.

    Parameters
    ----------
    common_mins : list[int]
        Ordered list of simulation minutes to extract.
    by_min : dict[int, IterationRecord]
        Minute-indexed cascade log.
    extractor : callable
        ``extractor(rec) -> NDArray | None``  —  returns the state vector for
        this record, or ``None`` if the controller did not fire this step.
    n_cols : int
        Length of the state vector.
    dtype : type, optional
        NumPy dtype for the output array.  Default: ``float``.

    Returns
    -------
    np.ndarray, shape (len(common_mins), n_cols)
        NaN where no state has been observed yet; forward-filled thereafter.
    """
    out = np.full((len(common_mins), n_cols), np.nan, dtype=float)
    last = None
    for i, m in enumerate(common_mins):
        rec = by_min.get(m)
        if rec is not None:
            val = extractor(rec)
            if val is not None:
                last = np.asarray(val, dtype=float)
        if last is not None:
            out[i] = last
    return out.astype(dtype) if dtype != float else out


# =============================================================================
#  Voltage RMSD
# =============================================================================

def plot_voltage_rmsd_comparison(
    time_minutes: list[int],
    ref_rmsd_tn: list[float],
    cas_rmsd_tn: list[float],
    ref_rmsd_dn: list[float],
    cas_rmsd_dn: list[float],
) -> None:
    """
    Two-panel time-series plot of per-timestep voltage RMSD (weight-free).

    Values are displayed in milli-p.u. (×1000) for readability since
    transmission-level deviations are typically 1–50 mp.u.

    Panel 1: TN (380 kV) buses  σ_V^TN(t).
    Panel 2: DN (110 kV) buses  σ_V^DN(t).
    """
    t = np.array(time_minutes)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(t, np.array(ref_rmsd_tn) * 1e3,
             label='MINLP Reference', linewidth=2,
             drawstyle='steps-post', color='#1f77b4')
    ax1.plot(t, np.array(cas_rmsd_tn) * 1e3,
             label='Cascaded OFO', linewidth=2, linestyle='--',
             drawstyle='steps-post', color='#ff7f0e')
    ax1.set_title(
        r'TN (380 kV) Voltage RMSD  '
        r'$\sigma_V^\mathrm{TN}(t) = \sqrt{\frac{1}{N_\mathrm{TN}}'
        r'\sum_i (V_i - V_\mathrm{set})^2}$'
    )
    ax1.set_ylabel(r'$\sigma_V^\mathrm{TN}$ [mp.u.]')
    ax1.grid(True, alpha=0.4)
    ax1.legend()

    ax2.plot(t, np.array(ref_rmsd_dn) * 1e3,
             label='MINLP Reference', linewidth=2,
             drawstyle='steps-post', color='#1f77b4')
    ax2.plot(t, np.array(cas_rmsd_dn) * 1e3,
             label='Cascaded OFO', linewidth=2, linestyle='--',
             drawstyle='steps-post', color='#ff7f0e')
    ax2.set_title(
        r'DN (110 kV) Voltage RMSD  '
        r'$\sigma_V^\mathrm{DN}(t)$'
    )
    ax2.set_ylabel(r'$\sigma_V^\mathrm{DN}$ [mp.u.]')
    ax2.set_xlabel('Simulation time [min]')
    ax2.grid(True, alpha=0.4)
    ax2.legend()

    fig.suptitle('Voltage RMSD Comparison (weight-free)', fontsize=13)
    fig.tight_layout()
    plt.show()


# =============================================================================
#  Voltage Max Deviation
# =============================================================================

def plot_voltage_maxdev_comparison(
    time_minutes: list[int],
    ref_max_tn: list[float],
    cas_max_tn: list[float],
    ref_max_dn: list[float],
    cas_max_dn: list[float],
) -> None:
    """
    Two-panel time-series plot of per-timestep maximum absolute voltage
    deviation from setpoint (weight-free).

    This is the worst-case bus metric, relevant for compliance with voltage
    band constraints.

    Panel 1: TN (380 kV) buses  ε_V^TN(t) = max_i |V_i(t) − V_set|.
    Panel 2: DN (110 kV) buses  ε_V^DN(t).
    """
    t = np.array(time_minutes)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(t, np.array(ref_max_tn) * 1e3,
             label='MINLP Reference', linewidth=2,
             drawstyle='steps-post', color='#1f77b4')
    ax1.plot(t, np.array(cas_max_tn) * 1e3,
             label='Cascaded OFO', linewidth=2, linestyle='--',
             drawstyle='steps-post', color='#ff7f0e')
    ax1.set_title(
        r'TN (380 kV) Max Deviation  '
        r'$\varepsilon_V^\mathrm{TN}(t) = \max_i\,|V_i(t) - V_\mathrm{set}|$'
    )
    ax1.set_ylabel(r'$\varepsilon_V^\mathrm{TN}$ [mp.u.]')
    ax1.grid(True, alpha=0.4)
    ax1.legend()

    ax2.plot(t, np.array(ref_max_dn) * 1e3,
             label='MINLP Reference', linewidth=2,
             drawstyle='steps-post', color='#1f77b4')
    ax2.plot(t, np.array(cas_max_dn) * 1e3,
             label='Cascaded OFO', linewidth=2, linestyle='--',
             drawstyle='steps-post', color='#ff7f0e')
    ax2.set_title(
        r'DN (110 kV) Max Deviation  '
        r'$\varepsilon_V^\mathrm{DN}(t)$'
    )
    ax2.set_ylabel(r'$\varepsilon_V^\mathrm{DN}$ [mp.u.]')
    ax2.set_xlabel('Simulation time [min]')
    ax2.grid(True, alpha=0.4)
    ax2.legend()

    fig.suptitle('Maximum Voltage Deviation Comparison (weight-free)', fontsize=13)
    fig.tight_layout()
    plt.show()


# =============================================================================
#  DER Q Comparison
# =============================================================================

def plot_der_q_comparison(
    time_minutes: list[int],
    ref_der_q_l1: list[float],
    cas_der_q_l1: list[float],
) -> None:
    """
    Plot of total absolute DER reactive power dispatch  Σ_k |Q_k(t)|  [Mvar].

    This is a weight-free actuator effort metric: larger values indicate that
    the controller relies more heavily on DER reactive reserves, which reduces
    headroom for active power curtailment and accelerates equipment ageing.
    The L1 norm is used (not L2) because individual DER capacities are
    heterogeneous — a high L1 norm directly corresponds to aggregate reactive
    current through cables and transformers.
    """
    t = np.array(time_minutes)
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(t, ref_der_q_l1,
            label='MINLP Reference', linewidth=2,
            drawstyle='steps-post', color='#1f77b4')
    ax.plot(t, cas_der_q_l1,
            label='Cascaded OFO', linewidth=2, linestyle='--',
            drawstyle='steps-post', color='#ff7f0e')
    ax.set_title(
        r'Total Absolute DER Reactive Dispatch  $\sum_k |Q_k(t)|$'
    )
    ax.set_xlabel('Simulation time [min]')
    ax.set_ylabel(r'$\sum_k |Q_k|$ [Mvar]')
    ax.grid(True, alpha=0.4)
    ax.legend()

    fig.suptitle('DER Q Utilisation Comparison (weight-free)', fontsize=13)
    fig.tight_layout()
    plt.show()


# =============================================================================
#  OLTC Switching Activity
# =============================================================================

def plot_oltc_switching_comparison(
    time_minutes: list[int],
    ref_oltc_switches: list[int],
    cas_oltc_switches: list[int],
) -> None:
    """
    Two-panel plot of per-timestep and cumulative OLTC tap activity.

    Panel 1 (bar chart): per-step total tap movement  Σ_i |Δs_i(t)|.
        Bars are side-by-side so individual switching events are visible.
        This is a wear metric — each tap operation contributes to contact
        erosion regardless of direction.

    Panel 2 (line): cumulative tap movements  Σ_{τ≤t} Σ_i |Δs_i(τ)|.
        The final value is directly comparable as a scalar summary statistic
        in the thesis table.
    """
    t = np.array(time_minutes, dtype=float)
    ref_sw = np.array(ref_oltc_switches, dtype=float)
    cas_sw = np.array(cas_oltc_switches, dtype=float)

    ref_cum = np.cumsum(ref_sw)
    cas_cum = np.cumsum(cas_sw)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Bar width: half the minimum inter-sample interval, capped at 0.4 min.
    dt = float(np.min(np.diff(t))) if len(t) > 1 else 1.0
    w = min(dt * 0.4, 0.4)

    ax1.bar(t - w / 2, ref_sw, width=w,
            label='MINLP Reference', color='#1f77b4', alpha=0.85)
    ax1.bar(t + w / 2, cas_sw, width=w,
            label='Cascaded OFO', color='#ff7f0e', alpha=0.85)
    ax1.set_title(r'Per-step OLTC Tap Activity  $\sum_i |\Delta s_i(t)|$')
    ax1.set_ylabel('Tap steps per minute')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()

    ax2.plot(t, ref_cum,
             label='MINLP Reference', linewidth=2,
             drawstyle='steps-post', color='#1f77b4')
    ax2.plot(t, cas_cum,
             label='Cascaded OFO', linewidth=2, linestyle='--',
             drawstyle='steps-post', color='#ff7f0e')
    ax2.set_title(r'Cumulative OLTC Tap Activity')
    ax2.set_ylabel('Cumulative tap steps')
    ax2.set_xlabel('Simulation time [min]')
    ax2.grid(True, alpha=0.4)
    ax2.legend()

    fig.suptitle('OLTC Switching Activity Comparison (weight-free)', fontsize=13)
    fig.tight_layout()
    plt.show()


# =============================================================================
#  Control Effort (L2 norm)
# =============================================================================

def plot_control_effort_comparison(
    time_minutes: list[int],
    ref_norms: np.ndarray,
    cas_norms: np.ndarray,
) -> None:
    """
    Plot of the L2 norm of the physical control vector  ||u(t)||_2.

    The control vector is assembled in a consistent order for both methods:
        u = [V_gen [p.u.] | s_OLTC [taps] | Q_DER [Mvar] | s_shunt [steps]]

    Note that the mixed physical units make the absolute magnitude of ||u||_2
    sensitive to scaling (e.g., a 1-tap OLTC step contributes the same as a
    1 Mvar DER dispatch).  This metric is therefore most meaningful when
    compared *relatively* between the two methods at the same timestep,
    and should be interpreted alongside the individual actuator plots.

    The Q_PCC_set vector (internal TSO→DSO coordination signal) is excluded
    because it has no counterpart in the monolithic MINLP formulation.

    Parameters
    ----------
    time_minutes : list[int]
        Common simulation minutes.
    ref_norms : np.ndarray
        ||u||_2 per timestep for the MINLP reference.
    cas_norms : np.ndarray
        ||u||_2 per timestep for the cascaded OFO.
    """
    t = np.array(time_minutes)
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(t, ref_norms,
            label='MINLP Reference', linewidth=2,
            drawstyle='steps-post', color='#1f77b4')
    ax.plot(t, cas_norms,
            label='Cascaded OFO', linewidth=2, linestyle='--',
            drawstyle='steps-post', color='#ff7f0e')
    ax.set_title(r'Control Vector Norm  $\|u(t)\|_2$  (physical actuators only)')
    ax.set_xlabel('Simulation time [min]')
    ax.set_ylabel(r'$\|u\|_2$ [mixed units]')
    ax.grid(True, alpha=0.4)
    ax.legend()

    fig.suptitle(
        r'Control Effort Comparison — $Q_\mathrm{PCC,set}$ excluded',
        fontsize=13
    )
    fig.tight_layout()
    plt.show()


# =============================================================================
#  Voltage Penalty (weighted)
# =============================================================================

def plot_voltage_penalty_comparison(
        time_minutes: list[int],
        ref_v_tn: list[float], cas_v_tn: list[float],
        ref_v_dn: list[float], cas_v_dn: list[float],
        ref_g_v_tn: float, cas_g_v_tso: float,
        ref_g_v_dn: float, cas_g_v_dso: float
) -> None:
    """
    Plots the pure voltage deviation penalty for TSO and DSO separately,
    annotating the respective g_v weights.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ── TSO Subplot (TN Buses) ──
    ax1.plot(time_minutes, np.array(ref_v_tn), label=f'MINLP Ref ($g_v$={ref_g_v_tn:g})',
             linewidth=2, drawstyle='steps-post', color='#1f77b4')
    ax1.plot(time_minutes, np.array(cas_v_tn), label=f'Cascaded OFO ($g_v$={cas_g_v_tso:g})',
             linewidth=2, linestyle='--', drawstyle='steps-post', color='#ff7f0e')
    ax1.set_title("TSO Voltage Penalty ($g_v \\sum(V_{TN} - V_{set})^2$)")
    ax1.set_ylabel("Penalty")
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.4, which='both')
    ax1.legend()

    # ── DSO Subplot (DN Buses) ──
    ax2.plot(time_minutes, np.array(ref_v_dn), label=f'MINLP Ref ($g_v$={ref_g_v_dn:g})',
             linewidth=2, drawstyle='steps-post', color='#1f77b4')
    ax2.plot(time_minutes, np.array(cas_v_dn), label=f'Cascaded OFO ($g_v$={cas_g_v_dso:g})',
             linewidth=2, linestyle='--', drawstyle='steps-post', color='#ff7f0e')
    ax2.set_title("DSO Voltage Penalty ($g_v \\sum(V_{DN} - V_{set})^2$)")
    ax2.set_xlabel("Simulation Time (minutes)")
    ax2.set_ylabel("Penalty")
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.4, which='both')
    ax2.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
#  Common Objective (η convergence ratio)
# =============================================================================

def plot_common_objective_comparison(
    time_minutes: list[int],
    ref_J: list[float],
    cas_J: list[float],
    eta_list: list[float],
    g_v_tn: float,
    g_v_dn: float,
    g_u: float,
) -> None:
    """
    Three-panel plot of the common-objective suboptimality comparison.

    Evaluates the *same* MINLP objective function at both the reference
    and cascade plant states, providing a true apples-to-apples comparison.

    Panel 1: Time-series of J_ref(t) and J_cas(t) on a log scale.
             Shows how the cascade converges towards the MINLP oracle.
    Panel 2: Convergence ratio  η(t) = J_ref(t) / J_cas(t)  ∈ [0, 1].
             η = 1 means oracle-optimal; η → 0 means far from optimal.
             More informative than percentage gap when orders of magnitude
             differ.
    Panel 3: Absolute objective difference  ΔJ = J_cas − J_ref.

    Parameters
    ----------
    time_minutes : list[int]
        Common simulation minutes.
    ref_J, cas_J : list[float]
        MINLP objective value at each timestep for reference / cascade.
    eta_list : list[float]
        Convergence ratio η = J_ref / J_cas at each timestep.
    g_v_tn, g_v_dn, g_u : float
        Objective weights (displayed in annotation).
    """
    t = np.array(time_minutes)
    ref_arr = np.array(ref_J)
    cas_arr = np.array(cas_J)
    eta_arr = np.array(eta_list)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    # ── Panel 1: Objective time-series (log scale) ────────────────────────
    ax1.plot(t, ref_arr,
             label='$J(u^*_\\mathrm{MINLP})$', linewidth=2,
             drawstyle='steps-post', color='#1f77b4')
    ax1.plot(t, cas_arr,
             label='$J(u^*_\\mathrm{cascade})$', linewidth=2, linestyle='--',
             drawstyle='steps-post', color='#ff7f0e')
    ax1.set_title(
        'Common Objective Value  '
        '$J(u) = g_v^{TN} \\sum(V_{TN}-V_{set})^2'
        ' + g_v^{DN} \\sum(V_{DN}-V_{set})^2'
        ' + g_u \\sum(Q_{DER}/s_{base})^2$'
    )
    ax1.set_ylabel('$J(u)$')
    # Use log scale only if values are positive
    if np.all(ref_arr > 0) and np.all(cas_arr > 0):
        ax1.set_yscale('log')
    ax1.grid(True, alpha=0.4, which='both')
    ax1.legend(fontsize=11)
    # Annotate weights
    weight_str = f'$g_v^{{TN}}$={g_v_tn:g},  $g_v^{{DN}}$={g_v_dn:g},  $g_u$={g_u:g}'
    ax1.annotate(weight_str, xy=(0.02, 0.92), xycoords='axes fraction',
                 fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='wheat', alpha=0.7))

    # ── Panel 2: Convergence ratio η(t) ──────────────────────────────────
    ax2.plot(t, eta_arr, linewidth=2, drawstyle='steps-post', color='#2ca02c')
    ax2.axhline(1.0, color='black', linewidth=0.8, linestyle='-',
                label=r'$\eta = 1$ (oracle-optimal)')
    ax2.fill_between(t, eta_arr, 1.0, alpha=0.2, color='#d62728', step='post')
    ax2.set_title(
        r'Convergence Ratio  $\eta(t) = J_\mathrm{ref}(t)\,/\,J_\mathrm{cas}(t)$'
        r'  ($\eta = 1$ $\Rightarrow$ oracle-optimal)'
    )
    ax2.set_ylabel(r'$\eta$')
    ax2.set_ylim(-0.05, 1.10)
    ax2.grid(True, alpha=0.4)
    ax2.legend(fontsize=10, loc='lower right')
    # Annotate final η
    if len(eta_arr) > 0:
        final_eta = eta_arr[-1]
        n_ss = max(1, len(eta_arr) // 10)
        ss_eta = float(np.mean(eta_arr[-n_ss:]))
        ax2.annotate(
            f'$\\eta(T)$ = {final_eta:.4f}\n'
            f'$\\bar{{\\eta}}$ last 10% = {ss_eta:.4f}',
            xy=(t[-1], final_eta),
            xytext=(-140, -40), textcoords='offset points',
            fontsize=9, color='#2ca02c',
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='honeydew', alpha=0.8))

    # ── Panel 3: Absolute objective difference ────────────────────────────
    diff = cas_arr - ref_arr
    ax3.fill_between(t, 0, diff, alpha=0.4, color='#d62728', step='post')
    ax3.plot(t, diff, linewidth=1.5, drawstyle='steps-post', color='#d62728')
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_title(
        r'Absolute Objective Difference  '
        r'$\Delta J(t) = J_\mathrm{cas}(t) - J_\mathrm{ref}(t)$'
    )
    ax3.set_ylabel(r'$\Delta J$')
    ax3.set_xlabel('Simulation time [min]')
    ax3.grid(True, alpha=0.4)

    fig.suptitle('Common-Objective Suboptimality Analysis', fontsize=14)
    fig.tight_layout()
    plt.show()


# =============================================================================
#  Actuator State Comparison — 3W OLTC + Shunts
# =============================================================================

def plot_3w_oltc_and_shunt_states(
        common_mins: list[int],
        ref: 'ReferenceResult',
        cas: 'CascadeResult',
) -> None:
    """
    Two-panel plot of 3W coupler OLTC tap positions and shunt step states
    (Reference vs Cascaded OFO).

    Same colour per element; solid line = MINLP Reference,
    dashed line = Cascaded OFO.  Cascade states are forward-filled across
    DSO/TSO firing periods.

    Panel 1 — 3W coupler OLTC tap positions [tap steps].
    Panel 2 — Shunt step states (tertiary DN shunts + TN 380 kV shunts).
    """
    t = np.array(common_mins)
    ref_bm = {r.minute: r for r in ref.log}
    cas_bm = {r.minute: r for r in cas.log}
    colors = plt.cm.tab10.colors

    # -- 3W OLTC (reference) --------------------------------------------------
    # Keys in rr.oltc_taps: '3w_{trafo3w_idx}'
    three_w_keys = sorted(
        k for k in ref.log[0].oltc_taps.keys()
        if k.startswith('3w_')
    )
    if not three_w_keys:
        raise RuntimeError(
            "plot_3w_oltc_and_shunt_states: no '3w_' keys found in "
            "ReferenceRecord.oltc_taps.  Check network metadata."
        )

    n_3w_ref = len(three_w_keys)
    ref_3w = np.zeros((len(common_mins), n_3w_ref), dtype=int)
    for i, m in enumerate(common_mins):
        rr = ref_bm[m]
        for j, key in enumerate(three_w_keys):
            ref_3w[i, j] = rr.oltc_taps.get(key, 0)

    # 3W OLTC (cascade, DSO controller)
    dso_oltc_idx = cas.dso_config.oltc_trafo_indices
    n_3w_cas = len(dso_oltc_idx)
    cas_3w = _forward_fill(
        common_mins, cas_bm,
        lambda r: r.dso_oltc_taps if (r.dso_active and r.dso_oltc_taps is not None) else None,
        n_3w_cas,
    )

    # -- Shunts ---------------------------------------------------------------
    # In extract_network_data(): all_shunt_indices = tertiary + TN.
    # => rr.shunt_steps[:n_tert] = tertiary, [n_tert:] = TN.
    n_tert = len(cas.dso_config.shunt_bus_indices)
    n_tn_sh = len(cas.tso_config.shunt_bus_indices)

    ref_tert = np.zeros((len(common_mins), n_tert), dtype=int)
    ref_tn_sh = np.zeros((len(common_mins), n_tn_sh), dtype=int)
    for i, m in enumerate(common_mins):
        s = ref_bm[m].shunt_steps
        if len(s) >= n_tert:
            ref_tert[i] = s[:n_tert]
        if len(s) >= n_tert + n_tn_sh:
            ref_tn_sh[i] = s[n_tert:n_tert + n_tn_sh]

    cas_tert = _forward_fill(
        common_mins, cas_bm,
        lambda r: r.dso_shunt_states if (r.dso_active and r.dso_shunt_states is not None) else None,
        n_tert,
    )
    cas_tn_sh = _forward_fill(
        common_mins, cas_bm,
        lambda r: r.tso_shunt_states if (r.tso_active and r.tso_shunt_states is not None) else None,
        n_tn_sh,
    )

    # -- Figure ---------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Panel 1 — 3W OLTC
    for j in range(n_3w_ref):
        c = colors[j % 10]
        key = three_w_keys[j]
        ax1.step(t, ref_3w[:, j], where='post',
                 color=c, linewidth=2, label=f'Ref {key}')
    for j in range(n_3w_cas):
        c = colors[j % 10]
        v = ~np.isnan(cas_3w[:, j])
        if v.any():
            ax1.step(t[v], cas_3w[v, j], where='post',
                     color=c, linewidth=2, linestyle='--',
                     label=f'Cas trafo3w_{dso_oltc_idx[j]}')

    ax1.set_title('3W Coupler OLTC Tap Positions')
    ax1.set_ylabel('Tap position [steps]')
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.4)
    ax1.legend(ncol=2, fontsize=8)

    # Panel 2 — Shunts
    for j in range(n_tert):
        c = colors[j % 10]
        sb = cas.dso_config.shunt_bus_indices[j]
        ax2.step(t, ref_tert[:, j], where='post',
                 color=c, linewidth=2, label=f'Ref tert (bus {sb})')
        v = ~np.isnan(cas_tert[:, j])
        if v.any():
            ax2.step(t[v], cas_tert[v, j], where='post',
                     color=c, linewidth=2, linestyle='--',
                     label=f'Cas tert (bus {sb})')

    for j in range(n_tn_sh):
        c = colors[(n_tert + j) % 10]
        sb = cas.tso_config.shunt_bus_indices[j]
        ax2.step(t, ref_tn_sh[:, j], where='post',
                 color=c, linewidth=2, label=f'Ref TN shunt (bus {sb})')
        v = ~np.isnan(cas_tn_sh[:, j])
        if v.any():
            ax2.step(t[v], cas_tn_sh[v, j], where='post',
                     color=c, linewidth=2, linestyle='--',
                     label=f'Cas TN shunt (bus {sb})')

    ax2.set_title('Shunt Step States  [0 = open, 1 = engaged]')
    ax2.set_ylabel('Step state')
    ax2.set_xlabel('Simulation time [min]')
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.set_ylim(-0.1, 1.5)
    ax2.grid(True, alpha=0.4)
    ax2.legend(ncol=2, fontsize=8)

    fig.suptitle(
        '3W Coupler OLTC Positions and Shunt States\n'
        'Solid = MINLP Reference  |  Dashed = Cascaded OFO',
        fontsize=13,
    )
    fig.tight_layout()
    plt.show()


# =============================================================================
#  Actuator State Comparison — 2W OLTC + Generators
# =============================================================================

def plot_2w_oltc_and_generator_states(
        common_mins: list[int],
        ref: 'ReferenceResult',
        cas: 'CascadeResult',
) -> None:
    """
    Three-panel plot of 2W machine transformer OLTC positions, generator AVR
    voltage setpoints, and generator reactive power injection
    (Reference vs Cascaded OFO).

    Panel 1 — 2W machine trafo OLTC tap positions [tap steps].
    Panel 2 — Generator AVR setpoints V_gen [p.u.].
    Panel 3 — Generator Q injection Q_gen [Mvar].

    Notes
    -----
    *Q_gen* for the reference is the AC-OPF optimal dispatch value.
    *Q_gen* for the cascade is the plant measurement after the verification
    power flow (stored in ``IterationRecord.tso_q_gen_mvar``), so it reflects
    the actual generator response to the AVR setpoint rather than a scheduled
    setpoint.
    """
    t = np.array(common_mins)
    ref_bm = {r.minute: r for r in ref.log}
    cas_bm = {r.minute: r for r in cas.log}
    colors = plt.cm.tab10.colors

    # -- 2W OLTC (reference) --------------------------------------------------
    two_w_keys = sorted(
        k for k in ref.log[0].oltc_taps.keys()
        if k.startswith('2w_')
    )
    if not two_w_keys:
        raise RuntimeError(
            "plot_2w_oltc_and_generator_states: no '2w_' keys found in "
            "ReferenceRecord.oltc_taps.  Check network metadata."
        )

    n_2w_ref = len(two_w_keys)
    ref_2w = np.zeros((len(common_mins), n_2w_ref), dtype=int)
    for i, m in enumerate(common_mins):
        rr = ref_bm[m]
        for j, key in enumerate(two_w_keys):
            ref_2w[i, j] = rr.oltc_taps.get(key, 0)

    # 2W OLTC (cascade, TSO controller)
    tso_oltc_idx = cas.tso_config.oltc_trafo_indices
    n_2w_cas = len(tso_oltc_idx)
    cas_2w = _forward_fill(
        common_mins, cas_bm,
        lambda r: r.tso_oltc_taps if (r.tso_active and r.tso_oltc_taps is not None) else None,
        n_2w_cas,
    )

    # -- Generator AVR setpoints ----------------------------------------------
    n_gen = len(cas.tso_config.gen_indices)
    ref_vgen = np.zeros((len(common_mins), n_gen))
    for i, m in enumerate(common_mins):
        rr = ref_bm[m]
        if len(rr.v_gen_pu) == n_gen:
            ref_vgen[i] = rr.v_gen_pu

    cas_vgen = _forward_fill(
        common_mins, cas_bm,
        lambda r: r.tso_v_gen_pu if (r.tso_active and r.tso_v_gen_pu is not None) else None,
        n_gen,
    )

    # -- Generator Q injection ------------------------------------------------
    ref_qgen = np.zeros((len(common_mins), n_gen))
    for i, m in enumerate(common_mins):
        rr = ref_bm[m]
        if len(rr.q_gen_mvar) == n_gen:
            ref_qgen[i] = rr.q_gen_mvar

    # tso_q_gen_mvar is a plant measurement available every minute (not just TSO steps)
    cas_qgen = _forward_fill(
        common_mins, cas_bm,
        lambda r: r.tso_q_gen_mvar if r.tso_q_gen_mvar is not None else None,
        n_gen,
    )

    # -- Figure ---------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    # Panel 1 — 2W OLTC
    for j in range(n_2w_ref):
        c = colors[j % 10]
        key = two_w_keys[j]
        ax1.step(t, ref_2w[:, j], where='post',
                 color=c, linewidth=2, label=f'Ref {key}')
    for j in range(n_2w_cas):
        c = colors[j % 10]
        v = ~np.isnan(cas_2w[:, j])
        if v.any():
            ax1.step(t[v], cas_2w[v, j], where='post',
                     color=c, linewidth=2, linestyle='--',
                     label=f'Cas trafo_{tso_oltc_idx[j]}')

    ax1.set_title('2W Machine Transformer OLTC Tap Positions')
    ax1.set_ylabel('Tap position [steps]')
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.4)
    ax1.legend(ncol=2, fontsize=8)

    # Panel 2 — V_gen AVR setpoints
    gen_idx = cas.tso_config.gen_indices
    for j in range(n_gen):
        c = colors[j % 10]
        ax2.step(t, ref_vgen[:, j], where='post',
                 color=c, linewidth=2, label=f'Ref gen_{gen_idx[j]}')
        v = ~np.isnan(cas_vgen[:, j])
        if v.any():
            ax2.step(t[v], cas_vgen[v, j], where='post',
                     color=c, linewidth=2, linestyle='--',
                     label=f'Cas gen_{gen_idx[j]}')

    ax2.set_title(r'Generator AVR Setpoints  $V_\mathrm{gen}$')
    ax2.set_ylabel(r'$V_\mathrm{gen}$ [p.u.]')
    ax2.grid(True, alpha=0.4)
    ax2.legend(ncol=2, fontsize=8)

    # Panel 3 — Q_gen injection
    for j in range(n_gen):
        c = colors[j % 10]
        ax3.step(t, ref_qgen[:, j], where='post',
                 color=c, linewidth=2, label=f'Ref gen_{gen_idx[j]}')
        v = ~np.isnan(cas_qgen[:, j])
        if v.any():
            ax3.step(t[v], cas_qgen[v, j], where='post',
                     color=c, linewidth=2, linestyle='--',
                     label=f'Cas gen_{gen_idx[j]}')

    ax3.set_title(r'Generator Reactive Power Injection  $Q_\mathrm{gen}$')
    ax3.set_ylabel(r'$Q_\mathrm{gen}$ [Mvar]')
    ax3.set_xlabel('Simulation time [min]')
    ax3.axhline(0, color='k', linewidth=0.8, linestyle=':')
    ax3.grid(True, alpha=0.4)
    ax3.legend(ncol=2, fontsize=8)

    fig.suptitle(
        '2W Machine Trafo OLTC Positions and Generator States\n'
        'Solid = MINLP Reference  |  Dashed = Cascaded OFO',
        fontsize=13,
    )
    fig.tight_layout()
    plt.show()


# =============================================================================
#  Actuator State Comparison — DER Q by Voltage Level
# =============================================================================

def plot_der_q_states(
        common_mins: list[int],
        ref: 'ReferenceResult',
        cas: 'CascadeResult',
) -> None:
    """
    Two-panel plot of DER reactive power dispatch, split by voltage level.

    Panel 1 — TS-side DER Q [Mvar]  (TN-connected DERs, TSO-controlled).
    Panel 2 — DN-side DER Q [Mvar]  (DN-connected DERs, DSO-controlled).

    For the reference, ``ref.ders_pp_buses`` is used to split
    ``ReferenceRecord.q_der_mvar`` into TN and DN subsets, matching the
    cascade's TSO/DSO DER partition.

    Each individual DER is shown as a separate trace.  The Σ|Q| envelope is
    overlaid as a thick semi-transparent line for readability.

    Notes
    -----
    The reference shows per-sgen Q values from the AC-OPF optimal dispatch.
    The cascade shows per-unique-bus Q aggregates from the MIQP controller —
    multiple sgens at the same bus are aggregated into one cascade trace.
    """
    t = np.array(common_mins)
    ref_bm = {r.minute: r for r in ref.log}
    cas_bm = {r.minute: r for r in cas.log}
    colors = plt.cm.tab10.colors

    # -- Reference TN / DN split via ders_pp_buses ----------------------------
    tso_bus_set = set(int(b) for b in cas.tso_config.der_bus_indices)
    dso_bus_set = set(int(b) for b in cas.dso_config.der_bus_indices)

    tn_der_idx = [
        d for d, bus in enumerate(ref.ders_pp_buses) if bus in tso_bus_set
    ]
    dn_der_idx = [
        d for d, bus in enumerate(ref.ders_pp_buses) if bus in dso_bus_set
    ]

    if not tn_der_idx:
        raise RuntimeError(
            "plot_der_q_states: no reference DERs matched the cascade TSO "
            "bus set.  Verify that ref.ders_pp_buses is populated and that "
            "both simulations use the same network."
        )
    if not dn_der_idx:
        raise RuntimeError(
            "plot_der_q_states: no reference DERs matched the cascade DSO "
            "bus set."
        )

    n_tn_ref = len(tn_der_idx)
    n_dn_ref = len(dn_der_idx)

    ref_tn_q = np.zeros((len(common_mins), n_tn_ref))
    ref_dn_q = np.zeros((len(common_mins), n_dn_ref))
    for i, m in enumerate(common_mins):
        q = ref_bm[m].q_der_mvar
        ref_tn_q[i] = q[tn_der_idx]
        ref_dn_q[i] = q[dn_der_idx]

    # -- Cascade TSO DER Q (forward-filled, fires every 3 min) ----------------
    n_tso_der = len(cas.tso_config.der_bus_indices)
    cas_tn_q = _forward_fill(
        common_mins, cas_bm,
        lambda r: r.tso_q_der_mvar if (r.tso_active and r.tso_q_der_mvar is not None) else None,
        n_tso_der,
    )

    # -- Cascade DSO DER Q (forward-filled, fires every 1 min) ----------------
    n_dso_der = len(cas.dso_config.der_bus_indices)
    cas_dn_q = _forward_fill(
        common_mins, cas_bm,
        lambda r: r.dso_q_der_mvar if (r.dso_active and r.dso_q_der_mvar is not None) else None,
        n_dso_der,
    )

    # -- Figure ---------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Panel 1 — TS-side DER Q
    ref_tn_buses = [ref.ders_pp_buses[d] for d in tn_der_idx]
    for j, bus in enumerate(ref_tn_buses):
        c = colors[j % 10]
        ax1.step(t, ref_tn_q[:, j], where='post',
                 color=c, linewidth=1.5, alpha=0.75,
                 label=f'Ref DER (bus {bus})')

    tso_buses = cas.tso_config.der_bus_indices
    for j, bus in enumerate(tso_buses):
        c = colors[j % 10]
        v = ~np.isnan(cas_tn_q[:, j])
        if v.any():
            ax1.step(t[v], cas_tn_q[v, j], where='post',
                     color=c, linewidth=1.5, linestyle='--', alpha=0.75,
                     label=f'Cas DER (bus {bus})')

    # Σ|Q| envelope (thick, semi-transparent)
    ref_tn_sum = np.sum(np.abs(ref_tn_q), axis=1)
    cas_tn_sum = np.nansum(np.abs(cas_tn_q), axis=1)
    ax1.step(t, ref_tn_sum, where='post', color='#1f77b4',
             linewidth=3, alpha=0.4, label='Ref Σ|Q|')
    ax1.step(t, cas_tn_sum, where='post', color='#ff7f0e',
             linewidth=3, alpha=0.4, linestyle='--', label='Cas Σ|Q|')

    ax1.set_title(r'TS-Side DER Reactive Dispatch  $Q_\mathrm{DER}^\mathrm{TS}$  (TSO-controlled)')
    ax1.set_ylabel(r'$Q_\mathrm{DER}$ [Mvar]')
    ax1.axhline(0, color='k', linewidth=0.8, linestyle=':')
    ax1.grid(True, alpha=0.4)
    ax1.legend(ncol=3, fontsize=7)

    # Panel 2 — DN-side DER Q
    dn_ref_buses = [ref.ders_pp_buses[d] for d in dn_der_idx]
    for j, bus in enumerate(dn_ref_buses):
        c = colors[j % 10]
        ax2.step(t, ref_dn_q[:, j], where='post',
                 color=c, linewidth=1.5, alpha=0.75,
                 label=f'Ref DER (bus {bus})')

    dso_buses = cas.dso_config.der_bus_indices
    for j, bus in enumerate(dso_buses):
        c = colors[j % 10]
        v = ~np.isnan(cas_dn_q[:, j])
        if v.any():
            ax2.step(t[v], cas_dn_q[v, j], where='post',
                     color=c, linewidth=1.5, linestyle='--', alpha=0.75,
                     label=f'Cas DER (bus {bus})')

    ref_dn_sum = np.sum(np.abs(ref_dn_q), axis=1)
    cas_dn_sum = np.nansum(np.abs(cas_dn_q), axis=1)
    ax2.step(t, ref_dn_sum, where='post', color='#1f77b4',
             linewidth=3, alpha=0.4, label='Ref Σ|Q|')
    ax2.step(t, cas_dn_sum, where='post', color='#ff7f0e',
             linewidth=3, alpha=0.4, linestyle='--', label='Cas Σ|Q|')

    ax2.set_title(r'DN-Side DER Reactive Dispatch  $Q_\mathrm{DER}^\mathrm{DN}$  (DSO-controlled)')
    ax2.set_ylabel(r'$Q_\mathrm{DER}$ [Mvar]')
    ax2.set_xlabel('Simulation time [min]')
    ax2.axhline(0, color='k', linewidth=0.8, linestyle=':')
    ax2.grid(True, alpha=0.4)
    ax2.legend(ncol=3, fontsize=7)

    fig.suptitle(
        'DER Reactive Power Dispatch by Voltage Level\n'
        'Solid = MINLP Reference  |  Dashed = Cascaded OFO  |  '
        'Thick band = Σ|Q|',
        fontsize=13,
    )
    fig.tight_layout()
    plt.show()
