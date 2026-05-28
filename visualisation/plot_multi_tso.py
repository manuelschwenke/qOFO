"""
Multi-TSO/DSO post-run comparison plotting
==========================================

Post-run comparison between two simulation logs (``log_a`` vs. ``log_b``),
typically "coordinated TSO-DSO OFO" vs. "local DSO control".  The live
plotters for MULTI-TSO CONTROLLER, CASCADE-DSO CONTROLLER and SYSTEM
POWER FLOW have moved to dedicated modules
:mod:`visualisation.plot_tso_controller`,
:mod:`visualisation.plot_cascade_dso`, and
:mod:`visualisation.plot_system_power_flow`.

This module keeps only the post-run comparison tooling
(``plot_coordination_comparison``) because that is the only public symbol
consumed outside the ``visualisation`` package.

Author: Manuel Schwenke / Claude Code
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from visualisation.style import TU_COLOURS, apply_x_fmt as _apply_x_fmt

if TYPE_CHECKING:
    from experiments.helpers import MultiTSOIterationRecord


# =============================================================================
#  Coordination comparison: coordinated vs. uncoordinated Q_PCC
# =============================================================================

# Zone colour mapping: Z1 -> dark blue, Z2 -> teal, Z3 -> dark orange
_ZONE_COLOUR_IDX = {1: 1, 2: 4, 3: 2}
# DSO group colours start from magenta (5) and cycle
_DSO_COLOUR_START = 5


def _extract_comparison_series(
    log: "List[MultiTSOIterationRecord]",
    v_setpoint_pu: float,
) -> dict:
    """Extract time-series arrays from a simulation log for comparison plots.

    Returns a dict with numpy arrays keyed by metric name.
    """
    if not log:
        raise ValueError("log is empty")

    time_min = np.array([r.time_s / 60.0 for r in log])

    # Infer zone and DSO group IDs from the log
    zone_ids: list[int] = sorted(
        {z for r in log for z in r.zone_v_mean.keys()}
    )
    dso_group_ids: list[str] = sorted(
        {g for r in log for g in r.dso_group_v_min_pu.keys()}
    )

    n = len(log)

    # ── Per-zone EHV voltages (every step) ──────────────────────────────
    v_min: dict[int, np.ndarray] = {}
    v_max: dict[int, np.ndarray] = {}
    v_mean: dict[int, np.ndarray] = {}
    for z in zone_ids:
        v_min[z] = np.array([r.zone_v_min.get(z, np.nan) for r in log])
        v_max[z] = np.array([r.zone_v_max.get(z, np.nan) for r in log])
        v_mean[z] = np.array([r.zone_v_mean.get(z, np.nan) for r in log])

    # ── Per-DSO-group HV voltages (every step) ─────────────────────────
    gv_min: dict[str, np.ndarray] = {}
    gv_max: dict[str, np.ndarray] = {}
    for g in dso_group_ids:
        gv_min[g] = np.array([r.dso_group_v_min_pu.get(g, np.nan) for r in log])
        gv_max[g] = np.array([r.dso_group_v_max_pu.get(g, np.nan) for r in log])

    # ── Q_PCC setpoints (TSO-active steps, forward-filled) ─────────────
    q_pcc_sum: dict[int, np.ndarray] = {z: np.full(n, np.nan) for z in zone_ids}
    last_q: dict[int, float] = {}
    for i, r in enumerate(log):
        for z in zone_ids:
            arr = r.zone_q_pcc_set.get(z)
            if arr is not None and len(arr) > 0:
                last_q[z] = float(np.sum(arr))
            if z in last_q:
                q_pcc_sum[z][i] = last_q[z]

    # ── Generator Q infeed per zone (TSO-active steps, forward-filled) ─
    q_gen_sum: dict[int, np.ndarray] = {z: np.full(n, np.nan) for z in zone_ids}
    last_qg: dict[int, float] = {}
    for i, r in enumerate(log):
        for z in zone_ids:
            arr = r.zone_q_gen.get(z)
            if arr is not None and hasattr(arr, '__len__') and len(arr) > 0:
                last_qg[z] = float(np.sum(arr))
            if z in last_qg:
                q_gen_sum[z][i] = last_qg[z]

    # ── Q_PCC actual per DSO group (forward-filled) ────────────────────
    q_actual: dict[str, np.ndarray] = {g: np.full(n, np.nan) for g in dso_group_ids}
    last_qa: dict[str, float] = {}
    for i, r in enumerate(log):
        for g in dso_group_ids:
            val = r.dso_q_actual_mvar.get(g)
            if val is not None:
                last_qa[g] = float(val)
            if g in last_qa:
                q_actual[g][i] = last_qa[g]

    # ── Summary metrics (every step) ───────────────────────────────────
    v_rmsd = np.full(n, np.nan)
    v_maxdev = np.full(n, np.nan)
    for i in range(n):
        devs = []
        for z in zone_ids:
            vm = v_mean[z][i]
            vlo = v_min[z][i]
            vhi = v_max[z][i]
            if not np.isnan(vm):
                devs.append(vm - v_setpoint_pu)
            if not np.isnan(vlo):
                v_maxdev[i] = max(
                    v_maxdev[i] if not np.isnan(v_maxdev[i]) else 0.0,
                    abs(vlo - v_setpoint_pu),
                    abs(vhi - v_setpoint_pu),
                )
        if devs:
            v_rmsd[i] = np.sqrt(np.mean(np.array(devs) ** 2))

    return dict(
        time_min=time_min,
        zone_ids=zone_ids,
        dso_group_ids=dso_group_ids,
        v_min=v_min,
        v_max=v_max,
        v_mean=v_mean,
        gv_min=gv_min,
        gv_max=gv_max,
        q_pcc_sum=q_pcc_sum,
        q_gen_sum=q_gen_sum,
        q_actual=q_actual,
        v_rmsd=v_rmsd,
        v_maxdev=v_maxdev,
    )


def _add_contingency_shading(
    axes,
    contingencies: "Optional[List]" = None,
) -> None:
    """Add grey vertical spans for each trip/restore contingency pair."""
    if not contingencies:
        return
    # Pair trips with restores by (element_type, element_index)
    trips: dict[tuple, float] = {}
    pairs: list[tuple[float, float]] = []
    for ev in contingencies:
        key = (ev.element_type, ev.element_index)
        t_min = ev.effective_time_s / 60.0
        if ev.action == "trip":
            trips[key] = t_min
        elif ev.action == "restore" and key in trips:
            pairs.append((trips.pop(key), t_min))
    # Add spans for unrestored trips extending to the right edge
    for key, t_trip in trips.items():
        pairs.append((t_trip, None))

    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        for t_start, t_end in pairs:
            if t_end is not None:
                ax.axvspan(t_start, t_end, color='grey', alpha=0.08, zorder=0)
            else:
                ax.axvline(t_start, color='grey', alpha=0.3, linestyle=':', zorder=0)


def plot_coordination_comparison(
    log_a: "List[MultiTSOIterationRecord]",
    log_b: "List[MultiTSOIterationRecord]",
    *,
    label_a: str = "Coordinated",
    label_b: str = "Uncoordinated",
    v_setpoint_pu: float = 1.03,
    v_min_pu: float = 0.9,
    v_max_pu: float = 1.1,
    contingencies: "Optional[List]" = None,
    gen_info: "Optional[List[dict]]" = None,
    show: bool = True,
) -> "tuple[plt.Figure, ...]":
    """Compare two simulation logs: coordinated vs. uncoordinated Q_PCC control.

    Parameters
    ----------
    log_a, log_b : list of MultiTSOIterationRecord
        Simulation logs for Scenario A (coordinated) and B (uncoordinated).
    label_a, label_b : str
        Legend labels for the two scenarios.
    v_setpoint_pu : float
        Voltage setpoint for reference lines.
    v_min_pu, v_max_pu : float
        Voltage limits for reference lines.
    contingencies : list, optional
        ContingencyEvent list for shading trip/restore periods.
    gen_info : list of dict, optional
        Per-generator capability data. Each dict has keys: zone, gen_idx,
        name, max_p_mw, min_p_mw, max_q_mvar, min_q_mvar.
    show : bool
        Call ``plt.show()`` after creating figures.

    Returns
    -------
    tuple of figures (fig_voltage, fig_metrics[, fig_capability])
    """
    da = _extract_comparison_series(log_a, v_setpoint_pu)
    db = _extract_comparison_series(log_b, v_setpoint_pu)

    zone_ids = da["zone_ids"]
    dso_group_ids = da["dso_group_ids"]

    # =====================================================================
    #  Figure 1: Voltage & Q_PCC Comparison
    #  Layout: N_zone EHV rows + 1 HV + 1 Q_PCC set + 1 Q_PCC act + 1 Q_gen
    # =====================================================================
    n_z = len(zone_ids)
    n_rows_fig1 = n_z + 4  # per-zone EHV + HV + Q_set + Q_act + Q_gen
    # Give EHV zone rows slightly less height than the other panels
    height_ratios = [1.0] * n_z + [1.2, 1.0, 1.0, 1.0]
    fig1, axes1 = plt.subplots(
        n_rows_fig1, 1,
        figsize=(14, 2.5 * n_z + 10),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig1.suptitle("Coordinated vs. Uncoordinated Q$_{PCC}$ Control",
                  fontweight="bold", fontsize=13)

    col_a_style = TU_COLOURS[1]  # dark blue for scenario A
    col_b_style = TU_COLOURS[2]  # dark orange for scenario B

    # ── Rows 0..N_z-1: EHV voltage envelope, one subplot per zone ──────
    for iz, z in enumerate(zone_ids):
        ax = axes1[iz]
        # Scenario A: filled blue band
        ax.fill_between(da["time_min"], da["v_min"][z], da["v_max"][z],
                        color=col_a_style, alpha=0.15, step="post")
        ax.plot(da["time_min"], da["v_min"][z], color=col_a_style,
                linewidth=1.0, drawstyle="steps-post")
        ax.plot(da["time_min"], da["v_max"][z], color=col_a_style,
                linewidth=1.0, drawstyle="steps-post")
        # Scenario B: filled orange band
        ax.fill_between(db["time_min"], db["v_min"][z], db["v_max"][z],
                        color=col_b_style, alpha=0.15, step="post")
        ax.plot(db["time_min"], db["v_min"][z], color=col_b_style,
                linewidth=1.0, linestyle="--", drawstyle="steps-post")
        ax.plot(db["time_min"], db["v_max"][z], color=col_b_style,
                linewidth=1.0, linestyle="--", drawstyle="steps-post")
        ax.axhline(v_setpoint_pu, color="black", linewidth=0.6, linestyle=":")
        ax.axhline(v_min_pu, color="red", linewidth=0.6, linestyle="--",
                   alpha=0.5)
        ax.axhline(v_max_pu, color="red", linewidth=0.6, linestyle="--",
                   alpha=0.5)
        ax.set_ylabel(f"Zone {z} V [p.u.]")
        ax.grid(True, alpha=0.3)
        if iz == 0:
            ax.legend(
                handles=[
                    Patch(facecolor=col_a_style, alpha=0.3, label=label_a),
                    Patch(facecolor=col_b_style, alpha=0.3, label=label_b),
                ],
                loc="upper right", fontsize=8,
            )

    # ── Next row: HV voltage per DSO group ──────────────────────────────
    ax = axes1[n_z]
    legend_handles = []
    for i, g in enumerate(dso_group_ids):
        ci = (_DSO_COLOUR_START + i) % len(TU_COLOURS)
        col = TU_COLOURS[ci]
        ax.fill_between(da["time_min"], da["gv_min"][g], da["gv_max"][g],
                        color=col, alpha=0.25, step="post")
        ax.plot(da["time_min"], da["gv_min"][g], color=col, linewidth=0.8,
                drawstyle="steps-post")
        ax.plot(da["time_min"], da["gv_max"][g], color=col, linewidth=0.8,
                drawstyle="steps-post")
        ax.plot(db["time_min"], db["gv_min"][g], color=col, linewidth=1.0,
                linestyle="--", drawstyle="steps-post")
        ax.plot(db["time_min"], db["gv_max"][g], color=col, linewidth=1.0,
                linestyle="--", drawstyle="steps-post")
        legend_handles.append(Patch(facecolor=col, alpha=0.4, label=g))
    ax.axhline(v_setpoint_pu, color="black", linewidth=0.6, linestyle=":")
    ax.axhline(v_min_pu, color="red", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axhline(v_max_pu, color="red", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=2)
    ax.set_ylabel("HV Voltage [p.u.]")
    ax.grid(True, alpha=0.3)

    # ── Q_PCC setpoint per zone (sum) ───────────────────────────────────
    ax = axes1[n_z + 1]
    for z in zone_ids:
        ci = _ZONE_COLOUR_IDX.get(z, z % len(TU_COLOURS))
        col = TU_COLOURS[ci]
        ax.plot(da["time_min"], da["q_pcc_sum"][z], color=col, linewidth=1.2,
                drawstyle="steps-post", label=f"Z{z} {label_a}")
        ax.plot(db["time_min"], db["q_pcc_sum"][z], color=col, linewidth=1.2,
                linestyle="--", drawstyle="steps-post", label=f"Z{z} {label_b}")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylabel("$Q_{PCC}^{set}$ [Mvar]")
    ax.grid(True, alpha=0.3)

    # ── Q_PCC actual per DSO group ──────────────────────────────────────
    ax = axes1[n_z + 2]
    for i, g in enumerate(dso_group_ids):
        ci = (_DSO_COLOUR_START + i) % len(TU_COLOURS)
        col = TU_COLOURS[ci]
        ax.plot(da["time_min"], da["q_actual"][g], color=col, linewidth=1.2,
                drawstyle="steps-post", label=f"{g} {label_a}")
        ax.plot(db["time_min"], db["q_actual"][g], color=col, linewidth=1.2,
                linestyle="--", drawstyle="steps-post", label=f"{g} {label_b}")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylabel("$Q_{PCC}^{actual}$ [Mvar]")
    ax.grid(True, alpha=0.3)

    # ── Generator Q infeed per zone (sum) ───────────────────────────────
    ax = axes1[n_z + 3]
    for z in zone_ids:
        ci = _ZONE_COLOUR_IDX.get(z, z % len(TU_COLOURS))
        col = TU_COLOURS[ci]
        ax.plot(da["time_min"], da["q_gen_sum"][z], color=col, linewidth=1.2,
                drawstyle="steps-post", label=f"Z{z} {label_a}")
        ax.plot(db["time_min"], db["q_gen_sum"][z], color=col, linewidth=1.2,
                linestyle="--", drawstyle="steps-post", label=f"Z{z} {label_b}")
    ax.axhline(0, color="black", linewidth=0.4, linestyle=":")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylabel("$\\Sigma Q_{gen}$ [Mvar]")
    ax.set_xlabel("Time [min]")
    ax.grid(True, alpha=0.3)

    # Contingency shading + x-axis formatting
    _add_contingency_shading(axes1, contingencies)
    for a in axes1:
        _apply_x_fmt(a)

    # =====================================================================
    #  Figure 2: Summary Metrics (2 rows)
    # =====================================================================
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                               constrained_layout=True)
    fig2.suptitle("Voltage Quality Metrics", fontweight="bold", fontsize=13)

    col_a = TU_COLOURS[1]  # dark blue
    col_b = TU_COLOURS[2]  # dark orange

    # ── Row 0: Voltage RMSD from setpoint ───────────────────────────────
    ax = axes2[0]
    ax.plot(da["time_min"], da["v_rmsd"], color=col_a, linewidth=1.5,
            label=label_a)
    ax.plot(db["time_min"], db["v_rmsd"], color=col_b, linewidth=1.5,
            linestyle="--", label=label_b)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylabel("Voltage RMSD [p.u.]")
    ax.grid(True, alpha=0.3)

    # ── Row 1: Max voltage deviation ────────────────────────────────────
    ax = axes2[1]
    ax.plot(da["time_min"], da["v_maxdev"], color=col_a, linewidth=1.5,
            label=label_a)
    ax.plot(db["time_min"], db["v_maxdev"], color=col_b, linewidth=1.5,
            linestyle="--", label=label_b)
    ax.axhline(0.05, color="red", linewidth=0.6, linestyle="--", alpha=0.5,
               label="5% limit")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylabel("Max $|\\Delta V|$ [p.u.]")
    ax.set_xlabel("Time [min]")
    ax.grid(True, alpha=0.3)

    # Contingency shading + x-axis formatting
    _add_contingency_shading(axes2, contingencies)
    for a in axes2:
        _apply_x_fmt(a)

    # =====================================================================
    #  Figure 3: Generator P-Q capability scatter (if gen_info provided)
    #  Capability curve: Milano (2010) §12.2.1 — three thermal constraints
    #    (i)   Stator:  p² + q² ≤ s_max²
    #    (ii)  Rotor:   p² + (q + v²/xd)² ≤ (v·i_f_max/xd)²
    #    (iii) Under-excitation:  q ≥ −q₀(v) + β·p_max
    # =====================================================================
    fig3 = None
    if gen_info:
        from core.actuator_bounds import GeneratorParameters, compute_generator_q_limits

        # Group generators by zone for indexing
        gens_by_zone: dict[int, list[dict]] = {}
        for gi in gen_info:
            gens_by_zone.setdefault(gi["zone"], []).append(gi)

        n_gens = len(gen_info)
        ncols = min(n_gens, 4)
        nrows = (n_gens + ncols - 1) // ncols
        fig3, axes3 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                                   constrained_layout=True, squeeze=False)
        fig3.suptitle("Generator P-Q Operating Points vs. Capability (Milano §12.2)",
                      fontweight="bold", fontsize=13)

        gen_flat_idx = 0
        for z in sorted(gens_by_zone.keys()):
            zone_gens = gens_by_zone[z]
            for k, gi in enumerate(zone_gens):
                row, col_idx = divmod(gen_flat_idx, ncols)
                ax = axes3[row, col_idx]

                gp = GeneratorParameters(
                    s_rated_mva=gi["s_rated_mva"],
                    p_max_mw=gi["p_max_mw"],
                    xd_pu=gi["xd_pu"],
                    i_f_max_pu=gi["i_f_max_pu"],
                    beta=gi["beta"],
                    q0_pu=gi["q0_pu"],
                )
                s_base = gi["s_rated_mva"]
                v = 1.0  # nominal voltage for capability envelope

                # ── Individual constraint curves (dashed, for reference) ──
                # Sweep P from 0 to S_rated to show full stator circle
                p_full = np.linspace(0, s_base, 300)
                p_pu = p_full / s_base

                # (i) Stator current limit: p^2 + q^2 <= 1
                disc_s = np.maximum(1.0 - p_pu**2, 0.0)
                q_stator_hi = np.sqrt(disc_s) * s_base
                q_stator_lo = -q_stator_hi

                # (ii) Rotor current limit: p^2 + (q + v^2/xd)^2 <= (v*if/xd)^2
                xd = gi["xd_pu"]
                i_f = gi["i_f_max_pu"]
                rotor_r = v * i_f / xd
                rotor_c = -v**2 / xd  # center in q (p.u.)
                disc_r = np.maximum(rotor_r**2 - p_pu**2, 0.0)
                q_rotor_hi = (rotor_c + np.sqrt(disc_r)) * s_base

                # (iii) Under-excitation: q >= -q0*v^2 + beta*p_max_pu
                p_max_pu = gi["p_max_mw"] / s_base
                q_ue = (-gi["q0_pu"] * v**2 + gi["beta"] * p_max_pu) * s_base

                # Plot individual constraints as thin dashed lines
                ax.plot(p_full, q_stator_hi, color="grey", linewidth=0.7,
                        linestyle=":", zorder=0, label="Stator limit")
                ax.plot(p_full, q_stator_lo, color="grey", linewidth=0.7,
                        linestyle=":", zorder=0)
                ax.plot(p_full, q_rotor_hi, color="grey", linewidth=0.7,
                        linestyle="-.", zorder=0, label="Rotor limit")
                ax.axhline(q_ue, color="grey", linewidth=0.7,
                           linestyle="--", zorder=0, label="UE limit")

                # ── Composite capability envelope (p_min to p_max) ────────
                p_min = gi.get("p_min_mw", 0.0)
                p_sweep = np.linspace(p_min, gi["p_max_mw"], 300)
                q_lo_cap = np.empty_like(p_sweep)
                q_hi_cap = np.empty_like(p_sweep)
                for ip, pp in enumerate(p_sweep):
                    q_lo_cap[ip], q_hi_cap[ip] = compute_generator_q_limits(
                        gp, p_mw=pp, v_pu=v,
                    )

                # Fill the feasible region
                ax.fill_between(p_sweep, q_lo_cap, q_hi_cap,
                                color="lightgrey", alpha=0.3, zorder=1,
                                label="Feasible region")
                ax.plot(p_sweep, q_hi_cap, color="black", linewidth=1.2,
                        zorder=1)
                ax.plot(p_sweep, q_lo_cap, color="black", linewidth=1.2,
                        zorder=1)
                # Vertical lines at P_min and P_max (turbine limits)
                if p_min > 0:
                    ax.axvline(p_min, color="black", linewidth=0.7,
                               linestyle="--", alpha=0.5, zorder=1)
                ax.axvline(gi["p_max_mw"], color="black", linewidth=0.7,
                           linestyle="--", alpha=0.5, zorder=1)

                # Collect operating points for this generator
                p_a, q_a = [], []
                p_b, q_b = [], []
                for r in log_a:
                    pv = r.zone_p_gen.get(z)
                    qv = r.zone_q_gen.get(z)
                    if pv is not None and qv is not None and k < len(pv):
                        p_a.append(float(pv[k]))
                        q_a.append(float(qv[k]))
                for r in log_b:
                    pv = r.zone_p_gen.get(z)
                    qv = r.zone_q_gen.get(z)
                    if pv is not None and qv is not None and k < len(pv):
                        p_b.append(float(pv[k]))
                        q_b.append(float(qv[k]))

                # Scatter: Scenario A vs B
                if p_a:
                    ax.scatter(p_a, q_a, color=TU_COLOURS[1], alpha=0.15,
                               s=12, label=label_a, zorder=2, edgecolors="none")
                if p_b:
                    ax.scatter(p_b, q_b, color=TU_COLOURS[2], alpha=0.15,
                               s=12, label=label_b, zorder=2, edgecolors="none")

                ax.set_title(f"{gi['name']} (Z{z})", fontsize=10)
                ax.set_xlabel("P [MW]")
                ax.set_ylabel("Q [Mvar]")
                ax.axhline(0, color="black", linewidth=0.3, linestyle=":")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7, loc="best")

                gen_flat_idx += 1

        # Hide unused subplots
        for idx in range(gen_flat_idx, nrows * ncols):
            row, col_idx = divmod(idx, ncols)
            axes3[row, col_idx].set_visible(False)

    import os
    os.makedirs("results", exist_ok=True)
    fig1.savefig("results/compare_voltage_q.png", dpi=120, bbox_inches="tight")
    fig2.savefig("results/compare_metrics.png",  dpi=120, bbox_inches="tight")
    if fig3 is not None:
        fig3.savefig("results/compare_gen_capability.png", dpi=120, bbox_inches="tight")

    if show:
        plt.ioff()
        plt.show()

    if fig3 is not None:
        return fig1, fig2, fig3
    return fig1, fig2
