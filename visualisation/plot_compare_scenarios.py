#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualisation/plot_compare_scenarios.py
=======================================
Static comparison plots for the four control modes evaluated by
``experiments/002_M_TSO_M_DSO_COMPARE.py``:

* L0  -- TSO + DSO at cos phi = 1 (worst case)
* L1  -- TSO Q(V), DSO cos phi = 1
* L2  -- TSO Q(V) + DSO Q(V)
* OFO -- Multi-TSO / Multi-DSO OFO controller

Plots produced (PNG + PDF):

1. ``compare_ts_voltage_envelope``     -- TRANSMISSION-system voltage
                                          envelope (focus): three stacked
                                          subplots v_min / v_mean / v_max
                                          across all TSO zones.  Setpoint
                                          line + reference bands annotated.
2. ``compare_ds_voltage_envelope``     -- DISTRIBUTION-system equivalent
                                          across all HV sub-networks.
3. ``compare_voltage_rmsd``            -- TS and DS RMSD-to-setpoint vs t
                                          (two stacked subplots).
4. ``compare_voltage_violations``      -- bar chart of total step-aggregate
                                          violations: TS / DS x low / high.
5. ``compare_losses``                  -- total network active losses vs t.
6. ``compare_gen_q_headroom``          -- minimum sync-gen Q headroom vs t.
7. ``compare_q_tie_deviation``         -- summed and worst-pair absolute
                                          deviation of inter-zone tie-line
                                          Q flows from the setpoint (Phase B
                                          Q_tie tracking diagnostic).
8. ``compare_gen_capability_pq``       -- per-generator P-Q operating-point
                                          scatter against the Milano §12.2
                                          capability envelope; one tile per
                                          gen with all four scenarios
                                          overlaid as transparent scatter.

Non-converged scenarios (empty record list) are annotated in a footer rather
than dropped, so the consumer can see at a glance which baselines failed.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from experiments.helpers.records import MultiTSOIterationRecord
from experiments.helpers.comparison_metrics import (
    V_SET_DEFAULT,
    gen_q_headroom_series,
    loss_series,
    q_tie_deviation_series,
    summary_table,
    voltage_envelope_ds,
    voltage_envelope_ts,
    voltage_rmsd_ds,
    voltage_rmsd_ts,
    voltage_violation_counts_ds,
    voltage_violation_counts_ts,
)
from visualisation.style import (
    COLOUR_TITLE_BAR,
    TU_COLOURS,
    apply_serif_style,
    draw_figure_header,
    tile_title,
)


# Default scenario palette: deliberately picks visually-distinct colours
# from the TU Darmstadt palette.
_DEFAULT_PALETTE: Dict[str, str] = {
    "L0":    TU_COLOURS[8],   # red
    "L1":    TU_COLOURS[3],   # gold
    "L2":    TU_COLOURS[4],   # teal
    "T-OFO": TU_COLOURS[5],   # magenta
    "C-OFO": TU_COLOURS[1],   # dark blue
}


def _resolve_palette(palette: Optional[Dict[str, str]],
                     scenario_names: List[str]) -> Dict[str, str]:
    if palette is not None:
        return {name: palette.get(name, TU_COLOURS[i % len(TU_COLOURS)])
                for i, name in enumerate(scenario_names)}
    return {name: _DEFAULT_PALETTE.get(name, TU_COLOURS[i % len(TU_COLOURS)])
            for i, name in enumerate(scenario_names)}


def _save(fig, out_dir: str, basename: str) -> None:
    """Save fig as both PNG and PDF.  Locked target files (PermissionError,
    OSError) are reported but do not abort the comparison run — the rest
    of the figures still render."""
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{basename}.{ext}")
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
        except (PermissionError, OSError) as exc:
            print(f"  [plot_compare] WARNING: could not write {path}: {exc} "
                  f"-- close the file in any viewer and re-run with --replot to retry")


def _failure_footer(fig, failed_names: List[str]) -> None:
    if not failed_names:
        return
    fig.text(
        0.012, 0.005,
        f"Failed to converge: {', '.join(failed_names)}",
        color="#666666", fontsize=8, ha="left", va="bottom",
        style="italic",
    )


def _annotate_voltage_axis(ax, v_set: float, low: float, high: float,
                           soft_low: float = 0.90, soft_high: float = 1.10,
                           ) -> None:
    """Add setpoint line plus ±0.05 / ±0.10 reference bands."""
    ax.axhline(v_set, color="#222222", linewidth=0.9, linestyle="-",
               alpha=0.7, zorder=1)
    for v in (low, high):
        ax.axhline(v, color="#888888", linewidth=0.6, linestyle="--",
                   alpha=0.6, zorder=1)
    for v in (soft_low, soft_high):
        ax.axhline(v, color="#aa4444", linewidth=0.6, linestyle=":",
                   alpha=0.6, zorder=1)


# ---------------------------------------------------------------------------
#  Voltage envelopes (TS and DS)
# ---------------------------------------------------------------------------


def _plot_voltage_envelope(envelope_fn, header: str, basename: str,
                           logs, palette, out_dir, failed, v_set: float,
                           ) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    plt.subplots_adjust(top=0.93, hspace=0.20, left=0.10, right=0.98)
    draw_figure_header(fig, header, color=COLOUR_TITLE_BAR)

    keys = ("v_min", "v_mean", "v_max")
    titles = ("MIN VOLTAGE [P.U.]",
              "MEAN VOLTAGE [P.U.]",
              "MAX VOLTAGE [P.U.]")

    for ax, key, title in zip(axes, keys, titles):
        for name, recs in logs.items():
            if not recs:
                continue
            env = envelope_fn(recs)
            ax.plot(env["t_min"], env[key], color=palette[name],
                    label=name, linewidth=1.4)
        _annotate_voltage_axis(ax, v_set, low=0.95, high=1.05)
        tile_title(ax, title)
        ax.set_ylabel("V [p.u.]")

    axes[-1].set_xlabel("Time [min]")
    # Build a custom legend that includes the setpoint line entry.
    from matplotlib.lines import Line2D
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="#222222", linewidth=0.9,
                          linestyle="-", alpha=0.7))
    labels.append(f"V_set = {v_set:.3f} pu")
    axes[0].legend(handles, labels, loc="upper right",
                   ncol=min(len(handles), 5), frameon=True, fontsize=8)
    _failure_footer(fig, failed)
    _save(fig, out_dir, basename)
    plt.close(fig)


def _plot_voltage_rmsd(logs, palette, out_dir, failed, v_set: float) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    plt.subplots_adjust(top=0.91, hspace=0.22, left=0.10, right=0.98)
    draw_figure_header(fig, f"Voltage RMSD to setpoint ({v_set:.3f} pu)",
                       color=COLOUR_TITLE_BAR)

    for ax, fn, title in (
        (axes[0], voltage_rmsd_ts, "TS — RMSD ACROSS TSO ZONES [P.U.]"),
        (axes[1], voltage_rmsd_ds, "DS — RMSD ACROSS HV SUB-NETWORKS [P.U.]"),
    ):
        for name, recs in logs.items():
            if not recs:
                continue
            r = fn(recs, v_set=v_set)
            ax.plot(r["t_min"], r["rmsd_pu"], color=palette[name],
                    label=name, linewidth=1.4)
        ax.axhline(0.0, color="#222222", linewidth=0.5, alpha=0.4)
        ax.set_ylabel("RMSD [p.u.]")
        ax.set_ylim(bottom=0.0)
        tile_title(ax, title)

    axes[-1].set_xlabel("Time [min]")
    axes[0].legend(loc="upper right", ncol=4, frameon=True, fontsize=8)
    _failure_footer(fig, failed)
    _save(fig, out_dir, "compare_voltage_rmsd")
    plt.close(fig)


def _plot_voltage_violations(logs, palette, out_dir, failed,
                             low: float = 0.95, high: float = 1.05) -> None:
    import matplotlib.pyplot as plt

    names = list(logs.keys())
    counts = {"TS_low": [], "TS_high": [], "DS_low": [], "DS_high": []}
    for name in names:
        recs = logs[name]
        if recs:
            ts = voltage_violation_counts_ts(recs, low, high)
            ds = voltage_violation_counts_ds(recs, low, high)
            counts["TS_low"].append(int(np.nansum(ts["n_low"])))
            counts["TS_high"].append(int(np.nansum(ts["n_high"])))
            counts["DS_low"].append(int(np.nansum(ds["n_low"])))
            counts["DS_high"].append(int(np.nansum(ds["n_high"])))
        else:
            for k in counts:
                counts[k].append(0)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    plt.subplots_adjust(top=0.86, left=0.10, right=0.98, bottom=0.13)
    draw_figure_header(fig, f"Voltage-band violations (V < {low:.2f} | V > {high:.2f} pu)",
                       color=COLOUR_TITLE_BAR)

    x = np.arange(len(names))
    w = 0.20
    bars_specs = [
        ("TS_low",  -1.5 * w, "#aa4444", "/"),     # TS low
        ("TS_high", -0.5 * w, "#cc8800", "/"),     # TS high
        ("DS_low",  +0.5 * w, "#aa4444", "."),     # DS low
        ("DS_high", +1.5 * w, "#cc8800", "."),     # DS high
    ]
    label_map = {
        "TS_low":  f"TS V < {low:.2f}",
        "TS_high": f"TS V > {high:.2f}",
        "DS_low":  f"DS V < {low:.2f}",
        "DS_high": f"DS V > {high:.2f}",
    }
    for key, dx, colour, hatch in bars_specs:
        bars = ax.bar(x + dx, counts[key], w, color=colour,
                      hatch=hatch, edgecolor="white",
                      label=label_map[key])
        for rect in bars:
            h = rect.get_height()
            if h:
                ax.annotate(f"{int(h)}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Total step-aggregate violations")
    ax.legend(loc="upper right", frameon=True, ncol=2, fontsize=8)
    tile_title(ax, "TS = TSO ZONES, DS = HV SUB-NETWORKS")
    _failure_footer(fig, failed)
    _save(fig, out_dir, "compare_voltage_violations")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Losses & headroom (unchanged behaviour, kept for completeness)
# ---------------------------------------------------------------------------


def _plot_losses(logs, palette, out_dir, failed):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    plt.subplots_adjust(top=0.88, left=0.10, right=0.98, bottom=0.14)
    draw_figure_header(fig, "Total network losses vs t", color=COLOUR_TITLE_BAR)

    for name, recs in logs.items():
        if not recs:
            continue
        ls = loss_series(recs)
        ax.plot(ls["t_min"], ls["losses_mw"], color=palette[name],
                label=name, linewidth=1.4)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Losses [MW]")
    ax.legend(loc="upper right", frameon=True, ncol=4)
    tile_title(ax, "LINE + 2W + 3W TRAFO LOSSES")
    _failure_footer(fig, failed)
    _save(fig, out_dir, "compare_losses")
    plt.close(fig)


def _plot_gen_q_headroom(logs, palette, out_dir, failed):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    plt.subplots_adjust(top=0.88, left=0.10, right=0.98, bottom=0.14)
    draw_figure_header(fig, "Min sync-gen Q headroom vs t",
                       color=COLOUR_TITLE_BAR)

    for name, recs in logs.items():
        if not recs:
            continue
        qh = gen_q_headroom_series(recs)
        ax.plot(qh["t_min"], qh["q_headroom_min_mvar"], color=palette[name],
                label=name, linewidth=1.4)
    ax.axhline(0.0, color="#666", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("min(Q_max - |Q_act|) [Mvar]")
    ax.legend(loc="upper right", frameon=True, ncol=4)
    tile_title(ax, "MOST-SATURATED SYNCHRONOUS MACHINE")
    _failure_footer(fig, failed)
    _save(fig, out_dir, "compare_gen_q_headroom")
    plt.close(fig)


def _plot_q_tie_deviation(logs, palette, out_dir, failed,
                          q_tie_setpoint_mvar: float = 0.0) -> None:
    """Two stacked subplots:
        (top)    Sum |Q_tie_pair - Q_set| across all zone pairs vs t.
        (bottom) Worst-pair |Q_tie - Q_set| vs t.
    Both metrics use the per-pair aggregate Q (sum over physical lines
    between two zones).  Default setpoint 0 Mvar matches the Phase B
    "no inter-zone reactive exchange" target.
    """
    import matplotlib.pyplot as plt

    fig, (ax_sum, ax_max) = plt.subplots(2, 1, figsize=(8.5, 6.2),
                                         sharex=True)
    plt.subplots_adjust(top=0.90, left=0.10, right=0.98, bottom=0.10,
                        hspace=0.20)
    draw_figure_header(
        fig,
        f"Tie-line Q deviation from setpoint ({q_tie_setpoint_mvar:.1f} Mvar)"
        " vs t",
        color=COLOUR_TITLE_BAR,
    )

    any_data = False
    for name, recs in logs.items():
        if not recs:
            continue
        qt = q_tie_deviation_series(recs, q_tie_setpoint_mvar=q_tie_setpoint_mvar)
        if not np.any(np.isfinite(qt["sum_abs_dev_mvar"])):
            continue
        any_data = True
        ax_sum.plot(qt["t_min"], qt["sum_abs_dev_mvar"],
                    color=palette[name], label=name, linewidth=1.4)
        ax_max.plot(qt["t_min"], qt["max_abs_dev_mvar"],
                    color=palette[name], label=name, linewidth=1.4)

    if not any_data:
        ax_sum.text(0.5, 0.5,
                    "no zone_tie_q_mvar records found",
                    transform=ax_sum.transAxes,
                    ha="center", va="center",
                    color="#888", style="italic")

    for ax in (ax_sum, ax_max):
        ax.axhline(0.0, color="#666", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.set_ylabel("|Q_tie - Q_set| [Mvar]")

    ax_max.set_xlabel("Time [min]")
    ax_sum.legend(loc="upper right", frameon=True, ncol=4)

    tile_title(ax_sum, "sum over zone pairs of |Q_tie - Q_set|")
    tile_title(ax_max, "worst-pair |Q_tie - Q_set|")
    _failure_footer(fig, failed)
    _save(fig, out_dir, "compare_q_tie_deviation")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Generator P-Q operating points vs. capability envelope
# ---------------------------------------------------------------------------


def _build_gen_info_for_scenario(scenario: str = "wind_replace") -> List[Dict]:
    """Re-derive the per-generator capability metadata from a fresh net.

    Mirrors the block in ``experiments/000_M_TSO_M_DSO.py`` near
    ``main_comparison()`` (the "Extract generator limits for capability
    curve plot" section).  Returned list is sorted by (zone, gen_idx) so
    plot tiles render in a stable order.
    """
    from network.ieee39.build import build_ieee39_net
    from network.zone_partition import fixed_zone_partition_ieee39

    net_tmp, _ = build_ieee39_net(scenario=scenario)
    _, bus_zone_tmp = fixed_zone_partition_ieee39(net_tmp, verbose=False)

    gen_info: List[Dict] = []
    for g_idx in net_tmp.gen.index:
        g_bus = int(net_tmp.gen.at[g_idx, "bus"])
        zone = bus_zone_tmp.get(g_bus)
        if zone is None:
            for ti in net_tmp.trafo.index:
                if int(net_tmp.trafo.at[ti, "lv_bus"]) == g_bus:
                    hv_bus = int(net_tmp.trafo.at[ti, "hv_bus"])
                    zone = bus_zone_tmp.get(hv_bus)
                    if zone is not None:
                        break
        if zone is None:
            continue
        sn       = float(net_tmp.gen.at[g_idx, "sn_mva"])
        p_max_mw = float(net_tmp.gen.at[g_idx, "max_p_mw"])
        gen_info.append(dict(
            zone=zone,
            gen_idx=int(g_idx),
            name=net_tmp.gen.at[g_idx, "name"] or f"Gen_{g_idx}",
            s_rated_mva=sn,
            p_max_mw=p_max_mw,
            p_min_mw=0.0,
            xd_pu=1.8,
            i_f_max_pu=2.7,
            beta=0.15,
            q0_pu=0.4,
        ))
    gen_info.sort(key=lambda g: (g["zone"], g["gen_idx"]))
    return gen_info


def _plot_gen_capability_pq(logs, palette, out_dir, failed,
                            scenario: str = "wind_replace") -> None:
    """Per-generator P-Q operating-point scatter vs. Milano §12.2 envelope.

    Mirrors Figure 3 of ``visualisation.plot_multi_tso.plot_coordination_comparison``
    but overlays *all* scenarios in ``logs`` (one scatter colour per scenario)
    instead of the original two.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from core.actuator_bounds import GeneratorParameters, compute_generator_q_limits

    try:
        gen_info = _build_gen_info_for_scenario(scenario)
    except Exception as exc:
        print(f"  [plot_compare] could not build gen_info: {exc} -- skipping P-Q plot")
        return
    if not gen_info:
        return

    gens_by_zone: Dict[int, List[Dict]] = {}
    for gi in gen_info:
        gens_by_zone.setdefault(gi["zone"], []).append(gi)

    n_gens = len(gen_info)
    ncols = min(n_gens, 4)
    nrows = (n_gens + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4 * nrows),
                             constrained_layout=True, squeeze=False)
    fig.suptitle("Generator P-Q operating points vs. capability "
                 "(Milano §12.2)", fontweight="bold", fontsize=13)

    flat = 0
    for z in sorted(gens_by_zone.keys()):
        for k, gi in enumerate(gens_by_zone[z]):
            row, col = divmod(flat, ncols)
            ax = axes[row, col]
            gp = GeneratorParameters(
                s_rated_mva=gi["s_rated_mva"],
                p_max_mw=gi["p_max_mw"],
                xd_pu=gi["xd_pu"],
                i_f_max_pu=gi["i_f_max_pu"],
                beta=gi["beta"],
                q0_pu=gi["q0_pu"],
            )
            s_base = gi["s_rated_mva"]
            v = 1.0

            # ── Constraint reference curves (thin dashed) ────────────────
            p_full = np.linspace(0, s_base, 300)
            p_pu = p_full / s_base
            disc_s = np.maximum(1.0 - p_pu ** 2, 0.0)
            q_stator_hi = np.sqrt(disc_s) * s_base
            q_stator_lo = -q_stator_hi
            xd  = gi["xd_pu"]
            i_f = gi["i_f_max_pu"]
            rotor_r = v * i_f / xd
            rotor_c = -v ** 2 / xd
            disc_r = np.maximum(rotor_r ** 2 - p_pu ** 2, 0.0)
            q_rotor_hi = (rotor_c + np.sqrt(disc_r)) * s_base
            p_max_pu = gi["p_max_mw"] / s_base
            q_ue = (-gi["q0_pu"] * v ** 2 + gi["beta"] * p_max_pu) * s_base

            ax.plot(p_full, q_stator_hi, color="grey", linewidth=0.7,
                    linestyle=":", zorder=0, label="Stator limit")
            ax.plot(p_full, q_stator_lo, color="grey", linewidth=0.7,
                    linestyle=":", zorder=0)
            ax.plot(p_full, q_rotor_hi, color="grey", linewidth=0.7,
                    linestyle="-.", zorder=0, label="Rotor limit")
            ax.axhline(q_ue, color="grey", linewidth=0.7,
                       linestyle="--", zorder=0, label="UE limit")

            # ── Composite feasible region ────────────────────────────────
            p_min = gi.get("p_min_mw", 0.0)
            p_sweep = np.linspace(p_min, gi["p_max_mw"], 300)
            q_lo_cap = np.empty_like(p_sweep)
            q_hi_cap = np.empty_like(p_sweep)
            for ip, ppv in enumerate(p_sweep):
                q_lo_cap[ip], q_hi_cap[ip] = compute_generator_q_limits(
                    gp, p_mw=ppv, v_pu=v,
                )
            ax.fill_between(p_sweep, q_lo_cap, q_hi_cap,
                            color="lightgrey", alpha=0.3, zorder=1,
                            label="Feasible region")
            ax.plot(p_sweep, q_hi_cap, color="black", linewidth=1.2, zorder=1)
            ax.plot(p_sweep, q_lo_cap, color="black", linewidth=1.2, zorder=1)
            if p_min > 0:
                ax.axvline(p_min, color="black", linewidth=0.7,
                           linestyle="--", alpha=0.5, zorder=1)
            ax.axvline(gi["p_max_mw"], color="black", linewidth=0.7,
                       linestyle="--", alpha=0.5, zorder=1)

            # ── Scenario operating-point scatters ────────────────────────
            for name, recs in logs.items():
                if not recs:
                    continue
                p_vals, q_vals = [], []
                for r in recs:
                    pv = r.zone_p_gen.get(z)
                    qv = r.zone_q_gen.get(z)
                    if pv is not None and qv is not None and k < len(pv):
                        p_vals.append(float(pv[k]))
                        q_vals.append(float(qv[k]))
                if p_vals:
                    ax.scatter(p_vals, q_vals, color=palette[name],
                               alpha=0.20, s=12, zorder=2, edgecolors="none",
                               label=name)

            ax.set_title(f"{gi['name']} (Z{z})", fontsize=10)
            ax.set_xlabel("P [MW]")
            ax.set_ylabel("Q [Mvar]")
            ax.axhline(0, color="black", linewidth=0.3, linestyle=":")
            ax.grid(True, alpha=0.3)

            # Build a deduplicated legend (avoid one entry per scatter
            # of repeated labels).
            seen = set()
            handles_unique = []
            labels_unique = []
            for h, lbl in zip(*ax.get_legend_handles_labels()):
                if lbl in seen:
                    continue
                seen.add(lbl)
                handles_unique.append(h)
                labels_unique.append(lbl)
            ax.legend(handles_unique, labels_unique,
                      fontsize=7, loc="best", framealpha=0.85)

            flat += 1

    for idx in range(flat, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    _failure_footer(fig, failed)
    _save(fig, out_dir, "compare_gen_capability_pq")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Public entry point
# ---------------------------------------------------------------------------


def plot_scenario_comparison(
    logs: Dict[str, List[MultiTSOIterationRecord]],
    out_dir: str,
    palette: Optional[Dict[str, str]] = None,
    use_tex: bool = False,
    v_set: float = V_SET_DEFAULT,
    v_low: float = 0.95,
    v_high: float = 1.05,
) -> None:
    """Render the comparison figures (PNG + PDF) into ``out_dir``.

    Also writes ``summary.csv`` with one scalar row per scenario, with TS
    and DS metrics split into separate columns.

    Parameters
    ----------
    logs
        Mapping ``scenario_name -> records``.  Empty record lists are
        treated as "scenario diverged" and annotated in figure footers.
    out_dir
        Output directory.  Created if missing.
    palette
        Optional override mapping scenario name to hex colour.
    use_tex
        Forwarded to :func:`apply_serif_style`.
    v_set
        Voltage setpoint (pu) used for the dashed reference line and the
        RMSD computation.  Defaults to 1.03 pu.
    v_low, v_high
        Soft-bound thresholds for violation counting.
    """
    apply_serif_style(use_tex=use_tex)
    os.makedirs(out_dir, exist_ok=True)

    scenario_names = list(logs.keys())
    pal = _resolve_palette(palette, scenario_names)
    failed = [n for n, recs in logs.items() if not recs]

    # Emit summary CSV first so it's available even if a plot crashes.
    summary_table(logs, v_set=v_set, low=v_low, high=v_high
                  ).to_csv(os.path.join(out_dir, "summary.csv"))

    # --- Voltage figures ---
    _plot_voltage_envelope(
        envelope_fn=voltage_envelope_ts,
        header="TS voltage envelope vs t (focus)",
        basename="compare_ts_voltage_envelope",
        logs=logs, palette=pal, out_dir=out_dir, failed=failed, v_set=v_set,
    )
    _plot_voltage_envelope(
        envelope_fn=voltage_envelope_ds,
        header="DS voltage envelope vs t",
        basename="compare_ds_voltage_envelope",
        logs=logs, palette=pal, out_dir=out_dir, failed=failed, v_set=v_set,
    )
    _plot_voltage_rmsd(logs, pal, out_dir, failed, v_set=v_set)
    _plot_voltage_violations(logs, pal, out_dir, failed,
                             low=v_low, high=v_high)

    # --- Losses + headroom ---
    _plot_losses(logs, pal, out_dir, failed)
    _plot_gen_q_headroom(logs, pal, out_dir, failed)

    # --- Tie-line Q deviation from setpoint (Phase B) ---
    _plot_q_tie_deviation(logs, pal, out_dir, failed)

    # --- Per-generator P-Q operating points vs. capability envelope ---
    _plot_gen_capability_pq(logs, pal, out_dir, failed)
