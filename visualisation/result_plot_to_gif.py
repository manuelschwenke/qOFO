#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animated GIF Export for Cascade / TSO-Only Results
===================================================

Converts a list of :class:`IterationRecord` into an animated GIF that
plays through the simulation timesteps, progressively building up all
subplot panels exactly as the :class:`LivePlotter` shows them — **except**
the TSO / DSO objective rows.

Usage::

    from visualisation.result_plot_to_gif import result_plot_to_gif

    result_plot_to_gif(
        log,
        tso_config,
        dso_config,
        output_path="cascade_result.gif",
        fps=15,
        frame_every=3,            # one frame every 3 minutes
        tso_line_max_i_ka=...,    # optional thermal limits
        dso_line_max_i_ka=...,
    )

Requires ``Pillow`` (``pip install Pillow``) for the GIF writer.

Colour Palette
--------------
Uses the same TU Darmstadt PANTONE palette as ``plot_cascade``.

Author: Manuel Schwenke
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from matplotlib.ticker import MaxNLocator
import time as _time
from matplotlib.animation import PillowWriter

if TYPE_CHECKING:
    from controller.dso_controller import DSOControllerConfig
    from controller.tso_controller import TSOControllerConfig
    from run.records import IterationRecord


# ─── TU Darmstadt PANTONE colour palette ────────────────────────────────────
TU_COLOURS: List[str] = [
    "#B1BD00",  # 0  –  5c
    "#004E8A",  # 1  –  1c
    "#CC4C03",  # 2  –  8c
    "#009CDA",  # 3  –  6c
    "#009D81",  # 4  –  3c
    "#721085",  # 5  – 10c
    "#E6A800",  # 6  –  4c
    "#EC6500",  # 7  –  2c
    "#A60084",  # 8  –  9c
    "#C9308E",  # 9  –  7c
    "#F5A300",  # 10 – 11c
]


def _c(idx: int) -> str:
    return TU_COLOURS[idx % len(TU_COLOURS)]


# ─────────────────────────────────────────────────────────────────────────────
#  Data extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_data(log: List[IterationRecord]):
    """
    Pre-extract all time-series arrays from the log so we can slice them
    efficiently per frame.

    Returns a dict of numpy arrays (or lists) keyed by name.
    """
    d = {
        "minutes": [],
        # TN
        "tn_v": [],
        "tn_i": [],
        "tso_min": [],
        "tso_q_pcc": [],
        "tso_q_der": [],
        "tso_v_gen": [],
        "tso_oltc": [],
        "tso_q_gen_min": [],
        "tso_q_gen": [],
        "tso_v_pen_min": [],
        "tso_v_pen": [],
        # DN
        "dn_v": [],
        "dn_i": [],
        "dso_min": [],
        "dso_q_set_min": [],
        "dso_q_set": [],
        "dso_q_act_min": [],
        "dso_q_act": [],
        "dso_q_der": [],
        "dso_oltc": [],
        "dso_shunt": [],
        # Contingency
        "contingency_events": [],
    }

    for rec in log:
        d["minutes"].append(rec.time_s / 60.0 if rec.time_s > 0 else float(rec.minute))

        if rec.plant_tn_voltages_pu is not None:
            d["tn_v"].append(rec.plant_tn_voltages_pu)
        if hasattr(rec, "plant_tn_currents_ka") and rec.plant_tn_currents_ka is not None:
            d["tn_i"].append(rec.plant_tn_currents_ka)
        if hasattr(rec, "plant_dn_voltages_pu") and rec.plant_dn_voltages_pu is not None:
            d["dn_v"].append(rec.plant_dn_voltages_pu)
        if hasattr(rec, "plant_dn_currents_ka") and rec.plant_dn_currents_ka is not None:
            d["dn_i"].append(rec.plant_dn_currents_ka)
        if hasattr(rec, "tso_q_gen_mvar") and rec.tso_q_gen_mvar is not None:
            d["tso_q_gen_min"].append(rec.time_s / 60.0 if rec.time_s > 0 else float(rec.minute))
            d["tso_q_gen"].append(rec.tso_q_gen_mvar)
        if hasattr(rec, "tso_v_penalty") and rec.tso_v_penalty is not None:
            d["tso_v_pen_min"].append(rec.time_s / 60.0 if rec.time_s > 0 else float(rec.minute))
            d["tso_v_pen"].append(rec.tso_v_penalty)

        if rec.tso_active and rec.tso_q_pcc_set_mvar is not None:
            d["tso_min"].append(rec.time_s / 60.0 if rec.time_s > 0 else float(rec.minute))
            d["tso_q_pcc"].append(rec.tso_q_pcc_set_mvar)
            if rec.tso_q_der_mvar is not None:
                d["tso_q_der"].append(rec.tso_q_der_mvar)
            if rec.tso_v_gen_pu is not None:
                d["tso_v_gen"].append(rec.tso_v_gen_pu)
            if rec.tso_oltc_taps is not None:
                d["tso_oltc"].append(rec.tso_oltc_taps)

        if rec.dso_active and rec.dso_q_setpoint_mvar is not None:
            d["dso_q_set_min"].append(rec.time_s / 60.0 if rec.time_s > 0 else float(rec.minute))
            d["dso_q_set"].append(rec.dso_q_setpoint_mvar)
        if rec.dso_active and rec.dso_q_actual_mvar is not None:
            d["dso_q_act_min"].append(rec.time_s / 60.0 if rec.time_s > 0 else float(rec.minute))
            d["dso_q_act"].append(rec.dso_q_actual_mvar)
        if rec.dso_active and rec.dso_q_der_mvar is not None:
            d["dso_min"].append(rec.time_s / 60.0 if rec.time_s > 0 else float(rec.minute))
            d["dso_q_der"].append(rec.dso_q_der_mvar)
            if rec.dso_oltc_taps is not None:
                d["dso_oltc"].append(rec.dso_oltc_taps)
            if rec.dso_shunt_states is not None:
                d["dso_shunt"].append(rec.dso_shunt_states)

        if hasattr(rec, "contingency_events") and rec.contingency_events:
            for entry in rec.contingency_events:
                if isinstance(entry, tuple):
                    _, short_label = entry
                else:
                    short_label = str(entry)
                d["contingency_events"].append(
                    (rec.time_s / 60.0 if rec.time_s > 0 else float(rec.minute), short_label)
                )

    return d


# ─────────────────────────────────────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_contingency_lines(ax, events, up_to_minute: float) -> None:
    """Draw a vertical dashed line for every contingency event up to the current frame."""
    for minute, _label in events:
        if minute > up_to_minute:
            continue
        ax.axvline(minute, color="black", ls="--", lw=1.5, alpha=0.9, zorder=5)


def _draw_contingency_labels(top_ax, events, up_to_minute: float) -> None:
    """Draw contingency event labels above the top subplot of a column.

    Labels are drawn only once per figure at y > 1.0 in axes-fraction
    coordinates, with clip_on=False so they are not clipped by the axes
    boundary. Mirrors the identical method in LivePlotter.
    """
    import matplotlib.transforms as mtransforms

    base_transform = top_ax.get_xaxis_transform()
    for minute, short_label in events:
        if minute > up_to_minute:
            continue
        top_ax.text(
            minute,
            1.05,
            short_label,
            rotation=0,
            va="bottom",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="black",
            clip_on=False,
            zorder=6,
            transform=base_transform,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                alpha=0.85,
            ),
        )



# ─────────────────────────────────────────────────────────────────────────────
#  Main export function
# ─────────────────────────────────────────────────────────────────────────────

def result_plot_to_gif(
    log: List[IterationRecord],
    tso_config: TSOControllerConfig,
    dso_config: DSOControllerConfig,
    output_path: str = "cascade_result.gif",
    fps: int = 15,
    frame_every: float = 1.0,      # minutes between frames (float now)
    tso_line_max_i_ka: Optional[np.ndarray] = None,
    dso_line_max_i_ka: Optional[np.ndarray] = None,
    dpi: int = 100,
    tso_der_names: Optional[List[str]] = None,
    dso_der_names: Optional[List[str]] = None,
    start_minute: Optional[float] = None,  # first animated frame (history already visible)
    end_minute: Optional[float] = None,    # last animated frame (inclusive)
    loop: int = 1,
) -> str:
    """
    Render an animated GIF of all cascade result panels (except objectives).

    Parameters
    ----------
    log : list[IterationRecord]
        Simulation log (from run_cascade or run_tso_only).
    tso_config : TSOControllerConfig
        TSO controller configuration (for labels / indices).
    dso_config : DSOControllerConfig
        DSO controller configuration (for labels / indices).
    output_path : str
        File path for the output GIF.
    fps : int
        Frames per second in the GIF (default 15).
    frame_every : int
        Render one frame every *frame_every* minutes (default 3).
        Lower = smoother but bigger file.
    tso_line_max_i_ka : ndarray, optional
        Thermal limits for TN lines [kA].
    dso_line_max_i_ka : ndarray, optional
        Thermal limits for DN lines [kA].
    dpi : int
        Resolution (default 100).

    Returns
    -------
    output_path : str
        The path to the saved GIF.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for rendering
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    # ── Extract all data ──
    data = _extract_data(log)
    total_minutes = data["minutes"][-1]

    # Resolve DER display names
    _tso_der_names = tso_der_names or [f"sgen {idx}" for idx in tso_config.der_indices]
    _dso_der_names = dso_der_names or [f"sgen {idx}" for idx in dso_config.der_indices]

    total_minutes = data["minutes"][-1]

    # Resolve DER display names
    _tso_der_names = tso_der_names or [f"sgen {idx}" for idx in tso_config.der_indices]
    _dso_der_names = dso_der_names or [f"sgen {idx}" for idx in dso_config.der_indices]

    # ── Determine animation window ──                          ← INSERT FROM HERE
    t_start = float(data["minutes"][0]) if start_minute is None else float(start_minute)
    t_end = float(total_minutes) if end_minute is None else float(end_minute)

    if t_start < float(data["minutes"][0]) or t_end > float(total_minutes):
        raise ValueError(
            f"Requested window [{t_start}, {t_end}] min is outside "
            f"the log range [{data['minutes'][0]}, {total_minutes}] min."
        )
    if t_start >= t_end:
        raise ValueError(
            f"start_minute ({t_start}) must be strictly less than end_minute ({t_end})."
        )

    max_frames = 1000
    duration = t_end - t_start
    min_frame_every = duration / max_frames
    if min_frame_every > frame_every:
        print(
            f"NOTE: auto-adjusted frame_every {frame_every} -> "
            f"{min_frame_every:.3f} to keep frame count <= {max_frames}"
        )
        frame_every = min_frame_every

    frame_minutes = list(np.arange(t_start, t_end, frame_every))
    if not frame_minutes or abs(frame_minutes[-1] - t_end) > 1e-9:
        frame_minutes.append(t_end)

    # ── Detect which subplot rows are needed ──
    has_tso_current = len(tso_config.current_line_indices) > 0
    has_tso_gen = len(tso_config.gen_indices) > 0
    has_tso_oltc = len(tso_config.oltc_trafo_indices) > 0
    has_dso_current = len(dso_config.current_line_indices) > 0
    has_dso_oltc = len(dso_config.oltc_trafo_indices) > 0
    has_dso_shunt = len(dso_config.shunt_bus_indices) > 0

    # TSO rows: V, Q_DER, [I], [Q_gen, V_gen], [OLTC]   (no objective)
    n_tso_rows = 2  # V + Q_DER
    if has_tso_current:
        n_tso_rows += 1
    if has_tso_gen:
        n_tso_rows += 2  # Q_gen + V_gen
    if has_tso_oltc:
        n_tso_rows += 1

    # DSO rows: V, interface Q, Q_DER, [I], [OLTC], [shunt]   (no objective)
    n_dso_rows = 3  # V + interface Q + Q_DER
    if has_dso_current:
        n_dso_rows += 1
    if has_dso_oltc:
        n_dso_rows += 1
    if has_dso_shunt:
        n_dso_rows += 1

    n_rows = max(n_tso_rows, n_dso_rows)

    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(18, 2.2 * n_rows),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(h_pad=0.05, hspace=0.05)  # ← add here

    # If only one row, axes is 1-D — normalise to 2-D
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Cascade OFO Simulation", fontweight="bold", fontsize=13)

    # Hide unused axes
    for r in range(n_tso_rows, n_rows):
        axes[r, 0].set_visible(False)
    for r in range(n_dso_rows, n_rows):
        axes[r, 1].set_visible(False)

    # ── Assign TSO axes (left column) ──
    ax_tso_v = axes[0, 0]
    ax_tso_qder = axes[1, 0]
    tso_row = 2
    ax_tso_current = None
    if has_tso_current:
        ax_tso_current = axes[tso_row, 0]
        tso_row += 1
    ax_tso_qgen = ax_tso_vgen = None
    if has_tso_gen:
        ax_tso_qgen = axes[tso_row, 0]
        tso_row += 1
        ax_tso_vgen = axes[tso_row, 0]
        tso_row += 1
    ax_tso_oltc = None
    if has_tso_oltc:
        ax_tso_oltc = axes[tso_row, 0]
        tso_row += 1

    # ── Assign DSO axes (right column) ──
    ax_dso_v = axes[0, 1]
    ax_dso_iface = axes[1, 1]
    ax_dso_qder = axes[2, 1]
    dso_row = 3
    ax_dso_current = None
    if has_dso_current:
        ax_dso_current = axes[dso_row, 1]
        dso_row += 1
    ax_dso_oltc = None
    if has_dso_oltc:
        ax_dso_oltc = axes[dso_row, 1]
        dso_row += 1
    ax_dso_shunt = None
    if has_dso_shunt:
        ax_dso_shunt = axes[dso_row, 1]
        dso_row += 1

    # ── Pre-convert data to numpy arrays ──
    all_tn_v = np.array(data["tn_v"]) if data["tn_v"] else None
    all_tn_i = np.array(data["tn_i"]) if data["tn_i"] else None
    all_dn_v = np.array(data["dn_v"]) if data["dn_v"] else None
    all_dn_i = np.array(data["dn_i"]) if data["dn_i"] else None

    all_tso_q_der = np.array(data["tso_q_der"]) if data["tso_q_der"] else None
    all_tso_v_gen = np.array(data["tso_v_gen"]) if data["tso_v_gen"] else None
    all_tso_oltc = np.array(data["tso_oltc"]) if data["tso_oltc"] else None
    all_tso_q_gen = np.array(data["tso_q_gen"]) if data["tso_q_gen"] else None
    all_tso_q_pcc = np.array(data["tso_q_pcc"]) if data["tso_q_pcc"] else None

    all_dso_q_set = np.array(data["dso_q_set"]) if data["dso_q_set"] else None
    all_dso_q_act = np.array(data["dso_q_act"]) if data["dso_q_act"] else None
    all_dso_q_der = np.array(data["dso_q_der"]) if data["dso_q_der"] else None
    all_dso_oltc = np.array(data["dso_oltc"]) if data["dso_oltc"] else None
    all_dso_shunt = np.array(data["dso_shunt"]) if data["dso_shunt"] else None

    mins_all = np.array(data["minutes"])
    tso_mins = np.array(data["tso_min"]) if data["tso_min"] else np.array([])
    dso_mins = np.array(data["dso_min"]) if data["dso_min"] else np.array([])
    dso_q_set_mins = np.array(data["dso_q_set_min"]) if data["dso_q_set_min"] else np.array([])
    dso_q_act_mins = np.array(data["dso_q_act_min"]) if data["dso_q_act_min"] else np.array([])
    tso_q_gen_mins = np.array(data["tso_q_gen_min"]) if data["tso_q_gen_min"] else np.array([])
    tso_v_pen_mins = np.array(data["tso_v_pen_min"]) if data["tso_v_pen_min"] else np.array([])
    tso_v_pen_vals = np.array(data["tso_v_pen"]) if data["tso_v_pen"] else np.array([])

    contingency_events = data["contingency_events"]

    # ── Helper: slice arrays up to a given minute ──
    def _up_to(arr_mins, arr_vals, minute):
        """Return slices of mins/vals where mins <= minute."""
        if len(arr_mins) == 0 or arr_vals is None:
            return np.array([]), None
        mask = arr_mins <= minute
        n = mask.sum()
        return arr_mins[:n], arr_vals[:n]

    # ── Frame drawing function ──
    def _draw_frame(frame_idx):
        minute = frame_minutes[frame_idx]

        # Index into per-minute arrays (tn_v, dn_v, etc.)
        # These arrays have one entry per minute (aligned with data["minutes"])
        n_mins = np.searchsorted(mins_all, minute, side="right")

        # ═══════════════ TSO (left column) ═══════════════

        # TSO voltages
        ax_tso_v.clear()
        ax_tso_v.set_ylabel("Voltage / p.u.")
        ax_tso_v.set_title("EHV Bus Voltages", fontsize=10, pad=20)   # ← pad=20
        ax_tso_v.grid(True, alpha=0.3)
        if tso_config.v_setpoints_pu is not None:
            ax_tso_v.axhline(tso_config.v_setpoints_pu[0], color="k", ls="--", lw=1.0)
        if all_tn_v is not None and n_mins > 0:
            tn = all_tn_v[:n_mins]
            m = mins_all[:n_mins]
            for j in range(tn.shape[1]):
                ax_tso_v.plot(m, tn[:, j], lw=0.7, alpha=0.7, color=_c(j))
        ax_tso_v.set_xlim(mins_all[0], t_end)
        _draw_contingency_lines(ax_tso_v, contingency_events, minute)

        # TSO Q_DER
        ax_tso_qder.clear()
        ax_tso_qder.set_ylabel(r"$Q_\mathrm{DER}$ / Mvar")
        ax_tso_qder.set_title("TSO DER Reactive Power", fontsize=10)
        ax_tso_qder.grid(True, alpha=0.3)
        if all_tso_q_der is not None:
            tm, qd = _up_to(tso_mins, all_tso_q_der, minute)
            if qd is not None and len(tm) > 0:
                for j in range(qd.shape[1]):
                    ax_tso_qder.plot(
                        tm, qd[:, j], lw=1.0, color=_c(j),
                        label=_tso_der_names[j],
                    )
                ax_tso_qder.legend(fontsize=6, ncol=4, loc="upper left")
        ax_tso_qder.set_xlim(mins_all[0], t_end)
        _draw_contingency_lines(ax_tso_qder, contingency_events, minute)

        # TSO Line Currents
        if ax_tso_current is not None:
            ax_tso_current.clear()
            ax_tso_current.set_ylabel(r"$I$ / kA")
            ax_tso_current.set_title("TN Line Currents vs. Thermal Limits", fontsize=10)
            ax_tso_current.grid(True, alpha=0.3)
            if all_tn_i is not None and n_mins > 0:
                ti = all_tn_i[:n_mins]
                m = mins_all[:n_mins]
                for j in range(ti.shape[1]):
                    ax_tso_current.plot(m, ti[:, j], lw=0.7, alpha=0.7, color=_c(j))
            if tso_line_max_i_ka is not None:
                for j in range(len(tso_line_max_i_ka)):
                    lim = tso_line_max_i_ka[j]
                    if lim < 1e5:
                        ax_tso_current.axhline(lim, color="r", ls="--", lw=0.8, alpha=0.5)
                ax_tso_current.axhline(np.nan, color="r", ls="--", lw=0.8, label="thermal limit")
                ax_tso_current.legend(fontsize=6, loc="upper left")
            ax_tso_current.set_xlim(mins_all[0], t_end)
            _draw_contingency_lines(ax_tso_current, contingency_events, minute)

        # TSO Q_gen
        if ax_tso_qgen is not None:
            ax_tso_qgen.clear()
            ax_tso_qgen.set_ylabel(r"$Q_\mathrm{gen}$ / Mvar")
            ax_tso_qgen.set_title("Synchronous Generator Q Output", fontsize=10)
            ax_tso_qgen.grid(True, alpha=0.3)
            if all_tso_q_gen is not None:
                tm, qg = _up_to(tso_q_gen_mins, all_tso_q_gen, minute)
                if qg is not None and len(tm) > 0:
                    for j in range(qg.shape[1]):
                        ax_tso_qgen.plot(
                            tm, qg[:, j], lw=1.0, color=_c(j),
                            label=f"Gen {tso_config.gen_indices[j]}",
                        )
                    ax_tso_qgen.legend(fontsize=6, ncol=4, loc="upper left")
            ax_tso_qgen.set_xlim(mins_all[0], t_end)
            _draw_contingency_lines(ax_tso_qgen, contingency_events, minute)

        # TSO V_gen
        if ax_tso_vgen is not None:
            ax_tso_vgen.clear()
            ax_tso_vgen.set_ylabel(r"$V_\mathrm{gen}$ / p.u.")
            ax_tso_vgen.set_title("Generator AVR Setpoints", fontsize=10)
            ax_tso_vgen.grid(True, alpha=0.3)
            if all_tso_v_gen is not None:
                tm, vg = _up_to(tso_mins, all_tso_v_gen, minute)
                if vg is not None and len(tm) > 0:
                    for j in range(vg.shape[1]):
                        ax_tso_vgen.plot(
                            tm, vg[:, j], lw=1.0, color=_c(j),
                            label=f"Gen {tso_config.gen_indices[j]}",
                        )
                    ax_tso_vgen.legend(fontsize=6, ncol=4, loc="upper left")
            ax_tso_vgen.set_xlim(mins_all[0], t_end)
            _draw_contingency_lines(ax_tso_vgen, contingency_events, minute)

        # TSO OLTC
        if ax_tso_oltc is not None:
            ax_tso_oltc.clear()
            ax_tso_oltc.set_ylabel(r"Tap Position $s$")
            ax_tso_oltc.set_title("Machine Transformer OLTC Taps", fontsize=10)
            ax_tso_oltc.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_tso_oltc.grid(True, alpha=0.3)
            if all_tso_oltc is not None:
                tm, ot = _up_to(tso_mins, all_tso_oltc, minute)
                if ot is not None and len(tm) > 0:
                    for j in range(ot.shape[1]):
                        ax_tso_oltc.plot(
                            tm, ot[:, j], lw=1.0, color=_c(j),
                            label=f"OLTC trafo {tso_config.oltc_trafo_indices[j]}",
                        )
                    ax_tso_oltc.legend(fontsize=6, ncol=4, loc="upper left")
            ax_tso_oltc.set_xlim(mins_all[0], t_end)
            _draw_contingency_lines(ax_tso_oltc, contingency_events, minute)

        # ═══════════════ DSO (right column) ═══════════════

        # DSO voltages
        ax_dso_v.clear()
        ax_dso_v.set_ylabel(r"Voltage $V$ / p.u.")
        ax_dso_v.set_title("DN Bus Voltages", fontsize=10, pad=20)
        ax_dso_v.grid(True, alpha=0.3)
        if dso_config.v_setpoints_pu is not None:
            ax_dso_v.axhline(dso_config.v_setpoints_pu[0], color="k", ls="--", lw=1.0)
        if all_dn_v is not None and n_mins > 0:
            dn = all_dn_v[:n_mins]
            m = mins_all[:n_mins]
            for j in range(dn.shape[1]):
                ax_dso_v.plot(m, dn[:, j], lw=0.7, alpha=0.7, color=_c(j))
        # Lock y-axis to data range before drawing limit lines.
        ax_dso_v.relim()
        ax_dso_v.autoscale_view()
        y_lo, y_hi = ax_dso_v.get_ylim()
        ax_dso_v.set_ylim(y_lo, y_hi)
        ax_dso_v.axhline(dso_config.v_min_pu, color="r", ls="--", lw=1.0)
        ax_dso_v.axhline(dso_config.v_max_pu, color="r", ls="--", lw=1.0)
        ax_dso_v.set_xlim(mins_all[0], t_end)
        _draw_contingency_lines(ax_dso_v, contingency_events, minute)

        # DSO Interface Q
        ax_dso_iface.clear()
        ax_dso_iface.set_ylabel(r"$Q$ / Mvar")
        ax_dso_iface.set_title(
            "TSO-DSO Interface Q (load conv., +Q into coupler from TN)",
            fontsize=10,
        )
        ax_dso_iface.grid(True, alpha=0.3)
        if all_tso_q_pcc is not None:
            tm, qp = _up_to(tso_mins, all_tso_q_pcc, minute)
            if qp is not None and len(tm) > 0:
                for j in range(qp.shape[1]):
                    ax_dso_iface.plot(
                        tm, qp[:, j], ls="--", lw=1.0, color=_c(j),
                        label=f"Q_set [{j}]",
                    )
        if all_dso_q_act is not None:
            tm, qa = _up_to(dso_q_act_mins, all_dso_q_act, minute)
            if qa is not None and len(tm) > 0:
                for j in range(qa.shape[1]):
                    ax_dso_iface.plot(
                        tm, qa[:, j], lw=1.8, color=_c(j), alpha=0.8,
                        label=f"Q_actual [{j}]",
                    )
        ax_dso_iface.legend(fontsize=6, ncol=4, loc="upper left")
        ax_dso_iface.set_xlim(mins_all[0], t_end)
        _draw_contingency_lines(ax_dso_iface, contingency_events, minute)

        # DSO Q_DER
        ax_dso_qder.clear()
        ax_dso_qder.set_ylabel(r"$Q_\mathrm{DER}$ / Mvar")
        ax_dso_qder.set_title("DSO DER Reactive Power", fontsize=10)
        ax_dso_qder.grid(True, alpha=0.3)
        if all_dso_q_der is not None:
            tm, qd = _up_to(dso_mins, all_dso_q_der, minute)
            if qd is not None and len(tm) > 0:
                for j in range(qd.shape[1]):
                    lbl = _dso_der_names[j] if j < 10 else None
                    ax_dso_qder.plot(
                        tm, qd[:, j], lw=0.8, alpha=0.7, color=_c(j), label=lbl,
                    )
                if len(dso_config.der_indices) <= 10:
                    ax_dso_qder.legend(fontsize=6, ncol=4, loc="upper left")
        ax_dso_qder.set_xlim(mins_all[0], t_end)
        _draw_contingency_lines(ax_dso_qder, contingency_events, minute)

        # DSO Line Currents
        if ax_dso_current is not None:
            ax_dso_current.clear()
            ax_dso_current.set_ylabel(r"$I$ / kA")
            ax_dso_current.set_title("DN Line Currents vs. Thermal Limits", fontsize=10)
            ax_dso_current.grid(True, alpha=0.3)
            if all_dn_i is not None and n_mins > 0:
                di = all_dn_i[:n_mins]
                m = mins_all[:n_mins]
                for j in range(di.shape[1]):
                    ax_dso_current.plot(m, di[:, j], lw=0.7, alpha=0.7, color=_c(j))
            if dso_line_max_i_ka is not None:
                for j in range(len(dso_line_max_i_ka)):
                    lim = dso_line_max_i_ka[j]
                    if lim < 1e5:
                        ax_dso_current.axhline(lim, color="r", ls="--", lw=0.8, alpha=0.5)
                ax_dso_current.axhline(np.nan, color="r", ls="--", lw=0.8, label="thermal limit")
                ax_dso_current.legend(fontsize=6, loc="upper left")
            ax_dso_current.set_xlim(mins_all[0], t_end)
            _draw_contingency_lines(ax_dso_current, contingency_events, minute)

        # DSO OLTC
        if ax_dso_oltc is not None:
            ax_dso_oltc.clear()
            ax_dso_oltc.set_ylabel(r"Tap Position $s$")
            ax_dso_oltc.set_title("Coupler OLTC Taps", fontsize=10)
            ax_dso_oltc.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_dso_oltc.grid(True, alpha=0.3)
            if all_dso_oltc is not None:
                tm, ot = _up_to(dso_mins, all_dso_oltc, minute)
                if ot is not None and len(tm) > 0:
                    for j in range(ot.shape[1]):
                        ax_dso_oltc.plot(
                            tm, ot[:, j], lw=1.0, color=_c(j),
                            label=f"OLTC trafo3w {dso_config.oltc_trafo_indices[j]}",
                        )
                    ax_dso_oltc.legend(fontsize=6, ncol=4, loc="upper left")
            ax_dso_oltc.set_xlim(mins_all[0], t_end)
            _draw_contingency_lines(ax_dso_oltc, contingency_events, minute)

        # DSO Shunt
        if ax_dso_shunt is not None:
            ax_dso_shunt.clear()
            ax_dso_shunt.set_ylabel("State")
            ax_dso_shunt.set_title("DSO Shunt States", fontsize=10)
            ax_dso_shunt.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_dso_shunt.grid(True, alpha=0.3)
            if all_dso_shunt is not None:
                tm, sh = _up_to(dso_mins, all_dso_shunt, minute)
                if sh is not None and len(tm) > 0:
                    for j in range(sh.shape[1]):
                        ax_dso_shunt.plot(
                            tm, sh[:, j], lw=1.0, color=_c(j),
                            label=f"Shunt bus {dso_config.shunt_bus_indices[j]}",
                        )
                    ax_dso_shunt.legend(fontsize=6, ncol=4, loc="upper left")
            ax_dso_shunt.set_xlim(mins_all[0], t_end)
            _draw_contingency_lines(ax_dso_shunt, contingency_events, minute)

        # Bottom row x-label
        for col in range(2):
            last_visible = None
            for r in range(n_rows - 1, -1, -1):
                if axes[r, col].get_visible():
                    last_visible = axes[r, col]
                    break
            if last_visible is not None:
                last_visible.set_xlabel(r"Time $t$ / min")

        _draw_contingency_labels(ax_tso_v, contingency_events, minute)
        _draw_contingency_labels(ax_dso_v, contingency_events, minute)
        # Progress indicator
        fig.suptitle(
            f"Cascade OFO Simulation  —  t = {minute} min",
            fontweight="bold",
            fontsize=13,
        )

    # ── Animate ──
    n_frames = len(frame_minutes)
    est_sec = n_frames * 0.4  # rough estimate: ~0.4 s per frame
    print(
        f"Rendering {n_frames} frames ({total_minutes} min, "
        f"1 frame / {frame_every} min) at {fps} fps  "
        f"(est. {est_sec/60:.0f}–{est_sec*2/60:.0f} min) ..."
    )

    class LoopAwarePillowWriter(PillowWriter):
        def __init__(self, *args, gif_loop=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._gif_loop = gif_loop

        def finish(self):
            save_kwargs = dict(
                save_all=True,
                append_images=self._frames[1:],
                duration=int(1000 / self.fps),
            )

            # Pillow GIF behavior:
            # - omit loop / None => play once, no looping
            # - 0 => loop forever
            # - positive int => finite loop count
            if self._gif_loop is not None:
                save_kwargs["loop"] = self._gif_loop

            self._frames[0].save(self.outfile, **save_kwargs)

    # We drive the writer manually instead of using FuncAnimation.save()
    # so we can print progress.
    gif_loop = None if loop == 1 else loop
    writer = LoopAwarePillowWriter(fps=fps, gif_loop=gif_loop)
    writer.setup(fig, output_path, dpi=dpi)

    t0 = _time.perf_counter()

    for i in range(n_frames):
        _draw_frame(i)
        writer.grab_frame()

        # Progress every 10 % (and first + last frame)
        pct = (i + 1) / n_frames * 100
        if (i + 1) % max(1, n_frames // 10) == 0 or i == n_frames - 1:
            elapsed = _time.perf_counter() - t0
            eta = elapsed / (i + 1) * (n_frames - i - 1)
            print(
                f"  [{pct:5.1f}%]  frame {i + 1}/{n_frames}  "
                f"(t={frame_minutes[i]} min)  "
                f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s"
            )

    writer.finish()
    plt.close(fig)
    wall = _time.perf_counter() - t0

    print(f"GIF saved to: {output_path}  ({wall:.1f}s)")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_run_dir(base_dir: str) -> str:
    """Return the path of the newest ``NNN_…`` run directory inside *base_dir*."""
    import re

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Results directory not found: {base_dir}")

    candidates = []
    for name in os.listdir(base_dir):
        if re.match(r"^\d{3,}_", name):
            full = os.path.join(base_dir, name)
            if os.path.isdir(full):
                candidates.append(full)

    if not candidates:
        raise FileNotFoundError(f"No run directories found in: {base_dir}")

    # Sort by directory name (NNN prefix gives chronological order)
    candidates.sort()
    return candidates[-1]


def main() -> None:
    """
    Load the latest ``run_cascade`` result from ``results/`` and export an
    animated GIF next to the pickle file.

    Usage::

        python -m visualisation.result_plot_to_gif          # latest run
        python -m visualisation.result_plot_to_gif results/042_20260317_160727
    """
    import argparse
    import os
    import sys

    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    from core.results_storage import DEFAULT_RESULTS_DIR, load_results

    parser = argparse.ArgumentParser(
        description="Export an animated GIF from a saved cascade result.",
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=r'Z:\Python_Projekte\qOFO_GH\results\051_20260324_165159',
        help=(
            "Path to a specific run directory.  "
            "If omitted, the latest run in results/ is used."
        ),
    )
    parser.add_argument(
        "--fps", type=int, default=15,
        help="Frames per second (default: 15).",
    )
    parser.add_argument(
        "--frame-every", type=int, default=1,
        help="Render one frame every N minutes (default: 3).",
    )
    parser.add_argument(
        "--dpi", type=int, default=80,
        help="Resolution in DPI (default: 100).",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output GIF path.  Default: <run_dir>/cascade_result.gif",
    )
    parser.add_argument(
        "--start-minute", type=float, default=1,
        help="First animated frame in minutes. History before this is shown statically.",
    )
    parser.add_argument(
        "--end-minute", type=float, default=720,
        help="Last animated frame in minutes. Defaults to end of simulation.",
    )

    parser.add_argument(
        "--loop", type=int, default=1,
        help="GIF loop count. 0 = infinite (default), 1 = play once.",
    )

    args = parser.parse_args()

    # Resolve run directory
    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        run_dir = _find_latest_run_dir(DEFAULT_RESULTS_DIR)
    print(f"Loading results from: {run_dir}")

    result, config = load_results(run_dir)

    # Derive thermal-limit arrays from the stored configs
    tso_cfg = result.tso_config
    dso_cfg = result.dso_config

    tso_lim = None
    if tso_cfg.current_line_max_i_ka is not None:
        tso_lim = np.array(tso_cfg.current_line_max_i_ka) * tso_cfg.i_max_pu

    dso_lim = None
    if dso_cfg.current_line_max_i_ka is not None:
        dso_lim = np.array(dso_cfg.current_line_max_i_ka) * dso_cfg.i_max_pu

    # Output path
    if args.output is not None:
        out_path = args.output
    else:
        out_path = os.path.join(run_dir, "cascade_result.gif")

    result_plot_to_gif(
        result.log,
        tso_cfg,
        dso_cfg,
        output_path=out_path,
        fps=args.fps,
        frame_every=args.frame_every,
        tso_line_max_i_ka=tso_lim,
        dso_line_max_i_ka=dso_lim,
        dpi=args.dpi,
        start_minute=args.start_minute,
        end_minute=args.end_minute,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()
