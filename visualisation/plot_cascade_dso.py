"""
visualisation/plot_cascade_dso.py
=================================
Live plotter for Figure 2 — CASCADE-DSO CONTROLLER.

The layout adapts to the actual number of DSO controllers at runtime.
For N_dso DSOs the grid is:

    MEASUREMENTS (orange)
        1.               DSO Voltages per HV Network Group
        2..(N_dso+1).    TSO-DSO Interface Q — one tile per DSO
        N_dso+2.         DSO Line Currents (Max / Mean / Min per group)

    ACTUATORS (dark blue)
        N_dso+3.                       DSO DER Reactive Power per HV Group
        (N_dso+4)..(2*N_dso+3).        TSO-DSO Transformer Tap Positions
                                       — one tile per DSO
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from visualisation.style import (
    COLOUR_ACT_BAND,
    COLOUR_MEAS_BAND,
    TITLE_BAR_HEIGHT_FRAC,
    _c,
    apply_serif_style,
    apply_x_fmt,
    draw_figure_header,
    fill_section_band,
    position_figure_in_slot,
    raise_figure_to_front,
    tile_title,
)

if TYPE_CHECKING:
    from experiments.helpers import MultiTSOIterationRecord


_NO_DATA_COLOR = "gray"


def _fill_empty(ax: plt.Axes, message: str) -> None:
    """Render a muted placeholder message on an empty tile."""
    ax.text(0.5, 0.5, message,
            transform=ax.transAxes, ha="center", va="center",
            color=_NO_DATA_COLOR, fontsize=8, style="italic")


def _dso_num(dso_id: str) -> str:
    """Return the short DSO index used on y-axis labels.

    ``"DSO_1"`` -> ``"1"``, anything else -> the full id.
    """
    if "_" in dso_id:
        tail = dso_id.rsplit("_", 1)[1]
        if tail:
            return tail
    return dso_id


def _trafo_short(trafo_key: str) -> str:
    """``"DSO_1|trafo_37"`` -> ``"T37"``; falls back to the raw key."""
    if "|trafo_" in trafo_key:
        return "T" + trafo_key.rsplit("|trafo_", 1)[1]
    return trafo_key


class CascadeDSOLivePlotter:
    """Live figure 2 — CASCADE-DSO CONTROLLER."""

    def __init__(
        self,
        dso_ids: Sequence[str],
        *,
        v_setpoint_pu: float = 1.03,
        v_min_pu: float = 0.9,
        v_max_pu: float = 1.1,
        sub_minute: bool = False,
        update_every: int = 1,
        slot_idx: int = 1,
        layout: str = "dual_screen",
        show_line_currents: bool = True,
        use_tex: bool = False,
    ) -> None:
        apply_serif_style(use_tex=use_tex)
        plt.ion()

        self._dso_ids: List[str] = list(dso_ids)
        self._v_set = v_setpoint_pu
        self._v_min = v_min_pu
        self._v_max = v_max_pu
        self._sub_minute = sub_minute
        self._update_every = max(1, int(update_every))
        self._show_iline = bool(show_line_currents)
        self._call_count = 0

        # Group IDs are learnt at first update() via rec.dso_controller_group.
        self._group_ids: List[str] = []
        # Per-DSO trafo IDs (trafo_key strings) populated at first update().
        self._dso_trafo_ids: Dict[str, List[str]] = {d: [] for d in self._dso_ids}

        # ── Accumulators ─────────────────────────────────────────────────
        self._t_all: List[float] = []
        self._t_dso: List[float] = []   # DSO-active updates only

        # Per-group (populated as group_ids are discovered)
        self._g_v_min:   Dict[str, List[float]] = {}
        self._g_v_mean:  Dict[str, List[float]] = {}
        self._g_v_max:   Dict[str, List[float]] = {}
        self._g_i_max:   Dict[str, List[float]] = {}
        self._g_i_mean:  Dict[str, List[float]] = {}
        self._g_i_min:   Dict[str, List[float]] = {}
        self._g_q_der:   Dict[str, List[float]] = {}
        self._g_q_der_min: Dict[str, List[float]] = {}
        self._g_q_der_max: Dict[str, List[float]] = {}

        # Per-trafo (populated as trafo_keys are discovered)
        self._trafo_q_set:    Dict[str, List[float]] = {}
        self._trafo_q_actual: Dict[str, List[float]] = {}
        self._trafo_tap:      Dict[str, List[float]] = {}

        # ── Figure + GridSpec ────────────────────────────────────────────
        n = len(self._dso_ids)
        # meas: band + 1 volt + N iface + (1 line currents if shown)
        # act:  band + 1 DER + N taps
        iline_row_count = 1 if self._show_iline else 0
        n_rows = 2 * n + 4 + iline_row_count
        fig_h = max(8.0, 1.35 * (2 * n + 3))
        self._fig = plt.figure(figsize=(6.2, min(fig_h, 13.0)))
        try:
            self._fig.canvas.manager.set_window_title("Cascade-DSO Controller")
        except Exception:
            pass

        self._fig.subplots_adjust(
            top=1.0 - TITLE_BAR_HEIGHT_FRAC - 0.005,
            bottom=0.04, left=0.10, right=0.985,
            hspace=0.38,
        )
        draw_figure_header(self._fig, "Cascade-DSO Controller")

        plot_h = 1.0
        band_h = 0.2
        heights = [band_h] + [plot_h] * (1 + n)   # meas band, voltages, iface Q (N)
        if self._show_iline:
            heights.append(plot_h)                 # optional line currents
        heights += [band_h] + [plot_h] * (1 + n)   # act band, DER Q, taps (N)

        gs = GridSpec(
            n_rows, 1, figure=self._fig, height_ratios=heights, hspace=0.38,
        )

        # Row 0: MEAS band
        ax = self._fig.add_subplot(gs[0, 0])
        fill_section_band(ax, "Measurements", COLOUR_MEAS_BAND)

        # Row 1: DSO voltages per group
        self._ax_v = self._fig.add_subplot(gs[1, 0])
        tile_title(self._ax_v, "DSO Voltages per HV Network Group")

        # Rows 2..n+1: interface Q per DSO (one axes each).  Only the
        # first DSO's axis carries the section title; the others are
        # identified via their y-axis label ("Q_{DSk} / Mvar") to free
        # vertical space.
        self._ax_iface: Dict[str, plt.Axes] = {}
        for idx, dso_id in enumerate(self._dso_ids):
            axi = self._fig.add_subplot(gs[2 + idx, 0], sharex=self._ax_v)
            if idx == 0:
                tile_title(axi, "TSO-DSO Interface Q")
            self._ax_iface[dso_id] = axi

        # Optional DSO line currents row
        row = 2 + n
        if self._show_iline:
            self._ax_iline = self._fig.add_subplot(gs[row, 0], sharex=self._ax_v)
            tile_title(self._ax_iline, "DSO Line Currents (Loading %)")
            row += 1
        else:
            self._ax_iline = None

        # ACT band
        ax_act = self._fig.add_subplot(gs[row, 0])
        fill_section_band(ax_act, "Actuators", COLOUR_ACT_BAND)
        row += 1

        # DSO DER Q per group
        self._ax_qder = self._fig.add_subplot(gs[row, 0], sharex=self._ax_v)
        tile_title(self._ax_qder, "DSO DER Q per HV Network Group")
        row += 1

        # Tap positions per DSO.  As for interface Q above, only the
        # first DSO's axis carries the section title; others use a
        # compact "DS k / Tap" y-axis label.
        self._ax_tap: Dict[str, plt.Axes] = {}
        for idx, dso_id in enumerate(self._dso_ids):
            axt = self._fig.add_subplot(gs[row + idx, 0], sharex=self._ax_v)
            if idx == 0:
                tile_title(axt, "TSO-DSO Transformer Taps")
            self._ax_tap[dso_id] = axt

        # Collect all plot axes for x-axis formatting
        self._plot_axes: List[plt.Axes] = (
            [self._ax_v]
            + [self._ax_iface[d] for d in self._dso_ids]
        )
        if self._ax_iline is not None:
            self._plot_axes.append(self._ax_iline)
        self._plot_axes.append(self._ax_qder)
        self._plot_axes += [self._ax_tap[d] for d in self._dso_ids]

        for ax in self._plot_axes:
            ax.tick_params(axis="both", labelsize=8)
        for ax in self._plot_axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        self._plot_axes[-1].set_xlabel(
            "Time / s" if sub_minute else "Time / min"
        )

        position_figure_in_slot(self._fig, slot_idx, layout=layout, n_slots=3)
        plt.pause(0.01)
        raise_figure_to_front(self._fig)

    # ─── update ─────────────────────────────────────────────────────────

    def update(self, rec: "MultiTSOIterationRecord") -> None:
        t_unit = rec.time_s if self._sub_minute else rec.time_s / 60.0
        self._t_all.append(t_unit)

        # Discover group IDs lazily from the record
        for g in sorted(rec.dso_group_v_min_pu.keys()):
            if g not in self._g_v_min:
                self._group_ids.append(g)
                self._g_v_min[g]     = [np.nan] * (len(self._t_all) - 1)
                self._g_v_mean[g]    = [np.nan] * (len(self._t_all) - 1)
                self._g_v_max[g]     = [np.nan] * (len(self._t_all) - 1)
                self._g_i_max[g]     = [np.nan] * (len(self._t_all) - 1)
                self._g_i_mean[g]    = [np.nan] * (len(self._t_all) - 1)
                self._g_i_min[g]     = [np.nan] * (len(self._t_all) - 1)
                self._g_q_der[g]     = [np.nan] * (len(self._t_all) - 1)
                self._g_q_der_min[g] = [np.nan] * (len(self._t_all) - 1)
                self._g_q_der_max[g] = [np.nan] * (len(self._t_all) - 1)

        # Append per-group voltages + line loadings (every step)
        for g in self._group_ids:
            self._g_v_min[g] .append(rec.dso_group_v_min_pu .get(g, np.nan))
            self._g_v_mean[g].append(rec.dso_group_v_mean_pu.get(g, np.nan))
            self._g_v_max[g] .append(rec.dso_group_v_max_pu .get(g, np.nan))
            self._g_i_max[g] .append(rec.dso_group_i_max_pct .get(g, np.nan))
            self._g_i_mean[g].append(rec.dso_group_i_mean_pct.get(g, np.nan))
            self._g_i_min[g] .append(rec.dso_group_i_min_pct .get(g, np.nan))
            self._g_q_der[g]     .append(rec.dso_group_q_der_mvar    .get(g, np.nan))
            self._g_q_der_min[g] .append(rec.dso_group_q_der_min_mvar.get(g, np.nan))
            self._g_q_der_max[g] .append(rec.dso_group_q_der_max_mvar.get(g, np.nan))

        # Per-trafo data — discover trafo keys lazily
        for trafo_key in list(rec.dso_trafo_q_actual_mvar.keys()) + list(rec.dso_trafo_tap_pos.keys()):
            if trafo_key not in self._trafo_q_actual:
                # Bucket by DSO id (first segment before "|")
                dso_id = trafo_key.split("|", 1)[0]
                if dso_id in self._dso_trafo_ids and trafo_key not in self._dso_trafo_ids[dso_id]:
                    self._dso_trafo_ids[dso_id].append(trafo_key)
                self._trafo_q_set   [trafo_key] = [np.nan] * (len(self._t_dso))
                self._trafo_q_actual[trafo_key] = [np.nan] * (len(self._t_dso))
                self._trafo_tap     [trafo_key] = [np.nan] * (len(self._t_dso))

        if rec.dso_active or any(
            (k in rec.dso_trafo_q_actual_mvar) for k in self._trafo_q_actual.keys()
        ):
            self._t_dso.append(t_unit)
            for trafo_key in self._trafo_q_actual.keys():
                self._trafo_q_set   [trafo_key].append(
                    rec.dso_trafo_q_set_mvar   .get(trafo_key, np.nan))
                self._trafo_q_actual[trafo_key].append(
                    rec.dso_trafo_q_actual_mvar.get(trafo_key, np.nan))
                self._trafo_tap     [trafo_key].append(
                    rec.dso_trafo_tap_pos      .get(trafo_key, np.nan))

        self._call_count += 1
        if self._call_count % self._update_every == 0:
            self._redraw()

    # ─── redraw ─────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        self._redraw_voltages()
        for dso_id in self._dso_ids:
            self._redraw_interface_q(dso_id)
        if self._ax_iline is not None:
            self._redraw_line_currents()
        self._redraw_der_q()
        for dso_id in self._dso_ids:
            self._redraw_taps(dso_id)
        # Shared-x axis: after ax.clear() each redraw exposes the
        # xticklabels again, so re-hide them on all but the last subplot
        # and re-apply the xlabel that ax.clear() wiped.
        for ax in self._plot_axes:
            apply_x_fmt(ax, sub_minute=self._sub_minute)
        for ax in self._plot_axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlabel("")
        self._plot_axes[-1].set_xlabel(
            "Time / s" if self._sub_minute else "Time / min"
        )
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    # ─── per-tile redraws ───────────────────────────────────────────────

    def _redraw_voltages(self) -> None:
        ax = self._ax_v
        ax.clear()
        tile_title(ax, "DSO Voltages per HV Network Group")
        ax.set_ylabel(r"V / p.u.")
        if not self._group_ids:
            _fill_empty(ax, "no HV-group measurements available")
            return
        ax.axhline(self._v_set, color="k",       ls=":",  lw=0.8, alpha=0.7)
        #ax.axhline(self._v_min, color="#B90F22", ls="--", lw=0.6, alpha=0.7) # ToDo: Not needed
        #ax.axhline(self._v_max, color="#B90F22", ls="--", lw=0.6, alpha=0.7) # ToDo: Not needed
        t = np.asarray(self._t_all, dtype=float)
        for i, g in enumerate(self._group_ids):
            c = _c(5 + i)
            lo = np.asarray(self._g_v_min[g],  dtype=float)
            hi = np.asarray(self._g_v_max[g],  dtype=float)
            mn = np.asarray(self._g_v_mean[g], dtype=float)
            ax.fill_between(t, lo, hi, color=c, alpha=0.15)
            ax.plot(t, mn, color=c, lw=1.0, label=g)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7,
                  ncol=min(len(self._group_ids), 3), frameon=False)

    def _redraw_interface_q(self, dso_id: str) -> None:
        ax = self._ax_iface[dso_id]
        ax.clear()
        # Only the first DSO's axis gets the section title.
        if dso_id == self._dso_ids[0]:
            tile_title(ax, "TSO-DSO Interface Q")
        num = _dso_num(dso_id)
        # Compact two-line y-label:  "Q_{DS<k>}"  /  "/ Mvar"
        ax.set_ylabel(rf"$Q_{{\mathrm{{DS{num}}}}}$" + "\n/ Mvar")
        trafo_ids = self._dso_trafo_ids.get(dso_id, [])
        if not trafo_ids or not self._t_dso:
            _fill_empty(ax, "interface Q not available")
            return
        t = np.asarray(self._t_dso, dtype=float)
        n = t.size
        # One line per coupling transformer: solid = actual, dashed = setpoint.
        # Colour-cycle per trafo index so the three trafos of a given DSO are
        # visually distinct; the same colour ordering is reused across DSOs.
        for k, trafo_key in enumerate(trafo_ids):
            c = _c(3 + k)
            set_vals = np.asarray(self._trafo_q_set[trafo_key],    dtype=float)
            act_vals = np.asarray(self._trafo_q_actual[trafo_key], dtype=float)
            if set_vals.size < n:
                set_vals = np.concatenate([set_vals, np.full(n - set_vals.size, np.nan)])
            if act_vals.size < n:
                act_vals = np.concatenate([act_vals, np.full(n - act_vals.size, np.nan)])
            set_vals = set_vals[:n]
            act_vals = act_vals[:n]
            t_label = _trafo_short(trafo_key)
            ax.plot(t, act_vals, color=c, lw=1.0, label=t_label)
            ax.plot(t, set_vals, color=c, lw=0.8, ls="--", drawstyle="steps-post")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=6,
                  ncol=min(len(trafo_ids), 3), frameon=False)

    def _redraw_line_currents(self) -> None:
        ax = self._ax_iline
        ax.clear()
        tile_title(ax, "DSO Line Currents (Loading %)")
        ax.set_ylabel(r"I / %")
        if not self._group_ids:
            _fill_empty(ax, "no HV-group line measurements available")
            return
        ax.axhline(100.0, color="#B90F22", ls="--", lw=0.6, alpha=0.7)
        t = np.asarray(self._t_all, dtype=float)
        for i, g in enumerate(self._group_ids):
            c = _c(5 + i)
            lo = np.asarray(self._g_i_min[g],  dtype=float)
            hi = np.asarray(self._g_i_max[g],  dtype=float)
            mn = np.asarray(self._g_i_mean[g], dtype=float)
            ax.fill_between(t, lo, hi, color=c, alpha=0.12)
            ax.plot(t, mn, color=c, lw=1.0, label=g)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7,
                  ncol=min(len(self._group_ids), 3), frameon=False)

    def _redraw_der_q(self) -> None:
        ax = self._ax_qder
        ax.clear()
        tile_title(ax, "DSO DER Q per HV Network Group")
        ax.set_ylabel(r"Q$_\mathrm{DER}$ / Mvar")
        if not self._group_ids:
            _fill_empty(ax, "no DSO DER dispatch available")
            return
        t = np.asarray(self._t_all, dtype=float)
        for i, g in enumerate(self._group_ids):
            c = _c(5 + i)
            #qmin = np.asarray(self._g_q_der_min[g], dtype=float) #ToDo: Not wanted at the moment
            #qmax = np.asarray(self._g_q_der_max[g], dtype=float) #ToDo: Not wanted at the moment
            qval = np.asarray(self._g_q_der[g],     dtype=float)
            #ax.fill_between(t, qmin, qmax, color=c, alpha=0.15) #ToDo: Not wanted at the moment
            ax.plot(t, qval, color=c, lw=1.0, label=g)
        ax.axhline(0.0, color="k", ls=":", lw=0.6, alpha=0.4)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7,
                  ncol=min(len(self._group_ids), 3), frameon=False)

    def _redraw_taps(self, dso_id: str) -> None:
        ax = self._ax_tap[dso_id]
        ax.clear()
        # Only the first DSO's axis gets the section title.
        if dso_id == self._dso_ids[0]:
            tile_title(ax, "TSO-DSO Transformer Taps")
        num = _dso_num(dso_id)
        # Compact two-line y-label: "DS <k>" / "Tap"
        ax.set_ylabel(f"DS {num}\nTap")
        trafo_ids = self._dso_trafo_ids.get(dso_id, [])
        if not trafo_ids or not self._t_dso:
            _fill_empty(ax, "no transformer taps available")
            return
        t = np.asarray(self._t_dso, dtype=float)
        for k, trafo_key in enumerate(trafo_ids):
            vals = np.asarray(self._trafo_tap[trafo_key], dtype=float)
            if vals.size < t.size:
                vals = np.concatenate([vals, np.full(t.size - vals.size, np.nan)])
            vals = vals[:t.size]
            ax.plot(t, vals, color=_c(3 + k), lw=0.9, drawstyle="steps-post",
                    label=_trafo_short(trafo_key))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=6,
                  ncol=min(len(trafo_ids), 3), frameon=False)
