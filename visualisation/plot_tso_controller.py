"""
visualisation/plot_tso_controller.py
====================================
Live plotter for Figure 1 — MULTI-TSO CONTROLLER.

Tiles grouped into two colour-banded sections.  Tiles 3-5 are optional
(``show_reserves`` / ``show_tie_flows`` / ``show_line_currents``).

    MEASUREMENTS (orange)
        1. TSO Voltages per Zone (v_min / mean / max band per zone)
        2. TSO Generator Reactive Power Injections
        3. TSO Reactive Reserve — per-zone mean normalised reserve of
           synchronous machines (solid) and TSO DER (dashed)   [optional]
        4. TSO Reactive Power Tie-Line Flows                    [optional]
        5. TSO Line Currents (max / mean / min loading % per zone) [optional]

    ACTUATORS (dark blue)
        6. TSO DER Reactive Power Infeed per Zone
        7. TSO Generator AVR Setpoints
        8. TSO Machine Transformer Setpoints (OLTC taps)
        9. TSO Shunt States (MSC/MSR)

The figure is sized to fit one third of the screen width at full height
and is placed into slot 0 by :func:`visualisation.style.position_figure_in_slot`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

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


class TSOControllerLivePlotter:
    """Live figure 1 — MULTI-TSO CONTROLLER (8 tiles, 2 sections)."""

    def __init__(
        self,
        zone_ids: Sequence[int],
        tie_line_pairs: Sequence[Tuple[int, int]],
        n_oltc_per_zone: Dict[int, int],
        n_shunt_per_zone: Dict[int, int],
        *,
        v_setpoint_pu: float = 1.03,
        v_min_pu: float = 0.9,
        v_max_pu: float = 1.1,
        sub_minute: bool = False,
        update_every: int = 1,
        slot_idx: int = 0,
        layout: str = "dual_screen",
        show_line_currents: bool = True,
        show_reserves: bool = True,
        show_tie_flows: bool = True,
        use_tex: bool = False,
    ) -> None:
        apply_serif_style(use_tex=use_tex)
        plt.ion()

        self._zone_ids = list(zone_ids)
        self._tie_pairs = list(tie_line_pairs)
        self._n_oltc = dict(n_oltc_per_zone)
        self._n_shunt = dict(n_shunt_per_zone)
        self._v_set = v_setpoint_pu
        self._v_min = v_min_pu
        self._v_max = v_max_pu
        self._sub_minute = sub_minute
        self._update_every = max(1, int(update_every))
        self._show_iline = bool(show_line_currents)
        self._show_reserve = bool(show_reserves)
        self._show_tie = bool(show_tie_flows)
        self._call_count = 0

        # ── Accumulators ─────────────────────────────────────────────────
        self._t_all: List[float] = []   # every update() call
        self._t_tso: List[float] = []   # TSO-active updates only

        self._zone_v_min:  Dict[int, List[float]] = {z: [] for z in self._zone_ids}
        self._zone_v_mean: Dict[int, List[float]] = {z: [] for z in self._zone_ids}
        self._zone_v_max:  Dict[int, List[float]] = {z: [] for z in self._zone_ids}
        self._zone_q_gen:  Dict[int, List[np.ndarray]] = {z: [] for z in self._zone_ids}
        self._zone_gen_reserve: Dict[int, List[np.ndarray]] = {z: [] for z in self._zone_ids}
        self._zone_der_reserve: Dict[int, List[np.ndarray]] = {z: [] for z in self._zone_ids}
        self._zone_i_max:  Dict[int, List[float]] = {z: [] for z in self._zone_ids}
        self._zone_i_mean: Dict[int, List[float]] = {z: [] for z in self._zone_ids}
        self._zone_i_min:  Dict[int, List[float]] = {z: [] for z in self._zone_ids}
        self._tie_q: Dict[Tuple[int, int], List[float]] = {
            p: [] for p in self._tie_pairs
        }

        self._zone_q_der: Dict[int, List[np.ndarray]] = {z: [] for z in self._zone_ids}
        self._zone_v_gen: Dict[int, List[np.ndarray]] = {z: [] for z in self._zone_ids}
        self._zone_oltc:  Dict[int, List[np.ndarray]] = {z: [] for z in self._zone_ids}
        self._zone_shunt: Dict[int, List[np.ndarray]] = {z: [] for z in self._zone_ids}

        # ── Figure + GridSpec ────────────────────────────────────────────
        self._fig = plt.figure(figsize=(6.2, 10.0))
        try:
            self._fig.canvas.manager.set_window_title("Multi-TSO Controller")
        except Exception:
            pass

        self._fig.subplots_adjust(
            top=1.0 - TITLE_BAR_HEIGHT_FRAC - 0.005,
            bottom=0.045, left=0.10, right=0.985,
            hspace=0.60,
        )
        draw_figure_header(self._fig, "Multi-TSO Controller")

        plot_h = 1.0
        band_h = 0.18
        # Measurement tiles: V + gen Q always; reserves / tie Q / line currents
        # optional.  Count the active ones to size the GridSpec.
        n_meas = 2 + int(self._show_reserve) + int(self._show_tie) + int(self._show_iline)
        heights = [band_h] + [plot_h] * n_meas + [band_h] + [plot_h] * 4
        gs = GridSpec(
            len(heights), 1, figure=self._fig,
            height_ratios=heights, hspace=0.60,
        )

        # Row 0: MEAS band
        self._ax_band_meas = self._fig.add_subplot(gs[0, 0])
        fill_section_band(self._ax_band_meas, "Measurements", COLOUR_MEAS_BAND)

        # Measurement rows (V + gen Q always; reserves / tie Q / line currents optional)
        row = 1
        self._ax_v    = self._fig.add_subplot(gs[row, 0]); row += 1
        self._ax_qgen = self._fig.add_subplot(gs[row, 0], sharex=self._ax_v); row += 1
        if self._show_reserve:
            self._ax_reserve = self._fig.add_subplot(gs[row, 0], sharex=self._ax_v); row += 1
        else:
            self._ax_reserve = None
        if self._show_tie:
            self._ax_tie = self._fig.add_subplot(gs[row, 0], sharex=self._ax_v); row += 1
        else:
            self._ax_tie = None
        if self._show_iline:
            self._ax_iline = self._fig.add_subplot(gs[row, 0], sharex=self._ax_v); row += 1
        else:
            self._ax_iline = None
        act_start = row

        # ACT band
        self._ax_band_act = self._fig.add_subplot(gs[act_start, 0])
        fill_section_band(self._ax_band_act, "Actuators", COLOUR_ACT_BAND)

        # First actuator row: TSO DER reactive-power infeed.
        self._ax_qder = self._fig.add_subplot(gs[act_start + 1, 0], sharex=self._ax_v)
        self._ax_vgen  = self._fig.add_subplot(gs[act_start + 2, 0], sharex=self._ax_v)
        self._ax_oltc  = self._fig.add_subplot(gs[act_start + 3, 0], sharex=self._ax_v)
        self._ax_shunt = self._fig.add_subplot(gs[act_start + 4, 0], sharex=self._ax_v)

        self._plot_axes = [self._ax_v, self._ax_qgen]
        if self._ax_reserve is not None:
            self._plot_axes.append(self._ax_reserve)
        if self._ax_tie is not None:
            self._plot_axes.append(self._ax_tie)
        if self._ax_iline is not None:
            self._plot_axes.append(self._ax_iline)
        self._plot_axes += [
            self._ax_qder, self._ax_vgen, self._ax_oltc, self._ax_shunt,
        ]

        tile_title(self._ax_v,     "TSO Voltages per Zone")
        tile_title(self._ax_qgen,  "TSO Generator Q Injections")
        if self._ax_reserve is not None:
            tile_title(self._ax_reserve, "TSO Reactive Reserve (SG solid / DER dashed)")
        if self._ax_tie is not None:
            tile_title(self._ax_tie,   "TSO Reactive Power Tie-Line Flows")
        if self._ax_iline is not None:
            tile_title(self._ax_iline, "TSO Line Currents (Loading %)")
        tile_title(self._ax_qder,      "TSO DER Q Infeed per Zone")
        tile_title(self._ax_vgen,  "TSO Generator AVR Setpoints")
        tile_title(self._ax_oltc,  "TSO Machine Transformer Taps")
        tile_title(self._ax_shunt, "TSO Shunt States  (+ reactor/MSR  /  - capacitor/MSC)")

        for ax in self._plot_axes:
            ax.tick_params(axis="both", labelsize=8)
        for ax in self._plot_axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        self._ax_shunt.set_xlabel("Time / s" if sub_minute else "Time / min")

        position_figure_in_slot(self._fig, slot_idx, layout=layout, n_slots=3)
        plt.pause(0.01)
        raise_figure_to_front(self._fig)

    # ─── update ─────────────────────────────────────────────────────────

    def update(self, rec: "MultiTSOIterationRecord") -> None:
        """Append this record's data and redraw if cadence matches."""
        t_unit = rec.time_s if self._sub_minute else rec.time_s / 60.0
        self._t_all.append(t_unit)

        for z in self._zone_ids:
            self._zone_v_min[z].append(rec.zone_v_min.get(z, float("nan")))
            self._zone_v_mean[z].append(rec.zone_v_mean.get(z, float("nan")))
            self._zone_v_max[z].append(rec.zone_v_max.get(z, float("nan")))
            self._zone_q_gen[z].append(
                np.asarray(rec.zone_q_gen.get(z, []), dtype=float)
            )
            self._zone_gen_reserve[z].append(
                np.asarray(rec.gen_q_reserve.get(z, []), dtype=float)
            )
            self._zone_der_reserve[z].append(
                np.asarray(rec.tso_der_q_reserve.get(z, []), dtype=float)
            )
            self._zone_i_max[z].append(rec.zone_line_loading_max_pct.get(z, float("nan")))
            self._zone_i_mean[z].append(rec.zone_line_loading_mean_pct.get(z, float("nan")))
            self._zone_i_min[z].append(rec.zone_line_loading_min_pct.get(z, float("nan")))
        for pair in self._tie_pairs:
            self._tie_q[pair].append(rec.zone_tie_q_mvar.get(pair, float("nan")))

        if rec.tso_active:
            self._t_tso.append(t_unit)
            for z in self._zone_ids:
                self._zone_q_der[z].append(
                    np.asarray(rec.zone_q_der.get(z, []), dtype=float)
                )
                self._zone_v_gen[z].append(
                    np.asarray(rec.zone_v_gen.get(z, []), dtype=float)
                )
                self._zone_oltc[z].append(
                    np.asarray(rec.zone_oltc_taps.get(z, []), dtype=float)
                )
                self._zone_shunt[z].append(
                    np.asarray(rec.zone_tso_shunt_states.get(z, []), dtype=float)
                )

        self._call_count += 1
        if self._call_count % self._update_every == 0:
            self._redraw()

    # ─── redraw ─────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        self._redraw_voltages()
        self._redraw_gen_q()
        if self._ax_reserve is not None:
            self._redraw_reserves()
        if self._ax_tie is not None:
            self._redraw_tie_q()
        if self._ax_iline is not None:
            self._redraw_line_currents()
        self._redraw_der_q()
        self._redraw_v_gen()
        self._redraw_oltc()
        self._redraw_shunts()
        for ax in self._plot_axes:
            apply_x_fmt(ax, sub_minute=self._sub_minute)
        # Shared-x axis cleanup — see note in CascadeDSOLivePlotter._redraw.
        for ax in self._plot_axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlabel("")
        self._plot_axes[-1].set_xlabel(
            "Time / s" if self._sub_minute else "Time / min"
        )
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    # ─── per-tile redraws ───────────────────────────────────────────────

    def _zone_legend_handles(self) -> List[plt.Line2D]:
        return [
            plt.Line2D([0], [0], color=_c(i + 1), lw=1.2, label=f"Z{z}")
            for i, z in enumerate(self._zone_ids)
        ]

    def _redraw_voltages(self) -> None:
        ax = self._ax_v
        ax.clear()
        tile_title(ax, "TSO Voltages per Zone")
        ax.axhline(self._v_set, color="k",       ls=":",  lw=0.8, alpha=0.7)
        #ax.axhline(self._v_min, color="#B90F22", ls="--", lw=0.6, alpha=0.7) # ToDo: Not needed
        #ax.axhline(self._v_max, color="#B90F22", ls="--", lw=0.6, alpha=0.7) # ToDo: Not needed
        t = np.asarray(self._t_all, dtype=float)
        for i, z in enumerate(self._zone_ids):
            c = _c(i + 1)
            v_lo = np.asarray(self._zone_v_min[z],  dtype=float)
            v_hi = np.asarray(self._zone_v_max[z],  dtype=float)
            v_mn = np.asarray(self._zone_v_mean[z], dtype=float)
            ax.fill_between(t, v_lo, v_hi, color=c, alpha=0.15)
            ax.plot(t, v_mn, color=c, lw=1.1, label=f"Z{z}")
        ax.set_ylabel(r"V / p.u.")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7,
                  ncol=min(len(self._zone_ids), 3), frameon=False)

    def _plot_padded_multi(
        self,
        ax: plt.Axes,
        t: np.ndarray,
        arrs_per_zone: Dict[int, List[np.ndarray]],
        *,
        drawstyle: str = "default",
        lw: float = 0.9,
        alpha: float = 0.85,
    ) -> None:
        """Plot multiple per-zone element series with NaN-padded arrays."""
        for i, z in enumerate(self._zone_ids):
            arrs = arrs_per_zone[z]
            if not arrs:
                continue
            n = max((a.size for a in arrs), default=0)
            if n == 0:
                continue
            c = _c(i + 1)
            series = np.full((len(arrs), n), np.nan)
            for r, a in enumerate(arrs):
                if a.size > 0:
                    series[r, :a.size] = a
            for k in range(n):
                kw = dict(color=c, lw=lw, alpha=alpha)
                if drawstyle != "default":
                    kw["drawstyle"] = drawstyle
                ax.plot(t, series[:, k], **kw)

    def _redraw_gen_q(self) -> None:
        ax = self._ax_qgen
        ax.clear()
        tile_title(ax, "TSO Generator Q Injections")
        t = np.asarray(self._t_all, dtype=float)
        self._plot_padded_multi(ax, t, self._zone_q_gen, lw=0.8)
        ax.set_ylabel(r"Q$_\mathrm{gen}$ / Mvar")
        ax.grid(True, alpha=0.3)
        ax.legend(handles=self._zone_legend_handles(),
                  loc="upper left", fontsize=7,
                  ncol=min(len(self._zone_ids), 3), frameon=False)

    def _redraw_reserves(self) -> None:
        ax = self._ax_reserve
        ax.clear()
        tile_title(ax, "TSO Reactive Reserve (SG solid / DER dashed)")
        t = np.asarray(self._t_all, dtype=float)

        def _mean_series(arrs: List[np.ndarray]) -> np.ndarray:
            # Per-step mean reserve across the zone's elements (NaN-safe;
            # empty / all-NaN steps -> NaN so the trace simply gaps there).
            out = np.full(len(arrs), np.nan)
            for i, a in enumerate(arrs):
                if a.size and np.any(np.isfinite(a)):
                    out[i] = np.nanmean(a)
            return out

        any_data = False
        for i, z in enumerate(self._zone_ids):
            c = _c(i + 1)
            sg = _mean_series(self._zone_gen_reserve[z])
            der = _mean_series(self._zone_der_reserve[z])
            if np.any(np.isfinite(sg)):
                ax.plot(t, sg, color=c, lw=1.1, ls="-")
                any_data = True
            if np.any(np.isfinite(der)):
                ax.plot(t, der, color=c, lw=1.0, ls="--", alpha=0.9)
                any_data = True
        ax.set_ylabel("reserve (norm.)")
        ax.set_ylim(0.0, 0.5)
        ax.grid(True, alpha=0.3)
        if any_data:
            # Zone colours + a solid/dashed proxy so SG vs DER is unambiguous.
            handles = self._zone_legend_handles()
            handles += [
                plt.Line2D([0], [0], color="0.3", lw=1.2, ls="-", label="SG"),
                plt.Line2D([0], [0], color="0.3", lw=1.0, ls="--", label="DER"),
            ]
            ax.legend(handles=handles, loc="upper left", fontsize=6,
                      ncol=min(len(self._zone_ids) + 2, 4), frameon=False)
        else:
            ax.text(0.5, 0.5, "no reserve data",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=8, style="italic")

    def _redraw_tie_q(self) -> None:
        ax = self._ax_tie
        ax.clear()
        tile_title(ax, "TSO Reactive Power Tie-Line Flows")
        if not self._tie_pairs:
            ax.text(0.5, 0.5, "no inter-zone tie lines",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=8, style="italic")
            ax.set_ylabel(r"Q$_\mathrm{tie}$ / Mvar")
            return
        ax.axhline(0.0, color="k", ls=":", lw=0.6, alpha=0.4)
        t = np.asarray(self._t_all, dtype=float)
        for k, (zi, zj) in enumerate(self._tie_pairs):
            vals = np.asarray(self._tie_q[(zi, zj)], dtype=float)
            ax.plot(t, vals, color=_c(3 + k), lw=1.0, label=f"Z{zi}$\\rightarrow$Z{zj}")
        ax.set_ylabel(r"Q$_\mathrm{tie}$ / Mvar")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7,
                  ncol=min(len(self._tie_pairs), 3), frameon=False)

    def _redraw_line_currents(self) -> None:
        ax = self._ax_iline
        ax.clear()
        tile_title(ax, "TSO Line Currents (Loading %)")
        ax.axhline(100.0, color="#B90F22", ls="--", lw=0.6, alpha=0.7)
        t = np.asarray(self._t_all, dtype=float)
        for i, z in enumerate(self._zone_ids):
            c = _c(i + 1)
            lo = np.asarray(self._zone_i_min[z],  dtype=float)
            hi = np.asarray(self._zone_i_max[z],  dtype=float)
            mn = np.asarray(self._zone_i_mean[z], dtype=float)
            ax.fill_between(t, lo, hi, color=c, alpha=0.12)
            ax.plot(t, mn, color=c, lw=1.0, label=f"Z{z}")
        ax.set_ylabel(r"I / %")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7,
                  ncol=min(len(self._zone_ids), 3), frameon=False)

    def _redraw_der_q(self) -> None:
        ax = self._ax_qder
        ax.clear()
        tile_title(ax, "TSO DER Q Infeed per Zone")
        t = np.asarray(self._t_tso, dtype=float)
        self._plot_padded_multi(ax, t, self._zone_q_der, drawstyle="default")
        ax.set_ylabel(r"Q$_\mathrm{DER}$ / Mvar")
        ax.grid(True, alpha=0.3)
        ax.legend(handles=self._zone_legend_handles(),
                  loc="upper left", fontsize=7,
                  ncol=min(len(self._zone_ids), 3), frameon=False)

    def _redraw_v_gen(self) -> None:
        ax = self._ax_vgen
        ax.clear()
        tile_title(ax, "TSO Generator AVR Setpoints")
        ax.axhline(self._v_set, color="k", ls=":", lw=0.8, alpha=0.6)
        t = np.asarray(self._t_tso, dtype=float)
        self._plot_padded_multi(ax, t, self._zone_v_gen, drawstyle="default")
        ax.set_ylabel(r"V$_\mathrm{gen,set}$ / p.u.")
        ax.grid(True, alpha=0.3)
        ax.legend(handles=self._zone_legend_handles(),
                  loc="upper left", fontsize=7,
                  ncol=min(len(self._zone_ids), 3), frameon=False)

    def _redraw_oltc(self) -> None:
        ax = self._ax_oltc
        ax.clear()
        tile_title(ax, "TSO Machine Transformer Taps")
        ax.set_ylabel("Tap")
        if sum(self._n_oltc.values()) == 0:
            ax.text(0.5, 0.5, "no machine-trafo OLTCs",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=8, style="italic")
            return
        t = np.asarray(self._t_tso, dtype=float)
        self._plot_padded_multi(ax, t, self._zone_oltc, drawstyle="steps-post")
        ax.grid(True, alpha=0.3)
        handles = [
            plt.Line2D([0], [0], color=_c(i + 1), lw=1.2, label=f"Z{z}")
            for i, z in enumerate(self._zone_ids) if self._n_oltc.get(z, 0) > 0
        ]
        if handles:
            ax.legend(handles=handles, loc="upper left",
                      fontsize=7, frameon=False)

    def _redraw_shunts(self) -> None:
        ax = self._ax_shunt
        ax.clear()
        tile_title(ax, "TSO Shunt States  (+ reactor/MSR  /  - capacitor/MSC)")
        ax.set_xlabel("Time / s" if self._sub_minute else "Time / min")
        ax.set_ylabel("Step  (+reac / -cap)")
        if sum(self._n_shunt.values()) == 0:
            ax.text(0.5, 0.5, "no shunts in network",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=8, style="italic")
            return
        t = np.asarray(self._t_tso, dtype=float)
        # Zero baseline separates inductive (MSR, +) from capacitive (MSC, -).
        ax.axhline(0.0, color="0.5", lw=0.6, alpha=0.6)
        self._plot_padded_multi(ax, t, self._zone_shunt, drawstyle="steps-post")
        ax.grid(True, alpha=0.3)
        handles = [
            plt.Line2D([0], [0], color=_c(i + 1), lw=1.2, label=f"Z{z}")
            for i, z in enumerate(self._zone_ids) if self._n_shunt.get(z, 0) > 0
        ]
        if handles:
            ax.legend(handles=handles, loc="upper left",
                      fontsize=7, frameon=False)
