"""
visualisation/plot_tracking.py
==============================
Live plotter for Figure 4 — TRACKING ERRORS & RESERVES.

Six tiles grouped into two colour-banded sections.  Unlike the
MULTI-TSO CONTROLLER / CASCADE-DSO / SYSTEM figures (which show the raw
measured/actuated quantities), this figure shows derived control-performance
KPIs: how well each control objective is tracked and how much reactive-power
reserve the continuous actuators retain.

    TRACKING ERRORS (orange)
        1. TS voltage tracking error — spatial RMS of (V − V_set) over each
           zone's observed EHV buses (one line per zone).
        2. TS voltage tracking error — system-wide RMS over all observed
           EHV buses (one aggregate line).
        3. TSO–DSO interface Q tracking error — per DSO, RMS over the DSO's
           interface transformers of (Q_actual − Q_set) (one line per DSO).
        4. Tie-line Q tracking error — RMS over all inter-zone tie groups of
           (Q_tie − Q_tie_set) (one aggregate line; reference defaults to 0).

    RESERVES (dark blue)
        5. Synchronous-generator reactive-power reserve r_Q(P)
           = min(Q_max(P) − Q, Q − Q_min(P)) / (Q_max(P) − Q_min(P))
           (one line per synchronous machine).
        6. TSO-connected DER reactive-power reserve, same definition
           (one line per TSO DER).

r_Q ranges from 0 (at a capability limit, no reserve) to 0.5 (mid-band,
maximum reserve); it can dip below 0 if the realised Q falls outside the
operating-point capability envelope.  NaN where the capability band has
zero width (e.g. a DER in the VDE dead zone below P/S_n = 0.1).

Activated like the other live plots, e.g. ``live_plot_tracking=True`` on the
:class:`configs.multi_tso_config.MultiTSOConfig`.
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


class TrackingLivePlotter:
    """Live figure 4 — TRACKING ERRORS & RESERVES (6 tiles, 2 sections)."""

    def __init__(
        self,
        zone_ids: Sequence[int],
        n_v_bus_per_zone: Dict[int, int],
        dso_ids: Sequence[str],
        dso_trafo_keys: Dict[str, Sequence[str]],
        tie_line_pairs: Sequence[Tuple[int, int]],
        *,
        tie_setpoints_mvar: Dict[Tuple[int, int], float] | None = None,
        sub_minute: bool = False,
        update_every: int = 1,
        slot_idx: int = 2,
        layout: str = "dual_screen",
        use_tex: bool = False,
    ) -> None:
        apply_serif_style(use_tex=use_tex)
        plt.ion()

        self._zone_ids = list(zone_ids)
        self._n_v_bus = {int(z): int(n) for z, n in n_v_bus_per_zone.items()}
        self._dso_ids = list(dso_ids)
        self._dso_trafo_keys = {d: list(k) for d, k in dso_trafo_keys.items()}
        self._tie_pairs = list(tie_line_pairs)
        self._tie_ref = dict(tie_setpoints_mvar or {})
        self._sub_minute = sub_minute
        self._update_every = max(1, int(update_every))
        self._call_count = 0

        # ── Accumulators (every update() call) ───────────────────────────
        self._t_all: List[float] = []

        self._zone_v_rms: Dict[int, List[float]] = {z: [] for z in self._zone_ids}
        self._agg_v_rms: List[float] = []
        self._dso_q_rms: Dict[str, List[float]] = {d: [] for d in self._dso_ids}
        self._tie_rms: List[float] = []
        self._gen_reserve: Dict[int, List[np.ndarray]] = {z: [] for z in self._zone_ids}
        self._der_reserve: Dict[int, List[np.ndarray]] = {z: [] for z in self._zone_ids}

        # ── Figure + GridSpec ────────────────────────────────────────────
        self._fig = plt.figure(figsize=(6.2, 10.0))
        try:
            self._fig.canvas.manager.set_window_title("Tracking & Reserves")
        except Exception:
            pass

        self._fig.subplots_adjust(
            top=1.0 - TITLE_BAR_HEIGHT_FRAC - 0.005,
            bottom=0.045, left=0.10, right=0.985,
            hspace=0.60,
        )
        draw_figure_header(self._fig, "Tracking & Reserves")

        plot_h = 1.0
        band_h = 0.18
        heights = [band_h] + [plot_h] * 4 + [band_h] + [plot_h] * 2
        gs = GridSpec(
            len(heights), 1, figure=self._fig,
            height_ratios=heights, hspace=0.60,
        )

        # Row 0: TRACKING ERRORS band
        self._ax_band_err = self._fig.add_subplot(gs[0, 0])
        fill_section_band(self._ax_band_err, "Tracking Errors", COLOUR_MEAS_BAND)

        self._ax_v_zone = self._fig.add_subplot(gs[1, 0])
        self._ax_v_agg  = self._fig.add_subplot(gs[2, 0], sharex=self._ax_v_zone)
        self._ax_q_dso  = self._fig.add_subplot(gs[3, 0], sharex=self._ax_v_zone)
        self._ax_q_tie  = self._fig.add_subplot(gs[4, 0], sharex=self._ax_v_zone)

        # Row 5: RESERVES band
        self._ax_band_res = self._fig.add_subplot(gs[5, 0])
        fill_section_band(self._ax_band_res, "Reserves", COLOUR_ACT_BAND)

        self._ax_res_gen = self._fig.add_subplot(gs[6, 0], sharex=self._ax_v_zone)
        self._ax_res_der = self._fig.add_subplot(gs[7, 0], sharex=self._ax_v_zone)

        self._plot_axes = [
            self._ax_v_zone, self._ax_v_agg, self._ax_q_dso, self._ax_q_tie,
            self._ax_res_gen, self._ax_res_der,
        ]

        tile_title(self._ax_v_zone, "TS Voltage Tracking Error per Zone")
        tile_title(self._ax_v_agg,  "TS Voltage Tracking Error (System RMS)")
        tile_title(self._ax_q_dso,  "TSO-DSO Interface Q Tracking Error per DSO")
        tile_title(self._ax_q_tie,  "Tie-Line Q Tracking Error (System RMS)")
        tile_title(self._ax_res_gen, "Synchronous Generator Q Reserve")
        tile_title(self._ax_res_der, "TSO DER Q Reserve")

        for ax in self._plot_axes:
            ax.tick_params(axis="both", labelsize=8)
        for ax in self._plot_axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        self._ax_res_der.set_xlabel("Time / s" if sub_minute else "Time / min")

        position_figure_in_slot(self._fig, slot_idx, layout=layout, n_slots=3)
        plt.pause(0.01)
        raise_figure_to_front(self._fig)

    # ─── update ───────────────────────────────────────────────────────────

    def update(self, rec: "MultiTSOIterationRecord") -> None:
        """Append this record's KPIs and redraw if cadence matches."""
        t_unit = rec.time_s if self._sub_minute else rec.time_s / 60.0
        self._t_all.append(t_unit)

        # Tile 1: per-zone voltage RMS error.
        num, den = 0.0, 0
        for z in self._zone_ids:
            r = float(rec.zone_v_rms_err_pu.get(z, float("nan")))
            self._zone_v_rms[z].append(r)
            n = self._n_v_bus.get(z, 0)
            if np.isfinite(r) and n > 0:
                num += n * r * r
                den += n
        # Tile 2: system-wide voltage RMS error (bus-count-weighted aggregate).
        self._agg_v_rms.append(float(np.sqrt(num / den)) if den > 0 else float("nan"))

        # Tile 3: per-DSO interface-Q RMS tracking error over the DSO's trafos.
        for dso in self._dso_ids:
            errs: List[float] = []
            for key in self._dso_trafo_keys.get(dso, []):
                q_set = rec.dso_trafo_q_set_mvar.get(key)
                q_act = rec.dso_trafo_q_actual_mvar.get(key)
                if q_set is not None and q_act is not None:
                    errs.append(float(q_act) - float(q_set))
            self._dso_q_rms[dso].append(
                float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
            )

        # Tile 4: tie-line Q RMS tracking error over all inter-zone tie groups.
        tie_errs: List[float] = []
        for pair in self._tie_pairs:
            q = rec.zone_tie_q_mvar.get(pair)
            if q is not None:
                tie_errs.append(float(q) - float(self._tie_ref.get(pair, 0.0)))
        self._tie_rms.append(
            float(np.sqrt(np.mean(np.square(tie_errs)))) if tie_errs else float("nan")
        )

        # Tiles 5 & 6: per-element reserve arrays (NaN-padded downstream).
        for z in self._zone_ids:
            self._gen_reserve[z].append(
                np.asarray(rec.gen_q_reserve.get(z, []), dtype=float)
            )
            self._der_reserve[z].append(
                np.asarray(rec.tso_der_q_reserve.get(z, []), dtype=float)
            )

        self._call_count += 1
        if self._call_count % self._update_every == 0:
            self._redraw()

    # ─── redraw ───────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        self._redraw_v_zone()
        self._redraw_v_agg()
        self._redraw_q_dso()
        self._redraw_q_tie()
        self._redraw_reserve(self._ax_res_gen, self._gen_reserve,
                             "Synchronous Generator Q Reserve")
        self._redraw_reserve(self._ax_res_der, self._der_reserve,
                             "TSO DER Q Reserve")
        for ax in self._plot_axes:
            apply_x_fmt(ax, sub_minute=self._sub_minute)
        # Shared-x cleanup (mirrors TSOControllerLivePlotter._redraw).
        for ax in self._plot_axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlabel("")
        self._plot_axes[-1].set_xlabel(
            "Time / s" if self._sub_minute else "Time / min"
        )
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    # ─── per-tile redraws ─────────────────────────────────────────────────

    def _redraw_v_zone(self) -> None:
        ax = self._ax_v_zone
        ax.clear()
        tile_title(ax, "TS Voltage Tracking Error per Zone")
        t = np.asarray(self._t_all, dtype=float)
        for i, z in enumerate(self._zone_ids):
            ax.plot(t, np.asarray(self._zone_v_rms[z], dtype=float),
                    color=_c(i + 1), lw=1.1, label=f"Z{z}")
        ax.set_ylabel(r"V RMS err / p.u.")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7,
                  ncol=min(len(self._zone_ids), 3), frameon=False)

    def _redraw_v_agg(self) -> None:
        ax = self._ax_v_agg
        ax.clear()
        tile_title(ax, "TS Voltage Tracking Error (System RMS)")
        t = np.asarray(self._t_all, dtype=float)
        ax.plot(t, np.asarray(self._agg_v_rms, dtype=float),
                color=_c(8), lw=1.3, label="all zones")
        ax.set_ylabel(r"V RMS err / p.u.")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7, frameon=False)

    def _redraw_q_dso(self) -> None:
        ax = self._ax_q_dso
        ax.clear()
        tile_title(ax, "TSO-DSO Interface Q Tracking Error per DSO")
        if not self._dso_ids:
            ax.text(0.5, 0.5, "no DSO interfaces",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=8, style="italic")
            ax.set_ylabel(r"Q RMS err / Mvar")
            return
        t = np.asarray(self._t_all, dtype=float)
        for i, dso in enumerate(self._dso_ids):
            ax.plot(t, np.asarray(self._dso_q_rms[dso], dtype=float),
                    color=_c(i + 1), lw=1.1, label=dso)
        ax.set_ylabel(r"Q RMS err / Mvar")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7,
                  ncol=min(len(self._dso_ids), 3), frameon=False)

    def _redraw_q_tie(self) -> None:
        ax = self._ax_q_tie
        ax.clear()
        tile_title(ax, "Tie-Line Q Tracking Error (System RMS)")
        if not self._tie_pairs:
            ax.text(0.5, 0.5, "no inter-zone tie lines",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=8, style="italic")
            ax.set_ylabel(r"Q RMS err / Mvar")
            return
        t = np.asarray(self._t_all, dtype=float)
        ax.plot(t, np.asarray(self._tie_rms, dtype=float),
                color=_c(8), lw=1.3, label="all tie lines")
        ax.set_ylabel(r"Q RMS err / Mvar")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7, frameon=False)

    def _redraw_reserve(
        self,
        ax: plt.Axes,
        arrs_per_zone: Dict[int, List[np.ndarray]],
        title: str,
    ) -> None:
        ax.clear()
        tile_title(ax, title)
        ax.axhline(0.0, color="#B90F22", ls="--", lw=0.6, alpha=0.6)
        ax.axhline(0.5, color="k", ls=":", lw=0.6, alpha=0.4)
        t = np.asarray(self._t_all, dtype=float)
        color_idx = 0
        any_line = False
        for z in self._zone_ids:
            arrs = arrs_per_zone[z]
            n = max((a.size for a in arrs), default=0)
            if n == 0:
                continue
            series = np.full((len(arrs), n), np.nan)
            for r, a in enumerate(arrs):
                if a.size > 0:
                    series[r, :a.size] = a
            for k in range(n):
                ax.plot(t, series[:, k], color=_c(color_idx), lw=0.9, alpha=0.85)
                color_idx += 1
                any_line = True
        ax.set_ylabel(r"$r_Q$ / -")
        ax.grid(True, alpha=0.3)
        if not any_line:
            ax.text(0.5, 0.5, "no elements",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=8, style="italic")
