"""
visualisation/plot_tie_coordination.py
=======================================
Live plotter for Figure 5 — HORIZONTAL TSO-TSO COORDINATION (two-loop ΔV_ref).

Per inter-zone tie line:

    1. Agreed vs realised boundary-voltage difference  (ΔV_ref vs V_i−V_j)
    2. Tie-line reactive flow Q_tie    — with the ±band soft cap shaded
    3. Boundary voltages V_i / V_j
    4. Coordination marginal m_e        — the consistency residual ("dual-like")

Series are keyed by tie-line id and read from the ``rec.tie_*`` fields the
runner populates when ``config.enable_tie_coordination`` is set.  Enabled via
``config.live_plot_tie_coordination``.

NOTE: tile_title / section bands upper-case their text, which corrupts ``$…$``
mathtext (``\\mathrm`` -> ``\\MATHRM``), so titles are plain text and mathtext
lives only in the (non-upper-cased) y-labels / legend handles.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from visualisation.style import (
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


class TieCoordinationLivePlotter:
    """Live figure 5 — HORIZONTAL TSO-TSO COORDINATION (4 tiles)."""

    def __init__(
        self,
        tie_ids: Sequence[int],
        tie_labels: Mapping[int, str],
        *,
        q_band_mvar: float = 30.0,
        deadband_v_pu: float = 0.002,
        sub_minute: bool = False,
        update_every: int = 1,
        slot_idx: int = 3,
        layout: str = "dual_screen",
        use_tex: bool = False,
    ) -> None:
        apply_serif_style(use_tex=use_tex)
        plt.ion()

        self._tie_ids = list(tie_ids)
        self._labels = {int(k): str(v) for k, v in tie_labels.items()}
        self._q_band = float(q_band_mvar)
        self._deadband = float(deadband_v_pu)
        self._sub_minute = sub_minute
        self._update_every = max(1, int(update_every))
        self._call_count = 0

        self._t_all: List[float] = []
        self._dvref:  Dict[int, List[float]] = {e: [] for e in self._tie_ids}
        self._dvreal: Dict[int, List[float]] = {e: [] for e in self._tie_ids}
        self._q:      Dict[int, List[float]] = {e: [] for e in self._tie_ids}
        self._vi:     Dict[int, List[float]] = {e: [] for e in self._tie_ids}
        self._vj:     Dict[int, List[float]] = {e: [] for e in self._tie_ids}
        self._gcomb:  Dict[int, List[float]] = {e: [] for e in self._tie_ids}

        self._fig = plt.figure(figsize=(6.2, 10.0))
        try:
            self._fig.canvas.manager.set_window_title("TSO-TSO Coordination")
        except Exception:
            pass
        self._fig.subplots_adjust(
            top=1.0 - TITLE_BAR_HEIGHT_FRAC - 0.005,
            bottom=0.045, left=0.12, right=0.985, hspace=0.60,
        )
        draw_figure_header(self._fig, "Horizontal TSO–TSO Coordination")

        band_h, plot_h = 0.18, 1.0
        gs = GridSpec(5, 1, figure=self._fig,
                      height_ratios=[band_h] + [plot_h] * 4, hspace=0.60)
        self._ax_band = self._fig.add_subplot(gs[0, 0])
        fill_section_band(self._ax_band, "Coordination Variables", COLOUR_MEAS_BAND)
        self._ax_dv   = self._fig.add_subplot(gs[1, 0])
        self._ax_q    = self._fig.add_subplot(gs[2, 0], sharex=self._ax_dv)
        self._ax_v    = self._fig.add_subplot(gs[3, 0], sharex=self._ax_dv)
        self._ax_marg = self._fig.add_subplot(gs[4, 0], sharex=self._ax_dv)
        self._plot_axes = [self._ax_dv, self._ax_q, self._ax_v, self._ax_marg]

        for ax in self._plot_axes:
            ax.tick_params(axis="both", labelsize=8)
        for ax in self._plot_axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        self._ax_marg.set_xlabel("Time / s" if sub_minute else "Time / min")

        position_figure_in_slot(self._fig, slot_idx, layout=layout, n_slots=4)
        plt.pause(0.01)
        raise_figure_to_front(self._fig)

    # ─── update ─────────────────────────────────────────────────────────

    def update(self, rec: "MultiTSOIterationRecord") -> None:
        """Append coordination observables (skip records without tie data so
        every trace shares a gap-free TSO-step time axis)."""
        if not rec.tie_dvref:
            return
        t_unit = rec.time_s if self._sub_minute else rec.time_s / 60.0
        self._t_all.append(t_unit)
        for e in self._tie_ids:
            self._dvref[e].append(rec.tie_dvref.get(e, float("nan")))
            self._dvreal[e].append(rec.tie_dv_realized.get(e, float("nan")))
            self._q[e].append(rec.tie_q_mvar.get(e, float("nan")))
            self._vi[e].append(rec.tie_v_i.get(e, float("nan")))
            self._vj[e].append(rec.tie_v_j.get(e, float("nan")))
            self._gcomb[e].append(rec.tie_grad_combined.get(e, float("nan")))
        self._call_count += 1
        if self._call_count % self._update_every == 0:
            self._redraw()

    # ─── redraw ─────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        self._redraw_dv()
        self._redraw_q()
        self._redraw_voltages()
        self._redraw_marginal()
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

    def _tie_legend_handles(self) -> List[plt.Line2D]:
        return [
            plt.Line2D([0], [0], color=_c(k + 1), lw=1.2,
                       label=self._labels.get(e, f"L{e}"))
            for k, e in enumerate(self._tie_ids)
        ]

    def _redraw_dv(self) -> None:
        ax = self._ax_dv
        ax.clear()
        tile_title(ax, "Agreed vs Realised Difference  (dv_ref solid / dv_real dashed)")
        ax.axhline(0.0, color="k", ls=":", lw=0.8, alpha=0.6)
        if self._deadband > 0:
            ax.axhspan(-self._deadband, self._deadband, color="0.5", alpha=0.08)
        t = np.asarray(self._t_all, dtype=float)
        for k, e in enumerate(self._tie_ids):
            c = _c(k + 1)
            ax.plot(t, np.asarray(self._dvref[e], dtype=float), color=c, lw=1.3, ls="-")
            ax.plot(t, np.asarray(self._dvreal[e], dtype=float), color=c, lw=1.0, ls="--")
        ax.set_ylabel(r"$\Delta V$ / p.u.")
        ax.grid(True, alpha=0.3)
        handles = self._tie_legend_handles()
        handles += [
            plt.Line2D([0], [0], color="0.3", lw=1.3, ls="-",  label=r"$\Delta V_\mathrm{ref}$"),
            plt.Line2D([0], [0], color="0.3", lw=1.0, ls="--", label="realised"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=6,
                  ncol=min(len(self._tie_ids) + 2, 4), frameon=False)

    def _redraw_q(self) -> None:
        ax = self._ax_q
        ax.clear()
        tile_title(ax, "Tie-Line Reactive Flow  (± band soft cap)")
        if self._q_band > 0:
            ax.axhspan(-self._q_band, self._q_band, color="0.5", alpha=0.10)
            ax.axhline(+self._q_band, color="0.4", ls="--", lw=0.6, alpha=0.6)
            ax.axhline(-self._q_band, color="0.4", ls="--", lw=0.6, alpha=0.6)
        ax.axhline(0.0, color="k", ls=":", lw=0.6, alpha=0.4)
        t = np.asarray(self._t_all, dtype=float)
        for k, e in enumerate(self._tie_ids):
            ax.plot(t, np.asarray(self._q[e], dtype=float), color=_c(k + 1), lw=1.1)
        ax.set_ylabel(r"$Q_\mathrm{tie}$ / Mvar")
        ax.grid(True, alpha=0.3)

    def _redraw_voltages(self) -> None:
        ax = self._ax_v
        ax.clear()
        tile_title(ax, "Boundary Voltages  V_i solid / V_j dashed")
        t = np.asarray(self._t_all, dtype=float)
        for k, e in enumerate(self._tie_ids):
            c = _c(k + 1)
            ax.plot(t, np.asarray(self._vi[e], dtype=float), color=c, lw=1.1, ls="-")
            ax.plot(t, np.asarray(self._vj[e], dtype=float), color=c, lw=1.0, ls="--")
        ax.set_ylabel(r"V / p.u.")
        ax.grid(True, alpha=0.3)
        ax.legend(handles=self._tie_legend_handles(), loc="upper left",
                  fontsize=7, ncol=min(len(self._tie_ids), 3), frameon=False)

    def _redraw_marginal(self) -> None:
        ax = self._ax_marg
        ax.clear()
        tile_title(ax, "Combined Boundary Gradient  G  (joint descent direction)")
        ax.axhline(0.0, color="k", ls=":", lw=0.6, alpha=0.4)
        t = np.asarray(self._t_all, dtype=float)
        for k, e in enumerate(self._tie_ids):
            ax.plot(t, np.asarray(self._gcomb[e], dtype=float), color=_c(k + 1), lw=1.0)
        ax.set_ylabel(r"$G$")
        ax.grid(True, alpha=0.3)
