"""
visualisation/plot_sbx.py
=========================
Live plotter for Figure 6 — SBX MECHANISM (Scheduled Boundary Exchange).

Per corridor (one tile each):

    * measured corridor flow q_meas (per plant step, computed by the
      runner AT each tie's reference-end terminal — q_from or q_to per
      line orientation; positive = export from the reference end A.
      ``rec.zone_tie_q_mvar`` is deliberately NOT used: it negates the
      from-end flow for lines oriented from the higher zone, which
      misstates heavily charged ties — e.g. IEEE 39 line 14 with
      ~107 Mvar of charging — relative to the SBX schedule convention),
    * the schedule staircase q_sched and the standard q_std (per cycle,
      from the scheduler's ``CorridorCycleRecord``s),
    * the NO-REMUNERATION BAND q_sched ± q_band (tier 1: deviations
      inside the shading are free and only enter the unmonetised
      netting ledger),
    * deal markers — requests that were DELIVERED: ▼ unilateral paid,
      ◆ mutual unpaid, △ unwind steps, ✕ scarcity (a request that
      found no counterpart),
    * need-flag strips at the tile bottom (end A / end B of the
      corridor — a set flag is an outstanding REQUEST precondition).

Below the corridor tiles: the signed surplus staircase per corridor
(the running deal balance q_sched − q_std) and the cumulative
settlement payments per area (tier 2 + tier 3).

The figure starts at the contract-freeze tick (``sbx_warmup_s``) —
before it there are no corridors, no schedule and no band, so there is
nothing SBX-specific to draw.  Enabled via ``config.live_plot_sbx``
(requires ``coordination_mode="sbx"``); the runner passes the adapter
handle and the per-step reference-end corridor flows alongside each
iteration record.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-08 (SBX Phase 7 follow-up)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from visualisation.style import (
    COLOUR_MEAS_BAND,
    TITLE_BAR_HEIGHT_FRAC,
    apply_serif_style,
    draw_figure_header,
    fill_section_band,
    position_figure_in_slot,
    raise_figure_to_front,
    tile_title,
)

if TYPE_CHECKING:
    from experiments.helpers import MultiTSOIterationRecord
    from sbx_h.adapter import SBXRunnerAdapter

_C_MEAS = "#cc6677"
_C_SCHED = "#4477aa"
_C_STD = "#888888"
_C_BAND = "#4477aa"
_C_DEAL_UNI = "#117733"
_C_DEAL_MUT = "#aa4499"
_C_UNWIND = "#ddaa33"
_C_SCARCITY = "#cc3311"
_ZONE_COLOURS = {1: "#4477aa", 2: "#ee6677", 3: "#228833"}


class SBXMechanismLivePlotter:
    """Live figure 6 — SBX MECHANISM (corridor tiles + surplus + payments).

    Constructed before the loop; ``update(rec, adapter, corridor_q)``
    is called once per plant step and redraws every ``update_every``
    calls.  ``adapter``/``corridor_q`` are ``None`` until the
    contract-freeze tick — the figure builds itself at the first update
    after the freeze (corridor keys from ``corridor_q``).
    """

    def __init__(
        self,
        *,
        update_every: int = 9,
        slot_idx: int = 3,
        layout: str = "dual_screen",
        use_tex: bool = False,
    ) -> None:
        apply_serif_style(use_tex=use_tex)
        plt.ion()
        self._update_every = max(1, int(update_every))
        self._call_count = 0
        self._slot_idx = slot_idx
        self._layout = layout

        self._keys: List[Tuple[int, int]] = []
        self._t: List[float] = []
        self._q: Dict[Tuple[int, int], List[float]] = {}
        #: Contract-freeze time [min], latched at the first update that
        #: sees a constructed adapter (the freeze tick's plant step) —
        #: the adapter rebases its cycle counter there.
        self._freeze_t_min: Optional[float] = None
        self._axes: Dict[Tuple[int, int], plt.Axes] = {}
        self._ax_surplus: Optional[plt.Axes] = None
        self._ax_pay: Optional[plt.Axes] = None
        self._built = False

        # The window opens IMMEDIATELY (like the other live figures) with
        # a placeholder; the axes are built at the contract-freeze tick,
        # when the corridors exist.
        self._fig = plt.figure(figsize=(7.0, 9.0))
        try:
            self._fig.canvas.manager.set_window_title("SBX Mechanism")
        except Exception:
            pass
        draw_figure_header(self._fig, "SBX — Scheduled Boundary Exchange")
        self._placeholder = self._fig.text(
            0.5, 0.5,
            "SBX contracts not yet frozen —\n"
            "corridor schedules appear at sbx_warmup_s.",
            ha="center", va="center", fontsize=11, color="0.35",
        )
        position_figure_in_slot(self._fig, self._slot_idx,
                                layout=self._layout, n_slots=4)
        plt.pause(0.01)
        raise_figure_to_front(self._fig)

    # ── lazy axes construction (needs the corridor keys) ──────────────
    def _build(self, keys: List[Tuple[int, int]]) -> None:
        self._keys = list(keys)
        for key in self._keys:
            self._q[key] = []
        n = len(self._keys)
        try:
            self._placeholder.remove()
        except Exception:
            pass
        self._fig.set_size_inches(7.0, 2.4 * n + 4.6, forward=True)
        self._fig.subplots_adjust(
            top=1.0 - TITLE_BAR_HEIGHT_FRAC - 0.005,
            bottom=0.05, left=0.115, right=0.985, hspace=0.62,
        )
        band_h, plot_h = 0.18, 1.0
        gs = GridSpec(n + 3, 1, figure=self._fig,
                      height_ratios=[band_h] + [plot_h] * (n + 2),
                      hspace=0.62)
        ax_band = self._fig.add_subplot(gs[0, 0])
        fill_section_band(ax_band, "Corridor Schedules and Deliveries",
                          COLOUR_MEAS_BAND)
        prev = None
        for k, key in enumerate(self._keys):
            ax = self._fig.add_subplot(gs[1 + k, 0], sharex=prev)
            prev = prev or ax
            self._axes[key] = ax
        self._ax_surplus = self._fig.add_subplot(gs[1 + n, 0], sharex=prev)
        self._ax_pay = self._fig.add_subplot(gs[2 + n, 0], sharex=prev)
        for ax in list(self._axes.values()) + [self._ax_surplus,
                                               self._ax_pay]:
            ax.tick_params(axis="both", labelsize=8)
        self._ax_pay.set_xlabel("Time / min")
        position_figure_in_slot(self._fig, self._slot_idx,
                                layout=self._layout, n_slots=4)
        plt.pause(0.01)
        raise_figure_to_front(self._fig)
        self._built = True

    # ── update ─────────────────────────────────────────────────────────
    def update(
        self,
        rec: "MultiTSOIterationRecord",
        adapter: Optional["SBXRunnerAdapter"],
        corridor_q_mvar: Optional[Dict[Tuple[int, int], float]],
    ) -> None:
        """One plant step.  ``corridor_q_mvar`` = reference-end corridor
        flows computed by the runner (None before the contract freeze)."""
        if adapter is None or corridor_q_mvar is None:
            # Keep the placeholder window responsive during the warmup
            # even when Figure 6 is the only live figure.
            self._call_count += 1
            if self._call_count % self._update_every == 0:
                try:
                    plt.pause(0.001)
                except Exception:
                    pass
            return
        if not self._built:
            self._build(sorted(corridor_q_mvar.keys()))
        if self._freeze_t_min is None:
            self._freeze_t_min = rec.time_s / 60.0
        self._t.append(rec.time_s / 60.0)
        for key in self._keys:
            self._q[key].append(float(corridor_q_mvar.get(key, np.nan)))
        self._call_count += 1
        if self._call_count % self._update_every:
            return
        self._redraw(adapter)

    def _cycle_time_min(self, adapter, cycle: int) -> float:
        """Boundary of cycle c in minutes (adapter rebases at the freeze)."""
        if self._freeze_t_min is None:
            return float("nan")
        return self._freeze_t_min + cycle * adapter.config.t_cycle_min

    def _redraw(self, adapter) -> None:
        t = np.asarray(self._t)
        for key in self._keys:
            ax = self._axes[key]
            ax.cla()
            ax.plot(t, self._q[key], color=_C_MEAS, lw=0.9,
                    label="q_meas")
            if adapter is not None:
                self._draw_schedule(ax, adapter, key)
            tile_title(ax, f"Corridor ({key[0]},{key[1]})  "
                           f"[+ = export from zone {key[0]}]")
            ax.set_ylabel("Q / Mvar", fontsize=8)
            ax.grid(alpha=0.3, lw=0.4)
            ax.legend(loc="upper left", fontsize=6.5, ncol=4,
                      frameon=False)
        self._draw_surplus_and_payments(adapter)
        try:
            self._fig.canvas.draw_idle()
            plt.pause(0.001)
        except Exception:
            pass

    def _draw_schedule(self, ax, adapter, key: Tuple[int, int]) -> None:
        sched = adapter.scheduler
        recs = sched.records.get(key, [])
        if not recs:
            return
        # Staircase over [boundary(c-1), boundary(c)] — the cycle whose
        # schedule was ACTIVE (records store the elapsed-cycle values at
        # each boundary; the freeze tick starts cycle 1).  The band is
        # per cycle (v3: hourly band schedules).
        t_edges = [self._cycle_time_min(adapter, recs[0].cycle - 1)]
        q_std, q_sched, band = [], [], []
        for r in recs:
            t_edges.append(self._cycle_time_min(adapter, r.cycle))
            q_std.append(r.q_std_mvar)
            q_sched.append(r.q_sched_mvar)
            band.append(r.q_band_mvar)
        t_edges = np.asarray(t_edges)
        q_std = np.asarray(q_std)
        q_sched = np.asarray(q_sched)
        band = np.asarray(band)
        tt = np.repeat(t_edges, 2)[1:-1]
        ax.fill_between(tt, np.repeat(q_sched - band, 2),
                        np.repeat(q_sched + band, 2),
                        color=_C_BAND, alpha=0.15, lw=0,
                        label=f"tier-1 band ±{band[-1]:.0f}")
        ax.plot(tt, np.repeat(q_std, 2), color=_C_STD, ls="--", lw=0.9,
                label="q_std")
        ax.plot(tt, np.repeat(q_sched, 2), color=_C_SCHED, lw=1.2,
                label="q_sched")
        # Deal / unwind / scarcity markers at the boundary that decided.
        seen = set()
        for r in recs:
            tb = self._cycle_time_min(adapter, r.cycle)
            if r.deal.dq_deal_mvar != 0.0:
                mut = r.deal.kind == "mutual"
                lab = "mutual deal" if mut else "unilateral deal"
                ax.scatter([tb], [r.q_sched_mvar],
                           marker="D" if mut else "v", s=26, zorder=5,
                           color=_C_DEAL_MUT if mut else _C_DEAL_UNI,
                           label=lab if lab not in seen else None)
                seen.add(lab)
            elif r.unwound_mvar != 0.0:
                ax.scatter([tb], [r.q_sched_mvar], marker="^", s=24,
                           zorder=5, color=_C_UNWIND,
                           label="unwind" if "unwind" not in seen
                           else None)
                seen.add("unwind")
            elif r.deal.kind == "scarcity":
                ax.scatter([tb], [r.q_sched_mvar], marker="x", s=26,
                           zorder=5, color=_C_SCARCITY,
                           label="scarcity" if "scarcity" not in seen
                           else None)
                seen.add("scarcity")
        # Need-flag strips along the tile bottom (request precondition).
        y0, y1 = ax.get_ylim()
        h = 0.04 * (y1 - y0)
        for r in recs:
            ta = self._cycle_time_min(adapter, r.cycle - 1)
            tb = self._cycle_time_min(adapter, r.cycle)
            if r.need_a:
                ax.fill_between([ta, tb], y0, y0 + h,
                                color=_ZONE_COLOURS.get(key[0], "0.4"),
                                alpha=0.85, lw=0)
            if r.need_b:
                ax.fill_between([ta, tb], y0 + h, y0 + 2 * h,
                                color=_ZONE_COLOURS.get(key[1], "0.4"),
                                alpha=0.85, lw=0)

    def _draw_surplus_and_payments(self, adapter) -> None:
        ax_s, ax_p = self._ax_surplus, self._ax_pay
        ax_s.cla()
        ax_p.cla()
        if adapter is not None:
            sched = adapter.scheduler
            for key in self._keys:
                recs = sched.records.get(key, [])
                if not recs:
                    continue
                tt = [self._cycle_time_min(adapter, recs[0].cycle - 1)]
                ss = [0.0]
                for r in recs:
                    tt.append(self._cycle_time_min(adapter, r.cycle))
                    ss.append(r.surplus_mvar)
                ax_s.step(tt, ss, where="post", lw=1.2,
                          label=f"({key[0]},{key[1]})")
            # Cumulative per-area payments from the per-cycle settlements.
            cum: Dict[int, float] = {}
            series: Dict[int, List[float]] = {}
            times: List[float] = []
            all_s = sorted(
                (s for sl in sched.settlements.values() for s in sl),
                key=lambda s: s.cycle,
            )
            for s in all_s:
                for z, x in s.payments_eur.items():
                    cum[z] = cum.get(z, 0.0) + x
                times.append(self._cycle_time_min(adapter, s.cycle))
                for z in sched.area_ids:
                    series.setdefault(z, []).append(cum.get(z, 0.0))
            for z, ys in sorted(series.items()):
                ax_p.plot(times, ys, lw=1.2,
                          color=_ZONE_COLOURS.get(z, "0.3"),
                          label=f"zone {z}")
        tile_title(ax_s, "Surplus q_sched - q_std (deal balance)")
        ax_s.set_ylabel("Mvar", fontsize=8)
        tile_title(ax_p, "Cumulative settlement payments")
        ax_p.set_ylabel("EUR", fontsize=8)
        ax_p.set_xlabel("Time / min")
        for ax in (ax_s, ax_p):
            ax.grid(alpha=0.3, lw=0.4)
            ax.legend(loc="upper left", fontsize=6.5, ncol=3,
                      frameon=False)

    # ── finalise ───────────────────────────────────────────────────────
    def save(self, path) -> None:
        """Force a final redraw-independent save of the figure."""
        if self._fig is not None:
            self._fig.savefig(path, dpi=160)
