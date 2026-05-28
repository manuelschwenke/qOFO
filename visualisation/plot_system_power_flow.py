"""
visualisation/plot_system_power_flow.py
=======================================
Live plotter for Figure 3 — SYSTEM POWER FLOW.

Eight measurement tiles (a single orange MEASUREMENTS band at the top):

    1. TSO-DSO Active Power Flow  (primary/HV side, per interface trafo)
    2. TSO-DSO Reactive Power Flow (primary/HV side, per interface trafo)
    3. Synchronous Generator Active Power (one line per gen, with P-limit band)
    4. Synchronous Generator Reactive Power (one line per gen, with Q-limit band)
    5. System P balance: Σ TSO DER P · Σ TSO Gen P · Σ TSO Load P · Σ TSO-DSO P out
    6. System Q balance: Σ TSO DER Q · Σ TSO Gen Q · Σ TSO Load Q · Σ TSO-DSO Q out
    7. DSO-side P balance: Σ DSO DER P · Σ DSO Load P
    8. DSO-side Q balance: Σ DSO DER Q · Σ DSO Load Q
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Sequence

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


def _fill_empty(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message,
            transform=ax.transAxes, ha="center", va="center",
            color="gray", fontsize=8, style="italic")


class SystemPowerFlowLivePlotter:
    """Live figure 3 — SYSTEM POWER FLOW."""

    def __init__(
        self,
        zone_ids: Sequence[int],
        dso_ids: Sequence[str],
        interface_trafo_ids: Sequence[str],
        zone_gen_indices: Dict[int, List[int]],
        gen_limits_static: Dict[int, Dict[str, float]],
        *,
        sub_minute: bool = False,
        update_every: int = 1,
        slot_idx: int = 2,
        layout: str = "thirds",
        use_tex: bool = False,
    ) -> None:
        apply_serif_style(use_tex=use_tex)
        plt.ion()

        self._zone_ids = list(zone_ids)
        self._dso_ids = list(dso_ids)
        self._trafo_ids = list(interface_trafo_ids)
        self._zone_gen = {z: list(zone_gen_indices.get(z, [])) for z in self._zone_ids}
        self._gen_lim = dict(gen_limits_static)
        self._sub_minute = sub_minute
        self._update_every = max(1, int(update_every))
        self._call_count = 0

        # Map each gen index to its zone (for colouring)
        self._gen_zone: Dict[int, int] = {}
        for z, gens in self._zone_gen.items():
            for g in gens:
                self._gen_zone[g] = z
        self._gen_ids_flat: List[int] = sorted(self._gen_zone.keys())

        # ── Accumulators ────────────────────────────────────────────────
        self._t_all: List[float] = []

        # Per interface trafo (discovered lazily if names differ from init)
        self._trafo_p: Dict[str, List[float]] = {k: [] for k in self._trafo_ids}
        self._trafo_q: Dict[str, List[float]] = {k: [] for k in self._trafo_ids}

        # Per generator P/Q
        self._gen_p: Dict[int, List[float]] = {g: [] for g in self._gen_ids_flat}
        self._gen_q: Dict[int, List[float]] = {g: [] for g in self._gen_ids_flat}

        # System P/Q balance (aggregated over all TSO zones)
        self._sum_tso_der_p:   List[float] = []
        self._sum_tso_der_q:   List[float] = []
        self._sum_tso_gen_p:   List[float] = []
        self._sum_tso_gen_q:   List[float] = []
        self._sum_tso_load_p:  List[float] = []
        self._sum_tso_load_q:  List[float] = []
        self._sum_tso_dso_p_out: List[float] = []
        self._sum_tso_dso_q_out: List[float] = []

        # DSO-side balance (aggregated over all HV groups)
        self._sum_dso_der_p:  List[float] = []
        self._sum_dso_der_q:  List[float] = []
        self._sum_dso_load_p: List[float] = []
        self._sum_dso_load_q: List[float] = []

        # ── Figure + GridSpec ───────────────────────────────────────────
        self._fig = plt.figure(figsize=(6.2, 10.0))
        try:
            self._fig.canvas.manager.set_window_title("System Power Flow")
        except Exception:
            pass
        self._fig.subplots_adjust(
            top=1.0 - TITLE_BAR_HEIGHT_FRAC - 0.005,
            bottom=0.045, left=0.11, right=0.985,
            hspace=0.60,
        )
        draw_figure_header(self._fig, "System Power Flow")

        plot_h = 1.0
        band_h = 0.18
        heights = [band_h] + [plot_h] * 8
        gs = GridSpec(9, 1, figure=self._fig, height_ratios=heights, hspace=0.65)

        ax_band = self._fig.add_subplot(gs[0, 0])
        fill_section_band(ax_band, "Measurements", COLOUR_MEAS_BAND)

        self._ax_p_iface = self._fig.add_subplot(gs[1, 0])
        self._ax_q_iface = self._fig.add_subplot(gs[2, 0], sharex=self._ax_p_iface)
        self._ax_p_gen   = self._fig.add_subplot(gs[3, 0], sharex=self._ax_p_iface)
        self._ax_q_gen   = self._fig.add_subplot(gs[4, 0], sharex=self._ax_p_iface)
        self._ax_tso_p   = self._fig.add_subplot(gs[5, 0], sharex=self._ax_p_iface)
        self._ax_tso_q   = self._fig.add_subplot(gs[6, 0], sharex=self._ax_p_iface)
        self._ax_dso_p   = self._fig.add_subplot(gs[7, 0], sharex=self._ax_p_iface)
        self._ax_dso_q   = self._fig.add_subplot(gs[8, 0], sharex=self._ax_p_iface)

        self._plot_axes = [
            self._ax_p_iface, self._ax_q_iface,
            self._ax_p_gen,   self._ax_q_gen,
            self._ax_tso_p,   self._ax_tso_q,
            self._ax_dso_p,   self._ax_dso_q,
        ]

        tile_title(self._ax_p_iface, "TSO-DSO Active Power Flow (primary)")
        tile_title(self._ax_q_iface, "TSO-DSO Reactive Power Flow (primary)")
        tile_title(self._ax_p_gen,   "Synchronous Generator P (with limits)")
        tile_title(self._ax_q_gen,   "Synchronous Generator Q (with limits)")
        tile_title(self._ax_tso_p,   "TSO P Balance (DER · Gen · Load · TSO-DSO)")
        tile_title(self._ax_tso_q,   "TSO Q Balance (DER · Gen · Load · TSO-DSO)")
        tile_title(self._ax_dso_p,   "DSO P Balance (DER · Load)")
        tile_title(self._ax_dso_q,   "DSO Q Balance (DER · Load)")

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

        # Interface trafos: append known + discover new
        for trafo_key, p_val in rec.dso_trafo_p_actual_mw.items():
            if trafo_key not in self._trafo_p:
                self._trafo_p[trafo_key] = [np.nan] * (len(self._t_all) - 1)
                self._trafo_q[trafo_key] = [np.nan] * (len(self._t_all) - 1)
        for trafo_key in list(self._trafo_p.keys()):
            self._trafo_p[trafo_key].append(rec.dso_trafo_p_actual_mw  .get(trafo_key, np.nan))
            self._trafo_q[trafo_key].append(rec.dso_trafo_q_actual_mvar.get(trafo_key, np.nan))

        # Per-gen P/Q — drawn from zone_balance_gen_* are sums; for per-gen
        # series we use the record's zone_q_gen / zone_p_gen arrays.
        for g in self._gen_ids_flat:
            z = self._gen_zone[g]
            # Locate g within zone_gen_indices[z]
            try:
                idx = self._zone_gen[z].index(g)
            except ValueError:
                self._gen_p[g].append(np.nan)
                self._gen_q[g].append(np.nan)
                continue
            p_arr = rec.zone_p_gen.get(z)
            q_arr = rec.zone_q_gen.get(z)
            self._gen_p[g].append(
                float(p_arr[idx]) if p_arr is not None and idx < len(p_arr) else np.nan
            )
            self._gen_q[g].append(
                float(q_arr[idx]) if q_arr is not None and idx < len(q_arr) else np.nan
            )

        # Aggregate TSO-side P/Q balance across all zones
        self._sum_tso_der_p  .append(_sum_record(rec.zone_balance_der_p_mw))
        self._sum_tso_der_q  .append(_sum_record(rec.zone_balance_der_q_mvar))
        self._sum_tso_gen_p  .append(_sum_record(rec.zone_balance_gen_p_mw))
        self._sum_tso_gen_q  .append(_sum_record(rec.zone_balance_gen_q_mvar))
        self._sum_tso_load_p .append(_sum_record(rec.zone_balance_load_p_mw))
        self._sum_tso_load_q .append(_sum_record(rec.zone_balance_load_q_mvar))
        self._sum_tso_dso_p_out.append(_sum_record(rec.zone_balance_tso_dso_p_out_mw))
        self._sum_tso_dso_q_out.append(_sum_record(rec.zone_balance_tso_dso_q_out_mvar))

        # Aggregate DSO-side balance across all HV groups
        self._sum_dso_der_p .append(_sum_record(rec.dso_group_der_p_mw))
        self._sum_dso_der_q .append(_sum_record(rec.dso_group_q_der_mvar))
        self._sum_dso_load_p.append(_sum_record(rec.dso_group_load_p_mw))
        self._sum_dso_load_q.append(_sum_record(rec.dso_group_load_q_mvar))

        self._call_count += 1
        if self._call_count % self._update_every == 0:
            self._redraw()

    # ─── redraw ─────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        self._redraw_iface_p()
        self._redraw_iface_q()
        self._redraw_gen(self._ax_p_gen, self._gen_p, "p", "P / MW")
        self._redraw_gen(self._ax_q_gen, self._gen_q, "q", "Q / Mvar")
        self._redraw_tso_balance_p()
        self._redraw_tso_balance_q()
        self._redraw_dso_balance_p()
        self._redraw_dso_balance_q()
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

    def _redraw_iface_p(self) -> None:
        self._redraw_iface(self._ax_p_iface, self._trafo_p, "P", "P / MW",
                            title="TSO-DSO Active Power Flow (primary)")

    def _redraw_iface_q(self) -> None:
        self._redraw_iface(self._ax_q_iface, self._trafo_q, "Q", "Q / Mvar",
                            title="TSO-DSO Reactive Power Flow (primary)")

    def _redraw_iface(
        self, ax: plt.Axes, data: Dict[str, List[float]],
        symbol: str, ylabel: str, *, title: str,
    ) -> None:
        ax.clear()
        tile_title(ax, title)
        ax.set_ylabel(ylabel)
        if not data:
            _fill_empty(ax, "no interface measurements available")
            return
        ax.axhline(0.0, color="k", ls=":", lw=0.6, alpha=0.4)
        t = np.asarray(self._t_all, dtype=float)
        # Colour by DSO-id prefix so trafos in the same DSO share a hue
        dso_colour: Dict[str, str] = {}
        for i, d in enumerate(self._dso_ids):
            dso_colour[d] = _c(5 + i)
        for k, (trafo_key, vals) in enumerate(data.items()):
            dso_id = trafo_key.split("|", 1)[0]
            c = dso_colour.get(dso_id, _c(k))
            arr = np.asarray(vals, dtype=float)
            if arr.size < t.size:
                arr = np.concatenate([arr, np.full(t.size - arr.size, np.nan)])
            arr = arr[:t.size]
            ax.plot(t, arr, color=c, lw=0.9, alpha=0.85)
        handles = [
            plt.Line2D([0], [0], color=dso_colour[d], lw=1.2, label=d)
            for d in self._dso_ids if d in dso_colour
        ]
        ax.grid(True, alpha=0.3)
        if handles:
            ax.legend(handles=handles, loc="upper left", fontsize=7,
                      ncol=min(len(handles), 3), frameon=False)

    def _redraw_gen(
        self, ax: plt.Axes, data: Dict[int, List[float]],
        which: str, ylabel: str,
    ) -> None:
        ax.clear()
        title = "Synchronous Generator P (with limits)" if which == "p" \
                else "Synchronous Generator Q (with limits)"
        tile_title(ax, title)
        ax.set_ylabel(ylabel)
        if not self._gen_ids_flat:
            _fill_empty(ax, "no synchronous generators available")
            return
        t = np.asarray(self._t_all, dtype=float)
        for g in self._gen_ids_flat:
            z = self._gen_zone[g]
            c = _c(1 + self._zone_ids.index(z))
            arr = np.asarray(data[g], dtype=float)
            if arr.size < t.size:
                arr = np.concatenate([arr, np.full(t.size - arr.size, np.nan)])
            arr = arr[:t.size]
            ax.plot(t, arr, color=c, lw=0.9, alpha=0.85)
            limits = self._gen_lim.get(g, {})
            key_min = "min_p_mw" if which == "p" else "min_q_mvar"
            key_max = "max_p_mw" if which == "p" else "max_q_mvar"
            lo = limits.get(key_min, float("nan"))
            hi = limits.get(key_max, float("nan"))
            if np.isfinite(lo):
                ax.axhline(lo, color=c, ls="--", lw=0.5, alpha=0.35)
            if np.isfinite(hi):
                ax.axhline(hi, color=c, ls="--", lw=0.5, alpha=0.35)
        handles = [
            plt.Line2D([0], [0], color=_c(1 + i), lw=1.2, label=f"Z{z}")
            for i, z in enumerate(self._zone_ids)
        ]
        ax.grid(True, alpha=0.3)
        ax.legend(handles=handles, loc="upper left", fontsize=7,
                  ncol=min(len(handles), 3), frameon=False)

    def _redraw_balance(
        self, ax: plt.Axes, title: str, ylabel: str,
        series: List[tuple],   # [(label, list, colour_idx, linestyle), ...]
        empty_msg: str,
    ) -> None:
        ax.clear()
        tile_title(ax, title)
        ax.set_ylabel(ylabel)
        if not any(any(np.isfinite(v) for v in ser[1]) for ser in series):
            _fill_empty(ax, empty_msg)
            return
        ax.axhline(0.0, color="k", ls=":", lw=0.6, alpha=0.4)
        t = np.asarray(self._t_all, dtype=float)
        for label, vals, c_idx, ls in series:
            arr = np.asarray(vals, dtype=float)
            if arr.size < t.size:
                arr = np.concatenate([arr, np.full(t.size - arr.size, np.nan)])
            arr = arr[:t.size]
            ax.plot(t, arr, color=_c(c_idx), lw=1.1, ls=ls, label=label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=False)

    def _redraw_tso_balance_p(self) -> None:
        self._redraw_balance(
            self._ax_tso_p,
            "TSO P Balance (DER · Gen · Load · TSO-DSO)",
            "P / MW",
            [
                (r"$\Sigma$ DER P",        self._sum_tso_der_p,    6, "-"),
                (r"$\Sigma$ Gen P",        self._sum_tso_gen_p,    1, "-"),
                (r"$\Sigma$ Load P",       self._sum_tso_load_p,   8, "--"),
                (r"$\Sigma$ TSO-DSO P",    self._sum_tso_dso_p_out,2, "-."),
            ],
            empty_msg="no TSO P balance data available",
        )

    def _redraw_tso_balance_q(self) -> None:
        self._redraw_balance(
            self._ax_tso_q,
            "TSO Q Balance (DER · Gen · Load · TSO-DSO)",
            "Q / Mvar",
            [
                (r"$\Sigma$ DER Q",        self._sum_tso_der_q,    6, "-"),
                (r"$\Sigma$ Gen Q",        self._sum_tso_gen_q,    1, "-"),
                (r"$\Sigma$ Load Q",       self._sum_tso_load_q,   8, "--"),
                (r"$\Sigma$ TSO-DSO Q",    self._sum_tso_dso_q_out,2, "-."),
            ],
            empty_msg="no TSO Q balance data available",
        )

    def _redraw_dso_balance_p(self) -> None:
        self._redraw_balance(
            self._ax_dso_p,
            "DSO P Balance (DER · Load)",
            "P / MW",
            [
                (r"$\Sigma$ DER P",  self._sum_dso_der_p,  6, "-"),
                (r"$\Sigma$ Load P", self._sum_dso_load_p, 8, "--"),
            ],
            empty_msg="no DSO P balance data available",
        )

    def _redraw_dso_balance_q(self) -> None:
        self._redraw_balance(
            self._ax_dso_q,
            "DSO Q Balance (DER · Load)",
            "Q / Mvar",
            [
                (r"$\Sigma$ DER Q",  self._sum_dso_der_q,  6, "-"),
                (r"$\Sigma$ Load Q", self._sum_dso_load_q, 8, "--"),
            ],
            empty_msg="no DSO Q balance data available",
        )


def _sum_record(d: Dict) -> float:
    """Sum the values of a dict of floats, treating NaNs as zero."""
    if not d:
        return float("nan")
    total = 0.0
    any_val = False
    for v in d.values():
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isnan(fv):
            total += fv
            any_val = True
    return total if any_val else float("nan")
