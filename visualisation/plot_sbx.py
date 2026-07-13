"""
Live visualization of the active SBX-H v6 mechanism.

For every TSO-TSO corridor the figure shows:

- measured reference-end reactive flow Q_meas;
- the measured-P baseline Q_0 and deadband Q_0 +/- B_Q;
- paid support Q_sup, signed in the corridor orientation;
- measured and scheduled terminal voltages on both sides;
- hold/sag/neither condition strips for side A and side B;
- cumulative bilateral settlement payments.

Positive corridor quantities point from area A to area B. Q_sup is
positive for A-to-B support and negative for B-to-A support. Strength
is deliberately absent because it is diagnostic only and is not a
traded SBX-H product.

Author: Manuel Schwenke / OpenAI Codex
Date: 2026-07-13
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

_C_Q_MEAS = "#cc6677"
_C_Q0 = "#4477aa"
_C_BAND = "#4477aa"
_C_SUPPORT = "#aa3377"
_C_A = "#4477aa"
_C_B = "#ee6677"
_C_HOLD = "#228833"
_C_SAG = "#cc3311"
_C_NEITHER = "#bbbbbb"
_C_ESCALATION = "#000000"
_ZONE_COLOURS = {1: "#4477aa", 2: "#ee6677", 3: "#228833"}


class SBXMechanismLivePlotter:
    """Live thesis figure for scheduled-voltage support settlement."""

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
        self._freeze_t_min: Optional[float] = None
        self._axes_q: Dict[Tuple[int, int], plt.Axes] = {}
        self._axes_v: Dict[Tuple[int, int], plt.Axes] = {}
        self._ax_support: Optional[plt.Axes] = None
        self._ax_pay: Optional[plt.Axes] = None
        self._last_adapter: Optional["SBXRunnerAdapter"] = None
        self._built = False

        self._fig = plt.figure(figsize=(12.0, 9.0))
        try:
            self._fig.canvas.manager.set_window_title(
                "SBX-H scheduled-voltage support"
            )
        except Exception:
            pass
        draw_figure_header(
            self._fig,
            "SBX-H - Scheduled Terminal Voltages and Reactive Support",
        )
        self._status = self._fig.text(
            0.5,
            0.947,
            "Waiting for SBX-H contract initialization",
            ha="center",
            va="top",
            fontsize=8,
            color="0.25",
        )
        self._placeholder = self._fig.text(
            0.5,
            0.5,
            "SBX-H contracts not yet initialized",
            ha="center",
            va="center",
            fontsize=11,
            color="0.35",
        )
        position_figure_in_slot(
            self._fig,
            self._slot_idx,
            layout=self._layout,
            n_slots=4,
        )
        plt.pause(0.01)
        raise_figure_to_front(self._fig)

    def _build(self, keys: Sequence[Tuple[int, int]]) -> None:
        self._keys = list(keys)
        for key in self._keys:
            self._q[key] = []
        try:
            self._placeholder.remove()
        except Exception:
            pass

        n_corridors = len(self._keys)
        self._fig.set_size_inches(
            12.0,
            2.35 * n_corridors + 4.0,
            forward=True,
        )
        self._fig.subplots_adjust(
            top=1.0 - TITLE_BAR_HEIGHT_FRAC - 0.035,
            bottom=0.055,
            left=0.075,
            right=0.985,
            hspace=0.68,
            wspace=0.18,
        )
        grid = GridSpec(
            n_corridors + 2,
            2,
            figure=self._fig,
            height_ratios=[0.18] + [1.0] * n_corridors + [1.0],
            hspace=0.68,
            wspace=0.18,
        )
        band_axis = self._fig.add_subplot(grid[0, :])
        fill_section_band(
            band_axis,
            "Corridor Flow, Terminal Voltages, and Hold/Sag State",
            COLOUR_MEAS_BAND,
        )

        previous_q = None
        previous_v = None
        for index, key in enumerate(self._keys):
            q_axis = self._fig.add_subplot(
                grid[1 + index, 0],
                sharex=previous_q,
            )
            v_axis = self._fig.add_subplot(
                grid[1 + index, 1],
                sharex=previous_v,
            )
            previous_q = previous_q or q_axis
            previous_v = previous_v or v_axis
            self._axes_q[key] = q_axis
            self._axes_v[key] = v_axis

        self._ax_support = self._fig.add_subplot(
            grid[1 + n_corridors, 0],
            sharex=previous_q,
        )
        self._ax_pay = self._fig.add_subplot(
            grid[1 + n_corridors, 1],
            sharex=previous_v,
        )
        for axis in (
            list(self._axes_q.values())
            + list(self._axes_v.values())
            + [self._ax_support, self._ax_pay]
        ):
            axis.tick_params(axis="both", labelsize=8)
        self._ax_support.set_xlabel("Time / min")
        self._ax_pay.set_xlabel("Time / min")
        position_figure_in_slot(
            self._fig,
            self._slot_idx,
            layout=self._layout,
            n_slots=4,
        )
        plt.pause(0.01)
        raise_figure_to_front(self._fig)
        self._built = True

    def update(
        self,
        rec: "MultiTSOIterationRecord",
        adapter: Optional["SBXRunnerAdapter"],
        corridor_q_mvar: Optional[Dict[Tuple[int, int], float]],
    ) -> None:
        """Record one plant step and redraw periodically."""
        self._call_count += 1
        if adapter is None or corridor_q_mvar is None:
            if self._call_count % self._update_every == 0:
                self._keep_responsive()
            return
        if not self._built:
            self._build(sorted(corridor_q_mvar))
        if self._freeze_t_min is None:
            self._freeze_t_min = rec.time_s / 60.0
        self._last_adapter = adapter
        self._t.append(rec.time_s / 60.0)
        for key in self._keys:
            self._q[key].append(
                float(corridor_q_mvar.get(key, np.nan))
            )
        if self._call_count % self._update_every == 0:
            self._redraw(adapter)

    def _keep_responsive(self) -> None:
        try:
            plt.pause(0.001)
        except Exception:
            pass

    def _cycle_time_min(
        self,
        adapter: "SBXRunnerAdapter",
        cycle: int,
    ) -> float:
        if self._freeze_t_min is None:
            return float("nan")
        return (
            self._freeze_t_min
            + cycle * adapter.config.t_cycle_min
        )

    def _step_series(
        self,
        adapter: "SBXRunnerAdapter",
        records,
        values,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not records:
            return np.asarray([]), np.asarray([])
        edges = [
            self._cycle_time_min(adapter, records[0].cycle - 1)
        ]
        edges.extend(
            self._cycle_time_min(adapter, item.cycle)
            for item in records
        )
        return (
            np.repeat(np.asarray(edges, dtype=float), 2)[1:-1],
            np.repeat(np.asarray(values, dtype=float), 2),
        )

    @staticmethod
    def _signed_support(record) -> float:
        if record.support_direction == "a_to_b":
            return float(record.support_mvar)
        if record.support_direction == "b_to_a":
            return -float(record.support_mvar)
        return 0.0

    def _redraw(self, adapter: "SBXRunnerAdapter") -> None:
        for key in self._keys:
            self._draw_corridor_q(adapter, key)
            self._draw_corridor_v(adapter, key)
        self._draw_support(adapter)
        self._draw_payments(adapter)
        self._draw_current_status(adapter)
        try:
            self._fig.canvas.draw_idle()
            plt.pause(0.001)
        except Exception:
            pass

    def _draw_corridor_q(
        self,
        adapter: "SBXRunnerAdapter",
        key: Tuple[int, int],
    ) -> None:
        axis = self._axes_q[key]
        axis.cla()
        axis.plot(
            np.asarray(self._t),
            np.asarray(self._q[key]),
            color=_C_Q_MEAS,
            lw=0.9,
            label="Q_meas",
        )
        records = adapter.scheduler.records.get(key, [])
        if records:
            times, q0 = self._step_series(
                adapter,
                records,
                [item.q_std_mvar for item in records],
            )
            _, band = self._step_series(
                adapter,
                records,
                [item.q_band_mvar for item in records],
            )
            _, support = self._step_series(
                adapter,
                records,
                [self._signed_support(item) for item in records],
            )
            axis.fill_between(
                times,
                q0 - band,
                q0 + band,
                color=_C_BAND,
                alpha=0.14,
                lw=0,
                label="Q_0 +/- B_Q",
            )
            axis.plot(
                times,
                q0,
                color=_C_Q0,
                lw=1.25,
                label="Q_0",
            )
            axis.plot(
                times,
                support,
                color=_C_SUPPORT,
                lw=1.1,
                ls="-.",
                label="signed Q_sup",
            )
            for item in records:
                if item.escalation:
                    axis.axvline(
                        self._cycle_time_min(adapter, item.cycle),
                        color=_C_ESCALATION,
                        lw=0.8,
                        ls=":",
                    )
        tile_title(
            axis,
            f"Corridor ({key[0]},{key[1]}): Q at area-{key[0]} end",
        )
        axis.set_ylabel("Q / Mvar", fontsize=8)
        axis.grid(alpha=0.3, lw=0.4)
        axis.legend(
            loc="upper left",
            fontsize=6.7,
            ncol=4,
            frameon=False,
        )

    def _terminal_series(
        self,
        adapter: "SBXRunnerAdapter",
        key: Tuple[int, int],
    ):
        corridor = adapter.scheduler.corridors[key]
        times: List[float] = []
        measured_a: List[float] = []
        measured_b: List[float] = []
        scheduled_a: List[float] = []
        scheduled_b: List[float] = []
        for iteration, measured, scheduled in adapter.terminal_history:
            buses_a = [line.bus_a for line in corridor.lines]
            buses_b = [line.bus_b for line in corridor.lines]
            if not all(
                bus in measured and bus in scheduled
                for bus in buses_a + buses_b
            ):
                continue
            times.append(
                adapter.freeze_time_s / 60.0
                + iteration * adapter.config.tso_period_s / 60.0
            )
            measured_a.append(min(measured[bus] for bus in buses_a))
            measured_b.append(min(measured[bus] for bus in buses_b))
            scheduled_a.append(min(scheduled[bus] for bus in buses_a))
            scheduled_b.append(min(scheduled[bus] for bus in buses_b))
        return (
            np.asarray(times),
            np.asarray(measured_a),
            np.asarray(measured_b),
            np.asarray(scheduled_a),
            np.asarray(scheduled_b),
        )

    @staticmethod
    def _state_colour(sags: bool, holds: bool) -> str:
        if sags:
            return _C_SAG
        if holds:
            return _C_HOLD
        return _C_NEITHER

    def _draw_corridor_v(
        self,
        adapter: "SBXRunnerAdapter",
        key: Tuple[int, int],
    ) -> None:
        axis = self._axes_v[key]
        axis.cla()
        time, va, vb, va_ref, vb_ref = self._terminal_series(
            adapter,
            key,
        )
        axis.plot(time, va, color=_C_A, lw=1.1, label="V_A meas")
        axis.plot(time, vb, color=_C_B, lw=1.1, label="V_B meas")
        axis.plot(
            time,
            va_ref,
            color=_C_A,
            lw=0.9,
            ls="--",
            label="V_A sched",
        )
        axis.plot(
            time,
            vb_ref,
            color=_C_B,
            lw=0.9,
            ls="--",
            label="V_B sched",
        )

        records = adapter.scheduler.records.get(key, [])
        for item in records:
            t_left = self._cycle_time_min(adapter, item.cycle - 1)
            t_right = self._cycle_time_min(adapter, item.cycle)
            axis.axvspan(
                t_left,
                t_right,
                ymin=0.00,
                ymax=0.035,
                color=self._state_colour(
                    item.a_sags,
                    item.a_holds,
                ),
                alpha=0.9,
                lw=0,
            )
            axis.axvspan(
                t_left,
                t_right,
                ymin=0.045,
                ymax=0.080,
                color=self._state_colour(
                    item.b_sags,
                    item.b_holds,
                ),
                alpha=0.9,
                lw=0,
            )
        axis.text(
            0.003,
            0.017,
            "A",
            transform=axis.transAxes,
            fontsize=5.5,
            va="center",
        )
        axis.text(
            0.003,
            0.062,
            "B",
            transform=axis.transAxes,
            fontsize=5.5,
            va="center",
        )
        tile_title(
            axis,
            f"Corridor ({key[0]},{key[1]}): terminal voltages "
            "(state strips A/B)",
        )
        axis.set_ylabel("V / pu", fontsize=8)
        axis.grid(alpha=0.3, lw=0.4)
        handles = [
            Line2D([], [], color=_C_A, lw=1.1, label="V_A meas"),
            Line2D([], [], color=_C_B, lw=1.1, label="V_B meas"),
            Line2D(
                [],
                [],
                color="0.35",
                lw=0.9,
                ls="--",
                label="scheduled",
            ),
            Patch(color=_C_HOLD, label="hold=True"),
            Patch(color=_C_SAG, label="sag=True"),
            Patch(color=_C_NEITHER, label="neither"),
        ]
        axis.legend(
            handles=handles,
            loc="upper left",
            fontsize=6.4,
            ncol=3,
            frameon=False,
        )

    def _draw_support(self, adapter: "SBXRunnerAdapter") -> None:
        axis = self._ax_support
        axis.cla()
        for key in self._keys:
            records = adapter.scheduler.records.get(key, [])
            if not records:
                continue
            times, values = self._step_series(
                adapter,
                records,
                [self._signed_support(item) for item in records],
            )
            axis.plot(
                times,
                values,
                lw=1.2,
                label=f"({key[0]},{key[1]})",
            )
        axis.axhline(0.0, color="0.4", lw=0.6)
        tile_title(
            axis,
            "Paid support Q_sup (+ A to B, - B to A)",
        )
        axis.set_ylabel("Q_sup / Mvar", fontsize=8)
        axis.set_xlabel("Time / min")
        axis.grid(alpha=0.3, lw=0.4)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(
                handles,
                labels,
                loc="upper left",
                fontsize=6.7,
                ncol=3,
                frameon=False,
            )

    def _draw_payments(self, adapter: "SBXRunnerAdapter") -> None:
        axis = self._ax_pay
        axis.cla()
        by_cycle: Dict[int, Dict[int, float]] = {}
        for settlements in adapter.scheduler.settlements.values():
            for item in settlements:
                row = by_cycle.setdefault(item.cycle, {})
                for area, amount in item.payments_eur.items():
                    row[area] = row.get(area, 0.0) + amount

        cumulative = {
            area: 0.0 for area in adapter.scheduler.area_ids
        }
        series = {
            area: [] for area in adapter.scheduler.area_ids
        }
        times: List[float] = []
        for cycle in sorted(by_cycle):
            for area, amount in by_cycle[cycle].items():
                cumulative[area] += amount
            times.append(self._cycle_time_min(adapter, cycle + 1))
            for area in series:
                series[area].append(cumulative[area])
        for area, values in sorted(series.items()):
            axis.plot(
                times,
                values,
                lw=1.2,
                color=_ZONE_COLOURS.get(area, "0.3"),
                label=f"area {area}",
            )
        axis.axhline(0.0, color="0.4", lw=0.6)
        tile_title(axis, "Cumulative bilateral settlement")
        axis.set_ylabel("Payment / EUR", fontsize=8)
        axis.set_xlabel("Time / min")
        axis.grid(alpha=0.3, lw=0.4)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(
                handles,
                labels,
                loc="upper left",
                fontsize=6.7,
                ncol=3,
                frameon=False,
            )

    def _draw_current_status(
        self,
        adapter: "SBXRunnerAdapter",
    ) -> None:
        parts = []
        for key in self._keys:
            records = adapter.scheduler.records.get(key, [])
            if not records:
                continue
            item = records[-1]
            parts.append(
                f"{key}: Q0={item.q_std_mvar:+.1f} Mvar, "
                f"Qsup={self._signed_support(item):+.1f} Mvar, "
                f"{item.support_state}"
            )
        self._status.set_text(" | ".join(parts))

    def save(self, path) -> None:
        """Force a final redraw and save the current figure."""
        if self._built and self._last_adapter is not None:
            self._redraw(self._last_adapter)
        self._fig.savefig(path, dpi=180)
