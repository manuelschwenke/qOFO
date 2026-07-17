"""
Live visualization of the active SBX-H v6 mechanism.

For every TSO-TSO corridor the figure shows:

- cycle residual dQ = Q_meas - Q_0 around the no-payment deadband;
- paid support Q_sup, signed in the corridor orientation;
- worst-terminal measured and scheduled voltages on both sides;
- hold/violation/transition strips for side A and side B;
- rolling all-area voltage-tracking RMSE and normalized Gini inequality;
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

from configs.color_config import TU_COLOURS, TU_PRIMARY

from visualisation.style import (
    TITLE_BAR_HEIGHT_FRAC,
    apply_serif_style,
    draw_figure_header,
    position_figure_in_slot,
    raise_figure_to_front,
    tile_title,
)

if TYPE_CHECKING:
    from experiments.helpers import MultiTSOIterationRecord
    from sbx_h.adapter import SBXRunnerAdapter

_C_Q_MEAS = TU_PRIMARY          # 5c yellow-green primary
_C_Q0 = TU_COLOURS[1]           # dark blue
_C_BAND = TU_COLOURS[1]         # dark blue
_C_SUPPORT = TU_COLOURS[5]      # magenta
_C_A = TU_PRIMARY               # 5c yellow-green primary
_C_B = TU_COLOURS[2]            # dark orange
_C_HOLD = TU_COLOURS[4]         # teal
_C_SAG = TU_COLOURS[8]          # red
_C_NEITHER = "#B8B8B8"          # neutral transition state
_C_ESCALATION = TU_COLOURS[10]  # purple
_ZONE_COLOURS = {
    1: TU_PRIMARY,
    2: TU_COLOURS[2],
    3: TU_COLOURS[4],
}


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
        self._freeze_t_min: Optional[float] = None
        self._axes_q: Dict[Tuple[int, int], plt.Axes] = {}
        self._axes_v: Dict[Tuple[int, int], plt.Axes] = {}
        self._ax_equity: Optional[plt.Axes] = None
        self._ax_equity_gini: Optional[plt.Axes] = None
        self._ax_pay: Optional[plt.Axes] = None
        self._last_adapter: Optional["SBXRunnerAdapter"] = None
        self._built = False

        self._fig = plt.figure(figsize=(11.5, 6.0))
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
        try:
            self._placeholder.remove()
        except Exception:
            pass

        n_corridors = len(self._keys)
        self._fig.set_size_inches(
            11.5,
            1.35 * n_corridors + 1.95,
            forward=True,
        )
        self._fig.subplots_adjust(
            top=1.0 - TITLE_BAR_HEIGHT_FRAC - 0.024,
            bottom=0.075,
            left=0.075,
            right=0.985,
            hspace=0.34,
            wspace=0.24,
        )
        grid = GridSpec(
            n_corridors + 1,
            2,
            figure=self._fig,
            height_ratios=[1.0] * n_corridors + [0.72],
            hspace=0.34,
            wspace=0.24,
        )

        shared_axis = None
        for index, key in enumerate(self._keys):
            q_axis = self._fig.add_subplot(
                grid[index, 0],
                sharex=shared_axis,
            )
            if shared_axis is None:
                shared_axis = q_axis
            v_axis = self._fig.add_subplot(
                grid[index, 1],
                sharex=shared_axis,
            )
            self._axes_q[key] = q_axis
            self._axes_v[key] = v_axis
            q_axis.tick_params(labelbottom=False)
            v_axis.tick_params(labelbottom=False)

        self._ax_equity = self._fig.add_subplot(
            grid[n_corridors, 0],
            sharex=shared_axis,
        )
        self._ax_equity_gini = self._ax_equity.twinx()
        self._ax_pay = self._fig.add_subplot(
            grid[n_corridors, 1],
            sharex=shared_axis,
        )
        for axis in (
            list(self._axes_q.values())
            + list(self._axes_v.values())
            + [self._ax_equity, self._ax_pay]
        ):
            axis.tick_params(axis="both", labelsize=8)
        self._ax_equity_gini.tick_params(axis="y", labelsize=8)
        self._ax_equity.set_xlabel("Time / min")
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
        self._draw_tracking_equity(adapter)
        self._draw_payments(adapter)
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
        records = adapter.scheduler.records.get(key, [])
        if records:
            residual = [
                item.deviation_mvar
                if np.isfinite(item.deviation_mvar)
                else item.q_meas_mvar - item.q_std_mvar
                for item in records
            ]
            times, delta_q = self._step_series(adapter, records, residual)
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
                -band,
                band,
                color=_C_BAND,
                alpha=0.12,
                lw=0,
                label="deadband +/- B_Q",
            )
            axis.plot(
                times,
                band,
                color=_C_BAND,
                lw=0.65,
                ls="--",
            )
            axis.plot(
                times,
                -band,
                color=_C_BAND,
                lw=0.65,
                ls="--",
            )
            axis.plot(
                times,
                delta_q,
                color=_C_Q_MEAS,
                lw=1.15,
                label="dQ = Q - Q_0",
            )
            axis.plot(
                times,
                support,
                color=_C_SUPPORT,
                lw=1.35,
                ls="-.",
                label="paid Q_sup",
            )
            axis.axhline(0.0, color=_C_Q0, lw=0.75)

            was_escalated = False
            for item in records:
                if item.escalation and not was_escalated:
                    axis.axvline(
                        self._cycle_time_min(adapter, item.cycle),
                        color=_C_ESCALATION,
                        lw=0.8,
                        ls=":",
                    )
                was_escalated = item.escalation

            current = records[-1]
            detail = (
                f"Q={current.q_meas_mvar:+.1f} | "
                f"Q0={current.q_std_mvar:+.1f} | "
                f"B_Q={current.q_band_mvar:.1f} Mvar"
            )
        else:
            detail = "awaiting first settlement cycle"

        tile_title(
            axis,
            f"({key[0]},{key[1]}) reactive residual | {detail}",
        )
        axis.title.set_fontsize(8.5)
        axis.set_ylabel("dQ, Q_sup / Mvar", fontsize=8)
        axis.grid(alpha=0.25, lw=0.4)
        if key == self._keys[0] and records:
            axis.legend(
                loc="upper left",
                fontsize=6.2,
                ncol=3,
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
            pairs_a = [(measured[bus], scheduled[bus]) for bus in buses_a]
            pairs_b = [(measured[bus], scheduled[bus]) for bus in buses_b]
            worst_a = max(pairs_a, key=lambda pair: abs(pair[0] - pair[1]))
            worst_b = max(pairs_b, key=lambda pair: abs(pair[0] - pair[1]))
            measured_a.append(worst_a[0])
            measured_b.append(worst_b[0])
            scheduled_a.append(worst_a[1])
            scheduled_b.append(worst_b[1])
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
        tolerance = adapter.config.v_hold_tolerance_pu
        axis.fill_between(
            time,
            va_ref - tolerance,
            va_ref + tolerance,
            color=_C_A,
            alpha=0.07,
            lw=0,
        )
        axis.fill_between(
            time,
            vb_ref - tolerance,
            vb_ref + tolerance,
            color=_C_B,
            alpha=0.07,
            lw=0,
        )
        axis.plot(time, va, color=_C_A, lw=1.15, label="V_A worst")
        axis.plot(time, vb, color=_C_B, lw=1.15, label="V_B worst")
        axis.plot(
            time,
            va_ref,
            color=_C_A,
            lw=0.85,
            ls="--",
            label="V_A scheduled",
        )
        axis.plot(
            time,
            vb_ref,
            color=_C_B,
            lw=0.85,
            ls="--",
            label="V_B scheduled",
        )

        records = adapter.scheduler.records.get(key, [])
        for item in records:
            t_left = self._cycle_time_min(adapter, item.cycle - 1)
            t_right = self._cycle_time_min(adapter, item.cycle)
            axis.axvspan(
                t_left,
                t_right,
                ymin=0.00,
                ymax=0.027,
                color=self._state_colour(item.a_sags, item.a_holds),
                alpha=0.92,
                lw=0,
            )
            axis.axvspan(
                t_left,
                t_right,
                ymin=0.034,
                ymax=0.061,
                color=self._state_colour(item.b_sags, item.b_holds),
                alpha=0.92,
                lw=0,
            )
        axis.text(
            0.003, 0.014, "A", transform=axis.transAxes,
            fontsize=5.5, va="center",
        )
        axis.text(
            0.003, 0.048, "B", transform=axis.transAxes,
            fontsize=5.5, va="center",
        )

        if time.size:
            detail = (
                f"A {va[-1]:.4f}/{va_ref[-1]:.4f} | "
                f"B {vb[-1]:.4f}/{vb_ref[-1]:.4f} pu (meas/sched)"
            )
        else:
            detail = "awaiting terminal measurements"
        tile_title(
            axis,
            f"({key[0]},{key[1]}) worst terminal voltage | {detail}",
        )
        axis.title.set_fontsize(8.5)
        axis.set_ylabel("V / pu", fontsize=8)
        axis.grid(alpha=0.25, lw=0.4)
        if key == self._keys[0]:
            handles = [
                Line2D([], [], color=_C_A, lw=1.1, label="V_A measured"),
                Line2D([], [], color=_C_B, lw=1.1, label="V_B measured"),
                Line2D(
                    [], [], color="0.35", lw=0.9, ls="--",
                    label="schedule +/- hold tol.",
                ),
                Patch(color=_C_HOLD, label="hold"),
                Patch(color=_C_SAG, label="violation"),
                Patch(color=_C_NEITHER, label="transition"),
            ]
            axis.legend(
                handles=handles,
                loc="upper left",
                fontsize=6.0,
                ncol=3,
                frameon=False,
            )

    def _draw_tracking_equity(
        self,
        adapter: "SBXRunnerAdapter",
    ) -> None:
        axis = self._ax_equity
        gini_axis = self._ax_equity_gini
        axis.cla()
        gini_axis.cla()
        gini_axis.yaxis.set_label_position("right")
        gini_axis.yaxis.tick_right()
        gini_axis.spines["right"].set_position(("axes", 1.0))

        history = adapter.tracking_equity_history
        handles = []
        if history:
            times = np.asarray([
                adapter.freeze_time_s / 60.0
                + iteration * adapter.config.tso_period_s / 60.0
                for iteration, _metric in history
            ])
            for area in adapter.scheduler.area_ids:
                line = axis.plot(
                    times,
                    [
                        metric.area_rmse_mpu[area]
                        for _iteration, metric in history
                    ],
                    lw=1.15,
                    color=_ZONE_COLOURS.get(area, "0.3"),
                    label=f"area {area} RMSE",
                )[0]
                handles.append(line)
            gini_line = gini_axis.plot(
                times,
                [metric.gini for _iteration, metric in history],
                lw=1.1,
                ls="--",
                color=_C_ESCALATION,
                label="inequality G_V",
            )[0]
            handles.append(gini_line)
            current = history[-1][1]
            detail = (
                f"mean={current.mean_rmse_mpu:.1f}, "
                f"max=z{current.worst_area} "
                f"{current.worst_rmse_mpu:.1f} mpu, "
                f"G_V={current.gini:.2f}"
            )
        else:
            detail = "awaiting TSO voltage samples"

        tile_title(axis, f"V-tracking equity | {detail}")
        axis.title.set_fontsize(8.0)
        axis.set_ylabel("Cycle RMSE / mpu", fontsize=8)
        axis.set_xlabel("Time / min")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.3, lw=0.4)
        gini_axis.set_ylabel("G_V / 1", fontsize=8, rotation=270, labelpad=5)
        gini_axis.set_ylim(0.0, 1.0)
        gini_axis.grid(False)
        gini_axis.tick_params(axis="y", labelsize=8)
        if handles:
            axis.legend(
                handles=handles,
                loc="upper left",
                fontsize=6.2,
                ncol=2,
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
        axis.title.set_fontsize(8.5)
        axis.set_ylabel("Net payment / EUR", fontsize=8)
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

    def save(self, path) -> None:
        """Force a final redraw and save the current figure."""
        if self._built and self._last_adapter is not None:
            self._redraw(self._last_adapter)
        self._fig.savefig(path, dpi=180)
