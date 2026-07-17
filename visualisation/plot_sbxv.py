"""Observational live plot for the SBX-V coordination mechanism."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np

from configs.color_config import TU_COLOURS, TU_PRIMARY
from sbx_v.directions import Direction
from sbx_v.settlement import DSO_DELIVERS, SettlementEngine, WindowObservation
from visualisation.style import (
    apply_serif_style,
    draw_figure_header,
    position_figure_in_slot,
    raise_figure_to_front,
    tile_title,
)

if TYPE_CHECKING:
    from experiments.helpers import MultiTSOIterationRecord
    from sbx_v.adapter import SBXVRunnerAdapter

_COLOURS = (TU_PRIMARY, TU_COLOURS[2], TU_COLOURS[4], TU_COLOURS[5])


class SBXVMechanismLivePlotter:
    """Plot normal bands, metering, requests, grants and remuneration."""

    def __init__(
        self, *, update_every: int = 3, slot_idx: int = 3,
        layout: str = "dual_screen", use_tex: bool = False,
    ) -> None:
        apply_serif_style(use_tex=use_tex)
        plt.ion()
        self._update_every = max(1, int(update_every))
        self._calls = 0
        self._last_adapter: Optional["SBXVRunnerAdapter"] = None
        self._time_min = 0.0
        self._fig, self._axes = plt.subplots(3, 1, figsize=(10.8, 7.1), sharex=True)
        self._fig.subplots_adjust(
            top=0.91, bottom=0.085, left=0.09, right=0.985, hspace=0.32
        )
        draw_figure_header(
            self._fig,
            "SBX-V - Normal Bands, Requests, Grants and Remuneration",
        )
        position_figure_in_slot(self._fig, slot_idx, layout=layout, n_slots=4)
        plt.pause(0.01)
        raise_figure_to_front(self._fig)

    def update(
        self, rec: "MultiTSOIterationRecord",
        adapter: Optional["SBXVRunnerAdapter"],
    ) -> None:
        """Read one completed plant step; never feed values to a controller."""
        self._calls += 1
        self._time_min = float(rec.time_s) / 60.0
        if adapter is not None:
            self._last_adapter = adapter
            if self._calls % self._update_every == 0:
                self._redraw(adapter)

    @staticmethod
    def _signed(direction: Direction, value: float) -> float:
        return float(direction.q_hv_sign) * float(value)

    def _redraw(self, adapter: "SBXVRunnerAdapter") -> None:
        self._draw_metering(adapter)
        self._draw_requests(adapter)
        self._draw_payments(adapter)
        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    def _draw_metering(self, adapter: "SBXVRunnerAdapter") -> None:
        axis = self._axes[0]
        axis.cla()
        for i, area_id in enumerate(sorted(adapter.areas)):
            colour = _COLOURS[i % len(_COLOURS)]
            band = adapter.bands[area_id]
            rows = adapter.meters[area_id].finalise()
            time = [row.t_end_s / 60.0 for row in rows]
            q_mean = [row.q_mean_mvar for row in rows]
            if rows:
                axis.step(time, q_mean, where="post", color=colour,
                          lw=1.35, label=f"{area_id}: metered Q")
            x_right = max(self._time_min, 1.0)
            axis.fill_between(
                [0.0, x_right], [-band.q_raise_mvar] * 2,
                [band.q_lower_mvar] * 2, color=colour, alpha=0.055,
            )
            axis.hlines(
                (-band.q_raise_mvar, band.q_lower_mvar), 0.0, x_right,
                color=colour, lw=0.65, ls="--",
            )
        axis.axhline(0.0, color="0.45", lw=0.6)
        tile_title(axis, "15-minute area metering and Normalbereich")
        axis.set_ylabel("Q [Mvar]")
        axis.grid(alpha=0.25, lw=0.4)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="upper left", fontsize=7, ncol=2, frameon=False)

    def _draw_requests(self, adapter: "SBXVRunnerAdapter") -> None:
        axis = self._axes[1]
        axis.cla()
        current = max(0, int(adapter._k_now) // adapter.config.k_window)
        windows = np.arange(current + 2, dtype=int)
        time = windows * adapter.config.window_s / 60.0
        for i, area_id in enumerate(sorted(adapter.areas)):
            colour = _COLOURS[i % len(_COLOURS)]
            scheduler = adapter.schedulers[adapter.areas[area_id].zone]
            grants = [
                sum(
                    self._signed(
                        direction,
                        scheduler.ledger.granted_mvar(
                            area_id, direction, int(window)
                        ),
                    )
                    for direction in (Direction.RAISING, Direction.LOWERING)
                )
                for window in windows
            ]
            axis.step(time, grants, where="post", color=colour, lw=1.45,
                      label=f"{area_id}: active grant")
            for event in scheduler.pipeline.log:
                if event[0] == "request" and event[2] == area_id:
                    _, window, _, name, n_quanta, _ = event
                    value = self._signed(
                        Direction(name),
                        n_quanta * adapter.config.dq_grant_mvar,
                    )
                    axis.scatter(
                        window * adapter.config.window_s / 60.0,
                        value, marker="D", s=24, color=colour, zorder=4,
                    )
        axis.axhline(0.0, color="0.45", lw=0.6)
        tile_title(axis, "Requests (diamonds) and confirmed active grants")
        axis.set_ylabel("extension [Mvar]")
        axis.grid(alpha=0.25, lw=0.4)
        if adapter.areas:
            axis.legend(loc="upper left", fontsize=7, ncol=2, frameon=False)

    @staticmethod
    def _settle(adapter: "SBXVRunnerAdapter"):
        observations = []
        complete = {}
        for area_id, meter in adapter.meters.items():
            rows = meter.finalise()
            complete[area_id] = {row.window_index for row in rows}
            for row in rows:
                q_values = adapter._q_set_acc.get((area_id, row.window_index))
                observations.append(WindowObservation(
                    area_id=area_id, window_index=row.window_index,
                    t_start_s=row.t_start_s, q_meas_mvar=row.q_mean_mvar,
                    q_set_mvar=float(np.mean(q_values)) if q_values else None,
                ))
        grants = [
            grant
            for scheduler in adapter.schedulers.values()
            for grant in scheduler.ledger.to_grant_records(
                delivering_party=DSO_DELIVERS
            )
            if all(
                window in complete.get(grant.area_id, set())
                for window in range(grant.window_first, grant.window_end)
            )
        ]
        if not observations:
            return None
        return SettlementEngine(adapter.config, adapter.bands).settle(
            observations, grants
        )

    def _draw_payments(self, adapter: "SBXVRunnerAdapter") -> None:
        axis = self._axes[2]
        axis.cla()
        result = self._settle(adapter)
        capacity = 0.0
        if result is not None:
            capacity = sum(
                row.pay_cap_avg_eur + row.pay_cap_grenz_eur
                for row in result.day_rows
            )
            for i, area_id in enumerate(sorted(adapter.areas)):
                rows = sorted(
                    (row for row in result.window_rows
                     if row.area_id == area_id),
                    key=lambda row: row.window_index,
                )
                payments = np.cumsum([
                    row.pay_energy_avg_eur + row.pay_energy_grenz_eur
                    for row in rows
                ])
                if rows:
                    axis.step(
                        [(row.t_start_s + adapter.config.window_s) / 60.0
                         for row in rows],
                        payments, where="post",
                        color=_COLOURS[i % len(_COLOURS)], lw=1.35,
                        label=area_id,
                    )
        tile_title(
            axis,
            f"Cumulative remuneration | capacity accrued: {capacity:.2f} EUR",
        )
        axis.set_ylabel("payment [EUR]")
        axis.set_xlabel("Simulation time [min]")
        axis.grid(alpha=0.25, lw=0.4)
        if result is not None:
            axis.legend(loc="upper left", fontsize=7, ncol=3, frameon=False)

    def save(self, path) -> None:
        """Force a final redraw and save the live figure."""
        if self._last_adapter is not None:
            self._redraw(self._last_adapter)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(target, dpi=180)
