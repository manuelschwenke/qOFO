"""Python-side live view of RMS trajectories during a co-simulation.

PowerFactory's own live plots are unavailable here: ``pf/session.py`` uses
``GetApplicationExt()`` (external engine mode), where PF acts as a
calculation server driven from outside.  Showing the desktop gives a
database viewer, but the simulation machinery -- progress bar, live result
plots -- only engages when the calculation is launched from inside PF.  So a
visible plot page never animates, and there is no on-screen sign that a run
is in progress.

This module provides the equivalent from the Python side instead: a
matplotlib window fed from ``PowerFactoryPlant.harvest_trajectories()`` after
each dispatch interval, showing the true RMS trace (0.1-0.5 s resolution)
rather than the 20 s staircase the record-based plots give.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("qofo.pf")


class LiveTrajectoryView:
    """A matplotlib window redrawn once per dispatch interval.

    Parameters
    ----------
    label_filter :
        Predicate on the monitor label; selects which signals to draw.
    title, ylabel :
        Figure annotation.
    max_lines :
        Safety cap -- a stray filter matching hundreds of signals would make
        every redraw crawl.
    """

    def __init__(self, label_filter: Callable[[str], bool],
                 title: str = "RMS live view",
                 ylabel: str = "value",
                 max_lines: int = 60):
        import matplotlib
        # The batch figure modules import matplotlib and select "Agg" at
        # import time, which cannot draw on screen.  Force an interactive
        # backend for this window; savefig() keeps working for the batch
        # figures regardless of the active backend.
        if matplotlib.get_backend().lower() in ("agg", "template"):
            try:
                matplotlib.use("TkAgg", force=True)
            except Exception as exc:               # noqa: BLE001
                raise RuntimeError(
                    f"no interactive matplotlib backend available ({exc}); "
                    f"run without --live-plot"
                ) from exc
        import matplotlib.pyplot as plt

        self._plt = plt
        self._filter = label_filter
        self._max_lines = int(max_lines)
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(11, 6))
        self.ax.set_title(title)
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel(ylabel)
        self.ax.grid(alpha=0.3)
        self._lines: Dict[str, object] = {}
        self._data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.fig.tight_layout()
        self.fig.show()
        self._pump()

    # ------------------------------------------------------------------
    def matches(self, label: str) -> bool:
        """Label predicate, passed straight to ``harvest_trajectories``.

        Harvesting only the plotted signals is what keeps the per-interval
        cost bounded: ``ScreeningContext.read`` walks the result file per
        variable, so filtering at the source matters more than filtering
        after.
        """
        return bool(self._filter(label))

    def _pump(self) -> None:
        """Let the GUI toolkit process events so the window stays alive."""
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as exc:                   # noqa: BLE001
            logger.debug("live view pump failed: %s", exc)

    def update(self, trajectories: Dict[str, Tuple[np.ndarray, np.ndarray]],
               ) -> None:
        """Replace the plotted data with ``{label: (t, y)}`` and redraw."""
        for label, (t, y) in trajectories.items():
            if not self._filter(label):
                continue
            if label not in self._lines:
                if len(self._lines) >= self._max_lines:
                    continue
                (line,) = self.ax.plot([], [], lw=0.9)
                self._lines[label] = line
            self._data[label] = (np.asarray(t), np.asarray(y))
            self._lines[label].set_data(self._data[label])
        if self._data:
            self.ax.relim()
            self.ax.autoscale_view()
        self._pump()

    def close(self) -> None:
        try:
            self._plt.close(self.fig)
        except Exception:                          # noqa: BLE001
            pass


def ts_bus_voltage_view() -> LiveTrajectoryView:
    """Live view of every monitored TN (transmission) bus voltage."""
    return LiveTrajectoryView(
        label_filter=lambda s: s.startswith("u_TN_bus"),
        title="TS bus voltages -- RMS plant (live)",
        ylabel="V [pu]",
    )
