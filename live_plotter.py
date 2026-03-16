#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Plotter for Cascaded OFO Controller Simulation
=====================================================

Provides a ``LivePlotter`` class that creates and updates matplotlib figures
during or after a simulation run.  Each call to ``update()`` appends the
latest iteration data to the existing plot lines without redrawing from
scratch, making it suitable for monitoring long-running simulations in an
interactive Python session.

Colour Palette
--------------
All plots use the official TU Darmstadt PANTONE colour palette in the
following order (as specified):

    Index  Code  HEX        Description
    0      5c    #B1BD00    Yellow-green
    1      1c    #004E8A    Dark blue
    2      8c    #CC4C03    Dark orange
    3      6c    #D7AC00    Gold
    4      3c    #008877    Teal
    5      10c   #951169    Magenta
    6      4c    #7FAB16    Olive green
    7      2c    #00689D    Mid blue
    8      9c    #B90F22    Red
    9      7c    #D28700    Amber
    10     11c   #611C73    Purple

Usage
-----
::

    from live_plotter import LivePlotter
    from run_cascade import run_cascade

    plotter = LivePlotter(n_scenarios=3)
    log = run_cascade(v_setpoint_pu=1.05, n_minutes=30)
    plotter.update(scenario_index=0, log=log)
    plotter.show()

Author: Manuel Schwenke
Date:   2026-03-16
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# TU Darmstadt PANTONE colour palette – ordered as specified
# ---------------------------------------------------------------------------

#: Colour sequence to be used for successive plots (lines, scatter, bar charts).
#: The order follows the specification:  5c, 1c, 8c, 6c, 3c, 10c, 4c, 2c, 9c, 7c, 11c
TU_COLOURS: List[str] = [
    "#B1BD00",  # 0  –  5c  Yellow-green   (PANTONE 390)
    "#004E8A",  # 1  –  1c  Dark blue      (PANTONE 2945)
    "#CC4C03",  # 2  –  8c  Dark orange    (PANTONE 173)
    "#D7AC00",  # 3  –  6c  Gold           (PANTONE 110)
    "#008877",  # 4  –  3c  Teal           (PANTONE 3285)
    "#951169",  # 5  –  10c Magenta        (PANTONE 249)
    "#7FAB16",  # 6  –  4c  Olive green    (PANTONE 376)
    "#00689D",  # 7  –  2c  Mid blue       (PANTONE 3015)
    "#B90F22",  # 8  –  9c  Red            (PANTONE 193)
    "#D28700",  # 9  –  7c  Amber          (PANTONE 124)
    "#611C73",  # 10 –  11c Purple         (PANTONE 268)
]


def tu_colour(index: int) -> str:
    """
    Return the TU Darmstadt PANTONE colour for the given zero-based index.

    Parameters
    ----------
    index:
        Zero-based colour index.  Wraps around if ``index >= len(TU_COLOURS)``.

    Returns
    -------
    str
        Hex colour string, e.g. ``"#B1BD00"``.
    """
    return TU_COLOURS[index % len(TU_COLOURS)]


# ---------------------------------------------------------------------------
# LivePlotter
# ---------------------------------------------------------------------------

class LivePlotter:
    """
    Incremental live plotter for cascaded OFO simulation results.

    The figure contains four subplots:

    1. **TN voltage** – plant TN (EHV) voltage magnitudes [p.u.] per iteration.
    2. **DN voltage** – plant DN (HV) voltage magnitudes [p.u.] per iteration.
    3. **TSO objective** – TSO solver objective value per active TSO step.
    4. **DSO objective** – DSO solver objective value per active DSO step.

    Each scenario / voltage setpoint is rendered with a distinct colour from
    the TU Darmstadt PANTONE palette (order: 5c, 1c, 8c, …).

    Parameters
    ----------
    n_scenarios:
        Expected number of scenarios (used only for legend pre-allocation;
        the plotter will accept any number).
    figsize:
        Matplotlib figure size ``(width_inches, height_inches)``.
    interactive:
        If ``True``, call ``plt.ion()`` and pause after each ``update()``
        so the figure refreshes during a long-running simulation.  Set to
        ``False`` when calling ``update()`` only after the simulation has
        finished.
    """

    def __init__(
        self,
        n_scenarios: int = 1,
        figsize: tuple = (14, 10),
        interactive: bool = False,
    ) -> None:
        self._n_scenarios = n_scenarios
        self._interactive = interactive
        self._scenario_count = 0  # incremented by update()

        if interactive:
            plt.ion()

        self._fig: Figure
        self._axes: List[Axes]
        self._fig, axes_array = plt.subplots(2, 2, figsize=figsize)
        self._axes = [
            axes_array[0, 0],  # TN voltage
            axes_array[0, 1],  # DN voltage
            axes_array[1, 0],  # TSO objective
            axes_array[1, 1],  # DSO objective
        ]

        self._setup_axes()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _setup_axes(self) -> None:
        """Initialise axis labels, titles, and reference lines."""
        titles = [
            "Plant TN (EHV) Voltage Magnitudes",
            "Plant DN (HV) Voltage Magnitudes",
            "TSO Objective Value",
            "DSO Objective Value",
        ]
        xlabels = [
            "Iteration / Minute",
            "Iteration / Minute",
            "TSO Active Step",
            "DSO Active Step",
        ]
        ylabels = [
            "Voltage Magnitude [p.u.]",
            "Voltage Magnitude [p.u.]",
            "Objective Value [–]",
            "Objective Value [–]",
        ]

        for ax, title, xlabel, ylabel in zip(
            self._axes, titles, xlabels, ylabels
        ):
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        # Voltage limit reference lines
        for ax in self._axes[:2]:
            ax.axhline(
                0.95,
                colour="#B90F22",  # 9c – red
                linestyle=":",
                linewidth=1.0,
                label="V limit (0.95 / 1.05 p.u.)",
            )
            ax.axhline(
                1.05,
                colour="#B90F22",
                linestyle=":",
                linewidth=1.0,
            )

        self._fig.tight_layout(pad=2.0)

    def _colour_for_scenario(self, scenario_index: int) -> str:
        """Return the TU Darmstadt colour for the given scenario index."""
        return tu_colour(scenario_index)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        log: list,
        scenario_index: Optional[int] = None,
        label: Optional[str] = None,
        v_setpoint_pu: Optional[float] = None,
    ) -> None:
        """
        Add one scenario's results to the live plots.

        Parameters
        ----------
        log:
            List of ``IterationRecord`` objects as returned by
            ``run_cascade()`` or ``run_tso_voltage_control()``.
        scenario_index:
            Zero-based scenario index that selects the colour.  If
            ``None``, the index is auto-incremented.
        label:
            Legend label for this scenario.  If ``None`` and
            ``v_setpoint_pu`` is provided, the label is constructed
            automatically.
        v_setpoint_pu:
            Voltage setpoint used in the scenario [p.u.].  Used for the
            automatic label and for drawing a horizontal reference line
            in the TN voltage subplot.
        """
        if scenario_index is None:
            scenario_index = self._scenario_count
        self._scenario_count += 1

        colour = self._colour_for_scenario(scenario_index)

        if label is None:
            if v_setpoint_pu is not None:
                label = f"V_set = {v_setpoint_pu:.3f} p.u."
            else:
                label = f"Scenario {scenario_index}"

        # ----------------------------------------------------------------
        # Extract time-series from the log
        # ----------------------------------------------------------------
        minutes: List[int] = []
        tn_v_mean: List[float] = []
        tn_v_min: List[float] = []
        tn_v_max: List[float] = []
        dn_v_mean: List[float] = []
        dn_v_min: List[float] = []
        dn_v_max: List[float] = []

        tso_steps: List[int] = []
        tso_obj: List[float] = []

        dso_steps: List[int] = []
        dso_obj: List[float] = []

        for rec in log:
            # Determine iteration / minute index
            t = getattr(rec, "minute", getattr(rec, "iteration", None))
            if t is None:
                raise ValueError(
                    "IterationRecord has neither 'minute' nor 'iteration' attribute."
                )
            minutes.append(int(t))

            # TN voltages
            v_tn = getattr(rec, "plant_tn_voltages_pu", None)
            if v_tn is not None and len(v_tn) > 0:
                tn_v_mean.append(float(np.mean(v_tn)))
                tn_v_min.append(float(np.min(v_tn)))
                tn_v_max.append(float(np.max(v_tn)))
            else:
                tn_v_mean.append(float("nan"))
                tn_v_min.append(float("nan"))
                tn_v_max.append(float("nan"))

            # DN voltages
            v_dn = getattr(rec, "plant_dn_voltages_pu", None)
            if v_dn is not None and len(v_dn) > 0:
                dn_v_mean.append(float(np.mean(v_dn)))
                dn_v_min.append(float(np.min(v_dn)))
                dn_v_max.append(float(np.max(v_dn)))
            else:
                dn_v_mean.append(float("nan"))
                dn_v_min.append(float("nan"))
                dn_v_max.append(float("nan"))

            # TSO objective
            tso_active = getattr(rec, "tso_active", True)  # True for TSO-only runs
            tso_obj_val = getattr(rec, "tso_objective", None)
            if tso_active and tso_obj_val is not None:
                tso_steps.append(int(t))
                tso_obj.append(float(tso_obj_val))

            # DSO objective
            dso_active = getattr(rec, "dso_active", False)
            dso_obj_val = getattr(rec, "dso_objective", None)
            if dso_active and dso_obj_val is not None:
                dso_steps.append(int(t))
                dso_obj.append(float(dso_obj_val))

        t_arr = np.array(minutes, dtype=float)

        # ----------------------------------------------------------------
        # Subplot 0: TN voltage
        # ----------------------------------------------------------------
        ax_tn = self._axes[0]
        ax_tn.plot(
            t_arr,
            tn_v_mean,
            colour=colour,
            linewidth=1.8,
            label=label,
        )
        ax_tn.fill_between(
            t_arr,
            tn_v_min,
            tn_v_max,
            colour=colour,
            alpha=0.15,
        )
        if v_setpoint_pu is not None:
            ax_tn.axhline(
                v_setpoint_pu,
                colour=colour,
                linestyle="--",
                linewidth=1.0,
            )

        # ----------------------------------------------------------------
        # Subplot 1: DN voltage
        # ----------------------------------------------------------------
        ax_dn = self._axes[1]
        ax_dn.plot(
            t_arr,
            dn_v_mean,
            colour=colour,
            linewidth=1.8,
            label=label,
        )
        ax_dn.fill_between(
            t_arr,
            dn_v_min,
            dn_v_max,
            colour=colour,
            alpha=0.15,
        )

        # ----------------------------------------------------------------
        # Subplot 2: TSO objective
        # ----------------------------------------------------------------
        if tso_steps:
            ax_tso = self._axes[2]
            ax_tso.semilogy(
                np.array(tso_steps, dtype=float),
                np.abs(tso_obj),
                colour=colour,
                linewidth=1.8,
                marker="o",
                markersize=3,
                label=label,
            )

        # ----------------------------------------------------------------
        # Subplot 3: DSO objective
        # ----------------------------------------------------------------
        if dso_steps:
            ax_dso = self._axes[3]
            ax_dso.semilogy(
                np.array(dso_steps, dtype=float),
                np.abs(dso_obj),
                colour=colour,
                linewidth=1.8,
                marker="o",
                markersize=3,
                label=label,
            )

        # Refresh legends
        for ax in self._axes:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles,
                    labels,
                    fontsize=7,
                    loc="best",
                    framealpha=0.7,
                )

        if self._interactive:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.05)

    def show(self) -> None:
        """Render the final figure (blocking call)."""
        self._fig.tight_layout(pad=2.0)
        plt.show()

    def save(self, path: str, dpi: int = 150) -> None:
        """
        Save the current figure to a file.

        Parameters
        ----------
        path:
            Output file path, e.g. ``"results/cascade_results.pdf"``.
        dpi:
            Dots per inch for raster formats.
        """
        self._fig.tight_layout(pad=2.0)
        self._fig.savefig(path, dpi=dpi, bbox_inches="tight")

    @property
    def figure(self) -> Figure:
        """Return the underlying :class:`matplotlib.figure.Figure`."""
        return self._fig

    @property
    def axes(self) -> List[Axes]:
        """Return the list of :class:`matplotlib.axes.Axes` objects."""
        return self._axes
