"""
Multi-TSO/DSO Simulation Plotting
===================================

Live and post-run plotting for multi-zone TSO-DSO OFO results
(``run/run_multi_tso_dso.py``).

Produces two figures:

* **Figure 1 (TSO overview)**: Per-zone voltage bands, DER Q setpoints,
  generator AVR setpoints, and TSO objective per zone.
* **Figure 2 (DSO + stability)**: Per-feeder DSO interface Q tracking,
  DSO DER Q outputs, and per-zone contraction stability metric.

Also provides a :class:`MultiTSOLivePlotter` callback for real-time
visualisation during simulation.

Colour Palette
--------------
All data series use the official TU Darmstadt PANTONE palette in the order:
    5c, 1c, 8c, 6c, 3c, 10c, 4c, 2c, 9c, 7c, 11c

Author: Manuel Schwenke / Claude Code
Date: 2026-03-27
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from run.run_multi_tso_dso import MultiTSOIterationRecord

import os

import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, MultipleLocator, FuncFormatter

os.environ["QT_API"] = "pyqt5"
mpl.use("Qt5Agg")


# ─── TU Darmstadt PANTONE colour palette ────────────────────────────────────

#: Ordered colour sequence for all data series (5c, 1c, 8c, 6c, 3c, 10c, 4c, 2c, 9c, 7c, 11c).
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


def _c(index: int) -> str:
    """Return the TU Darmstadt PANTONE colour for a zero-based series index."""
    return TU_COLOURS[index % len(TU_COLOURS)]


# ─── x-axis formatting ───────────────────────────────────────────────────────


def _apply_x_fmt(ax: plt.Axes, sub_minute: bool = False) -> None:
    """Apply adaptive minute-resolution tick formatting to a time axis."""
    x_min, x_max = ax.get_xlim()
    if sub_minute:
        duration_min = (x_max - x_min) / 60.0
    else:
        duration_min = x_max - x_min

    if duration_min > 240:
        spacing_min = 60
    elif duration_min > 120:
        spacing_min = 30
    elif duration_min > 60:
        spacing_min = 15
    else:
        spacing_min = 3

    if sub_minute:
        ax.xaxis.set_major_locator(MultipleLocator(spacing_min * 60))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _pos: f'{int(round(x / 60))}')
        )
    else:
        ax.xaxis.set_major_locator(MultipleLocator(spacing_min))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _pos: f'{int(round(x))}')
        )


# =============================================================================
#  MultiTSOLivePlotter
# =============================================================================

class MultiTSOLivePlotter:
    """
    Real-time live plotter for the multi-TSO/DSO simulation.

    Call :meth:`update` once per simulation step with the
    :class:`~run.run_multi_tso_dso.MultiTSOIterationRecord` from that step.

    Two figure windows are created:

    * **Figure 1 (TSO)**: Zone voltage bands (V_min/V_max/V_mean), per-zone
      DER Q setpoints, generator AVR setpoints, and TSO objective per zone.
    * **Figure 2 (DSO/stability)**: DSO interface Q tracking (set vs actual),
      DSO DER Q outputs, and zone contraction stability metric over time.

    Parameters
    ----------
    zone_ids : sequence of int
        Sorted list of TSO zone IDs (e.g. [0, 1, 2]).
    dso_ids : sequence of str
        DSO controller ID strings (e.g. ['dso_zone2_0', 'dso_zone2_1']).
    v_setpoint_pu : float, optional
        Nominal voltage setpoint [p.u.] drawn as a reference line.
    v_min_pu, v_max_pu : float
        Hard voltage limits drawn as dashed red lines.
    sub_minute : bool
        If True, the x-axis unit is seconds; otherwise minutes.
    update_every : int
        Redraw DSO figure every this many ``update()`` calls.
    tso_update_every : int
        Redraw TSO figure every this many ``update()`` calls.
    """

    def __init__(
        self,
        zone_ids: Sequence[int],
        dso_ids: Sequence[str],
        *,
        v_setpoint_pu: float = 1.0,
        v_min_pu: float = 0.95,
        v_max_pu: float = 1.05,
        sub_minute: bool = False,
        update_every: int = 1,
        tso_update_every: int = 1,
    ) -> None:
        plt.ion()

        self._zone_ids: List[int] = list(zone_ids)
        self._dso_ids: List[str] = list(dso_ids)
        self._v_set = v_setpoint_pu
        self._v_min = v_min_pu
        self._v_max = v_max_pu
        self._sub_minute = sub_minute
        self._update_every = update_every
        self._tso_update_every = tso_update_every
        self._call_count = 0

        # ── Accumulated time series ───────────────────────────────────────────
        # Global time axis (one entry per update() call)
        self._minutes: List[float] = []

        # Per-zone TSO data (only at TSO-active steps)
        self._tso_min:    List[float] = []
        # zone_id → list of per-step arrays
        self._zone_v_min:  Dict[int, List[float]] = {z: [] for z in self._zone_ids}
        self._zone_v_max:  Dict[int, List[float]] = {z: [] for z in self._zone_ids}
        self._zone_v_mean: Dict[int, List[float]] = {z: [] for z in self._zone_ids}
        self._zone_q_der:  Dict[int, List] = {z: [] for z in self._zone_ids}
        self._zone_v_gen:  Dict[int, List] = {z: [] for z in self._zone_ids}
        self._zone_obj:    Dict[int, List[float]] = {z: [] for z in self._zone_ids}
        self._zone_obj_min: Dict[int, List[float]] = {z: [] for z in self._zone_ids}
        self._zone_contraction: Dict[int, List[float]] = {z: [] for z in self._zone_ids}

        # Per-DSO data (only at DSO-active steps)
        self._dso_min: List[float] = []
        self._dso_q_der: Dict[str, List] = {d: [] for d in self._dso_ids}
        self._dso_q_set:    Dict[str, List[float]] = {d: [] for d in self._dso_ids}
        self._dso_q_actual: Dict[str, List[float]] = {d: [] for d in self._dso_ids}
        self._dso_q_set_min:    Dict[str, List[float]] = {d: [] for d in self._dso_ids}
        self._dso_q_actual_min: Dict[str, List[float]] = {d: [] for d in self._dso_ids}
        self._dso_obj: Dict[str, List[float]] = {d: [] for d in self._dso_ids}

        # ── Build TSO figure ──────────────────────────────────────────────────
        # Rows: voltage bands | DER Q | V_gen | TSO objective
        _n_tso_rows = 4
        self._fig_tso, self._axes_tso = plt.subplots(
            _n_tso_rows, 1,
            figsize=(11, 2.8 * _n_tso_rows),
            sharex=True,
            constrained_layout=True,
        )
        self._fig_tso.suptitle("Multi-TSO Controller (live)", fontweight="bold")
        self._fig_tso.set_constrained_layout_pads(h_pad=0.05, hspace=0.05)

        ax = self._axes_tso[0]
        ax.set_ylabel("Voltage [p.u.]")
        ax.set_title("Zone Bus Voltages (V_min / V_mean / V_max bands)")
        ax.axhline(self._v_set, color="k", ls="--", lw=1.0, label=f"V_set={self._v_set:.3f}")
        ax.axhline(self._v_min, color="r", ls=":", lw=0.8, alpha=0.7)
        ax.axhline(self._v_max, color="r", ls=":", lw=0.8, alpha=0.7)
        ax.grid(True, alpha=0.3)
        self._ax_v = ax

        ax = self._axes_tso[1]
        ax.set_ylabel(r"$Q_\mathrm{DER}$ [Mvar]")
        ax.set_title("TSO DER Reactive Power per Zone")
        ax.grid(True, alpha=0.3)
        self._ax_qder = ax

        ax = self._axes_tso[2]
        ax.set_ylabel(r"$V_\mathrm{gen}$ [p.u.]")
        ax.set_title("Generator AVR Setpoints per Zone")
        ax.grid(True, alpha=0.3)
        self._ax_vgen = ax

        ax = self._axes_tso[3]
        ax.set_ylabel("Objective")
        ax.set_title("TSO Objective Value per Zone")
        ax.grid(True, alpha=0.3)
        self._ax_tso_obj = ax

        self._axes_tso[-1].set_xlabel(
            "Time [s]" if sub_minute else "Time [min]"
        )

        # ── Build DSO / stability figure ─────────────────────────────────────
        # Rows: interface Q tracking | DSO DER Q | contraction metric
        _n_dso_rows = 3
        self._fig_dso, self._axes_dso = plt.subplots(
            _n_dso_rows, 1,
            figsize=(11, 2.8 * _n_dso_rows),
            sharex=True,
            constrained_layout=True,
        )
        self._fig_dso.suptitle("Multi-DSO / Stability (live)", fontweight="bold")
        self._fig_dso.set_constrained_layout_pads(h_pad=0.05, hspace=0.05)

        ax = self._axes_dso[0]
        ax.set_ylabel(r"$Q$ [Mvar]")
        ax.set_title("TSO-DSO Interface Q  (set vs actual, load convention)")
        ax.grid(True, alpha=0.3)
        self._ax_iface = ax

        ax = self._axes_dso[1]
        ax.set_ylabel(r"$Q_\mathrm{DER}$ [Mvar]")
        ax.set_title("DSO DER Reactive Power per Feeder")
        ax.grid(True, alpha=0.3)
        self._ax_dso_qder = ax

        ax = self._axes_dso[2]
        ax.set_ylabel(r"Contraction $\|\cdot\|$")
        ax.set_title("Per-Zone Coupling Contraction Metric")
        ax.grid(True, alpha=0.3)
        self._ax_contraction = ax

        self._axes_dso[-1].set_xlabel(
            "Time [s]" if sub_minute else "Time [min]"
        )

        self._position_windows_side_by_side()

    # ── window placement ─────────────────────────────────────────────────────

    def _position_windows_side_by_side(self) -> None:
        """Position TSO window on the left half and DSO on the right half."""
        try:
            backend = mpl.get_backend()
            if "Qt" in backend:
                from PyQt5.QtWidgets import QApplication

                app = QApplication.instance()
                if app is not None:
                    screen = app.primaryScreen().availableGeometry()
                    half_w = screen.width() // 2
                    h = screen.height()
                    x0 = screen.x()
                    y0 = screen.y()

                    mgr_tso = self._fig_tso.canvas.manager
                    mgr_dso = self._fig_dso.canvas.manager
                    mgr_tso.window.setGeometry(x0, y0, half_w, h)
                    mgr_dso.window.setGeometry(x0 + half_w, y0, half_w, h)
        except Exception:
            pass

    # ── public API ───────────────────────────────────────────────────────────

    def update(self, rec: "MultiTSOIterationRecord") -> None:
        """Feed one MultiTSOIterationRecord and refresh the plots."""
        self._call_count += 1

        if self._sub_minute:
            t_val = float(rec.time_s)
        else:
            t_val = rec.time_s / 60.0
        self._minutes.append(t_val)

        # ── Accumulate TSO data ──────────────────────────────────────────────
        if rec.tso_active:
            self._tso_min.append(t_val)
            for z in self._zone_ids:
                self._zone_v_min[z].append(rec.zone_v_min.get(z, float("nan")))
                self._zone_v_max[z].append(rec.zone_v_max.get(z, float("nan")))
                self._zone_v_mean[z].append(rec.zone_v_mean.get(z, float("nan")))
                q_der = rec.zone_q_der.get(z)
                if q_der is not None:
                    self._zone_q_der[z].append(q_der)
                v_gen = rec.zone_v_gen.get(z)
                if v_gen is not None:
                    self._zone_v_gen[z].append(v_gen)
                obj = rec.zone_tso_objective.get(z)
                if obj is not None:
                    self._zone_obj[z].append(obj)
                    self._zone_obj_min[z].append(t_val)
                lhs = rec.zone_contraction_lhs.get(z)
                if lhs is not None and not (
                    isinstance(lhs, float) and (lhs != lhs)  # nan guard
                ):
                    self._zone_contraction[z].append(lhs)

        # ── Accumulate DSO data ──────────────────────────────────────────────
        if rec.dso_active:
            self._dso_min.append(t_val)
            for d in self._dso_ids:
                q_der_d = rec.dso_q_der.get(d)
                if q_der_d is not None:
                    self._dso_q_der[d].append(q_der_d)
                q_set = rec.dso_q_set_mvar.get(d) if rec.dso_q_set_mvar else None
                if q_set is not None:
                    self._dso_q_set[d].append(q_set)
                    self._dso_q_set_min[d].append(t_val)
                q_act = rec.dso_q_actual_mvar.get(d)
                if q_act is not None:
                    self._dso_q_actual[d].append(q_act)
                    self._dso_q_actual_min[d].append(t_val)
                obj_d = rec.dso_objective.get(d)
                if obj_d is not None:
                    self._dso_obj[d].append(obj_d)

        # ── Redraw ───────────────────────────────────────────────────────────
        _redrew = False
        if self._call_count % self._tso_update_every == 0:
            self._redraw_tso()
            _redrew = True
        if self._call_count % self._update_every == 0:
            self._redraw_dso()
            _redrew = True
        if _redrew:
            plt.pause(0.01)

    # ── internal redraw ──────────────────────────────────────────────────────

    def _redraw_tso(self) -> None:
        """Redraw the TSO figure."""
        tso_t = np.array(self._tso_min)

        # ── Zone voltage bands ───────────────────────────────────────────────
        ax = self._ax_v
        ax.clear()
        ax.set_ylabel("Voltage [p.u.]")
        ax.set_title("Zone Bus Voltages (V_min / V_mean / V_max bands)", pad=20)
        ax.axhline(self._v_set, color="k", ls="--", lw=1.0,
                   label=f"V_set={self._v_set:.3f}")
        ax.axhline(self._v_min, color="r", ls=":", lw=0.8, alpha=0.7)
        ax.axhline(self._v_max, color="r", ls=":", lw=0.8, alpha=0.7)
        ax.grid(True, alpha=0.3)
        _apply_x_fmt(ax, self._sub_minute)

        for zi, z in enumerate(self._zone_ids):
            v_min_arr = np.array(self._zone_v_min[z])
            v_max_arr = np.array(self._zone_v_max[z])
            v_mean_arr = np.array(self._zone_v_mean[z])
            t_arr = tso_t[:len(v_mean_arr)]
            if len(t_arr) == 0:
                continue
            col = _c(zi)
            ax.fill_between(t_arr, v_min_arr, v_max_arr,
                            alpha=0.15, color=col, linewidth=0)
            ax.plot(t_arr, v_mean_arr, lw=1.2, color=col,
                    label=f"Zone {z} V_mean")
            ax.plot(t_arr, v_min_arr, lw=0.6, color=col, ls="--", alpha=0.6)
            ax.plot(t_arr, v_max_arr, lw=0.6, color=col, ls="--", alpha=0.6)
        ax.legend(fontsize=7, ncol=4, loc="upper left")

        # ── TSO DER Q ────────────────────────────────────────────────────────
        ax = self._ax_qder
        ax.clear()
        ax.set_ylabel(r"$Q_\mathrm{DER}$ [Mvar]")
        ax.set_title("TSO DER Reactive Power per Zone")
        ax.grid(True, alpha=0.3)
        _apply_x_fmt(ax, self._sub_minute)

        for zi, z in enumerate(self._zone_ids):
            q_list = self._zone_q_der[z]
            if not q_list:
                continue
            q_arr = np.array(q_list)
            t_arr = tso_t[:len(q_arr)]
            for j in range(q_arr.shape[1]):
                lbl = f"Z{z}-DER{j}" if j == 0 else f"_Z{z}-DER{j}"
                ax.plot(t_arr, q_arr[:, j], lw=0.9,
                        color=_c(zi * 4 + j), alpha=0.85, label=lbl)
        if any(self._zone_q_der[z] for z in self._zone_ids):
            ax.legend(fontsize=7, ncol=4, loc="upper left")

        # ── Generator AVR setpoints ──────────────────────────────────────────
        ax = self._ax_vgen
        ax.clear()
        ax.set_ylabel(r"$V_\mathrm{gen}$ [p.u.]")
        ax.set_title("Generator AVR Setpoints per Zone")
        ax.grid(True, alpha=0.3)
        _apply_x_fmt(ax, self._sub_minute)

        for zi, z in enumerate(self._zone_ids):
            vg_list = self._zone_v_gen[z]
            if not vg_list:
                continue
            vg_arr = np.array(vg_list)
            t_arr = tso_t[:len(vg_arr)]
            for j in range(vg_arr.shape[1]):
                lbl = f"Z{z}-Gen{j}" if j == 0 else f"_Z{z}-Gen{j}"
                ax.plot(t_arr, vg_arr[:, j], lw=0.9,
                        color=_c(zi * 4 + j), alpha=0.85, label=lbl)
        if any(self._zone_v_gen[z] for z in self._zone_ids):
            ax.legend(fontsize=7, ncol=4, loc="upper left")

        # ── TSO objective ────────────────────────────────────────────────────
        ax = self._ax_tso_obj
        ax.clear()
        ax.set_ylabel("Objective")
        ax.set_title("TSO Objective Value per Zone")
        ax.grid(True, alpha=0.3)
        _apply_x_fmt(ax, self._sub_minute)

        has_obj = False
        for zi, z in enumerate(self._zone_ids):
            obj_list = self._zone_obj[z]
            if not obj_list:
                continue
            obj_arr = np.array(obj_list)
            t_arr = np.array(self._zone_obj_min[z])
            ax.plot(t_arr, obj_arr, lw=1.0, color=_c(zi),
                    marker=".", markersize=3, label=f"Zone {z}")
            has_obj = True
        if has_obj:
            try:
                ax.set_yscale("log")
            except Exception:
                pass
            ax.legend(fontsize=7, ncol=3, loc="upper right")

        self._fig_tso.canvas.draw_idle()
        self._fig_tso.canvas.flush_events()

    def _redraw_dso(self) -> None:
        """Redraw the DSO + stability figure."""
        dso_t = np.array(self._dso_min)

        # ── Interface Q tracking ─────────────────────────────────────────────
        ax = self._ax_iface
        ax.clear()
        ax.set_ylabel(r"$Q$ [Mvar]")
        ax.set_title("TSO-DSO Interface Q  (set vs actual, load convention)")
        ax.grid(True, alpha=0.3)
        _apply_x_fmt(ax, self._sub_minute)

        for di, d in enumerate(self._dso_ids):
            col_set = _c(di * 2)
            col_act = _c(di * 2 + 1)
            q_set_list = self._dso_q_set[d]
            q_act_list = self._dso_q_actual[d]
            if q_set_list:
                t_s = np.array(self._dso_q_set_min[d])
                ax.plot(t_s, np.array(q_set_list), lw=1.2, color=col_set,
                        ls="--", label=f"{d} set")
            if q_act_list:
                t_a = np.array(self._dso_q_actual_min[d])
                ax.plot(t_a, np.array(q_act_list), lw=1.2, color=col_act,
                        label=f"{d} actual")
        if any(self._dso_q_set[d] or self._dso_q_actual[d]
               for d in self._dso_ids):
            ax.legend(fontsize=7, ncol=4, loc="upper left")

        # ── DSO DER Q ────────────────────────────────────────────────────────
        ax = self._ax_dso_qder
        ax.clear()
        ax.set_ylabel(r"$Q_\mathrm{DER}$ [Mvar]")
        ax.set_title("DSO DER Reactive Power per Feeder")
        ax.grid(True, alpha=0.3)
        _apply_x_fmt(ax, self._sub_minute)

        for di, d in enumerate(self._dso_ids):
            q_list = self._dso_q_der[d]
            if not q_list:
                continue
            q_arr = np.array(q_list)
            t_arr = dso_t[:len(q_arr)]
            for j in range(q_arr.shape[1]):
                lbl = f"{d}-DER{j}" if j == 0 else f"_{d}-DER{j}"
                ax.plot(t_arr, q_arr[:, j], lw=0.9,
                        color=_c(di * 4 + j), alpha=0.85, label=lbl)
        if any(self._dso_q_der[d] for d in self._dso_ids):
            ax.legend(fontsize=7, ncol=4, loc="upper left")

        # ── Contraction metric ───────────────────────────────────────────────
        ax = self._ax_contraction
        ax.clear()
        ax.set_ylabel(r"Contraction LHS")
        ax.set_title("Per-Zone Coupling Contraction Metric")
        ax.grid(True, alpha=0.3)
        _apply_x_fmt(ax, self._sub_minute)

        tso_t = np.array(self._tso_min)
        has_contraction = False
        for zi, z in enumerate(self._zone_ids):
            c_list = self._zone_contraction[z]
            if not c_list:
                continue
            c_arr = np.array(c_list)
            t_arr = tso_t[:len(c_arr)]
            ax.plot(t_arr, c_arr, lw=1.0, color=_c(zi),
                    marker=".", markersize=3, label=f"Zone {z}")
            has_contraction = True
        if has_contraction:
            ax.legend(fontsize=7, ncol=3, loc="upper right")

        self._axes_dso[-1].set_xlabel(
            "Time [s]" if self._sub_minute else "Time [min]"
        )
        self._fig_dso.canvas.draw_idle()
        self._fig_dso.canvas.flush_events()


# =============================================================================
#  Post-run plot function
# =============================================================================

def plot_multi_tso(
    log: "List[MultiTSOIterationRecord]",
    zone_ids: Optional[List[int]] = None,
    dso_ids: Optional[List[str]] = None,
    *,
    v_setpoint_pu: float = 1.0,
    v_min_pu: float = 0.95,
    v_max_pu: float = 1.05,
    show: bool = True,
) -> tuple:
    """
    Post-run plot from a completed simulation log.

    Parameters
    ----------
    log : list of MultiTSOIterationRecord
    zone_ids : list of int, optional
        Zone IDs to include.  Inferred from log if None.
    dso_ids : list of str, optional
        DSO IDs to include.  Inferred from log if None.
    v_setpoint_pu, v_min_pu, v_max_pu : float
        Voltage reference lines.
    show : bool
        Call ``plt.show()`` after creating figures.

    Returns
    -------
    (fig_tso, fig_dso)
    """
    from typing import List as _List

    if not log:
        raise ValueError("log is empty")

    if zone_ids is None:
        zone_ids_set: set = set()
        for r in log:
            zone_ids_set.update(r.zone_v_mean.keys())
        zone_ids = sorted(zone_ids_set)

    if dso_ids is None:
        dso_ids_set: set = set()
        for r in log:
            dso_ids_set.update(r.dso_q_der.keys())
            dso_ids_set.update(r.dso_q_actual_mvar.keys())
        dso_ids = sorted(dso_ids_set)

    # Build a temporary plotter to reuse redraw logic
    plotter = MultiTSOLivePlotter(
        zone_ids,
        dso_ids,
        v_setpoint_pu=v_setpoint_pu,
        v_min_pu=v_min_pu,
        v_max_pu=v_max_pu,
        sub_minute=False,
        update_every=10**9,     # suppress incremental redraws
        tso_update_every=10**9,
    )

    for rec in log:
        plotter._call_count -= 1  # prevent auto-redraw in update()
        plotter.update(rec)

    plotter._redraw_tso()
    plotter._redraw_dso()

    if show:
        plt.ioff()
        plt.show()

    return plotter._fig_tso, plotter._fig_dso
