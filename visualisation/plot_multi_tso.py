"""
Multi-TSO/DSO Simulation Plotting
===================================

Live and post-run plotting for multi-zone TSO-DSO OFO results
(``run/run_M_TSO_M_DSO.py``).

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

if TYPE_CHECKING:
    from experiments.records import MultiTSOIterationRecord

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
        self._zone_q_gen:  Dict[int, List] = {z: [] for z in self._zone_ids}
        self._zone_oltc_taps: Dict[int, List] = {z: [] for z in self._zone_ids}
        self._zone_obj:    Dict[int, List[float]] = {z: [] for z in self._zone_ids}
        self._zone_obj_min: Dict[int, List[float]] = {z: [] for z in self._zone_ids}
        self._zone_contraction: Dict[int, List[float]] = {z: [] for z in self._zone_ids}

        # Per-DSO / network-group / transformer data (only at DSO-active steps)
        self._dso_min: List[float] = []

        self._group_ids: List[str] = []
        self._trafo_ids: List[str] = []

        self._dso_group_q_der: Dict[str, List[float]] = {}
        self._dso_group_q_der_min: Dict[str, List[float]] = {}
        self._dso_group_q_der_max: Dict[str, List[float]] = {}
        self._dso_group_v_min: Dict[str, List[float]] = {}
        self._dso_group_v_mean: Dict[str, List[float]] = {}
        self._dso_group_v_max: Dict[str, List[float]] = {}

        self._dso_trafo_group: Dict[str, str] = {}
        self._dso_trafo_q_set: Dict[str, List[float]] = {}
        self._dso_trafo_q_actual: Dict[str, List[float]] = {}
        self._dso_trafo_tap_pos: Dict[str, List[float]] = {}

        self._dso_trafo_q_set_t: Dict[str, List[float]] = {}
        self._dso_trafo_q_actual_t: Dict[str, List[float]] = {}
        self._dso_trafo_tap_t: Dict[str, List[float]] = {}

        # ── Build TSO figure ──────────────────────────────────────────────────
        # Subplots: Voltage | DER Q | V_gen | Q_gen | OLTC taps
        _n_tso_rows = 5
        self._fig_tso, self._axes_tso = plt.subplots(
            _n_tso_rows, 1,
            figsize=(11, 2.6 * _n_tso_rows),
            sharex=True,
            constrained_layout=True,
        )
        self._fig_tso.suptitle("Multi-TSO Controller (live)", fontweight="bold")
        self._fig_tso.set_constrained_layout_pads(h_pad=0.05, hspace=0.05)

        _row = 0
        ax = self._axes_tso[_row]
        ax.set_ylabel("Voltage [p.u.]")
        ax.set_title("Zone Bus Voltages (V_min / V_mean / V_max bands)")
        ax.axhline(self._v_set, color="k", ls="--", lw=1.0, label=f"V_set={self._v_set:.3f}")
        ax.grid(True, alpha=0.3)
        self._ax_v = ax

        _row += 1
        ax = self._axes_tso[_row]
        ax.set_ylabel(r"$Q_\mathrm{DER}$ [Mvar]")
        ax.set_title("TSO DER Reactive Power per Zone")
        ax.grid(True, alpha=0.3)
        self._ax_qder = ax

        _row += 1
        ax = self._axes_tso[_row]
        ax.set_ylabel(r"$V_\mathrm{gen}$ [p.u.]")
        ax.set_title("Generator AVR Setpoints per Zone")
        ax.grid(True, alpha=0.3)
        self._ax_vgen = ax

        _row += 1
        ax = self._axes_tso[_row]
        ax.set_ylabel(r"$Q_\mathrm{gen}$ [Mvar]")
        ax.set_title("Generator Reactive Power Injection per Zone")
        ax.grid(True, alpha=0.3)
        self._ax_qgen = ax

        _row += 1
        ax = self._axes_tso[_row]
        ax.set_ylabel("Tap pos.")
        ax.set_title("TSO Machine Transformer Tap Positions")
        ax.grid(True, alpha=0.3)
        self._ax_tso_oltc = ax

        # Preserved but inactive for now:
        self._ax_tso_obj = None
        
        self._axes_tso[-1].set_xlabel(
            "Time [s]" if sub_minute else "Time [min]"
        )

        # ── Build DSO / stability figure ─────────────────────────────────────
        # Layout (2N + 2 rows for N DSOs):
        #   rows 0..N-1   : interface Q (set vs actual) for each DSO
        #   row  N        : DSO DER reactive power per group + capability band
        #   rows N+1..2N  : transformer tap positions for each DSO
        #   row  2N+1     : DSO voltage bands per group
        n = len(self._dso_ids)
        if n < 1:
            raise ValueError("MultiTSOLivePlotter requires at least one DSO id.")

        self._dso_groups_sorted: List[str] = sorted(self._dso_ids)
        n_rows = 2 * n + 2
        height_ratios = [1.2] * n + [1.0] + [0.7] * n + [1.2]

        self._fig_dso, self._axes_dso = plt.subplots(
            n_rows, 1,
            figsize=(11, min(28.0, sum(height_ratios) * 2.2)),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        self._fig_dso.suptitle("Multi-DSO / Stability (live)", fontweight="bold")
        self._fig_dso.set_constrained_layout_pads(h_pad=0.05, hspace=0.05)

        # Slice the axes array into the four blocks. np.atleast_1d guards
        # against the degenerate single-axis case (n_rows == 1 cannot happen
        # here because n >= 1 implies n_rows >= 4, but be defensive).
        axes_arr = np.atleast_1d(self._axes_dso)
        self._ax_iface_list: List[plt.Axes] = list(axes_arr[:n])
        self._ax_dso_qder = axes_arr[n]
        self._ax_tap_list: List[plt.Axes] = list(axes_arr[n + 1 : 2 * n + 1])
        self._ax_dso_v = axes_arr[-1]

        self._ax_iface_by_group: Dict[str, plt.Axes] = dict(
            zip(self._dso_groups_sorted, self._ax_iface_list)
        )
        self._ax_tap_by_group: Dict[str, plt.Axes] = dict(
            zip(self._dso_groups_sorted, self._ax_tap_list)
        )

        # Static decoration so empty subplots are still informative.
        for g, ax in self._ax_iface_by_group.items():
            ax.set_title(f"TSO-DSO Interface Q — {g} (set vs actual)")
            ax.set_ylabel(r"$Q$ / Mvar")
            ax.grid(True, alpha=0.3)

        ax = self._ax_dso_qder
        ax.set_ylabel(r"$Q_\mathrm{DER}$ / Mvar")
        ax.set_title("DSO DER Reactive Power per HV Network Group (line: actual, band: DER capability)")
        ax.grid(True, alpha=0.3)

        for g, ax in self._ax_tap_by_group.items():
            ax.set_title(f"TSO-DSO Transformer Tap Position — {g}")
            ax.set_ylabel("Tap position")
            ax.grid(True, alpha=0.3)

        ax = self._ax_dso_v
        ax.set_ylabel("Voltage / p.u.")
        ax.set_title("DSO Voltages per HV Network Group (V_min / V_mean / V_max bands)")
        ax.axhline(self._v_set, color="k", ls="--", lw=1.0, label=f"V_set={self._v_set:.3f}")
        ax.grid(True, alpha=0.3)

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
                q_gen = rec.zone_q_gen.get(z)
                if q_gen is not None:
                    self._zone_q_gen[z].append(q_gen)
                oltc_taps = rec.zone_oltc_taps.get(z)
                if oltc_taps is not None:
                    self._zone_oltc_taps[z].append(oltc_taps)
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

            required_group_fields = (
                rec.dso_group_q_der_mvar,
                rec.dso_group_v_min_pu,
                rec.dso_group_v_mean_pu,
                rec.dso_group_v_max_pu,
                rec.dso_trafo_q_set_mvar,
                rec.dso_trafo_q_actual_mvar,
                rec.dso_trafo_tap_pos,
                rec.dso_trafo_group,
            )
            if any(field is None for field in required_group_fields):
                raise RuntimeError("DSO grouped plotting fields are missing in record.")

            # Optional fields for backward compatibility with replayed logs
            # that pre-date the DER capability bound recording.
            q_min_map = getattr(rec, "dso_group_q_der_min_mvar", None) or {}
            q_max_map = getattr(rec, "dso_group_q_der_max_mvar", None) or {}

            for group_id in sorted(rec.dso_group_q_der_mvar.keys()):
                if group_id not in rec.dso_group_v_min_pu:
                    raise KeyError(f"Missing dso_group_v_min_pu for group '{group_id}'.")
                if group_id not in rec.dso_group_v_mean_pu:
                    raise KeyError(f"Missing dso_group_v_mean_pu for group '{group_id}'.")
                if group_id not in rec.dso_group_v_max_pu:
                    raise KeyError(f"Missing dso_group_v_max_pu for group '{group_id}'.")

                if group_id not in self._ax_iface_by_group:
                    raise RuntimeError(
                        f"Group '{group_id}' was not in dso_ids passed at init "
                        f"({self._dso_groups_sorted!r}); the per-DSO split-row "
                        f"layout requires static groups."
                    )

                if group_id not in self._group_ids:
                    self._group_ids.append(group_id)
                    self._dso_group_q_der[group_id] = []
                    self._dso_group_q_der_min[group_id] = []
                    self._dso_group_q_der_max[group_id] = []
                    self._dso_group_v_min[group_id] = []
                    self._dso_group_v_mean[group_id] = []
                    self._dso_group_v_max[group_id] = []

                self._dso_group_q_der[group_id].append(float(rec.dso_group_q_der_mvar[group_id]))
                self._dso_group_q_der_min[group_id].append(
                    float(q_min_map.get(group_id, float("nan")))
                )
                self._dso_group_q_der_max[group_id].append(
                    float(q_max_map.get(group_id, float("nan")))
                )
                self._dso_group_v_min[group_id].append(float(rec.dso_group_v_min_pu[group_id]))
                self._dso_group_v_mean[group_id].append(float(rec.dso_group_v_mean_pu[group_id]))
                self._dso_group_v_max[group_id].append(float(rec.dso_group_v_max_pu[group_id]))

            for trafo_id, group_id in rec.dso_trafo_group.items():
                if trafo_id not in rec.dso_trafo_q_set_mvar:
                    raise KeyError(f"Missing q-setpoint for transformer '{trafo_id}'.")
                if trafo_id not in rec.dso_trafo_q_actual_mvar:
                    raise KeyError(f"Missing q-actual for transformer '{trafo_id}'.")
                if trafo_id not in rec.dso_trafo_tap_pos:
                    raise KeyError(f"Missing tap position for transformer '{trafo_id}'.")

                if trafo_id not in self._trafo_ids:
                    self._trafo_ids.append(trafo_id)
                    self._dso_trafo_group[trafo_id] = group_id
                    self._dso_trafo_q_set[trafo_id] = []
                    self._dso_trafo_q_actual[trafo_id] = []
                    self._dso_trafo_tap_pos[trafo_id] = []
                    self._dso_trafo_q_set_t[trafo_id] = []
                    self._dso_trafo_q_actual_t[trafo_id] = []
                    self._dso_trafo_tap_t[trafo_id] = []
                else:
                    if self._dso_trafo_group[trafo_id] != group_id:
                        raise ValueError(
                            f"Transformer '{trafo_id}' changed network group from "
                            f"'{self._dso_trafo_group[trafo_id]}' to '{group_id}'."
                        )

                self._dso_trafo_q_set[trafo_id].append(float(rec.dso_trafo_q_set_mvar[trafo_id]))
                self._dso_trafo_q_actual[trafo_id].append(float(rec.dso_trafo_q_actual_mvar[trafo_id]))
                self._dso_trafo_tap_pos[trafo_id].append(float(rec.dso_trafo_tap_pos[trafo_id]))
                self._dso_trafo_q_set_t[trafo_id].append(t_val)
                self._dso_trafo_q_actual_t[trafo_id].append(t_val)
                self._dso_trafo_tap_t[trafo_id].append(t_val)

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
        #ax.axhline(self._v_min, color="r", ls=":", lw=0.8, alpha=0.7)
        #ax.axhline(self._v_max, color="r", ls=":", lw=0.8, alpha=0.7)
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

        # ── TSO DER Q (only when TSO DERs are present) ──────────────────────
        if self._ax_qder is not None:
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

        # ── Generator Q injection ────────────────────────────────────────────
        ax = self._ax_qgen
        ax.clear()
        ax.set_ylabel(r"$Q_\mathrm{gen}$ [Mvar]")
        ax.set_title("Generator Reactive Power Injection per Zone")
        ax.grid(True, alpha=0.3)
        _apply_x_fmt(ax, self._sub_minute)

        for zi, z in enumerate(self._zone_ids):
            qg_list = self._zone_q_gen[z]
            if not qg_list:
                continue
            qg_arr = np.array(qg_list)
            t_arr = tso_t[: len(qg_arr)]
            for j in range(qg_arr.shape[1]):
                lbl = f"Z{z}-Gen{j}" if j == 0 else f"_Z{z}-Gen{j}"
                ax.plot(t_arr, qg_arr[:, j], lw=0.9,
                        color=_c(zi * 4 + j), alpha=0.85, label=lbl)
        if any(self._zone_q_gen[z] for z in self._zone_ids):
            ax.legend(fontsize=7, ncol=4, loc="upper left")

        # ── TSO machine transformer taps ─────────────────────────────────────
        ax = self._ax_tso_oltc
        ax.clear()
        ax.set_ylabel("Tap pos.")
        ax.set_title("TSO Machine Transformer Tap Positions")
        ax.grid(True, alpha=0.3)
        _apply_x_fmt(ax, self._sub_minute)

        for zi, z in enumerate(self._zone_ids):
            tap_list = self._zone_oltc_taps[z]
            if not tap_list:
                continue
            tap_arr = np.array(tap_list)
            t_arr = tso_t[:len(tap_arr)]
            for j in range(tap_arr.shape[1]):
                lbl = f"Z{z}-Tap{j}" if j == 0 else f"_Z{z}-Tap{j}"
                ax.step(t_arr, tap_arr[:, j], where="post", lw=1.0,
                        color=_c(zi * 4 + j), alpha=0.85, label=lbl)
        if any(self._zone_oltc_taps[z] for z in self._zone_ids):
            ax.legend(fontsize=7, ncol=4, loc="upper left")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # ── TSO objective (Preserved but commented out for now) ──────────────
        """
        ax = self._ax_tso_obj
        if ax is not None:
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
                ax.plot(t_arr, np.abs(obj_arr), lw=1.0, color=_c(zi),
                        marker=".", markersize=3, label=f"Zone {z}")
                has_obj = True
            if has_obj:
                try:
                    ax.set_yscale("log")
                except Exception:
                    pass
                ax.legend(fontsize=7, ncol=3, loc="upper right")
        """

        self._fig_tso.canvas.draw_idle()
        self._fig_tso.canvas.flush_events()

    def _redraw_dso(self) -> None:
        """Redraw the DSO / stability figure."""
        dso_t = np.array(self._dso_min)

        # Colour index uses the static init-time group ordering so that the
        # split subplots and the per-group line colours stay in sync across
        # redraws.
        group_colour_index = {
            g: i for i, g in enumerate(self._dso_groups_sorted)
        }

        # Pre-group transformer IDs by group for the per-DSO subplots.
        trafo_ids_by_group: Dict[str, List[str]] = {
            g: [] for g in self._dso_groups_sorted
        }
        for trafo_id in self._trafo_ids:
            if trafo_id not in self._dso_trafo_group:
                raise KeyError(f"Missing group assignment for transformer '{trafo_id}'.")
            g = self._dso_trafo_group[trafo_id]
            if g in trafo_ids_by_group:
                trafo_ids_by_group[g].append(trafo_id)

        # ------------------------------------------------------------------
        # Block A: interface Q per DSO (one subplot per DSO)
        # ------------------------------------------------------------------
        for gi, g in enumerate(self._dso_groups_sorted):
            ax = self._ax_iface_by_group[g]
            ax.clear()
            ax.set_title(f"TSO-DSO Interface Q — {g} (set vs actual)")
            ax.set_ylabel(r"$Q$ / Mvar")
            ax.grid(True, alpha=0.3)
            _apply_x_fmt(ax, self._sub_minute)

            col = _c(group_colour_index[g])

            for trafo_id in trafo_ids_by_group[g]:
                q_set = np.array(self._dso_trafo_q_set[trafo_id], dtype=float)
                q_act = np.array(self._dso_trafo_q_actual[trafo_id], dtype=float)
                t_set = np.array(self._dso_trafo_q_set_t[trafo_id], dtype=float)
                t_act = np.array(self._dso_trafo_q_actual_t[trafo_id], dtype=float)

                if q_set.size > 0:
                    ax.plot(t_set, q_set, lw=1.1, ls="--", color=col, alpha=0.9)
                if q_act.size > 0:
                    ax.plot(t_act, q_act, lw=1.3, ls="-", color=col, alpha=0.9)

            # Show the linestyle legend on the first interface subplot only.
            if gi == 0:
                style_handles = [
                    Line2D([0], [0], color="0.35", lw=1.1, ls="--", label="setpoint"),
                    Line2D([0], [0], color="0.35", lw=1.3, ls="-", label="actual"),
                ]
                ax.legend(handles=style_handles, fontsize=7, loc="upper left")

        # ------------------------------------------------------------------
        # Block B: DSO DER reactive power per group, with capability band
        # ------------------------------------------------------------------
        ax = self._ax_dso_qder
        ax.clear()
        ax.set_ylabel(r"$Q_\mathrm{DER}$ / Mvar")
        ax.set_title("DSO DER Reactive Power per HV Network Group (line: actual, band: DER capability)")
        ax.grid(True, alpha=0.3)
        _apply_x_fmt(ax, self._sub_minute)

        line_handles: List[Line2D] = []
        has_band = False
        for g in self._dso_groups_sorted:
            if g not in self._dso_group_q_der:
                continue  # group has not yet emitted any DSO step
            q_arr = np.array(self._dso_group_q_der[g], dtype=float)
            q_min = np.array(self._dso_group_q_der_min.get(g, []), dtype=float)
            q_max = np.array(self._dso_group_q_der_max.get(g, []), dtype=float)
            col = _c(group_colour_index[g])

            if q_arr.size > 0:
                t_line = dso_t[:len(q_arr)]
                line, = ax.plot(t_line, q_arr, lw=1.3, color=col, label=str(g))
                line_handles.append(line)

                # Shaded capability band where bound data is available. # ToDo: band was deactivated for now
                # m = min(len(q_arr), len(q_min), len(q_max))
                # if m > 0:
                #     t_band = dso_t[:m]
                #     ax.fill_between(
                #         t_band, q_min[:m], q_max[:m],
                #         color=col, alpha=0.15, linewidth=0.0,
                #     )
                #     has_band = True

        if line_handles:
            handles: List = list(line_handles)
            if has_band:
                handles.append(
                    Patch(facecolor="0.5", alpha=0.25, label="DER Q capability")
                )
            ax.legend(handles=handles, fontsize=7, ncol=3, loc="upper left")

        # ------------------------------------------------------------------
        # Block C: transformer tap positions per DSO (one subplot per DSO)
        # ------------------------------------------------------------------
        for g in self._dso_groups_sorted:
            ax = self._ax_tap_by_group[g]
            ax.clear()
            ax.set_title(f"TSO-DSO Transformer Tap Position — {g}")
            ax.set_ylabel("Tap position")
            ax.grid(True, alpha=0.3)
            _apply_x_fmt(ax, self._sub_minute)

            col = _c(group_colour_index[g])
            for trafo_id in trafo_ids_by_group[g]:
                tap_arr = np.array(self._dso_trafo_tap_pos[trafo_id], dtype=float)
                t_arr = np.array(self._dso_trafo_tap_t[trafo_id], dtype=float)
                if tap_arr.size > 0:
                    ax.step(
                        t_arr, tap_arr,
                        where="post",
                        lw=1.2, color=col,
                    )
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # ------------------------------------------------------------------
        # Row 4: DSO voltage min/mean/max per HV network group
        # ------------------------------------------------------------------
        ax = self._ax_dso_v
        ax.clear()
        ax.set_ylabel("Voltage [p.u.]")
        ax.set_title("DSO Voltages per HV Network Group (V_min / V_mean / V_max bands)")
        ax.axhline(self._v_set, color="k", ls="--", lw=1.0, label=f"V_set={self._v_set:.3f}")
        #ax.axhline(self._v_min, color="r", ls=":", lw=0.8, alpha=0.7)
        #ax.axhline(self._v_max, color="r", ls=":", lw=0.8, alpha=0.7)
        ax.grid(True, alpha=0.3)
        _apply_x_fmt(ax, self._sub_minute)

        has_v = False
        for group_id in self._dso_groups_sorted:
            if group_id not in self._dso_group_v_min:
                continue  # group not yet observed in any DSO step

            v_min = np.array(self._dso_group_v_min[group_id], dtype=float)
            v_mean = np.array(self._dso_group_v_mean[group_id], dtype=float)
            v_max = np.array(self._dso_group_v_max[group_id], dtype=float)
            t_arr = dso_t[:len(v_mean)]
            col = _c(group_colour_index[group_id])

            if v_mean.size > 0:
                ax.fill_between(t_arr, v_min, v_max, alpha=0.15, color=col, linewidth=0.0)
                ax.plot(t_arr, v_mean, lw=1.2, color=col, label=f"{group_id} V_mean")
                ax.plot(t_arr, v_min, lw=0.6, color=col, ls="--", alpha=0.6)
                ax.plot(t_arr, v_max, lw=0.6, color=col, ls="--", alpha=0.6)
                has_v = True

        if has_v:
            ax.legend(fontsize=7, ncol=3, loc="upper left")

        self._axes_dso[-1].set_xlabel("Time [s]" if self._sub_minute else "Time [min]")
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


# =============================================================================
#  Coordination comparison: coordinated vs. uncoordinated Q_PCC
# =============================================================================

# Zone colour mapping: Z1 → dark blue, Z2 → teal, Z3 → dark orange
_ZONE_COLOUR_IDX = {1: 1, 2: 4, 3: 2}
# DSO group colours start from magenta (5) and cycle
_DSO_COLOUR_START = 5


def _extract_comparison_series(
    log: "List[MultiTSOIterationRecord]",
    v_setpoint_pu: float,
) -> dict:
    """Extract time-series arrays from a simulation log for comparison plots.

    Returns a dict with numpy arrays keyed by metric name.
    """
    if not log:
        raise ValueError("log is empty")

    time_min = np.array([r.time_s / 60.0 for r in log])

    # Infer zone and DSO group IDs from the log
    zone_ids: list[int] = sorted(
        {z for r in log for z in r.zone_v_mean.keys()}
    )
    dso_group_ids: list[str] = sorted(
        {g for r in log for g in r.dso_group_v_min_pu.keys()}
    )

    n = len(log)

    # ── Per-zone EHV voltages (every step) ──────────────────────────────
    v_min: dict[int, np.ndarray] = {}
    v_max: dict[int, np.ndarray] = {}
    v_mean: dict[int, np.ndarray] = {}
    for z in zone_ids:
        v_min[z] = np.array([r.zone_v_min.get(z, np.nan) for r in log])
        v_max[z] = np.array([r.zone_v_max.get(z, np.nan) for r in log])
        v_mean[z] = np.array([r.zone_v_mean.get(z, np.nan) for r in log])

    # ── Per-DSO-group HV voltages (every step) ─────────────────────────
    gv_min: dict[str, np.ndarray] = {}
    gv_max: dict[str, np.ndarray] = {}
    for g in dso_group_ids:
        gv_min[g] = np.array([r.dso_group_v_min_pu.get(g, np.nan) for r in log])
        gv_max[g] = np.array([r.dso_group_v_max_pu.get(g, np.nan) for r in log])

    # ── Q_PCC setpoints (TSO-active steps, forward-filled) ─────────────
    q_pcc_sum: dict[int, np.ndarray] = {z: np.full(n, np.nan) for z in zone_ids}
    last_q: dict[int, float] = {}
    for i, r in enumerate(log):
        for z in zone_ids:
            arr = r.zone_q_pcc_set.get(z)
            if arr is not None and len(arr) > 0:
                last_q[z] = float(np.sum(arr))
            if z in last_q:
                q_pcc_sum[z][i] = last_q[z]

    # ── Generator Q infeed per zone (TSO-active steps, forward-filled) ─
    q_gen_sum: dict[int, np.ndarray] = {z: np.full(n, np.nan) for z in zone_ids}
    last_qg: dict[int, float] = {}
    for i, r in enumerate(log):
        for z in zone_ids:
            arr = r.zone_q_gen.get(z)
            if arr is not None and hasattr(arr, '__len__') and len(arr) > 0:
                last_qg[z] = float(np.sum(arr))
            if z in last_qg:
                q_gen_sum[z][i] = last_qg[z]

    # ── Q_PCC actual per DSO group (forward-filled) ────────────────────
    q_actual: dict[str, np.ndarray] = {g: np.full(n, np.nan) for g in dso_group_ids}
    last_qa: dict[str, float] = {}
    for i, r in enumerate(log):
        for g in dso_group_ids:
            val = r.dso_q_actual_mvar.get(g)
            if val is not None:
                last_qa[g] = float(val)
            if g in last_qa:
                q_actual[g][i] = last_qa[g]

    # ── Summary metrics (every step) ───────────────────────────────────
    v_rmsd = np.full(n, np.nan)
    v_maxdev = np.full(n, np.nan)
    for i in range(n):
        devs = []
        for z in zone_ids:
            vm = v_mean[z][i]
            vlo = v_min[z][i]
            vhi = v_max[z][i]
            if not np.isnan(vm):
                devs.append(vm - v_setpoint_pu)
            if not np.isnan(vlo):
                v_maxdev[i] = max(
                    v_maxdev[i] if not np.isnan(v_maxdev[i]) else 0.0,
                    abs(vlo - v_setpoint_pu),
                    abs(vhi - v_setpoint_pu),
                )
        if devs:
            v_rmsd[i] = np.sqrt(np.mean(np.array(devs) ** 2))

    return dict(
        time_min=time_min,
        zone_ids=zone_ids,
        dso_group_ids=dso_group_ids,
        v_min=v_min,
        v_max=v_max,
        v_mean=v_mean,
        gv_min=gv_min,
        gv_max=gv_max,
        q_pcc_sum=q_pcc_sum,
        q_gen_sum=q_gen_sum,
        q_actual=q_actual,
        v_rmsd=v_rmsd,
        v_maxdev=v_maxdev,
    )


def _add_contingency_shading(
    axes,
    contingencies: "Optional[List]" = None,
) -> None:
    """Add grey vertical spans for each trip/restore contingency pair."""
    if not contingencies:
        return
    # Pair trips with restores by (element_type, element_index)
    trips: dict[tuple, float] = {}
    pairs: list[tuple[float, float]] = []
    for ev in contingencies:
        key = (ev.element_type, ev.element_index)
        t_min = ev.effective_time_s / 60.0
        if ev.action == "trip":
            trips[key] = t_min
        elif ev.action == "restore" and key in trips:
            pairs.append((trips.pop(key), t_min))
    # Add spans for unrestored trips extending to the right edge
    for key, t_trip in trips.items():
        pairs.append((t_trip, None))

    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        for t_start, t_end in pairs:
            if t_end is not None:
                ax.axvspan(t_start, t_end, color='grey', alpha=0.08, zorder=0)
            else:
                ax.axvline(t_start, color='grey', alpha=0.3, linestyle=':', zorder=0)


def plot_coordination_comparison(
    log_a: "List[MultiTSOIterationRecord]",
    log_b: "List[MultiTSOIterationRecord]",
    *,
    label_a: str = "Coordinated",
    label_b: str = "Uncoordinated",
    v_setpoint_pu: float = 1.03,
    v_min_pu: float = 0.95,
    v_max_pu: float = 1.05,
    contingencies: "Optional[List]" = None,
    gen_info: "Optional[List[dict]]" = None,
    show: bool = True,
) -> "tuple[plt.Figure, ...]":
    """Compare two simulation logs: coordinated vs. uncoordinated Q_PCC control.

    Parameters
    ----------
    log_a, log_b : list of MultiTSOIterationRecord
        Simulation logs for Scenario A (coordinated) and B (uncoordinated).
    label_a, label_b : str
        Legend labels for the two scenarios.
    v_setpoint_pu : float
        Voltage setpoint for reference lines.
    v_min_pu, v_max_pu : float
        Voltage limits for reference lines.
    contingencies : list, optional
        ContingencyEvent list for shading trip/restore periods.
    gen_info : list of dict, optional
        Per-generator capability data. Each dict has keys: zone, gen_idx,
        name, max_p_mw, min_p_mw, max_q_mvar, min_q_mvar.
    show : bool
        Call ``plt.show()`` after creating figures.

    Returns
    -------
    tuple of figures (fig_voltage, fig_metrics[, fig_capability])
    """
    da = _extract_comparison_series(log_a, v_setpoint_pu)
    db = _extract_comparison_series(log_b, v_setpoint_pu)

    zone_ids = da["zone_ids"]
    dso_group_ids = da["dso_group_ids"]

    # =====================================================================
    #  Figure 1: Voltage & Q_PCC Comparison
    #  Layout: N_zone EHV rows + 1 HV + 1 Q_PCC set + 1 Q_PCC act + 1 Q_gen
    # =====================================================================
    n_z = len(zone_ids)
    n_rows_fig1 = n_z + 4  # per-zone EHV + HV + Q_set + Q_act + Q_gen
    # Give EHV zone rows slightly less height than the other panels
    height_ratios = [1.0] * n_z + [1.2, 1.0, 1.0, 1.0]
    fig1, axes1 = plt.subplots(
        n_rows_fig1, 1,
        figsize=(14, 2.5 * n_z + 10),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig1.suptitle("Coordinated vs. Uncoordinated Q$_{PCC}$ Control",
                  fontweight="bold", fontsize=13)

    col_a_style = TU_COLOURS[1]  # dark blue for scenario A
    col_b_style = TU_COLOURS[2]  # dark orange for scenario B

    # ── Rows 0..N_z-1: EHV voltage envelope, one subplot per zone ──────
    for iz, z in enumerate(zone_ids):
        ax = axes1[iz]
        # Scenario A: filled blue band
        ax.fill_between(da["time_min"], da["v_min"][z], da["v_max"][z],
                        color=col_a_style, alpha=0.15, step="post")
        ax.plot(da["time_min"], da["v_min"][z], color=col_a_style,
                linewidth=1.0, drawstyle="steps-post")
        ax.plot(da["time_min"], da["v_max"][z], color=col_a_style,
                linewidth=1.0, drawstyle="steps-post")
        # Scenario B: filled orange band
        ax.fill_between(db["time_min"], db["v_min"][z], db["v_max"][z],
                        color=col_b_style, alpha=0.15, step="post")
        ax.plot(db["time_min"], db["v_min"][z], color=col_b_style,
                linewidth=1.0, linestyle="--", drawstyle="steps-post")
        ax.plot(db["time_min"], db["v_max"][z], color=col_b_style,
                linewidth=1.0, linestyle="--", drawstyle="steps-post")
        ax.axhline(v_setpoint_pu, color="black", linewidth=0.6, linestyle=":")
        ax.axhline(v_min_pu, color="red", linewidth=0.6, linestyle="--",
                   alpha=0.5)
        ax.axhline(v_max_pu, color="red", linewidth=0.6, linestyle="--",
                   alpha=0.5)
        ax.set_ylabel(f"Zone {z} V [p.u.]")
        ax.grid(True, alpha=0.3)
        if iz == 0:
            ax.legend(
                handles=[
                    Patch(facecolor=col_a_style, alpha=0.3, label=label_a),
                    Patch(facecolor=col_b_style, alpha=0.3, label=label_b),
                ],
                loc="upper right", fontsize=8,
            )

    # ── Next row: HV voltage per DSO group ──────────────────────────────
    ax = axes1[n_z]
    legend_handles = []
    for i, g in enumerate(dso_group_ids):
        ci = (_DSO_COLOUR_START + i) % len(TU_COLOURS)
        col = TU_COLOURS[ci]
        ax.fill_between(da["time_min"], da["gv_min"][g], da["gv_max"][g],
                        color=col, alpha=0.25, step="post")
        ax.plot(da["time_min"], da["gv_min"][g], color=col, linewidth=0.8,
                drawstyle="steps-post")
        ax.plot(da["time_min"], da["gv_max"][g], color=col, linewidth=0.8,
                drawstyle="steps-post")
        ax.plot(db["time_min"], db["gv_min"][g], color=col, linewidth=1.0,
                linestyle="--", drawstyle="steps-post")
        ax.plot(db["time_min"], db["gv_max"][g], color=col, linewidth=1.0,
                linestyle="--", drawstyle="steps-post")
        legend_handles.append(Patch(facecolor=col, alpha=0.4, label=g))
    ax.axhline(v_setpoint_pu, color="black", linewidth=0.6, linestyle=":")
    ax.axhline(v_min_pu, color="red", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axhline(v_max_pu, color="red", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=2)
    ax.set_ylabel("HV Voltage [p.u.]")
    ax.grid(True, alpha=0.3)

    # ── Q_PCC setpoint per zone (sum) ───────────────────────────────────
    ax = axes1[n_z + 1]
    for z in zone_ids:
        ci = _ZONE_COLOUR_IDX.get(z, z % len(TU_COLOURS))
        col = TU_COLOURS[ci]
        ax.plot(da["time_min"], da["q_pcc_sum"][z], color=col, linewidth=1.2,
                drawstyle="steps-post", label=f"Z{z} {label_a}")
        ax.plot(db["time_min"], db["q_pcc_sum"][z], color=col, linewidth=1.2,
                linestyle="--", drawstyle="steps-post", label=f"Z{z} {label_b}")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylabel("$Q_{PCC}^{set}$ [Mvar]")
    ax.grid(True, alpha=0.3)

    # ── Q_PCC actual per DSO group ──────────────────────────────────────
    ax = axes1[n_z + 2]
    for i, g in enumerate(dso_group_ids):
        ci = (_DSO_COLOUR_START + i) % len(TU_COLOURS)
        col = TU_COLOURS[ci]
        ax.plot(da["time_min"], da["q_actual"][g], color=col, linewidth=1.2,
                drawstyle="steps-post", label=f"{g} {label_a}")
        ax.plot(db["time_min"], db["q_actual"][g], color=col, linewidth=1.2,
                linestyle="--", drawstyle="steps-post", label=f"{g} {label_b}")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylabel("$Q_{PCC}^{actual}$ [Mvar]")
    ax.grid(True, alpha=0.3)

    # ── Generator Q infeed per zone (sum) ───────────────────────────────
    ax = axes1[n_z + 3]
    for z in zone_ids:
        ci = _ZONE_COLOUR_IDX.get(z, z % len(TU_COLOURS))
        col = TU_COLOURS[ci]
        ax.plot(da["time_min"], da["q_gen_sum"][z], color=col, linewidth=1.2,
                drawstyle="steps-post", label=f"Z{z} {label_a}")
        ax.plot(db["time_min"], db["q_gen_sum"][z], color=col, linewidth=1.2,
                linestyle="--", drawstyle="steps-post", label=f"Z{z} {label_b}")
    ax.axhline(0, color="black", linewidth=0.4, linestyle=":")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylabel("$\\Sigma Q_{gen}$ [Mvar]")
    ax.set_xlabel("Time [min]")
    ax.grid(True, alpha=0.3)

    # Contingency shading + x-axis formatting
    _add_contingency_shading(axes1, contingencies)
    for a in axes1:
        _apply_x_fmt(a)

    # =====================================================================
    #  Figure 2: Summary Metrics (2 rows)
    # =====================================================================
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                               constrained_layout=True)
    fig2.suptitle("Voltage Quality Metrics", fontweight="bold", fontsize=13)

    col_a = TU_COLOURS[1]  # dark blue
    col_b = TU_COLOURS[2]  # dark orange

    # ── Row 0: Voltage RMSD from setpoint ───────────────────────────────
    ax = axes2[0]
    ax.plot(da["time_min"], da["v_rmsd"], color=col_a, linewidth=1.5,
            label=label_a)
    ax.plot(db["time_min"], db["v_rmsd"], color=col_b, linewidth=1.5,
            linestyle="--", label=label_b)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylabel("Voltage RMSD [p.u.]")
    ax.grid(True, alpha=0.3)

    # ── Row 1: Max voltage deviation ────────────────────────────────────
    ax = axes2[1]
    ax.plot(da["time_min"], da["v_maxdev"], color=col_a, linewidth=1.5,
            label=label_a)
    ax.plot(db["time_min"], db["v_maxdev"], color=col_b, linewidth=1.5,
            linestyle="--", label=label_b)
    ax.axhline(0.05, color="red", linewidth=0.6, linestyle="--", alpha=0.5,
               label="5% limit")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylabel("Max $|\\Delta V|$ [p.u.]")
    ax.set_xlabel("Time [min]")
    ax.grid(True, alpha=0.3)

    # Contingency shading + x-axis formatting
    _add_contingency_shading(axes2, contingencies)
    for a in axes2:
        _apply_x_fmt(a)

    # =====================================================================
    #  Figure 3: Generator P-Q capability scatter (if gen_info provided)
    #  Capability curve: Milano (2010) §12.2.1 — three thermal constraints
    #    (i)   Stator:  p² + q² ≤ s_max²
    #    (ii)  Rotor:   p² + (q + v²/xd)² ≤ (v·i_f_max/xd)²
    #    (iii) Under-excitation:  q ≥ −q₀(v) + β·p_max
    # =====================================================================
    fig3 = None
    if gen_info:
        from core.actuator_bounds import GeneratorParameters, compute_generator_q_limits

        # Group generators by zone for indexing
        gens_by_zone: dict[int, list[dict]] = {}
        for gi in gen_info:
            gens_by_zone.setdefault(gi["zone"], []).append(gi)

        n_gens = len(gen_info)
        ncols = min(n_gens, 4)
        nrows = (n_gens + ncols - 1) // ncols
        fig3, axes3 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                                   constrained_layout=True, squeeze=False)
        fig3.suptitle("Generator P-Q Operating Points vs. Capability (Milano §12.2)",
                      fontweight="bold", fontsize=13)

        gen_flat_idx = 0
        for z in sorted(gens_by_zone.keys()):
            zone_gens = gens_by_zone[z]
            for k, gi in enumerate(zone_gens):
                row, col_idx = divmod(gen_flat_idx, ncols)
                ax = axes3[row, col_idx]

                gp = GeneratorParameters(
                    s_rated_mva=gi["s_rated_mva"],
                    p_max_mw=gi["p_max_mw"],
                    xd_pu=gi["xd_pu"],
                    i_f_max_pu=gi["i_f_max_pu"],
                    beta=gi["beta"],
                    q0_pu=gi["q0_pu"],
                )
                s_base = gi["s_rated_mva"]
                v = 1.0  # nominal voltage for capability envelope

                # ── Individual constraint curves (dashed, for reference) ──
                # Sweep P from 0 to S_rated to show full stator circle
                p_full = np.linspace(0, s_base, 300)
                p_pu = p_full / s_base

                # (i) Stator current limit: p^2 + q^2 <= 1
                disc_s = np.maximum(1.0 - p_pu**2, 0.0)
                q_stator_hi = np.sqrt(disc_s) * s_base
                q_stator_lo = -q_stator_hi

                # (ii) Rotor current limit: p^2 + (q + v^2/xd)^2 <= (v*if/xd)^2
                xd = gi["xd_pu"]
                i_f = gi["i_f_max_pu"]
                rotor_r = v * i_f / xd
                rotor_c = -v**2 / xd  # center in q (p.u.)
                disc_r = np.maximum(rotor_r**2 - p_pu**2, 0.0)
                q_rotor_hi = (rotor_c + np.sqrt(disc_r)) * s_base

                # (iii) Under-excitation: q >= -q0*v^2 + beta*p_max_pu
                p_max_pu = gi["p_max_mw"] / s_base
                q_ue = (-gi["q0_pu"] * v**2 + gi["beta"] * p_max_pu) * s_base

                # Plot individual constraints as thin dashed lines
                ax.plot(p_full, q_stator_hi, color="grey", linewidth=0.7,
                        linestyle=":", zorder=0, label="Stator limit")
                ax.plot(p_full, q_stator_lo, color="grey", linewidth=0.7,
                        linestyle=":", zorder=0)
                ax.plot(p_full, q_rotor_hi, color="grey", linewidth=0.7,
                        linestyle="-.", zorder=0, label="Rotor limit")
                ax.axhline(q_ue, color="grey", linewidth=0.7,
                           linestyle="--", zorder=0, label="UE limit")

                # ── Composite capability envelope (p_min to p_max) ────────
                p_min = gi.get("p_min_mw", 0.0)
                p_sweep = np.linspace(p_min, gi["p_max_mw"], 300)
                q_lo_cap = np.empty_like(p_sweep)
                q_hi_cap = np.empty_like(p_sweep)
                for ip, pp in enumerate(p_sweep):
                    q_lo_cap[ip], q_hi_cap[ip] = compute_generator_q_limits(
                        gp, p_mw=pp, v_pu=v,
                    )

                # Fill the feasible region
                ax.fill_between(p_sweep, q_lo_cap, q_hi_cap,
                                color="lightgrey", alpha=0.3, zorder=1,
                                label="Feasible region")
                ax.plot(p_sweep, q_hi_cap, color="black", linewidth=1.2,
                        zorder=1)
                ax.plot(p_sweep, q_lo_cap, color="black", linewidth=1.2,
                        zorder=1)
                # Vertical lines at P_min and P_max (turbine limits)
                if p_min > 0:
                    ax.axvline(p_min, color="black", linewidth=0.7,
                               linestyle="--", alpha=0.5, zorder=1)
                ax.axvline(gi["p_max_mw"], color="black", linewidth=0.7,
                           linestyle="--", alpha=0.5, zorder=1)

                # Collect operating points for this generator
                p_a, q_a = [], []
                p_b, q_b = [], []
                for r in log_a:
                    pv = r.zone_p_gen.get(z)
                    qv = r.zone_q_gen.get(z)
                    if pv is not None and qv is not None and k < len(pv):
                        p_a.append(float(pv[k]))
                        q_a.append(float(qv[k]))
                for r in log_b:
                    pv = r.zone_p_gen.get(z)
                    qv = r.zone_q_gen.get(z)
                    if pv is not None and qv is not None and k < len(pv):
                        p_b.append(float(pv[k]))
                        q_b.append(float(qv[k]))

                # Scatter: Scenario A vs B
                if p_a:
                    ax.scatter(p_a, q_a, color=TU_COLOURS[1], alpha=0.15,
                               s=12, label=label_a, zorder=2, edgecolors="none")
                if p_b:
                    ax.scatter(p_b, q_b, color=TU_COLOURS[2], alpha=0.15,
                               s=12, label=label_b, zorder=2, edgecolors="none")

                ax.set_title(f"{gi['name']} (Z{z})", fontsize=10)
                ax.set_xlabel("P [MW]")
                ax.set_ylabel("Q [Mvar]")
                ax.axhline(0, color="black", linewidth=0.3, linestyle=":")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7, loc="best")

                gen_flat_idx += 1

        # Hide unused subplots
        for idx in range(gen_flat_idx, nrows * ncols):
            row, col_idx = divmod(idx, ncols)
            axes3[row, col_idx].set_visible(False)

    if show:
        plt.ioff()
        plt.show()

    if fig3 is not None:
        return fig1, fig2, fig3
    return fig1, fig2


# =============================================================================
#  LoadBalanceLivePlotter
# =============================================================================

class LoadBalanceLivePlotter:
    """
    Real-time live plotter showing system-wide load balance over time.

    Plots:
      - **Top**: Active power [MW] — total load, DER infeed, generator infeed,
        residual load (load − DER), with original IEEE 39 base-case load as
        a dashed reference line.
      - **Bottom**: Reactive power [Mvar] — total load Q and total generator Q,
        with original IEEE 39 base-case Q as a dashed reference line.

    Parameters
    ----------
    original_load_p_mw : float
        Original IEEE 39-bus total active load (base_p_mw sum) [MW].
    original_load_q_mvar : float
        Original IEEE 39-bus total reactive load (base_q_mvar sum) [Mvar].
    sub_minute : bool
        If True, x-axis unit is seconds; otherwise minutes.
    update_every : int
        Redraw every N calls to :meth:`update`.
    """

    def __init__(
        self,
        original_load_p_mw: float,
        original_load_q_mvar: float,
        *,
        sub_minute: bool = False,
        update_every: int = 1,
    ) -> None:
        plt.ion()

        self._orig_p = original_load_p_mw
        self._orig_q = original_load_q_mvar
        self._sub_minute = sub_minute
        self._update_every = update_every
        self._call_count = 0

        # Time series storage
        self._t: List[float] = []
        self._load_p: List[float] = []
        self._load_q: List[float] = []
        self._sgen_p: List[float] = []
        self._gen_p: List[float] = []
        self._gen_q: List[float] = []
        self._residual_p: List[float] = []

        # Build figure: 2 subplots (P and Q)
        self._fig, self._axes = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True, constrained_layout=True,
        )
        self._fig.suptitle("System Load Balance (live)", fontweight="bold")

        # ── Top: Active power ────────────────────────────────────────────
        ax_p = self._axes[0]
        ax_p.set_ylabel("Active Power [MW]")
        ax_p.set_title("System P Balance: Load, DER, Generators, Residual")
        ax_p.axhline(self._orig_p, color="k", ls=":", lw=1.2, alpha=0.6,
                      label=f"IEEE 39 base load ({self._orig_p:.0f} MW)")
        ax_p.grid(True, alpha=0.3)
        self._ax_p = ax_p

        # ── Bottom: Reactive power ───────────────────────────────────────
        ax_q = self._axes[1]
        ax_q.set_ylabel("Reactive Power [Mvar]")
        ax_q.set_title("System Q Balance: Load Q and Generator Q")
        ax_q.axhline(self._orig_q, color="k", ls=":", lw=1.2, alpha=0.6,
                      label=f"IEEE 39 base Q ({self._orig_q:.0f} Mvar)")
        ax_q.grid(True, alpha=0.3)
        ax_q.set_xlabel("Time [s]" if sub_minute else "Time [min]")
        self._ax_q = ax_q

        self._fig.show()

    def update(self, rec: "MultiTSOIterationRecord") -> None:
        """Append one timestep and redraw."""
        self._call_count += 1

        t = rec.time_s / 60.0 if not self._sub_minute else rec.time_s
        self._t.append(t)
        self._load_p.append(rec.total_load_p_mw)
        self._load_q.append(rec.total_load_q_mvar)
        self._sgen_p.append(rec.total_sgen_p_mw)
        self._gen_p.append(rec.total_gen_p_mw)
        self._gen_q.append(rec.total_gen_q_mvar)
        self._residual_p.append(rec.residual_load_p_mw)

        if self._call_count % self._update_every != 0:
            return

        # ── Redraw P subplot ─────────────────────────────────────────────
        ax = self._ax_p
        ax.clear()
        ax.set_ylabel("Active Power [MW]")
        ax.set_title("System P Balance: Load, DER, Generators, Residual")
        ax.axhline(self._orig_p, color="k", ls=":", lw=1.2, alpha=0.6,
                    label=f"IEEE 39 base load ({self._orig_p:.0f} MW)")
        ax.plot(self._t, self._load_p,     color=_c(1), lw=1.5, label="Total load P")
        ax.plot(self._t, self._sgen_p,     color=_c(0), lw=1.5, label="DER infeed P")
        ax.plot(self._t, self._gen_p,      color=_c(2), lw=1.5, label="Generator P")
        ax.plot(self._t, self._residual_p, color=_c(4), lw=1.5, ls="--",
                label="Residual (load-DER)")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        _apply_x_fmt(ax, self._sub_minute)

        # ── Redraw Q subplot ─────────────────────────────────────────────
        ax = self._ax_q
        ax.clear()
        ax.set_ylabel("Reactive Power [Mvar]")
        ax.set_title("System Q Balance: Load Q and Generator Q")
        ax.axhline(self._orig_q, color="k", ls=":", lw=1.2, alpha=0.6,
                    label=f"IEEE 39 base Q ({self._orig_q:.0f} Mvar)")
        ax.plot(self._t, self._load_q, color=_c(1), lw=1.5, label="Total load Q")
        ax.plot(self._t, self._gen_q,  color=_c(2), lw=1.5, label="Generator Q")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time [s]" if self._sub_minute else "Time [min]")
        _apply_x_fmt(ax, self._sub_minute)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
