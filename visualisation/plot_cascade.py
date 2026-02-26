"""
Cascade Simulation Plotting
=============================

Post-simulation plotting for cascaded TSO-DSO OFO results.
Produces two figures:
  - **Figure 1 (TSO)**: EHV voltage trajectories + setpoints, TSO control
    inputs (DER Q, Q_gen, V_gen, OLTC taps), and TSO objective value.
  - **Figure 2 (DSO)**: DN voltage trajectories with limits, TSO-DSO interface
    Q tracking, DSO control inputs (DER Q, OLTC taps, shunt states), and DSO
    objective value.

Also provides a :class:`LivePlotter` callback for real-time visualisation
during simulation.

Author: Manuel Schwenke
Date: 2026-02-13
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from run_cascade import IterationRecord
    from controller.tso_controller import TSOControllerConfig
    from controller.dso_controller import DSOControllerConfig

import matplotlib as mpl
import os
os.environ["QT_API"] = "pyqt5"
mpl.use('Qt5Agg')


# ─── helpers ────────────────────────────────────────────────────────────────

def _collect_series(log: List[IterationRecord]):
    """Extract time series arrays from iteration log."""
    minutes = np.array([r.minute for r in log])

    # Plant voltages (available every minute after PF)
    tn_v = np.array([r.plant_tn_voltages_pu for r in log])  # (T, n_tn_buses)
    dn_v = np.array([r.plant_dn_voltages_pu for r in log])  # (T, n_dn_buses)

    # TSO series (only at TSO-active minutes)
    tso_mask = np.array([r.tso_active and r.tso_q_der_mvar is not None
                         for r in log])
    tso_min = minutes[tso_mask]
    tso_recs = [r for r in log if r.tso_active and r.tso_q_der_mvar is not None]

    tso_q_der = np.array([r.tso_q_der_mvar for r in tso_recs])
    tso_q_pcc = np.array([r.tso_q_pcc_set_mvar for r in tso_recs])
    tso_v_gen = np.array([r.tso_v_gen_pu for r in tso_recs])
    tso_oltc = np.array([r.tso_oltc_taps for r in tso_recs]) if tso_recs and tso_recs[0].tso_oltc_taps is not None and len(tso_recs[0].tso_oltc_taps) > 0 else None
    tso_shunt = np.array([r.tso_shunt_states for r in tso_recs]) if tso_recs and tso_recs[0].tso_shunt_states is not None and len(tso_recs[0].tso_shunt_states) > 0 else None
    tso_obj = np.array([r.tso_objective for r in tso_recs # ToDo: Manually disabled
                        if r.tso_objective is not None])
    tso_obj_min = np.array([r.minute for r in tso_recs
                            if r.tso_objective is not None])

    # DSO series
    dso_mask = np.array([r.dso_active and r.dso_q_der_mvar is not None
                         for r in log])
    dso_min = minutes[dso_mask]
    dso_recs = [r for r in log if r.dso_active and r.dso_q_der_mvar is not None]

    dso_q_der = np.array([r.dso_q_der_mvar for r in dso_recs])
    dso_oltc = np.array([r.dso_oltc_taps for r in dso_recs]) if dso_recs and dso_recs[0].dso_oltc_taps is not None and len(dso_recs[0].dso_oltc_taps) > 0 else None
    dso_shunt = np.array([r.dso_shunt_states for r in dso_recs]) if dso_recs and dso_recs[0].dso_shunt_states is not None and len(dso_recs[0].dso_shunt_states) > 0 else None
    dso_q_set = np.array([r.dso_q_setpoint_mvar for r in dso_recs]) if dso_recs and dso_recs[0].dso_q_setpoint_mvar is not None else None
    dso_q_act = np.array([r.dso_q_actual_mvar for r in dso_recs
                          if r.dso_q_actual_mvar is not None])
    dso_q_act_min = np.array([r.minute for r in dso_recs
                              if r.dso_q_actual_mvar is not None])
    dso_obj = np.array([r.dso_objective for r in dso_recs  # ToDo: Manually disabled
                        if r.dso_objective is not None])
    dso_obj_min = np.array([r.minute for r in dso_recs
                            if r.dso_objective is not None])

    # Penalty term series (recorded every minute after PF)
    tso_v_pen = np.array([r.tso_v_penalty for r in log
                          if r.tso_v_penalty is not None])
    tso_v_pen_min = np.array([r.minute for r in log
                              if r.tso_v_penalty is not None])
    dso_q_pen = np.array([r.dso_q_penalty for r in log
                          if r.dso_q_penalty is not None])
    dso_q_pen_min = np.array([r.minute for r in log
                              if r.dso_q_penalty is not None])

    return dict(
        minutes=minutes,
        tn_v=tn_v, dn_v=dn_v,
        tso_min=tso_min, tso_q_der=tso_q_der, tso_q_pcc=tso_q_pcc,
        tso_v_gen=tso_v_gen, tso_oltc=tso_oltc, tso_shunt=tso_shunt,
        tso_obj=tso_obj, tso_obj_min=tso_obj_min,
        tso_v_pen=tso_v_pen, tso_v_pen_min=tso_v_pen_min,
        dso_min=dso_min, dso_q_der=dso_q_der,
        dso_oltc=dso_oltc, dso_shunt=dso_shunt,
        dso_q_set=dso_q_set, dso_q_act=dso_q_act,
        dso_q_act_min=dso_q_act_min,
        dso_obj=dso_obj, dso_obj_min=dso_obj_min,
        dso_q_pen=dso_q_pen, dso_q_pen_min=dso_q_pen_min,
    )


# ─── TSO figure ─────────────────────────────────────────────────────────────

def plot_tso(
    log: List[IterationRecord],
    tso_config: TSOControllerConfig,
    *,
    show: bool = True,
) -> plt.Figure:
    """
    Plot TSO controller results.

    Subplots:
      1) EHV bus voltages + setpoint band
      2) TSO DER Q
      3) Q_gen (if generators)
      4) V_gen (if generators)
      5) OLTC taps (if OLTCs)
      6) TSO Objective value
    """
    s = _collect_series(log)
    v_buses = tso_config.voltage_bus_indices

    # Count input subplot rows needed
    n_input_rows = 1  # Q_DER always present
    if s["tso_v_gen"].shape[1] > 0:
        n_input_rows += 1
    if s["tso_oltc"] is not None:
        n_input_rows += 1
    if s["tso_shunt"] is not None:
        n_input_rows += 1

    n_rows = 1 + n_input_rows + 1  # voltages + inputs + objective
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.0 * n_rows),
                             sharex=True, constrained_layout=True)
    fig.suptitle("TSO Controller", fontsize=14, fontweight="bold")

    # ── 1) EHV voltages ──
    ax = axes[0]
    for j in range(s["tn_v"].shape[1]):
        ax.plot(s["minutes"], s["tn_v"][:, j], lw=0.8, alpha=0.7,
                label=f"Bus {v_buses[j]}")
    if tso_config.v_setpoints_pu is not None:
        v_set = tso_config.v_setpoints_pu[0]
        ax.axhline(v_set, color="k", ls="--", lw=1.2, label=f"V_set = {v_set:.3f}")
    ax.axhline(tso_config.v_min_pu, color="r", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(tso_config.v_max_pu, color="r", ls=":", lw=0.8, alpha=0.6)
    ax.set_ylabel("Voltage [p.u.]")
    ax.set_title("EHV Bus Voltages")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── 2+) Control inputs ──
    row = 1

    # Q_DER
    ax = axes[row]
    for j in range(s["tso_q_der"].shape[1]):
        ax.plot(s["tso_min"], s["tso_q_der"][:, j], lw=1.0,
                label=f"DER bus {tso_config.der_bus_indices[j]}")
    ax.set_ylabel("Q_DER [Mvar]")
    ax.set_title("TSO DER Reactive Power")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.grid(True, alpha=0.3)
    row += 1

    # V_gen
    if s["tso_v_gen"].shape[1] > 0:
        ax = axes[row]
        for j in range(s["tso_v_gen"].shape[1]):
            ax.plot(s["tso_min"], s["tso_v_gen"][:, j], lw=1.0,
                    label=f"Gen {tso_config.gen_indices[j]}")
        ax.set_ylabel("V_gen [p.u.]")
        ax.set_title("Generator AVR Setpoints")
        ax.legend(fontsize=7, ncol=4, loc="upper left")
        ax.grid(True, alpha=0.3)
        row += 1

    # OLTC taps
    if s["tso_oltc"] is not None:
        ax = axes[row]
        for j in range(s["tso_oltc"].shape[1]):
            ax.plot(s["tso_min"], s["tso_oltc"][:, j], lw=1.0,
                    label=f"OLTC trafo {tso_config.oltc_trafo_indices[j]}")
        ax.set_ylabel("Tap Position")
        ax.set_title("Machine Transformer OLTC Taps")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=7, ncol=4, loc="upper left")
        ax.grid(True, alpha=0.3)
        row += 1

    # Shunt states
    if s["tso_shunt"] is not None:
        ax = axes[row]
        for j in range(s["tso_shunt"].shape[1]):
            ax.plot(s["tso_min"], s["tso_shunt"][:, j], lw=1.0,
                    label=f"Shunt bus {tso_config.shunt_bus_indices[j]}")
        ax.set_ylabel("State")
        ax.set_title("TSO Shunt States")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=7, ncol=4, loc="upper left")
        ax.grid(True, alpha=0.3)
        row += 1

    # ── TSO Objective ──
    ax = axes[row]
    if len(s["tso_obj"]) > 0:
        ax.plot(s["tso_obj_min"], s["tso_obj"], lw=1.2, color="darkblue",
                marker=".", markersize=3, label="Total objective")
    if len(s["tso_v_pen"]) > 0:
        ax.plot(s["tso_v_pen_min"], s["tso_v_pen"], lw=1.0, color="orangered",
                ls="--", marker=".", markersize=2, alpha=0.8,
                label=r"$g_v \cdot \Sigma(V - V_{set})^2$")
    ax.set_ylabel("Objective")
    ax.set_title("TSO Objective Value")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [min]")

    if show:
        plt.show()
    return fig


# ─── DSO figure ─────────────────────────────────────────────────────────────

def plot_dso(
    log: List[IterationRecord],
    dso_config: DSOControllerConfig,
    *,
    show: bool = True,
) -> plt.Figure:
    """
    Plot DSO controller results.

    Subplots:
      1) DN bus voltages + min/max limits
      2) TSO-DSO Interface Q: setpoint vs actual
      3) DSO DER Q
      4) OLTC taps (if present)
      5) Shunt states (if present)
      6) DSO Objective value
    """
    s = _collect_series(log)
    v_buses = dso_config.voltage_bus_indices

    n_input_rows = 1  # Q_DER
    if s["dso_oltc"] is not None:
        n_input_rows += 1
    if s["dso_shunt"] is not None:
        n_input_rows += 1

    n_rows = 2 + n_input_rows + 1  # voltages + interface Q + inputs + objective
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.0 * n_rows),
                             sharex=True, constrained_layout=True)
    fig.suptitle("DSO Controller", fontsize=14, fontweight="bold")

    # ── 1) DN voltages ──
    ax = axes[0]
    for j in range(s["dn_v"].shape[1]):
        ax.plot(s["minutes"], s["dn_v"][:, j], lw=0.8, alpha=0.7,
                label=f"Bus {v_buses[j]}" if j < 10 else None)
    ax.axhline(dso_config.v_min_pu, color="r", ls="--", lw=1.2,
               label=f"V_min = {dso_config.v_min_pu:.2f}")
    ax.axhline(dso_config.v_max_pu, color="r", ls="--", lw=1.2,
               label=f"V_max = {dso_config.v_max_pu:.2f}")
    ax.set_ylabel("Voltage [p.u.]")
    ax.set_title("DN Bus Voltages")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── 2) Interface Q tracking ──
    ax = axes[1]
    prop_cycle_dso = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if s["dso_q_set"] is not None:
        for j in range(s["dso_q_set"].shape[1]):
            c = prop_cycle_dso[j % len(prop_cycle_dso)]
            ax.plot(s["dso_min"], s["dso_q_set"][:, j], ls="--", lw=1.0,
                    color=c, label=f"Q_set [{j}] (from TSO)")
    if len(s["dso_q_act"]) > 0:
        for j in range(s["dso_q_act"].shape[1]):
            c = prop_cycle_dso[j % len(prop_cycle_dso)]
            ax.plot(s["dso_q_act_min"], s["dso_q_act"][:, j], lw=1.8,
                    color=c, alpha=0.8, label=f"Q_actual [{j}]")
    ax.set_ylabel("Q [Mvar]")
    ax.set_title("TSO-DSO Interface Reactive Power (load conv., +Q into coupler from TN)")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── 3+) Control inputs ──
    row = 2

    # Q_DER
    ax = axes[row]
    for j in range(s["dso_q_der"].shape[1]):
        ax.plot(s["dso_min"], s["dso_q_der"][:, j], lw=0.8,
                alpha=0.7,
                label=f"DER bus {dso_config.der_bus_indices[j]}" if j < 10 else None)
    ax.set_ylabel("Q_DER [Mvar]")
    ax.set_title("DSO DER Reactive Power")
    if len(dso_config.der_bus_indices) <= 10:
        ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.grid(True, alpha=0.3)
    row += 1

    # OLTC taps
    if s["dso_oltc"] is not None:
        ax = axes[row]
        for j in range(s["dso_oltc"].shape[1]):
            ax.plot(s["dso_min"], s["dso_oltc"][:, j], lw=1.0,
                    label=f"OLTC trafo3w {dso_config.oltc_trafo_indices[j]}")
        ax.set_ylabel("Tap Position")
        ax.set_title("Coupler OLTC Taps")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=7, ncol=4, loc="upper left")
        ax.grid(True, alpha=0.3)
        row += 1

    # Shunt states
    if s["dso_shunt"] is not None:
        ax = axes[row]
        for j in range(s["dso_shunt"].shape[1]):
            ax.plot(s["dso_min"], s["dso_shunt"][:, j], lw=1.0,
                    label=f"Shunt bus {dso_config.shunt_bus_indices[j]}")
        ax.set_ylabel("State")
        ax.set_title("DSO Shunt States")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=7, ncol=4, loc="upper left")
        ax.grid(True, alpha=0.3)
        row += 1

    # ── DSO Objective ──
    ax = axes[row]
    if len(s["dso_obj"]) > 0:
        ax.plot(s["dso_obj_min"], s["dso_obj"], lw=1.2, color="darkgreen",
                marker=".", markersize=3, label="Total objective")
    if len(s["dso_q_pen"]) > 0:
        ax.plot(s["dso_q_pen_min"], s["dso_q_pen"], lw=1.0, color="orangered",
                ls="--", marker=".", markersize=2, alpha=0.8,
                label=r"$g_q \cdot \Sigma(Q - Q_{set})^2$")
    ax.set_ylabel("Objective")
    ax.set_title("DSO Objective Value")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [min]")

    if show:
        plt.show()
    return fig


# ─── convenience wrapper ────────────────────────────────────────────────────

def plot_all(
    log: List[IterationRecord],
    tso_config: TSOControllerConfig,
    dso_config: DSOControllerConfig,
    *,
    show: bool = True,
) -> tuple:
    """Plot both TSO and DSO figures. Returns (fig_tso, fig_dso)."""
    fig_tso = plot_tso(log, tso_config, show=False)
    fig_dso = plot_dso(log, dso_config, show=False)
    if show:
        plt.show()
    return fig_tso, fig_dso


# ─── live plotter ───────────────────────────────────────────────────────────

class LivePlotter:
    """
    Real-time visualisation callback for the cascade simulation loop.

    Usage::

        live = LivePlotter(tso_config, dso_config)
        # inside loop:
        live.update(rec)
        # after loop:
        live.finish()

    Plots are updated in-place using ``plt.pause`` with interactive mode.
    Two figures are created:

    **TSO figure:** EHV voltages, Q_DER, Q_gen, V_gen, OLTC taps, objective.
    **DSO figure:** DN voltages, interface Q tracking, Q_DER, OLTC taps,
    shunt states, objective.
    """

    def __init__(
        self,
        tso_config: TSOControllerConfig,
        dso_config: DSOControllerConfig,
        update_every: int = 1,
        tso_line_max_i_ka: Optional[np.ndarray] = None,
        dso_line_max_i_ka: Optional[np.ndarray] = None,
    ) -> None:
        self._tso_cfg = tso_config
        self._dso_cfg = dso_config
        self._update_every = update_every
        self._call_count = 0

        # Thermal limits for current plots (kA, one per monitored line)
        self._tso_line_max_i_ka = tso_line_max_i_ka
        self._dso_line_max_i_ka = dso_line_max_i_ka

        # Accumulated data
        self._minutes: List[int] = []
        self._tn_v: List[np.ndarray] = []
        self._dn_v: List[np.ndarray] = []
        self._tn_i: List[np.ndarray] = []   # TN line currents [kA]
        self._dn_i: List[np.ndarray] = []   # DN line currents [kA]
        self._tso_min: List[int] = []
        self._tso_q_pcc: List[np.ndarray] = []
        self._tso_q_der: List[np.ndarray] = []
        self._tso_v_gen: List[np.ndarray] = []
        self._tso_oltc: List[np.ndarray] = []
        self._tso_q_gen: List[np.ndarray] = []
        self._tso_q_gen_min: List[int] = []
        self._tso_obj: List[float] = []
        self._tso_obj_min: List[int] = []
        self._dso_min: List[int] = []
        self._dso_q_act: List[np.ndarray] = []
        self._dso_q_act_min: List[int] = []
        self._dso_q_set: List[np.ndarray] = []
        self._dso_q_set_min: List[int] = []
        self._dso_q_der: List[np.ndarray] = []
        self._dso_oltc: List[np.ndarray] = []
        self._dso_shunt: List[np.ndarray] = []
        self._dso_obj: List[float] = []
        self._dso_obj_min: List[int] = []
        self._tso_v_pen: List[float] = []
        self._tso_v_pen_min: List[int] = []
        self._dso_q_pen: List[float] = []
        self._dso_q_pen_min: List[int] = []

        # Detect which input rows are needed
        self._has_tso_oltc = len(tso_config.oltc_trafo_indices) > 0
        self._has_tso_gen = len(tso_config.gen_indices) > 0
        self._has_dso_oltc = len(dso_config.oltc_trafo_indices) > 0
        self._has_dso_shunt = len(dso_config.shunt_bus_indices) > 0

        plt.ion()

        # ── TSO figure: V + Q_DER + I_line + Q_gen + V_gen + OLTC + objective ──
        self._has_tso_current = len(tso_config.current_line_indices) > 0
        n_tso_rows = 2  # voltages + Q_DER always present
        if self._has_tso_current:
            n_tso_rows += 1  # line currents
        if self._has_tso_gen:
            n_tso_rows += 2  # Q_gen + V_gen
        if self._has_tso_oltc:
            n_tso_rows += 1
        n_tso_rows += 1  # objective
        self._fig_tso, self._axes_tso = plt.subplots(
            n_tso_rows, 1, figsize=(10, 2.5 * n_tso_rows),
            sharex=True, constrained_layout=True)
        self._fig_tso.suptitle("TSO Controller (live)", fontweight="bold")

        self._ax_tso_v = self._axes_tso[0]
        self._ax_tso_v.set_ylabel("Voltage [p.u.]")
        self._ax_tso_v.set_title("EHV Bus Voltages")
        self._ax_tso_v.grid(True, alpha=0.3)

        if tso_config.v_setpoints_pu is not None:
            v_set = tso_config.v_setpoints_pu[0]
            self._ax_tso_v.axhline(v_set, color="k", ls="--", lw=1.0)

        self._ax_tso_qder = self._axes_tso[1]
        self._ax_tso_qder.set_ylabel("Q_DER [Mvar]")
        self._ax_tso_qder.set_title("TSO DER Reactive Power")
        self._ax_tso_qder.grid(True, alpha=0.3)

        tso_row = 2
        self._ax_tso_current = None
        if self._has_tso_current:
            self._ax_tso_current = self._axes_tso[tso_row]
            self._ax_tso_current.set_ylabel("I [kA]")
            self._ax_tso_current.set_title("TN Line Currents vs. Thermal Limits")
            self._ax_tso_current.grid(True, alpha=0.3)
            tso_row += 1

        self._ax_tso_qgen = None
        self._ax_tso_vgen = None
        if self._has_tso_gen:
            self._ax_tso_qgen = self._axes_tso[tso_row]
            self._ax_tso_qgen.set_ylabel("Q_gen [Mvar]")
            self._ax_tso_qgen.set_title("Synchronous Generator Q Output")
            self._ax_tso_qgen.grid(True, alpha=0.3)
            tso_row += 1

            self._ax_tso_vgen = self._axes_tso[tso_row]
            self._ax_tso_vgen.set_ylabel("V_gen [p.u.]")
            self._ax_tso_vgen.set_title("Generator AVR Setpoints")
            self._ax_tso_vgen.grid(True, alpha=0.3)
            tso_row += 1

        self._ax_tso_oltc = None
        if self._has_tso_oltc:
            self._ax_tso_oltc = self._axes_tso[tso_row]
            self._ax_tso_oltc.set_ylabel("Tap Position")
            self._ax_tso_oltc.set_title("Machine Transformer OLTC Taps")
            self._ax_tso_oltc.yaxis.set_major_locator(MaxNLocator(integer=True))
            self._ax_tso_oltc.grid(True, alpha=0.3)
            tso_row += 1

        self._ax_tso_obj = self._axes_tso[tso_row]
        self._ax_tso_obj.set_ylabel("Objective")
        self._ax_tso_obj.set_title("TSO Objective Value")
        self._ax_tso_obj.grid(True, alpha=0.3)

        self._axes_tso[-1].set_xlabel("Time [min]")

        # ── DSO figure: V + interface Q + Q_DER + I_line + OLTC + shunt + objective ──
        self._has_dso_current = len(dso_config.current_line_indices) > 0
        n_dso_rows = 3  # voltages + interface Q + Q_DER always present
        if self._has_dso_current:
            n_dso_rows += 1  # line currents
        if self._has_dso_oltc:
            n_dso_rows += 1
        if self._has_dso_shunt:
            n_dso_rows += 1
        n_dso_rows += 1  # objective
        self._fig_dso, self._axes_dso = plt.subplots(
            n_dso_rows, 1, figsize=(10, 2.5 * n_dso_rows),
            sharex=True, constrained_layout=True)
        self._fig_dso.suptitle("DSO Controller (live)", fontweight="bold")

        # Store axes references
        self._ax_dso_v = self._axes_dso[0]
        self._ax_dso_v.set_ylabel("Voltage [p.u.]")
        self._ax_dso_v.set_title("DN Bus Voltages")
        self._ax_dso_v.grid(True, alpha=0.3)
        self._ax_dso_v.axhline(dso_config.v_min_pu, color="r", ls="--", lw=1.0)
        self._ax_dso_v.axhline(dso_config.v_max_pu, color="r", ls="--", lw=1.0)

        self._ax_dso_iface = self._axes_dso[1]
        self._ax_dso_iface.set_ylabel("Q [Mvar]")
        self._ax_dso_iface.set_title(
            "TSO-DSO Interface Q (load conv., +Q into coupler from TN)")
        self._ax_dso_iface.grid(True, alpha=0.3)

        self._ax_dso_qder = self._axes_dso[2]
        self._ax_dso_qder.set_ylabel("Q_DER [Mvar]")
        self._ax_dso_qder.set_title("DSO DER Reactive Power")
        self._ax_dso_qder.grid(True, alpha=0.3)

        dso_row = 3
        self._ax_dso_current = None
        if self._has_dso_current:
            self._ax_dso_current = self._axes_dso[dso_row]
            self._ax_dso_current.set_ylabel("I [kA]")
            self._ax_dso_current.set_title("DN Line Currents vs. Thermal Limits")
            self._ax_dso_current.grid(True, alpha=0.3)
            dso_row += 1

        self._ax_dso_oltc = None
        if self._has_dso_oltc:
            self._ax_dso_oltc = self._axes_dso[dso_row]
            self._ax_dso_oltc.set_ylabel("Tap Position")
            self._ax_dso_oltc.set_title("Coupler OLTC Taps")
            self._ax_dso_oltc.yaxis.set_major_locator(MaxNLocator(integer=True))
            self._ax_dso_oltc.grid(True, alpha=0.3)
            dso_row += 1

        self._ax_dso_shunt = None
        if self._has_dso_shunt:
            self._ax_dso_shunt = self._axes_dso[dso_row]
            self._ax_dso_shunt.set_ylabel("State")
            self._ax_dso_shunt.set_title("DSO Shunt States")
            self._ax_dso_shunt.yaxis.set_major_locator(MaxNLocator(integer=True))
            self._ax_dso_shunt.grid(True, alpha=0.3)
            dso_row += 1

        self._ax_dso_obj = self._axes_dso[dso_row]
        self._ax_dso_obj.set_ylabel("Objective")
        self._ax_dso_obj.set_title("DSO Objective Value")
        self._ax_dso_obj.grid(True, alpha=0.3)

        self._axes_dso[-1].set_xlabel("Time [min]")

        # Dock figures side-by-side: TSO on the left, DSO on the right
        self._position_windows_side_by_side()

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
            pass  # fall back to default overlapping placement

    def update(self, rec) -> None:
        """Feed one IterationRecord and refresh plots."""
        self._call_count += 1
        self._minutes.append(rec.minute)

        if rec.plant_tn_voltages_pu is not None:
            self._tn_v.append(rec.plant_tn_voltages_pu)
        if rec.plant_dn_voltages_pu is not None:
            self._dn_v.append(rec.plant_dn_voltages_pu)
        if hasattr(rec, "plant_tn_currents_ka") and rec.plant_tn_currents_ka is not None:
            self._tn_i.append(rec.plant_tn_currents_ka)
        if hasattr(rec, "plant_dn_currents_ka") and rec.plant_dn_currents_ka is not None:
            self._dn_i.append(rec.plant_dn_currents_ka)
        if hasattr(rec, "tso_q_gen_mvar") and rec.tso_q_gen_mvar is not None:
            self._tso_q_gen_min.append(rec.minute)
            self._tso_q_gen.append(rec.tso_q_gen_mvar)
        if rec.tso_active and rec.tso_q_pcc_set_mvar is not None:
            self._tso_min.append(rec.minute)
            self._tso_q_pcc.append(rec.tso_q_pcc_set_mvar)
            if rec.tso_q_der_mvar is not None:
                self._tso_q_der.append(rec.tso_q_der_mvar)
            if rec.tso_v_gen_pu is not None:
                self._tso_v_gen.append(rec.tso_v_gen_pu)
            if rec.tso_oltc_taps is not None:
                self._tso_oltc.append(rec.tso_oltc_taps)
            if rec.tso_objective is not None: # ToDo: Manually disabled
                self._tso_obj_min.append(rec.minute)
                self._tso_obj.append(rec.tso_objective)
        if rec.dso_active and rec.dso_q_setpoint_mvar is not None:
            self._dso_q_set_min.append(rec.minute)
            self._dso_q_set.append(rec.dso_q_setpoint_mvar)
        if rec.dso_active and rec.dso_q_actual_mvar is not None:
            self._dso_q_act_min.append(rec.minute)
            self._dso_q_act.append(rec.dso_q_actual_mvar)
        if rec.dso_active and rec.dso_q_der_mvar is not None:
            self._dso_min.append(rec.minute)
            self._dso_q_der.append(rec.dso_q_der_mvar)
            if rec.dso_oltc_taps is not None:
                self._dso_oltc.append(rec.dso_oltc_taps)
            if rec.dso_shunt_states is not None:
                self._dso_shunt.append(rec.dso_shunt_states)
            if rec.dso_objective is not None: # ToDo: Manually disabled
                self._dso_obj_min.append(rec.minute)
                self._dso_obj.append(rec.dso_objective)

        # Penalty terms (available every minute after PF)
        if hasattr(rec, "tso_v_penalty") and rec.tso_v_penalty is not None:
            self._tso_v_pen_min.append(rec.minute)
            self._tso_v_pen.append(rec.tso_v_penalty)
        if hasattr(rec, "dso_q_penalty") and rec.dso_q_penalty is not None:
            self._dso_q_pen_min.append(rec.minute)
            self._dso_q_pen.append(rec.dso_q_penalty)

        if self._call_count % self._update_every != 0:
            return

        self._redraw()

    def _redraw(self) -> None:
        mins = np.array(self._minutes[:len(self._tn_v)])

        # ═══════════════ TSO FIGURE ═══════════════

        # TSO voltages
        if len(self._tn_v) > 0:
            tn = np.array(self._tn_v)
            self._ax_tso_v.clear()
            self._ax_tso_v.set_ylabel("Voltage [p.u.]")
            self._ax_tso_v.set_title("EHV Bus Voltages")
            self._ax_tso_v.grid(True, alpha=0.3)
            if self._tso_cfg.v_setpoints_pu is not None:
                v_set = self._tso_cfg.v_setpoints_pu[0]
                self._ax_tso_v.axhline(v_set, color="k", ls="--", lw=1.0)
            for j in range(tn.shape[1]):
                self._ax_tso_v.plot(mins, tn[:, j], lw=0.7, alpha=0.7)

        # TSO Q_DER
        if len(self._tso_q_der) > 0:
            qd_arr = np.array(self._tso_q_der)
            td_arr = np.array(self._tso_min[:len(self._tso_q_der)])
            self._ax_tso_qder.clear()
            self._ax_tso_qder.set_ylabel("Q_DER [Mvar]")
            self._ax_tso_qder.set_title("TSO DER Reactive Power")
            self._ax_tso_qder.grid(True, alpha=0.3)
            for j in range(qd_arr.shape[1]):
                self._ax_tso_qder.plot(
                    td_arr, qd_arr[:, j], lw=1.0,
                    label=f"DER bus {self._tso_cfg.der_bus_indices[j]}")
            self._ax_tso_qder.legend(fontsize=7, ncol=4, loc="upper left")

        # TSO Line Currents vs. Thermal Limits
        if self._ax_tso_current is not None and len(self._tn_i) > 0:
            tn_i = np.array(self._tn_i)
            mins_i = np.array(self._minutes[:len(self._tn_i)])
            self._ax_tso_current.clear()
            self._ax_tso_current.set_ylabel("I [kA]")
            self._ax_tso_current.set_title("TN Line Currents vs. Thermal Limits")
            self._ax_tso_current.grid(True, alpha=0.3)
            for j in range(tn_i.shape[1]):
                self._ax_tso_current.plot(mins_i, tn_i[:, j], lw=0.7, alpha=0.7)
            # Draw thermal limits as horizontal dashed lines
            if self._tso_line_max_i_ka is not None:
                for j in range(len(self._tso_line_max_i_ka)):
                    lim = self._tso_line_max_i_ka[j]
                    if lim < 1e5:  # skip absurdly large limits
                        self._ax_tso_current.axhline(
                            lim, color="r", ls="--", lw=0.8, alpha=0.5)
                # Draw one visible legend entry for the limit band
                self._ax_tso_current.axhline(
                    np.nan, color="r", ls="--", lw=0.8, label="thermal limit")
                self._ax_tso_current.legend(fontsize=7, loc="upper left")

        # TSO Q_gen (synchronous generator reactive power output)
        if self._ax_tso_qgen is not None and len(self._tso_q_gen) > 0:
            qg_arr = np.array(self._tso_q_gen)
            tg_arr = np.array(self._tso_q_gen_min)
            self._ax_tso_qgen.clear()
            self._ax_tso_qgen.set_ylabel("Q_gen [Mvar]")
            self._ax_tso_qgen.set_title("Synchronous Generator Q Output")
            self._ax_tso_qgen.grid(True, alpha=0.3)
            for j in range(qg_arr.shape[1]):
                self._ax_tso_qgen.plot(
                    tg_arr, qg_arr[:, j], lw=1.0,
                    label=f"Gen {self._tso_cfg.gen_indices[j]}")
            self._ax_tso_qgen.legend(fontsize=7, ncol=4, loc="upper left")

        # TSO V_gen (AVR setpoints)
        if self._ax_tso_vgen is not None and len(self._tso_v_gen) > 0:
            vg_arr = np.array(self._tso_v_gen)
            td_arr = np.array(self._tso_min[:len(self._tso_v_gen)])
            self._ax_tso_vgen.clear()
            self._ax_tso_vgen.set_ylabel("V_gen [p.u.]")
            self._ax_tso_vgen.set_title("Generator AVR Setpoints")
            self._ax_tso_vgen.grid(True, alpha=0.3)
            for j in range(vg_arr.shape[1]):
                self._ax_tso_vgen.plot(
                    td_arr, vg_arr[:, j], lw=1.0,
                    label=f"Gen {self._tso_cfg.gen_indices[j]}")
            self._ax_tso_vgen.legend(fontsize=7, ncol=4, loc="upper left")

        # TSO OLTC taps
        if self._ax_tso_oltc is not None and len(self._tso_oltc) > 0:
            ot_arr = np.array(self._tso_oltc)
            td_arr = np.array(self._tso_min[:len(self._tso_oltc)])
            self._ax_tso_oltc.clear()
            self._ax_tso_oltc.set_ylabel("Tap Position")
            self._ax_tso_oltc.set_title("Machine Transformer OLTC Taps")
            self._ax_tso_oltc.yaxis.set_major_locator(MaxNLocator(integer=True))
            self._ax_tso_oltc.grid(True, alpha=0.3)
            for j in range(ot_arr.shape[1]):
                self._ax_tso_oltc.plot(
                    td_arr, ot_arr[:, j], lw=1.0,
                    label=f"OLTC trafo {self._tso_cfg.oltc_trafo_indices[j]}")
            self._ax_tso_oltc.legend(fontsize=7, ncol=4, loc="upper left")

        # TSO Objective
        if len(self._tso_obj) > 0 or len(self._tso_v_pen) > 0:
            self._ax_tso_obj.clear()
            self._ax_tso_obj.set_ylabel("Objective")
            self._ax_tso_obj.set_title("TSO Objective Value")
            self._ax_tso_obj.grid(True, alpha=0.3)
            # Determine if log scale is safe
            all_vals = []
            if len(self._tso_obj) > 0:
                all_vals.extend(self._tso_obj)
            if len(self._tso_v_pen) > 0:
                all_vals.extend(self._tso_v_pen)
            if all_vals and all(v > 0 for v in all_vals): # ToDo: Manually disabled
                self._ax_tso_obj.set_yscale("log")
            if len(self._tso_obj) > 0:
                obj_arr = np.array(self._tso_obj)
                obj_min_arr = np.array(self._tso_obj_min)
                self._ax_tso_obj.plot(obj_min_arr, obj_arr, lw=1.2,
                                      color="darkblue", marker=".", markersize=3,
                                      label="Total objective")
            if len(self._tso_v_pen) > 0:
                pen_arr = np.array(self._tso_v_pen)
                pen_min_arr = np.array(self._tso_v_pen_min)
                self._ax_tso_obj.plot(pen_min_arr, pen_arr, lw=1.0,
                                      color="orangered", ls="--", marker=".",
                                      markersize=2, alpha=0.8,
                                      label=r"$g_v \cdot \Sigma(V - V_{set})^2$")
            self._ax_tso_obj.legend(fontsize=7, ncol=2, loc="upper left")

        self._fig_tso.canvas.draw_idle()
        self._fig_tso.canvas.flush_events()

        # ═══════════════ DSO FIGURE ═══════════════

        # DSO voltages
        mins_dn = np.array(self._minutes[:len(self._dn_v)])
        if len(self._dn_v) > 0:
            dn = np.array(self._dn_v)
            self._ax_dso_v.clear()
            self._ax_dso_v.set_ylabel("Voltage [p.u.]")
            self._ax_dso_v.set_title("DN Bus Voltages")
            self._ax_dso_v.grid(True, alpha=0.3)
            for j in range(dn.shape[1]):
                self._ax_dso_v.plot(mins_dn, dn[:, j], lw=0.7, alpha=0.7)

        # DSO Interface Q (setpoint vs actual) — moved here from TSO figure
        if len(self._tso_q_pcc) > 0 or len(self._dso_q_act) > 0:
            self._ax_dso_iface.clear()
            self._ax_dso_iface.set_ylabel("Q [Mvar]")
            self._ax_dso_iface.set_title(
                "TSO-DSO Interface Q (load conv., +Q into coupler from TN)")
            self._ax_dso_iface.grid(True, alpha=0.3)

            prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            if len(self._tso_q_pcc) > 0:
                qp_arr = np.array(self._tso_q_pcc)
                tp_arr = np.array(self._tso_min)
                for j in range(qp_arr.shape[1]):
                    c = prop_cycle[j % len(prop_cycle)]
                    self._ax_dso_iface.plot(tp_arr, qp_arr[:, j], ls="--",
                                            lw=1.0, color=c,
                                            label=f"Q_set [{j}]")

            if len(self._dso_q_act) > 0:
                qa_arr = np.array(self._dso_q_act)
                ta_arr = np.array(self._dso_q_act_min)
                for j in range(qa_arr.shape[1]):
                    c = prop_cycle[j % len(prop_cycle)]
                    self._ax_dso_iface.plot(ta_arr, qa_arr[:, j], lw=1.8,
                                            color=c, alpha=0.8,
                                            label=f"Q_actual [{j}]")

            self._ax_dso_iface.legend(fontsize=7, ncol=4, loc="upper left")

        # DSO Q_DER
        if len(self._dso_q_der) > 0:
            qd_arr = np.array(self._dso_q_der)
            td_arr = np.array(self._dso_min)
            self._ax_dso_qder.clear()
            self._ax_dso_qder.set_ylabel("Q_DER [Mvar]")
            self._ax_dso_qder.set_title("DSO DER Reactive Power")
            self._ax_dso_qder.grid(True, alpha=0.3)
            for j in range(qd_arr.shape[1]):
                lbl = (f"DER bus {self._dso_cfg.der_bus_indices[j]}"
                       if j < 10 else None)
                self._ax_dso_qder.plot(td_arr, qd_arr[:, j], lw=0.8,
                                       alpha=0.7, label=lbl)
            if len(self._dso_cfg.der_bus_indices) <= 10:
                self._ax_dso_qder.legend(fontsize=7, ncol=4, loc="upper left")

        # DSO Line Currents vs. Thermal Limits
        if self._ax_dso_current is not None and len(self._dn_i) > 0:
            dn_i = np.array(self._dn_i)
            mins_i_dn = np.array(self._minutes[:len(self._dn_i)])
            self._ax_dso_current.clear()
            self._ax_dso_current.set_ylabel("I [kA]")
            self._ax_dso_current.set_title("DN Line Currents vs. Thermal Limits")
            self._ax_dso_current.grid(True, alpha=0.3)
            for j in range(dn_i.shape[1]):
                self._ax_dso_current.plot(mins_i_dn, dn_i[:, j], lw=0.7, alpha=0.7)
            # Draw thermal limits as horizontal dashed lines
            if self._dso_line_max_i_ka is not None:
                for j in range(len(self._dso_line_max_i_ka)):
                    lim = self._dso_line_max_i_ka[j]
                    if lim < 1e5:  # skip absurdly large limits
                        self._ax_dso_current.axhline(
                            lim, color="r", ls="--", lw=0.8, alpha=0.5)
                # Draw one visible legend entry for the limit band
                self._ax_dso_current.axhline(
                    np.nan, color="r", ls="--", lw=0.8, label="thermal limit")
                self._ax_dso_current.legend(fontsize=7, loc="upper left")

        # DSO OLTC taps
        if self._ax_dso_oltc is not None and len(self._dso_oltc) > 0:
            ot_arr = np.array(self._dso_oltc)
            td_arr = np.array(self._dso_min[:len(self._dso_oltc)])
            self._ax_dso_oltc.clear()
            self._ax_dso_oltc.set_ylabel("Tap Position")
            self._ax_dso_oltc.set_title("Coupler OLTC Taps")
            self._ax_dso_oltc.yaxis.set_major_locator(MaxNLocator(integer=True))
            self._ax_dso_oltc.grid(True, alpha=0.3)
            for j in range(ot_arr.shape[1]):
                self._ax_dso_oltc.plot(
                    td_arr, ot_arr[:, j], lw=1.0,
                    label=f"OLTC trafo3w {self._dso_cfg.oltc_trafo_indices[j]}")
            self._ax_dso_oltc.legend(fontsize=7, ncol=4, loc="upper left")

        # DSO shunt states
        if self._ax_dso_shunt is not None and len(self._dso_shunt) > 0:
            sh_arr = np.array(self._dso_shunt)
            td_arr = np.array(self._dso_min[:len(self._dso_shunt)])
            self._ax_dso_shunt.clear()
            self._ax_dso_shunt.set_ylabel("State")
            self._ax_dso_shunt.set_title("DSO Shunt States")
            self._ax_dso_shunt.yaxis.set_major_locator(MaxNLocator(integer=True))
            self._ax_dso_shunt.grid(True, alpha=0.3)
            for j in range(sh_arr.shape[1]):
                self._ax_dso_shunt.plot(
                    td_arr, sh_arr[:, j], lw=1.0,
                    label=f"Shunt bus {self._dso_cfg.shunt_bus_indices[j]}")
            self._ax_dso_shunt.legend(fontsize=7, ncol=4, loc="upper left")

        # DSO Objective
        if len(self._dso_obj) > 0 or len(self._dso_q_pen) > 0:
            self._ax_dso_obj.clear()
            self._ax_dso_obj.set_ylabel("Objective")
            self._ax_dso_obj.set_title("DSO Objective Value")
            self._ax_dso_obj.grid(True, alpha=0.3)
            # Determine if log scale is safe
            all_vals = []
            if len(self._dso_obj) > 0:
                all_vals.extend(self._dso_obj)
            if len(self._dso_q_pen) > 0:
                all_vals.extend(self._dso_q_pen)
            if all_vals and all(v > 0 for v in all_vals): # ToDo: Manually disabled
                self._ax_dso_obj.set_yscale("log")
            if len(self._dso_obj) > 0:
                obj_arr = np.array(self._dso_obj)
                obj_min_arr = np.array(self._dso_obj_min)
                self._ax_dso_obj.plot(obj_min_arr, obj_arr, lw=1.2,
                                      color="darkgreen", marker=".", markersize=3,
                                      label="Total objective")
            if len(self._dso_q_pen) > 0:
                pen_arr = np.array(self._dso_q_pen)
                pen_min_arr = np.array(self._dso_q_pen_min)
                self._ax_dso_obj.plot(pen_min_arr, pen_arr, lw=1.0,
                                      color="orangered", ls="--", marker=".",
                                      markersize=2, alpha=0.8,
                                      label=r"$g_q \cdot \Sigma(Q - Q_{set})^2$")
            self._ax_dso_obj.legend(fontsize=7, ncol=2, loc="upper left")

        self._fig_dso.canvas.draw_idle()
        self._fig_dso.canvas.flush_events()
        plt.pause(0.01)

    def finish(self) -> None:
        """Final redraw and switch to blocking mode."""
        self._redraw()
        plt.ioff()
        plt.show()
