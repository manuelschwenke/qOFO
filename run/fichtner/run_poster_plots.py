#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poster-Plots für den Fichtner-Award
=====================================
Simuliert 90 min mit Generatorausfall bei min 30.
Vergleicht: nur ÜNB-Controller vs. kaskadierter ÜNB+VNB-Controller.

Erzeugt drei Plots:
  1) ÜN-Spannungsband (Median/Min/Max) – Vergleich
  2) Aktoreinsatz im Verteilnetz
  3) Generator Q, Sollspannung, MT-Stufensteller

Author: Manuel Schwenke / Claude Code
"""

from __future__ import annotations

import time as _time
from datetime import datetime
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from numpy.typing import NDArray

from run.records import (
    ContingencyEvent,
    IterationRecord,
    CascadeResult,
)

from run.run_cascade import run_cascade


from core.cascade_config import CascadeConfig

# ─── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 32,
    "axes.titlesize": 38,
    "axes.labelsize": 34,
    "legend.fontsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 3.0,
})

# Feste Ränder für gleiche Zeichenbreite über alle Plots
PLOT_LEFT = 0.12      # Platz für breitestes Y-Label
PLOT_RIGHT = 0.97     # rechter Rand

COLORS = {
    "mit": "#B1BD00",      # Grün – mit Kaskade
    "ohne": "#00689D",     # Blau – ohne Kaskade
    "band_mit": "#d8e07f",
    "band_ohne": "#7fb4ce",
    "setpoint": "#2ca02c",
    "gen_q": "#9467bd",
    "gen_v": "#ff7f0e",
    "oltc": "#8c564b",
    "shunt": "#e377c2",
    "der": "#B1BD00",
    "q_soll": "#CC4C03",
    "q_ist": "#B1BD00",
    "stufen": ['#B1BD00', '#00689D', '#CC4C03']
}


# ═══════════════════════════════════════════════════════════════════════════
#  Helper: extract time series from log
# ═══════════════════════════════════════════════════════════════════════════

def _extract(log: List[IterationRecord]):
    """Return dict of numpy arrays extracted from an IterationRecord log."""
    mins = np.array([r.minute for r in log])
    # TN voltages: (n_minutes, n_buses)
    tn_v = np.array([r.plant_tn_voltages_pu for r in log])  # shape (T, B)
    tn_median = np.median(tn_v, axis=1)
    tn_min = np.min(tn_v, axis=1)
    tn_max = np.max(tn_v, axis=1)

    # Generator Q and V_set (forward-fill from TSO steps)
    n_gen = log[0].tso_q_gen_mvar.shape[0] if log[0].tso_q_gen_mvar is not None else 0
    gen_q = np.full((len(log), n_gen), np.nan)
    gen_v = np.full((len(log), n_gen), np.nan)
    for i, r in enumerate(log):
        if r.tso_q_gen_mvar is not None:
            gen_q[i] = r.tso_q_gen_mvar
        if r.tso_v_gen_pu is not None:
            gen_v[i] = r.tso_v_gen_pu
    # forward-fill gen_v
    last_v = None
    for i in range(len(log)):
        if np.isnan(gen_v[i, 0]) if n_gen > 0 else True:
            if last_v is not None:
                gen_v[i] = last_v
        else:
            last_v = gen_v[i].copy()

    # TSO OLTC taps (machine trafos) – forward-fill
    first_tso = next((r for r in log if r.tso_oltc_taps is not None), None)
    n_oltc_tso = first_tso.tso_oltc_taps.shape[0] if first_tso is not None else 0
    oltc_mt = np.full((len(log), n_oltc_tso), np.nan)
    for i, r in enumerate(log):
        if r.tso_oltc_taps is not None:
            oltc_mt[i] = r.tso_oltc_taps
    last_tap = None
    for i in range(len(log)):
        if np.isnan(oltc_mt[i, 0]) if n_oltc_tso > 0 else True:
            if last_tap is not None:
                oltc_mt[i] = last_tap
        else:
            last_tap = oltc_mt[i].copy()

    # TSO shunt states – forward-fill
    first_shunt = next((r for r in log if r.tso_shunt_states is not None), None)
    n_shunt_tso = first_shunt.tso_shunt_states.shape[0] if first_shunt is not None else 0
    shunt_tso = np.full((len(log), n_shunt_tso), np.nan)
    for i, r in enumerate(log):
        if r.tso_shunt_states is not None:
            shunt_tso[i] = r.tso_shunt_states
    last_sh = None
    for i in range(len(log)):
        if np.isnan(shunt_tso[i, 0]) if n_shunt_tso > 0 else True:
            if last_sh is not None:
                shunt_tso[i] = last_sh
        else:
            last_sh = shunt_tso[i].copy()

    # DSO DER Q (sum) – forward-fill
    first_dso = next((r for r in log if r.dso_q_der_mvar is not None), None)
    n_der_dso = first_dso.dso_q_der_mvar.shape[0] if first_dso is not None else 0
    der_q_sum = np.full(len(log), np.nan)
    for i, r in enumerate(log):
        if r.dso_q_der_mvar is not None:
            der_q_sum[i] = np.sum(r.dso_q_der_mvar)
    last_dq = None
    for i in range(len(log)):
        if np.isnan(der_q_sum[i]):
            if last_dq is not None:
                der_q_sum[i] = last_dq
        else:
            last_dq = der_q_sum[i]

    # DSO OLTC taps (3W couplers) – forward-fill
    first_dso_oltc = next((r for r in log if r.dso_oltc_taps is not None), None)
    n_oltc_dso = first_dso_oltc.dso_oltc_taps.shape[0] if first_dso_oltc is not None else 0
    oltc_dso = np.full((len(log), n_oltc_dso), np.nan)
    for i, r in enumerate(log):
        if r.dso_oltc_taps is not None:
            oltc_dso[i] = r.dso_oltc_taps
    last_do = None
    for i in range(len(log)):
        if np.isnan(oltc_dso[i, 0]) if n_oltc_dso > 0 else True:
            if last_do is not None:
                oltc_dso[i] = last_do
        else:
            last_do = oltc_dso[i].copy()

    # DSO shunt states – forward-fill
    first_dso_sh = next((r for r in log if r.dso_shunt_states is not None), None)
    n_shunt_dso = first_dso_sh.dso_shunt_states.shape[0] if first_dso_sh is not None else 0
    shunt_dso = np.full((len(log), n_shunt_dso), np.nan)
    for i, r in enumerate(log):
        if r.dso_shunt_states is not None:
            shunt_dso[i] = r.dso_shunt_states
    last_ds = None
    for i in range(len(log)):
        if np.isnan(shunt_dso[i, 0]) if n_shunt_dso > 0 else True:
            if last_ds is not None:
                shunt_dso[i] = last_ds
        else:
            last_ds = shunt_dso[i].copy()

    # Q_Trafo: setpoint and actual – forward-fill
    first_qset = next((r for r in log if r.dso_q_setpoint_mvar is not None), None)
    n_iface = first_qset.dso_q_setpoint_mvar.shape[0] if first_qset is not None else 0
    q_soll = np.full((len(log), n_iface), np.nan)
    q_ist = np.full((len(log), n_iface), np.nan)
    for i, r in enumerate(log):
        if r.dso_q_setpoint_mvar is not None:
            q_soll[i] = r.dso_q_setpoint_mvar
        if r.dso_q_actual_mvar is not None:
            q_ist[i] = r.dso_q_actual_mvar
    for arr in (q_soll, q_ist):
        last = None
        for i in range(len(log)):
            if np.isnan(arr[i, 0]) if n_iface > 0 else True:
                if last is not None:
                    arr[i] = last
            else:
                last = arr[i].copy()

    return dict(
        mins=mins, tn_median=tn_median, tn_min=tn_min, tn_max=tn_max,
        gen_q=gen_q, gen_v=gen_v,
        oltc_mt=oltc_mt, shunt_tso=shunt_tso,
        der_q_sum=der_q_sum, oltc_dso=oltc_dso, shunt_dso=shunt_dso,
        q_soll=q_soll, q_ist=q_ist,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 1 – TN voltage band: with vs. without cascade
# ═══════════════════════════════════════════════════════════════════════════

def plot_voltage_comparison(d_mit: dict, d_ohne: dict, v_set: float, event_min: int):
    """Plot median, minimum and maximum transmission network voltage band."""

    # Fail-fast: Abort if any required argument is missing (Space instruction)
    if any(arg is None for arg in (d_mit, d_ohne, v_set, event_min)):
        raise ValueError("Missing arguments: d_mit, d_ohne, v_set, and event_min are strictly required.")

    fig, ax = plt.subplots(figsize=(24, 7))

    # Shaded voltage bands
    ax.fill_between(d_ohne["mins"], d_ohne["tn_min"], d_ohne["tn_max"],
                    alpha=0.25, color=COLORS["band_ohne"], label="_nolegend_")
    ax.fill_between(d_mit["mins"], d_mit["tn_min"], d_mit["tn_max"],
                    alpha=0.25, color=COLORS["band_mit"], label="_nolegend_")

    # Median lines
    ax.plot(d_ohne["mins"], d_ohne["tn_median"],
            color=COLORS["ohne"], label="Ohne VN (Med.)")
    ax.plot(d_mit["mins"], d_mit["tn_median"],
            color=COLORS["mit"], label="Mit VN (Med.)")

    # Min/Max thin dashed lines
    ax.plot(d_ohne["mins"], d_ohne["tn_min"], color=COLORS["ohne"],
            ls="--", lw=1.5, alpha=0.7, label="Ohne (Min./Max.)")
    ax.plot(d_ohne["mins"], d_ohne["tn_max"], color=COLORS["ohne"],
            ls="--", lw=1.5, alpha=0.7, label="_nolegend_")
    ax.plot(d_mit["mins"], d_mit["tn_min"], color=COLORS["mit"],
            ls="--", lw=1.5, alpha=0.7, label="Mit (Min./Max.)")
    ax.plot(d_mit["mins"], d_mit["tn_max"], color=COLORS["mit"],
            ls="--", lw=1.5, alpha=0.7, label="_nolegend_")

    # Setpoint
    ax.axhline(v_set, color='#CC4C03', ls=":", lw=2.0,
               label="Sollwert")

    # Event marker
    y_top = ax.get_ylim()[1]
    offset = (y_top - ax.get_ylim()[0]) * 0.02
    ax.axvline(event_min, color="grey", ls="-.", lw=2.0, alpha=0.7)
    ax.annotate("Generatorausfall", xy=(event_min, y_top),
                xytext=(event_min + 1, y_top - offset),
                fontsize=30, color="grey", va="top")

    # Labels and layout
    ax.set_xlabel(r"Zeit $k$ / min")
    ax.set_ylabel("Spannung / p.u.")
    ax.set_title("Spannungsband im Übertragungsnetz (380 kV)")

    # Adjusted legend for compact layout
    ax.legend(
        loc="lower left",
        ncol=5,
        framealpha=0.2,
        fontsize="medium",  # Optional: scales font size down slightly
        columnspacing=0.70,  # Reduces space between columns
        handlelength=1.1,  # Shortens the legend line/box handles
        handletextpad=0.4,  # Reduces gap between handle and text
        borderpad=0.3,  # Reduces inner padding of the legend box
        borderaxespad=0.3  # Pushes legend closer to the axis edge
    )

    ax.set_xlim(d_mit["mins"][0], d_mit["mins"][-1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(left=PLOT_LEFT, right=PLOT_RIGHT)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 2 – Aktoreinsatz im Verteilnetz (nur kaskadierter Lauf)
# ═══════════════════════════════════════════════════════════════════════════

def plot_actuator_detail(d: dict, event_min: int):
    """DER Q sum, 3 coupler OLTCs, shunts, Q_Trafo soll/ist."""
    n_oltc = d["oltc_dso"].shape[1]
    n_shunt = d["shunt_dso"].shape[1]
    n_iface = d["q_soll"].shape[1]

    fig, axes = plt.subplots(4, 1, figsize=(24, 14), sharex=True)

    mins = d["mins"]

    # (a) DER Q Summenlinie
    ax = axes[0]
    ax.plot(mins, d["der_q_sum"], color=COLORS["der"], label="ΣQ EZA (VN)")
    ax.axhline(0, color="grey", lw=1.0)
    ax.axvline(event_min, color="grey", ls="-.", lw=2.0, alpha=0.5)
    ax.set_ylabel(r"$Q$ / Mvar")
    ax.set_title("Blindleistungseinsatz EZA im Verteilnetz")
    ax.legend(loc="best", framealpha=0.2)
    ax.set_xlim(d["mins"][0], d["mins"][-1])
    ax.grid(True, alpha=0.3)

    # (b) 3W Coupler OLTCs
    ax = axes[1]
    coupler_labels = [f"Kupplung {k+1}" for k in range(n_oltc)]
    for k in range(n_oltc):
        ax.step(mins, d["oltc_dso"][:, k], where="post", label=coupler_labels[k], color=COLORS['stufen'][k])
    ax.axvline(event_min, color="grey", ls="-.", lw=2.0, alpha=0.5)
    ax.set_ylabel("Stufenposition")
    ax.set_title("Stufensteller der Kuppeltransformatoren (110 kV)")
    ax.legend(loc="best", ncol=n_oltc, framealpha=0.2)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(d["mins"][0], d["mins"][-1])
    ax.grid(True, alpha=0.3)

    # (c) Shunts
    ax = axes[2]
    shunt_labels = [f"Shunt {k+1}" for k in range(n_shunt)]
    for k in range(n_shunt):
        ax.step(mins, d["shunt_dso"][:, k], where="post", label=shunt_labels[k])
    ax.axvline(event_min, color="grey", ls="-.", lw=2.0, alpha=0.5)
    ax.set_ylabel("Schaltzustand")
    ax.set_title("Tertiär-Shunts (20 kV)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Aus", "Ein"])
    ax.legend(loc="best", ncol=n_shunt, framealpha=0.2)
    ax.set_xlim(d["mins"][0], d["mins"][-1])
    ax.grid(True, alpha=0.3)

    # (d) Q_Trafo soll vs. ist (sum over interfaces)
    ax = axes[3]
    q_soll_sum = np.nansum(d["q_soll"], axis=1)
    q_ist_sum = np.nansum(d["q_ist"], axis=1)
    ax.plot(mins, q_soll_sum, color=COLORS["q_soll"], label=r"$Q_\mathrm{VN,soll}^{k}$ (ÜNB→VNB)")
    ax.plot(mins, q_ist_sum, color=COLORS["q_ist"], ls="--", label=r"$Q_\mathrm{VN,ist}^{k}$ (ÜNB→VNB)")
    ax.axhline(0, color="grey", lw=1.0)
    ax.axvline(event_min, color="grey", ls="-.", lw=2.0, alpha=0.5)
    ax.set_xlabel(r"Zeit $k$ / min")
    ax.set_ylabel(r"$Q$ / Mvar")
    ax.set_title("Blindleistung an den Kuppelstellen (Summe)")
    ax.legend(loc="best", framealpha=0.2)
    ax.set_xlim(d["mins"][0], d["mins"][-1])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.subplots_adjust(left=PLOT_LEFT, right=PLOT_RIGHT)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 3 – Generator: Q, Sollspannung, MT-Stufensteller
# ═══════════════════════════════════════════════════════════════════════════

def plot_generator_detail(d: dict, event_min: int):
    """Combined plot with 2 y-axes: Gen Q + Gen V_set + MT OLTC."""
    n_gen = d["gen_q"].shape[1]

    fig, axes = plt.subplots(3, 1, figsize=(24, 11), sharex=True)
    mins = d["mins"]

    # (a) Generator Q
    ax = axes[0]
    for g in range(n_gen):
        ax.plot(mins, d["gen_q"][:, g], color=COLORS["gen_q"],
                label=f"Gen {g+1}" if n_gen > 1 else "Generator")
    ax.axhline(0, color="grey", lw=1.0)
    ax.axvline(event_min, color="grey", ls="-.", lw=2.0, alpha=0.5)
    ax.set_ylabel("Q / Mvar")
    ax.set_title("Blindleistung Synchrongenerator")
    ax.legend(loc="best", framealpha=0.2)
    ax.set_xlim(d["mins"][0], d["mins"][-1])
    ax.grid(True, alpha=0.3)

    # (b) Generator Sollspannung
    ax = axes[1]
    for g in range(n_gen):
        ax.plot(mins, d["gen_v"][:, g], color=COLORS["gen_v"],
                label=f"Gen {g+1}" if n_gen > 1 else "Generator")
    ax.axvline(event_min, color="grey", ls="-.", lw=2.0, alpha=0.5)
    ax.set_ylabel("Spannung / p.u.")
    ax.set_title("AVR-Sollspannung Generator")
    ax.legend(loc="best", framealpha=0.2)
    ax.set_xlim(d["mins"][0], d["mins"][-1])
    ax.grid(True, alpha=0.3)

    # (c) Maschinentrafo Stufensteller
    ax = axes[2]
    n_mt = d["oltc_mt"].shape[1]
    for k in range(n_mt):
        ax.step(mins, d["oltc_mt"][:, k], where="post", color=COLORS["oltc"],
                label=f"MT {k+1}" if n_mt > 1 else "Maschinentrafo")
    ax.axvline(event_min, color="grey", ls="-.", lw=2.0, alpha=0.5)
    ax.set_xlabel(r"Zeit $k$ / min")
    ax.set_ylabel("Stufenposition")
    ax.set_title("Stufensteller Maschinentransformator")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="best", framealpha=0.2)
    ax.set_xlim(d["mins"][0], d["mins"][-1])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.subplots_adjust(left=PLOT_LEFT, right=PLOT_RIGHT)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration builders
# ═══════════════════════════════════════════════════════════════════════════

def _base_config(*, enable_dso: bool) -> CascadeConfig:
    """Build a 90-min config with gen outage at min 30."""
    return CascadeConfig(
        # Simulation
        v_setpoint_pu=1.05,
        n_minutes=90,
        tso_period_min=3,
        dso_period_min=1,
        start_time=datetime(2016, 6, 1, 8, 0),
        use_profiles=True,
        verbose=1,
        live_plot=False,

        # Objective weights
        g_v=250000,
        g_q=1 if enable_dso else 0,
        dso_g_v=100000.0 if enable_dso else 0.0,

        # OFO
        alpha=1.0,
        g_z=1e12,

        # TSO g_w
        gw_tso_q_der=0.2,
        gw_tso_q_pcc=0.1,
        gw_tso_v_gen=5e6,
        gw_tso_oltc=10.0,
        gw_tso_shunt=2000.0,

        # DSO g_w
        gw_dso_q_der=10.0 if enable_dso else 1e12,
        gw_dso_oltc=100.0 if enable_dso else 1e12,
        gw_dso_shunt=4000.0 if enable_dso else 1e12,

        # DSO integral Q-tracking (PI-like)
        g_qi=0.15 if enable_dso else 0,
        lambda_qi=0.95,
        q_integral_max_mvar=50.0,

        # Generator capability
        gen_xd_pu=1.2,
        gen_i_f_max_pu=2.65,
        gen_beta=0.15,
        gen_q0_pu=0.4,

        # Reserve Observer
        enable_reserve_observer=False,#enable_dso,
        reserve_q_threshold_mvar=50.0,
        reserve_q_release_mvar=-50.0,
        reserve_cooldown_min=15,

        # Achievable Value Tracking
        k_t_avt=1.0,

        # Contingency: generator trip at minute 30
        # contingencies=[
        #     ContingencyEvent(minute=30, element_type="gen", element_index=0),
        # ],
        # Contingencies
        contingencies=[
            ContingencyEvent(minute=30, element_type="gen", element_index=0),
        ],

    )


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

EVENT_MIN = 30  # Generatorausfall

def main():
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "poster_plots")
    os.makedirs(out_dir, exist_ok=True)

    # ── Run 1: MIT kaskadiertem VNB-Controller ────────────────────────────
    print("=" * 60)
    print("  Lauf 1/2: MIT kaskadiertem VNB-Controller")
    print("=" * 60)
    t0 = _time.perf_counter()
    cfg_mit = _base_config(enable_dso=True)
    res_mit = run_cascade(cfg_mit)
    print(f"  Lauf 1 fertig in {_time.perf_counter() - t0:.1f} s\n")

    # ── Run 2: OHNE kaskadierten VNB-Controller ──────────────────────────
    print("=" * 60)
    print("  Lauf 2/2: OHNE kaskadierten VNB-Controller")
    print("=" * 60)
    t0 = _time.perf_counter()
    cfg_ohne = _base_config(enable_dso=False)
    res_ohne = run_cascade(cfg_ohne)
    print(f"  Lauf 2 fertig in {_time.perf_counter() - t0:.1f} s\n")

    # ── Extract data ─────────────────────────────────────────────────────
    d_mit = _extract(res_mit.log)
    d_ohne = _extract(res_ohne.log)

    # ── Plot 1: Spannungsvergleich ───────────────────────────────────────
    fig1 = plot_voltage_comparison(d_mit, d_ohne, cfg_mit.v_setpoint_pu, EVENT_MIN)
    fig1.savefig(os.path.join(out_dir, "poster_01_spannung.pdf"))
    fig1.savefig(os.path.join(out_dir, "poster_01_spannung.svg"))
    fig1.savefig(os.path.join(out_dir, "poster_01_spannung.png"))
    print("  Plot 1 gespeichert: poster_01_spannung.pdf/.svg/.png")

    # ── Plot 2: Aktoreinsatz VN (nur kaskadierter Lauf) ──────────────────
    fig2 = plot_actuator_detail(d_mit, EVENT_MIN)
    fig2.savefig(os.path.join(out_dir, "poster_02_aktoren.pdf"))
    fig2.savefig(os.path.join(out_dir, "poster_02_aktoren.svg"))
    fig2.savefig(os.path.join(out_dir, "poster_02_aktoren.png"))
    print("  Plot 2 gespeichert: poster_02_aktoren.pdf/.svg/.png")

    # ── Plot 3: Generator-Details (kaskadierter Lauf) ────────────────────
    fig3 = plot_generator_detail(d_mit, EVENT_MIN)
    fig3.savefig(os.path.join(out_dir, "poster_03_generator.pdf"))
    fig3.savefig(os.path.join(out_dir, "poster_03_generator.svg"))
    fig3.savefig(os.path.join(out_dir, "poster_03_generator.png"))
    print("  Plot 3 gespeichert: poster_03_generator.pdf/.svg/.png")

    plt.close("all")
    print(f"\nAlle Poster-Plots gespeichert in: {out_dir}")


if __name__ == "__main__":
    main()
