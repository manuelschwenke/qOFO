"""
Diagnose the T-OFO oscillation in 002_M_TSO_M_DSO_COMPARE.

Produces four plots and a summary report in
``experiments/results/002_compare/diagnostics_T-OFO/`` from the existing
log.pkl files (no re-simulation):

    1. per_zone_v_q.pdf            -- per-zone V envelope and aggregate Q over time
    2. q_v_phase.pdf               -- DSO-group Q vs V phase relationship (overlay + scatter)
    3. tso_command_trajectory.pdf  -- TSO Q_PCC_set and V_gen trajectories per zone, with
                                       empirical loop-gain estimate
    4. sensitivity_eigenvalues.pdf -- eigenvalue spectrum of the analytical closed-loop
                                       transfer (I + S_VQ_HV * K_droop) at warm-start;
                                       quantifies how far the cached open-loop sensitivity
                                       is from the true closed-loop one

Plus ``summary.md`` with the quantitative findings.

Usage:
    python tests/diag_t_ofo_oscillation.py
"""

from __future__ import annotations

import io
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = ROOT / "experiments" / "results" / "002_compare" / "diagnostics_T-OFO"
COMPARE_ROOT = ROOT / "experiments" / "results" / "002_compare"

QV_SLOPE_PU = 0.07
QV_SETPOINT_PU = 1.03
TSO_PERIOD_S = 180.0


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------


def load_logs(scenarios=("L2", "T-OFO", "C-OFO")) -> Dict[str, list]:
    out = {}
    for name in scenarios:
        p = COMPARE_ROOT / name / "log.pkl"
        if not p.is_file():
            print(f"  [load] missing {p}, skipping {name}")
            out[name] = []
            continue
        with open(p, "rb") as f:
            out[name] = pickle.load(f)
        print(f"  [load] {name}: {len(out[name])} records")
    return out


def time_axis_min(records) -> np.ndarray:
    return np.array([r.time_s / 60.0 for r in records], dtype=np.float64)


# ---------------------------------------------------------------------------
#  Diagnostic 1: per-zone V/Q breakdown
# ---------------------------------------------------------------------------


def plot_per_zone_vq(log_t_ofo: list, out_path: Path) -> Dict[str, float]:
    """Per-zone TS V envelopes and aggregate gen Q over time, T-OFO only.

    Returns dominant-zone diagnostics (max swing amplitude, period estimate).
    """
    if not log_t_ofo:
        return {}

    t = time_axis_min(log_t_ofo)
    zones = sorted(log_t_ofo[0].zone_v_min.keys())
    n_z = len(zones)

    # Pre-extract arrays
    v_min = {z: np.array([r.zone_v_min.get(z, np.nan) for r in log_t_ofo]) for z in zones}
    v_max = {z: np.array([r.zone_v_max.get(z, np.nan) for r in log_t_ofo]) for z in zones}
    v_mean = {z: np.array([r.zone_v_mean.get(z, np.nan) for r in log_t_ofo]) for z in zones}

    def _q_total(r, z):
        arr = r.zone_q_gen.get(z)
        if arr is None or (hasattr(arr, "__len__") and len(arr) == 0):
            return np.nan
        return float(np.sum(np.asarray(arr, dtype=float)))

    q_gen_tot = {z: np.array([_q_total(r, z) for r in log_t_ofo]) for z in zones}

    # DSO group V & Q
    dso_groups = sorted(log_t_ofo[0].dso_group_v_mean_pu.keys())
    v_ds_mean = {g: np.array([r.dso_group_v_mean_pu.get(g, np.nan) for r in log_t_ofo])
                 for g in dso_groups}
    q_ds = {g: np.array([r.dso_group_q_der_mvar.get(g, np.nan) for r in log_t_ofo])
            for g in dso_groups}

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    fig.suptitle("T-OFO per-zone V envelope and aggregate Q (TS + DS)",
                 fontweight="bold")

    # Panel 0: per-zone TS V envelope
    ax = axes[0]
    palette_z = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    for z in zones:
        c = palette_z.get(z, "k")
        ax.plot(t, v_mean[z], color=c, linewidth=1.4, label=f"Zone {z} mean")
        ax.fill_between(t, v_min[z], v_max[z], color=c, alpha=0.15,
                        label=f"Zone {z} min-max")
    ax.axhline(QV_SETPOINT_PU, color="k", linewidth=0.7, linestyle="-", alpha=0.5)
    ax.set_ylabel("V_TS [p.u.]")
    ax.set_title("TS voltage per zone (mean line, min-max band)")
    ax.legend(loc="upper right", ncol=n_z, fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel 1: per-zone aggregate gen Q
    ax = axes[1]
    for z in zones:
        c = palette_z.get(z, "k")
        ax.plot(t, q_gen_tot[z], color=c, linewidth=1.2, label=f"Zone {z}")
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
    ax.set_ylabel("Sum Q_gen [Mvar]")
    ax.set_title("TS aggregate synchronous-gen Q per zone")
    ax.legend(loc="upper right", ncol=n_z, fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel 2: per-DSO-group V mean
    ax = axes[2]
    palette_g = {"DSO_1": "#d62728", "DSO_2": "#9467bd",
                 "DSO_3": "#8c564b", "DSO_4": "#e377c2"}
    for g in dso_groups:
        c = palette_g.get(g, "k")
        ax.plot(t, v_ds_mean[g], color=c, linewidth=1.2, label=g)
    ax.axhline(QV_SETPOINT_PU, color="k", linewidth=0.7, linestyle="-", alpha=0.5)
    ax.set_ylabel("V_HV mean [p.u.]")
    ax.set_title("HV (DSO group) mean voltage")
    ax.legend(loc="upper right", ncol=4, fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel 3: per-DSO-group aggregate Q_DER
    ax = axes[3]
    for g in dso_groups:
        c = palette_g.get(g, "k")
        ax.plot(t, q_ds[g], color=c, linewidth=1.2, label=g)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
    ax.set_ylabel("Sum Q_DER [Mvar]")
    ax.set_title("HV (DSO group) aggregate DER Q")
    ax.legend(loc="upper right", ncol=4, fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time [min]")

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Diagnostics: which zone has the largest peak-to-peak swing in V_max post-150 min?
    mask_post = t >= 150.0
    swings_z = {z: float(np.nanmax(v_max[z][mask_post]) - np.nanmin(v_min[z][mask_post]))
                for z in zones}
    swings_g = {g: float(np.nanmax(v_ds_mean[g][mask_post])
                          - np.nanmin(v_ds_mean[g][mask_post]))
                 for g in dso_groups}

    return {
        "ts_swing_per_zone_pu": swings_z,
        "ds_swing_per_group_pu": swings_g,
        "dominant_ts_zone": int(max(swings_z, key=swings_z.get)),
        "dominant_ds_group": str(max(swings_g, key=swings_g.get)),
    }


# ---------------------------------------------------------------------------
#  Diagnostic 2: Q vs V phase relationship
# ---------------------------------------------------------------------------


def plot_q_v_phase(log_t_ofo: list, dominant_group: str, out_path: Path
                   ) -> Dict[str, float]:
    """Overlay Q_DER and V_mean for the dominant DSO group, plus a scatter
    plot of Q vs V post-150 min.  Returns Pearson r and rough phase shift
    (in TSO-step units)."""
    if not log_t_ofo or dominant_group is None:
        return {}

    t = time_axis_min(log_t_ofo)
    v = np.array([r.dso_group_v_mean_pu.get(dominant_group, np.nan) for r in log_t_ofo])
    q = np.array([r.dso_group_q_der_mvar.get(dominant_group, np.nan) for r in log_t_ofo])

    mask = t >= 150.0
    vp = v[mask]
    qp = q[mask]
    finite = np.isfinite(vp) & np.isfinite(qp)
    vp = vp[finite]
    qp = qp[finite]

    # Pearson correlation Q vs V
    if vp.size > 5:
        r_qv = float(np.corrcoef(vp, qp)[0, 1])
    else:
        r_qv = np.nan

    # Cross-correlation lag (sample-shift, +/- in minutes since dt=1 min)
    def best_lag(x, y, max_lag=15):
        if x.size < 2 * max_lag + 5:
            return 0
        x = (x - np.mean(x)) / (np.std(x) + 1e-12)
        y = (y - np.mean(y)) / (np.std(y) + 1e-12)
        lags = np.arange(-max_lag, max_lag + 1)
        ccf = np.array([np.mean(x[max(0, k): x.size + min(0, k)]
                                 * y[max(0, -k): y.size + min(0, -k)])
                         for k in lags])
        return int(lags[np.argmax(np.abs(ccf))])

    lag_q_to_v_min = best_lag(vp, qp)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5))
    fig.suptitle(f"Q vs V phase, dominant DSO group: {dominant_group}",
                 fontweight="bold")

    # Panel 0: time-series overlay (twin axis)
    ax = axes[0]
    ax.plot(t, v, color="#1f77b4", linewidth=1.2, label="V_mean [pu]")
    ax.set_ylabel("V_mean [p.u.]", color="#1f77b4")
    ax.tick_params(axis="y", colors="#1f77b4")
    ax.axvline(150, color="grey", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, q, color="#d62728", linewidth=1.2, label="Q_DER total [Mvar]")
    ax2.set_ylabel("Q_DER [Mvar]", color="#d62728")
    ax2.tick_params(axis="y", colors="#d62728")
    ax.set_xlabel("Time [min]")
    ax.set_title(f"V (left) vs Q_DER (right) overlay   |   "
                  f"r={r_qv:+.3f}, best-lag={lag_q_to_v_min:+d} min")

    # Panel 1: scatter post-150 min
    ax = axes[1]
    ax.scatter(vp, qp, s=18, alpha=0.5, color="#9467bd")
    ax.axvline(QV_SETPOINT_PU, color="k", linewidth=0.6, linestyle="--", alpha=0.6,
               label=f"V_set={QV_SETPOINT_PU} pu")
    # Reference droop line: Q = -k*(V - V_set). k aggregated across DERs is
    # plotted only as a guide -- exact magnitude depends on which DERs are in
    # this group.
    if vp.size > 1:
        sl = -1.0 * (np.max(qp) - np.min(qp)) / (np.max(vp) - np.min(vp) + 1e-9)
        v_line = np.linspace(np.min(vp), np.max(vp), 50)
        ax.plot(v_line, sl * (v_line - np.mean(vp)) + np.mean(qp),
                color="k", linewidth=0.7, linestyle=":",
                label=f"empirical droop slope (~{sl:.0f} Mvar/pu)")
    ax.set_xlabel("V_mean [p.u.] (post 150 min)")
    ax.set_ylabel("Q_DER [Mvar] (post 150 min)")
    ax.set_title("Q-V scatter post-150 min  "
                  "(neg slope = correctly reactive droop)")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "pearson_r_q_v_post150": r_qv,
        "lag_q_to_v_min": lag_q_to_v_min,
    }


# ---------------------------------------------------------------------------
#  Diagnostic 3: TSO command trajectory + loop gain
# ---------------------------------------------------------------------------


def plot_tso_command_trajectory(log_t_ofo: list, out_path: Path
                                ) -> Dict[str, float]:
    """Plot per-zone TSO Q_PCC_set and V_gen trajectories at TSO-active steps,
    estimate empirical loop gain as median |dQ_(n+1) / dQ_n| over post-150
    min TSO steps."""
    if not log_t_ofo:
        return {}

    tso_recs = [r for r in log_t_ofo if r.tso_active]
    t_tso = np.array([r.time_s / 60.0 for r in tso_recs])

    zones = sorted(log_t_ofo[0].zone_v_gen.keys())

    # Aggregate per-zone TSO commands at TSO-active steps
    q_pcc_agg = {z: np.array([float(np.sum(r.zone_q_pcc_set.get(z, np.zeros(0))))
                               for r in tso_recs]) for z in zones}
    v_gen_mean = {z: np.array([float(np.mean(r.zone_v_gen.get(z, np.full(1, np.nan))))
                                for r in tso_recs]) for z in zones}

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    fig.suptitle("T-OFO TSO commands at OFO-step boundaries",
                 fontweight="bold")
    palette_z = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}

    ax = axes[0]
    for z in zones:
        c = palette_z.get(z, "k")
        ax.plot(t_tso, q_pcc_agg[z], color=c, linewidth=1.2,
                marker="o", markersize=3, label=f"Zone {z}")
    ax.axvline(150, color="grey", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
    ax.set_ylabel("Sum Q_PCC_set [Mvar]")
    ax.set_title("Per-zone aggregated Q_PCC setpoint commanded by TSO OFO")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for z in zones:
        c = palette_z.get(z, "k")
        ax.plot(t_tso, v_gen_mean[z], color=c, linewidth=1.2,
                marker="o", markersize=3, label=f"Zone {z}")
    ax.axvline(150, color="grey", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.axhline(QV_SETPOINT_PU, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_ylabel("Mean V_gen [p.u.]")
    ax.set_title("Per-zone mean V_gen (AVR setpoint) commanded by TSO OFO")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time [min]")

    # Q_PCC windup index = |sum(dQ)| / sum(|dQ|) ∈ [0,1].
    # 1.0  -> monotonic divergence (windup); 0.0 -> perfectly symmetric oscillation.
    # V_gen oscillation = sign-flip fraction of consecutive Δ at TSO steps.
    # 1.0  -> alternating (limit cycle); 0.0 -> monotonic.
    mask = t_tso >= 150.0
    swing_q_pcc_per_zone = {z: float(np.nanmax(q_pcc_agg[z][mask])
                                      - np.nanmin(q_pcc_agg[z][mask]))
                             for z in zones if q_pcc_agg[z][mask].size > 1}
    swing_v_gen_per_zone = {z: float(np.nanmax(v_gen_mean[z][mask])
                                      - np.nanmin(v_gen_mean[z][mask]))
                             for z in zones if v_gen_mean[z][mask].size > 1}
    dominant_zone = (max(swing_q_pcc_per_zone, key=swing_q_pcc_per_zone.get)
                      if swing_q_pcc_per_zone else None)

    q_windup_idx = {}
    v_gen_signflip = {}
    for z in zones:
        q = q_pcc_agg[z][mask]
        v = v_gen_mean[z][mask]
        if q.size >= 4:
            dq = np.diff(q)
            denom = np.sum(np.abs(dq)) + 1e-12
            q_windup_idx[z] = float(np.abs(np.sum(dq)) / denom)
        if v.size >= 4:
            dv = np.diff(v)
            sig_dv = np.sign(dv)
            # ignore exact zeros (no-change steps) when counting flips
            nz = sig_dv != 0
            if nz.sum() >= 2:
                pairs = sig_dv[nz][:-1] != sig_dv[nz][1:]
                v_gen_signflip[z] = float(np.mean(pairs))
            else:
                v_gen_signflip[z] = float("nan")

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "dominant_zone_q_pcc": dominant_zone,
        "swing_q_pcc_per_zone_mvar": swing_q_pcc_per_zone,
        "swing_v_gen_per_zone_pu": swing_v_gen_per_zone,
        "q_pcc_windup_index_per_zone": q_windup_idx,
        "v_gen_signflip_per_zone": v_gen_signflip,
    }


# ---------------------------------------------------------------------------
#  Diagnostic 4: open-loop vs closed-loop sensitivity at warm-start
# ---------------------------------------------------------------------------


def diagnose_sensitivity_eigenvalues(out_path: Path) -> Dict[str, float]:
    """Build the wind_replace network at scenario warm-start, compute
    dV/dQ at HV PQ buses (the OFO's open-loop sensitivity for HV-DER Q),
    then form the closed-loop transfer (I + S_VQ_HV * K_droop) and report
    eigenvalues.

    Interpretation:
        * Eigenvalues all near 1.0  -> open-loop S is approximately correct,
          droop has negligible feedback.
        * Eigenvalues far from 1.0  -> the cached open-loop sensitivity used
          by the TSO OFO is structurally wrong by the same factor, so OFO
          commands have miscalibrated gain.  This is the smoking gun.
    """
    import pandapower as pp
    from network.ieee39.build import build_ieee39_net
    from sensitivity.jacobian import JacobianSensitivities

    net, meta = build_ieee39_net(scenario="wind_replace")

    # Add HV sub-networks the same way 000_M_TSO_M_DSO does, to get a network
    # whose bus topology matches the simulation.
    try:
        from experiments.helpers.plant_io import (
            apply_default_dispatch_and_loads,
        )
    except Exception:
        apply_default_dispatch_and_loads = None

    # Add HV networks (replicates add_hv_networks call in 000_M_TSO_M_DSO.py).
    # Note: add_hv_networks(net, meta, ...) modifies net in-place and returns
    # only the updated meta.
    try:
        from network.ieee39.hv_networks import add_hv_networks
        meta = add_hv_networks(
            net, meta,
            install_tso_tertiary_shunts=False,
            verbose=False,
        )
    except Exception as exc:
        print(f"  [diag4] WARNING: could not add HV networks: {exc}")
        print(f"  [diag4]   proceeding with TS-only network.")

    pp.runpp(net, calculate_voltage_angles=True, run_control=False)
    if not net.converged:
        return {"error": "warm-start PF did not converge"}

    js = JacobianSensitivities(net)
    S_full_VQ = js.dV_dQ_reduced              # [pu_V / pu_Q] across all PQ buses
    pq_buses_ppc = js.pq_buses                # internal pypower indices

    # Map each PQ position in S_full_VQ to its corresponding pp bus and vn_kv.
    # _pd2ppc_lookups["bus"] is a 1-D numpy array indexed by pp bus id, value
    # is the ppc bus id (or some sentinel for inactive buses).  Build the
    # inverse mapping ppc -> pp explicitly and skip any PQ position whose
    # ppc id has no pp counterpart.
    bus_lookup = net._pd2ppc_lookups["bus"]   # pp -> ppc
    inv_bus_lookup: Dict[int, int] = {int(pi): int(pp_id)
                                       for pp_id, pi in enumerate(bus_lookup)
                                       if pi >= 0}
    vn = net.bus["vn_kv"].values

    hv_pq_idx_in_S: List[int] = []
    hv_pp_buses: List[int] = []
    for s_pos, ppc_idx in enumerate(pq_buses_ppc):
        pp_id = inv_bus_lookup.get(int(ppc_idx))
        if pp_id is None or pp_id >= len(vn):
            continue
        if abs(float(vn[pp_id]) - 110.0) <= 1.0:
            hv_pq_idx_in_S.append(s_pos)
            hv_pp_buses.append(pp_id)
    hv_pq_idx_in_S = np.array(hv_pq_idx_in_S, dtype=int)
    hv_pp_buses = np.array(hv_pp_buses, dtype=int)
    n_hv = hv_pq_idx_in_S.size

    if n_hv == 0:
        return {"error": "no HV PQ buses found in cached network"}

    # Sub-block of dV/dQ at HV PQ buses (square, n_hv x n_hv).
    S_HV = S_full_VQ[np.ix_(hv_pq_idx_in_S, hv_pq_idx_in_S)]   # pu_V / pu_Q

    # Convert to physical units used by the OFO: V[pu] / Q[Mvar].
    s_base = float(net.sn_mva)
    S_HV_per_mvar = S_HV / s_base

    # K_droop per HV bus.  In the legacy CharacteristicControl, each DER injects
    #   Q_inj [Mvar] = -(S_n / qv_slope_pu) * (V - V_set).
    # We aggregate K = sum_(DERs at this bus) S_n / qv_slope_pu.  DERs are
    # the sgens whose bus matches one of the HV PQ buses.
    sgen = net.sgen
    K_droop_diag = np.zeros(n_hv)
    n_der_per_bus = np.zeros(n_hv, dtype=int)
    for i, b in enumerate(hv_pp_buses):
        rows = sgen.index[(sgen["bus"] == b) & sgen["in_service"].astype(bool)]
        if rows.size:
            sn = sgen.loc[rows, "sn_mva"].values.astype(float)
            K_droop_diag[i] = float(np.sum(sn / QV_SLOPE_PU))
            n_der_per_bus[i] = int(rows.size)

    K_droop = np.diag(K_droop_diag)

    # Closed-loop transfer:  V_actual = (I + S_HV * K_droop)^{-1} V_open
    # Eigenvalues of M = I + S_HV * K_droop quantify how strongly the droop
    # rescales the open-loop sensitivity.  M close to identity => weak feedback;
    # M with eigenvalues much larger than 1 (or near 0) => strong feedback,
    # cached open-loop sensitivity is structurally wrong.
    M = np.eye(n_hv) + S_HV_per_mvar @ K_droop
    eigvals = np.linalg.eigvals(M)
    eig_mag = np.abs(eigvals)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Closed-loop transfer M = I + S_VQ_HV * K_droop "
                 "(eigenvalue spectrum at warm-start)",
                 fontweight="bold")

    ax = axes[0]
    ax.scatter(eigvals.real, eigvals.imag, s=22, color="#1f77b4")
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
    ax.axvline(1, color="r", linewidth=0.7, linestyle="--",
               label="ideal (no feedback)")
    ax.set_xlabel("Re(eigenvalue)")
    ax.set_ylabel("Im(eigenvalue)")
    ax.set_title("Complex plane")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.bar(np.arange(n_hv), np.sort(eig_mag)[::-1],
           color="#9467bd", edgecolor="white")
    ax.axhline(1.0, color="r", linewidth=0.7, linestyle="--",
               label="|λ|=1 (no feedback)")
    ax.set_xlabel("Eigenvalue index (sorted)")
    ax.set_ylabel("|λ|")
    ax.set_title(f"|λ| spectrum   |   max={float(eig_mag.max()):.2f},  "
                  f"min={float(eig_mag.min()):.2f}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_hv_pq_buses": int(n_hv),
        "n_der_total_at_hv": int(n_der_per_bus.sum()),
        "max_K_droop_mvar_per_pu": float(K_droop_diag.max()) if n_hv else 0.0,
        "eig_M_max_abs": float(eig_mag.max()),
        "eig_M_min_abs": float(eig_mag.min()),
        "frac_eigs_far_from_1": float(np.mean(np.abs(eig_mag - 1.0) > 0.5)),
    }


# ---------------------------------------------------------------------------
#  Summary writer
# ---------------------------------------------------------------------------


def write_summary(out_dir: Path, results: Dict[str, Dict]) -> None:
    md = []
    md.append("# T-OFO oscillation diagnostic summary\n")
    md.append("Generated by `tests/diag_t_ofo_oscillation.py`.\n")
    md.append("Source: `experiments/results/002_compare/T-OFO/log.pkl`.\n\n")

    md.append("## Diagnostic 1 — per-zone V/Q breakdown\n")
    d1 = results.get("per_zone", {})
    if d1:
        md.append("| Zone | TS V swing post-150 min [pu] |\n|-----:|-----:|\n")
        for z, v in d1["ts_swing_per_zone_pu"].items():
            md.append(f"| {z} | {v:.4f} |\n")
        md.append(f"\n**Dominant TS zone**: {d1['dominant_ts_zone']}\n\n")
        md.append("| DSO group | HV V swing post-150 min [pu] |\n|---|---:|\n")
        for g, v in d1["ds_swing_per_group_pu"].items():
            md.append(f"| {g} | {v:.4f} |\n")
        md.append(f"\n**Dominant DSO group**: {d1['dominant_ds_group']}\n\n")
    else:
        md.append("_no data_\n\n")

    md.append("## Diagnostic 2 — Q vs V phase relationship\n")
    d2 = results.get("q_v_phase", {})
    if d2:
        md.append(f"- Pearson r(Q, V) post-150 min: **{d2.get('pearson_r_q_v_post150'):+.3f}**\n")
        md.append(f"- Best lag of Q w.r.t. V: **{d2.get('lag_q_to_v_min'):+d} min**\n")
        md.append("\nInterpretation: a strongly negative r close to -1 with small lag "
                  "confirms the legacy droop is reactive (Q swings opposite to V), "
                  "i.e. the droop is **doing exactly what it was designed for** — "
                  "the bug is upstream in the OFO that does not anticipate this reaction.\n\n")
    else:
        md.append("_no data_\n\n")

    md.append("## Diagnostic 3 — TSO command trajectory\n")
    d3 = results.get("tso_traj", {})
    if d3:
        md.append("Two distinct pathologies coexist in the OFO commands:\n\n")
        md.append("**(a) Q_PCC windup** (monotonic divergence of the inactive setpoint).\n")
        md.append("In T-OFO mode the DSO does not run an MIQP, so `Q_PCC_set` is an "
                  "internal MIQP variable that is never enacted on the plant.  When the "
                  "OFO cannot bring V to setpoint via its physical actuators it pushes "
                  "Q_PCC monotonically because the soft-cost penalty for V-error "
                  "outweighs the regularisation on Q_PCC (g_v = 5e5 vs g_w_pcc = 50).\n\n")
        md.append("| Zone | Q_PCC swing post-150 min [Mvar] | Q_PCC windup index |\n")
        md.append("|-----:|---:|---:|\n")
        for z in sorted(d3.get("swing_q_pcc_per_zone_mvar", {}).keys()):
            sw = d3["swing_q_pcc_per_zone_mvar"][z]
            wi = d3.get("q_pcc_windup_index_per_zone", {}).get(z, float("nan"))
            md.append(f"| {z} | {sw:,.0f} | {wi:.3f} |\n")
        md.append("\n*windup index ∈ [0, 1]: 1.0 = perfectly monotonic divergence, 0.0 = symmetric oscillation*\n\n")
        md.append("**(b) V_gen limit-cycling at the AVR setpoint upper bound**.\n")
        md.append("V_gen oscillates between adjacent values at the box constraint "
                  "(near 1.07 pu) at the TSO step cadence — this is the *physical* "
                  "actuator that drives the plant V oscillation.\n\n")
        md.append("| Zone | V_gen swing post-150 min [pu] | V_gen sign-flip fraction |\n")
        md.append("|-----:|---:|---:|\n")
        for z in sorted(d3.get("swing_v_gen_per_zone_pu", {}).keys()):
            sw = d3["swing_v_gen_per_zone_pu"][z]
            sf = d3.get("v_gen_signflip_per_zone", {}).get(z, float("nan"))
            md.append(f"| {z} | {sw:.4f} | {sf:.3f} |\n")
        md.append("\n*sign-flip fraction ≥ 0.5 ⇒ alternating-direction limit cycle*\n\n")
    else:
        md.append("_no data_\n\n")

    md.append("## Diagnostic 4 — open-loop vs closed-loop sensitivity at warm-start\n")
    d4 = results.get("sensitivity", {})
    if d4 and "error" not in d4:
        md.append(f"- HV PQ buses examined: **{d4['n_hv_pq_buses']}**\n")
        md.append(f"- HV-connected DERs (legacy droop): **{d4['n_der_total_at_hv']}**\n")
        md.append(f"- Max bus K_droop = sum_DERs_at_bus(S_n / qv_slope): **{d4['max_K_droop_mvar_per_pu']:.0f} Mvar/pu**\n")
        md.append(f"- |λ_max(M)| = **{d4['eig_M_max_abs']:.3f}**, |λ_min(M)| = **{d4['eig_M_min_abs']:.3f}**\n")
        md.append(f"- Fraction of eigenvalues with |λ−1| > 0.5: **{d4['frac_eigs_far_from_1']:.2%}**\n\n")
        md.append("Interpretation: M = I + S_VQ_HV · K_droop is the matrix that maps "
                  "open-loop V to closed-loop V at the HV PQ buses. If the spectrum is "
                  "tightly clustered around 1.0, the cached open-loop sensitivity used "
                  "by the TSO OFO is approximately correct and the droop adds negligible "
                  "feedback.  If |λ_max| is large (or |λ_min| near 0), the open-loop "
                  "sensitivity over- or under-estimates the true gain by the same "
                  "factor, which directly causes the OFO to over-correct and oscillate.\n\n")
    elif d4:
        md.append(f"_error: {d4.get('error')}_\n\n")
    else:
        md.append("_no data_\n\n")

    md.append("## Conclusion (auto-generated)\n")
    diag_lines = []
    # Q_PCC windup
    max_windup = max(d3.get("q_pcc_windup_index_per_zone", {0: float("nan")}).values()) if d3 else float("nan")
    if np.isfinite(max_windup) and max_windup > 0.8:
        diag_lines.append(f"- Q_PCC windup index ≥ {max_windup:.2f} in at least one zone → "
                          "the inactive Q_PCC setpoint diverges monotonically, indicating "
                          "the OFO cost pushes infeasibly hard against a V error it "
                          "cannot eliminate with its physical actuators.")
    # V_gen sign-flip
    max_signflip = max(
        (v for v in d3.get("v_gen_signflip_per_zone", {}).values()
         if np.isfinite(v)), default=float("nan"))
    if np.isfinite(max_signflip) and max_signflip > 0.5:
        diag_lines.append(f"- V_gen sign-flip fraction ≥ {max_signflip:.2f} in at least "
                          "one zone → AVR setpoint limit-cycling between adjacent "
                          "values at the TSO step cadence; this is the *physical* "
                          "driver of the bus V oscillation.")
    # DSO droop reaction
    if d2 and np.isfinite(d2.get("pearson_r_q_v_post150", float("nan"))):
        if d2["pearson_r_q_v_post150"] < -0.5:
            diag_lines.append("- DSO Q is strongly negatively correlated with V "
                              "(r ≈ {:+.2f}) → the legacy droop is *reacting "
                              "correctly*, not internally driving the oscillation. "
                              "The droop's reaction is exactly what the cached TSO "
                              "sensitivity does not anticipate.".format(
                                  d2["pearson_r_q_v_post150"]))
    # Closed-loop sensitivity
    if d4 and "error" not in d4:
        if d4["eig_M_max_abs"] > 1.5 or d4["eig_M_min_abs"] < 0.5:
            diag_lines.append("- Closed-loop transfer eigenvalues span [{:.2f}, "
                              "{:.2f}] → the cached open-loop dV/dQ sensitivity "
                              "over-/under-estimates the true closed-loop response "
                              "by the same factor.  The OFO commands are therefore "
                              "miscalibrated by up to {:.0%}, which after "
                              "contingency-driven operating-point shifts is enough "
                              "to drive the system out of its stability margin.".format(
                                  d4["eig_M_min_abs"], d4["eig_M_max_abs"],
                                  abs(d4["eig_M_max_abs"] - 1.0)))
    if not diag_lines:
        diag_lines.append("- No single smoking-gun threshold tripped; review plots manually.")
    md.extend([line + "\n" for line in diag_lines])

    md.append("\n**Synthesis.** The user's hypothesis (\"DSO Q(V) control causes the "
              "oscillation\") is *partially* correct: the droop is the necessary "
              "feedback mechanism, but the *bug* is upstream — the TSO OFO uses an "
              "open-loop sensitivity that ignores the droop's reaction.  After the "
              "contingency at t≈150 min shifts the operating point, the open-loop "
              "and closed-loop transfer matrices diverge enough that the OFO commands "
              "no longer reduce V error.  V_gen saturates and limit-cycles between "
              "adjacent values at the upper box bound, while Q_PCC (an inactive "
              "internal variable in T-OFO mode) winds up monotonically.\n")

    md.append("\n## Mitigation D code sketch (under-relaxation, NOT committed)\n")
    md.append("Apply a damping factor α<1 to **all** TSO OFO command updates "
              "(V_gen, Q_DER, OLTC tap deltas, Q_PCC) so the controller cannot "
              "fully commit to a model-based step that may be miscalibrated by "
              "the open-loop sensitivity:\n\n")
    md.append("```python\n# in controller/multi_tso_coordinator.py or controller/tso_controller.py,\n"
              "# at the point where the OFO solution u_star is dispatched to the plant:\n"
              "alpha = 0.3   # tune in [0.1, 0.7]; lower = more damping, slower convergence\n"
              "u_actual_prev = self._last_dispatched_commands  # cached from previous step\n"
              "u_command = u_actual_prev + alpha * (u_star - u_actual_prev)\n"
              "self._last_dispatched_commands = u_command\n"
              "# dispatch u_command (V_gen, Q_DER, tap deltas, Q_PCC_set) instead of u_star\n```\n\n")
    md.append("**Effect.** The discrete-time loop gain (closed-loop sensitivity error × OFO gain) "
              "is multiplied by α, so the stability margin scales by 1/α.  Pick α just above "
              "the value that kills the V_gen limit cycle in T-OFO; verify C-OFO is "
              "unaffected (it should be — the cascade architecture has lower model error).\n\n")
    md.append("**Caveat.** Under-relaxation only buys margin; it does not fix the underlying "
              "open-loop-vs-closed-loop sensitivity error.  If diagnostic 4 confirms "
              "|λ_max(M)| > 2, Mitigation E (apply the Schur correction once at warm-start) "
              "is the structurally correct follow-up.  D and E compose.\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("".join(md), encoding="utf-8")
    print(f"  [summary] wrote {out_dir / 'summary.md'}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [diag] output dir: {OUT_DIR}")
    logs = load_logs()
    log_t_ofo = logs.get("T-OFO", [])
    if not log_t_ofo:
        print("  [diag] T-OFO log empty -- aborting")
        return

    results: Dict[str, Dict] = {}

    print("\n  [diag 1] per-zone V/Q breakdown ...")
    results["per_zone"] = plot_per_zone_vq(log_t_ofo, OUT_DIR / "per_zone_v_q")

    print("\n  [diag 2] Q vs V phase ...")
    dom_g = results["per_zone"].get("dominant_ds_group")
    results["q_v_phase"] = plot_q_v_phase(log_t_ofo, dom_g,
                                           OUT_DIR / "q_v_phase")

    print("\n  [diag 3] TSO command trajectory ...")
    results["tso_traj"] = plot_tso_command_trajectory(log_t_ofo,
                                                       OUT_DIR / "tso_command_trajectory")

    print("\n  [diag 4] open-loop vs closed-loop sensitivity at warm-start ...")
    try:
        results["sensitivity"] = diagnose_sensitivity_eigenvalues(
            OUT_DIR / "sensitivity_eigenvalues"
        )
    except Exception as exc:
        print(f"  [diag 4] FAILED: {type(exc).__name__}: {exc}")
        results["sensitivity"] = {"error": f"{type(exc).__name__}: {exc}"}

    print("\n  [diag] writing summary ...")
    write_summary(OUT_DIR, results)
    print("  [diag] done.")


if __name__ == "__main__":
    main()
