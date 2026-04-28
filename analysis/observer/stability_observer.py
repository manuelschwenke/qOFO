"""
Stability Observer for Profile-Driven Simulation
================================================

Passive, non-invasive companion to the MIQP-OFO controller execution.
Instead of Monte Carlo sampling of synthetic operating points, this
observer attaches to the running ``MultiTSOCoordinator`` and records
the stability-relevant quantities at every cross-sensitivity refresh
during a normal simulation run.

The recorded trajectory — already computed at every timestep by the
existing pipeline — is post-processed by the aggregation tools from
``stability_tuning`` to produce the same Pareto plots and per-block
histograms, but over operating points that were actually visited by
the simulation rather than sampled from a prior.

Why this is better than Monte Carlo for an applications thesis
-------------------------------------------------------------
- Operating points are driven by real 15-minute profiles (load, PV,
  wind), capturing the correlation structure that random sampling
  ignores (e.g. high load correlates with low PV at evening peak).
- N-1 contingency tests already fire during the simulation; the
  observer automatically captures their effect on H.
- Zero extra computation at runtime: H is already built by
  ``coordinator.compute_cross_sensitivities()``, we just read it.
- Results are directly tied to the scenarios the thesis validates.

Usage (minimal, non-invasive)
-----------------------------
Insert one line in ``run_multi_tso_dso`` after the coordinator is
created, and one line inside the main loop right after a refresh:

    # After coordinator = MultiTSOCoordinator(...)
    observer = StabilityObserver(coordinator, zone_defs, config)

    # Inside the main loop, after coordinator.step(...) with refresh_H=True:
    if refresh_H:
        observer.record(time_s=time_s)

    # At end of simulation:
    observer.write_results(config.result_dir)

The observer uses zero bandwidth when ``refresh_H`` is False, and
negligible bandwidth when it fires (one matrix copy + two eigvalsh
per zone).

Author: Manuel Schwenke (drafted with Claude 2026-04-18)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# Import the core tuning primitives — these do not depend on the
# coordinator or network, only on H matrices and layout.
from .stability_tuning import (
    BlockLayout,
    StabilityResult,
    compute_min_gw_per_block,
    compute_min_gw_lmi,
    aggregate_monte_carlo,
)

__all__ = [
    "ObservationRecord",
    "ZoneTrajectory",
    "StabilityObserver",
    "write_trajectory_report",
    "plot_trajectory",
]


# --------------------------------------------------------------------------- #
#  Record types
# --------------------------------------------------------------------------- #

@dataclass
class ObservationRecord:
    """Single observation captured at one refresh step of one zone."""
    time_s: float
    zone_id: int
    result: StabilityResult            # output of compute_min_gw_per_block
    H_op_norm: float                   # ‖H_V‖_op (diagnostic)
    H_Q_op_norm: float                 # ‖H_Q‖_op (diagnostic)


@dataclass
class ZoneTrajectory:
    """Trajectory of observations for one zone over the full simulation."""
    zone_id: int
    layout: BlockLayout
    records: List[ObservationRecord] = field(default_factory=list)

    def times_s(self) -> NDArray[np.float64]:
        return np.array([r.time_s for r in self.records], dtype=np.float64)

    def gw_block_trajectory(self) -> NDArray[np.float64]:
        """Per-block g_w^min over time, shape (T, 4)."""
        return np.stack([r.result.gw_min_block for r in self.records], axis=0)

    def gw_full_trajectory(self) -> NDArray[np.float64]:
        """Per-actuator g_w^min over time, shape (T, m)."""
        return np.stack([r.result.gw_min_full for r in self.records], axis=0)

    def op_norm_trajectory(self) -> NDArray[np.float64]:
        """‖M‖_op over time (diagnostic), shape (T,)."""
        return np.array([r.result.op_norm for r in self.records],
                        dtype=np.float64)

    def aggregate(
        self,
        statistic: str = "max",
        percentile: float = 95.0,
    ) -> NDArray[np.float64]:
        """Reduce trajectory to a single per-actuator g_w* vector."""
        return aggregate_monte_carlo(
            [r.result for r in self.records],
            statistic=statistic,
            percentile=percentile,
        )

    def aggregate_haberle(
        self,
        statistic: str = "percentile",
        percentile: float = 95.0,
    ) -> float:
        """
        Haeberle 2021 floor (constrained projected gradient with explicit
        box on Delta u): ``g_w >= ||M||_op / 2``.  Returns one scalar per zone
        — the smallest proximal weight that any actuator must satisfy.

        This is generally tighter than the unconstrained spectral-gap floor
        ``g_w >= ||M||_op - lam_min(M)`` whenever the cost-Hessian eigenspectrum
        is rank-deficient (typical for power-flow Jacobians).

        Citation: Haeberle, Hauswirth, Ortmann, Bolognani, Doerfler (2021)
        L-CSS 5(1):343, Theorem 1 (descent step for projected gradient).
        Verify the constant 1/2 against the published condition before
        using as a thesis claim.
        """
        op_norms = self.op_norm_trajectory()
        if op_norms.size == 0:
            return float("nan")
        if statistic == "max":
            val = float(op_norms.max())
        elif statistic == "percentile":
            val = float(np.percentile(op_norms, percentile))
        elif statistic == "mean":
            val = float(op_norms.mean())
        else:
            raise ValueError(f"Unknown statistic '{statistic}'")
        return val / 2.0


# --------------------------------------------------------------------------- #
#  Observer
# --------------------------------------------------------------------------- #

class StabilityObserver:
    """
    Passive stability observer that attaches to a running coordinator.

    Collects ``StabilityResult`` snapshots at each sensitivity refresh
    and builds per-zone trajectories.  Can be serialised to JSON or
    plotted to image files at the end of the simulation.

    Parameters
    ----------
    coordinator : MultiTSOCoordinator
        The running coordinator (typed ``Any`` to avoid a circular import
        with the qOFO_GH package).  Must expose
        ``get_H_block(zone_id, zone_id)`` which returns the in-zone
        sensitivity matrix.
    zone_defs : dict[int, ZoneDefinition]
        The same zone definitions passed to the coordinator.  Used to
        determine block layouts and the output row scaling.
    g_v, g_q : float
        Objective weights from config (voltage tracking, Q_gen capability).
        Held fixed across the observation window.
    ratio_priors : sequence of 4 floats, optional
        Block-weight ratios (DER, PCC, V_gen, OLTC).  Default reflects
        physical prior that OLTC > V_gen > PCC > DER.
    safety_margin : float
        Fractional inflation of the computed minimum g_w.
    tracked_zone_ids : sequence of int, optional
        Restrict observation to a subset of zones.  Default: all zones.
    method : {"block", "lmi"}
        Which tuning function to evaluate at each step.  ``"block"`` is
        the thesis-default.  ``"lmi"`` runs the Gershgorin per-actuator
        bound (slower but tighter).

    Notes
    -----
    The observer assumes the coordinator's internal ``_H_blocks`` have
    been refreshed at the current operating point before ``record()``
    is called (i.e. call it right after ``coordinator.step(...,
    recompute_cross_sensitivities=True)``).  If ``_H_blocks`` is stale,
    the observer silently records a stale snapshot — this is by design
    so it matches the controller's own view of the plant.
    """

    def __init__(
        self,
        coordinator: Any,
        zone_defs: Dict[int, Any],
        *,
        g_v: float,
        g_q: float,
        ratio_priors: Sequence[float] = (1.0, 2.0, 3.0, 10.0),
        safety_margin: float = 0.3,
        tracked_zone_ids: Optional[Sequence[int]] = None,
        method: str = "lmi",
    ) -> None:
        if method not in ("block", "lmi"):
            raise ValueError(f"method must be 'block' or 'lmi', got '{method}'")
        self.coordinator = coordinator
        self.zone_defs = zone_defs
        self.g_v = float(g_v)
        self.g_q = float(g_q)
        self.ratio_priors = tuple(ratio_priors)
        self.safety_margin = float(safety_margin)
        self.method = method

        if tracked_zone_ids is None:
            tracked_zone_ids = sorted(zone_defs.keys())
        self.tracked_zone_ids: Tuple[int, ...] = tuple(tracked_zone_ids)

        # Build per-zone layouts once.  Each zone's layout is fixed for the
        # whole simulation (n_der, n_pcc, n_gen, n_oltc do not change after
        # init — N-1 contingencies zero rows/columns but do not resize).
        self._layouts: Dict[int, BlockLayout] = {}
        for z in self.tracked_zone_ids:
            zd = zone_defs[z]
            self._layouts[z] = BlockLayout(
                n_der=len(zd.tso_der_indices),
                n_pcc=len(zd.pcc_trafo_indices),
                n_gen=len(zd.gen_indices),
                n_oltc=len(zd.oltc_trafo_indices),
            )

        # Trajectories, one per zone.
        self.trajectories: Dict[int, ZoneTrajectory] = {
            z: ZoneTrajectory(zone_id=z, layout=self._layouts[z])
            for z in self.tracked_zone_ids
        }

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record(self, time_s: float) -> None:
        """Capture one snapshot per tracked zone at the current operating point."""
        for z in self.tracked_zone_ids:
            try:
                H = self.coordinator.get_H_block(z, z)
            except (KeyError, AttributeError):
                # No in-zone block available (e.g. coordinator not yet
                # refreshed).  Silently skip this zone for this step.
                continue
            if H is None:
                continue

            H_V, H_Q, W_V, W_Q = self._split_H(z, np.asarray(H))
            if H_V.size == 0:
                continue

            if self.method == "block":
                res = compute_min_gw_per_block(
                    g_v=self.g_v, g_q=self.g_q,
                    H_V=H_V, H_Q=H_Q, W_V=W_V, W_Q=W_Q,
                    layout=self._layouts[z],
                    ratio_priors=self.ratio_priors,
                    safety_margin=self.safety_margin,
                )
            else:
                res = compute_min_gw_lmi(
                    g_v=self.g_v, g_q=self.g_q,
                    H_V=H_V, H_Q=H_Q, W_V=W_V, W_Q=W_Q,
                    layout=self._layouts[z],
                    safety_margin=self.safety_margin,
                    method="gershgorin",   # solver-free
                )

            rec = ObservationRecord(
                time_s=float(time_s),
                zone_id=z,
                result=res,
                H_op_norm=float(np.linalg.norm(H_V, 2)) if H_V.size else 0.0,
                H_Q_op_norm=float(np.linalg.norm(H_Q, 2)) if H_Q.size else 0.0,
            )
            self.trajectories[z].records.append(rec)

    # ------------------------------------------------------------------ #
    # H splitting
    # ------------------------------------------------------------------ #

    def _split_H(
        self, zone_id: int, H: NDArray[np.float64],
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Split the zone's sensitivity matrix into voltage and Q_gen row blocks.

        The TSO controller in this code-base produces an output vector
        ordered as ``[V_bus | I_line]`` (line currents are hard-constrained
        and carry zero objective weight).  Q_gen rows are not yet present
        in the current controller; we expose ``H_Q`` as an empty (0, m)
        block so the downstream code handles it correctly.  Once the
        Q_gen feature is added, update this method to split out those rows.

        Row scaling
        -----------
        - ``W_V = 1`` (identity).  The MIQP objective in
          ``multi_tso_coordinator.py`` uses ``g_v * ||V - V_set||^2`` in raw
          pu units, so the voltage-tracking Hessian is ``g_v * H_V^T H_V``
          with no further normalisation.  (An earlier version of this
          module set ``W_V = 1/sigma_V`` on the assumption that ``g_v``
          encoded a probability weighting rather than a direct squared-
          deviation penalty; that introduced a spurious factor of
          ``1/sigma_V^2 ~= 10^4`` and made the computed ``g_w_min`` too
          large by four decades.  Do NOT reintroduce sigma_V here unless
          you also divide ``g_v`` by ``sigma_V^2`` in the objective.)
        - ``W_Q = 1 / (Q_max - Q_min)`` per generator (once Q_gen rows
          are populated).  This scaling IS appropriate because ``g_q``
          in the objective has no built-in normalisation.
        """
        zd = self.zone_defs[zone_id]
        n_v = len(zd.v_bus_indices)
        # n_i = len(zd.line_indices)   # currents, not penalised → ignored

        if H.shape[0] < n_v:
            # Defensive: some zones may have zero monitored buses.
            return (np.zeros((0, H.shape[1])),
                    np.zeros((0, H.shape[1])),
                    np.zeros(0), np.zeros(0))

        # Voltage-tracking rows.  W_V = 1 (see scaling note above).
        H_V = H[:n_v, :]
        W_V = np.ones(n_v, dtype=np.float64)

        # Q_gen rows: empty for now (not yet part of y in this codebase).
        # When the Q_gen feature lands, replace with H[n_v+n_i:n_v+n_i+n_gen, :]
        # and W_Q = 1 / (Q_max - Q_min) per generator.
        H_Q = np.zeros((0, H.shape[1]))
        W_Q = np.zeros(0)

        return H_V, H_Q, W_V, W_Q

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def summary(self, statistic: str = "max",
                percentile: float = 95.0) -> Dict[int, Dict[str, Any]]:
        """Summarise all trajectories with a single aggregation statistic."""
        out: Dict[int, Dict[str, Any]] = {}
        for z, traj in self.trajectories.items():
            if not traj.records:
                out[z] = {"n_samples": 0}
                continue
            gw_agg = traj.aggregate(statistic=statistic, percentile=percentile)
            block_means = []
            for k in range(4):
                sl = traj.layout.block_slice(k)
                if sl.stop > sl.start:
                    block_means.append(float(gw_agg[sl].mean()))
                else:
                    block_means.append(float("nan"))
            out[z] = {
                "n_samples": len(traj.records),
                "layout": traj.layout.sizes,
                "gw_per_block_{}".format(statistic): dict(
                    zip(traj.layout.names, block_means),
                ),
                "gw_full": gw_agg.tolist(),
                # Haeberle (constrained-projected-gradient) floor: scalar per zone.
                "gw_haberle_{}".format(statistic): traj.aggregate_haberle(
                    statistic=statistic, percentile=percentile,
                ),
            }
        return out

    def write_results(
        self,
        result_dir: str,
        *,
        basename: str = "stability_observer",
        plot: bool = True,
    ) -> None:
        """Write JSON summary and optional per-zone plots to ``result_dir``."""
        os.makedirs(result_dir, exist_ok=True)
        json_path = os.path.join(result_dir, f"{basename}.json")
        out_dict = {
            "g_v": self.g_v,
            "g_q": self.g_q,
            "ratio_priors": list(self.ratio_priors),
            "safety_margin": self.safety_margin,
            "method": self.method,
            "aggregations": {
                stat: self.summary(statistic=stat)
                for stat in ("max", "percentile", "mean")
            },
        }
        with open(json_path, "w") as fh:
            json.dump(out_dict, fh, indent=2)
        if plot:
            for z in self.tracked_zone_ids:
                plot_trajectory(
                    self.trajectories[z],
                    os.path.join(result_dir, f"{basename}_zone{z}.png"),
                )


# --------------------------------------------------------------------------- #
#  Reporting helpers
# --------------------------------------------------------------------------- #

def write_trajectory_report(
    path: str,
    trajectories: Dict[int, ZoneTrajectory],
) -> None:
    """Write a human-readable markdown summary of the observed trajectories."""
    lines = ["# Stability Observer Report\n"]
    for z in sorted(trajectories):
        traj = trajectories[z]
        T = len(traj.records)
        if T == 0:
            lines.append(f"## Zone {z}\n\n_No observations recorded._\n")
            continue
        gw_max = traj.aggregate(statistic="max")
        gw_p95 = traj.aggregate(statistic="percentile", percentile=95.0)
        gw_mean = traj.aggregate(statistic="mean")

        lines.append(f"## Zone {z}\n")
        lines.append(f"- Observations: **{T}** refresh snapshots")
        lines.append(f"- Layout (DER, PCC, V_gen, OLTC): "
                     f"{traj.layout.sizes}")
        lines.append("")
        lines.append("| Block | mean | p95 | max |")
        lines.append("|---|---:|---:|---:|")
        for k, name in enumerate(traj.layout.names):
            sl = traj.layout.block_slice(k)
            if sl.stop == sl.start:
                lines.append(f"| {name} | — | — | — |")
                continue
            lines.append(
                f"| {name} | {gw_mean[sl][0]:.2f} | "
                f"{gw_p95[sl][0]:.2f} | {gw_max[sl][0]:.2f} |"
            )
        lines.append("")

    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def plot_trajectory(traj: ZoneTrajectory, out_path: str) -> None:
    """Render per-block g_w^min time-series + histogram."""
    # Use Figure + FigureCanvasAgg directly so we never touch the global
    # pyplot backend. Calling matplotlib.use("Agg") here previously killed
    # the Qt5Agg live-plot backend for anything that ran after the first
    # observer report (e.g. Scenario B in main_comparison).
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        return

    if not traj.records:
        return

    t = traj.times_s() / 60.0  # minutes
    gw_block = traj.gw_block_trajectory()  # (T, 4)
    gw_max = traj.aggregate(statistic="max")
    gw_p95 = traj.aggregate(statistic="percentile", percentile=95.0)

    fig = Figure(figsize=(16, 7), dpi=110)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 4)

    # Top row: time series per block.
    for k, name in enumerate(traj.layout.names):
        ax = axes[0, k]
        sl = traj.layout.block_slice(k)
        if sl.stop == sl.start:
            ax.text(0.5, 0.5, f"{name}: no actuators",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray")
            ax.set_title(f"{name} — (empty in this zone)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        ax.plot(t, gw_block[:, k], lw=0.9, color="#4c72b0")
        ax.axhline(gw_p95[sl][0], color="#c44e52", ls="--", lw=1.0,
                   label=f"p95 = {gw_p95[sl][0]:.1f}")
        ax.axhline(gw_max[sl][0], color="#8172b2", ls="-", lw=1.0,
                   label=f"max = {gw_max[sl][0]:.1f}")
        ax.set_title(f"{name} — $g_{{w,{name}}}^{{\\min}}(t)$")
        ax.set_xlabel("time [min]")
        ax.set_ylabel(f"$g_{{w,{name}}}^{{\\min}}$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Bottom row: histograms per block.
    for k, name in enumerate(traj.layout.names):
        ax = axes[1, k]
        sl = traj.layout.block_slice(k)
        if sl.stop == sl.start:
            ax.text(0.5, 0.5, f"{name}: no actuators",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray")
            ax.set_title(f"{name} — (empty)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        data = gw_block[:, k]
        ax.hist(data, bins=40, color="#4c72b0", alpha=0.75,
                edgecolor="white", linewidth=0.4)
        ax.axvline(gw_p95[sl][0], color="#c44e52", ls="--", lw=1.2,
                   label=f"p95 = {gw_p95[sl][0]:.1f}")
        ax.axvline(gw_max[sl][0], color="#8172b2", ls="-", lw=1.2,
                   label=f"max = {gw_max[sl][0]:.1f}")
        ax.set_title(f"{name} — distribution")
        ax.set_xlabel(f"$g_{{w,{name}}}^{{\\min}}$")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Zone {traj.zone_id}: stability-minimum $g_w$ trajectory "
        f"(N = {len(traj.records)} observations)",
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
