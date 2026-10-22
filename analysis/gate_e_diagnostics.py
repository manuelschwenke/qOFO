"""Gate-E diagnostics: discrete actuator and per-bus voltage comparison.

Answers two questions the aggregate endpoint metrics cannot:

1.  **Do the two runs end on the same discrete actuator state?**  The static
    and RMS runs are *independent closed loops* -- each controller sees only
    its own plant -- so they may issue different ``u``.  OLTC taps and shunt
    steps are integrators: once they part, the difference persists, and a
    persistent zone-voltage offset follows.  Distinguishing "the plants
    differ" from "the control trajectories diverged" requires seeing the
    actuator states side by side.

2.  **Is the voltage gap uniform across a zone or carried by a few buses?**
    The zone *mean* hides this.  The RMS side records every TN bus at the
    monitor stride; the static records carry only the zone min/mean/max
    envelope, so the comparison is per-bus (RMS) against envelope (static).
    See ``KNOWN_GAPS`` below.

Usage::

    python analysis/gate_e_diagnostics.py results/rms_phase6_replay/0012_...

Figures land in ``<run>/figures/``.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Comparisons this script cannot make from the stored artefacts, and why.
#: Closing them needs extra monitors plus a re-run (see the module docstring
#: of ``pf/screening.py`` and the runner's record schema).
KNOWN_GAPS = (
    "static per-bus voltages: records store zone min/mean/max only",
    "RMS DSO feeder-bus voltages: monitored_outputs omits them by design",
)

STATIC_C = "#c0392b"
RMS_C = "#2471a3"


# ---------------------------------------------------------------- loading
def load_run(run_dir: Path) -> Dict[str, Any]:
    with (run_dir / "static_records.pkl").open("rb") as fh:
        static = pickle.load(fh)
    with (run_dir / "rms_records.pkl").open("rb") as fh:
        rms = pickle.load(fh)
    snaps = [p for p in (run_dir / "snapshot").glob("*.json")
             if "roundtrip" not in p.name and "sync" not in p.name]
    with snaps[0].open(encoding="utf-8") as fh:
        snapshot = json.load(fh)
    monitors = pd.read_csv(run_dir / "csv" / "rms_monitors_raw.csv")
    return {"static": static, "rms": rms, "snapshot": snapshot,
            "monitors": monitors}


def _times(records: Sequence) -> np.ndarray:
    return np.asarray([float(r.time_s) for r in records], dtype=float)


def _tap_series(records: Sequence, attr: str) -> Dict[str, np.ndarray]:
    """{label: series} for a per-record dict-of-scalars or dict-of-arrays."""
    out: Dict[str, List[float]] = {}
    for rec in records:
        table = getattr(rec, attr, None) or {}
        for key, value in table.items():
            if np.isscalar(value):
                out.setdefault(f"{key}", []).append(float(value))
            else:
                for i, v in enumerate(np.atleast_1d(value)):
                    out.setdefault(f"z{key}[{i}]", []).append(float(v))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


# ---------------------------------------------------------------- figures
def plot_actuators(data: Dict[str, Any], out: Path) -> Path:
    """Every discrete actuator, static vs RMS, on a shared time axis."""
    ts, tr = _times(data["static"]), _times(data["rms"])
    panels: List[Tuple[str, Dict[str, np.ndarray], Dict[str, np.ndarray]]] = []
    for attr, title in (
        ("zone_oltc_taps", "TSO zone OLTC taps"),
        ("dso_trafo_tap_pos", "DSO coupler 3W taps"),
        ("zone_tso_shunt_states", "TSO tertiary shunt steps"),
    ):
        s, r = _tap_series(data["static"], attr), _tap_series(data["rms"], attr)
        if s or r:
            panels.append((title, s, r))

    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 3.6 * len(panels)),
                             squeeze=False)
    for ax, (title, s, r) in zip(axes[:, 0], panels):
        keys = sorted(set(s) | set(r))
        cmap = plt.get_cmap("tab20")
        for i, k in enumerate(keys):
            col = cmap(i % 20)
            if k in s:
                ax.step(ts[:len(s[k])], s[k], where="post", color=col,
                        lw=1.6, alpha=0.9, label=f"{k} static")
            if k in r:
                ax.step(tr[:len(r[k])], r[k], where="post", color=col,
                        lw=1.6, ls="--", alpha=0.9, label=f"{k} RMS")
        ax.set_title(f"{title}  (solid = static, dashed = RMS, "
                     f"colour = actuator)")
        ax.set_ylabel("tap / step position")
        ax.grid(alpha=0.3)
        ax.margins(x=0.01)
        if len(keys) <= 12:
            ax.legend(ncol=4, fontsize=7, framealpha=0.9)
    axes[-1, 0].set_xlabel("Time [s]")
    fig.suptitle("Gate E: discrete actuator trajectories, static vs RMS",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = out / "gate_e_actuators.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_zone_voltages(data: Dict[str, Any], out: Path) -> Path:
    """Per-bus RMS voltages against the static envelope, one panel per zone.

    Static per-bus values are not recorded (see KNOWN_GAPS), so the static
    side is drawn as its min/mean/max envelope.  A per-bus RMS trace leaving
    that band is a genuine disagreement; one inside it is not resolvable
    from these artefacts.
    """
    zone_map = {int(k): set(v) for k, v in data["snapshot"]["zone_map"].items()}
    mon = data["monitors"]
    volt = mon[mon.signal.str.startswith("u_TN_bus")].copy()
    volt["bus"] = volt.signal.str.replace("u_TN_bus", "", regex=False).astype(int)
    ts = _times(data["static"])

    zones = sorted(zone_map)
    fig, axes = plt.subplots(len(zones), 1, figsize=(13, 3.6 * len(zones)),
                             squeeze=False, sharex=True)
    for ax, z in zip(axes[:, 0], zones):
        buses = sorted(b for b in volt.bus.unique() if b in zone_map[z])
        for b in buses:
            g = volt[volt.bus == b].sort_values("time_s")
            ax.plot(g.time_s, g.value, color=RMS_C, lw=0.7, alpha=0.55)
        if buses:                      # legend proxy
            ax.plot([], [], color=RMS_C, lw=1.2,
                    label=f"RMS, individual TN buses (n={len(buses)})")
        lo = np.asarray([r.zone_v_min[z] for r in data["static"]], dtype=float)
        hi = np.asarray([r.zone_v_max[z] for r in data["static"]], dtype=float)
        mu = np.asarray([r.zone_v_mean[z] for r in data["static"]], dtype=float)
        ax.fill_between(ts, lo, hi, step="post", color=STATIC_C, alpha=0.16,
                        label="static min-max envelope")
        ax.step(ts, mu, where="post", color=STATIC_C, lw=1.8,
                label="static mean")
        ax.set_title(f"TS zone {z}")
        ax.set_ylabel("V [pu]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[-1, 0].set_xlabel("Time [s]")
    fig.suptitle("Gate E: per-bus RMS voltages vs static envelope, by zone",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = out / "gate_e_zone_voltages_all_buses.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_dso_voltages(data: Dict[str, Any], out: Path) -> Path:
    """DSO-area voltage envelopes, static vs RMS.

    Both sides are envelopes here: the RMS run does not monitor DSO feeder
    buses (KNOWN_GAPS), so the RMS envelope comes from its own records --
    which are the RMS plant's measurement image, sampled at dispatch
    instants, not a continuous trajectory.
    """
    groups = sorted(getattr(data["static"][0], "dso_group_v_mean_pu", {}))
    ts, tr = _times(data["static"]), _times(data["rms"])
    fig, axes = plt.subplots(len(groups), 1, figsize=(13, 3.0 * len(groups)),
                             squeeze=False, sharex=True)
    for ax, g in zip(axes[:, 0], groups):
        for recs, t, col, tag in ((data["static"], ts, STATIC_C, "static"),
                                  (data["rms"], tr, RMS_C, "RMS")):
            lo = np.asarray([r.dso_group_v_min_pu[g] for r in recs], float)
            hi = np.asarray([r.dso_group_v_max_pu[g] for r in recs], float)
            mu = np.asarray([r.dso_group_v_mean_pu[g] for r in recs], float)
            ax.fill_between(t, lo, hi, step="post", color=col, alpha=0.15)
            ax.step(t, mu, where="post", color=col, lw=1.8,
                    label=f"{tag} mean (band = min-max)")
        ax.set_title(f"{g}")
        ax.set_ylabel("V [pu]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[-1, 0].set_xlabel("Time [s]")
    fig.suptitle("Gate E: DSO-area voltage envelopes, static vs RMS",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = out / "gate_e_dso_voltages.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def tap_divergence_table(data: Dict[str, Any]) -> pd.DataFrame:
    """Final discrete-actuator state, static vs RMS, worst mismatch first."""
    rows = []
    for attr in ("zone_oltc_taps", "dso_trafo_tap_pos",
                 "zone_tso_shunt_states"):
        s = _tap_series(data["static"], attr)
        r = _tap_series(data["rms"], attr)
        for k in sorted(set(s) | set(r)):
            sv = s[k][-1] if k in s and len(s[k]) else np.nan
            rv = r[k][-1] if k in r and len(r[k]) else np.nan
            rows.append({"actuator_class": attr, "actuator": k,
                         "static_final": sv, "rms_final": rv,
                         "delta": rv - sv})
    df = pd.DataFrame(rows)
    return df.reindex(df.delta.abs().sort_values(ascending=False).index)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args(argv)

    data = load_run(args.run_dir)
    figs = args.run_dir / "figures"
    figs.mkdir(exist_ok=True)

    paths = [plot_actuators(data, figs),
             plot_zone_voltages(data, figs),
             plot_dso_voltages(data, figs)]

    table = tap_divergence_table(data)
    csv = args.run_dir / "csv" / "actuator_divergence.csv"
    table.to_csv(csv, index=False)

    print("figures:")
    for p in paths:
        print("  ", p)
    print("  ", csv)
    print("\nDiscrete actuator divergence (worst first):")
    print(table.head(15).to_string(index=False))
    n_diff = int((table.delta.abs() > 0).sum())
    print(f"\n{n_diff}/{len(table)} discrete actuators end in a different "
          f"state.")
    print("\nNot resolvable from these artefacts:")
    for gap in KNOWN_GAPS:
        print("  -", gap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
