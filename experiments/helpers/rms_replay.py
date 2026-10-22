"""Pure extraction, comparison and plotting for Phase 6 RMS replay.

No PowerFactory API object crosses into this module.  Inputs are runner
records, a parsed dynamic snapshot, and numeric (time, value) arrays.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


Trajectory = Tuple[np.ndarray, np.ndarray]
Trajectories = Dict[str, Trajectory]

_Q_RMS_RE = re.compile(r"^qSTS_.+_t(?P<index>\d+)$")
_Q_STATIC_RE = re.compile(r"(?:^|\|)trafo_(?P<index>\d+)$")
_Q_LOGICAL_RE = re.compile(r"^qSTS_t(?P<index>\d+)$")
_V_LOGICAL_RE = re.compile(r"^vZone_(?P<zone>\d+)$")


def _logical_q_name(index: int) -> str:
    return f"qSTS_t{int(index)}"


def _signal_sort_key(name: str):
    q_match = _Q_LOGICAL_RE.match(name)
    if q_match:
        return (0, int(q_match.group("index")))
    v_match = _V_LOGICAL_RE.match(name)
    if v_match:
        return (1, int(v_match.group("zone")))
    return (2, name)


def signal_type(name: str) -> str:
    if _Q_LOGICAL_RE.match(name):
        return "interface_q"
    if _V_LOGICAL_RE.match(name):
        return "zone_voltage"
    return "other"


def static_controlled_trajectories(
    records: Sequence,
) -> Trajectories:
    """Extract static endpoint trajectories from iteration records.

    The runner stamps ``record.time_s = step * dt_s`` and records plant truth
    after its end-of-step solve.  The record timestamp is therefore already
    the static-equilibrium interval endpoint.
    """

    values: Dict[str, list[Tuple[float, float]]] = {}
    for record in records:
        endpoint = float(record.time_s)
        seen_q = set()
        for key, value in record.dso_trafo_q_actual_mvar.items():
            match = _Q_STATIC_RE.search(str(key))
            if match is None:
                raise ValueError(f"unrecognised interface key {key!r}")
            index = int(match.group("index"))
            if index in seen_q:
                raise ValueError(
                    f"record at t={record.time_s} contains trafo {index} twice"
                )
            seen_q.add(index)
            values.setdefault(_logical_q_name(index), []).append(
                (endpoint, float(value))
            )
        for zone, value in record.zone_v_mean.items():
            values.setdefault(f"vZone_{int(zone)}", []).append(
                (endpoint, float(value))
            )

    trajectories: Trajectories = {}
    for name, rows in values.items():
        rows.sort(key=lambda row: row[0])
        trajectories[name] = (
            np.asarray([row[0] for row in rows], dtype=float),
            np.asarray([row[1] for row in rows], dtype=float),
        )
    return dict(
        sorted(trajectories.items(), key=lambda item: _signal_sort_key(item[0]))
    )


def rms_controlled_trajectories(
    raw: Mapping[str, Trajectory],
    snapshot_doc: Mapping,
) -> Trajectories:
    """Normalize raw PF monitor labels and derive TN-PQ zone-mean voltage."""

    out: Trajectories = {}
    q_indices = set()
    for label, (time, value) in raw.items():
        match = _Q_RMS_RE.match(label)
        if match is None:
            continue
        index = int(match.group("index"))
        if index in q_indices:
            raise ValueError(f"duplicate RMS interface monitor for trafo {index}")
        q_indices.add(index)
        out[_logical_q_name(index)] = (
            np.asarray(time, dtype=float),
            np.asarray(value, dtype=float),
        )

    expected_q = {int(key) for key in snapshot_doc["model"]["trafo3w"]}
    if q_indices != expected_q:
        raise ValueError(
            "RMS interface monitor set differs from snapshot: "
            f"missing={sorted(expected_q - q_indices)}, "
            f"extra={sorted(q_indices - expected_q)}"
        )

    model_buses = snapshot_doc["model"]["bus"]
    gen_buses = {
        int(record["bus"])
        for record in snapshot_doc["model"]["gen"].values()
        if bool(record.get("in_service", True))
    }
    for zone_text, zone_buses in snapshot_doc["zone_map"].items():
        zone = int(zone_text)
        pq_tn_buses = []
        for bus in zone_buses:
            bus = int(bus)
            record = model_buses.get(str(bus))
            if record is None:
                continue
            if (
                record.get("subnet") == "TN"
                and float(record["vn_kv"]) >= 100.0
                and bus not in gen_buses
            ):
                pq_tn_buses.append(bus)
        if not pq_tn_buses:
            raise ValueError(f"zone {zone} has no monitored TN PQ buses")

        labels = [f"u_TN_bus{bus}" for bus in sorted(pq_tn_buses)]
        missing = [label for label in labels if label not in raw]
        if missing:
            raise ValueError(
                f"zone {zone} is missing RMS voltage monitors {missing}"
            )
        common_t = np.asarray(raw[labels[0]][0], dtype=float)
        series = []
        for label in labels:
            time = np.asarray(raw[label][0], dtype=float)
            if time.shape != common_t.shape or not np.allclose(
                time, common_t, rtol=0.0, atol=1e-9
            ):
                raise ValueError(
                    f"voltage monitor {label} has a different RMS time grid"
                )
            series.append(np.asarray(raw[label][1], dtype=float))
        out[f"vZone_{zone}"] = (
            common_t,
            np.mean(np.vstack(series), axis=0),
        )

    return dict(sorted(out.items(), key=lambda item: _signal_sort_key(item[0])))


def trajectories_long_frame(
    trajectories: Mapping[str, Trajectory],
) -> pd.DataFrame:
    rows = []
    for name, (time, value) in trajectories.items():
        for t_value, y_value in zip(time, value):
            rows.append(
                {
                    "signal": name,
                    "signal_type": signal_type(name),
                    "time_s": float(t_value),
                    "value": float(y_value),
                }
            )
    return pd.DataFrame.from_records(rows)


def endpoint_comparison(
    static: Mapping[str, Trajectory],
    rms: Mapping[str, Trajectory],
) -> pd.DataFrame:
    """Compare the two plants at each static interval endpoint."""

    rows = []
    for name in sorted(set(static).intersection(rms), key=_signal_sort_key):
        static_t, static_y = static[name]
        rms_t, rms_y = rms[name]
        if len(rms_t) == 0:
            continue
        for t_value, static_value in zip(static_t, static_y):
            if t_value < rms_t[0] or t_value > rms_t[-1]:
                continue
            rms_value = float(np.interp(t_value, rms_t, rms_y))
            error = rms_value - float(static_value)
            rows.append(
                {
                    "signal": name,
                    "signal_type": signal_type(name),
                    "time_s": float(t_value),
                    "static_value": float(static_value),
                    "rms_value": rms_value,
                    "error": error,
                    "abs_error": abs(error),
                }
            )
    return pd.DataFrame.from_records(rows)


def endpoint_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame(
            columns=["signal", "signal_type", "n", "rmse", "mae", "max_abs"]
        )
    rows = []
    for (name, kind), frame in comparison.groupby(
        ["signal", "signal_type"], sort=False
    ):
        error = frame["error"].to_numpy(dtype=float)
        rows.append(
            {
                "signal": name,
                "signal_type": kind,
                "n": len(error),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "mae": float(np.mean(np.abs(error))),
                "max_abs": float(np.max(np.abs(error))),
            }
        )
    return pd.DataFrame.from_records(rows)


def interval_settling_table(
    trajectories: Mapping[str, Trajectory],
    *,
    interval_s: float | None = None,
    total_s: float,
    relative_band: float = 0.02,
    q_abs_floor_mvar: float = 1.0,
    voltage_abs_floor_pu: float = 1e-3,
    final_mean_window_s: float = 0.5,
    windows: Sequence[Tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Compute a bounded 2-percent settling metric per analysis window.

    Parameters
    ----------
    interval_s
        Width of a uniform window grid covering ``[0, total_s)``.  Required
        unless ``windows`` is given.
    windows
        Explicit ``[(start_s, end_s), ...]`` windows, overriding ``interval_s``.
        Use this to anchor the analysis on *events* rather than on a fixed
        cadence: a settling metric is only meaningful relative to a disturbance,
        and a uniform grid mostly measures quiet intervals in which nothing
        stepped.  Offline controller tuning wants one window per contingency.

    Notes
    -----
    ``settled_within_interval=False`` means the signal was still outside the
    tolerance band at the window edge, and ``settling_time_s`` is then censored
    at the window width — treat it as a lower bound, not a measurement.
    """
    if windows is None and interval_s is None:
        raise ValueError("pass either interval_s or windows")

    rows = []
    if windows is not None:
        bounds = [(float(a), float(b)) for a, b in windows]
    else:
        starts = np.arange(0.0, float(total_s), float(interval_s))
        bounds = [
            (float(s), min(float(s) + float(interval_s), float(total_s)))
            for s in starts
        ]
    for name, (time, value) in trajectories.items():
        time = np.asarray(time, dtype=float)
        value = np.asarray(value, dtype=float)
        if time.ndim != 1 or value.ndim != 1 or time.shape != value.shape:
            raise ValueError(f"{name}: trajectory arrays must be aligned 1-D")
        if len(time) == 0 or np.any(np.diff(time) < 0.0):
            raise ValueError(f"{name}: trajectory time grid is empty or unsorted")
        kind = signal_type(name)
        abs_floor = (
            q_abs_floor_mvar
            if kind == "interface_q"
            else voltage_abs_floor_pu
        )

        for start, end in bounds:
            window_mask = (time >= start - 1e-9) & (time <= end + 1e-9)
            if not np.any(window_mask):
                raise ValueError(f"{name}: no samples in interval [{start}, {end}]")
            wt = time[window_mask]
            wy = value[window_mask]
            before = int(np.searchsorted(time, start, side="right") - 1)
            before = max(before, 0)
            y_initial = float(value[before])
            final_mask = (time >= end - final_mean_window_s) & (
                time <= end + 1e-9
            )
            if not np.any(final_mask):
                y_final = float(wy[-1])
            else:
                y_final = float(np.mean(value[final_mask]))
            signed_step = y_final - y_initial
            step = abs(signed_step)
            tolerance = max(relative_band * step, abs_floor)
            outside = np.abs(wy - y_final) > tolerance

            if np.any(outside):
                last_outside = int(np.flatnonzero(outside)[-1])
                if last_outside == len(wt) - 1:
                    settled = False
                    settling_s = end - start
                else:
                    settled = True
                    settling_s = max(0.0, float(wt[last_outside + 1] - start))
            else:
                settled = True
                settling_s = 0.0

            if step > 1e-12:
                direction = np.sign(signed_step)
                overshoot = max(
                    0.0,
                    float(np.max((wy - y_final) * direction) / step),
                )
            else:
                overshoot = 0.0
            rows.append(
                {
                    "signal": name,
                    "signal_type": kind,
                    "interval_start_s": float(start),
                    "interval_end_s": float(end),
                    "y_initial": y_initial,
                    "y_final": y_final,
                    "step_magnitude": step,
                    "tolerance": tolerance,
                    "settling_time_s": float(settling_s),
                    "settled_within_interval": bool(settled),
                    "overshoot_fraction": overshoot,
                }
            )
    return pd.DataFrame.from_records(rows)


def settling_summary(
    settling: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-signal and per-quantity settling summaries."""

    columns = [
        "n_intervals",
        "unsettled_intervals",
        "max_settling_s",
        "p95_settling_s",
        "max_overshoot_fraction",
    ]
    if settling.empty:
        return (
            pd.DataFrame(columns=["signal", "signal_type", *columns]),
            pd.DataFrame(columns=["signal_type", *columns]),
        )

    def _rows(group_fields: Iterable[str]) -> pd.DataFrame:
        rows = []
        grouper = list(group_fields)
        for keys, frame in settling.groupby(grouper, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(grouper, keys))
            row.update(
                {
                    "n_intervals": int(len(frame)),
                    "unsettled_intervals": int(
                        (~frame["settled_within_interval"]).sum()
                    ),
                    "max_settling_s": float(frame["settling_time_s"].max()),
                    "p95_settling_s": float(
                        frame["settling_time_s"].quantile(0.95)
                    ),
                    "max_overshoot_fraction": float(
                        frame["overshoot_fraction"].max()
                    ),
                }
            )
            rows.append(row)
        return pd.DataFrame.from_records(rows)

    return _rows(("signal", "signal_type")), _rows(("signal_type",))


def plot_controlled_output_overlays(
    static: Mapping[str, Trajectory],
    rms: Mapping[str, Trajectory],
    out_dir: Path,
    *,
    dt_s: float,
    tso_period_s: float,
    write_pdf: bool = True,
) -> list[Path]:
    """Render complete interface-Q and zone-voltage static/RMS overlays."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _decorate(ax, total_s: float) -> None:
        for t_value in np.arange(tso_period_s, total_s + 1e-9, tso_period_s):
            ax.axvline(t_value, color="0.85", lw=0.7, zorder=0)
        ax.grid(True, alpha=0.25)

    q_names = [
        name for name in rms if signal_type(name) == "interface_q"
    ]
    q_names.sort(key=_signal_sort_key)
    if q_names:
        ncols = 3
        nrows = int(np.ceil(len(q_names) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(13.5, 2.7 * nrows),
            sharex=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).ravel()
        for ax, name in zip(axes, q_names):
            rt, ry = rms[name]
            ax.plot(rt, ry, color="#1f77b4", lw=1.0, label="RMS plant")
            if name in static:
                st, sy = static[name]
                ax.step(
                    np.maximum(st - dt_s, 0.0),
                    sy,
                    where="post",
                    color="#d62728",
                    lw=1.1,
                    label="static equilibrium",
                )
            ax.set_title(name.replace("qSTS_", "Interface "))
            ax.set_ylabel("Q HV side [Mvar]")
            _decorate(ax, float(rt[-1]))
        for ax in axes[len(q_names):]:
            ax.set_visible(False)
        for ax in axes[-ncols:]:
            if ax.get_visible():
                ax.set_xlabel("Time [s]")
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle("Phase 6 Gate E: all TS-STS interface flows")
        png = out_dir / "interface_q_static_vs_rms.png"
        fig.savefig(png, dpi=180)
        written.append(png)
        if write_pdf:
            pdf = out_dir / "interface_q_static_vs_rms.pdf"
            fig.savefig(pdf)
            written.append(pdf)
        plt.close(fig)

    v_names = [
        name for name in rms if signal_type(name) == "zone_voltage"
    ]
    v_names.sort(key=_signal_sort_key)
    if v_names:
        fig, axes = plt.subplots(
            len(v_names),
            1,
            figsize=(11.0, 2.8 * len(v_names)),
            sharex=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, v_names):
            rt, ry = rms[name]
            ax.plot(rt, ry, color="#1f77b4", lw=1.0, label="RMS plant")
            if name in static:
                st, sy = static[name]
                ax.step(
                    np.maximum(st - dt_s, 0.0),
                    sy,
                    where="post",
                    color="#d62728",
                    lw=1.1,
                    label="static equilibrium",
                )
            ax.set_ylabel("mean V [pu]")
            ax.set_title(name.replace("vZone_", "TS zone "))
            _decorate(ax, float(rt[-1]))
        axes[-1].set_xlabel("Time [s]")
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle("Phase 6 Gate E: controlled TN-PQ zone voltages")
        png = out_dir / "zone_voltage_static_vs_rms.png"
        fig.savefig(png, dpi=180)
        written.append(png)
        if write_pdf:
            pdf = out_dir / "zone_voltage_static_vs_rms.pdf"
            fig.savefig(pdf)
            written.append(pdf)
        plt.close(fig)

    return written
