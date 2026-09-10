"""Matplotlib quick look at a saved Ch 10 ladder; writes only RUN/figures.

From the project root::

    python -m experiments.ch_10_case_study.ch_10_2_preview --run-dir results/ch10_ladder/0002

Without --run-dir, use the newest numbered ch10_ladder run. PNG and PDF are
saved by default; --formats png saves PNG only. No simulation or TeX export.

Voltage curves use logged post-power-flow spatial RMS errors over the TS
area's observed buses, including the references used during that run. The
aggregate weights the three areas equally. Missing area errors remain NaN;
they are never replaced by the error of the area's mean voltage. CSV means
and RMS values are sample-weighted, as in the ladder's RMS report.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import pickle
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Support both python -m and execution of this file from an IDE.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.ch_10_case_study.ch_10_2_figures import LADDER, PALETTE, STYLE
from experiments.helpers.comparison_metrics import q_iface_per_trafo
from experiments.paths import RESULTS_ROOT

ZONES = (1, 2, 3)


def latest_run() -> Path:
    base = Path(RESULTS_ROOT) / "ch10_ladder"
    runs = [p for p in base.iterdir() if p.is_dir() and p.name.isdigit()] if base.is_dir() else []
    if not runs:
        raise ValueError(f"No numbered ladder runs in {base}; specify --run-dir")
    return max(runs, key=lambda p: int(p.name))


def load_logs(run_dir: Path, names: list[str]) -> dict:
    logs = {}
    for name in names:
        path = run_dir / name / "log.pkl"
        if not path.is_file():
            warnings.warn(f"{name}: missing log; skipped", stacklevel=2)
            continue
        with path.open("rb") as stream:
            records = pickle.load(stream)
        if not records:
            warnings.warn(f"{name}: empty/failed run; skipped", stacklevel=2)
            continue
        t = np.asarray([r.time_s / 60 for r in records], float)
        if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError(f"{name}: timestamps must be finite and strictly increasing")
        logs[name] = records
        print(f"{name}: {len(records)} records, {t[0]:g} to {t[-1]:g} min", flush=True)
    if not logs:
        raise ValueError(f"No nonempty variant logs in {run_dir}")
    times = [np.asarray([r.time_s for r in records]) for records in logs.values()]
    if any(not np.array_equal(times[0], t) for t in times[1:]):
        warnings.warn("Variant time grids differ; curves and summaries use each log's own samples", stacklevel=2)
    return logs


def voltage_data(records) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, (sample, area) logged errors, and equal-area RMS.

    Require all three areas for the aggregate to keep its meaning constant.
    """
    t = np.asarray([r.time_s / 60 for r in records], float)
    errors = np.asarray([
        [(getattr(r, "zone_v_rms_err_pu", None) or {}).get(z, np.nan) for z in ZONES]
        for r in records
    ], float)
    errors[~np.isfinite(errors)] = np.nan
    if np.any(np.isnan(errors)):
        warnings.warn("Missing recorded area voltage errors: gaps retained in curves", stacklevel=2)
    return t, errors, np.sqrt(np.mean(errors ** 2, axis=1))


def decorate(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save_figure(fig, out: Path, stem: str, formats: list[str], dpi: int) -> None:
    for extension in formats:
        path = out / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  wrote {path.name}", flush=True)
    plt.close(fig)


def plot_voltage(data: dict, out: Path, formats: list[str], dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    for name, (t, _, aggregate) in data.items():
        ax.plot(t, aggregate, label=name, color=PALETTE[name], **STYLE[name])
    decorate(ax, r"$e_{\mathrm{v}}$ / p.u.")
    ax.set(title="TS voltage tracking - equal-area RMS (post-PF)", xlabel="Time / min", autoscaley_on=True)
    ax.set_ylim(bottom=0)
    ax.legend(ncol=len(data), loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    save_figure(fig, out, "ch10_voltage_tracking", formats, dpi)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, sharey=True, layout="constrained")
    for column, (z, ax) in enumerate(zip(ZONES, axes)):
        for name, (t, errors, _) in data.items():
            ax.plot(t, errors[:, column], label=name, color=PALETTE[name], **STYLE[name])
        decorate(ax, rf"$e_{{\mathrm{{v}},{z}}}$ / p.u.")
        ax.set(title=f"TS area {z}", autoscaley_on=True)
    # Set the shared lower bound only after all areas have contributed to autoscaling.
    axes[0].set_ylim(bottom=0)
    axes[0].legend(ncol=len(data), loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False)
    axes[-1].set_xlabel("Time / min")
    fig.suptitle("Voltage tracking per TS area - logged spatial RMS (post-PF)")
    save_figure(fig, out, "ch10_voltage_tracking_per_area", formats, dpi)


def plot_interfaces(logs: dict, name: str, out: Path, formats: list[str], dpi: int) -> None:
    if name not in logs:
        warnings.warn(f"{name} unavailable; interface figure skipped", stacklevel=2)
        return
    data = q_iface_per_trafo(logs[name])
    if not data["groups"]:
        warnings.warn(f"{name}: no interface data; figure skipped", stacklevel=2)
        return
    fig, axes = plt.subplots(len(data["groups"]), 1, figsize=(11, 2.8 * len(data["groups"])),
                             sharex=True, squeeze=False, layout="constrained")
    for ax, group in zip(axes[:, 0], data["groups"]):
        for i, key in enumerate(data["trafos"][group]):
            color = f"C{i % 10}"
            ax.plot(data["t_min"], data["actual_mvar"][key], color=color, label=f"{key}: actual")
            if data["has_setpoint"][key]:
                ax.step(data["t_min"], data["set_mvar"][key], where="post", color=color,
                        linestyle="--", linewidth=1.1, label=f"{key}: setpoint")
        decorate(ax, r"$Q_{\mathrm{interface}}$ / Mvar")
        ax.set_title(str(group))
        ax.legend(fontsize=8, ncol=2, frameon=False)
    axes[-1, 0].set_xlabel("Time / min")
    fig.suptitle(f"{name}: TS-DS interface reactive power (post-PF actual, dispatched reference)")
    save_figure(fig, out, f"ch10_iface_tracking_{name}", formats, dpi)


def plot_envelopes(logs: dict, out: Path, formats: list[str], dpi: int) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, sharey=True, layout="constrained")
    for z, ax in zip(ZONES, axes):
        for name, records in logs.items():
            t = [r.time_s / 60 for r in records]
            low = [(getattr(r, "zone_v_min", None) or {}).get(z, np.nan) for r in records]
            high = [(getattr(r, "zone_v_max", None) or {}).get(z, np.nan) for r in records]
            ax.plot(t, low, color=PALETTE[name], label=name, **STYLE[name])
            ax.plot(t, high, color=PALETTE[name], **STYLE[name])
        decorate(ax, "Voltage / p.u.")
        ax.set_title(f"TS area {z}")
    axes[0].legend(ncol=len(logs), loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False)
    axes[-1].set_xlabel("Time / min")
    fig.suptitle("TS voltage envelopes - minimum and maximum observed bus voltage (post-PF)")
    save_figure(fig, out, "ch10_voltage_envelopes_per_area", formats, dpi)


def write_metrics(data: dict, out: Path) -> None:
    rows = []
    for name, (t, errors, aggregate) in data.items():
        for area, values in [(str(z), errors[:, i]) for i, z in enumerate(ZONES)] + [("all_equal_area", aggregate)]:
            finite = values[np.isfinite(values)]
            rows.append(dict(variant=name, area=area, start_min=t[0], end_min=t[-1],
                             samples=len(t), valid_samples=len(finite),
                             mean_e_v_pu=float(finite.mean()) if finite.size else np.nan,
                             rms_e_v_pu=float(np.sqrt(np.mean(finite ** 2))) if finite.size else np.nan,
                             peak_e_v_pu=float(finite.max()) if finite.size else np.nan))
    with (out / "ch10_voltage_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nEqual-area voltage error: sample mean / sample RMS / peak [p.u.]")
    for row in rows:
        if row["area"] == "all_equal_area":
            print(f"  {row['variant']:>2}: {row['mean_e_v_pu']:.6f} / {row['rms_e_v_pu']:.6f} / {row['peak_e_v_pu']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, help="saved ladder directory; default: newest numbered run")
    parser.add_argument("--only", help="comma-separated variants; default: all ladder variants")
    parser.add_argument("--iface-variant", choices=LADDER, default="O3")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    names = [n.strip() for n in args.only.split(",")] if args.only else list(LADDER)
    if args.dpi <= 0 or any(n not in LADDER for n in names):
        parser.error("--dpi must be positive and --only must contain ladder variant names")
    names = [n for n in LADDER if n in names]
    try:
        run_dir = (args.run_dir or latest_run()).resolve()
        logs = load_logs(run_dir, names)
    except (ValueError, OSError, EOFError, pickle.UnpicklingError) as exc:
        parser.exit(1, f"Cannot preview run: {exc}\n")
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    data = {name: voltage_data(records) for name, records in logs.items()}
    with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans", "font.size": 10}):
        plot_voltage(data, out, args.formats, args.dpi)
        plot_interfaces(logs, args.iface_variant, out, args.formats, args.dpi)
        plot_envelopes(logs, out, args.formats, args.dpi)
    write_metrics(data, out)
    (out / "preview_notes.txt").write_text(
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}\n"
        f"Source: {run_dir}\nVariants: {', '.join(logs)}\n"
        "Voltage: recorded post-PF spatial RMS over observed TS buses and the run's references.\n"
        "Aggregate: sqrt((e_v,1^2 + e_v,2^2 + e_v,3^2)/3), equal area weights.\n"
        "Missing area errors remain NaN; aggregate requires all three areas.\n"
        "CSV mean and RMS use finite samples (no duration weighting); valid counts are included.\n"
        "Curves show plant outcomes, not the noisy pre-control measurements seen by controllers.\n"
        "Interface actual Q is post-PF, using the logged transformer-side sign convention.\n"
        "No saved configuration/event schedule is assumed: no inferred event or voltage-limit lines.\n"
        "Single-run descriptive comparison; no statistical confidence statement.\n",
        encoding="utf-8",
    )
    print(f"\nPreview saved in {out}")


if __name__ == "__main__":
    main()
