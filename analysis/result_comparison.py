"""Interactive comparison helpers for saved multi-system OFO runs.

The public entry point :func:`create_dashboard` is used by
``analysis/compare_results.ipynb``.  Data extraction and plotting are kept in
this module so they can be tested without executing notebook widget state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
import json
from numbers import Number
from pathlib import Path
import pickle
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunInfo:
    """One immutable result directory containing ``records.pkl``."""

    identifier: str
    directory: Path
    records_path: Path
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuantitySpec:
    """Fields and presentation metadata belonging to one plot quantity."""

    label: str
    fields: tuple[str, ...]
    ylabel: str
    aliases: Mapping[str, str] = field(default_factory=dict)


QUANTITIES: Mapping[str, QuantitySpec] = {
    "tso_voltage": QuantitySpec(
        "Transmission-system voltages (min / mean / max)",
        ("zone_v_min", "zone_v_mean", "zone_v_max"),
        "Voltage [p.u.]",
        {
            "zone_v_min": "minimum",
            "zone_v_mean": "mean",
            "zone_v_max": "maximum",
        },
    ),
    "dso_voltage": QuantitySpec(
        "Distribution-system voltages (min / mean / max)",
        (
            "dso_group_v_min_pu",
            "dso_group_v_mean_pu",
            "dso_group_v_max_pu",
        ),
        "Voltage [p.u.]",
        {
            "dso_group_v_min_pu": "minimum",
            "dso_group_v_mean_pu": "mean",
            "dso_group_v_max_pu": "maximum",
        },
    ),
    "tso_der_q": QuantitySpec(
        "TSO DER reactive-power infeed",
        ("zone_q_der",),
        "Reactive power [Mvar]",
    ),
    "dso_der_q": QuantitySpec(
        "DSO DER reactive-power infeed (group aggregate)",
        ("dso_group_q_der_mvar",),
        "Reactive power [Mvar]",
    ),
    "generator_q": QuantitySpec(
        "Synchronous-generator reactive-power infeed",
        ("zone_q_gen",),
        "Reactive power [Mvar]",
    ),
    "tso_shunts": QuantitySpec(
        "TSO shunt positions (MSC / MSR)",
        ("zone_tso_shunt_states",),
        "Shunt state [step]",
    ),
    "tso_oltc": QuantitySpec(
        "TSO OLTC positions",
        ("zone_oltc_taps",),
        "Tap position [step]",
    ),
    "dso_oltc": QuantitySpec(
        "DSO OLTC positions",
        ("dso_trafo_tap_pos",),
        "Tap position [step]",
    ),
    "avr_setpoints": QuantitySpec(
        "Generator AVR voltage setpoints",
        ("zone_v_gen",),
        "Voltage setpoint [p.u.]",
    ),
    "dso_interface_q": QuantitySpec(
        "DSO interface Q (setpoint / actual)",
        ("dso_q_set_mvar", "dso_q_actual_mvar"),
        "Reactive power [Mvar]",
        {
            "dso_q_set_mvar": "setpoint",
            "dso_q_actual_mvar": "actual",
        },
    ),
    "dso_trafo_q": QuantitySpec(
        "DSO transformer Q (setpoint / actual)",
        ("dso_trafo_q_set_mvar", "dso_trafo_q_actual_mvar"),
        "Reactive power [Mvar]",
        {
            "dso_trafo_q_set_mvar": "setpoint",
            "dso_trafo_q_actual_mvar": "actual",
        },
    ),
    "tie_q": QuantitySpec(
        "Inter-zone tie-line reactive-power flow",
        ("zone_tie_q_mvar",),
        "Reactive power [Mvar]",
    ),
    "tso_loading": QuantitySpec(
        "Transmission line loading (min / mean / max)",
        (
            "zone_line_loading_min_pct",
            "zone_line_loading_mean_pct",
            "zone_line_loading_max_pct",
        ),
        "Loading [%]",
        {
            "zone_line_loading_min_pct": "minimum",
            "zone_line_loading_mean_pct": "mean",
            "zone_line_loading_max_pct": "maximum",
        },
    ),
    "dso_loading": QuantitySpec(
        "Distribution line loading (min / mean / max)",
        (
            "dso_group_i_min_pct",
            "dso_group_i_mean_pct",
            "dso_group_i_max_pct",
        ),
        "Loading [%]",
        {
            "dso_group_i_min_pct": "minimum",
            "dso_group_i_mean_pct": "mean",
            "dso_group_i_max_pct": "maximum",
        },
    ),
    "losses": QuantitySpec(
        "System active-power losses",
        ("total_losses_mw",),
        "Active-power loss [MW]",
    ),
}


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find the nearest parent containing both ``analysis`` and ``results``."""
    here = Path(start or Path.cwd()).resolve()
    candidates = (here, *here.parents)
    for candidate in candidates:
        if (candidate / "analysis").is_dir() and (candidate / "results").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the project root containing analysis/ and results/."
    )


def discover_runs(results_root: Path) -> list[RunInfo]:
    """Return every run below ``results_root`` that contains ``records.pkl``."""
    root = Path(results_root).resolve()
    runs: list[RunInfo] = []
    if not root.exists():
        return runs
    for records_path in root.rglob("records.pkl"):
        directory = records_path.parent
        identifier = directory.relative_to(root).as_posix()
        meta_path = directory / "meta.json"
        meta: Mapping[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
        runs.append(RunInfo(identifier, directory, records_path, meta))
    return sorted(runs, key=lambda run: run.identifier)


@lru_cache(maxsize=16)
def _load_pickle(path: str, modified_ns: int) -> Any:
    del modified_ns  # part of the cache key; the value itself is not needed
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_records(run: RunInfo) -> list[Any]:
    """Load a local trusted result pickle and normalize its record container."""
    payload = _load_pickle(
        str(run.records_path), run.records_path.stat().st_mtime_ns
    )
    if hasattr(payload, "log"):
        payload = payload.log
    elif isinstance(payload, Mapping) and "records" in payload:
        payload = payload["records"]
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise TypeError(
            f"{run.records_path} does not contain a record sequence"
        )
    return list(payload)


def _record_value(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _record_time_s(record: Any, fallback: int) -> float:
    value = _record_value(record, "time_s")
    if value is not None:
        return float(value)
    minute = _record_value(record, "minute")
    if minute is not None:
        return 60.0 * float(minute)
    step = _record_value(record, "step")
    return float(step if step is not None else fallback)


def _flatten_numeric(value: Any, path: tuple[Any, ...] = ()):
    """Yield ``(path, scalar)`` leaves from nested dict/array structures."""
    if value is None or isinstance(value, (str, bytes, bool, np.bool_)):
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _flatten_numeric(value[key], path + (key,))
        return
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            yield from _flatten_numeric(value.item(), path)
            return
        for index in np.ndindex(value.shape):
            item_path = path + tuple(int(item) for item in index)
            yield from _flatten_numeric(value[index], item_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            yield from _flatten_numeric(item, path + (index,))
        return
    if isinstance(value, Number):
        yield path, float(value)


def _base_entity(field_name: str, value: Any) -> str:
    if isinstance(value, tuple):
        joined = "–".join(str(item) for item in value)
        return f"Zone {joined}" if field_name.startswith("zone_") else joined
    if field_name.startswith("zone_"):
        return f"Zone {value}"
    if field_name.startswith("dso_"):
        return str(value)
    if field_name.startswith("tie_"):
        return f"Tie line {value}"
    return str(value)


def _channel_label(
    field_name: str,
    path: tuple[Any, ...],
    alias: Optional[str],
) -> str:
    if not path:
        parts = ["system"]
    else:
        parts = [_base_entity(field_name, path[0])]
        for level, item in enumerate(path[1:], start=1):
            if isinstance(item, tuple):
                parts.append("–".join(str(value) for value in item))
            elif isinstance(item, (int, np.integer)):
                parts.append(f"item {int(item)}" if level == 1 else str(item))
            else:
                parts.append(str(item))
    if alias:
        parts.append(alias)
    return " · ".join(parts)


def extract_field(
    records: Sequence[Any],
    field_name: str,
    *,
    alias: Optional[str] = None,
) -> pd.DataFrame:
    """Convert one nested record field to a time-indexed numeric DataFrame."""
    rows: list[dict[str, float]] = []
    times: list[float] = []
    for position, record in enumerate(records):
        row = {
            _channel_label(field_name, path, alias): scalar
            for path, scalar in _flatten_numeric(
                _record_value(record, field_name)
            )
        }
        rows.append(row)
        times.append(_record_time_s(record, position))
    frame = pd.DataFrame(rows, index=pd.Index(times, name="time_s"))
    if not frame.empty:
        frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def extract_quantity(
    records: Sequence[Any], spec: QuantitySpec
) -> pd.DataFrame:
    """Extract and combine every field in a quantity specification."""
    frames = [
        extract_field(records, name, alias=spec.aliases.get(name))
        for name in spec.fields
    ]
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, axis=1) if nonempty else pd.DataFrame()


def numeric_record_fields(records: Sequence[Any]) -> list[str]:
    """Discover fields having at least one numeric scalar leaf."""
    fields: set[str] = set()
    for record in records[: min(len(records), 5)]:
        names = record.keys() if isinstance(record, Mapping) else vars(record)
        for name in names:
            if next(_flatten_numeric(_record_value(record, name)), None) is not None:
                fields.add(str(name))
    return sorted(fields)


def _unit_for_field(field_name: str) -> str:
    lower = field_name.lower()
    if lower.endswith("_pu") or "_v_" in lower or lower == "zone_v_gen":
        return "Value [p.u.]"
    if "mvar" in lower or "_q_" in lower or lower.endswith("_q"):
        return "Reactive power [Mvar]"
    if lower.endswith("_mw") or "_p_" in lower:
        return "Active power [MW]"
    if lower.endswith("_ka"):
        return "Current [kA]"
    if lower.endswith("_pct"):
        return "Value [%]"
    if "tap" in lower or "shunt" in lower or "state" in lower:
        return "Discrete position [step]"
    return field_name


def specification(key: str) -> QuantitySpec:
    """Resolve a curated key or a dynamic ``field:<record field>`` key."""
    if key in QUANTITIES:
        return QUANTITIES[key]
    if key.startswith("field:"):
        name = key.partition(":")[2]
        return QuantitySpec(f"Record field: {name}", (name,), _unit_for_field(name))
    raise KeyError(f"Unknown quantity key: {key}")


def comparison_frames(
    runs: Iterable[RunInfo], spec: QuantitySpec
) -> dict[str, pd.DataFrame]:
    """Extract the selected quantity from each run."""
    return {
        run.identifier: extract_quantity(load_records(run), spec)
        for run in runs
    }


def available_channels(frames: Mapping[str, pd.DataFrame]) -> list[str]:
    """Return the sorted union of channel columns across runs."""
    return sorted(
        {str(column) for frame in frames.values() for column in frame.columns}
    )


def _difference_series(
    series: pd.Series, baseline: pd.Series
) -> pd.Series:
    index = series.index.union(baseline.index).sort_values()
    current = series.reindex(index).interpolate(method="index", limit_area="inside")
    reference = baseline.reindex(index).interpolate(
        method="index", limit_area="inside"
    )
    return (current - reference).dropna()


def plot_comparison(
    frames: Mapping[str, pd.DataFrame],
    channels: Sequence[str],
    spec: QuantitySpec,
    *,
    layout: str = "small_multiples",
    difference_to_first: bool = False,
    time_unit: str = "minutes",
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot selected channels and return a compact per-trace summary table."""
    if len(frames) < 2:
        raise ValueError("Select at least two result runs.")
    chosen = [channel for channel in channels if channel]
    if not chosen:
        raise ValueError("Select at least one channel.")

    run_ids = list(frames)
    scale, xlabel = {
        "seconds": (1.0, "Simulation time [s]"),
        "minutes": (60.0, "Simulation time [min]"),
        "hours": (3600.0, "Simulation time [h]"),
    }[time_unit]
    ylabel = f"Difference in {spec.ylabel.lower()}" if difference_to_first else spec.ylabel

    if layout == "small_multiples":
        ncols = 2 if len(chosen) > 1 else 1
        nrows = int(np.ceil(len(chosen) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(7.0 * ncols, 3.4 * nrows),
            squeeze=False,
            sharex=True,
        )
        plot_axes = list(axes.flat)
    else:
        fig, axis = plt.subplots(figsize=(13, 6))
        plot_axes = [axis]

    colors = plt.get_cmap("tab10")
    linestyles = ("-", "--", "-.", ":")
    summary_rows: list[dict[str, Any]] = []
    baseline_id = run_ids[0]

    for channel_index, channel in enumerate(chosen):
        axis = plot_axes[channel_index] if layout == "small_multiples" else plot_axes[0]
        baseline_frame = frames[baseline_id]
        baseline = (
            baseline_frame[channel].dropna()
            if channel in baseline_frame
            else pd.Series(dtype=float)
        )
        for run_index, run_id in enumerate(run_ids):
            frame = frames[run_id]
            if channel not in frame:
                continue
            series = frame[channel].dropna()
            if difference_to_first:
                if baseline.empty:
                    continue
                series = _difference_series(series, baseline)
            if series.empty:
                continue
            label = run_id if layout == "small_multiples" else f"{channel} | {run_id}"
            color_index = run_index if layout == "small_multiples" else channel_index
            axis.plot(
                series.index.to_numpy(dtype=float) / scale,
                series.to_numpy(dtype=float),
                label=label,
                color=colors(color_index % 10),
                linestyle=linestyles[run_index % len(linestyles)],
                linewidth=1.6,
            )
            summary_rows.append(
                {
                    "run": run_id,
                    "channel": channel,
                    "minimum": float(series.min()),
                    "mean": float(series.mean()),
                    "maximum": float(series.max()),
                    "final": float(series.iloc[-1]),
                }
            )
        if layout == "small_multiples":
            axis.set_title(channel)
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=8)

    if layout != "small_multiples":
        axis = plot_axes[0]
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8, ncols=2)

    for axis in plot_axes[: len(chosen)]:
        axis.set_xlabel(xlabel)
    for axis in plot_axes[len(chosen) :]:
        axis.set_visible(False)
    title = spec.label
    if difference_to_first:
        title += f" — difference to {baseline_id}"
    fig.suptitle(title)
    fig.tight_layout()
    return fig, pd.DataFrame(summary_rows)


def create_dashboard(results_root: Optional[Path] = None):
    """Build the ipywidgets dashboard used by the comparison notebook."""
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError as exc:  # pragma: no cover - depends on notebook runtime
        raise ImportError(
            "The interactive notebook requires ipywidgets. Install it in the "
            "active kernel with `%pip install ipywidgets`, restart the kernel, "
            "and rerun the notebook."
        ) from exc

    root = Path(results_root or (find_project_root() / "results")).resolve()
    run_lookup: dict[str, RunInfo] = {}

    run_widget = widgets.SelectMultiple(
        description="Runs",
        rows=9,
        layout=widgets.Layout(width="98%"),
    )
    quantity_widget = widgets.Dropdown(
        description="Quantity",
        layout=widgets.Layout(width="98%"),
    )
    channel_widget = widgets.SelectMultiple(
        description="Channels",
        rows=10,
        layout=widgets.Layout(width="98%"),
    )
    layout_widget = widgets.Dropdown(
        options=[("Small multiples", "small_multiples"), ("Overlay", "overlay")],
        value="small_multiples",
        description="Layout",
    )
    difference_widget = widgets.Checkbox(
        value=False,
        description="Difference to first selected run",
        indent=False,
    )
    time_widget = widgets.Dropdown(
        options=("seconds", "minutes", "hours"),
        value="minutes",
        description="Time unit",
    )
    plot_button = widgets.Button(
        description="Compare selected runs",
        button_style="primary",
        icon="line-chart",
    )
    rescan_button = widgets.Button(description="Rescan results", icon="refresh")
    status = widgets.HTML()
    output = widgets.Output()

    def selected_runs() -> list[RunInfo]:
        return [run_lookup[key] for key in run_widget.value if key in run_lookup]

    def selected_spec() -> QuantitySpec:
        return specification(quantity_widget.value)

    def update_quantity_options(*_args) -> None:
        current = quantity_widget.value
        fields: set[str] = set()
        errors: list[str] = []
        for run in selected_runs():
            try:
                fields.update(numeric_record_fields(load_records(run)))
            except Exception as exc:  # keep the remaining valid runs usable
                errors.append(f"{run.identifier}: {exc}")
        options = [(item.label, key) for key, item in QUANTITIES.items()]
        options.extend((f"Record field: {name}", f"field:{name}") for name in sorted(fields))
        quantity_widget.options = options
        values = {value for _, value in options}
        quantity_widget.value = current if current in values else "tso_voltage"
        if errors:
            status.value = "<br>".join(f"<b>Load error:</b> {item}" for item in errors)

    def update_channels(*_args) -> None:
        prior = set(channel_widget.value)
        try:
            frames = comparison_frames(selected_runs(), selected_spec())
            channels = available_channels(frames)
        except Exception as exc:
            channel_widget.options = ()
            channel_widget.value = ()
            status.value = f"<b>Selection error:</b> {exc}"
            return
        channel_widget.options = channels
        retained = tuple(item for item in channels if item in prior)
        channel_widget.value = retained or tuple(channels[: min(8, len(channels))])
        run_count = len(selected_runs())
        status.value = (
            f"Found <b>{len(run_lookup)}</b> saved runs; selected "
            f"<b>{run_count}</b> runs and <b>{len(channels)}</b> available channels."
        )

    def rescan(*_args) -> None:
        prior = set(run_widget.value)
        found = discover_runs(root)
        run_lookup.clear()
        run_lookup.update((run.identifier, run) for run in found)
        identifiers = list(run_lookup)
        run_widget.options = identifiers
        retained = tuple(item for item in identifiers if item in prior)
        run_widget.value = retained or tuple(identifiers[-2:])
        update_quantity_options()
        update_channels()

    def run_changed(*_args) -> None:
        update_quantity_options()
        update_channels()

    def quantity_changed(*_args) -> None:
        update_channels()

    def draw(*_args) -> None:
        with output:
            clear_output(wait=True)
            if len(selected_runs()) < 2:
                print("Select at least two runs before plotting.")
                return
            try:
                spec = selected_spec()
                frames = comparison_frames(selected_runs(), spec)
                figure, summary = plot_comparison(
                    frames,
                    channel_widget.value,
                    spec,
                    layout=layout_widget.value,
                    difference_to_first=difference_widget.value,
                    time_unit=time_widget.value,
                )
                display(figure)
                plt.close(figure)
                if not summary.empty:
                    display(summary.set_index(["channel", "run"]).round(5))
            except Exception as exc:
                print(f"Comparison failed: {type(exc).__name__}: {exc}")

    run_widget.observe(run_changed, names="value")
    quantity_widget.observe(quantity_changed, names="value")
    plot_button.on_click(draw)
    rescan_button.on_click(rescan)

    controls = widgets.VBox(
        [
            widgets.HTML(
                "<b>1.</b> Select two or more result identifiers. "
                "<b>2.</b> Choose a quantity and one or more channels. "
                "<b>3.</b> Compare."
            ),
            run_widget,
            quantity_widget,
            channel_widget,
            widgets.HBox([layout_widget, time_widget]),
            difference_widget,
            widgets.HBox([plot_button, rescan_button]),
            status,
        ]
    )
    rescan()
    return widgets.VBox([controls, output])


__all__ = [
    "QUANTITIES",
    "QuantitySpec",
    "RunInfo",
    "available_channels",
    "comparison_frames",
    "create_dashboard",
    "discover_runs",
    "extract_field",
    "extract_quantity",
    "find_project_root",
    "load_records",
    "numeric_record_fields",
    "plot_comparison",
    "specification",
]
