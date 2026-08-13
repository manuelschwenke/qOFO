"""Bulk PowerFactory ``ElmRes`` export and sampled trajectory loading.

``ElmRes.GetValue`` is useful for small interactive reads, but one Python API
call per result cell is prohibitively slow for long RMS traces.  This module
uses ``ComRes`` for one native CSV export, then performs the row sampling in
pandas/NumPy without further PowerFactory calls.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pf.session import PFSessionError


Monitor = Tuple[Any, str, str]
Trajectory = Tuple[np.ndarray, np.ndarray]

_CONTAINER_CLASS_SUFFIX = re.compile(
    r"\.(?:IntUser|IntPrj|IntPrjfolder|IntCase|ElmNet|ElmComp)$"
)


def _normalise_object_path(value: str) -> Tuple[str, ...]:
    """Return comparable PF/ComRes path segments.

    ``GetFullName()`` includes database container classes and the full
    user/project prefix.  A ComRes header starts at the calculation-relevant
    grid or study-case folder and omits container classes.  Removing class
    suffixes from non-leaf segments makes the latter a unique suffix of the
    former while retaining the leaf object's class.
    """

    parts = [
        part.strip()
        for part in str(value).strip().strip('"').lstrip("\\").split("\\")
        if part.strip()
    ]
    for index in range(max(0, len(parts) - 1)):
        parts[index] = _CONTAINER_CLASS_SUFFIX.sub("", parts[index])
    return tuple(parts)


def _base_variable(value: str) -> str:
    """Strip the unit text which ComRes appends to a variable identifier."""

    return str(value).strip().strip('"').split(" in ", 1)[0].strip()


def export_comres_csv(app: Any, result: Any, csv_path: Path | str) -> Path:
    """Export the complete ``ElmRes`` through PowerFactory's native ComRes.

    The validated ``iopt_sep=1`` format uses semicolons and decimal commas.
    ``iopt_csel=0`` is intentional: the complete export includes ``b:tnow``;
    selected-column exports omit the time column in PowerFactory 2025 SP4.
    """

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comres = app.GetFromStudyCase("ComRes")
    if comres is None:
        raise PFSessionError("ComRes not found in the active study case")

    settings = {
        "pResult": result,
        "iopt_exp": 6,       # CSV
        "f_name": str(path),
        "iopt_sep": 1,       # system separator: ';' and decimal comma here
        "iopt_honly": 0,     # header plus data
        "iopt_csel": 0,      # all columns, including b:tnow
    }
    for name, value in settings.items():
        comres.SetAttribute(name, value)
    if comres.Execute():
        raise PFSessionError(f"ComRes export failed: {path}")
    if not path.is_file() or path.stat().st_size == 0:
        raise PFSessionError(f"ComRes produced no CSV data: {path}")
    return path


def _read_comres_headers(csv_path: Path) -> Tuple[list[str], list[str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        try:
            objects = next(reader)
            variables = next(reader)
        except StopIteration as exc:
            raise ValueError(f"ComRes CSV has fewer than two header rows: {csv_path}") from exc
    if len(objects) != len(variables):
        raise ValueError(
            "ComRes object/variable header widths differ: "
            f"{len(objects)} != {len(variables)}"
        )
    return objects, variables


def _monitor_columns(
    objects: Sequence[str],
    variables: Sequence[str],
    monitors: Iterable[Monitor],
    labels: Optional[set[str] | Callable[[str], bool]],
) -> Tuple[int, list[Tuple[str, int]]]:
    header_paths = [_normalise_object_path(value) for value in objects]
    header_vars = [_base_variable(value) for value in variables]
    time_columns = [
        index for index, variable in enumerate(header_vars)
        if variable == "b:tnow"
    ]
    if len(time_columns) != 1:
        raise ValueError(
            f"expected one b:tnow column in ComRes CSV, found {time_columns}"
        )

    mapped: list[Tuple[str, int]] = []
    for obj, variable, label in monitors:
        if labels is not None:
            keep = labels(label) if callable(labels) else label in labels
            if not keep:
                continue
        full_path = _normalise_object_path(obj.GetFullName())
        candidates = []
        for index, (header_path, header_var) in enumerate(
            zip(header_paths, header_vars)
        ):
            if header_var != variable or len(header_path) > len(full_path):
                continue
            if full_path[-len(header_path):] == header_path:
                candidates.append(index)
        if not candidates:
            raise ValueError(
                f"monitor not found in ComRes CSV: {label!r}, "
                f"{obj.GetFullName()!r}, {variable!r}"
            )
        # Duplicate AddVars registrations may yield duplicate identical
        # columns.  They carry the same signal; selecting the first is stable.
        mapped.append((label, candidates[0]))
    return time_columns[0], mapped


def load_comres_trajectories(
    csv_path: Path | str,
    monitors: Iterable[Monitor],
    *,
    since_s: float = 0.0,
    stride: int = 5,
    labels: Optional[set[str] | Callable[[str], bool]] = None,
    chunksize: int = 100_000,
) -> Dict[str, Trajectory]:
    """Load monitored trajectories with pandas and sample rows in NumPy.

    Sampling is performed chunk-wise, so memory is bounded for multi-hour RMS
    runs.  As in :meth:`ScreeningContext.read`, row zero and the final result
    row are always retained.
    """

    if stride < 1:
        raise ValueError("stride must be >= 1")
    if chunksize < 1:
        raise ValueError("chunksize must be >= 1")

    path = Path(csv_path)
    objects, variables = _read_comres_headers(path)
    time_column, mapped = _monitor_columns(
        objects, variables, monitors, labels
    )
    if not mapped:
        return {}

    selected_columns = sorted({time_column, *(column for _, column in mapped)})
    samples: Dict[int, list[np.ndarray]] = {
        column: [] for column in selected_columns
    }
    final_row: Optional[pd.Series] = None
    row_offset = 0

    chunks = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        header=None,
        skiprows=2,
        usecols=selected_columns,
        dtype=np.float64,
        chunksize=chunksize,
    )
    for chunk in chunks:
        if chunk.empty:
            continue
        local_rows = np.arange(len(chunk), dtype=np.int64)
        take = ((row_offset + local_rows) % stride) == 0
        sampled = chunk.loc[take]
        for column in selected_columns:
            samples[column].append(
                sampled[column].to_numpy(dtype=float, copy=True)
            )
        final_row = chunk.iloc[-1]
        row_offset += len(chunk)

    if row_offset == 0 or final_row is None:
        raise ValueError(f"ComRes CSV contains no result rows: {path}")

    arrays = {
        column: (
            np.concatenate(parts) if parts else np.empty(0, dtype=float)
        )
        for column, parts in samples.items()
    }
    final_index = row_offset - 1
    if final_index % stride != 0:
        for column in selected_columns:
            arrays[column] = np.append(
                arrays[column], float(final_row[column])
            )

    time = arrays[time_column]
    keep = time >= float(since_s)
    return {
        label: (time[keep].copy(), arrays[column][keep].copy())
        for label, column in mapped
    }
