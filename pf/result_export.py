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


# =====================================================================
#  Q(V) dead-band internals: parameter/derived-signal realignment
# =====================================================================

#: Rows whose identity residual exceeds this are treated as candidates for an
#: event-instant split.  The ComRes CSV carries six decimals, so the rounding
#: quantum alone is 1e-6 and a threshold an order above it separates a genuine
#: split (which is the size of the parameter jump) from export rounding.
QV_SPLIT_EPS: float = 1e-5

#: A realignment is only applied when using the previous parameter value cuts
#: the residual by at least this factor.  Guards the case where the droop
#: contribution ``Kdroop*qcorr`` is itself comparable to the setpoint step, in
#: which case neither alignment is decisive and the row is left alone.
QV_REALIGN_MARGIN: float = 0.5


def _realign_one(param: np.ndarray, resid_now: np.ndarray,
                 resid_prev: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Revert ``param`` to its previous sample where that is the consistent
    reading.  Returns ``(param, n_realigned, n_ambiguous)``."""
    out = param.copy()
    changed = np.zeros(param.shape, dtype=bool)
    changed[1:] = np.abs(np.diff(param)) > 1e-12
    split = changed & (resid_now > QV_SPLIT_EPS)
    decisive = split & (resid_prev < QV_REALIGN_MARGIN * resid_now)
    ambiguous = split & ~decisive
    idx = np.flatnonzero(decisive)
    out[idx] = param[idx - 1]
    return out, int(idx.size), int(ambiguous.sum())


def realign_qv_parameter_channels(
    qv_raw: Dict[str, Trajectory],
) -> Tuple[Dict[str, Trajectory], Dict[str, int]]:
    """Restore simultaneity between QVPRE parameters and derived signals.

    ``Vanchor`` and ``qset`` are written by pooled ``EvtParam`` events.  Within
    the integration step that applies such an event, PowerFactory may store the
    result row *after* the parameter write but *before* the DSL block
    re-evaluates its internal variables.  That row then carries the new
    parameter beside the pre-event ``veff``/``qcorr``/``qraw``/``Qext`` -- a
    state the block never actually held.

    It is a race, not a fixed lag: on the 2026-09-16 verification run only 157
    of 1165 anchor updates and 152 of 736 setpoint writes were split, so a
    blanket shift would corrupt the ~87 % that are already consistent.  Each
    change row is therefore tested against the block's own algebra
    (``veff == x1 - Vanchor`` for the anchor, ``qraw == qset - Kdroop*qcorr``
    for the setpoint) and the parameter is reverted only where the previous
    value is decisively the consistent one.

    Left uncorrected this is not cosmetic: ``qraw - qset`` is F1's incremental
    droop command, and a split row reports a spurious command the size of the
    whole setpoint step (0.152 pu on that run) at every dispatch, which would
    fail the F1 confirmation for a reason unrelated to the dead band.

    Returns the realigned channel dict and a per-channel count of the rows
    touched, for the run summary.
    """
    out = dict(qv_raw)
    stats: Dict[str, int] = {
        "anchor_rows_realigned": 0, "anchor_rows_ambiguous": 0,
        "qset_rows_realigned": 0, "qset_rows_ambiguous": 0,
        "parks": 0,
    }
    parks = sorted(label[len("vanchorDER_"):] for label in qv_raw
                   if label.startswith("vanchorDER_"))
    stats["parks"] = len(parks)

    for park in parks:
        def get(prefix: str) -> Optional[np.ndarray]:
            entry = qv_raw.get(f"{prefix}{park}")
            return None if entry is None else np.asarray(entry[1], dtype=float)

        v_anchor, v_filt, v_eff = get("vanchorDER_"), get("vfiltDER_"), get("veffDER_")
        q_set, q_raw = get("qsetDER_"), get("qrawDER_")

        if (v_anchor is not None and v_filt is not None and v_eff is not None
                and v_anchor.size == v_filt.size == v_eff.size
                and v_anchor.size > 1):
            prev = np.concatenate(([v_anchor[0]], v_anchor[:-1]))
            fixed, n, n_amb = _realign_one(
                v_anchor,
                np.abs(v_eff - (v_filt - v_anchor)),
                np.abs(v_eff - (v_filt - prev)),
            )
            out[f"vanchorDER_{park}"] = (qv_raw[f"vanchorDER_{park}"][0], fixed)
            stats["anchor_rows_realigned"] += n
            stats["anchor_rows_ambiguous"] += n_amb

        if (q_set is not None and q_raw is not None
                and q_set.size == q_raw.size and q_set.size > 1):
            prev = np.concatenate(([q_set[0]], q_set[:-1]))
            fixed, n, n_amb = _realign_one(
                q_set, np.abs(q_raw - q_set), np.abs(q_raw - prev))
            out[f"qsetDER_{park}"] = (qv_raw[f"qsetDER_{park}"][0], fixed)
            stats["qset_rows_realigned"] += n
            stats["qset_rows_ambiguous"] += n_amb

    return out, stats
