"""Event-free PowerFactory RMS playback of known exogenous profiles.

The online OFO commands remain simulation events because they depend on the
preceding RMS measurement.  Only trajectories known before ``ComInc`` are
handled here:

* ``ElmFile.y1/y2 -> ElmLod.Pext/Qext`` for profile-driven loads;
* ``ElmFile.y1 -> WTGWGO_A.Pref_in`` for profile-driven DER active power.

All created database objects carry the ``qOFO RMS Profile`` ownership prefix.
The installer is deliberately idempotent: it restores the original WECC frame
and removes an earlier installation before constructing the current one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from pf.session import PFSessionError, get_all
from pf.wecc_apply import LOCAL_FRAME_NAME


OWNER_PREFIX = "qOFO RMS Profile"
LOAD_FRAME_NAME = f"{OWNER_PREFIX} Load Frame"
DER_FRAME_NAME = f"{OWNER_PREFIX} WECC Frame"
LOAD_COMPOSITE_PREFIX = f"{OWNER_PREFIX} Load "
LOAD_SOURCE_PREFIX = f"{OWNER_PREFIX} Load Source "
DER_SOURCE_PREFIX = f"{OWNER_PREFIX} DER Source "
DER_FILE_SLOT_NAME = "Profile File"
UNITY_COLUMN = "__unity__"
MAX_ELMFILE_CHANNELS = 24


@dataclass(frozen=True)
class ProfileSchedule:
    """Dense, pre-ComInc measurement-file payload."""

    columns: Tuple[str, ...]
    time_s: np.ndarray
    values: np.ndarray

    def channel(self, name: str) -> int:
        """Return the 1-based ElmFile data-column index for *name*."""

        try:
            return self.columns.index(str(name)) + 1
        except ValueError as exc:
            raise PFSessionError(
                f"profile column {name!r} is absent from the RMS schedule"
            ) from exc


@dataclass(frozen=True)
class ProfilePlaybackInstallation:
    """Provenance and event-pruning targets returned by the installer."""

    file_path: Path
    columns: Tuple[str, ...]
    time_rows: int
    load_indices: Tuple[int, ...]
    sgen_indices: Tuple[int, ...]
    load_targets: Tuple[Any, ...]
    pref_targets: Tuple[Any, ...]
    removed_objects: Mapping[str, int]


def _profile_name(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def referenced_profile_columns(net) -> Tuple[str, ...]:
    """Return the distinct file channels used by load and DER playback."""

    names = set()
    for table_name, columns in (
        ("load", ("profile_p", "profile_q")),
        ("sgen", ("profile",)),
    ):
        table = getattr(net, table_name)
        for column in columns:
            if column not in table.columns:
                continue
            for value in table[column]:
                name = _profile_name(value)
                if name is not None:
                    names.add(name)
    return tuple(sorted(names))


def build_profile_schedule(
    profiles: pd.DataFrame,
    *,
    columns: Sequence[str],
    start_time: datetime,
    dt_s: float,
    duration_s: float,
    transition_delay_s: float,
) -> ProfileSchedule:
    """Translate runner wall-clock rows to the existing RMS dispatch timing.

    ``run_multi_tso_dso`` applies the initial row before plant construction.
    During interval ``k`` it applies wall time ``start + k*dt`` at paused RMS
    time ``(k-1)*dt``.  Parameter/load events scheduled at ``+0.5 s`` become
    visible on the next 10-ms result row, hence the production file transition
    delay is normally 0.51 s.
    """

    dt_s = float(dt_s)
    duration_s = float(duration_s)
    transition_delay_s = float(transition_delay_s)
    if dt_s <= 0.0 or duration_s <= 0.0:
        raise ValueError("profile dt_s and duration_s must be positive")
    steps = duration_s / dt_s
    if abs(steps - round(steps)) > 1.0e-9:
        raise ValueError("profile duration_s must be a multiple of dt_s")
    if not (0.0 < transition_delay_s < dt_s):
        raise ValueError("profile transition delay must lie inside one interval")
    if not isinstance(profiles.index, pd.DatetimeIndex) or profiles.empty:
        raise ValueError("profiles must be a non-empty DatetimeIndex DataFrame")
    if not profiles.index.is_monotonic_increasing:
        raise ValueError("profile index must be monotone increasing")

    requested = tuple(str(column) for column in columns)
    missing = [column for column in requested if column not in profiles.columns]
    if missing:
        raise PFSessionError(
            f"RMS profile playback references missing columns: {missing}"
        )
    schedule_columns = (UNITY_COLUMN,) + requested
    if len(schedule_columns) > MAX_ELMFILE_CHANNELS:
        raise PFSessionError(
            f"ElmFile supports at most {MAX_ELMFILE_CHANNELS} channels; "
            f"{len(schedule_columns)} are required"
        )

    n_steps = int(round(steps))
    wall_times = [start_time] + [
        start_time + timedelta(seconds=k * dt_s)
        for k in range(1, n_steps + 1)
    ]
    positions = profiles.index.get_indexer(wall_times, method="nearest")
    if np.any(positions < 0):
        raise PFSessionError("one or more RMS profile timestamps cannot be resolved")

    selected = profiles.iloc[positions].loc[:, requested]
    profile_values = selected.to_numpy(dtype=float, copy=True)
    if not np.isfinite(profile_values).all():
        bad = np.argwhere(~np.isfinite(profile_values))[0]
        raise PFSessionError(
            "non-finite RMS profile value at "
            f"wall time {wall_times[int(bad[0])]!s}, "
            f"column {requested[int(bad[1])]!r}"
        )

    time_s = np.asarray(
        [0.0] + [
            (k - 1) * dt_s + transition_delay_s
            for k in range(1, n_steps + 1)
        ],
        dtype=float,
    )
    values = np.column_stack(
        (np.ones(len(wall_times), dtype=float), profile_values)
    )
    return ProfileSchedule(schedule_columns, time_s, values)


def write_measurement_file(path: Path, schedule: ProfileSchedule) -> Path:
    """Write PowerFactory's plain measurement-file format as UTF-8 ASCII."""

    path = Path(path)
    if schedule.values.shape != (
        len(schedule.time_s),
        len(schedule.columns),
    ):
        raise ValueError("profile schedule shape is inconsistent")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(schedule.columns))]
    for time_s, row in zip(schedule.time_s, schedule.values):
        line = " ".join(
            [format(float(time_s), ".12g")]
            + [format(float(value), ".12g") for value in row]
        )
        if len(line) > 1024:
            raise PFSessionError(
                "PowerFactory measurement-file row exceeds 1024 characters"
            )
        lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _objects_named(parent, class_name: str, name: str) -> list[Any]:
    return list(parent.GetContents(f"{name}.{class_name}"))


def _unique_base_frame(app):
    folder = app.GetProjectFolder("blk")
    hits = _objects_named(folder, "BlkDef", LOCAL_FRAME_NAME)
    if len(hits) != 1:
        raise PFSessionError(
            f"expected one {LOCAL_FRAME_NAME!r} frame, found {len(hits)}"
        )
    return folder, hits[0]


def _mapping_by_slot(comp) -> Dict[str, Any]:
    pblk = list(comp.GetAttribute("pblk") or [])
    pelm = list(comp.GetAttribute("pelm") or [])
    if len(pblk) != len(pelm):
        raise PFSessionError(
            f"{comp.loc_name}: pblk/pelm lengths differ "
            f"({len(pblk)} != {len(pelm)})"
        )
    return {
        str(slot.loc_name): element
        for slot, element in zip(pblk, pelm)
        if slot is not None
    }


def _bind_by_slot_name(comp, mapping: Mapping[str, Any]) -> None:
    pblk = list(comp.GetAttribute("pblk") or [])
    missing = [
        str(slot.loc_name)
        for slot in pblk
        if slot is not None and str(slot.loc_name) not in mapping
    ]
    if missing:
        raise PFSessionError(
            f"{comp.loc_name}: no element mapping for frame slots {missing}"
        )
    comp.SetAttribute(
        "pelm",
        [mapping[str(slot.loc_name)] for slot in pblk],
    )


def remove_profile_playback_models(app) -> Dict[str, int]:
    """Remove only objects owned by this module and restore WECC frames."""

    app.ResetCalculation()
    frame_folder, base_frame = _unique_base_frame(app)
    derived_frames = _objects_named(frame_folder, "BlkDef", DER_FRAME_NAME)

    restored = 0
    for comp in list(get_all(app, "ElmComp")):
        typ = comp.GetAttribute("typ_id")
        if typ is None or str(typ.loc_name) != DER_FRAME_NAME:
            continue
        mapping = _mapping_by_slot(comp)
        mapping.pop(DER_FILE_SLOT_NAME, None)
        comp.SetAttribute("typ_id", base_frame)
        _bind_by_slot_name(comp, mapping)
        restored += 1

    removed_composites = 0
    for comp in list(get_all(app, "ElmComp")):
        if str(comp.loc_name).startswith(LOAD_COMPOSITE_PREFIX):
            comp.Delete()
            removed_composites += 1

    removed_sources = 0
    for source in list(get_all(app, "ElmFile")):
        if str(source.loc_name).startswith(
            (LOAD_SOURCE_PREFIX, DER_SOURCE_PREFIX)
        ):
            source.Delete()
            removed_sources += 1

    removed_frames = 0
    for name in (LOAD_FRAME_NAME, DER_FRAME_NAME):
        for frame in _objects_named(frame_folder, "BlkDef", name):
            frame.Delete()
            removed_frames += 1
    return {
        "wecc_restored": restored,
        "composites_removed": removed_composites,
        "sources_removed": removed_sources,
        "frames_removed": removed_frames,
    }


def _create_load_frame(frame_folder):
    frame = frame_folder.CreateObject("BlkDef", LOAD_FRAME_NAME)
    if frame is None:
        raise PFSessionError("failed to create RMS profile load frame")
    load_slot = frame.CreateObject("BlkSlot", "Load")
    file_slot = frame.CreateObject("BlkSlot", "File")
    if load_slot is None or file_slot is None:
        raise PFSessionError("failed to create RMS profile load-frame slots")
    load_slot.SetAttribute("sInput", ["Pext,Qext"])
    file_slot.SetAttribute("sOutput", ["y1,y2"])
    for index, name in enumerate(("Pext", "Qext")):
        signal = frame.CreateObject("BlkSig", name)
        if signal is None:
            raise PFSessionError(f"failed to create load-frame signal {name}")
        signal.SetAttribute("pnodfrom", file_slot)
        signal.SetAttribute("pnodto", load_slot)
        signal.SetAttribute("inodfrom", index)
        signal.SetAttribute("inodto", index)
        signal.SetAttribute("iconfrom", 2)
        signal.SetAttribute("iconto", 1)
    return frame, load_slot, file_slot


def _create_der_frame(frame_folder, base_frame):
    frame = frame_folder.AddCopy(base_frame, DER_FRAME_NAME)
    if frame is None:
        raise PFSessionError("failed to copy the WECC profile frame")
    frame.loc_name = DER_FRAME_NAME
    slots = {
        str(obj.loc_name): obj
        for obj in frame.GetContents()
        if obj.GetClassName() == "BlkSlot"
    }
    wgo_slot = slots.get("Weak Grid Option")
    if wgo_slot is None:
        raise PFSessionError("copied WECC frame has no Weak Grid Option slot")
    file_slot = frame.CreateObject("BlkSlot", DER_FILE_SLOT_NAME)
    if file_slot is None:
        raise PFSessionError("failed to create WECC profile-file slot")
    file_slot.SetAttribute("sOutput", ["y1"])
    signal = frame.CreateObject("BlkSig", "Pref_in_profile")
    if signal is None:
        raise PFSessionError("failed to create WECC Pref_in profile signal")
    signal.SetAttribute("pnodfrom", file_slot)
    signal.SetAttribute("pnodto", wgo_slot)
    signal.SetAttribute("inodfrom", 0)
    signal.SetAttribute("inodto", 0)
    signal.SetAttribute("iconfrom", 2)
    signal.SetAttribute("iconto", 1)
    return frame


def _configure_source(
    source,
    *,
    file_path: Path,
    channels: Sequence[int],
    scales: Sequence[float],
) -> None:
    if len(channels) != len(scales):
        raise ValueError("ElmFile channel/scale lengths differ")
    if not channels or len(channels) > MAX_ELMFILE_CHANNELS:
        raise ValueError("ElmFile requires between 1 and 24 channels")
    n = MAX_ELMFILE_CHANNELS
    icol = list(map(int, channels)) + [1] * (n - len(channels))
    afac = list(map(float, scales)) + [1.0] * (n - len(scales))
    source.SetAttribute("iopt_imp", 1)
    source.SetAttribute("f_name", str(Path(file_path).resolve()))
    source.SetAttribute("icol", icol)
    source.SetAttribute("afac", afac)
    source.SetAttribute("bfac", [0.0] * n)
    source.SetAttribute("tini", 0.0)
    source.SetAttribute("approx", 0)


def install_profile_playback(
    app,
    net,
    *,
    loads: Mapping[int, Any],
    sgens: Mapping[int, Any],
    wgo: Mapping[int, Any],
    wecc_composites: Mapping[int, Any],
    sgen_sn: Mapping[int, float],
    profiles: pd.DataFrame,
    start_time: datetime,
    dt_s: float,
    duration_s: float,
    file_path: Path,
    transition_delay_s: float,
) -> ProfilePlaybackInstallation:
    """Install all known load/DER-P trajectories before ``ComInc``."""

    removed = remove_profile_playback_models(app)
    columns = referenced_profile_columns(net)
    schedule = build_profile_schedule(
        profiles,
        columns=columns,
        start_time=start_time,
        dt_s=dt_s,
        duration_s=duration_s,
        transition_delay_s=transition_delay_s,
    )
    file_path = write_measurement_file(file_path, schedule)

    frame_folder, base_frame = _unique_base_frame(app)
    load_frame, load_slot, load_file_slot = _create_load_frame(frame_folder)
    der_frame = _create_der_frame(frame_folder, base_frame)

    load_indices = []
    load_targets = []
    for idx, load in sorted(loads.items()):
        if idx not in net.load.index:
            continue
        p_profile = (
            _profile_name(net.load.at[idx, "profile_p"])
            if "profile_p" in net.load.columns else None
        )
        q_profile = (
            _profile_name(net.load.at[idx, "profile_q"])
            if "profile_q" in net.load.columns else None
        )
        if p_profile is None and q_profile is None:
            continue
        p_channel = schedule.channel(p_profile or UNITY_COLUMN)
        q_channel = schedule.channel(q_profile or UNITY_COLUMN)
        p_base = float(net.load.at[idx, "base_p_mw"])
        q_base = float(net.load.at[idx, "base_q_mvar"])
        parent = load.GetParent()
        source = parent.CreateObject(
            "ElmFile", f"{LOAD_SOURCE_PREFIX}{int(idx)}"
        )
        comp = parent.CreateObject(
            "ElmComp", f"{LOAD_COMPOSITE_PREFIX}{int(idx)}"
        )
        if source is None or comp is None:
            raise PFSessionError(
                f"failed to create RMS profile objects for load[{idx}]"
            )
        _configure_source(
            source,
            file_path=file_path,
            channels=(p_channel, q_channel),
            scales=(p_base, q_base),
        )
        comp.SetAttribute("typ_id", load_frame)
        comp.SetAttribute("pblk", [load_slot, load_file_slot])
        comp.SetAttribute("pelm", [load, source])
        load_indices.append(int(idx))
        load_targets.append(load)

    sgen_indices = []
    pref_targets = []
    for idx, gen in sorted(sgens.items()):
        if idx not in net.sgen.index or "profile" not in net.sgen.columns:
            continue
        profile = _profile_name(net.sgen.at[idx, "profile"])
        if profile is None:
            continue
        if str(net.sgen.at[idx, "name"]).startswith("BOUND_"):
            continue
        pref = wgo.get(idx)
        comp = wecc_composites.get(idx)
        sn = float(sgen_sn[idx])
        if pref is None or comp is None or sn <= 0.0:
            raise PFSessionError(
                f"sgen[{idx}] cannot be wired to RMS profile playback"
            )
        parent = comp.GetParent()
        source = parent.CreateObject(
            "ElmFile", f"{DER_SOURCE_PREFIX}{int(idx)}"
        )
        if source is None:
            raise PFSessionError(
                f"failed to create RMS profile source for sgen[{idx}]"
            )
        _configure_source(
            source,
            file_path=file_path,
            channels=(schedule.channel(profile),),
            scales=(float(net.sgen.at[idx, "base_p_mw"]) / sn,),
        )
        mapping = _mapping_by_slot(comp)
        mapping[DER_FILE_SLOT_NAME] = source
        comp.SetAttribute("typ_id", der_frame)
        _bind_by_slot_name(comp, mapping)
        sgen_indices.append(int(idx))
        pref_targets.append(pref)

    if not load_indices and not sgen_indices:
        raise PFSessionError(
            "RMS profile playback installed no load or DER profile targets"
        )
    return ProfilePlaybackInstallation(
        file_path=file_path,
        columns=schedule.columns,
        time_rows=len(schedule.time_s),
        load_indices=tuple(load_indices),
        sgen_indices=tuple(sgen_indices),
        load_targets=tuple(load_targets),
        pref_targets=tuple(pref_targets),
        removed_objects=removed,
    )

