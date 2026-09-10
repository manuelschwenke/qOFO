"""Actuator-only location sweep for open-loop settling experiment A.

Discrete actuator locations/directions are exhaustive in RMS. Continuous
Q/V-reference locations are ranked by deterministic AC-power-flow impact and
the global/per-area extrema plus a median audit are run in RMS. Controlled
outputs are nodal voltages and DSO/TSO interface Q only: no currents and no
disturbances are present.

OLTCs use an explicit command-to-contact delay followed by a short PT1. The
defaults retain a 5 s mechanical delay but set ``Tmech=0.05 s`` so the contact
change is nearly direct at the fixed 10 ms RMS step. Settling is measured from
the dispatch instant and therefore includes the 5 s waiting time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from export.dynamic_snapshot import load_snapshot, load_snapshot_to_pandapower  # noqa: E402
from pf.naming import build_name_map, machine_template_name  # noqa: E402
from pf.session import DEFAULT_PROJECT_PATH, PFSessionError  # noqa: E402

DEFAULT_SNAPSHOT = REPO_ROOT / "export" / "snapshots" / "full_t0_20160105-0800.json"

BAND_VOLTAGE_PU = 1.0e-3
BAND_Q_MVAR = 1.0
PREFLIGHT_V_DRIFT_PU = BAND_VOLTAGE_PU
PREFLIGHT_Q_DRIFT_MVAR = BAND_Q_MVAR
FINAL_WINDOW_S = 5.0
CONFIRM_WINDOW_S = 5.0
READ_STRIDE = 5

#: The two output families the settling time is reduced over SEPARATELY.
#:
#: Both are reported for every case and both belong in the Ch. 9.1 table:
#: ``t_settle_s`` / ``worst_signal`` over :data:`V_OUTPUT_KINDS`, and
#: ``t_settle_q_s`` / ``worst_signal_q`` over :data:`Q_OUTPUT_KINDS`.  There is
#: no single headline number, because the two are not commensurable and
#: collapsing them to one ``max`` hides which family bound the result --
#: over run ``actuator_location_sweep_t0/20260822-014209`` that max was a Q row
#: in 53 of 70 cases, so a single column read as a voltage claim and was not.
#:
#: A settling time is only interpretable against its band, and the two bands
#: differ in kind.  The voltage band (1 mpu) is a network quantity with the same
#: meaning at every bus, so a worst over the 70 voltage rows is one comparable
#: statement.  The interface-Q band (1 Mvar) is absolute against flows whose own
#: step sizes differ by more than an order of magnitude across the 12 DSO
#: interfaces and 5 TSO corridors, so a Q row can be worst because its band is
#: tight relative to its step rather than because that corridor is slow.  That
#: is an argument for reporting the Q worst in its own column against its own
#: band -- not for dropping it: interface Q is the tracked output of both
#: layers, and a timescale-separation claim that omits it is incomplete.
V_OUTPUT_KINDS: Tuple[str, ...] = ("voltage",)
Q_OUTPUT_KINDS: Tuple[str, ...] = ("dso_interface_q", "tso_boundary_q")

#: The Q family split by the layer that tracks it, reported as
#: ``t_settle_dso_q_s`` / ``t_settle_tso_q_s``.  Not in the dissertation table,
#: but the two are tracked at different periods (DSO 20 s, TSO 180 s), so the
#: pooled Q worst is the wrong quantity to compare against either one alone.
DSO_Q_OUTPUT_KINDS: Tuple[str, ...] = ("dso_interface_q",)
TSO_Q_OUTPUT_KINDS: Tuple[str, ...] = ("tso_boundary_q",)

#: Every kind :func:`build_output_specs` produces.  Used for the provenance
#: record and to assert the two families above partition it.
ALL_OUTPUT_KINDS: Tuple[str, ...] = (
    "voltage", "dso_interface_q", "tso_boundary_q")

assert set(V_OUTPUT_KINDS) | set(Q_OUTPUT_KINDS) == set(ALL_OUTPUT_KINDS)
assert not set(V_OUTPUT_KINDS) & set(Q_OUTPUT_KINDS)

MAX_Q_STEP_MVAR = 60.0
MAX_Q_STEP_FRACTION_SN = 0.5
MAX_AVR_STEP_PU = 0.02
AVR_MIN_PU = 0.95
AVR_MAX_PU = 1.07
DEFAULT_OLTC_DELAY_S = 5.0
DEFAULT_OLTC_TMECH_S = 0.05


@dataclass
class Candidate:
    name: str
    actuator_class: str
    domain: str
    group: str
    mode: str
    kind: str
    table: str
    index: int
    pf_name: str
    direction: int
    variable: str
    delta: float
    physical_delta: float
    unit: str
    event_offset_s: float = 0.0
    note: str = ""
    qss_score: Optional[float] = None
    qss_worst_output: str = ""
    qss_worst_delta: Optional[float] = None
    qss_error: str = ""
    selected: bool = False
    selection_reasons: List[str] = field(default_factory=list)

    @property
    def disturbance(self) -> bool:
        return False


@dataclass(frozen=True)
class OutputSpec:
    kind: str
    table: str
    index: int
    pf_class: str
    pf_name: str
    variable: str
    label: str
    tolerance: float


@dataclass
class ResolvedCase:
    candidate: Candidate
    target: Any
    event_target: Any
    diagnostic_obj: Any
    diagnostic_variable: str
    diagnostic_label: str
    tap_ctrl: Any = None


class Tee:
    def __init__(self, stream: Any, path: Path, *, append: bool = False) -> None:
        self.stream = stream
        self.fh = path.open("a" if append else "w", encoding="utf-8")

    def write(self, value: str) -> int:
        self.stream.write(value)
        self.fh.write(value)
        self.fh.flush()
        return len(value)

    def flush(self) -> None:
        self.stream.flush()
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def _token(value: float, digits: int = 3) -> str:
    number = f"{abs(float(value)):.{digits}f}".rstrip("0").rstrip(".")
    return ("p" if value >= 0.0 else "m") + number.replace(".", "p")


def _zone_lookup(doc: Mapping[str, Any]) -> Dict[int, str]:
    result: Dict[int, str] = {}
    for zone, buses in doc["zone_map"].items():
        for bus in buses:
            result.setdefault(int(bus), f"Z{zone}")
    return result


def _dso_maps(doc: Mapping[str, Any]) -> Tuple[Dict[int, str], Dict[int, str]]:
    sgen_owner: Dict[int, str] = {}
    tr3_owner: Dict[int, str] = {}
    for hv in doc["meta"]["hv_networks"]:
        dso = str(hv["net_id"])
        sgen_owner.update({int(i): dso for i in hv["sgen_indices"]})
        tr3_owner.update({int(i): dso for i in hv["coupling_trafo_indices"]})
    return sgen_owner, tr3_owner


def der_q_capability_mvar(
    p_mw: float, sn_mva: float, op_diagram: str = "VDE-AR-N-4120-v2"
) -> Tuple[float, float]:
    """Same operating-diagram rule as ``ActuatorBounds``."""
    sn = float(sn_mva)
    if sn <= 0.0:
        return 0.0, 0.0
    ratio = abs(float(p_mw)) / sn
    if op_diagram == "STATCOM":
        q = math.sqrt(max(1.0 - min(ratio * ratio, 1.0), 0.0)) * sn
        return -q, q
    if ratio < 0.1:
        return 0.0, 0.0
    if ratio < 0.2:
        alpha = (ratio - 0.1) / 0.1
        return (
            (-0.10 + alpha * (-0.33 + 0.10)) * sn,
            (0.10 + alpha * (0.41 - 0.10)) * sn,
        )
    return -0.33 * sn, 0.41 * sn


def build_candidates(
    doc: Mapping[str, Any], *, oltc_delay_s: float = DEFAULT_OLTC_DELAY_S
) -> List[Candidate]:
    """All feasible one-dispatch actuator steps at the snapshot point."""
    model = doc["model"]
    inventory = doc["actuators"]
    names = build_name_map(doc)
    zone_of = _zone_lookup(doc)
    sgen_owner, tr3_owner = _dso_maps(doc)
    cases: List[Candidate] = []

    def add_q(index: int, actuator_class: str, domain: str, group: str) -> None:
        rec = model["sgen"][str(index)]
        if not bool(rec.get("in_service", True)):
            return
        sn = float(rec["sn_mva"])
        qset = float(rec.get("q_set_mvar", rec.get("q_mvar", 0.0)))
        qmin, qmax = der_q_capability_mvar(
            rec["p_mw"], sn, str(rec.get("op_diagram") or "VDE-AR-N-4120-v2")
        )
        pf_name = names[("sgen", index)]
        for direction, rail in ((-1, qmin), (1, qmax)):
            headroom = max((rail - qset) * direction, 0.0)
            magnitude = min(MAX_Q_STEP_MVAR, MAX_Q_STEP_FRACTION_SN * sn, headroom)
            if magnitude <= 1.0e-9:
                continue
            signed = direction * magnitude
            cases.append(Candidate(
                name=f"q_{actuator_class}_{_token(signed)}Mvar_{pf_name}",
                actuator_class=actuator_class, domain=domain, group=group,
                mode="continuous", kind="param", table="sgen", index=index,
                pf_name=pf_name, direction=direction, variable="qset",
                delta=signed / sn, physical_delta=signed, unit="Mvar",
                note=(f"QVPRE qset; S_n={sn:.3f} MVA; capability "
                      f"[{qmin:.3f}, {qmax:.3f}] Mvar"),
            ))

    for raw in inventory["tso_der_sgen_indices"]:
        idx = int(raw)
        bus = int(model["sgen"][str(idx)]["bus"])
        add_q(idx, "tso_der_q", "TSO", zone_of.get(bus, "unassigned"))

    # This formal inventory includes DER_* and WPC_*; both are DSO-controller
    # Q columns and therefore both are actuator locations.
    for raw in inventory["dso_der_sgen_indices"]:
        idx = int(raw)
        add_q(idx, "dso_der_q", "DSO", sgen_owner.get(idx, "unassigned"))

    gen_grid_bus = {
        int(g): int(b) for g, b in zip(
            doc["meta"]["gen_indices"], doc["meta"]["gen_grid_bus_indices"]
        )
    }
    for raw in inventory["avr_gen_indices"]:
        idx = int(raw)
        rec = model["gen"][str(idx)]
        if bool(rec.get("slack", False)) or not bool(rec.get("in_service", True)):
            continue
        pf_name = machine_template_name(rec)
        base = float(rec["vm_pu"])
        group = zone_of.get(gen_grid_bus.get(idx, int(rec["bus"])), "unassigned")
        for direction, rail in ((-1, AVR_MIN_PU), (1, AVR_MAX_PU)):
            magnitude = min(MAX_AVR_STEP_PU, max((rail - base) * direction, 0.0))
            if magnitude <= 1.0e-9:
                continue
            delta = direction * magnitude
            cases.append(Candidate(
                name=f"avr_{_token(delta)}pu_{pf_name.replace(' ', '')}",
                actuator_class="avr_vref", domain="TSO", group=group,
                mode="continuous", kind="param", table="gen", index=idx,
                pf_name=pf_name, direction=direction, variable="usetp",
                delta=delta, physical_delta=delta, unit="pu",
                note=f"AVR V-reference within [{AVR_MIN_PU}, {AVR_MAX_PU}] pu",
            ))

    def add_tap(table: str, index: int, actuator_class: str, group: str) -> None:
        rec = model[table][str(index)]
        if not bool(rec.get("in_service", True)):
            return
        position = int(round(float(rec.get("tap_pos", 0.0))))
        lower = int(round(float(rec["tap_min"])))
        upper = int(round(float(rec["tap_max"])))
        pf_name = names[(table, index)]
        for direction in (-1, 1):
            if not lower <= position + direction <= upper:
                continue
            cases.append(Candidate(
                name=f"tap_{_token(direction, 0)}_{pf_name}",
                actuator_class=actuator_class,
                domain="DSO" if table == "trafo3w" else "TSO", group=group,
                mode="discrete", kind="tap", table=table, index=index,
                pf_name=pf_name, direction=direction, variable="ntapcmd",
                delta=float(direction), physical_delta=float(direction), unit="tap",
                event_offset_s=float(oltc_delay_s),
                note=(f"one tap; delay={oltc_delay_s:g} s; snapshot range "
                      f"[{lower}, {upper}]"),
            ))

    for raw in inventory["coupler_oltc_trafo3w_indices"]:
        idx = int(raw)
        add_tap("trafo3w", idx, "coupler_oltc", tr3_owner.get(idx, "unassigned"))

    slack_gens = {
        int(i) for i, rec in model["gen"].items() if bool(rec.get("slack", False))
    }
    gen_for_trafo = {
        int(t): int(g) for t, g in zip(
            doc["meta"]["machine_trafo_indices"],
            doc["meta"]["machine_trafo_gen_map"],
        )
    }
    tso_taps = (
        list(inventory["machine_oltc_trafo_indices"])
        + list(inventory["network_oltc_trafo_indices"])
    )
    for raw in tso_taps:
        idx = int(raw)
        if gen_for_trafo.get(idx) in slack_gens:
            continue
        bus = int(model["trafo"][str(idx)]["hv_bus"])
        add_tap("trafo", idx, "tso_oltc", zone_of.get(bus, "unassigned"))

    shunt_kind = {
        int(i): str(k) for i, k in zip(
            inventory["tso_tertiary_shunt_indices"],
            inventory["tso_tertiary_shunt_kinds"],
        )
    }
    shunt_zone = {
        int(i): f"Z{z}" for i, z in zip(
            doc["meta"]["tso_tertiary_shunt_indices"],
            doc["meta"]["tso_tertiary_shunt_zones"],
        )
    }
    for raw in inventory["tso_tertiary_shunt_indices"]:
        idx = int(raw)
        rec = model["shunt"][str(idx)]
        if not bool(rec.get("in_service", True)):
            continue
        position, upper = int(rec.get("step", 0)), int(rec["max_step"])
        pf_name = names[("shunt", idx)]
        for direction in (-1, 1):
            if not 0 <= position + direction <= upper:
                continue
            cases.append(Candidate(
                name=f"shunt_{_token(direction, 0)}_{pf_name}",
                actuator_class="shunt", domain="TSO",
                group=shunt_zone.get(idx, "unassigned"), mode="discrete",
                kind="tap", table="shunt", index=idx, pf_name=pf_name,
                direction=direction, variable="ncapa", delta=float(direction),
                physical_delta=float(direction), unit="step", event_offset_s=0.0,
                note=f"direct {shunt_kind.get(idx, 'shunt')} breaker step",
            ))

    if len({c.name for c in cases}) != len(cases):
        raise ValueError("candidate names are not unique")
    return sorted(cases, key=lambda c: c.name)


def build_output_specs(doc: Mapping[str, Any]) -> List[OutputSpec]:
    """Nodal voltage and boundary/interface Q output vector; no current rows."""
    model, names = doc["model"], build_name_map(doc)
    specs: List[OutputSpec] = []
    for raw in doc["meta"]["tn_bus_indices"]:
        idx = int(raw)
        if str(idx) not in model["bus"] or not model["bus"][str(idx)].get("in_service", True):
            continue
        pf_name = names[("bus", idx)]
        if pf_name.startswith("TN_bus"):
            specs.append(OutputSpec(
                "voltage", "bus", idx, "ElmTerm", pf_name, "m:u",
                f"V_{pf_name}", BAND_VOLTAGE_PU,
            ))
    for hv in doc["meta"]["hv_networks"]:
        for raw in hv["bus_indices"]:
            idx = int(raw)
            pf_name = names[("bus", idx)]
            specs.append(OutputSpec(
                "voltage", "bus", idx, "ElmTerm", pf_name, "m:u",
                f"V_{pf_name}", BAND_VOLTAGE_PU,
            ))
    for raw in doc["actuators"]["coupler_oltc_trafo3w_indices"]:
        idx = int(raw)
        pf_name = names[("trafo3w", idx)]
        specs.append(OutputSpec(
            "dso_interface_q", "trafo3w", idx, "ElmTr3", pf_name,
            "m:Q:bushv", f"Q_DSO_{pf_name}", BAND_Q_MVAR,
        ))
    zone_of = _zone_lookup(doc)
    for raw in doc["meta"]["tn_line_indices"]:
        idx = int(raw)
        rec = model["line"][str(idx)]
        z1, z2 = zone_of.get(int(rec["from_bus"])), zone_of.get(int(rec["to_bus"]))
        if z1 is None or z2 is None or z1 == z2:
            continue
        pf_name = names[("line", idx)]
        specs.append(OutputSpec(
            "tso_boundary_q", "line", idx, "ElmLne", pf_name,
            "m:Q:bus1", f"Q_TSO_{z1}_{z2}_{pf_name}", BAND_Q_MVAR,
        ))
    unique: Dict[Tuple[str, int, str], OutputSpec] = {}
    for spec in specs:
        unique.setdefault((spec.table, spec.index, spec.variable), spec)
    result = sorted(unique.values(), key=lambda s: (s.kind, s.index, s.label))
    if any("m:I" in s.variable or "current" in s.kind for s in result):
        raise AssertionError("current monitor leaked into output vector")
    return result


def _qss_values(net: Any, specs: Sequence[OutputSpec]) -> np.ndarray:
    values: List[float] = []
    for spec in specs:
        if spec.kind == "voltage":
            value = net.res_bus.at[spec.index, "vm_pu"]
        elif spec.kind == "dso_interface_q":
            value = net.res_trafo3w.at[spec.index, "q_hv_mvar"]
        elif spec.kind == "tso_boundary_q":
            value = net.res_line.at[spec.index, "q_from_mvar"]
        else:
            raise ValueError(f"unknown QSS output kind {spec.kind!r}")
        values.append(float(value))
    return np.asarray(values, dtype=float)


def screen_continuous_candidates(
    snapshot: Path, candidates: Sequence[Candidate], specs: Sequence[OutputSpec]
) -> None:
    """Deterministic amplitude proxy; failed cases are retained for RMS."""
    import pandapower as pp

    net, doc = load_snapshot_to_pandapower(snapshot)
    solver = dict(doc["solver_options"])
    pp.runpp(net, **solver)
    base = _qss_values(net, specs)
    tolerances = np.asarray([s.tolerance for s in specs], dtype=float)
    warm_solver = dict(solver)
    warm_solver["init"] = "results"
    continuous = [c for c in candidates if c.mode == "continuous"]
    print(f"[qss] ranking {len(continuous)} feasible continuous commands")
    for number, case in enumerate(continuous, 1):
        column = "q_mvar" if case.table == "sgen" else "vm_pu"
        table = net.sgen if case.table == "sgen" else net.gen
        old = float(table.at[case.index, column])
        increment = case.physical_delta if case.table == "sgen" else case.delta
        table.at[case.index, column] = old + increment
        try:
            pp.runpp(net, **warm_solver)
            delta = _qss_values(net, specs) - base
            normalized = np.abs(delta) / tolerances
            worst = int(np.argmax(normalized))
            case.qss_score = float(normalized[worst])
            case.qss_worst_output = specs[worst].label
            case.qss_worst_delta = float(delta[worst])
        except Exception as exc:
            case.qss_score = math.inf
            case.qss_worst_output = "QSS_FAILED"
            case.qss_error = f"{type(exc).__name__}: {exc}"
        finally:
            table.at[case.index, column] = old
        if number % 20 == 0 or number == len(continuous):
            print(f"[qss] {number}/{len(continuous)}")


def select_candidates(
    candidates: Sequence[Candidate], *, top_global: int, top_per_group: int,
    median_audit: bool, all_continuous: bool,
) -> List[Candidate]:
    """Exhaustive discrete selection plus deterministic continuous strata."""
    for case in candidates:
        case.selected = case.mode == "discrete"
        case.selection_reasons.clear()
        if case.selected:
            case.selection_reasons.append("all feasible discrete locations")
    continuous = [c for c in candidates if c.mode == "continuous"]
    if all_continuous:
        for case in continuous:
            case.selected = True
            case.selection_reasons.append("all-continuous override")
        return [c for c in candidates if c.selected]

    def mark(case: Candidate, reason: str) -> None:
        case.selected = True
        if reason not in case.selection_reasons:
            case.selection_reasons.append(reason)

    for actuator_class, direction in sorted(
        {(c.actuator_class, c.direction) for c in continuous}
    ):
        pool = [c for c in continuous
                if c.actuator_class == actuator_class and c.direction == direction]
        failed = [c for c in pool
                  if c.qss_score is None or not math.isfinite(c.qss_score)]
        finite = sorted(
            (c for c in pool if c.qss_score is not None and math.isfinite(c.qss_score)),
            key=lambda c: (-float(c.qss_score), c.name),
        )
        for case in failed:
            mark(case, "QSS failed: retained conservatively")
        for case in finite[:max(0, int(top_global))]:
            mark(case, f"top {top_global} global in class/direction")
        for group in sorted({c.group for c in finite}):
            members = [c for c in finite if c.group == group]
            for case in members[:max(0, int(top_per_group))]:
                mark(case, f"top {top_per_group} in {group}")
        if median_audit and finite:
            ascending = sorted(finite, key=lambda c: (float(c.qss_score), c.name))
            mark(ascending[len(ascending) // 2], "median QSS audit")
    return [c for c in candidates if c.selected]


def fixed_band_metrics(
    t: Sequence[float], y: Sequence[float], t_event: float, tolerance: float,
    *, final_window_s: float = FINAL_WINDOW_S,
    confirm_window_s: float = CONFIRM_WINDOW_S,
) -> Dict[str, Any]:
    """Fixed physical band, final-window mean, and 5 s confirmation."""
    ta, ya = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if ta.size != ya.size or ta.size < 2:
        raise ValueError("settling metric needs equal t/y arrays with >=2 samples")
    before = (ta < t_event) & (ta >= t_event - 1.0)
    if before.any():
        initial = float(np.mean(ya[before]))
    else:
        initial = float(ya[max(int(np.searchsorted(ta, t_event)) - 1, 0)])
    tail = ta >= ta[-1] - float(final_window_s)
    final = float(np.mean(ya[tail]))
    after = ta >= t_event
    if not after.any():
        raise ValueError("trajectory ends before the event")
    outside = after & (np.abs(ya - final) > float(tolerance))
    if outside.any():
        last_outside = float(ta[outside][-1])
        settling = max(last_outside - float(t_event), 0.0)
        confirmation = float(ta[-1]) - last_outside
    else:
        last_outside, settling = float(t_event), 0.0
        confirmation = float(ta[-1]) - float(t_event)
    tail_values = ya[tail]
    split = max(1, tail_values.size // 2)
    tail_drift = abs(float(np.mean(tail_values[-split:]))
                     - float(np.mean(tail_values[:split])))
    excursion = float(np.max(np.abs(ya[after] - initial)))
    censored = confirmation + 1.0e-9 < confirm_window_s or tail_drift > tolerance
    return {
        "y_initial": initial, "y_final": final, "signed_step": final - initial,
        "excursion": excursion, "t_settle_s": settling,
        "last_outside_s": last_outside, "confirmation_s": confirmation,
        "tail_drift": tail_drift, "tolerance": float(tolerance),
        "subband": bool(excursion <= tolerance), "censored": bool(censored),
    }


def reduce_worst(signal_rows: Sequence[Mapping[str, Any]],
                 kinds: Sequence[str]) -> Mapping[str, Any]:
    """The worst of ``signal_rows`` restricted to the given ``output_kind``s.

    Ordered by (censored, settling time, excursion in bands), so a censored row
    outranks any settled one and the band-normalised excursion breaks ties
    between rows that never left their band.  Raises rather than returning a
    default when ``kinds`` selects nothing: an empty reduction would otherwise
    report a settling time of zero for a case that was never evaluated.

    Flags are read with :func:`_truth`, not ``bool``, because ``signal_rows``
    is either live metric dicts (real booleans) or rows re-read from
    ``signal_metrics.csv`` (the strings ``"True"`` / ``"False"``), and
    ``bool("False")`` is ``True`` -- which marks every archived row censored.
    """
    rows = [row for row in signal_rows if row["output_kind"] in kinds]
    if not rows:
        raise ValueError(
            f"no measured output has kind in {tuple(kinds)!r}; "
            f"present: {sorted({r['output_kind'] for r in signal_rows})}")
    return max(rows, key=lambda row: (
        _truth(row["censored"]), float(row["t_settle_s"]),
        float(row["excursion"]) / float(row["tolerance"])))


def _family_columns(
    signal_rows: Sequence[Mapping[str, Any]],
    families: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    """One ``worst_*`` column group per output family, suffixed by its key.

    ``{"": V_OUTPUT_KINDS, "_q": Q_OUTPUT_KINDS}`` yields ``worst_signal`` /
    ``t_settle_s`` / ... for the voltages and ``worst_signal_q`` /
    ``t_settle_q_s`` / ... for the Q flows.  A family that selects no measured
    output yields ``None`` for its columns rather than raising: the per-layer Q
    split is optional detail, and a network without one of the two Q kinds must
    still produce a row.  The two families the table is built from are asserted
    non-empty by the caller, not here.
    """
    columns: Dict[str, Any] = {}
    for suffix, kinds in families.items():
        rows = [r for r in signal_rows if r["output_kind"] in kinds]
        stem = f"t_settle{suffix}_s" if suffix else "t_settle_s"
        if not rows:
            columns.update({
                f"worst_output_kinds{suffix}": "|".join(kinds),
                f"worst_signal{suffix}": None,
                f"worst_output_kind{suffix}": None,
                stem: None,
                f"censored{suffix}": None, f"subband{suffix}": None,
            })
            continue
        worst = reduce_worst(rows, kinds)
        columns.update({
            f"worst_output_kinds{suffix}": "|".join(kinds),
            f"worst_signal{suffix}": worst["signal"],
            f"worst_output_kind{suffix}": worst["output_kind"],
            stem: float(worst["t_settle_s"]),
            f"censored{suffix}": any(_truth(r["censored"]) for r in rows),
            f"subband{suffix}": all(_truth(r["subband"]) for r in rows),
        })
    return columns


def _read_scalar(obj: Any, variable: str) -> float:
    for attr in (variable, f"s:{variable}", f"c:{variable}"):
        try:
            return float(obj.GetAttribute(attr))
        except Exception:
            continue
    raise PFSessionError(
        f"cannot read {variable!r} on {getattr(obj, 'loc_name', obj)!r}"
    )


def _by_name(app: Any, class_name: str) -> Dict[str, Any]:
    from pf.session import get_all

    result: Dict[str, Any] = {}
    for obj in get_all(app, class_name):
        name = str(obj.loc_name)
        if name in result:
            raise PFSessionError(f"duplicate {class_name} name {name!r}")
        result[name] = obj
    return result


def resolve_output_monitors(
    app: Any, specs: Sequence[OutputSpec]
) -> List[Tuple[Any, str, str]]:
    classes = {s.pf_class for s in specs}
    objects = {class_name: _by_name(app, class_name) for class_name in classes}
    result: List[Tuple[Any, str, str]] = []
    for spec in specs:
        obj = objects[spec.pf_class].get(spec.pf_name)
        if obj is None:
            raise PFSessionError(
                f"controlled output {spec.label}: {spec.pf_name}.{spec.pf_class} missing"
            )
        result.append((obj, spec.variable, spec.label))
    return result


def resolve_live_case(app: Any, case: Candidate) -> ResolvedCase:
    from pf.screening import _machine_avrs, _qvpre_of
    from pf.tap_ctrl import tapctrl_of

    if case.table == "sgen":
        park = _by_name(app, "ElmGenstat").get(case.pf_name)
        if park is None:
            raise PFSessionError(f"{case.name}: park not found")
        pre = _qvpre_of(app, case.pf_name)
        if pre is None:
            raise PFSessionError(f"{case.name}: QVPRE write handle missing")
        return ResolvedCase(
            case, park, pre, pre, "s:qset", f"diag_qset_{case.pf_name}"
        )
    if case.table == "gen":
        match = next(
            ((sym, avr) for sym, avr in _machine_avrs(app).items()
             if str(sym.loc_name) == case.pf_name), None,
        )
        if match is None:
            raise PFSessionError(f"{case.name}: AVR not found")
        sym, avr = match
        return ResolvedCase(
            case, sym, avr, avr, "s:usetp",
            f"diag_vref_{case.pf_name.replace(' ', '')}",
        )
    if case.table == "trafo3w":
        target = _by_name(app, "ElmTr3").get(case.pf_name)
        variable = "c:n3tap_h"
    elif case.table == "trafo":
        target = _by_name(app, "ElmTr2").get(case.pf_name)
        variable = "c:nntap"
    elif case.table == "shunt":
        target = _by_name(app, "ElmShnt").get(case.pf_name)
        if target is None:
            raise PFSessionError(f"{case.name}: shunt not found")
        return ResolvedCase(
            case, target, target, target, "c:ncapa",
            f"diag_step_{case.pf_name}", None,
        )
    else:
        raise PFSessionError(f"{case.name}: unsupported table {case.table!r}")
    if target is None:
        raise PFSessionError(f"{case.name}: transformer not found")
    ctrl = tapctrl_of(app, target)
    if ctrl is None:
        raise PFSessionError(
            f"{case.name}: TAPCTRL missing; delay/PT1 experiment is undefined"
        )
    return ResolvedCase(
        case, target, ctrl, target, variable, f"diag_tap_{case.pf_name}", ctrl
    )


def _tap_tmech(ctrl: Any) -> float:
    params = list(ctrl.GetAttribute("params") or [])
    if len(params) < 2:
        raise PFSessionError(f"{ctrl.loc_name}: incomplete TAPCTRL params")
    return float(params[1])


def _set_tap_tmech(ctrl: Any, value: float) -> None:
    ctrl.SetAttribute("params:1", float(value))


def read_monitors_bulk(
    ctx: Any, monitors: Sequence[Tuple[Any, str, str]], *, stride: int = READ_STRIDE
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Read the PF time column once rather than once per monitored signal."""
    ctx.res.Load()
    count = int(ctx.res.GetNumberOfRows())
    if count < 2:
        raise PFSessionError(f"RMS result has only {count} row(s)")
    rows = list(range(0, count, max(1, int(stride))))
    if rows[-1] != count - 1:
        rows.append(count - 1)
    t = np.asarray([ctx.res.GetValue(i, -1)[1] for i in rows], dtype=float)
    values: Dict[str, np.ndarray] = {}
    for obj, variable, label in monitors:
        column = int(ctx.res.FindColumn(obj, variable))
        if column < 0:
            raise PFSessionError(
                f"{variable!r} not recorded on {getattr(obj, 'loc_name', obj)!r}"
            )
        values[label] = np.asarray(
            [ctx.res.GetValue(i, column)[1] for i in rows], dtype=float
        )
    return t, values


def _mean_near(t: np.ndarray, y: np.ndarray, start: float, stop: float) -> float:
    mask = (t >= start) & (t < stop)
    if mask.any():
        return float(np.mean(y[mask]))
    index = min(max(int(np.searchsorted(t, stop)) - 1, 0), len(t) - 1)
    return float(y[index])


def _event_delta(resolved: ResolvedCase) -> Tuple[float, float]:
    case = resolved.candidate
    if case.table == "sgen":
        params = list(resolved.event_target.GetAttribute("params") or [])
        if len(params) < 7:
            raise PFSessionError(f"{case.name}: incomplete QVPRE params")
        current = _read_scalar(resolved.event_target, "qset")
        rail = float(params[6] if case.direction > 0 else params[5])
        magnitude = min(abs(case.delta), max((rail - current) * case.direction, 0.0))
        if magnitude <= 1.0e-9:
            raise PFSessionError(f"{case.name}: no live Q headroom")
        delta = case.direction * magnitude
        return delta, delta * float(resolved.target.GetAttribute("sgn"))
    if case.table == "gen":
        current = _read_scalar(resolved.event_target, "usetp")
        rail = AVR_MAX_PU if case.direction > 0 else AVR_MIN_PU
        magnitude = min(abs(case.delta), max((rail - current) * case.direction, 0.0))
        if magnitude <= 1.0e-9:
            raise PFSessionError(f"{case.name}: no live AVR headroom")
        delta = case.direction * magnitude
        return delta, delta
    return float(case.direction), float(case.direction)


def _arm_event(
    ctx: Any, resolved: ResolvedCase, event_time: float, event_delta: float
) -> None:
    case = resolved.candidate
    if case.table in ("sgen", "gen"):
        current = _read_scalar(resolved.event_target, case.variable)
        ctx.add_param_event(
            resolved.event_target, case.variable, current + event_delta, event_time
        )
    elif case.table in ("trafo", "trafo3w"):
        base = _read_scalar(resolved.tap_ctrl, "ntapcmd")
        ctx.add_param_event(
            resolved.tap_ctrl, "ntapcmd", base + event_delta, event_time
        )
    elif case.table == "shunt":
        ctx.add_tap_event(resolved.target, int(event_delta), event_time, seq=0)
    else:
        raise PFSessionError(f"{case.name}: cannot arm event")


def _diagnostic_liveness(
    resolved: ResolvedCase, t: np.ndarray, y: np.ndarray,
    event_time: float, expected: float,
) -> Dict[str, Any]:
    initial = _mean_near(t, y, event_time - 0.5, event_time)
    final = _mean_near(t, y, float(t[-1]) - 0.5, float(t[-1]) + 1.0e-9)
    observed = final - initial
    if resolved.candidate.mode == "discrete":
        tolerance = 0.5
        live = observed * expected > 0.0 and abs(observed) >= tolerance
    else:
        tolerance = max(0.10 * abs(expected), 1.0e-5)
        live = abs(observed - expected) <= tolerance
    return {
        "event_live": bool(live), "diagnostic_initial": initial,
        "diagnostic_final": final, "diagnostic_change": observed,
        "diagnostic_expected_change": expected,
        "diagnostic_tolerance": tolerance,
    }


def run_case_once(
    ctx: Any, resolved: ResolvedCase,
    controlled: Sequence[Tuple[Any, str, str]],
    spec_by_label: Mapping[str, OutputSpec], *, pre_settle_s: float,
    t_event_s: float, horizon_s: float, oltc_tmech_s: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray, Dict[str, np.ndarray]]:
    """Run one independent RMS attempt and prove that its event fired."""
    case = resolved.candidate
    diagnostic = (
        resolved.diagnostic_obj, resolved.diagnostic_variable,
        resolved.diagnostic_label,
    )
    monitors = list(controlled)
    if not any(obj is diagnostic[0] and variable == diagnostic[1]
               for obj, variable, _ in monitors):
        monitors.append(diagnostic)
    original_tmech: Optional[float] = None
    try:
        ctx.purge_events()
        if resolved.tap_ctrl is not None:
            original_tmech = _tap_tmech(resolved.tap_ctrl)
            _set_tap_tmech(resolved.tap_ctrl, oltc_tmech_s)
        ctx.set_monitors(monitors)
        ctx.initialise()
        if pre_settle_s > 0.0:
            ctx.simulate(float(pre_settle_s))
        dispatch_time = float(pre_settle_s) + float(t_event_s)
        physical_event_time = dispatch_time + float(case.event_offset_s)
        event_delta, physical_delta = _event_delta(resolved)
        _arm_event(ctx, resolved, physical_event_time, event_delta)
        admission_horizon = physical_event_time - float(pre_settle_s)
        if admission_horizon <= 0.0:
            raise ValueError("event must lie after the pre-settle clock")
        ctx.admit_new_events(float(pre_settle_s), admission_horizon)
        ctx.simulate(dispatch_time + float(horizon_s))
        t, series = read_monitors_bulk(ctx, monitors)
        diag = _diagnostic_liveness(
            resolved, t, series[resolved.diagnostic_label],
            physical_event_time, event_delta,
        )
        if not diag["event_live"]:
            raise PFSessionError(
                f"{case.name}: actuator diagnostic changed "
                f"{diag['diagnostic_change']:.6g}; expected "
                f"{diag['diagnostic_expected_change']:.6g}"
            )
        signal_rows: List[Dict[str, Any]] = []
        for _obj, variable, label in controlled:
            spec = spec_by_label[label]
            metric = fixed_band_metrics(t, series[label], dispatch_time, spec.tolerance)
            signal_rows.append({
                "case": case.name, "signal": label, "variable": variable,
                "output_kind": spec.kind, **metric,
            })
        family = {
            "": V_OUTPUT_KINDS, "_q": Q_OUTPUT_KINDS,
            "_dso_q": DSO_Q_OUTPUT_KINDS, "_tso_q": TSO_Q_OUTPUT_KINDS,
        }
        result = {
            "case": case.name, "actuator_class": case.actuator_class,
            "domain": case.domain, "group": case.group,
            "direction": case.direction, "mode": case.mode,
            "target": case.pf_name, "requested_delta": case.physical_delta,
            "actual_delta": physical_delta, "unit": case.unit,
            "dispatch_time_s": dispatch_time,
            "physical_event_time_s": physical_event_time,
            "event_offset_s": case.event_offset_s, "horizon_s": float(horizon_s),
            # One reduction per output family, never pooled into a single
            # headline: "" is the voltage worst, "_q" the interface-Q worst,
            # and the two _dso_q / _tso_q columns split Q by the layer that
            # tracks it.  See V_OUTPUT_KINDS.
            **_family_columns(signal_rows, family),
            "qss_score": case.qss_score,
            "qss_worst_output": case.qss_worst_output,
            "selection_reasons": "; ".join(case.selection_reasons), **diag,
        }
        return result, signal_rows, t, series
    finally:
        if original_tmech is not None:
            ctx.app.ResetCalculation()
            ctx._calculation_active = False
            _set_tap_tmech(resolved.tap_ctrl, original_tmech)


def run_case_adaptive(
    ctx: Any, resolved: ResolvedCase,
    controlled: Sequence[Tuple[Any, str, str]],
    spec_by_label: Mapping[str, OutputSpec], *, pre_settle_s: float,
    t_event_s: float, initial_horizon_s: float, max_horizon_s: float,
    oltc_tmech_s: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray, Dict[str, np.ndarray]]:
    horizon, attempted = float(initial_horizon_s), []
    while True:
        attempted.append(horizon)
        result, signals, t, series = run_case_once(
            ctx, resolved, controlled, spec_by_label,
            pre_settle_s=pre_settle_s, t_event_s=t_event_s,
            horizon_s=horizon, oltc_tmech_s=oltc_tmech_s,
        )
        if not result["censored"] or horizon >= max_horizon_s - 1.0e-9:
            result["attempted_horizons_s"] = ";".join(f"{x:g}" for x in attempted)
            return result, signals, t, series
        next_horizon = min(float(max_horizon_s), 2.0 * horizon)
        print(f"    censored at {horizon:g} s; rerun at {next_horizon:g} s")
        horizon = next_horizon


def run_preflight(
    ctx: Any, monitors: Sequence[Tuple[Any, str, str]],
    spec_by_label: Mapping[str, OutputSpec], *, pre_settle_s: float,
    window_s: float,
) -> Dict[str, Any]:
    """Flat-run output drift and monitor validity gate; no event exists."""
    ctx.purge_events()
    ctx.set_monitors(list(monitors))
    ctx.initialise()
    ctx.simulate(float(pre_settle_s) + float(window_s))
    t, series = read_monitors_bulk(ctx, monitors)
    rows: List[Dict[str, Any]] = []
    for _obj, variable, label in monitors:
        spec = spec_by_label[label]
        values = series[label][t >= float(pre_settle_s)]
        if values.size == 0:
            raise PFSessionError(f"preflight has no samples for {label}")
        drift = float(np.max(values) - np.min(values))
        limit = PREFLIGHT_V_DRIFT_PU if spec.kind == "voltage" else PREFLIGHT_Q_DRIFT_MVAR
        rows.append({
            "signal": label, "variable": variable, "kind": spec.kind,
            "peak_to_peak": drift, "limit": limit, "ratio": drift / limit,
        })
    worst = max(rows, key=lambda row: float(row["ratio"]))
    passed = float(worst["ratio"]) < 1.0
    print(
        f"[preflight] worst drift ratio {worst['ratio']:.3g}: "
        f"{worst['signal']} ({worst['peak_to_peak']:.3g}/"
        f"{worst['limit']:.3g})"
    )
    report = {
        "pre_settle_s": float(pre_settle_s), "window_s": float(window_s),
        "worst_signal": worst["signal"], "worst_ratio": worst["ratio"],
        "passed": passed, "signals": rows,
    }
    if not passed:
        raise PFSessionError(
            f"flat RMS plant is not settled: {worst['signal']} drift "
            f"{worst['peak_to_peak']:.6g} >= {worst['limit']:.6g}"
        )
    return report


def _git(*args: str) -> Optional[str]:
    executable = shutil.which("git") or r"C:\Program Files\Git\cmd\git.exe"
    try:
        result = subprocess.run(
            (executable, *args), cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _field_order(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    fields = sorted({key for row in rows for key in row})
    return (["case"] if "case" in fields else []) + [f for f in fields if f != "case"]


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = _field_order(rows)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _candidate_rows(candidates: Sequence[Candidate], scheduled: Sequence[Candidate]) -> List[Dict[str, Any]]:
    scheduled_names = {c.name for c in scheduled}
    rows: List[Dict[str, Any]] = []
    for case in candidates:
        row = asdict(case)
        row["selection_reasons"] = "; ".join(case.selection_reasons)
        row["disturbance"] = False
        row["scheduled"] = case.name in scheduled_names
        rows.append(row)
    return rows


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truth(value: Any) -> bool:
    return value is True or str(value).strip().lower() in ("1", "true", "yes")


def _case_order(cases: Sequence[Candidate]) -> List[Candidate]:
    rank = {"coupler_oltc": 0, "tso_oltc": 1, "shunt": 2}
    return sorted(
        cases,
        key=lambda c: (
            0 if c.mode == "discrete" else 1,
            rank.get(c.actuator_class, 9),
            -float(c.qss_score) if c.qss_score is not None and math.isfinite(c.qss_score) else -math.inf,
            c.name,
        ),
    )


def _worst_row(rows: Sequence[Mapping[str, Any]], suffix: str) -> Mapping[str, Any]:
    """The result row holding the worst settling of one output family.

    Same ordering as :func:`reduce_worst` minus the excursion tie-break, which
    is not carried up into ``results.csv``.  Rows whose family column is empty
    (a resumed run written before that family existed) sort last instead of
    raising.
    """
    stem = f"t_settle{suffix}_s" if suffix else "t_settle_s"
    return max(rows, key=lambda r: (
        r.get(stem) not in (None, ""),
        _truth(r.get(f"censored{suffix}")),
        _as_float(r.get(stem), float("-inf")),
    ))


def _settle_cell(row: Mapping[str, Any], suffix: str) -> str:
    """``12.472``, ``12.472 (c)`` if censored, ``0.072 (sb)`` if sub-band."""
    stem = f"t_settle{suffix}_s" if suffix else "t_settle_s"
    if row.get(stem) in (None, ""):
        return "n/a"
    text = f"{_as_float(row.get(stem)):.3f}"
    if _truth(row.get(f"censored{suffix}")):
        return text + " (c)"
    if _truth(row.get(f"subband{suffix}")):
        return text + " (sb)"
    return text


_KIND_PHRASES = {
    "voltage": "nodal voltages",
    "dso_interface_q": "DSO interface-Q flows",
    "tso_boundary_q": "TSO boundary-Q flows",
}


def _kinds_phrase(kinds: Sequence[str]) -> str:
    """``("voltage",)`` -> ``"nodal voltages"``, for the summary prose."""
    names = [_KIND_PHRASES.get(k, k) for k in kinds]
    if len(names) <= 1:
        return names[0] if names else "no outputs"
    return ", ".join(names[:-1]) + " and " + names[-1]


def write_summary(
    out_dir: Path, scheduled: Sequence[Candidate], outputs: Sequence[OutputSpec],
    results: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]],
    *, status: str, args: argparse.Namespace,
) -> None:
    complete_names = {str(row.get("case")) for row in results}
    discrete = [c for c in scheduled if c.mode == "discrete"]
    continuous = [c for c in scheduled if c.mode == "continuous"]
    lines = [
        "# Actuator-only settling-time location sweep", "",
        f"Status: **{status}** ({len(complete_names)}/{len(scheduled)} cases complete).",
        "", "## Design", "",
        f"- {len(discrete)} feasible discrete steps are exhaustive in RMS.",
        f"- {len(continuous)} continuous cases are RMS-verified after deterministic QSS screening.",
        f"- Controlled outputs: {sum(s.kind == 'voltage' for s in outputs)} nodal voltages, "
        f"{sum(s.kind == 'dso_interface_q' for s in outputs)} DSO interface-Q flows, and "
        f"{sum(s.kind == 'tso_boundary_q' for s in outputs)} TSO boundary-Q flows.",
        "- Currents and disturbances are excluded.",
        f"- Fixed bands: {BAND_VOLTAGE_PU:g} pu and {BAND_Q_MVAR:g} Mvar; "
        f"confirmation window {CONFIRM_WINDOW_S:g} s.",
        f"- OLTC: {args.oltc_delay_s:g} s pure delay followed by "
        f"Tmech={args.oltc_tmech_s:g} s; settling starts at dispatch.",
        f"- Each case pre-settles for {args.pre_settle_s:g} s and starts at "
        f"{args.initial_horizon_s:g} s, extending to {args.max_horizon_s:g} s if censored.",
        f"- Worst settling is reduced over {_kinds_phrase(V_OUTPUT_KINDS)} and "
        f"over {_kinds_phrase(Q_OUTPUT_KINDS)} SEPARATELY, against their own "
        f"bands ({BAND_VOLTAGE_PU:g} pu, {BAND_Q_MVAR:g} Mvar). The two are not "
        f"pooled into one number: `t_settle_s` is the voltage worst and "
        f"`t_settle_q_s` the interface-Q worst, and `results.csv` additionally "
        f"splits Q by tracking layer (`t_settle_dso_q_s`, `t_settle_tso_q_s`).",
        "", "The claim supported by this run is the worst RMS settling time, "
        "reported per output family, among all feasible discrete locations and "
        "the deterministically screened continuous locations. It is not a "
        "mathematical proof over untested continuous locations.",
    ]
    if results:
        lines += ["", "## Current worst by actuator class and direction", "",
                  "Worst nodal voltage and worst interface Q are reduced "
                  "independently, so the two halves of a row may be different "
                  "cases. `(c)` = censored, `(sb)` = never left its band.", "",
                  "| Class | Dir | Case (worst V) | V [s] | V output "
                  "| Case (worst Q) | Q [s] | Q output |",
                  "|---|---:|---|---:|---|---|---:|---|"]
        keys = sorted({(str(r.get("actuator_class")), str(r.get("direction"))) for r in results})
        for key in keys:
            rows = [r for r in results
                    if (str(r.get("actuator_class")), str(r.get("direction"))) == key]
            v, q = _worst_row(rows, ""), _worst_row(rows, "_q")
            lines.append(
                f"| {key[0]} | {key[1]} "
                f"| `{v.get('case')}` | {_settle_cell(v, '')} | {v.get('worst_signal')} "
                f"| `{q.get('case')}` | {_settle_cell(q, '_q')} | {q.get('worst_signal_q')} |"
            )
        v, q = _worst_row(results, ""), _worst_row(results, "_q")
        lines += ["",
                  f"Current global worst {_kinds_phrase(V_OUTPUT_KINDS)}: "
                  f"`{v.get('case')}`, {_settle_cell(v, '')} s at "
                  f"`{v.get('worst_signal')}`.",
                  f"Current global worst {_kinds_phrase(Q_OUTPUT_KINDS)}: "
                  f"`{q.get('case')}`, {_settle_cell(q, '_q')} s at "
                  f"`{q.get('worst_signal_q')}`."]
    if failures:
        lines += ["", "## Failures", ""]
        for failure in failures:
            lines.append(f"- `{failure.get('case')}`: {failure.get('error')}")
    pending = [c.name for c in scheduled if c.name not in complete_names]
    if pending:
        lines += ["", "## Pending", "", *[f"- `{name}`" for name in pending]]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_worst_trajectory(
    out_dir: Path, result: Mapping[str, Any], t: np.ndarray,
    series: Mapping[str, np.ndarray],
) -> None:
    label = str(result["worst_signal"])
    path = out_dir / f"traj_worst_{result['case']}.csv"
    rows = ({"t_s": float(tt), "signal": label, "value": float(yy)}
            for tt, yy in zip(t, series[label]))
    _atomic_csv(path, list(rows))


def _provenance(args: argparse.Namespace, snapshot: Path) -> Dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "argv": sys.argv, "args": vars(args),
        "started": datetime.now().astimezone().isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "cwd": os.getcwd(),
        "snapshot": str(snapshot.resolve()), "snapshot_sha256": _sha256(snapshot),
        "assumptions": {
            "disturbances": "excluded", "currents": "excluded",
            "continuous_screen": "QSS amplitude proxy plus median audit",
            "discrete_locations": "all feasible locations and directions",
            "settling_bands": {"voltage_pu": BAND_VOLTAGE_PU, "q_mvar": BAND_Q_MVAR},
            "worst_output_kinds": {
                "v": list(V_OUTPUT_KINDS), "q": list(Q_OUTPUT_KINDS),
                "dso_q": list(DSO_Q_OUTPUT_KINDS),
                "tso_q": list(TSO_Q_OUTPUT_KINDS),
            },
            "measured_output_kinds": list(ALL_OUTPUT_KINDS),
            "oltc": "pure delay plus experiment-local fast PT1",
        },
    }


def copy_provenance(out_dir: Path, snapshot: Path, meta: Dict[str, Any]) -> None:
    folder = out_dir / "provenance"
    folder.mkdir(exist_ok=True)
    paths = [Path(__file__), REPO_ROOT / "pf" / "screening.py",
             REPO_ROOT / "pf" / "tap_ctrl.py"]
    meta["source_files"] = {}
    for source in paths:
        target = folder / source.name
        shutil.copy2(source, target)
        meta["source_files"][str(source.resolve())] = _sha256(source)
    meta["source_files"][str(snapshot.resolve())] = _sha256(snapshot)


def _validate_args(args: argparse.Namespace) -> None:
    if args.pre_settle_s < 0 or args.t_event_s <= 0:
        raise ValueError("pre-settle must be >=0 and event time must be >0")
    if args.oltc_delay_s < 0 or args.oltc_tmech_s <= 0:
        raise ValueError("OLTC delay must be >=0 and Tmech must be >0")
    if args.oltc_tmech_s + 1.0e-12 < args.rms_step_ms / 1000.0:
        raise ValueError("OLTC Tmech must not be shorter than the fixed RMS step")
    if args.initial_horizon_s <= args.oltc_delay_s + CONFIRM_WINDOW_S:
        raise ValueError("initial horizon must exceed OLTC delay + confirmation window")
    if args.max_horizon_s < args.initial_horizon_s:
        raise ValueError("max horizon must be >= initial horizon")
    if args.top_global < 0 or args.top_per_group < 0:
        raise ValueError("screening counts must be non-negative")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--project", default=DEFAULT_PROJECT_PATH)
    parser.add_argument("--label", default="actuator_location_sweep_t0")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument("--pre-settle-s", type=float, default=300.0)
    parser.add_argument("--preflight-window-s", type=float, default=60.0)
    parser.add_argument("--t-event-s", type=float, default=5.0)
    parser.add_argument("--initial-horizon-s", type=float, default=30.0)
    parser.add_argument("--max-horizon-s", type=float, default=120.0)
    parser.add_argument("--rms-step-ms", type=float, default=10.0)
    parser.add_argument("--oltc-delay-s", type=float, default=DEFAULT_OLTC_DELAY_S)
    parser.add_argument("--oltc-tmech-s", type=float, default=DEFAULT_OLTC_TMECH_S)
    parser.add_argument("--top-global", type=int, default=2)
    parser.add_argument("--top-per-group", type=int, default=1)
    parser.add_argument("--no-median-audit", action="store_true")
    parser.add_argument("--all-continuous", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--save-worst-trajectories", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _prepare_design(
    args: argparse.Namespace, snapshot: Path
) -> Tuple[Mapping[str, Any], List[Candidate], List[OutputSpec], List[Candidate]]:
    doc = load_snapshot(snapshot)
    candidates = build_candidates(doc, oltc_delay_s=args.oltc_delay_s)
    outputs = build_output_specs(doc)
    screen_continuous_candidates(snapshot, candidates, outputs)
    selected = select_candidates(
        candidates, top_global=args.top_global, top_per_group=args.top_per_group,
        median_audit=not args.no_median_audit,
        all_continuous=args.all_continuous,
    )
    selected = _case_order(selected)
    if args.only:
        selected = [c for c in selected if any(token in c.name for token in args.only)]
        if not selected:
            raise ValueError(f"--only {args.only!r} matched no selected case")
    return doc, candidates, outputs, selected


def _print_design(
    candidates: Sequence[Candidate], outputs: Sequence[OutputSpec],
    selected: Sequence[Candidate],
) -> None:
    discrete_all = [c for c in candidates if c.mode == "discrete"]
    continuous_all = [c for c in candidates if c.mode == "continuous"]
    discrete_run = [c for c in selected if c.mode == "discrete"]
    continuous_run = [c for c in selected if c.mode == "continuous"]
    print(
        f"[design] candidates: {len(discrete_all)} discrete + "
        f"{len(continuous_all)} continuous; RMS: {len(discrete_run)} + "
        f"{len(continuous_run)} = {len(selected)}"
    )
    counts = {kind: sum(s.kind == kind for s in outputs)
              for kind in ("voltage", "dso_interface_q", "tso_boundary_q")}
    print(f"[design] outputs: {counts}; currents=0, disturbances=0")
    for case in selected:
        score = "n/a" if case.qss_score is None else f"{case.qss_score:.3g}"
        print(f"  {case.name}  [{case.group}; QSS={score}; "
              f"{'; '.join(case.selection_reasons)}]")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    if args.out_dir and args.resume_dir:
        raise ValueError("--out-dir and --resume-dir are mutually exclusive")
    snapshot = Path(args.snapshot).resolve()
    if not snapshot.exists():
        raise FileNotFoundError(snapshot)

    if args.dry_run:
        _doc, candidates, outputs, selected = _prepare_design(args, snapshot)
        _print_design(candidates, outputs, selected)
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.resume_dir:
        out_dir = Path(args.resume_dir).resolve()
        if not out_dir.exists():
            raise FileNotFoundError(out_dir)
        append_log = True
    elif args.out_dir:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        append_log = False
    else:
        label_dir = REPO_ROOT / "results" / "THESIS_ch9_1_timescale_settling" / args.label
        out_dir = label_dir / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        append_log = False

    tee = Tee(sys.stdout, out_dir / "run.log", append=append_log)
    sys.stdout = tee  # type: ignore[assignment]
    meta: Dict[str, Any] = {}
    scheduled: List[Candidate] = []
    outputs: List[OutputSpec] = []
    results: List[Dict[str, Any]] = []
    signal_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    try:
        print(f"\n[{datetime.now().astimezone().isoformat()}] campaign -> {out_dir}")
        _doc, candidates, outputs, scheduled = _prepare_design(args, snapshot)
        _print_design(candidates, outputs, scheduled)
        manifest_rows = _candidate_rows(candidates, scheduled)
        manifest_path = out_dir / "cases.csv"
        if args.resume_dir and manifest_path.exists():
            old_scheduled = {
                row["name"] for row in _read_csv(manifest_path) if _truth(row.get("scheduled"))
            }
            if old_scheduled != {c.name for c in scheduled}:
                raise RuntimeError("resume design differs from existing cases.csv")
        else:
            _atomic_csv(manifest_path, manifest_rows)

        meta = _provenance(args, snapshot)
        meta.update({
            "out_dir": str(out_dir), "status": "starting",
            "candidate_count": len(candidates), "scheduled_count": len(scheduled),
            "output_count": len(outputs),
        })
        copy_provenance(out_dir, snapshot, meta)
        _atomic_json(out_dir / "run_meta.json", meta)

        results = _read_csv(out_dir / "results.csv")
        signal_rows = _read_csv(out_dir / "signal_metrics.csv")
        failures = _read_csv(out_dir / "failures.csv")
        completed = {str(row.get("case")) for row in results}

        from pf.screening import RMS_STUDY_CASE, ScreeningContext
        from pf.session import connect

        app = connect(args.project, study_case=RMS_STUDY_CASE)
        ctx = ScreeningContext(
            app, rms_step_ms=args.rms_step_ms, adaptive_step=False,
            persistent_event_pool=False,
        )
        controlled = resolve_output_monitors(app, outputs)
        spec_by_label = {spec.label: spec for spec in outputs}
        if not args.skip_preflight:
            preflight = run_preflight(
                ctx, controlled, spec_by_label,
                pre_settle_s=args.pre_settle_s,
                window_s=args.preflight_window_s,
            )
            _atomic_json(out_dir / "preflight.json", preflight)
            meta["preflight"] = {
                "passed": preflight["passed"],
                "worst_signal": preflight["worst_signal"],
                "worst_ratio": preflight["worst_ratio"],
            }
        else:
            print("[preflight] SKIPPED by explicit request")
            meta["preflight"] = {"skipped": True}
        if args.preflight_only:
            meta["status"] = "preflight complete"
            meta["finished"] = datetime.now().astimezone().isoformat()
            _atomic_json(out_dir / "run_meta.json", meta)
            write_summary(
                out_dir, [], outputs, [], [], status="preflight complete", args=args
            )
            return 0

        meta["status"] = "running"
        _atomic_json(out_dir / "run_meta.json", meta)
        write_summary(
            out_dir, scheduled, outputs, results, failures, status="running", args=args
        )
        for number, case in enumerate(scheduled, 1):
            if case.name in completed:
                print(f"[{number}/{len(scheduled)}] {case.name}: already complete")
                continue
            print(f"[{number}/{len(scheduled)}] {case.name}")
            try:
                resolved = resolve_live_case(app, case)
                result, metrics, t, series = run_case_adaptive(
                    ctx, resolved, controlled, spec_by_label,
                    pre_settle_s=args.pre_settle_s,
                    t_event_s=args.t_event_s,
                    initial_horizon_s=args.initial_horizon_s,
                    max_horizon_s=args.max_horizon_s,
                    oltc_tmech_s=args.oltc_tmech_s,
                )
                results.append(result)
                signal_rows.extend(metrics)
                failures = [f for f in failures if str(f.get("case")) != case.name]
                completed.add(case.name)
                _atomic_csv(out_dir / "results.csv", results)
                _atomic_csv(out_dir / "signal_metrics.csv", signal_rows)
                _atomic_csv(out_dir / "failures.csv", failures)
                if args.save_worst_trajectories:
                    save_worst_trajectory(out_dir, result, t, series)
                flag = " CENSORED" if result["censored"] else (
                    " sub-band" if result["subband"] else ""
                )
                print(
                    f"    {result['t_settle_s']:.3f} s at "
                    f"{result['worst_signal']}; event diagnostic OK{flag}"
                )
            except Exception as exc:
                failure = {
                    "case": case.name,
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                failures = [f for f in failures if str(f.get("case")) != case.name]
                failures.append(failure)
                _atomic_csv(out_dir / "failures.csv", failures)
                print(f"    FAILED: {failure['error']}")
            meta["completed_count"] = len(completed)
            meta["failure_count"] = len(failures)
            meta["last_update"] = datetime.now().astimezone().isoformat()
            _atomic_json(out_dir / "run_meta.json", meta)
            write_summary(
                out_dir, scheduled, outputs, results, failures,
                status="running", args=args,
            )

        censored = [row for row in results if _truth(row.get("censored"))]
        missing = [case.name for case in scheduled if case.name not in completed]
        final_status = "complete" if not failures and not censored and not missing else "incomplete"
        meta.update({
            "status": final_status,
            "finished": datetime.now().astimezone().isoformat(),
            "completed_count": len(completed), "failure_count": len(failures),
            "censored_count": len(censored), "missing_count": len(missing),
        })
        _atomic_json(out_dir / "run_meta.json", meta)
        write_summary(
            out_dir, scheduled, outputs, results, failures,
            status=final_status, args=args,
        )
        if not args.out_dir and not args.resume_dir:
            label_dir = out_dir.parent
            (label_dir / "_latest.txt").write_text(out_dir.name + "\n", encoding="utf-8")
        print(f"[campaign] {final_status}; summary -> {out_dir / 'summary.md'}")
        return 0 if final_status == "complete" else 2
    except Exception:
        print("[campaign] ABORTED")
        traceback.print_exc(file=sys.stdout)
        if meta:
            meta["status"] = "aborted"
            meta["finished"] = datetime.now().astimezone().isoformat()
            meta["fatal_traceback"] = traceback.format_exc()
            _atomic_json(out_dir / "run_meta.json", meta)
        if scheduled and outputs:
            write_summary(
                out_dir, scheduled, outputs, results, failures,
                status="aborted", args=args,
            )
        raise
    finally:
        sys.stdout = tee.stream  # type: ignore[assignment]
        tee.close()


if __name__ == "__main__":
    sys.exit(main())

