#!/usr/bin/env python3
r"""Experiment F1: the anchor-referenced measurement, from the plant's own signals.

What this measures, and why it needed new instrumentation
---------------------------------------------------------
The DER Q(V) droop compares the *filtered* terminal voltage against an
**anchor** ``V_anchor``, which the OFO re-writes every time it dispatches that
park.  So the dead-zone half-width :math:`\delta` does not discriminate voltage
levels -- it discriminates **drift since the last dispatch**.  F1 therefore asks

    what is :math:`\max |V_{\mathrm{filt}} - V_{\mathrm{anchor}}|` between each
    device's *actual* anchor updates, over the operating windows,

and chooses :math:`\delta` above that maximum plus a declared margin, so the
local droop stays inert in routine operation.

Every earlier dead-band analysis in this package (``analysis.deadband_n1``,
``deadband_activity``, ``deadband_threshold``, ``deadband_2d``) works from
``csv/rms_der_raw.csv``, which carries only delivered Q (``qDER_*``) and
terminal voltage (``uDER_*``).  Neither the anchor nor the filter state is in
there, so those analyses had to *assume* the anchor-update schedule -- the
nominal 20 s / 180 s grid -- rather than read it.  Two things go wrong:

* the nominal grid is not the realised one.  Anchors move when the OFO
  actually writes a setpoint, and a park whose setpoint did not change does
  not re-anchor at all.  Measuring drift over nominal intervals both splits
  genuine drift windows and merges others.
* ``Q_actual - Q_set`` is **not** the droop signal.  The converter tracking and
  the REEC filter live in that difference, and a park pinned at a capability
  limit shows no incremental Q while its dead zone is fully active -- which is
  precisely the case F1 must not miss.

Since 2026-09-15 the RMS plant can export the block's own signals
(``--qv-internals``, see ``pf.screening.QV_INTERNAL_SIGNALS``), so this module
reads them instead of reconstructing them:

===============  ==========================================================
``vanchorDER_``  ``V_anchor``; its CHANGES are the effective update instants
``vfiltDER_``    ``x1``, the Tf-filtered terminal voltage
``veffDER_``     ``x1 - V_anchor`` -- the F1 quantity, computed in the block
``qcorrDER_``    dead-zone output; non-zero iff the dead zone acted
``qrawDER_``     ``qset - Kdroop*qcorr`` -- UNSATURATED droop command
``qextDER_``     ``clip(qraw, qmin, qmax)`` -- SATURATED reference
``qsetDER_``     the OFO setpoint the above are referenced to
===============  ==========================================================

Two modes
---------
``reference``   droop suppressed.  Reports the per-park maximum ``|veff|`` and
                the pooled maximum over parks and runs, which is the number a
                candidate half-width is chosen above.

``confirm``     a candidate width active from initialisation.  Reports
                dead-zone activation and incremental droop command, both of
                which must be zero within the declared tolerances, and lists
                **zero-headroom parks separately** -- a saturated park can show
                zero incremental Q while its dead zone is active, and counting
                it as quiet would be the one way this test lies.

Usage::

    python -m analysis.deadband_anchor --self-test
    python -m analysis.deadband_anchor --run results/rms_cosim/<stamp> --mode reference
    python -m analysis.deadband_anchor --run <dir> --mode confirm --delta 0.02

Author: Manuel Schwenke / Claude Code (2026-09-15)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: Channel file written by the RMS entry points under ``--qv-internals``.
QV_CSV = Path("csv") / "rms_qv_internals.csv"

#: Signal prefix -> short name used inside this module.
CHANNELS: Dict[str, str] = {
    "vanchorDER_": "v_anchor",
    "vfiltDER_": "v_filt",
    "veffDER_": "v_eff",
    "qcorrDER_": "q_corr",
    "qrawDER_": "q_raw",
    "qextDER_": "q_ext",
    "qsetDER_": "q_set",
}

#: Anchor movement [pu] counted as a re-anchoring rather than solver noise.
#: Two orders of magnitude below the smallest half-width the study considers,
#: so it cannot merge two genuine drift windows.
ANCHOR_EPS_PU = 1e-9

#: Dead-zone output [pu of S_n] counted as activation.  The block computes
#: ``qcorr`` arithmetically from ``abs()``, so an exactly-inside sample is
#: exactly zero and this only rejects floating-point dust.
QCORR_EPS_PU = 1e-9

#: Incremental droop command [pu of S_n] counted as action.
QCMD_EPS_PU = 1e-9

#: Directional headroom [pu of S_n] below which a park is reported as
#: zero-headroom rather than quiet.
HEADROOM_EPS_PU = 1e-6


@dataclass
class ParkDrift:
    """One park's anchor-referenced drift over one run."""

    park: str
    n_anchor_updates: int
    max_abs_veff: float
    p95_abs_veff: float
    #: Per-interval maxima, one per inter-anchor-update window.
    interval_max: List[float] = field(default_factory=list)
    #: Longest gap between effective anchor updates [s].
    max_interval_s: float = 0.0
    #: True when the anchor never moved: the whole run is ONE interval, and
    #: the drift measured is drift since initialisation.
    never_reanchored: bool = False


@dataclass
class ParkActivation:
    """One park's dead-zone and droop activity over one run."""

    park: str
    n_dead_zone_active: int
    max_abs_qcorr: float
    max_abs_incremental_unsat: float
    max_abs_incremental_sat: float
    headroom_up: float
    headroom_down: float
    zero_headroom: bool


def load_channels(run_dir: Path) -> Dict[str, pd.DataFrame]:
    """``{short name: wide frame indexed by time, one column per park}``.

    Raises rather than returning empty: a dead-band result read off a run that
    did not export the channels would be a measurement of nothing.
    """
    path = run_dir / QV_CSV
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Re-run with --qv-internals; without it the "
            f"anchor is not in the result file and F1 cannot be measured.")
    raw = pd.read_csv(path)
    missing = {c for c in ("signal", "time_s", "value") if c not in raw.columns}
    if missing:
        raise SystemExit(f"{path}: missing column(s) {sorted(missing)}")

    out: Dict[str, pd.DataFrame] = {}
    for prefix, short in CHANNELS.items():
        sub = raw[raw["signal"].str.startswith(prefix)]
        if sub.empty:
            continue
        sub = sub.assign(park=sub["signal"].str.slice(len(prefix)))
        out[short] = sub.pivot_table(index="time_s", columns="park",
                                     values="value", aggfunc="last")
    if "v_eff" not in out and not {"v_filt", "v_anchor"} <= set(out):
        raise SystemExit(
            f"{path}: neither veffDER_* nor both of vfiltDER_*/vanchorDER_* "
            f"present; nothing to measure.")
    return out


def effective_veff(ch: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """``V_filt - V_anchor`` per park.

    Prefers the block's own ``veff`` -- it is what the dead zone actually
    compared -- and falls back to the difference only if that channel is
    absent, so a run exported before ``veff`` existed can still be read.
    """
    if "v_eff" in ch:
        return ch["v_eff"]
    common = ch["v_filt"].columns.intersection(ch["v_anchor"].columns)
    return ch["v_filt"][common] - ch["v_anchor"][common]


def anchor_update_times(anchor: pd.Series,
                        eps: float = ANCHOR_EPS_PU) -> List[float]:
    """Times at which this park's anchor actually moved.

    The first sample is NOT an update: the anchor is initialised there, and
    counting it would report an extra interval of zero length in every run.
    """
    values = anchor.to_numpy(dtype=float)
    times = anchor.index.to_numpy(dtype=float)
    if values.size < 2:
        return []
    moved = np.abs(np.diff(values)) > float(eps)
    return [float(t) for t in times[1:][moved]]


def park_drift(veff: pd.Series, anchor: Optional[pd.Series],
               *, startup_exclude_s: float = 0.0) -> ParkDrift:
    """Per-interval and overall maxima of ``|veff|`` for one park.

    ``startup_exclude_s`` is applied BEFORE any reduction, and the caller is
    expected to have declared it; see the module docstring of the runbook.
    """
    park = str(veff.name)
    series = veff[veff.index >= float(startup_exclude_s)].dropna()
    if series.empty:
        return ParkDrift(park, 0, float("nan"), float("nan"))

    times = series.index.to_numpy(dtype=float)
    values = np.abs(series.to_numpy(dtype=float))

    updates: List[float] = []
    if anchor is not None:
        anchor_win = anchor[anchor.index >= float(startup_exclude_s)].dropna()
        updates = anchor_update_times(anchor_win)

    # Interval edges: the observation start, every anchor update, the end.
    edges = [float(times[0])] + updates + [float(times[-1])]
    interval_max: List[float] = []
    gaps: List[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        # Half-open on the right so an update instant belongs to the interval
        # it OPENS -- the anchor has already moved by then.
        mask = (times >= lo) & (times < hi)
        if not mask.any():
            continue
        interval_max.append(float(values[mask].max()))
        gaps.append(hi - lo)

    return ParkDrift(
        park=park,
        n_anchor_updates=len(updates),
        max_abs_veff=float(values.max()),
        p95_abs_veff=float(np.percentile(values, 95)),
        interval_max=interval_max,
        max_interval_s=float(max(gaps)) if gaps else 0.0,
        never_reanchored=not updates,
    )


def park_activation(ch: Dict[str, pd.DataFrame], park: str,
                    *, startup_exclude_s: float = 0.0) -> ParkActivation:
    """Dead-zone activation and droop command for one park.

    The incremental command is reported twice.  ``unsat`` is ``qraw - qset``,
    what the droop law asked for; ``sat`` is ``qext - qset``, what survived the
    capability clip.  A park at a limit has ``sat = 0`` with ``unsat`` large,
    and reporting only the second would score an actively-commanding dead zone
    as quiet.
    """
    def col(name: str) -> Optional[pd.Series]:
        frame = ch.get(name)
        if frame is None or park not in frame.columns:
            return None
        series = frame[park]
        return series[series.index >= float(startup_exclude_s)].dropna()

    q_corr = col("q_corr")
    q_raw, q_ext, q_set = col("q_raw"), col("q_ext"), col("q_set")

    n_active = 0
    max_qcorr = 0.0
    if q_corr is not None and not q_corr.empty:
        magnitude = np.abs(q_corr.to_numpy(dtype=float))
        n_active = int((magnitude > QCORR_EPS_PU).sum())
        max_qcorr = float(magnitude.max())

    def peak(a: Optional[pd.Series], b: Optional[pd.Series]) -> float:
        if a is None or b is None or a.empty or b.empty:
            return float("nan")
        common = a.index.intersection(b.index)
        if common.empty:
            return float("nan")
        return float(np.abs(a.loc[common].to_numpy(dtype=float)
                            - b.loc[common].to_numpy(dtype=float)).max())

    # Directional headroom at the setpoint.  qmin/qmax are block parameters
    # and are not exported as trajectories, so this is inferred from how far
    # the saturated reference ever moved from the setpoint -- zero in BOTH
    # directions while the dead zone was active is the signature of a park
    # with nowhere to go.
    head_up = head_dn = float("nan")
    if q_ext is not None and q_set is not None and not q_ext.empty:
        common = q_ext.index.intersection(q_set.index)
        if not common.empty:
            delta = (q_ext.loc[common].to_numpy(dtype=float)
                     - q_set.loc[common].to_numpy(dtype=float))
            head_up = float(max(delta.max(), 0.0))
            head_dn = float(max(-delta.min(), 0.0))

    zero_head = bool(
        n_active > 0
        and np.isfinite(head_up) and np.isfinite(head_dn)
        and head_up <= HEADROOM_EPS_PU and head_dn <= HEADROOM_EPS_PU
    )

    return ParkActivation(
        park=park,
        n_dead_zone_active=n_active,
        max_abs_qcorr=max_qcorr,
        max_abs_incremental_unsat=peak(q_raw, q_set),
        max_abs_incremental_sat=peak(q_ext, q_set),
        headroom_up=head_up,
        headroom_down=head_dn,
        zero_headroom=zero_head,
    )


def measure_reference(run_dir: Path, *, startup_exclude_s: float = 0.0
                      ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """F1-reference: the drift a candidate half-width must sit above."""
    ch = load_channels(run_dir)
    veff = effective_veff(ch)
    anchor = ch.get("v_anchor")

    rows: List[ParkDrift] = []
    for park in veff.columns:
        a = anchor[park] if anchor is not None and park in anchor else None
        rows.append(park_drift(veff[park], a,
                               startup_exclude_s=startup_exclude_s))

    frame = pd.DataFrame([{
        "park": r.park,
        "n_anchor_updates": r.n_anchor_updates,
        "max_abs_veff_pu": r.max_abs_veff,
        "p95_abs_veff_pu": r.p95_abs_veff,
        "max_interval_s": r.max_interval_s,
        "never_reanchored": r.never_reanchored,
    } for r in rows]).sort_values("max_abs_veff_pu", ascending=False)

    finite = frame["max_abs_veff_pu"].replace([np.inf, -np.inf], np.nan).dropna()
    summary = {
        "n_parks": int(len(frame)),
        "n_never_reanchored": int(frame["never_reanchored"].sum()),
        "max_abs_veff_pu": float(finite.max()) if len(finite) else float("nan"),
        "p95_park_max_pu": (float(np.percentile(finite, 95))
                            if len(finite) else float("nan")),
        "startup_exclude_s": float(startup_exclude_s),
    }
    return frame, summary


def measure_confirm(run_dir: Path, *, delta_pu: float,
                    startup_exclude_s: float = 0.0
                    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """F1-confirmation: the width is inert, and saturation is not hiding it."""
    ch = load_channels(run_dir)
    veff = effective_veff(ch)

    rows = [park_activation(ch, park, startup_exclude_s=startup_exclude_s)
            for park in veff.columns]
    frame = pd.DataFrame([{
        "park": r.park,
        "n_dead_zone_active": r.n_dead_zone_active,
        "max_abs_qcorr_pu": r.max_abs_qcorr,
        "max_incremental_unsat_pu": r.max_abs_incremental_unsat,
        "max_incremental_sat_pu": r.max_abs_incremental_sat,
        "headroom_up_pu": r.headroom_up,
        "headroom_down_pu": r.headroom_down,
        "zero_headroom": r.zero_headroom,
    } for r in rows]).sort_values("n_dead_zone_active", ascending=False)

    n_active_parks = int((frame["n_dead_zone_active"] > 0).sum())
    summary = {
        "delta_pu": float(delta_pu),
        "n_parks": int(len(frame)),
        "n_parks_dead_zone_active": n_active_parks,
        "n_parks_zero_headroom": int(frame["zero_headroom"].sum()),
        "max_abs_qcorr_pu": float(frame["max_abs_qcorr_pu"].max()),
        "max_incremental_unsat_pu": float(
            frame["max_incremental_unsat_pu"].max(skipna=True)),
        "passes": bool(n_active_parks == 0),
        "startup_exclude_s": float(startup_exclude_s),
    }
    return frame, summary


def select_half_width(reference_summaries: Sequence[Dict[str, float]],
                      *, margin_pu: float) -> Dict[str, float]:
    """Common half-width from the maximum across all parks and windows.

    One number for the whole population, per the experiment plan: a per-park
    width would be a per-park parameter set, which this study does not select.
    The margin is a DECLARED numerical/measurement allowance and must be frozen
    before candidates are compared, not tuned until a candidate passes.
    """
    maxima = [float(s["max_abs_veff_pu"]) for s in reference_summaries
              if np.isfinite(s.get("max_abs_veff_pu", np.nan))]
    if not maxima:
        raise SystemExit("no finite reference maximum; nothing to select from")
    observed = max(maxima)
    return {
        "n_runs": len(maxima),
        "observed_max_pu": observed,
        "margin_pu": float(margin_pu),
        "delta_pu": observed + float(margin_pu),
    }


# =====================================================================
#  Offline self-test
# =====================================================================

def _synthetic(tmp: Path, *, anchor_steps: Sequence[float],
               drift_amplitude: float, qcorr: float = 0.0,
               saturate: bool = False, decay: float = 1.0) -> Path:
    """Write a synthetic rms_qv_internals.csv with a known answer.

    ``decay`` scales each successive interval's drift amplitude, so a fixture
    can be built whose LARGEST drift is in the first interval -- which is what
    makes a startup-exclusion assertion discriminating.  With ``decay = 1`` the
    sawtooth reaches the same peak in every interval, and excluding a prefix
    changes nothing, because the later intervals still get there.
    """
    times = np.arange(0.0, 60.0, 1.0)
    anchor = np.zeros_like(times)
    for t_step in anchor_steps:
        anchor[times >= t_step] += 0.1
    # Sawtooth drift that resets at each anchor step, peaking at the interval's
    # amplitude just before the next one.
    v_eff = np.zeros_like(times)
    edges = [0.0] + list(anchor_steps) + [times[-1] + 1.0]
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (times >= lo) & (times < hi)
        n = int(mask.sum())
        if n:
            v_eff[mask] = np.linspace(
                0.0, drift_amplitude * (float(decay) ** i), n)

    q_set = np.full_like(times, 0.20)
    q_corr = np.full_like(times, float(qcorr))
    q_raw = q_set - 2.0 * q_corr
    q_ext = q_set.copy() if saturate else q_raw.copy()

    rows = []
    for label, series in (("vanchorDER_P1", anchor), ("vfiltDER_P1", v_eff + anchor),
                          ("veffDER_P1", v_eff), ("qcorrDER_P1", q_corr),
                          ("qrawDER_P1", q_raw), ("qextDER_P1", q_ext),
                          ("qsetDER_P1", q_set)):
        for t_value, y_value in zip(times, series):
            rows.append({"signal": label, "signal_type": "other",
                         "time_s": float(t_value), "value": float(y_value)})
    csv_dir = tmp / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(csv_dir / "rms_qv_internals.csv",
                                           index=False)
    return tmp


def self_test() -> int:
    import tempfile

    print("[self-test] F1 anchor-referenced measurement")
    ok = True

    def check(label: str, condition: bool) -> None:
        nonlocal ok
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
        ok = ok and bool(condition)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        run = _synthetic(root / "ref", anchor_steps=[20.0, 40.0],
                         drift_amplitude=0.004)
        frame, summary = measure_reference(run)
        check("two anchor updates are detected",
              int(frame.loc[0, "n_anchor_updates"]) == 2)
        check("the initial sample is not counted as an update",
              int(frame.loc[0, "n_anchor_updates"]) == 2)
        check("the pooled maximum is the drift amplitude",
              abs(summary["max_abs_veff_pu"] - 0.004) < 1e-9)
        check("a re-anchoring run is not flagged never_reanchored",
              not bool(frame.loc[0, "never_reanchored"]))

        held = _synthetic(root / "held", anchor_steps=[],
                          drift_amplitude=0.004)
        frame_h, summary_h = measure_reference(held)
        check("an anchor that never moves is reported, not silently one interval",
              bool(frame_h.loc[0, "never_reanchored"])
              and summary_h["n_never_reanchored"] == 1)

        quiet = _synthetic(root / "quiet", anchor_steps=[20.0],
                           drift_amplitude=0.004, qcorr=0.0)
        _f, s_quiet = measure_confirm(quiet, delta_pu=0.02)
        check("zero dead-zone output passes the confirmation",
              s_quiet["passes"] and s_quiet["n_parks_dead_zone_active"] == 0)

        active = _synthetic(root / "active", anchor_steps=[20.0],
                            drift_amplitude=0.004, qcorr=0.003)
        f_act, s_act = measure_confirm(active, delta_pu=0.002)
        check("a non-zero dead-zone output FAILS the confirmation",
              (not s_act["passes"]) and s_act["n_parks_dead_zone_active"] == 1)
        check("the unsaturated incremental command is reported",
              abs(float(f_act.loc[0, "max_incremental_unsat_pu"])
                  - 0.006) < 1e-9)

        pinned = _synthetic(root / "pinned", anchor_steps=[20.0],
                            drift_amplitude=0.004, qcorr=0.003, saturate=True)
        f_pin, s_pin = measure_confirm(pinned, delta_pu=0.002)
        check("a SATURATED park still fails -- saturation does not hide it",
              not s_pin["passes"])
        check("and it is reported as zero-headroom, not as quiet",
              bool(f_pin.loc[0, "zero_headroom"])
              and s_pin["n_parks_zero_headroom"] == 1)
        check("its saturated incremental command is zero while unsat is not",
              abs(float(f_pin.loc[0, "max_incremental_sat_pu"])) < 1e-12
              and float(f_pin.loc[0, "max_incremental_unsat_pu"]) > 1e-9)

        sel = select_half_width([summary, summary_h], margin_pu=0.001)
        check("the half-width is the pooled maximum plus the declared margin",
              abs(sel["delta_pu"] - 0.005) < 1e-9)

        # Largest drift in the FIRST interval, so excluding it must move the
        # answer.  With a flat sawtooth it would not, and the assertion would
        # pass for the wrong reason.
        decaying = _synthetic(root / "decay", anchor_steps=[20.0, 40.0],
                              drift_amplitude=0.004, decay=0.5)
        full = measure_reference(decaying)[1]
        trimmed = measure_reference(decaying, startup_exclude_s=20.0)[1]
        check("without exclusion the first (largest) interval sets the maximum",
              abs(full["max_abs_veff_pu"] - 0.004) < 1e-9)
        check("the startup exclusion is applied BEFORE reduction",
              abs(trimmed["max_abs_veff_pu"] - 0.002) < 1e-9)
        check("and it is recorded in the summary",
              trimmed["startup_exclude_s"] == 20.0)

    print(f"[self-test] {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="analysis.deadband_anchor")
    ap.add_argument("--run", type=Path, action="append", default=None,
                    help="run directory (repeatable; the selection pools "
                         "every run given)")
    ap.add_argument("--mode", choices=("reference", "confirm"),
                    default="reference")
    ap.add_argument("--delta", type=float, default=None,
                    help="candidate half-width [pu]; required for --mode confirm")
    ap.add_argument("--margin", type=float, default=0.001,
                    help="declared numerical/measurement margin [pu] added to "
                         "the observed maximum. FREEZE THIS before comparing "
                         "candidates.")
    ap.add_argument("--startup-exclude-s", type=float, default=0.0,
                    help="seconds discarded at the start of every leg. Declare "
                         "it before reduction and use the same value in every "
                         "leg; convergence is NOT implied by 180 s elapsing.")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the per-park table here as CSV")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test()
    if not a.run:
        ap.error("--run is required (repeatable)")
    if a.mode == "confirm" and a.delta is None:
        ap.error("--mode confirm requires --delta")

    frames: List[pd.DataFrame] = []
    summaries: List[Dict[str, float]] = []
    for run_dir in a.run:
        if a.mode == "reference":
            frame, summary = measure_reference(
                run_dir, startup_exclude_s=a.startup_exclude_s)
        else:
            frame, summary = measure_confirm(
                run_dir, delta_pu=float(a.delta),
                startup_exclude_s=a.startup_exclude_s)
        frame = frame.assign(run=run_dir.name)
        frames.append(frame)
        summaries.append(summary)
        print(f"\n=== {run_dir} ===")
        print(frame.head(12).to_string(index=False))
        print(f"  summary: {summary}")

    if a.mode == "reference":
        sel = select_half_width(summaries, margin_pu=float(a.margin))
        print(f"\n[F1] observed max |V_filt - V_anchor| = "
              f"{sel['observed_max_pu']:.6g} pu over {sel['n_runs']} run(s)")
        print(f"[F1] + declared margin {sel['margin_pu']:.6g} pu "
              f"-> candidate half-width delta = {sel['delta_pu']:.6g} pu")
        print("[F1] this is a LOWER BOUND on the width that keeps the droop "
              "inert on THESE windows; it is not a probabilistic guarantee.")
    else:
        n_bad = sum(1 for s in summaries if not s["passes"])
        print(f"\n[F1] confirmation at delta = {a.delta:g} pu: "
              f"{len(summaries) - n_bad}/{len(summaries)} run(s) inert")
        n_pinned = sum(int(s["n_parks_zero_headroom"]) for s in summaries)
        if n_pinned:
            print(f"[F1] {n_pinned} zero-headroom park-run(s) reported "
                  f"separately: saturation must not be read as quiet.")

    if a.out and frames:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(a.out, index=False)
        print(f"[F1] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
