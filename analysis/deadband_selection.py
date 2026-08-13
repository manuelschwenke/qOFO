#!/usr/bin/env python3
r"""Dead-band selection study: collect metrics, write CSVs, draw both figures.

Answers thesis Ch. 8 §2: which DER Q(V) dead-zone half-width delta to choose.

The dead band is a two-sided design choice. Too narrow and the local droop answers
every small profile variation, so the DER chatter continuously and the OFO keeps
re-anchoring against them. Too wide and the droop is inactive between dispatches, so
there is no local support at all and the interface Q drifts until the next OFO step.
Both extremes therefore degrade the controlled quantities, and the useful delta is
the interior minimum.

Three controlled quantities, all measured on the SAME runs::

    interface Q   mean |Q_act - Q_set| over the TS-DSO interfaces        [Mvar]
    TS voltage    per-zone RMS voltage error                             [pu]
    DS voltage    RMS deviation of each DSO group's mean V from V_ref    [pu]

Outputs (into ``results/deadband_selection/``)::

    deadband_metrics.csv      one row per run: window, delta, run, the 3 metrics
    deadband_optima.csv       argmin delta per window per metric
    figures/deadband_selection.png/.pdf   per-window U-curves, one panel
    figures/deadband_band.png/.pdf        min/max band across windows

Every number is read from the run directories; nothing is hard-coded. Runs are
admitted only if their own ``config.json`` matches the study configuration (see
``ADMIT``), which is what keeps a differently-configured run from silently entering
a curve -- on 2026-07-29 a sweep launched without an explicit ``--scenario``
produced one ``base_410`` run in the middle of a ``rural_700`` series, and a
run-number threshold would not have caught it.

Usage::

    python -m analysis.deadband_selection
    python -m analysis.deadband_selection --scenario rural_700
    python -m analysis.deadband_selection --results-root <dir> --out <dir>

Author: Manuel Schwenke / Claude Code (2026-07-31)
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "rms_phase6_replay"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "results" / "deadband_selection"

#: Voltage reference the DS group deviation is measured against [pu].
V_REF = 1.03

#: Configuration a run must match to enter the study.  Compared against the
#: ``runner_static`` block of each run's own ``config.json``.  ``None`` means the
#: key must be absent or null.
ADMIT: Dict[str, object] = {
    "der_q_capability_override_pu": None,   # physical VDE capability, not a stub
    "use_profiles": True,                   # profiled operation, not a step test
    "dso_qv_slope_pu": 0.06,                # the droop slope the study fixes
    "seed_der_anchor_to_local_v": False,
    "disable_qv_seed": False,
    # Per-DSO scenario multipliers.  The runner itself warns that a scaled run
    # is "NOT comparable with an unscaled run", so they belong in the filter:
    # run 0080 (2026-07-30) predates the DSO_3 x2 default and records neither
    # key, and without these two entries it was admitted into a curve otherwise
    # built from x2 runs.  It only escaped notice because the sweep happens to
    # re-run 0080's exact (window, delta) cell and overwrite it.
    "dso_der_scale": {"DSO_3": 2.0},
    "dso_load_p_scale": {"DSO_3": 2.0},
    # Undisturbed operation only. The disturbance-rejection study writes into
    # the SAME results root with an otherwise identical configuration, so
    # without this key its runs would be admitted into the profiled-operation
    # curves and silently corrupt them -- the same failure mode as the
    # unscaled run 0080. Runs carrying a load step are selected by
    # analysis/deadband_disturbance.py instead.
    "load_step_time_s": None,
    # Code-version boundary. The reduced-network sensitivities were corrected on
    # 2026-08-01 (rev 2): before that the DSO reductions solved on the wrong
    # power-flow branch and were 0.10-0.36 pu from the plant. Nothing else in
    # config.json distinguishes the two generations, and the metric moves ~12%
    # across the boundary, so runs from rev 1 must not enter the same curve.
    # Runs predating the field have no key at all and are rejected here.
    "sensitivity_reduction_rev": 2.0,
    # Simulation horizon.  Every metric is a mean over the run, so a run of a
    # different length is not comparable -- and the N-1 study uses a LONGER
    # horizon (its recovery is governed by the 180 s TSO period), so without
    # this key its undisturbed twins would be admitted here as ordinary cells.
    # All 214 runs of this study are 300 s.
    "n_total_s": 300.0,
}

def undisturbed_topology(cfg: dict) -> bool:
    """True if the run carries no contingency.

    This study is profiled operation on an intact network; the N-1 study runs
    the SAME configuration with a generator trip, and its runs are undisturbed
    in every key ``ADMIT`` checks (no load step, profiled, rev 2, scaled DSO_3,
    diagonal dead band). Without this guard they are admitted here on equal
    terms -- observed 2026-08-02, when the gen-trip probe 0283 silently
    superseded study run 0201 in the delta = 0.01 cell of window
    2016-01-05T08:00 and the curve was then built from a run containing a
    generator outage.

    ``contingencies`` is ``[]`` on an intact run, not absent, so this cannot be
    expressed as a ``None`` entry in ``ADMIT``.
    """
    return not (cfg.get("contingencies") or [])


def uniform_deadband(cfg: dict) -> bool:
    """True if the run applied ONE dead band to both voltage levels.

    This study is by definition the diagonal ``delta_TS == delta_DS``: it varies
    a single number applied to every park, and its curves are indexed by
    ``tso_qv_deadband_pu`` alone.  The 2D study (``analysis/deadband_2d.py``)
    writes into the SAME results root with an otherwise identical configuration
    but with the two levels set independently, so without this guard an
    off-diagonal run would be admitted here under its TS value and silently
    supersede the genuine cell -- the same failure mode as the unscaled run 0080
    that motivated the ``dso_der_scale`` entries in ``ADMIT``.

    A run predating the split (no ``dso_qv_deadband_pu`` key) is accepted: the
    field is old, and every such run set both levels from one flag.
    """
    got = cfg.get("dso_qv_deadband_pu")
    if got is None:
        return True
    try:
        return abs(float(got) - float(cfg.get("tso_qv_deadband_pu"))) < 1e-12
    except (TypeError, ValueError):
        return False


#: QSS per-interval voltage excursion per window [pu], from the Tier-1 season
#: screening.  Measured on an older topology, so used ONLY to order the windows
#: and to justify that they span the annual range -- never quoted as a result.
EXCURSION: Dict[str, float] = {
    "2016-01-05T08:00": 0.00828,
    "2016-01-15T03:00": 0.01573,
    "2016-07-15T03:00": 0.02051,
}

METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("ifq", "interface-Q tracking", "#c0392b"),
    ("tsv", "TS zone voltage", "#2171b5"),
    ("dsv", "DS group voltage", "#08856b"),
)


# =====================================================================
#  Collection
# =====================================================================

def _admit(run_dir: Path, scenario: str) -> Optional[Tuple[str, float]]:
    """Return ``(window, delta)`` if the run belongs in the study, else ``None``."""
    try:
        raw = json.loads(
            (run_dir / "config.json").read_text(encoding="utf-8", errors="ignore")
        )
    except Exception:
        return None
    cfg = raw.get("runner_static", {})
    if not cfg:
        return None
    if str(cfg.get("scenario", "")) != scenario:
        return None
    for key, want in ADMIT.items():
        got = cfg.get(key)
        if want is None:
            if got is not None:
                return None
        elif isinstance(want, dict):
            # A missing key is NOT an empty dict: a run predating the multiplier
            # records nothing, and treating that as "no scaling" would silently
            # admit it beside scaled runs.  Absent -> reject.
            if not isinstance(got, dict):
                return None
            try:
                if {str(k): float(v) for k, v in got.items()} != want:
                    return None
            except (TypeError, ValueError):
                return None
        elif isinstance(want, float):
            try:
                if abs(float(got) - want) > 1e-9:
                    return None
            except (TypeError, ValueError):
                return None
        elif bool(got) != bool(want):
            return None
    if not uniform_deadband(cfg) or not undisturbed_topology(cfg):
        return None
    try:
        delta = float(cfg.get("tso_qv_deadband_pu"))
    except (TypeError, ValueError):
        return None
    if delta < 0:
        return None
    return str(cfg.get("start_time", ""))[:16], delta


def _at_edge(best: float, deltas) -> bool:
    """True if ``best`` is the smallest or largest delta swept.

    An argmin at the edge of the swept range is NOT a measured optimum: the
    metric may still be improving beyond it. Reporting it as ``delta*`` invites
    exactly the wrong conclusion -- e.g. window 2016-01-15T03:00, where
    interface Q falls monotonically to the top of the range and the true optimum
    is somewhere above it, unbracketed.
    """
    ds = sorted(deltas)
    return len(ds) >= 2 and (best == ds[0] or best == ds[-1])


def _flat(values) -> bool:
    """True if a metric does not vary with delta at all across a window.

    An argmin over identical values is an artefact of dict ordering, not a
    result. Observed 2026-07-31 at window 2016-07-15T03:00, where all five dead
    bands returned bit-identical runs because every DER park had P = 0 MW and
    therefore zero Q capability under the VDE-AR-N-4120 diagram -- with no DER
    able to move, the dead zone cannot bind. Reporting delta* = 0.0025 there
    (simply the first key) would have put a meaningless number into the
    cross-window comparison.
    """
    vals = [v for v in values if np.isfinite(v)]
    if len(vals) < 2:
        return False
    span = max(vals) - min(vals)
    scale = max(abs(v) for v in vals) or 1.0
    return span / scale < 1e-9


def _metrics(run_dir: Path) -> Tuple[float, float, float]:
    """Mean interface-Q error [Mvar], TS voltage RMS error [pu], DS RMS dev [pu]."""
    with (run_dir / "rms_records.pkl").open("rb") as fh:
        records = pickle.load(fh)

    q_err: List[float] = []
    ts_v: List[float] = []
    ds_v: List[float] = []
    for rec in records:
        want = dict(rec.dso_trafo_q_set_mvar)
        got = dict(rec.dso_trafo_q_actual_mvar)
        q_err += [abs(got[k] - want[k]) for k in want if k in got]

        zone = getattr(rec, "zone_v_rms_err_pu", None)
        if isinstance(zone, dict) and zone:
            ts_v.append(float(np.mean([abs(v) for v in zone.values()])))

        group = dict(rec.dso_group_v_mean_pu)
        if group:
            ds_v.append(
                float(np.sqrt(np.mean([(v - V_REF) ** 2 for v in group.values()])))
            )

    return (
        float(np.mean(q_err)) if q_err else float("nan"),
        float(np.mean(ts_v)) if ts_v else float("nan"),
        float(np.mean(ds_v)) if ds_v else float("nan"),
    )


def collect(results_root: Path, scenario: str, verbose: bool = True):
    """Scan run directories; return ``{window: {delta: (run, ifq, tsv, dsv)}}``."""
    data: Dict[str, Dict[float, Tuple[str, float, float, float]]] = {}
    skipped = 0
    for run_dir in sorted(results_root.glob("0*")):
        if not (run_dir / "rms_records.pkl").exists():
            continue                      # aborted run: no results to read
        admitted = _admit(run_dir, scenario)
        if admitted is None:
            skipped += 1
            continue
        window, delta = admitted
        cell = data.setdefault(window, {})
        if delta in cell and verbose:
            # Last-writer-wins is fine for a genuine re-run, but it must never
            # be silent: this is how an out-of-study run would replace a study
            # run (or be replaced by one) with no trace in the CSV.
            print(f"[admit] duplicate cell {window} delta={delta:g}: "
                  f"run {run_dir.name[:4]} supersedes {cell[delta][0]}")
        cell[delta] = (run_dir.name[:4],) + _metrics(run_dir)
    if verbose:
        print(f"[admit] scenario={scenario!r}: {sum(len(v) for v in data.values())} "
              f"run(s) admitted, {skipped} skipped on scenario/config mismatch")
    return data


# =====================================================================
#  CSV
# =====================================================================

def write_csvs(data, out_root: Path) -> Tuple[Path, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    metrics_path = out_root / "deadband_metrics.csv"
    optima_path = out_root / "deadband_optima.csv"

    order = sorted(data, key=lambda w: EXCURSION.get(w, 9.0))
    with metrics_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["window", "screening_excursion_pu_old_topology", "delta_pu", "run",
                    "ifq_mean_abs_err_mvar", "ts_v_rms_err_pu",
                    "ds_v_rms_dev_pu"])
        for window in order:
            for delta in sorted(data[window]):
                run, ifq, tsv, dsv = data[window][delta]
                w.writerow([window, EXCURSION.get(window, ""), f"{delta:g}", run,
                            f"{ifq:.6f}", f"{tsv:.8f}", f"{dsv:.8f}"])

    with optima_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["window", "screening_excursion_pu_old_topology", "metric", "delta_star_pu",
                    "value_at_optimum", "n_deltas", "at_range_edge"])
        for window in order:
            rows = data[window]
            if len(rows) < 3:
                continue                  # an argmin over <3 points is not a curve
            for col, (key, label, _c) in enumerate(METRICS, start=1):
                vals = {d: rows[d][col] for d in rows if np.isfinite(rows[d][col])}
                if not vals:
                    continue
                best = min(vals, key=vals.get)
                w.writerow([window, EXCURSION.get(window, ""), key,
                            f"{best:g}", f"{vals[best]:.8f}", len(vals),
                            int(_at_edge(best, vals))])
    return metrics_path, optima_path


# =====================================================================
#  Figures
# =====================================================================

def _label(window: str) -> str:
    exc = EXCURSION.get(window)
    return f"{window}" + (f"  ({exc:.4f} pu)" if exc else "")


def figure_selection(data, out_dir: Path, scenario: str) -> Optional[Path]:
    """Per-window U-curves: the three controlled quantities against delta.

    One panel, twin axes (Mvar left, pu right). Each metric's minimum is ringed.
    With several windows present, each is drawn in its own line style so the
    interior minimum can be seen to persist -- or not -- across operating points.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [w for w in sorted(data, key=lambda w: EXCURSION.get(w, 9.0))
             if len(data[w]) >= 2]
    if not order:
        print("[figure] deadband_selection: need >=2 dead bands in a window; skipped")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax2 = ax.twinx()
    ax.axvspan(0.005, 0.01, color="#f0e442", alpha=.16, zorder=0,
               label="_nolegend_")

    styles = ["o-", "s--", "^:"]
    handles = []
    for wi, window in enumerate(order):
        rows = data[window]
        deltas = np.array(sorted(rows))
        ifq = np.array([rows[d][1] for d in deltas])
        tsv = np.array([rows[d][2] for d in deltas])
        dsv = np.array([rows[d][3] for d in deltas])
        style = styles[wi % len(styles)]
        suffix = f"  [{window[5:10]}]" if len(order) > 1 else ""
        for arr, axis, (key, label, colour) in zip(
            (ifq, tsv, dsv), (ax, ax2, ax2), METRICS
        ):
            ln, = axis.plot(deltas, arr, style, color=colour, lw=2.1, ms=6.5,
                            alpha=1.0 if wi == 0 else 0.65,
                            label=f"{label}{suffix}")
            if wi == 0:
                handles.append(ln)
            if np.any(np.isfinite(arr)):
                i = int(np.nanargmin(arr))
                axis.plot(deltas[i], arr[i], "o", ms=15, mfc="none", mec=colour,
                          mew=2.0, zorder=5)

    ax.set_xlabel(r"DER Q(V) dead-zone half-width  $\delta$  [pu]")
    ax.set_ylabel(r"interface-Q tracking  mean $|e|$  [Mvar]", color="#c0392b")
    ax2.set_ylabel("voltage error  [pu]")
    ax.tick_params(axis="y", labelcolor="#c0392b")
    ax.set_title(
        "Selecting the DER Q(V) dead band: all three controlled quantities\n"
        "degrade at both extremes  "
        f"(IEEE 39, {scenario}, physical VDE capability,\n"
        "profiled operation; rings mark each curve's minimum)",
        fontsize=10.5,
    )
    ax.grid(alpha=.3)
    # The two voltage curves live on the twin axis, so the handles must be passed
    # explicitly: ax.legend() alone sees only the artists drawn on ax and would
    # silently show the interface-Q entry by itself.
    if len(order) > 1:
        handles.append(
            ax.plot([], [], " ",
                    label=f"({len(order)} windows, styles o- s-- ^:)")[0]
        )
    ax.legend(handles=handles, fontsize=8.5, loc="upper left", framealpha=.95)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"deadband_selection.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {out_dir / 'deadband_selection.png'}")
    return out_dir / "deadband_selection.png"


def figure_band(data, out_dir: Path, scenario: str) -> Optional[Path]:
    """Min/max band across windows, each metric normalised to its own best.

    The windows sit at very different absolute levels, so raw curves cannot be
    overlaid. Normalising each metric by its best value WITHIN each window makes
    them comparable: 1.0 means "as good as that window ever gets", 2.0 means
    "twice that window's best error". The shaded band is the operating-point
    sensitivity; where its floor sits is the delta that is good across all of them.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = [w for w in sorted(data, key=lambda w: EXCURSION.get(w, 9.0))
               if len(data[w]) >= 2]
    if len(windows) < 2:
        print("[figure] deadband_band: needs >=2 windows; skipped "
              f"(have {len(windows)})")
        return None
    shared = sorted(set.intersection(*[set(data[w]) for w in windows]))
    if len(shared) < 2:
        print("[figure] deadband_band: windows share <2 dead bands; skipped")
        return None

    norm = {}
    for w in windows:
        M = np.array([[data[w][d][c] for c in (1, 2, 3)] for d in shared])
        norm[w] = M / np.nanmin(M, axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.array(shared)
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.axvspan(0.005, 0.01, color="#f0e442", alpha=.16, zorder=0)

    for k, (key, label, colour) in enumerate(METRICS):
        Y = np.array([norm[w][:, k] for w in windows])
        lo, hi = np.nanmin(Y, axis=0), np.nanmax(Y, axis=0)
        ax.fill_between(x, lo, hi, color=colour, alpha=.18, zorder=1)
        ax.plot(x, lo, "-", color=colour, lw=1.1, alpha=.75, zorder=2)
        ax.plot(x, hi, "-", color=colour, lw=1.1, alpha=.75, zorder=2)
        mid = np.nanmean(Y, axis=0)
        ax.plot(x, mid, "o-", color=colour, lw=2.4, ms=6, zorder=3,
                label=f"{label}  (band = min/max over {len(windows)} windows)")
        i = int(np.nanargmin(mid))
        ax.plot(x[i], mid[i], "o", ms=15, mfc="none", mec=colour, mew=2.0,
                zorder=4)

    ax.axhline(1.0, ls=":", color="k", lw=1.2)
    ax.text(x[-1], 1.02, "each window's own best", ha="right", va="bottom",
            fontsize=8, style="italic", color="#555555")
    ax.set_xlabel(r"DER Q(V) dead-zone half-width  $\delta$  [pu]")
    ax.set_ylabel("error relative to that window's best  [-]")
    exc = ", ".join(f"{EXCURSION[w]:.4f}" for w in windows if w in EXCURSION)
    ax.set_title(
        "Dead-band selection across operating points\n"
        f"band = min/max over profile windows spanning the annual range of\n"
        f"per-interval voltage excursion ({exc} pu)  --  {scenario}",
        fontsize=10.5,
    )
    ax.grid(alpha=.3)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=.95)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"deadband_band.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {out_dir / 'deadband_band.png'}")
    return out_dir / "deadband_band.png"


# =====================================================================
#  Report
# =====================================================================

def print_tables(data) -> None:
    order = sorted(data, key=lambda w: EXCURSION.get(w, 9.0))
    optima: Dict[str, Tuple[float, float, float]] = {}
    for window in order:
        rows = data[window]
        exc = EXCURSION.get(window)
        # Never present this as a property of the run. 2016-07-15T03:00 entered
        # the matrix labelled "0.02051 pu, annual maximum" on the strength of
        # this column and turned out to be a zero-DER-capability hour on the
        # current topology -- the least stressed window of the three, not the
        # most. The figure selected the window; it does not describe it.
        head = window + (f"   (old-topology screening value {exc:.5f} pu "
                         f"-- selection only, NOT a property of this run)"
                         if exc else "")
        print(f"\n=== {head} ===")
        print(f"{'delta':>8} {'run':>6} {'ifQ mean|e|':>12} {'TS V rms':>10} "
              f"{'DS V rms':>10}")
        for delta in sorted(rows):
            run, ifq, tsv, dsv = rows[delta]
            print(f"{delta:8.4f} {run:>6} {ifq:12.3f} {tsv:10.5f} {dsv:10.5f}")
        if len(rows) >= 3:
            best = []
            for col in (1, 2, 3):
                vals = {d: rows[d][col] for d in rows
                        if np.isfinite(rows[d][col])}
                best.append(min(vals, key=vals.get) if vals else float("nan"))
            flat = [_flat([rows[d][col] for d in rows]) for col in (1, 2, 3)]
            # A flat metric has no optimum to report; storing one would feed a
            # dict-ordering artefact into the cross-window verdict.
            optima[window] = tuple(None if f else b for f, b in zip(flat, best))
            cells = []
            for name, b, f in zip(("ifQ", "TS V", "DS V"), best, flat):
                if f:
                    cells.append(f"{name}=flat")
                else:
                    cells.append(f"{name}={b:g}" + ("*" if _at_edge(b, rows) else ""))
            print("  optimum:  " + "  ".join(cells))
            if any(not f and _at_edge(b, rows) for b, f in zip(best, flat)):
                print("            * = at the edge of the swept range: the metric "
                      "is still improving there,")
                print("              so this is a bound, NOT a measured optimum.")
            if all(flat):
                print("            DEGENERATE WINDOW: delta changes nothing here. "
                      "Check DER Q capability --")
                print("            a window whose parks are all at P=0 has no "
                      "continuous actuator for the")
                print("            dead band to act on, so it cannot inform "
                      "delta* selection.")
            elif any(flat):
                print("            'flat' = this metric does not vary with delta "
                      "in this window.")
        else:
            print(f"  (only {len(rows)} dead band(s): no optimum yet)")

    if len(optima) > 1:
        print("\n" + "=" * 68)
        print("DOES THE OPTIMUM MOVE WITH THE OPERATING POINT?")
        print("  'screen' is the OLD-TOPOLOGY screening value used to pick the")
        print("  windows. It does not describe these runs and must not be")
        print("  quoted: the window it ranked most stressed (2016-07-15T03:00)")
        print("  turned out to have zero DER capability on this topology.")
        print(f"{'window':>18} {'screen':>10} {'ifQ*':>8} {'TS V*':>8} "
              f"{'DS V*':>8}")
        for window in order:
            if window not in optima:
                continue
            # None = the metric is flat in this window, so it has no argmin.
            cells = " ".join(f"{'flat':>8}" if v is None else f"{v:8g}"
                             for v in optima[window])
            exc = EXCURSION.get(window)
            # A window absent from EXCURSION has no comparable figure: those
            # values were measured on the older topology, and inventing one
            # here would silently mix two incompatible screenings.
            exc_s = f"{exc:10.5f}" if exc is not None else f"{'n/a':>10}"
            print(f"{window:>18} {exc_s} {cells}")
        flat_windows = [w for w in optima if optima[w][0] is None]
        live = {w: optima[w] for w in optima if optima[w][0] is not None}
        q_opt = [live[w][0] for w in live]
        edge_windows = [w for w in live if _at_edge(live[w][0], data[w])]
        print()
        if flat_windows:
            print("  EXCLUDED (delta has no effect there, so no optimum exists): "
                  + ", ".join(sorted(flat_windows)))
            print()
        if len(q_opt) < 2:
            print("  fewer than 2 windows with a delta-dependent interface-Q "
                  "metric;")
            print("  the operating-point question cannot be answered yet.")
            return
        if len(set(q_opt)) == 1:
            print(f"  interface-Q optimum is INVARIANT at delta = {q_opt[0]:g}")
            print("  => a system property, not an artefact of the operating point;")
            print("     the thesis claim can be stated generally.")
        else:
            print(f"  interface-Q optimum MOVES: {q_opt}")
            print("  => delta* is operating-point dependent.")
            print("     Do NOT regress it against the 'screen' column: that is")
            print("     an old-topology figure. Characterising the operating")
            print("     points needs a current-topology quantity measured on")
            print("     these runs (e.g. DER Q headroom, or a fresh screening).")
        if edge_windows:
            print()
            print("  CAVEAT: the interface-Q argmin is at the edge of the swept")
            print("  range in: " + ", ".join(sorted(edge_windows)))
            print("  Those are bounds, not optima -- the true minimum lies at or")
            print("  beyond the edge, so the spread above is a LOWER bound on how")
            print("  far delta* actually moves. Widen the sweep to bracket it.")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="rural_700",
                    help="network scenario a run must record to be admitted "
                         "(default: rural_700). Runs on other scenarios are NOT "
                         "comparable and are skipped.")
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
                    help="directory holding the numbered run folders")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT,
                    help="output directory for CSVs and figures")
    ap.add_argument("--no-figures", action="store_true",
                    help="write the CSVs only")
    args = ap.parse_args(argv)

    if not args.results_root.is_dir():
        print(f"error: results root not found: {args.results_root}")
        return 2

    data = collect(args.results_root, args.scenario)
    if not data:
        print(f"\nNo admitted runs for scenario={args.scenario!r}.")
        print("Run the sweep first:  experiments/run_deadband_sweep.ps1")
        return 1

    print_tables(data)
    metrics_path, optima_path = write_csvs(data, args.out)
    print(f"\n[csv] wrote {metrics_path}")
    print(f"[csv] wrote {optima_path}")
    if not args.no_figures:
        fig_dir = args.out / "figures"
        figure_selection(data, fig_dir, args.scenario)
        figure_band(data, fig_dir, args.scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
