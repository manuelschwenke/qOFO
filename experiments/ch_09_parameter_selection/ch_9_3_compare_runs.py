#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_3_compare_runs.py
===========================================================
Run the same disturbed scenario under several parameter sets and plot them
against each other.

Why not just call ``run_multi_system_ofo.py`` three times
--------------------------------------------------------
That script saves ``records.pkl`` and **nothing else** -- the live plots it
enables are interactive matplotlib windows, which neither survive a headless
batch nor get written to disk. The figures here are therefore produced
afterwards from the stored records, which also lets the three runs share one
set of axes instead of being compared by eye across three windows.

The runs are headless (``live_plot_* = False``) and parallel, one process per
parameter set.

What is plotted, per figure, one panel row per parameter set
------------------------------------------------------------
``ts_voltages``   per-zone min / mean / max envelope and the RMS error against
                  the schedule, with the hard corridor marked
``ds_voltages``   the same per DSO area
``interface_q``   ``Q_set`` from the supervisory layer against the ``Q``
                  actually delivered at each interface -- the tracking the
                  cascade exists to perform
``actuators``     every actuator class: TS DER Q, PCC setpoints, AVR setpoints,
                  TS OLTC taps, TS shunt steps, DSO DER Q, DSO tap positions

Contingencies are drawn as vertical markers on every panel, since a
trajectory feature that is not aligned to an event is a different claim from
one that is.

Usage::

    python experiments/ch_09_parameter_selection/ch_9_3_compare_runs.py \\
        --paramsets stage1_start ch9_selected --hours 5 --workers 3
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _pin_threads() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _run_one(job: Dict[str, Any]) -> Dict[str, Any]:
    """One parameter set, one scenario.  Module-level, so picklable."""
    _pin_threads()
    import dataclasses
    import warnings
    warnings.filterwarnings("ignore")

    out = Path(job["out"])
    if out.exists() and not job.get("force"):
        return {"paramset": job["paramset"], "status": "cached"}

    from configs.paramsets import apply_paramset
    from experiments.run_multi_system_ofo import make_config
    from experiments.runners import run_multi_tso_dso

    cfg = apply_paramset(make_config(), job["paramset"], verbose=False)
    cfg = dataclasses.replace(
        cfg,
        n_total_s=float(job["hours"]) * 3600.0,
        # Headless: the live plots are interactive windows and would either
        # block or be discarded.  The figures come from the records instead.
        live_plot_controller=False, live_plot_cascade=False,
        live_plot_system=False, live_plot_tracking=False, live_plot_sbx=False,
        verbose=0,
    )
    t0 = time.perf_counter()
    try:
        log = run_multi_tso_dso(cfg)
    except Exception as exc:                        # noqa: BLE001
        import traceback
        return {"paramset": job["paramset"], "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}
    wall = time.perf_counter() - t0

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump(log, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return {"paramset": job["paramset"], "status": "ok",
            "records": len(log), "wall_s": wall}


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def _event_times_min(cfg) -> List[tuple]:
    out = []
    for ev in (cfg.contingencies or []):
        t = getattr(ev, "effective_time_s", None)
        if t is None:
            t = float(getattr(ev, "minute", 0.0)) * 60.0
        label = f"{getattr(ev, 'element_type', '?')} {getattr(ev, 'action', '')}"
        out.append((float(t) / 60.0, label))
    return sorted(out)


def _mark_events(ax, events, tmax_min) -> None:
    for t, _lab in events:
        if 0 <= t <= tmax_min:
            ax.axvline(t, color="0.6", lw=0.7, ls=":", zorder=0)


def make_figures(runs: Dict[str, list], cfg, out_dir: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(runs)
    events = _event_times_min(cfg)
    written: List[Path] = []

    def tmin(log):
        return np.array([r.time_s for r in log], float) / 60.0

    v_set = float(getattr(cfg, "v_setpoint_pu", 1.03))
    v_lo = float(getattr(cfg, "v_min_pu", 0.90))
    v_hi = float(getattr(cfg, "v_max_pu", 1.10))

    # ---- 1. TS voltages -------------------------------------------------
    zones = sorted({z for log in runs.values() for r in log for z in r.zone_v_mean})
    fig, axes = plt.subplots(len(names), len(zones), figsize=(5 * len(zones), 3.1 * len(names)),
                             sharex=True, sharey=True, squeeze=False)
    for i, nm in enumerate(names):
        log = runs[nm]; t = tmin(log)
        for j, z in enumerate(zones):
            ax = axes[i][j]
            lo = np.array([r.zone_v_min.get(z, np.nan) for r in log], float)
            me = np.array([r.zone_v_mean.get(z, np.nan) for r in log], float)
            hi = np.array([r.zone_v_max.get(z, np.nan) for r in log], float)
            ax.fill_between(t, lo, hi, alpha=0.25, lw=0, label="min-max")
            ax.plot(t, me, lw=1.1, label="mean")
            ax.axhline(v_set, color="k", lw=0.8, ls="--", label="schedule")
            _mark_events(ax, events, t[-1] if len(t) else 0)
            if i == 0:
                ax.set_title(f"TS zone {z}")
            if j == 0:
                ax.set_ylabel(f"{nm}\nV [pu]", fontsize=9)
            if i == len(names) - 1:
                ax.set_xlabel("t [min]")
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Transmission voltages  (corridor {v_lo}-{v_hi} pu, schedule {v_set} pu)")
    fig.tight_layout()
    p = out_dir / "ts_voltages.png"; fig.savefig(p, dpi=140); plt.close(fig)
    written.append(p)

    # ---- 2. DS voltages -------------------------------------------------
    areas = sorted({a for log in runs.values() for r in log for a in r.dso_group_v_mean_pu})
    if areas:
        fig, axes = plt.subplots(len(names), len(areas),
                                 figsize=(4.2 * len(areas), 3.1 * len(names)),
                                 sharex=True, sharey=True, squeeze=False)
        for i, nm in enumerate(names):
            log = runs[nm]; t = tmin(log)
            for j, a in enumerate(areas):
                ax = axes[i][j]
                lo = np.array([r.dso_group_v_min_pu.get(a, np.nan) for r in log], float)
                me = np.array([r.dso_group_v_mean_pu.get(a, np.nan) for r in log], float)
                hi = np.array([r.dso_group_v_max_pu.get(a, np.nan) for r in log], float)
                ax.fill_between(t, lo, hi, alpha=0.25, lw=0, color="C2")
                ax.plot(t, me, lw=1.1, color="C2")
                ax.axhline(v_set, color="k", lw=0.8, ls="--")
                _mark_events(ax, events, t[-1] if len(t) else 0)
                if i == 0:
                    ax.set_title(a)
                if j == 0:
                    ax.set_ylabel(f"{nm}\nV [pu]", fontsize=9)
                if i == len(names) - 1:
                    ax.set_xlabel("t [min]")
        fig.suptitle("Distribution (HV) voltages, per area: min-max envelope and mean")
        fig.tight_layout()
        p = out_dir / "ds_voltages.png"; fig.savefig(p, dpi=140); plt.close(fig)
        written.append(p)

    # ---- 3. Interface Q tracking ---------------------------------------
    if areas:
        fig, axes = plt.subplots(len(names), len(areas),
                                 figsize=(4.2 * len(areas), 3.1 * len(names)),
                                 sharex=True, squeeze=False)
        for i, nm in enumerate(names):
            log = runs[nm]; t = tmin(log)
            for j, a in enumerate(areas):
                ax = axes[i][j]
                qs = np.array([r.dso_q_set_mvar.get(a, np.nan)
                               if r.dso_q_set_mvar.get(a) is not None else np.nan
                               for r in log], float)
                # Held between supervisory dispatches, like the PCC block.
                last = np.nan
                for _i in range(qs.size):
                    if np.isfinite(qs[_i]):
                        last = qs[_i]
                    elif np.isfinite(last):
                        qs[_i] = last
                qa = np.array([r.dso_q_actual_mvar.get(a, np.nan)
                               if r.dso_q_actual_mvar.get(a) is not None else np.nan
                               for r in log], float)
                ax.plot(t, qs, lw=1.2, ls="--", color="k", label="Q_set (TSO)")
                ax.plot(t, qa, lw=1.1, color="C3", label="Q delivered")
                _mark_events(ax, events, t[-1] if len(t) else 0)
                if i == 0:
                    ax.set_title(a)
                if j == 0:
                    ax.set_ylabel(f"{nm}\nQ [Mvar]", fontsize=9)
                if i == len(names) - 1:
                    ax.set_xlabel("t [min]")
        axes[0][0].legend(fontsize=7)
        fig.suptitle("Interface reactive power: setpoint vs delivered")
        fig.tight_layout()
        p = out_dir / "interface_q.png"; fig.savefig(p, dpi=140); plt.close(fig)
        written.append(p)

    # ---- 4. Actuators ---------------------------------------------------
    def _ffill(m):
        """Hold the last value forward over gaps.

        Setpoints are written to the record only on the tick that dispatches
        them -- ``zone_q_pcc_set`` and ``dso_q_set_mvar`` appear in 6 of 45
        records on a 15 min run, because the supervisory layer fires every
        180 s against a 20 s plant step.  Plotting them raw draws isolated
        points with nothing between, which reads as "no setpoint" rather than
        "the setpoint is being held".  A held setpoint IS the physical
        quantity, so it is drawn as one.
        """
        out = m.copy()
        for c in range(out.shape[1]):
            col = out[:, c]
            last = np.nan
            for i in range(col.size):
                if np.isfinite(col[i]):
                    last = col[i]
                elif np.isfinite(last):
                    col[i] = last
        return out

    def stack(log, attr, keys, ffill=False):
        """Time series matrix for a dict-of-array record field."""
        out = []
        for r in log:
            d = getattr(r, attr, {}) or {}
            vals = []
            for k in keys:
                v = d.get(k)
                if v is None:
                    continue
                arr = np.atleast_1d(np.asarray(v, float))
                if arr.size == 0:
                    continue
                vals.extend(arr.tolist())
            out.append(vals)
        n = max((len(v) for v in out), default=0)
        if n == 0:
            return None
        m = np.full((len(out), n), np.nan)
        for i, v in enumerate(out):
            m[i, :len(v)] = v
        return _ffill(m) if ffill else m

    # DSO tap positions are keyed per transformer ("DSO_1|trafo_0"), not per
    # area, and the DSO DER block is recorded as a per-group aggregate rather
    # than under ``dso_q_der`` (which is never populated on this path).
    tap_keys = sorted({k for log in runs.values() for r in log
                       for k in (r.dso_trafo_tap_pos or {})})

    panels = [
        ("TS DER Q [Mvar]",      "zone_q_der",            zones, False),
        ("PCC setpoint [Mvar]",  "zone_q_pcc_set",        zones, True),
        ("AVR setpoint [pu]",    "zone_v_gen",            zones, False),
        ("TS OLTC tap",          "zone_oltc_taps",        zones, False),
        ("TS shunt step",        "zone_tso_shunt_states", zones, False),
        ("DSO DER Q [Mvar]",     "dso_group_q_der_mvar",  areas, False),
    ]
    fig, axes = plt.subplots(len(names), len(panels),
                             figsize=(3.5 * len(panels), 3.0 * len(names)),
                             sharex=True, squeeze=False)
    for i, nm in enumerate(names):
        log = runs[nm]; t = tmin(log)
        for j, (label, attr, keys, ff) in enumerate(panels):
            ax = axes[i][j]
            m = stack(log, attr, keys, ffill=ff)
            if m is not None:
                step = "post" if ("tap" in label or "shunt" in label) else None
                for c in range(m.shape[1]):
                    if step:
                        ax.step(t, m[:, c], where="post", lw=0.9)
                    else:
                        ax.plot(t, m[:, c], lw=0.9)
            else:
                ax.text(0.5, 0.5, "none", ha="center", va="center",
                        transform=ax.transAxes, color="0.6", fontsize=9)
            _mark_events(ax, events, t[-1] if len(t) else 0)
            if i == 0:
                ax.set_title(label, fontsize=9)
            if j == 0:
                ax.set_ylabel(nm, fontsize=9)
            if i == len(names) - 1:
                ax.set_xlabel("t [min]")
    fig.suptitle("Actuators, all classes (taps and shunts drawn as steps)")
    fig.tight_layout()
    p = out_dir / "actuators.png"; fig.savefig(p, dpi=140); plt.close(fig)
    written.append(p)

    # ---- 5. DSO tap positions (integers, own figure) ---------------------
    if tap_keys:
        fig, axes = plt.subplots(len(names), 1,
                                 figsize=(10, 2.8 * len(names)),
                                 sharex=True, sharey=True, squeeze=False)
        for i, nm in enumerate(names):
            ax = axes[i][0]
            log = runs[nm]; t = tmin(log)
            moved = 0
            for k in tap_keys:
                v = np.array([(r.dso_trafo_tap_pos or {}).get(k, np.nan)
                              for r in log], float)
                if not np.isfinite(v).any():
                    continue
                if np.nanmax(v) != np.nanmin(v):
                    moved += 1
                    ax.step(t, v, where="post", lw=1.2, label=k)
                else:
                    ax.step(t, v, where="post", lw=0.6, color="0.8", zorder=0)
            _mark_events(ax, events, t[-1] if len(t) else 0)
            ax.set_ylabel(f"{nm}\ntap position", fontsize=9)
            ax.set_title(f"{moved} of {len(tap_keys)} coupling transformers move",
                         fontsize=9)
            if moved:
                ax.legend(fontsize=6, ncol=4)
            if i == len(names) - 1:
                ax.set_xlabel("t [min]")
        fig.suptitle("Distribution coupling-transformer taps "
                     "(grey = never moves in this run)")
        fig.tight_layout()
        p = out_dir / "ds_taps.png"; fig.savefig(p, dpi=140); plt.close(fig)
        written.append(p)

    return written


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="ch_9_3_compare_runs")
    p.add_argument("--paramsets", nargs="+", required=True)
    p.add_argument("--hours", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results" / "THESIS_ch9_3_compare")
    p.add_argument("--force", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / f"{args.hours:g}h"
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs = [{"paramset": ps, "hours": args.hours, "force": args.force,
             "out": str(run_dir / f"records_{ps}.pkl")} for ps in args.paramsets]

    if not args.plot_only:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed({ex.submit(_run_one, j): j for j in jobs}):
                r = fut.result()
                if r["status"] == "ok":
                    print(f"[compare] {r['paramset']}: {r['records']} records "
                          f"({r['wall_s'] / 60:.1f} min)", flush=True)
                elif r["status"] == "cached":
                    print(f"[compare] {r['paramset']}: cached", flush=True)
                else:
                    print(f"[compare] {r['paramset']}: ERROR {r['error']}",
                          flush=True)
                    print(r.get("traceback", ""))
        print(f"[compare] simulations done in "
              f"{(time.perf_counter() - t0) / 60:.1f} min")

    runs: Dict[str, list] = {}
    for ps in args.paramsets:
        f = run_dir / f"records_{ps}.pkl"
        if not f.exists():
            print(f"[compare] {ps}: no records, skipped in the plots")
            continue
        with f.open("rb") as fh:
            runs[ps] = pickle.load(fh)
    if not runs:
        raise SystemExit("no runs to plot")

    from configs.paramsets import apply_paramset
    from experiments.run_multi_system_ofo import make_config
    import dataclasses
    cfg = dataclasses.replace(
        apply_paramset(make_config(), args.paramsets[0], verbose=False),
        n_total_s=args.hours * 3600.0)

    figs = make_figures(runs, cfg, run_dir / "figures")
    for f in figs:
        print(f"[compare] wrote {f}")

    (run_dir / "run_meta.json").write_text(json.dumps({
        "stamp": stamp, "paramsets": args.paramsets, "hours": args.hours,
        "records": {k: len(v) for k, v in runs.items()},
        "argv": sys.argv,
    }, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
