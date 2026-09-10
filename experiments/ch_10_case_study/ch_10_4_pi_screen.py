"""
ch_10_4_pi_screen.py
====================

Full-horizon screen of the S1 PI gains, with and without the pilot-voltage
priority safeguard.

Why this exists: the gains were screened on a 42-minute case, which is long
enough to see the first contingency and nothing else.  The 360-minute horizon
carries six, including a line trip at 210 min whose response the short case
never reaches.  A gain chosen on the short case is a guess about the long one.

Grid: k_p in {0.1, 0.2, 0.4} x safeguard {on, off}.  Both loops take the same
k_p, matching the "guard plus both Kp" case that screened best.  The two
k_p = 0 corners are NOT re-run -- they exist already:
    I-only + guard    the ladder rerun of 2026-09-04
    I-only, no guard  results/.../\_archive_S1_classical/S1_Ionly_noguard.pkl

Reported per case: horizon RMS over each area's reference set and on the pilot
nodes, plus the peak error in the 30 min after the generator trip (t = 30) and
after the line trip (t = 210) -- the two events the safeguard was built for.

Run:
    python -m experiments.ch_10_case_study.ch_10_4_pi_screen
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiments.paths import RESULTS_ROOT

RESULT_DIR = Path(RESULTS_ROOT) / "THESIS_ch10_variants_single_run"
OUT_DIR = Path(RESULTS_ROOT) / "ch10_s1_pi_screen"

#: k_p = 0.2 is taken from the 42-minute screen and NOT re-tuned.  S1 is a
#: reference, not the contribution: it needs to be a fair opponent, not an
#: optimal one, and a gain sweep on the baseline would spend a lot of machine
#: time on a number the thesis makes no claim about.  The one thing the short
#: case could not settle -- whether the safeguard is still wanted once the
#: proportional term is present -- is settled here, on the full horizon.
K_P_GRID = [0.0, 0.2]
GUARD_GRID = [True, False]
PILOT_BUSES = {1: 25, 2: 7, 3: 20}
V_SET = 1.03
#: (label, t_start, t_end) windows in which a peak is taken, minutes.
PEAK_WINDOWS = [("gen trip", 30, 60), ("line trip", 210, 240)]


def _rms(v) -> float:
    a = np.asarray([x for x in v if x is not None], float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")


def evaluate(log) -> Dict[str, float]:
    """Horizon metrics plus the two post-contingency peaks."""
    ev = [_rms([r.zone_v_rms_err_pu.get(z, np.nan) for r in log
                if getattr(r, "zone_v_rms_err_pu", None)]) for z in (1, 2, 3)]
    ep = [_rms([float(r.bus_vm_pu[PILOT_BUSES[z]]) - V_SET for r in log
                if getattr(r, "bus_vm_pu", None) and PILOT_BUSES[z] in r.bus_vm_pu])
          for z in (1, 2, 3)]

    t = np.array([r.time_s / 60.0 for r in log], float)
    agg = np.array([
        _rms([r.zone_v_rms_err_pu.get(z, np.nan) for z in (1, 2, 3)])
        if getattr(r, "zone_v_rms_err_pu", None) else np.nan
        for r in log], float)

    out = {"e_v": _rms(ev), "e_p": _rms(ep)}
    horizon = float(np.nanmax(t)) if t.size else 0.0
    for label, a, b in PEAK_WINDOWS:
        if a > horizon:          # event never fires on this horizon
            out[f"peak_{label.replace(' ', '_')}"] = float("nan")
            continue
        m = (t >= a) & (t <= min(b, horizon)) & np.isfinite(agg)
        out[f"peak_{label.replace(' ', '_')}"] = float(agg[m].max()) if m.any() else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=360)
    args = ap.parse_args()

    import importlib.util as u
    here = Path(__file__).parent
    spec = u.spec_from_file_location("lad", here / "ch_10_1_variant_ladder.py")
    lad = u.module_from_spec(spec)
    spec.loader.exec_module(lad)
    lad.HORIZON_MIN = args.minutes

    from experiments.runners.multi_tso_dso import run_multi_tso_dso

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}

    for guard in GUARD_GRID:
        for k_p in K_P_GRID:
            tag = f"kp{k_p:g}_{'guard' if guard else 'noguard'}"
            d = OUT_DIR / tag
            d.mkdir(parents=True, exist_ok=True)
            cfg = lad.make_thesis_config()
            for k, v in lad.VARIANTS["S1"].items():
                setattr(cfg, k, v)
            cfg.svr_k_p_rvr = k_p
            cfg.svr_k_p_rpr = k_p
            cfg.svr_rpr_voltage_priority = bool(guard)
            cfg.verbose = 1
            cfg.result_dir = str(d)
            print("\n" + "=" * 70)
            print(f"  S1  k_p={k_p:g}  safeguard={'on' if guard else 'off'}")
            print("=" * 70, flush=True)
            try:
                log = run_multi_tso_dso(cfg)
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
                results[tag] = None
                continue
            with open(d / "log.pkl", "wb") as f:
                pickle.dump(log, f)
            results[tag] = evaluate(log)
            print(f"  [{tag}] {len(log)} records  {results[tag]}", flush=True)

    # The k_p = 0 corners are run HERE rather than read from the ladder: those
    # logs are a 360-minute horizon and a peak or an RMS taken over a different
    # window is not comparable to one taken over this one.  Four corners, one
    # horizon, or the table means nothing.

    (OUT_DIR / "screen.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'case':>16}{'e_v':>10}{'e_v,p':>10}{'peak gen':>11}{'peak line':>11}")
    print("-" * 78)
    order = [f"kp{k:g}_{g}" for k in K_P_GRID for g in ("noguard", "guard")]
    for tag in order:
        r = results.get(tag)
        if not r:
            continue
        print(f"{tag:>16}{r['e_v']:>10.5f}{r['e_p']:>10.5f}"
              f"{r['peak_gen_trip']:>11.5f}{r['peak_line_trip']:>11.5f}")
    print(f"\nwritten: {OUT_DIR/'screen.json'}")


if __name__ == "__main__":
    main()
