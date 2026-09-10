"""
ch_10_3_export_tikz_data.py
===========================

Export the Ch 10 ladder time series as pgfplots ``.dat`` tables, for the
hand-authored TikZ figures in ``graphics/Ch10``.

The thesis draws its own figures: this writes DATA only, never TikZ.  That is
the house convention (see ``graphics/Ch9/data/``) and it is the right split --
styling belongs with the document, numbers belong with the run that produced
them.

Writes into ``graphics/Ch10/data/<date>_ch10_ladder/``:

    voltage_tracking.dat   t + one column per variant, the system-wide RMS
                           voltage-tracking error [pu]
    iface_<group>.dat      t + set/act column pair per interface of that DSO
                           group, under the proposed cascade [Mvar]
    metrics.dat            one row per variant, the horizon-aggregated metrics
    README.md              provenance

Run:
    python -m experiments.ch_10_case_study.ch_10_3_export_tikz_data
"""

from __future__ import annotations

import argparse
import pickle
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiments.helpers.comparison_metrics import (
    q_iface_per_trafo,
    voltage_rms_err_all,
)
from experiments.paths import RESULTS_ROOT

RESULT_DIR = Path(RESULTS_ROOT) / "THESIS_ch10_variants_single_run"
THESIS_DATA = Path(
    r"C:\Users\Manuel Schwenke\Desktop\Daten\01_Forschung\12_Dissertation"
    r"\latex_diss_ms\graphics\Ch10\data"
)

LADDER = ["L1", "L2", "S1", "O1", "O2", "O3", "M"]
PROPOSED = "O3"
PILOT_BUSES = {1: 25, 2: 7, 3: 20}
V_SET = 1.03

#: Keep every nth sample.  1 = full resolution.  The voltage figure stays at 1:
#: the classical scheme's contingency spikes are two or three samples wide and
#: decimation removes exactly the feature the figure is for.  The interface
#: figure carries 24 traces and is thinned.
STRIDE_V = 1
STRIDE_IFACE = 2


def load(run_dir: Path) -> Dict[str, List]:
    out = {}
    for n in LADDER:
        p = run_dir / n / "log.pkl"
        if p.exists():
            with open(p, "rb") as f:
                out[n] = pickle.load(f)
    return out


def _write(path: Path, header: List[str], cols: List[np.ndarray]) -> None:
    m = np.column_stack(cols)
    with path.open("w", encoding="utf-8") as f:
        f.write(" ".join(header) + "\n")
        for row in m:
            f.write(" ".join("nan" if not np.isfinite(v) else f"{v:.6g}"
                             for v in row) + "\n")
    print(f"  {path.name}: {m.shape[0]} rows x {m.shape[1]} cols")


def export_voltage(logs, out: Path) -> None:
    t = None
    cols, header = [], ["t"]
    for n in LADDER:
        if n not in logs:
            continue
        d = voltage_rms_err_all(logs[n], v_set=V_SET)
        tt = np.asarray(d["t_min"], float)[::STRIDE_V]
        yy = np.asarray(d["rms_err_pu"], float)[::STRIDE_V]
        if t is None:
            t = tt
            cols.append(t)
        cols.append(yy)
        header.append(n)
    _write(out / "voltage_tracking.dat", header, cols)


def export_iface(logs, out: Path) -> None:
    if PROPOSED not in logs:
        print(f"  !! {PROPOSED} missing; no interface data")
        return
    d = q_iface_per_trafo(logs[PROPOSED])
    t = np.asarray(d["t_min"], float)[::STRIDE_IFACE]
    for g in d["groups"]:
        header, cols = ["t"], [t]
        for k, key in enumerate(d["trafos"][g], start=1):
            header += [f"set{k}", f"act{k}"]
            cols.append(np.asarray(d["set_mvar"][key], float)[::STRIDE_IFACE])
            cols.append(np.asarray(d["actual_mvar"][key], float)[::STRIDE_IFACE])
        _write(out / f"iface_{g}.dat", header, cols)


def _rms(v) -> float:
    a = np.asarray([x for x in v if x is not None], float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")


def export_metrics(logs, out: Path) -> Dict[str, Dict[str, float]]:
    rows, m = [], {}
    for i, n in enumerate(LADDER):
        if n not in logs:
            continue
        lg = logs[n]
        ev = [_rms([r.zone_v_rms_err_pu.get(z, np.nan) for r in lg
                    if getattr(r, "zone_v_rms_err_pu", None)]) for z in (1, 2, 3)]
        ep = [_rms([float(r.bus_vm_pu[PILOT_BUSES[z]]) - V_SET for r in lg
                    if getattr(r, "bus_vm_pu", None)
                    and PILOT_BUSES[z] in r.bus_vm_pu]) for z in (1, 2, 3)]
        vmin = min(float(r.zone_v_min[z]) for r in lg
                   for z in (1, 2, 3) if z in getattr(r, "zone_v_min", {}))
        vmax = max(float(r.zone_v_max[z]) for r in lg
                   for z in (1, 2, 3) if z in getattr(r, "zone_v_max", {}))
        taps, prev = 0, {}
        for r in lg:
            for z, tp in (getattr(r, "zone_oltc_taps", {}) or {}).items():
                tp = np.asarray(tp, float)
                if z in prev and prev[z].shape == tp.shape:
                    taps += int(np.sum(np.abs(tp - prev[z]) > 0.5))
                prev[z] = tp
        m[n] = dict(e_v=_rms(ev), e_p=_rms(ep), v_min=vmin, v_max=vmax, n_sw=taps)
        rows.append((i, n, m[n]))

    p = out / "metrics.dat"
    with p.open("w", encoding="utf-8") as f:
        f.write("idx variant e_v e_p v_min v_max n_sw\n")
        for i, n, d in rows:
            f.write(f"{i} {n} {d['e_v']:.6f} {d['e_p']:.6f} "
                    f"{d['v_min']:.4f} {d['v_max']:.4f} {d['n_sw']}\n")
    print(f"  metrics.dat: {len(rows)} rows")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(RESULT_DIR / "g_w_gen_1e9"))
    ap.add_argument("--tag", default=f"{date.today().isoformat()}_ch10_ladder")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = THESIS_DATA / args.tag
    out.mkdir(parents=True, exist_ok=True)

    logs = load(run_dir)
    print(f"read {len(logs)} variants from {run_dir}")
    export_voltage(logs, out)
    export_iface(logs, out)
    m = export_metrics(logs, out)

    (out / "README.md").write_text(
        f"# Ch 10 variant ladder -- pgfplots data\n\n"
        f"Exported {date.today().isoformat()} by "
        f"`experiments/ch_10_case_study/ch_10_3_export_tikz_data.py`.\n\n"
        f"Source run: `{run_dir}`\n\n"
        f"Validated weight set (`experiments/run_multi_system_ofo.make_config`),\n"
        f"scenario rural_700, 360 min at dt = 20 s, all six contingencies,\n"
        f"**one run per variant** -- no confidence statement is supported.\n\n"
        f"Full provenance, including the five configuration defects repaired\n"
        f"during the campaign, is in the README beside the logs:\n"
        f"`results/THESIS_ch10_variants_single_run/README.md`.\n\n"
        f"## Files\n\n"
        f"- `voltage_tracking.dat` -- `t` [min] + one column per variant, the\n"
        f"  system-wide RMS voltage-tracking error [pu]. Full resolution\n"
        f"  (stride {STRIDE_V}): the classical scheme's contingency spikes are a\n"
        f"  few samples wide and decimation would remove them.\n"
        f"- `iface_DSO_*.dat` -- `t` + `setN`/`actN` per interface of that group\n"
        f"  under the proposed cascade O3 [Mvar]. Stride {STRIDE_IFACE}.\n"
        f"- `metrics.dat` -- horizon-aggregated metrics, one row per variant.\n\n"
        f"## Contingencies (minute)\n\n"
        f"30 gen trip - 120 load connect - 180 gen restore - 210 line trip -\n"
        f"300 line restore - 330 load off\n",
        encoding="utf-8")
    print(f"\nwritten to {out}")
    print(f"\n{'':>4}{'e_v':>10}{'e_v,p':>10}{'v_min':>8}{'v_max':>8}{'N_sw':>7}")
    for n in LADDER:
        if n in m:
            d = m[n]
            print(f"{n:>4}{d['e_v']:>10.5f}{d['e_p']:>10.5f}"
                  f"{d['v_min']:>8.3f}{d['v_max']:>8.3f}{d['n_sw']:>7d}")


if __name__ == "__main__":
    main()
