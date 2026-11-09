"""Do machines reach their Q capability bound in v5's windows?  (2026-11-09)

Fix A (sn_mva slip in the Q_gen rows) changes only the TSO's soft constraint on
machine Q (``rho_q_gen`` / ``g_z_q_gen``), which acts when a machine's Q is
predicted outside the capability band [Q_min, Q_max] the controller draws from
the PQ curve each iteration.  This script re-runs one v5 candidate on W1-W6
with the AS-BUILT v5 code and logs, once per TS iteration and per machine:

* measured Q (``net.res_gen``) -- logging only, the controllers are unchanged;
* the controller's capability band (``TSOController._get_output_limits``, Q_gen rows);
* the plant's Q limits (``net.gen.min_q_mvar / max_q_mvar``, enforced by the plant).

Window setup copied from ``ch_9_3_stage2_screen._run_one_window`` (paramset, frozen
network, config_set, evented 12-h condition set).

Run from the as-built checkout:
    cd F:\\qOFO_v5chk
    python <this file> --root F:\\qOFO_v5chk --out <dir> --workers 6
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PARAMSET = "stage1_v5_nr_2026_11_pq10"
CONFIG_SET = {"sts_coupler_hold_after_bank": 1}
KEY = "158fdfc8e812"          # II-b round-1 best of run 0002 (leader on 2026-11-09 08:00)
WEIGHTS = {"g_v": 40000.0, "g_q": 0.4, "dso_g_v": 10000.0, "g_w_der": 0.02750069849003403,
           "g_w_pcc": 0.1806370308851905, "g_w_gen": 371939.18359590974,
           "g_w_tso_oltc": 9.960507526267806, "g_w_tso_shunt": 20.03096026843382,
           "g_w_dso_der": 1.5280510783923393, "g_w_dso_oltc": 11.342714141101698}
WINDOWS = ("W1", "W2", "W3", "W4", "W5", "W6")


def run_window(root: str, window: str, out: str, n_total_s: float = 0.0) -> dict:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    sys.path.insert(0, root)
    os.chdir(root)
    import warnings
    warnings.filterwarnings("ignore")
    import contextlib
    import io

    from configs.paramsets import apply_paramset
    from experiments.run_multi_system_ofo import make_config
    from experiments.ch_09_parameter_selection._ch9_selected_design import (
        FROZEN_DSO_DER_SCALE, FROZEN_DSO_LOAD_P_SCALE)
    from tuning.parameters import apply_to_config
    from tuning.scenarios_mc_v2 import condition_set_evented
    from tuning._sim_loader import get_run_multi_tso_dso
    import experiments.runners.multi_tso_dso as runner

    base = apply_to_config(apply_paramset(make_config(), PARAMSET, verbose=False), WEIGHTS)
    base = dataclasses.replace(base, dso_der_scale=dict(FROZEN_DSO_DER_SCALE),
                               dso_load_p_scale=dict(FROZEN_DSO_LOAD_P_SCALE))
    base = dataclasses.replace(base, **CONFIG_SET)
    sc = next(s for s in condition_set_evented(hours=12.0) if s.name == window)
    cfg = sc.overlay_on(base)
    if n_total_s:
        cfg = dataclasses.replace(cfg, n_total_s=float(n_total_s))
    cfg = dataclasses.replace(cfg, **{n: False for n in (
        "live_plot_controller", "live_plot_cascade", "live_plot_overview", "live_plot_sbxv")
        if getattr(cfg, n, False)})

    state = {}
    trace = []                       # (step, zone, gen, q, cap_lo, cap_hi, plant_lo, plant_hi, in_service)

    def pre(s):
        state["net"] = s["net"]
        state["tso"] = s["tso_controllers"]
        return None                  # continue into the main loop

    def hook(step, log):
        if step % 9:                 # once per TS iteration (T_TS / T_STS)
            return
        net = state["net"]
        for z, ctrl in state["tso"].items():
            gens = list(ctrl.config.gen_indices)
            if not gens:
                continue
            try:
                lo, hi = ctrl._get_output_limits()
            except Exception:        # no measurement cached yet
                continue
            lo, hi = np.asarray(lo)[-len(gens):], np.asarray(hi)[-len(gens):]
            for k, g in enumerate(gens):
                trace.append((step, int(z) if str(z).isdigit() else str(z), int(g),
                              float(net.res_gen.at[g, "q_mvar"]), float(lo[k]), float(hi[k]),
                              float(net.gen.at[g, "min_q_mvar"]), float(net.gen.at[g, "max_q_mvar"]),
                              bool(net.gen.at[g, "in_service"])))

    runner.STEP_HOOK = hook
    t0 = time.perf_counter()
    err = ""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            get_run_multi_tso_dso()(cfg, pre_loop_hook=pre)
    except Exception as exc:          # report, do not hide
        err = f"{type(exc).__name__}: {exc}"
    finally:
        runner.STEP_HOOK = None
    wall = time.perf_counter() - t0

    Path(out).mkdir(parents=True, exist_ok=True)
    cols = ("step", "zone", "gen", "q", "cap_lo", "cap_hi", "plant_lo", "plant_hi", "in_service")
    with open(Path(out) / f"trace_{window}.json", "w") as fh:
        json.dump({"cols": cols, "rows": trace}, fh)
    return {"window": window, "wall_min": wall / 60.0, "error": err, "n_rows": len(trace)}


def summarise(out: Path) -> str:
    lines = [f"candidate {KEY}, paramset {PARAMSET}, config_set {CONFIG_SET}; as-built v5 code",
             "per window and machine (in-service TS iterations only):",
             f"{'win':>3} {'zone':>4} {'gen':>3} {'n':>5} {'Q min..max [Mvar]':>20} "
             f"{'cap band (median)':>20} {'plant lim':>14} {'min margin cap':>15} "
             f"{'#out cap':>8} {'#<10 Mvar':>9} {'#at plant':>9}"]
    for w in WINDOWS:
        f = out / f"trace_{w}.json"
        if not f.exists():
            lines.append(f"{w}: no trace")
            continue
        d = json.load(open(f))
        rows = [dict(zip(d["cols"], r)) for r in d["rows"] if r[8]]
        for g in sorted({(r["zone"], r["gen"]) for r in rows}, key=str):
            rr = [r for r in rows if (r["zone"], r["gen"]) == g]
            q = np.array([r["q"] for r in rr])
            lo = np.array([r["cap_lo"] for r in rr])
            hi = np.array([r["cap_hi"] for r in rr])
            plo = np.array([r["plant_lo"] for r in rr])
            phi = np.array([r["plant_hi"] for r in rr])
            margin = np.minimum(hi - q, q - lo)
            out_cap = int(np.sum((q > hi + 1e-6) | (q < lo - 1e-6)))
            near = int(np.sum(margin < 10.0))
            at_plant = int(np.sum((q >= phi - 0.5) | (q <= plo + 0.5)))
            lines.append(f"{w:>3} {str(g[0]):>4} {g[1]:>3} {len(rr):>5} {q.min():>9.1f}..{q.max():>8.1f} "
                         f"{np.median(lo):>9.1f}..{np.median(hi):>8.1f} {plo[0]:>6.0f}..{phi[0]:>5.0f} "
                         f"{margin.min():>15.1f} {out_cap:>8d} {near:>9d} {at_plant:>9d}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="W1 only, 30 min of plant time")
    a = ap.parse_args()
    out = Path(a.out)
    if a.smoke:
        print(json.dumps(run_window(a.root, "W1", str(out), n_total_s=1800.0)))
        return
    if not a.summary_only:
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(run_window, a.root, w, str(out)) for w in WINDOWS]
            for f in as_completed(futs):
                print(json.dumps(f.result()), flush=True)
    print(summarise(out))


if __name__ == "__main__":
    main()
