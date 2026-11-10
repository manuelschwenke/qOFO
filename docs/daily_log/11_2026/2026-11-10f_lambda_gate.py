r"""Gate before the reruns (handover 2026-11-10_avr_init_and_tap_relays.md, section Tuning): lambda_max(B) of the
selected weights ch9_selected_nr_2026_11_v5 on the v5 lambda bank (W1-W6 at 12 h + the ch_9_13 grid at 12 h and 3 h,
banks excluded), without re-selecting.

    control: the stored points of F:\qOFO_v5 run 0002 (old start, 1.03 pu terminals) -> must give 1.5301
    new    : the same conditions captured on res/rerun-base-2026-11-10 (terminals 1.00 pu) into this folder

    python lambda_gate.py control | capture <workers> | evaluate
"""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # before numpy; workers inherit it
    os.environ[_k] = "1"
import json
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
OLD = Path(r"F:\qOFO_v5\results\ch_09_parameter_selection\0002_2026-11-08_185554_ch_9_17_procedure_v5")
PARAMSET = "ch9_selected_nr_2026_11_v5"


def _load(d):
    ops = []
    for f in sorted(Path(d).glob("op_*.npz")):
        with f.open("rb") as fh:
            ops.append(pickle.load(fh))
    return ops


def _lam(ops, label):
    from configs.paramsets import load_paramset
    from tuning import stage1_screen as s1
    w = {k: float(v) for k, v in load_paramset(PARAMSET)["fields"].items()}
    rows = s1.screen_weights(ops, w, exclude_classes=("shunt",))
    wr = s1.worst(rows)
    per = defaultdict(float)
    for r in rows:
        per[r.key] = max(per[r.key], r.lambda_max)
    print(f"{label}: {len(ops)} points, lambda_max(B) worst {wr.lambda_max:.4f} at {wr.condition} {wr.key}; "
          f"ceiling {s1.select_ceiling():.4f}")
    print("   per layer (max over points): " + ", ".join(f"{k} {v:.4f}" for k, v in sorted(per.items())))
    return wr.lambda_max, dict(per), {(r.condition, r.key): r.lambda_max for r in rows}


def _design_capture(out):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P
    from experiments.ch_09_parameter_selection import ch_9_17_procedure_v5 as V5
    from tuning import stage1_screen as s1
    from tuning.scenarios_mc_v2 import CONDITIONS
    t0 = time.time()
    s1.load_or_build_operating_points(V5._capture_config(), [(r[0], r[1]) for r in CONDITIONS], Path(out),
                                      verbose=False, horizon_s=3600.0 * P.HOURS)
    return "design", time.time() - t0, None


def capture(workers):
    from experiments.ch_09_parameter_selection import ch_9_17_procedure_v5 as V5
    from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P
    cfg = V5._capture_config()
    print(f"capture config: scenario {cfg.scenario}, gen_vm_init_pu {cfg.gen_vm_init_pu}, "
          f"coupler_relay_tol_pu {cfg.coupler_relay_tol_pu}, HOURS {P.HOURS}", flush=True)
    if cfg.scenario != V5.NETWORK:
        raise RuntimeError(f"scenario {cfg.scenario} != {V5.NETWORK}")
    (HERE / "operating_points").mkdir(exist_ok=True)
    (HERE / "operating_points_lambda").mkdir(exist_ok=True)
    jobs = [(n, s.isoformat(), h, str(HERE / "operating_points_lambda")) for n, s, h in V5.bank_conditions()]
    from experiments.ch_09_parameter_selection import ch_9_14_procedure_v4 as V4
    V4._pin(V4.AFFINITY_MASK)               # 10 logical processors (e5wimi cap); the workers inherit it
    failed = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut_d = ex.submit(_design_capture, str(HERE / "operating_points"))
        for i, (name, dt, err) in enumerate(ex.map(V5._capture_one, jobs), 1):
            if err:
                failed.append((name, err))
            if i % 20 == 0 or err:
                print(f"  {i}/{len(jobs)} {name} {dt:.0f} s {err or ''} (elapsed {time.time() - t0:.0f} s)", flush=True)
        print("design:", fut_d.result(), flush=True)
    (HERE / "capture_failures.json").write_text(json.dumps(failed, indent=1), "utf-8")
    print(f"capture done in {time.time() - t0:.0f} s, {len(failed)} failed", flush=True)


def evaluate():
    lo, po, ro = _lam(_load(OLD / "operating_points") + _load(OLD / "operating_points_lambda"), "OLD start (v5 run 0002)")
    ln, pn, rn = _lam(_load(HERE / "operating_points") + _load(HERE / "operating_points_lambda"), "NEW start (rerun base)")
    common = sorted(set(ro) & set(rn))
    d = [rn[k] - ro[k] for k in common]
    import numpy as np
    d = np.array(d)
    print(f"paired (condition, layer) values: {len(common)}; new - old: mean {d.mean():+.4f}, "
          f"min {d.min():+.4f}, max {d.max():+.4f}; relative change of the worst {ln / lo - 1:+.2%}")
    json.dump({"old_worst": lo, "new_worst": ln, "old_per_layer": po, "new_per_layer": pn,
               "n_paired": len(common)}, open(HERE / "lambda_gate.json", "w"), indent=1)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "control":
        _lam(_load(OLD / "operating_points") + _load(OLD / "operating_points_lambda"), "OLD start (v5 run 0002)")
    elif mode == "capture":
        capture(int(sys.argv[2]))
    else:
        evaluate()
