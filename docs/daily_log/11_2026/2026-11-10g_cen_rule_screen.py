"""Screen (not a run of record): CEN (M) under the rule agreed with the author on 2026-11-10, R2 (E2), new base.

Rule (variants.py / runner of branch res/cen-rule-2026-11-10, no hook):
  1. STS voltage weight gamma * dso_g_v (gamma in {1, 0.5, 0.25}; the one tuned factor);
  2. block-wise loop gain: TS block -> lambda_max_B 1.530, STS block -> design_lambda_max_B_sts 1.40, common factor
     back to joint 1.530 (central_kappa_scope="blocks");
  3. C2: per-switch g_w of the MT taps and banks from the owning TS-OFO's curvature on M's full model
     (central_switch_g_w_rule="owner", definition B).
Selection fixed before the screen: the largest gamma at which M's e_v (run_ref.report) lies below every other
variant's; on R2 that is O3, rerun here on the same code.  Everything else (e_vp, f_V^STS, switching, period-2
cycling) is reported, not a criterion.

    python cen_rule_screen.py <out_root> <label>[,<label>...]     labels: M_g100, M_g050, M_g025, O3
"""
import contextlib
import dataclasses
import datetime as _dt
import json
import os
import pickle
import socket
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

SCENARIO = "E2"
GAMMA = {"M_g100": 1.0, "M_g050": 0.5, "M_g025": 0.25}


def run_one(out, label):
    from experiments.ch_11_case2_multisystem import variants as v
    from experiments.ch_11_case2_multisystem.ref import scenarios as sc
    from experiments.ch_11_case2_multisystem.ref.run_ref import HORIZON_MIN, NOISE_SEED
    from experiments.runners.multi_tso_dso import run_multi_tso_dso
    d = Path(out) / SCENARIO / label
    d.mkdir(parents=True, exist_ok=True)
    cfg = v.resolved("M" if label in GAMMA else label, HORIZON_MIN, sc.events(SCENARIO), NOISE_SEED)
    if label in GAMMA:
        if v.M_STS_G_V_FACTOR != 1.0:
            raise RuntimeError("screen expects M_STS_G_V_FACTOR = 1 in the code")
        cfg = dataclasses.replace(cfg, central_dso_g_v=GAMMA[label] * float(cfg.dso_g_v))
    cfg = dataclasses.replace(cfg, result_dir=str(d), verbose=1)
    if sc.start_time(SCENARIO) is not None:
        cfg = dataclasses.replace(cfg, start_time=sc.start_time(SCENARIO))
    if sc.ramps(SCENARIO):
        cfg = dataclasses.replace(cfg, profile_ramps=tuple(sc.ramps(SCENARIO)))
    (d / "config_repr.txt").write_text(repr(cfg), encoding="utf-8")
    t0 = _dt.datetime.now()
    with open(d / "console.log", "w", encoding="utf-8") as fh, \
            contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
        try:
            log = run_multi_tso_dso(cfg)
            status = "ok"
        except Exception as exc:
            log = list(getattr(exc, "partial_log", None) or [])
            status = f"failed: {type(exc).__name__}: {exc}"
            import traceback
            traceback.print_exc()
    with open(d / "log.pkl", "wb") as fh:
        pickle.dump(log, fh)
    st = {"variant": label, "scenario": SCENARIO, "status": status, "n_records": len(log),
          "gamma": GAMMA.get(label), "central_dso_g_v": cfg.central_dso_g_v, "host": socket.gethostname(),
          "wall_s": (_dt.datetime.now() - t0).total_seconds()}
    (d / "status.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
    return st


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out, labels = sys.argv[1], sys.argv[2].split(",")
    Path(out).mkdir(parents=True, exist_ok=True)
    print(f"{_dt.datetime.now():%H:%M:%S} start {labels} pid {os.getpid()}", flush=True)
    with ProcessPoolExecutor(max_workers=len(labels)) as pool:
        for f in as_completed([pool.submit(run_one, out, s) for s in labels]):
            print(f"{_dt.datetime.now():%H:%M:%S} " + json.dumps(f.result()), flush=True)
