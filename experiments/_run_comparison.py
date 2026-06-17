import os, sys, glob, importlib, subprocess, traceback
from datetime import datetime
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root); sys.path.insert(0, _root)
exp = importlib.import_module("experiments.003_S_DSO_CIGRE_2026")
ana = importlib.import_module("experiments.003_analysis")
ana.plot_comparison = lambda *a, **k: None            # silence per-run pop-ups
RES = os.path.join(_root, "results", "003_cigre_2026")

MODES  = {"identity": "Constant H", "kalman": "Kalman", "ann": "ANN"}
exp.H_INIT_BIAS_STD, exp.H_INIT_BIAS_SEED = 0.10, 0
exp.H_PREDICTOR_ROWS, exp.FROZEN_OP_POINT = "q_trafo+v", False

_orig = exp.make_config                                # <-- current name, NOT make_base_config
def _cfg(_o=_orig):
    cfg = _o()
    cfg.n_total_s, cfg.dso_period_s = 2*3600.0, 60.0   # 120 steps (~5 min/mode)
    cfg.start_time, cfg.contingencies = datetime(2016, 9, 7, 6, 0), []
    return cfg
exp.make_config = _cfg

runs = []
for mode, label in MODES.items():
    exp.H_PREDICTOR_MODE = mode
    before = set(glob.glob(os.path.join(RES, "*.pkl")))
    try:
        exp.run()
    except Exception:
        print(f"[cmp] mode={mode} FAILED (skipping):\n{traceback.format_exc()}")
        continue
    new = sorted(set(glob.glob(os.path.join(RES, "*.pkl"))) - before, key=os.path.getmtime)
    if not new: continue
    pkl  = new[-1]
    side = os.path.join(RES, os.path.basename(pkl).split("_")[0] + "_dso2_ctrl.npz")
    runs.append((pkl, side if os.path.exists(side) else None, label))

if not runs:
    print("[cmp] no successful runs — nothing to plot."); sys.exit(1)

ts  = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
png = os.path.join(RES, f"comparison_changing_{ts}.png")
ana.plot_multi_comparison(runs, save_path=png)
print("done ->", png)

# pop the figure up in a window
try:
    os.startfile(png)                                   # Windows default image viewer
except AttributeError:                                  # macOS / Linux fallback
    subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", png], check=False)