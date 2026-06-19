import os, sys, glob, importlib, subprocess, traceback
from datetime import datetime

import matplotlib as mpl
# Render the final comparison figure in an interactive window.  Qt5Agg needs
# PyQt5; if it is missing we fall back silently to the active backend.  Must run
# BEFORE pyplot is imported anywhere -- i.e. before importing the
# experiments.003_* modules below, which import matplotlib.pyplot.
if not os.environ.get("MPLBACKEND"):
    os.environ["QT_API"] = "pyqt5"
    try:
        mpl.use("Qt5Agg", force=True)
    except (ImportError, ValueError):
        pass  # PyQt5 missing or backend unavailable — fall back silently.
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root); sys.path.insert(0, _root)
exp = importlib.import_module("experiments.003_S_DSO_CIGRE_2026")
ana = importlib.import_module("experiments.003_analysis")
ana.plot_comparison = lambda *a, **k: None            # silence per-run pop-ups
RES = os.path.join(_root, "results", "003_cigre_2026")

MODES  = {"identity": "Constant H", "kalman": "Kalman"}#, "ann": "ANN"}
exp.H_INIT_BIAS_STD, exp.H_INIT_BIAS_SEED = 0.5, 64
exp.H_PREDICTOR_ROWS, exp.FROZEN_OP_POINT = "all", False
# Enable the NIS topology detector (Gap 5) so the Kalman re-opens P on the line
# trip scheduled below.  Needs the recalibrated R (run the MC regen first) so the
# NIS is χ²-calibrated; set False for a keep-alive-only baseline (no event reset).
exp.KALMAN_NIS_DETECT_ENABLED = True

_orig = exp.make_config                                # <-- current name, NOT make_base_config
def _cfg(_o=_orig):
    cfg = _o()
    cfg.n_total_s, cfg.dso_period_s = 600*60.0, 20.0   # 600 min per mode (long settling window)
    cfg.start_time = datetime(2016,  9,  7,  8, 0)
    # Single DSO_2 line contingency (110 kV island = lines 46-56).  Timed past the
    # detector settle-hold (KALMAN_NIS_COOLDOWN_STEPS=30) and the cold-start convergence
    # (the floored R lowers the gain → slower settle) so the trip lands on a low-NIS
    # baseline.  The second trip (line 54 @ 240) was REMOVED: the N-2 state (49+54 both
    # out) destabilised the controller power-flow ~min 271 (state-dependent, not a static
    # bad point) and aborted the run — separate N-2 robustness issue, out of scope here.
    #   min 120 : trip line 49 (bus 57->58) -- ~40% H step vs nominal (largest single mover).
    # No restore, so the topology stays changed and the re-identification shows.
    cfg.contingencies = [
        exp.ContingencyEvent(minute=120, element_type="line", element_index=49, action="trip"),
    ]
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
# NB: the per-run live-plot windows are intentionally left OPEN here -- no
# plt.close("all") -- so they remain visible after the runs, alongside the
# comparison figure that plot_multi_comparison adds as a new figure.
ana.plot_multi_comparison(runs, save_path=png)
print("done ->", png)

# Show the comparison figure at the end.  plot_multi_comparison leaves the
# figure open when save_path is set, so a blocking show pops it up in the
# Qt5Agg window.  If the active backend is non-interactive (PyQt5 missing ->
# Agg fallback, or MPLBACKEND forced headless), open the saved PNG instead.
if mpl.get_backend().lower() in ("agg", "pdf", "ps", "svg", "template", "cairo"):
    try:
        os.startfile(png)                               # Windows default image viewer
    except AttributeError:                              # macOS / Linux fallback
        subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", png], check=False)
else:
    plt.show(block=True)