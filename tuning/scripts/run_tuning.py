"""tuning/scripts/run_tuning.py — canonical BO-tuning invocation.

Calls ``tuning.tune.main()`` with the project's canonical defaults so
the user does not have to remember the long flag list.  Any extra argv
is forwarded to ``tuning.tune``; argparse takes the *last* occurrence of
each flag, so passing e.g. ``--n-trials 5`` after the script name
overrides the default.

Usage::

    # full run
    python -m tuning.scripts.run_tuning

    # smoke test (3 trials, throwaway storage)
    python -m tuning.scripts.run_tuning \\
        --n-trials 3 --n-startup-trials 2 \\
        --study-name smoke \\
        --storage sqlite:///results/tuning/smoke.db \\
        --output /tmp/p.yaml --report /tmp/t.html \\
        --no-progress-bar --no-cache-ceilings

Notes on the 2026-07-31 revision
--------------------------------
* **Paths resolve relative to this file, not the process CWD.**  The previous
  defaults (``configs/baseline_002_ieee39.yaml``,
  ``sqlite:///results/tuning/studies.db``) only resolved when the process was
  started from ``tuning/scripts/`` — which is why the study database sits under
  ``tuning/scripts/results/`` rather than the project ``results/``.  From the
  repository root the baseline path did not exist and the script could not run.
* **Trial budget fixed.**  The previous defaults were ``--n-trials 20
  --n-startup-trials 15``, i.e. **five** TPE-guided trials, while the docstring
  claimed "30 random trials + 120 trials".  With a median scenario-run of ~13 s
  the budget was never the binding constraint.  At the corrected ``dt_s=20`` a
  75-min scenario is 225 steps (~40 s), so a 5-scenario trial is ~3.5 min and
  120 trials is roughly 7 h.
* **Warm start re-enabled.**  ``--no-warm-start-baseline`` was hard-coded
  because the baseline lay *outside* the search box (``g_v=1e5`` at the edge,
  ``g_w_pcc=100`` against a ceiling of 30), so ``enqueue_trial`` would raise.
  A search space that cannot express the operating point being benchmarked is
  the defect; suppressing the warm start only hid it.
* **Study name not pinned to a previous study.**  It defaulted to
  ``v4_002_ieee39_metric_adapt``, so a re-run silently *resumed* that study and
  mixed trials scored under a different cost function.
"""
from __future__ import annotations

import sys
from pathlib import Path

from tuning import tune as tune_cli


#: Repository root, derived from this file's location so the defaults below do
#: not depend on where the interpreter was started.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent


#: Baseline operating point.  This should be regenerated from
#: ``experiments.run_multi_system_ofo.make_config()`` — the hand-tuned
#: configuration that demonstrably controls well — rather than from the older
#: ``baseline_002*`` snapshots, which describe a *different plant*: no tertiary
#: shunts, no ``sbx_h`` coordination, ``dt_s=60``.  See
#: ``tuning/scripts/save_baseline.py``.
BASELINE = _SCRIPT_DIR / "configs" / "baseline_002_ieee39.yaml"

DEFAULT_ARGS = [
    "--baseline",               str(BASELINE),
    "--n-trials",               "120",
    "--n-startup-trials",       "16",
    "--n-ei-candidates",        "200",
    "--study-name",             "v5_reparam",
    "--storage",
    "sqlite:///" + str(_REPO_ROOT / "results" / "tuning" / "studies.db"),
    "--output",
    str(_REPO_ROOT / "configs" / "tuned_params_ieee39.yaml"),
    "--report",
    str(_REPO_ROOT / "results" / "tuning" / "tuning_report_ieee39.html"),
]


if __name__ == "__main__":
    if not BASELINE.is_file():
        raise SystemExit(
            f"Baseline config not found: {BASELINE}\n"
            f"Generate it with `python -m tuning.scripts.save_baseline`."
        )
    sys.exit(tune_cli.main(DEFAULT_ARGS + sys.argv[1:]))
