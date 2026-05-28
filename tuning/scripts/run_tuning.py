"""tuning/scripts/run_tuning.py — canonical BO-tuning invocation.

Calls ``tuning.tune.main()`` with the project's canonical defaults so
the user does not have to remember the long flag list.  Any extra argv
is forwarded to ``tuning.tune``; argparse takes the *last* occurrence of
each flag, so passing e.g. ``--n-trials 5`` after the script name
overrides the default.

Usage::

    # full thesis-quality run (30 random trials + 120 trials × 5 design scenarios)
    python -m tuning.scripts.run_tuning

    # smoke test (3 trials, throwaway storage)
    python -m tuning.scripts.run_tuning \\
        --n-trials 3 --n-startup-trials 2 \\
        --study-name smoke \\
        --storage sqlite:///results/tuning/smoke.db \\
        --output /tmp/p.yaml --report /tmp/t.html \\
        --no-progress-bar --no-cache-ceilings
"""
from __future__ import annotations

import sys

from tuning import tune as tune_cli


DEFAULT_ARGS = [
    "--baseline",               "configs/baseline_002_ieee39.yaml",
    "--n-trials",               "20",
    "--n-startup-trials",       "15",
    "--n-ei-candidates",        "200",
    "--study-name",             "v4_002_ieee39_metric_adapt",
    "--storage",                "sqlite:///results/tuning/studies.db",
    "--output",                 "configs/tuned_params_002_ieee39.yaml",
    "--report",                 "results/tuning/tuning_report_002_iee39.html",
    "--no-warm-start-baseline",
]


if __name__ == "__main__":
    sys.exit(tune_cli.main(DEFAULT_ARGS + sys.argv[1:]))
