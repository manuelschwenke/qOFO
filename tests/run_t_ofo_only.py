"""
Run only the T-OFO scenario from 002_M_TSO_M_DSO_COMPARE.py with the
current SCENARIOS["T-OFO"] config.  Reuses cached L0/L1/L2/C-OFO logs
to regenerate the comparison plots quickly.

Used to verify mitigations D (under-relaxation) and H (drop Q_PCC
dispatch) without the cost of re-running all five scenarios.
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

spec = importlib.util.spec_from_file_location(
    "exp_002_compare",
    ROOT / "experiments" / "002_M_TSO_M_DSO_COMPARE.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from visualisation.plot_compare_scenarios import plot_scenario_comparison

OUT_ROOT = ROOT / "experiments" / "results" / "002_compare"


def main() -> None:
    print("=" * 72)
    print("  Running T-OFO scenario only (other logs reused from cache)")
    print("=" * 72)
    mod.run_one_scenario("T-OFO", mod.SCENARIOS["T-OFO"], str(OUT_ROOT))
    print()
    print("=" * 72)
    print("  Reloading all logs and replotting comparison")
    print("=" * 72)
    logs = mod.load_logs(str(OUT_ROOT))
    plot_scenario_comparison(logs, out_dir=str(OUT_ROOT))
    print(f"  Wrote summary.csv and comparison figures to {OUT_ROOT}")


if __name__ == "__main__":
    main()
