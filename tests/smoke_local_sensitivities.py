"""Smoke test for the per-controller local-network Jacobian option.

Runs the multi-TSO/DSO simulation three times on a 4-minute window:

    1.  Baseline: local_sensitivities_tso/dso both False (uses shared_jac).
    2.  Local TSO only: local_sensitivities_tso=True.
    3.  Local DSO only: local_sensitivities_dso=True.
    4.  Both: full Ward-style decentralised sensitivities.

The purpose is to verify each mode constructs the reduced nets, builds
the per-controller Jacobians, runs the post-Phase-2 + main-loop steps,
and produces a non-empty log without crashing.
"""

from __future__ import annotations

import io
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Force UTF-8 stdout so unicode arrows from the runner's print statements
# don't crash on the Windows cp1252 default console.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.multi_tso_config import MultiTSOConfig
from experiments.runners.multi_tso_dso import run_multi_tso_dso


def _make_config(*, local_tso: bool, local_dso: bool) -> MultiTSOConfig:
    return MultiTSOConfig(
        dt_s=60.0,
        n_total_s=4.0 * 60.0,
        tso_period_s=3.0 * 60.0,
        dso_period_s=1.0 * 60.0,
        verbose=2,
        live_plot_controller=False,
        live_plot_cascade=False,
        live_plot_system=False,
        run_stability_analysis=False,
        # Profiles must stay on so the runner's _t_init_total guard is satisfied
        # (it lives inside the `if use_profiles:` block, and several later prints
        # reference it unconditionally — a pre-existing wart in the runner).
        use_profiles=True,
        use_zonal_gen_dispatch=True,
        scenario="wind_replace",
        start_time=datetime(2016, 4, 15, 10, 0),
        contingencies=[],
        local_sensitivities_tso=local_tso,
        local_sensitivities_dso=local_dso,
    )


def _run(label: str, cfg: MultiTSOConfig) -> bool:
    print(f"\n{'=' * 72}\n[smoke_local_sens] {label}\n{'=' * 72}")
    try:
        log = run_multi_tso_dso(cfg)
    except Exception:
        traceback.print_exc()
        print(f"\n[smoke_local_sens] {label}: FAILED")
        return False
    print(f"\n[smoke_local_sens] {label}: OK ({len(log)} records)")
    return True


def main() -> None:
    cases = [
        ("baseline (shared_jac)",  _make_config(local_tso=False, local_dso=False)),
        ("local TSO only",         _make_config(local_tso=True,  local_dso=False)),
        ("local DSO only",         _make_config(local_tso=False, local_dso=True)),
        ("local TSO + DSO",        _make_config(local_tso=True,  local_dso=True)),
    ]
    results = {label: _run(label, cfg) for label, cfg in cases}
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    for label, ok in results.items():
        print(f"  {'OK ' if ok else 'FAIL'}  {label}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
