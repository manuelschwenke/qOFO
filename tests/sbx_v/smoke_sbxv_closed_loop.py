"""
SBX-V closed-loop smoke — regression R1 (plan §10) and wiring sanity.

Runs the shared 005 scenario (IEEE 39 three-zone partition + CIGRE HV
networks) twice over a short horizon:

* arm ``none``  — the CAIR baseline;
* arm ``sbxv``  — full SBX-V wiring with an UNREACHABLE need flag
  (``v_dev_threshold_pu = 0.5``): no request ever fires, the ledger
  stays empty, every solve takes the PricingSolver neutral bypass.

**R1**: the two dispatch trajectories must be BYTE-IDENTICAL (every
recorded control vector, every step).  Additionally the sbxv arm's
metering/settlement plane must have run: complete windows exist and
``finalise()`` settles without error.

Not pytest-collected (smoke_* naming, repo convention) — run directly:

    python tests/sbx_v/smoke_sbxv_closed_loop.py [minutes]
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

# Runnable from anywhere: put the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from sbx_v.config import SBXVConfig

_005 = importlib.import_module("experiments.CIGRE_2026.005_CIGRE_MULTI")
from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402


def make_config(arm: str, minutes: float):
    cfg = _005.make_cigre_config()
    cfg.n_total_s = 60.0 * minutes
    cfg.verbose = 0
    # Identical controller/model path on both arms (sbx v1 idiom).
    cfg.local_sensitivities_tso = True
    cfg.local_sensitivities_dso = True
    cfg.refresh_shared_jac_on_tso = False
    if arm == "none":
        cfg.coordination_mode = "none"
    elif arm == "sbxv":
        cfg.coordination_mode = "sbxv"
        # NEUTRAL/R1 configuration: pricing disabled (explicit bypass)
        # and an unreachable need flag (0.5 pu deviation).
        cfg.sbxv_config = SBXVConfig(
            tso_period_s=float(cfg.tso_period_s),
            miqp_pricing_enabled=False,
            v_dev_threshold_pu=0.5,
        )
    else:
        raise ValueError(arm)
    return cfg


def run_arm(arm: str, minutes: float):
    captured: dict = {}

    def hook(state):
        captured.update(state)
        return None

    t0 = time.perf_counter()
    recs = run_multi_tso_dso(make_config(arm, minutes),
                             pre_loop_hook=hook)
    print(f"  [{arm}] {len(recs)} steps in "
          f"{time.perf_counter() - t0:.0f} s wall")
    return recs, captured.get("sbxv_runtime", {})


def main(minutes: float = 45.0) -> None:
    recs_none, _ = run_arm("none", minutes)
    recs_sbxv, runtime = run_arm("sbxv", minutes)

    # ── R1: byte-identical dispatch ──────────────────────────────────
    assert len(recs_none) == len(recs_sbxv), "record count differs"
    n_checked = 0
    for ra, rb in zip(recs_none, recs_sbxv):
        assert ra.step == rb.step
        for field in ("zone_q_pcc_set", "zone_q_der", "zone_v_gen",
                      "zone_oltc_taps"):
            da, db = getattr(ra, field), getattr(rb, field)
            assert set(da) == set(db), (field, ra.step)
            for z in da:
                assert np.array_equal(np.asarray(da[z]),
                                      np.asarray(db[z])), \
                    f"R1 VIOLATION: {field} zone {z} step {ra.step}"
                n_checked += 1
        assert ra.total_losses_mw == rb.total_losses_mw, \
            f"R1 VIOLATION: losses step {ra.step}"
    print(f"  R1 OK: {n_checked} control arrays byte-identical over "
          f"{len(recs_none)} steps")

    # ── Settlement plane ran ─────────────────────────────────────────
    adapter = runtime.get("adapter")
    assert adapter is not None, "sbxv adapter missing from the runtime"
    final = adapter.finalise()
    obs = final["observations"]
    assert obs, "no metered windows — extend the horizon"
    assert final["grant_records"] == [], \
        "neutral arm must not confirm grants"
    result = final["settlement"]
    assert result is not None
    print(f"  settlement OK: {len(obs)} window observation(s), "
          f"{len(result.window_rows)} window row(s), cases: "
          f"{sorted({r.case for r in result.window_rows})}")
    print("SMOKE PASS")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 45.0)
