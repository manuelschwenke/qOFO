"""
Inspect the slack-Q trajectory of the L0 scenario log to verify whether
the new load split keeps the slack inside its capability envelope.

Reports per-step slack P, Q, V (computed from total record counters) and
flags any step where the saturation indicator fires.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np


def main() -> None:
    base = Path("results/002_compare")

    for scen in ["L0", "L1", "L2", "T-OFO", "C-OFO"]:
        pkl = base / scen / "log.pkl"
        with open(pkl, "rb") as f:
            log = pickle.load(f)
        if not log:
            print(f"{scen:8s}  EMPTY (diverged)")
            continue

        sp = np.array([r.slack_p_mw   for r in log])
        sq = np.array([r.slack_q_mvar for r in log])
        sl = np.array([r.slack_q_at_limit for r in log])
        t  = np.array([r.time_s / 60.0 for r in log])

        print(f"{scen:8s}  steps={len(log)}  "
              f"P_slack [{sp.min():+8.1f}, {sp.max():+8.1f}] MW  "
              f"Q_slack [{sq.min():+8.1f}, {sq.max():+8.1f}] Mvar  "
              f"|Q|_max={np.max(np.abs(sq)):.1f} Mvar  "
              f"saturated_steps={int(sl.sum())}")

        # Highlight Q peaks (top 3 abs values) with timestamps
        top_idx = np.argsort(-np.abs(sq))[:3]
        for i in top_idx:
            print(f"            t={t[i]:5.0f} min: P={sp[i]:+8.1f} MW  "
                  f"Q={sq[i]:+8.1f} Mvar  saturated={bool(sl[i])}")
        print()


if __name__ == "__main__":
    main()
