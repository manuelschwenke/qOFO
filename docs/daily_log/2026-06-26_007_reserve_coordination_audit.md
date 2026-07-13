# 2026-06-26 - 007 reserve coordination audit

Timestamp: 2026-06-26
Reason: audit whether experiments/007_TIE_COORDINATION.py --reserve can prove the reserve-scarcity extension of horizontal TSO-TSO coordination.

Changed:
- experiments/007_TIE_COORDINATION.py: explicitly disables enable_tie_coordination and tie_econ_gamma in the baseline configuration before selectively enabling coordination for COORD-sub and COORD-econ. This fixes the prior methodological issue where 005_CIGRE_MULTI.make_cigre_config() already enabled tie coordination, so the nominal OFF run was not a true OFF baseline.
- experiments/runners/multi_tso_dso.py: records the same capability-weighted zone_reserve_scarcity diagnostic for true OFF baselines, including the equivalent/slack generator fold, so OFF and coordinated runs can be compared with the same signal.

Method:
- Preserved the two-loop Delta V_ref coordination law and the no-price/no-extra-objective invariant.
- Verified syntax with py_compile on 007_TIE_COORDINATION.py and multi_tso_dso.py.
- Re-ran focused tests: tests/test_tie_coordinator.py, tests/test_tie_coordination_hooks.py, tests/test_tso_output_gradient.py -> 32 passed.
- Re-ran python experiments/007_TIE_COORDINATION.py --reserve after fixing the OFF baseline.

Corrected result:
~~~text
steady zone reserve scarcity (0 abundant .. 1 saturated):
  OFF        : Z1=0.398  Z2=0.000  Z3=0.181
  COORD-sub  : Z1=0.403  Z2=0.000  Z3=0.200
  COORD-econ : Z1=0.403  Z2=0.000  Z3=0.198
~~~

Interpretation:
- The reserve-economic anchor is implemented and active, but this 007 --reserve case does not prove beneficial reserve sharing.
- Relative to COORD-sub, COORD-econ slightly reduces the strained-zone scarcity (Z3: 0.200 -> 0.198).
- Relative to the corrected OFF baseline, both coordinated runs increase Z3 scarcity (OFF: 0.181), so the current scenario is not a clean demonstration that one zone helps another.

Open point:
- A convincing proof case likely needs either a reserve signal based on absolute remaining headroom or a test network/disturbance where the scarce zone is correctly identified and coordination has a controllable reactive path from an abundant zone.
