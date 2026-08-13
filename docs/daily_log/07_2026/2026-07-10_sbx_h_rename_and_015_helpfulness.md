# 2026-07-10 — SBX package rename (sbx→sbx_h, sbxv→sbx_v) + 015 helpfulness evaluation

Session: Claude Code (Fable), on Manuel's request.

## 1. Package rename

**What changed.** `sbx/` → `sbx_h/` (horizontal, TSO–TSO) and `sbxv/` →
`sbx_v/` (vertical, TSO–DSO), including `tests/sbx` → `tests/sbx_h` and
`tests/sbxv` → `tests/sbx_v`. Requested names were "sbx-h"/"sbx-v";
hyphens are not valid in Python package names, so underscores are used
(display names in prose remain SBX-H / SBX-V).

**Method.** `git mv` for the tracked `sbx/` (history-preserving rename
staged); plain `mv` for the untracked `sbxv/`, `tests/sbx`, `tests/sbxv`.
All imports rewritten with word-boundary-safe regexes
(`from sbx…`/`import sbx…`, dotted references `sbx.<module>` /
`sbxv.<module>` in docstrings and error strings, module-path form
`tests.sbx.…`, docstring headers `sbx/<file>.py`). Consumers touched:
`sbx_h/*`, `sbx_v/*`, `tests/sbx_h/*`, `tests/sbx_v/*`,
`experiments/013–019`, `experiments/runners/multi_tso_dso.py`,
`visualisation/plot_sbx.py`, `configs/multi_tso_config.py` (docstrings).

**Deliberately NOT renamed** (documented decision, zero-risk surface):

- `coordination_mode` strings `"sbx"` / `"sbxv"` remain the internal
  mode values (configs, pickles, ~165 tests predate the rename). The
  runner now ACCEPTS `"sbx_h"` / `"sbx_v"` as aliases and normalises
  them at validation time (`experiments/runners/multi_tso_dso.py`,
  whitelist block).
- `MultiTSOConfig` field names (`sbx_config`, `sbxv_config`,
  `sbx_warmup_s`, `sbx_v_std_schedule_path`, `live_plot_sbx`) and the
  `pre_loop_hook` state keys (`sbx_runtime`, `sbxv_runtime`).
- Historical entries in STATUS_SBX.md / STATUS_SBXV.md / handover docs
  (they document the past; a dated rename note was added instead).

**Verification.** `python -m compileall` clean;
`pytest tests/sbx_h tests/sbx_v` → **165 passed** (2:25); import smoke
of all 7 SBX experiment modules + adapters green.

**Reason.** With the vertical mechanism (SBX-V) implemented alongside,
the unqualified `sbx` name became ambiguous; `sbx_h` / `sbx_v` mirror
the thesis terminology (horizontal vs vertical coordination).

## 2. `experiments/015_SBX_COMPARE.py` — rewritten as the SBX-H helpfulness evaluation

**Question (Manuel).** "I am still not sure if sbx-h has any use over
the case without explicit communication." Hypothesis: SBX-H helps only
when area A's tie-line-near voltage is persistently below
schedule/reference AND A has no actuators left to raise it — then, on
top of the passive physical support from B, explicit requests can
deliver more.

**Formalised helpfulness conditions (all necessary):**

- **C1** persistent local infeasibility: violation > need threshold for
  ≥ n_need iterations AND A's gen/DER reactive reserves ≈ 0 in the
  relieving direction (otherwise deals merely substitute local action;
  F1's pinning lottery may even cost);
- **C2** corridor controllability of the violated buses (relieving-sign
  sensitivity, magnitude vs quantum × cycles);
- **C3** supporter capability AND v4 deliverability (t ≥ 1 quantum,
  delivery ratio ≈ 1);
- **C4** counterfactual gap: passive support + neighbours' autonomous
  tracking (mode `none`) must not already close the violation.

Model facts bounding the benefit: the F9 stopping rule (need clears
below threshold ⇒ steady state "just below flag depth", never zero) and
the contract cap `dq_contract_max_mvar`.

**Method / structure.** Full rewrite of 015 (old 2026-07-08 figure-only
version absorbed: its voltage/corridor/infeed overlays live on as
FIG_B/FIG_C; old file in git history). Constructed 2×2 matrix on the
005 scenario, zone A = zone 3 (validated requester), supporters z1/z2:

- **D axis** (deficit): D1 = 500 Mvar sink @ bus 15 + z3 v_min = 1.00
  (validated smoke stress, persists from min 60 to horizon end);
  D0 = 150 Mvar (self-manageable).
- **S axis** (supporter headroom): S1 = untouched; S0 = extra sinks
  z1 (500 Mvar @ bus 27) and z2 (450 Mvar @ bus 6) consuming actuator
  headroom without violating their default bounds → capability
  t_z1/t_z2 collapses → offers below dust.

Arms per cell: `none` / `sbx_inert` (pinning only) / `sbx` (v4).
Decomposition: pinning cost = expo(inert) − expo(none); deal benefit =
expo(inert) − expo(sbx); NET value vs no communication = expo(none) −
expo(sbx). SBX knobs pinned in the script (defaults drift):
k_sched = 2, quantum rate 30 Mvar/15 min (12 Mvar per 6-min cycle),
n_need = 2; tier-1 band per cell from the inert arm's pre-stress clean
cycles (2×RMS rule). Flags H1–H6 encode "helpful exactly iff D1 ∧ S1".
Evidence instruments: `zone_v_min`, `gen_q_reserve`/`tso_der_q_reserve`
(C1 saturation), `t_a`/`t_b` medians (S axis), delivery ratio
(C3), tie-flow proxy shift (C4 passive support; F7 caveat noted).

Outputs: `results/015_SBX_COMPARE/{matrix.csv, FIG_A_matrix.png,
REPORT.md}` + per-cell mechanism/infeed figures, settlement ledgers,
arm pickles. CLI: `--run/--evaluate/--cells/--minutes/--case-study`
(360-min D1S1 with stress 60–300).

**Run.** 120-min matrix (calibration-horizon rule) launched this
session; results appended to STATUS_SBX.md once evaluated.

**Reason.** Settles the "does SBX-H have any use at all" question with
a constructed existence proof + boundary cells, as Manuel specified.

## 3. Results (added after the campaign, same day)

Matrix ran 6 cells (D2/D1/D0 × S1/S0, 120 min) + 2 D2S1 variants
(band-35, no-gate); flags 33 PASS / 1 deliberate FAIL (H3 in D2S1:
gated deal benefit +0.001 — the verdict itself). Headline numbers
(zone-3 exposure, stress window): D2S1 none 5.207 / inert 1.444 / sbx
1.442; D1S1 1.458 / 0.119 / 0.115; D0 rows all 0. Findings G1–G6 in
`results/015_SBX_COMPARE/REPORT.md` and STATUS_SBX.md (2026-07-10):
three deficit regimes (misdirected vs exhausted); the CONTRACT/pinning
layer carries essentially the entire value (+3.76 / +1.34 pu·step,
inverts pre-v4 F1); the deal layer self-suppresses under deep stress
band-independently (delivery gate + magnitude classifier vs the
natural flow shift — F6/D-P7-5 root cause binds the control side; v5
candidate: gate on the settlement attribution); ungated deals deliver
+0.166 pu·step at delivery ratio 0.16 (D2S1_nogate); supporter
capability is a continuum (sinks don't collapse it, margin-zero bounds
cut it 4–5×). Answer to Manuel's question: SBX-H helps in exactly the
constructed corner, but through the scheduled boundary-voltage
contract, not the explicit runtime communication.

## 4. SBX-H v5 "evidence-based SBX" (same day, Manuel: "implement it")

**What changed.** The G1–G6-derived redesign, all as new SBXConfig
defaults with v4 semantics available per knob (STATUS_SBX.md v5 entry
for the full inventory): Move 1 C1 arming — requests only when the
requester is exhausted; two OR-combined paths: optimistic H-weighted
setpoint-headroom bound + **measured stall** (flag persists >
`c1_stall_cycles` boundaries with < 30 % depth recovery). The stall
path was added after the first v5 closed-loop run: the model bound
never armed the truly exhausted zone because AVR setpoint boxes
overstate deliverable Q on saturated machines. Move 2 —
voltage-referenced delivery verification (acting-terminal tracking at
the elapsed cycle's last sample, tol 2.5 mpu) replaces the
stress-blind magnitude test in the gate; tier-2 billing suspended for
undelivered cycles (`delivered_frac`). Move 3 — preventive release
(need hysteresis, release 1 mpu in 015) + gap-sized requests (up to 4
quanta; matching accepts multiples; offers scale). Pruned: modal
recording default off; magnitude classifier demoted to diagnostic.

**Method.** `sbx_h/{config,need,scheduler,matching,capability,
settlement}.py`; v4 protocol pinned in `tests/sbx_h/test_scheduler.py
REF_CFG`; new `tests/sbx_h/test_v5_redesign.py` (10 tests). Suite
74 + 102 passed. 015 re-ran only the sbx arms (none/inert are
mechanism-independent); v4 outputs preserved under
`results/015_SBX_COMPARE/v4_baseline/`.

**Reason.** Make the deal layer earn its place: silent where the
contract suffices (D1: 0 deals, unarmed cycles recorded), active and
verifiable only in the exhausted corner. Results in STATUS_SBX.md.
