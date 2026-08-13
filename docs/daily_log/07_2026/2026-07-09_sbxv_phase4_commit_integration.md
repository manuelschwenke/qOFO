# 2026-07-09 — SBX-V Phase 4: commit-instant integration + emergency

**What:** Implemented Phase 4 of the SBX-V build plan (continuing from a parallel session that
completed Phase 3 and added the deferred config knobs): `sbxv/commit.py` (`CommitScheduler`),
`sbxv/emergency.py` (`EmergencyHandler`, flag-gated), `__post_init__` validation for the new
config fields, and `tests/sbxv/test_commit_phase4.py` (21 tests). Full regression run 186
green (SBX-V 100, SBX-H + solver 86). No existing module modified.

**Key method / structure:**

- `CommitScheduler.specs_for(k)` makes the priced segment structure a pure function of the
  iteration index and ledger state frozen at scheduled instants — R3 is testable via tuple
  IDENTITY within a stretch. Neutral windows return `None` → `PricingSolver` bypass (R1).
- Expiry ramp: linear descent of the granted bound over the final `ramp_steps` iterations of a
  grant's last window (no confirmed follow-up), frozen once at ramp start — deterministic even
  if a late follow-up lands mid-ramp.
- Scheduled-envelope feedforward (`envelope_step_mvar`): the MSR/MSC offset pattern's SBX-V
  analogue; steps exactly at commit instants (+grant) and ramp iterations (−grant/ramp_steps).
- Incapability: `IncapabilityDeclaration` → Reserve-Observer log → `IncapabilityRecord` for
  TSO-delivers grants; end-to-end test yields settlement case 8.1-3a with pro-rata capacity.
  DSO-delivers declarations stay logged events (metering detects Tabelle 8.2 case 2).
- Emergency: fail-fast when disabled; enabled calls log loudly and never rebuild specs — under
  the open-tail pricing a Notfall-Abruf is consent bookkeeping, not a price change (R3
  preserved by construction).

**Deferred (documented in STATUS_SBXV.md §4.3):** the closed-loop DSO tracking-error invariant
and R1/R2 closed-loop re-runs move into the Phase-5 runner wiring (`coordination_mode="sbxv"`
in `experiments/runners/multi_tso_dso.py`) — the same wiring E1 needs; bundling avoids touching
the 4000-line runner twice.

**Why:** Plan §9 Phase 4; keeps the commercial plane piecewise-constant between commit instants
(R3) while the control plane stays untouched.
