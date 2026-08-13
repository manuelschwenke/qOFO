# 2026-07-20 (late) — Phase 6: plant-side Q(V) DSL layer built and verified exactly

**Context.** Gate E was `BLOCKED_DER_QV_MISMATCH`: the static plant applies
the re-anchored Q(V) droop (`QVLocalLoop`), the RMS DERs held constant Q.
Diagnosis verified from the replay figures (same start, growing closed-loop
divergence; static staircase at every 20 s DSO dispatch vs RMS plateaus).
QDSL was ruled out (**Quasi-Dynamic** DSL — not an RMS mechanism; likely
what consumed the prior session's probes). Implemented instead as a real
DSL block in the WECC composite's Plant Control slot.

## Architecture (verified on WP_TSO_s0_b18)

- **`QVPRE_qOFO.BlkDef`** (project `User Defined Models`), authored fully
  via API: `sInput=['u']`, `sOutput=['Qext']`,
  `sParams=['qset,Vanchor,Kdroop,db,Tf']` (comma-joined string!),
  `sStates=['x1']`, `sIntern=['veff,qcorr']`, equations:
  `inc(x1)=u; x1.=(u-x1)/Tf; veff=x1-Vanchor;`
  `qcorr=(veff-db+abs(veff-db))/2+(veff+db-abs(veff+db))/2;`
  `Qext=qset-Kdroop*qcorr` — the exact pandapower law (deadband via
  arithmetic max/min).
- **`Frame WECC PV qOFO`** (local copy of the WECC Large-scale PV Plant
  frame): Plant Control slot got `sInput=['u']` (API) plus **one
  user-drawn wire** Voltage-Measurement `u` (pin 3, the `Vt` net) → Plant
  Control `u` (GUI; wires are graphic-authoritative, headless creation not
  feasible). Slot OUTPUT matching is **by name**: the block's `Qext` output
  feeds REEC_D's `Qext` input through the frame's existing `Qext` signal.
- Composite: retarget `typ_id` to the local frame; **fresh** `ElmDsl QVPRE`
  bound into the Plant Control slot.

## Verification (exact)

`x1` tracks the terminal voltage to 8.7e-8. Law residuals: hold 0.000 Mvar;
qset +0.1 pu → +50.80 Mvar (−0.005 resid); re-anchor −0.02 → −17.09 Mvar
(−0.002 resid) — closed-loop equilibrium matches
`Q = S_n·(qset − K·db(u−Vanchor))` to milli-var.

## PF 2025 API landmines burned into code/memory

1. **`pelm` order follows `ElmComp.pblk`, NOT the frame's `GetContents`
   slot order** (Plant Control is pblk index 8, not 6). Index-based pelm
   writes silently corrupted the composite (Generator=None ⇒ the park ran
   as a plain constant-Q genstat, masking everything). Always map by
   `pblk[i].loc_name`.
2. **DSL declaration format**: list attributes take ONE comma-joined string
   (`['a,b,c']`); states must be declared in **`sStates`** (else
   "Derivative only for state variable useful" at the parser).
3. **A stale `ElmDsl` keeps a dead parameter table** if created before its
   BlkDef was finalised — runtime params all read 0 while the `params`
   vector accepts writes. Recreate the element after the type is final;
   verify with `pre.GetAttribute('c:<param>')` post-ComInc.
4. **`GetOutputWindow()` works in engine mode** (`ow.Clear()` /
   `GetContent()`) — ends init-error blindness; PF logs `evt`/`warn` lines
   per event ("Variable qset not found. Event ignored" was the smoking gun).
5. **`EvtParam` works on DSL parameters and unconnected input signals**
   (resolves `s:`/parameter names); **`EvtLod` with `dP/dQ/iopt_type`
   attrs does NOT fire** in this setup (silently ignored, no OW line) —
   don't use it for disturbances; use parameter events (e.g. on QVPRE
   `Vanchor`) or genstat Qext instead.

## Next (mechanical)

1. Roll out to all 44 parks (`pf/wecc_apply.py` extension): local frame,
   fresh QVPRE, pblk-name pelm fill, per-park params (`qset=Q_LF/S_n`,
   `Vanchor=V_LF`, `K=1/qv_slope_pu`, `db=qv_deadband_pu` from the
   snapshot sgen columns, `Tf=0.02`).
2. `pf/plant.py apply_u`: DER dispatch → EvtParam `qset` (pu) **and**
   `Vanchor` (measured park-bus V at the dispatch instant — the
   re-anchoring step); init writes params per park after sync.
3. Re-run the 900 s Gate-E replay and re-evaluate endpoint errors.
