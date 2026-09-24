# Modal analysis of the coupled plant (thesis Experiment A, Task M)

Timestamp: 2026-09-22T22:02+02:00
Reason: select/justify `T_STS = 20 s` from the time constants of the coupled
plant (`ch:timescales:plant`, Ch 8) by linearising the PF RMS model at W1–W6
with participation factors. Handoff of 2026-09-22 (overrides
`docs/PF_modal_and_outage_parity_prompt.md` of 2026-09-20 where they differ).

## Environment gate (§1 of the handoff)

- **Repository:** work happens in `Z:\Python_Projekte\qOFO_2027` (user
  correction: not `qOFO_GH`), branch `weights-reset-2026-09-10`, HEAD
  `c889384`. Mapped drive `Z:` = `\\e5server\homefolders$\mschwenke`.
- **Not in this checkout** (exist only as untracked files in `qOFO_GH`):
  `docs/PF_modal_and_outage_parity_prompt.md` (read from `qOFO_GH`,
  read-only) and the six `export/snapshots/*_dso3x2.json` snapshots.
- **Python:** `F:\python_environments\qOFO_clean\python.exe`, 3.12.14.
  `import powerfactory` works from
  `C:\Program Files\DIgSILENT\PowerFactory 2025 SP4\Python\3.12`
  (module `__version__ = 25.0.6`). PF 2024 SP7 also installed.
- **GUI/licence:** no PF process of this user. Three `PowerFactory.exe`
  processes belong to other users' disconnected sessions (fkorff, dnickel,
  ajoshi); they did not block the seat.
- **Snapshots** (in `qOFO_GH\export\snapshots`, all `provenance.phase == "full"`;
  "dso3x2" = `dso_overrides={'der_scale': {'DSO_3': 2.0}, 'load_p_scale': {'DSO_3': 2.0}}`,
  a parameter scaling, not a topology change):

  | W | file | sha256 |
  |---|---|---|
  | W1 | `full_t0_20160105-0800_dso3x2.json` | `3a90274550dfe9257101ced6e0d63547712f41fdf70e4ae756837c2ef03124d4` |
  | W2 | `full_at20160222-1300_20160222-1300_dso3x2.json` | `c2d350968c3f02fe5130e29e30ad68b281cd5431c755cbc1b93513cc864586bf` |
  | W3 | `full_at20160104-0800_20160104-0800_dso3x2.json` | `cb7cd57333dff386a695bb36b1459bfc70d71ff37ad6a0c4574a13a27b70fd22` |
  | W4 | `full_at20161126-1000_20161126-1000_dso3x2.json` | `fcd2c21adcaccc9d56de54797f64ce161f016e99c5d8841d397c1d4b90c21a71` |
  | W5 | `full_at20160203-2000_20160203-2000_dso3x2.json` | `552f07b93f28aafca2192953c1ecfb1eadcd47d9f599447d85e9625fcc8ccd83` |
  | W6 | `full_at20160718-1400_20160718-1400_dso3x2.json` | `8df934e63f4c553ebd0b132ba008c39339ed85c4a588ec3c8114142b115c9eed` |

- **Code landmarks:** `pf/screening.py::cmd_modal` at L1656 (moved from
  ~L1590), `RMS_STUDY_CASE = "02_RMS_CoSim"` L66, `COMMOD_EXPORT_ATTRS` L1651,
  `pf/pf_sync.py` CLI has `snapshot --phase --study-case --dry-run`.
- `rtk` is not installed on this host; commands run unprefixed.

## M0 — offline reduction (no seat)

- New `analysis/modal_reduce.py`: merges `modal_modes.csv` files, recomputes
  `tau = 1/|Re|` and L_VQ membership from the participation columns
  (declared rule: group = argmax participation; material `p >= 0.10`;
  L_VQ = rotor, excitation, converter groups plus any mode with a material
  share in a V/Q-relevant group; `other` counted V/Q-relevant by default,
  conservative), evaluates the declared rule, CHECKPOINT R1 (slow L_VQ modes;
  governor modes with material V/Q share), and writes `modal_groups.tex`
  (plus one per droop case), `near_zero_modes.md` (heuristic *candidate*
  explanations only), `modal_spectrum.pdf` (symlog σ–ω, boundary
  σ = −κ/T_STS = −0.15 s⁻¹), `modal_summary.md`. Exit code 1 if R1 triggers.
- New `tests/test_modal_reduce.py` (9 unittest cases): rule edge cases
  (0.09 vs 0.10, per-group vs summed), R1 detection, tap term, near-zero
  exclusion, emitter/recomputed L_VQ disagreement, table shape. All pass.
- Dry run on a synthetic 12-mode set (2 windows × 2 cases): emitter works;
  R1 fires on the planted governor mode with `p_rotor = 0.12`, as intended.

## M1 — probe (`python pf\probes\probe_modal_residue.py`), verbatim verdict

Runtime 572 s (engine start + ComInc + ComMod + result write). Output archived
at `results/screening/probe_modal_residue/2026-09-22_probe_output.txt`.

```
ComMod options present:  iLeft = 1, iRight = 0, iPart = 0, cinitMode = 1
absent: iExpMat, iMatExp, iExport, iExportMat, iWriteMat, dirMat, dirMatrix,
        sExpPath, fileMat, iMatlab, iFormat, iLeftEV, iRightEV, iEigenvec,
        iEigVec, iPartFac, iParticipation, iObserv, iObservability, iContr,
        iControllability, iResidue, iCalcType, iAlgorithm, nMode, iSelect,
        iSelMode, repMode, iShowAll, iFreqRange, fmin, fmax
-> NO matrix-export attribute found on this ComMod.
coupler interface Q 12 ALGEBRAIC; TN bus voltages 30 ALGEBRAIC; machine speeds 6 STATE
ElmRes 'Modal/Eigenvalue Analysis(1)': rows = 973, columns = 2486
  col 0 b:eigvalr, col 1 b:eigvali,
  col 2.. per state '<state>,c:lEVec_mag' / '<state>,c:lEVec_phi'
  (e.g. AVR 03 xa/xe/xf/xr, GOV 03 x0)
```

Reading: (1) matrix export: **no** (none of the candidate spellings);
(2) the modal ElmRes carries **left eigenvectors per state** (iLeft = 1);
right eigenvectors (`iRight`) and participation factors (`iPart`) exist as
options but were off; (3) eigenvector rows are **states only**.
Open: the eigenvalue-method attribute was not among the candidates; the
column count (2484 = 2 × 1242 state entries) versus 973 rows is unexplained
so far.

## M2 — smoke test, unchanged `cmd_modal` (2026-09-22T22:06+02:00)

Snapshots: the six `*_dso3x2.json` copied from `qOFO_GH\export\snapshots`
into `qOFO_2027\export\snapshots` on the user's instruction (untracked);
sha256 re-verified identical to the table above.

Run: scratchpad wrapper calling `pf.screening.main(["modal", "--label",
"smoke_stored_op"])` unchanged, then read-only ComMod/ElmRes diagnostics in
the same process. Folder
`results/screening/smoke_stored_op/20260922-220622/` (`modal.md`,
`commod_diagnostics.txt`). Operating point = whatever the project held
(NOT a thesis window).

- PF output window: "System dimensions: 1242 state variables, 4307 load flow
  variables. Dimension of eigenvalue problem: 1242. Starting QR-algorithm ...
  Results for the 269 calculated eigenvalues not recorded, since they are not
  within the recording limits."
- **n = 1242**, method **QR, all eigenvalues** (`iopt_met = 0`,
  `iQRorQZ = 0`). Runtime 11.7 s for connect + ComInc + ComMod + table.
- **Recording filter active:** `isRecMinMagn = 1, ignoreBelow = 1e-6` and
  `isRecMaxMagn = 1, ignoreAbove = 1e4`. Only 973 of 1242 eigenvalues are
  in the ElmRes (99 pairs + 775 real = 874 distinct modes). The min-magnitude
  filter hides exactly the structural zeros stop rule 2 needs.
- Recorded: 0 with Re > 0; 0 with |Re| < 1e-3; Re in [-1e4, -0.01738];
  min ζ (|Im| > 0.1) 0.028. Slowest τ: 57.5 s (real), 12.7 s (0.174 Hz),
  5.27 s (0.136 Hz), 5.16 s (1.088 Hz, ζ 0.028). **24 modes at exactly
  Re = -0.2000 (τ = 5.000 s)** = the 24 `TAPCTRL_qOFO` states
  (`x1. = (ntapcmd - x1)/Tmech`, Tmech = 5 s) -- inference until confirmed
  by participation.
- ComMod attribute names actually present (PF 2025 SP4), via `dir()`:
  `iopt_met` (method), `iQRorQZ`, `iLeft`, `iRight`, `iPart`,
  `iopt_noevec`, `i_nr`, `iopt_which`, `shiftr/shifti`, recording filters
  `isRec*`/`ignoreAbove`/`ignoreBelow`/`min*`/`max*`, and **Matlab export:
  `dirMatl`, `iSysMatsMatl` (system matrices), `iEValMatl`, `iLEVMatl`,
  `iREVMatl`, `iPartMatl`**. So a matrix export *does* exist; the probe's
  candidate spellings (and `COMMOD_EXPORT_ATTRS` in `cmd_modal`) miss it.
- State inventory (ElmRes columns, left eigenvectors, 1242 states):
  REGC_C 352/44, REEC_D 308/44, WTGWGO_A 132/44, ElmLod 113/113,
  Protection V_f_ROCOF 88/44, StaVmea 88/44, QVPRE_qOFO 44/44, ElmSym 35/6,
  gov_IEEEG1 30/5, avr_IEEET1 24/6, TAPCTRL_qOFO 24/24, gov_IEEEG3 4/1
  (states/objects). Six in-service machines.

Open after M2: (a) ElmLod carries one state per load although the loads are
documented as "no recovery dynamics"; (b) QVPRE parameters (`qset`,
`Vanchor`, `db`, `qmin`, `qmax`) are written only by `pf/plant.py`
(`_anchor_qv` pass) and `pf/wecc_apply.py`, **not by `pf_sync`**, so after a
sync they hold the values of whatever ran last.

## M3 — participation factors in `cmd_modal` (2026-09-22T22:30+02:00)

User go for M3 and for re-anchoring the QVPRE blocks per window.

**Exploration (two short PF sessions, ComMod options and TAPCTRL flags
restored and verified identical):**
- ElmRes with `iRight = iPart = 1` carries `<state>,c:p_mag/_phi` and
  `c:rEVec_*`; but one `GetValue` costs ~0.6 ms, i.e. ~15 min per
  1242x1242 participation matrix. **Route taken: ComMod's Matlab export**
  (`dirMatl`, `iEValMatl`, `iPartMatl`, `iSysMatsMatl`): `EVals.mtl`,
  `PartFacs.mtl` (states x modes), `Amat.mtl`, `Jacobian.mtl` (full DAE
  Jacobian incl. algebraic variables), `VariableToIdx_*.txt`. Parse 0.2 s.
- Offline checks: `eig(Amat)` reproduces `EVals` to 2.1e-11 relative (all
  1059 nonzero); 183 exact zeros on both sides. `PartFacs` equals
  normalised `w∘v` for simple real modes (median error 0); the exported
  `lEV/rEV` columns are **not** consistently paired for complex/repeated
  eigenvalues, so participation is never recomputed from them.
- With all recording filters off: 1242 eigenvalues, 183 with |λ| < 1e-6.
  `repConstantStates = 1` makes PF list the states it holds constant at the
  OP: AVR `xr` ×6, `G 01 phi`, Protection `xf` ×44, REEC_D `x2, x3` ×44
  (139). The other 44 zeros are REGC_C `x_phiref` (PLL angle integrators).
- PF reports "132 objects use buffers (delay, movingavg, ...)"; the object
  list is not in the output-window text. Buffers are not represented in a
  linearisation -- limitation to state.
- ElmLod state: `TypLod` "General Load Type", 100 % dynamic, `t1 = 0.1 s`
  ("Time constants: Delay"), `kpf = kqf = 0` (also answers Task O's
  open question on load frequency dependence).
- TAPCTRL out of service: n 1242 -> 1218, ComInc fine, restore verified.
- Stored QVPRE parameters (the "originals"): `db = 0.5` on all 44,
  `qset ≈ 0`, `Kdroop = 16.667` (slope 0.06), `Tf = 0.02`, placeholder
  limits `qmin/qmax = -0.33/0.41`, `Vanchor` 0.994–1.036.

**Code (additive):**
- `pf/modal_participation.py` (new, no PF import): readers for the export,
  constant-state parser, `group_of` (ElmSym -> rotor; avr -> exc; gov ->
  gov; REGC/REEC/REPC/QVPRE/WTG and anything inside a `WECC_` composite
  (StaVmea, Protection, WTGWGO) -> conv; else other), `mode_table`
  (normalisation `|P_kj| / Σ_k |P_kj|`, group = argmax of group sums,
  L_VQ via `analysis.modal_reduce.in_l_vq`), `degenerate_clusters`.
- `pf/screening.py`: `COMMOD_PARTICIPATION_OPTS`, `_CommodOptions`
  (save/restore), `_state_owner_meta`, `_modal_participation`; `cmd_modal(...,
  participation, window, droop_case)`; CLI `--participation --modal-window
  --droop-case`; `iSysMatsMatl`/`dirMatl` added to the export superset.
  `modal.md` unchanged in format; with `--participation` it is built from
  the same full-spectrum run.
- `pf/modal_window.py` (new): per-window orchestration (originals ->
  diagnostic before change -> re-anchor as in `plant.py` -> TAPCTRL out ->
  `db_wide`/`db_zero` -> restore + read-back verification -> acceptance
  check -> `modal_reduce`), `--restore originals.json` for recovery.
- `tests/test_modal_participation.py` (new). 13/13 offline tests pass.
- Bug found and fixed on the way: `GetContent()` returns one element per
  message, a message can span lines; the constant-state parser now splits.

**Validation run** `results/screening/m3_stored_op/20260922-2230xx`
(stored OP, not a thesis window): 1242 states, 1143 modes (Im ≥ 0), 0
unstable, 183 near-zero (139 dominated by a PF constant state), 0 unresolved
owners, ElmRes vs export 1.8e-12.

**Snapshot finding for W5 (to raise before W5):** the four TSO parks
(`WP_STATCOM|…`, TSO-WP) carry 42–239 Mvar at W5 while their snapshot
`op_diagram` (VDE-AR-N-4120-v2 at P/S_n < 0.1) gives a zero-width box; W1
has no park outside its box.

## M4 — W1 end to end (2026-09-22T22:34+02:00)

**Sync and verification** (`results/screening/modal_campaign/W1/`):
`pf_sync … full_t0_20160105-0800_dso3x2.json --phase full --study-case
02_RMS_CoSim`: created 0, renamed 0, **updated 170** (load plini/qlini 53
each, pgini 50, qgini 4, qsetp 4, taps 6), deleted 0 (`sync.log`). Dry-run
right after: **0/0/0/0** (`dryrun.log`); repeated before the final window run:
0/0/0/0 (`dryrun2.log`).

**Window run** `python -m pf.modal_window --window W1 …` ->
`modal_campaign/W1/20260922-223416/` (report, state JSON, originals, diag
CSVs, reduction), runs `W1_db_wide/20260922-223432`,
`W1_db_zero/20260922-223451`. A first attempt (`…/20260922-223122`, runs
`…223139`/`…223200`) completed both PF runs and the restore but crashed in
the post-processing (sorting complex numbers in `acceptance`); fixed and
re-run after a fresh dry-run; numbers identical.

§5 diagnostics:
1. Before any change (stored QVPRE params vs synced LF): |V_LF − Vanchor|
   max 0.055 pu, 44/44 above 1e-4 -> **PF's reference was not anchored at
   the synced point**; inside the stored db (0.5): 44/0; inside the
   snapshot/thesis db (0.01): 1 inside / 43 outside; stored qset vs Q_LF
   max 128.5 Mvar off.
2. After re-anchoring + ComInc: max |x1 − Vanchor| 3.7e-12, max |Qext −
   Q_LF/S_n| 2.5e-17 pu, no init warnings -> linearisation at the
   dispatch-anchored equilibrium. Kdroop 16.667 (slope 0.06) everywhere.
   16 parks with zero-width capability box (P/S_n < 0.1): droop clipped,
   inert in both cases. No park pinned, none outside its box.
3. Dead band before change 0.5 (all 44) and after restore 0.5 (all 44);
   restore max deviation 0.0; TAPCTRL flags restored; ComMod options
   unchanged.
- Tap controllers: 24 states (24 modes at exactly −0.2 s⁻¹ at the stored
  OP); set out of service for the modal runs -> n = 1218; restored.
- **db_zero acceptance: PASSED.** db_wide: 44 QVPRE-dominated modes all at
  σ = −50.000 s⁻¹ (= −1/T_f, inert). db_zero: 43 QVPRE-dominated modes,
  σ ∈ [−74.8, −50.0]; none slower than 50 s⁻¹ (consistent with λ(D) ≥ 1);
  converter spectra differ.

Results (n = 1218 both cases; 1118 / 1122 modes with Im ≥ 0; 0 unstable):

| | db_wide | db_zero |
|---|---|---|
| slowest in L_VQ | 13.9 s, 0.172 Hz, ζ 0.066, G 01 psifd (+AVR 01 xe/xf), label rotor (p_rot .46 / p_exc .44 / p_gov .09) | 15.0 s, 0.165 Hz, ζ 0.064, same mode, label exc (.49/.49/.03) |
| further R1 modes | 11.3 s, 0.791 Hz, ζ 0.018, G 01/G 10/G 09/G 07 speed+angle (rotor .90) | 10.6 s, 0.779 Hz, ζ 0.019 (same); 7.07 s, 0.061 Hz, G 04 psifd + AVR 04 |
| tau_exc / tap term | 5.53 s (G 10 psifd + AVR 10, 0.136 Hz) / 21.6 s | 15.0 s / 49.9 s |
| governor modes with material V/Q share | 5 (τ ≤ 4.2 s; e.g. 0.214 Hz GOV 01 + G 01 speed, p_rot .28) | 4 (τ ≤ 4.8 s) |
| slowest overall | 57.6 s, GOV 10 xr (IEEEG3), p_gov .998, max V/Q share .001 -> excluded | 57.5 s, same |
| required T_STS / margin | 41.6 s / −21.6 s | 49.9 s / −29.9 s |

**CHECKPOINT R1 TRIGGERED at W1 in both droop cases -> stopped (stop rule
3).** Converter group τ ≤ 0.3 s; ElmLod (other) τ 0.07–0.1 s.

Near-zero (stop rule 2), 199 per case, Re within ±3e-11: 155 dominated by a
state PF lists as constant at the OP (Protection xf 44, REEC_D x2 44, x3
44, x5 16 [the 16 zero-box parks], AVR xr 6, G 01 phi 1) -- these are
exactly the 155 all-zero rows of A; 44 dominated by REGC_C x_phiref, whose
A-rows are identical (sole entry 376.99 at G 01 speed: x_phiref' =
ω_n Δω_G01) -> linearly dependent rows, structural. rank deficit of A =
199 = number of zeros (semisimple). All 199 explained.

Group-label sensitivity: the binding mode is a G 01 field-flux/AVR 01
mode; psifd is an ElmSym state (-> rotor by the declared mapping), xe/xf
are AVR states (-> exc), and the two sums are within 0.02 of each other,
so the label (and with it tau_exc and the tap term) flips between the
cases. L_VQ membership is unaffected.

## R1 option 1 — residue-based observability (2026-09-23)

Author's go (2026-09-23): "you can do 1". Questions answered in chat: the
binding modes are not tap modes (TAPCTRL was out of service in the modal
runs); the 5 s was taken as the declared tap command-to-contact dead time,
never modelled as a dead time in the linearisation (a dead time has no
eigenvalue). Fact noted: in the PF project TAPCTRL is a **first-order lag**
Tmech = 5 s (the 24 modes at −0.2 s⁻¹), whereas the thesis and the Ch 9
actuator-location sweep use a 5 s pure delay + 0.05 s PT1.

Pass criterion and delay convention were NOT declared by the author (the
question dialog was dismissed); defaults used and labelled as such:
absolute Gate-D bands (1e-3 pu V, 1 Mvar Q) with Gate-D step sizes
(DER min(60 Mvar, 0.5 S_n, upward headroom), AVR 0.02 pu, 1 tap); tap
responses timed from contact (threshold 15 s), other inputs 20 s; the
communication-delay reading (15 s for all) and a relative e^-3 settling
time are reported alongside.

**Method** (`analysis/modal_observability.py`, new; `pf/modal_window.py`
extended):
- PF exports the full DAE Jacobian (5397 vars: 1218 states + 4179
  algebraic) and `M`; Kron reduction reproduces `Amat` to 3e-13 relative.
- Inputs from PF's own equations: DER `qset` enters the QVPRE `Qext` row
  (`g = h − Qext`, +1 if unclipped; clipped zero-box parks are not
  dispatchable); AVR `usetp` enters the IEEET1 `xa` row with −(coefficient of
  `ut`) (verr = usetp − ut; e.g. +83.33 on AVR 04); taps via `nntapin` of an
  extra run with TAPCTRL in service (PF then gives the tapped transformers
  explicit branch-current variables, so the taps-in system is used in its own
  coordinates; its reduced A has the same spectrum as the taps-out A to
  8e-11). `G 01` AVR excluded as input (network equivalent, decision
  2026-08-07). Shunts not included (admittance scale in PF's network
  equations not established).
- Outputs: TN and PCC voltage magnitudes from `ur/ui` (op values recorded
  by `modal_window` in `op.json`); coupler HV-side Q (12) and TSO tie-line Q
  (5, zone boundaries from `zone_map`) from pandapower's branch model at PF's
  voltages -- validated against PF's own flows: tie lines ≤ 4e-6 Mvar;
  couplers a constant +0.040 Mvar offset (no-load representation).
- Step responses y(t) = D + Σ_k R_k/λ_k (e^{λ_k t} − 1) with cluster
  projectors for repeated eigenvalues; the 199 structural zeros give a ramp
  ≤ 2e-13 (i.e. none). One fast cluster (λ = −100, 84 modes) is defective
  (cond 1e11); settling is therefore resolved from 0.5 s on (its term is
  < e^-50 there). Modal tails validated against stiff ODE integration
  (BDF) at 15 s and 20 s: ≤ 1e-7 of the band.
- `modal_window` now also writes `op.json` and runs `<W>_<case>_tapsin`.

**W1 results** (`modal_campaign/W1/20260923-144506/observability/`:
`pairs.csv`, `summary.md/json`, validations). Window re-run 2026-09-23
14:45 after a fresh dry-run (0/0/0/0); restore ok, ComMod options unchanged;
modal numbers identical to 2026-09-22.

Inputs 57 dispatchable (28 DER, 5 AVR, 24 taps); skipped 17 (16 zero-box
DER, G 01 AVR). Outputs 59 (30 V_TS, 12 V_PCC, 12 Q_STS, 5 Q_tie).
Validations: A 3e-13; taps-in spectrum 8e-11 / 2e-10; tie-line Q ≤ 4e-6
Mvar, coupler Q +0.040 Mvar offset; ODE vs modal ≤ 5e-6 of the band;
zero-mode ramp ≤ 3e-13; no unstable mode.

Max settling time T_s (absolute Gate-D bands, worst-case steps; taps from
contact):

| input class | db_wide | db_zero | threshold (tap-only dead time) | comm.-delay reading (15 s) |
|---|---|---|---|---|
| AVR (0.02 pu) | 18.06 s (G 10 -> Q tie line14) | 17.70 s (G 10 -> Q tie line2) | 20 s: met | NOT met |
| DER Q | 12.94 s (WP_TSO_s0 -> V_TN_bus18) | 4.56 s | 20 s: met | met |
| TAP_MT | 11.94 s (MT_g2_t6 -> V_TN_bus18) | 12.98 s | 15 s: met | met |
| TAP_NC3W / other | ≤ 0.5 s | ≤ 0.5 s | 15 s: met | met |

The G 01 excitation mode (τ 13.9/15.0 s) is observable: it is the dominant
tail term at 20 s for AVR -> Q_tie (0.39 Mvar of band 1), AVR -> V_TS
(6.5e-4 pu of band 1e-3) and DER -> Q_STS, but below the band there. The
relative e^-3 settling time fails everywhere (up to ~290 s): small static
changes against slow residual tails (e.g. the 57 s IEEEG3 governor mode) --
that criterion is not usable here.

Not established: shunt inputs; DSO internal bus voltages (excluded as in the
Gate-D output set, although CLAUDE.md lists DSO nodal voltages as Layer-2
controlled outputs); nonlinear effects of worst-case steps (QVPRE clipping,
limits); W2–W6.

## M5 — campaign W2–W6 with DSO voltages (2026-09-23T15:40–16:23+02:00)

User go (2026-09-23): W2–W6, DSO bus voltages added; no communication
delay (taps 15 s from contact, AVR/DER 20 s). New `pf/modal_campaign.py`
(sync -> dry-run gate -> `modal_window` -> `modal_observability`, each PF step
in its own process; stops on unclean dry-run, error, failed restore, unstable
mode, failed db_zero acceptance; always re-syncs W1 at the end).
DSO voltage outputs = the DSO controller's own set `meta.dn_bus_indices`
(40 DSO_k_bus* at 110 kV incl. 12 PCC, 12 tertiary at 20 kV) -> outputs 99
(30 V_TS, 12 V_PCC, 40 V_DS, 12 Q_STS, 5 Q_tie). W1 re-analysed offline.

`modal_window` change: a park whose snapshot load-flow Q lies outside its
own capability box gets the box pinned at that Q (linearisation stays at the
snapshot OP, droop inert). **Data finding:** W3 (3 TSO parks above qmax),
W4 and W5 (4 TSO parks, 32–239 Mvar with a zero-width VDE box at P/S_n <
0.1) -- the snapshots' TSO-park Q is inconsistent with the thesis plant's
capability model (likely the PV seeding in `make_snapshots`).

Syncs: W2 172, W3 173, W4 166, W5 166 updates; every dry-run 0/0/0/0; final
W1 re-sync 151 updates, dry-run 0/0/0/0. Every window: restore ok (db 0.5),
ComMod options unchanged, 0 unstable modes.

**STOP at W5** (§5 acceptance): db_zero converter spectrum identical to
db_wide -- structural, not a failed slope case: all 44 parks inert (40
zero-width boxes, 4 pinned). W5 residue analysis and W6 not run.

Max residue settling time T_s [s] (absolute Gate-D bands, worst-case steps):

| class | W1 wide/zero | W2 | W3 | W4 | thr |
|---|---|---|---|---|---|
| AVR | 18.06 / 17.70 | 18.12 / 17.72 | 16.62 / **20.26** | 16.74 / 19.84 | 20 |
| DER Q | 12.94 / 4.56 | 11.44 / 4.58 | 10.10 / 0.54 | 17.76 / 0.96 | 20 |
| TAP_MT | 11.94 / 12.98 | 11.64 / 13.00 | 11.72 / 12.66 | 11.74 / 12.34 | 15 |
| TAP_NC3W, other | ≤ 0.5 | ≤ 0.5 | ≤ 0.5 | ≤ 0.5 | 15 |

1 of 45 342 pairs exceeds its threshold: W3 db_zero, AVR G 10 (+0.02 pu)
-> Q tie line 2 (z1–z2), 20.26 s; tail −2.20 Mvar @15 s, +1.18 @20 s;
dominant: G 10 excitation mode (0.13 Hz, ζ 0.195, τ 6.14 s; G 10 psifd .46,
AVR 10 xe .29 / xf .17) -- a mode inside the rule's τ ≤ 6.67 s with a large
residue. Modes with τ > 6.67 s contribute at most 65 % of the band at 20 s
(AVR G 10 -> V_TN_bus38, G 01 excitation mode). DSO voltages never bind.

Mode summary (declared rule, all windows): L_VQ max τ 12.7–15.9 s (G 01
excitation mode; W3 wide 0.776 Hz EM mode), except **W4 db_wide and W5:
0.765 Hz electromechanical mode, ζ = 0.0044, τ = 47.2 s (G 09 dominant)**;
in W4 db_zero the droop slope damps it to ζ 0.014. Slowest overall
everywhere: GOV 10 xr (IEEEG3), 56.8–57.6 s, outside L_VQ.

## AVR step 0.01 pu and the per-mode settling view (2026-09-23T16:40+02:00)

Author's decision: maximum AVR set-point step 0.01 pu (was 0.02 pu in the
Gate-D catalogue). `AVR_STEP_PU = 0.01` in `analysis/modal_observability.py`.
Offline reruns W1–W5 (no PF), `--no-ode-check` (same matrices as the
ODE-checked runs; the step scales responses linearly), written to
`<window>/observability_avr0.01/` (the 0.02 pu results kept). W5 analysed
as a single case (db_wide ≡ db_zero, structural); W5 has no dispatchable DER.
One transient `WinError 10106` (Winsock, at import) on W4; retried, fine.

New per-mode output (`modal_settling_<case>.csv`): for each mode k, its
largest amplitude over all dispatch -> output pairs, A_k = |R_k/λ_k|·Δu (×2
for a conjugate pair), and T_k = τ_k·ln(A_k/band) (0 if A_k ≤ band) -- the
settling time read directly off the modal data. For an oscillatory mode this
is the envelope crossing, an upper value for that mode alone; the exact
multi-mode T_s (last band exceedance of the full modal sum) can be up to half
a period earlier. κ = 3 in the declared rule is T = τ·ln(A/b) with A/b = e³
assumed for every mode.

Results (max T_s [s], wide/zero):

| class | W1 | W2 | W3 | W4 | W5 | thr |
|---|---|---|---|---|---|---|
| AVR 0.01 pu | 12.0/13.4 | 11.8/13.5 | 11.6/16.0 | 11.7/12.6 | 11.7 | 20 |
| DER Q | 12.9/4.6 | 11.4/4.6 | 10.1/0.5 | 17.8/1.0 | -- | 20 |
| TAP_MT | 11.9/13.0 | 11.6/13.0 | 11.7/12.7 | 11.7/12.3 | 11.8 | 15 |
| TAP_NC3W, other | ≤ 0.5 | ≤ 0.5 | ≤ 0.5 | ≤ 0.5 | ≤ 0.5 | 15 |

0 of 51 084 pairs exceed their threshold. Minimum margins: AVR 4.0 s (W3
zero), DER 2.2 s (W4 wide, tail dominated by the 0.77 Hz mode, τ 47 s),
TAP_MT 2.0 s (W2 zero, from contact).

Per mode (worst over pairs): G 10 excitation mode τ 5.6–6.1 s, A/b up to 15
-> 16.7 s; G 01 excitation mode τ 15 s, A/b 2.8 -> 15.3 s; G 04 field mode
τ 5.8–6.7 s, A/b up to 12 -> 14.4 s; 0.77 Hz electromechanical mode W4 wide
τ 47 s but A/b 1.25 -> 10.7 s, W5 A/b 0.64 -> never leaves the band.
W6 not run (campaign stopped at W5; awaiting the author's go).

## W6 and the dissertation bundle (2026-09-23T18:00–18:30+02:00)

Author's go: W6, then the bundle. `python -m pf.modal_campaign W6`: sync
164 updates, dry-run 0/0/0/0; restore ok (db 0.5 -> 0.5), ComMod options
unchanged, acceptance passed, no pinned or zero-box parks; residue analysis
with ODE check (≤ 3e-6 of the band), AVR step 0.01 pu. Final W1 re-sync 164
updates, dry-run 0/0/0/0 -> project left as found.

W6 result: 1 exceedance -- db_zero, MT tap `MT_g9_t12` -> Q tie line 14,
15.36 s after contact (threshold 15 s); tail −1.004 Mvar @15 s; dominant
the G 01 excitation mode (τ 15.2 s, A/b 2.8). `MT_g9_t12` is the unit
transformer of **G 01** (the network equivalent, whose AVR is excluded as an
input). Over W1–W6: 1 of 65 538 pairs above threshold; without G 01's tap
0 of 64 350, max MT tap 13.0 s. Whether that tap is a dispatch input is the
author's decision -- reported both ways, not decided here.

Bundle: new `analysis/build_modal_bundle.py` ->
`results/THESIS_ch8_modal_2026-09-23/` (README with provenance, §5
diagnostics, answers to §9, residue method and results, what is not
established; `modal_modes.csv`, `modal_groups*.tex`, `modal_spectrum.pdf`,
`near_zero_modes.md` + per-run A-row verification (all 12 runs: zero rows
== PF constant list, x_phiref rows identical, rank deficit == #zeros),
`modal_summary.md`, `residue_settling.csv/.tex`,
`modal_settling_modes.csv/.tex`). Declared rule over W1–W6: slowest L_VQ
mode 47.2 s (0.77 Hz EM mode, W4/W5) -> margin −121.7 s, R1 triggered in
both droop cases.

## Measured variables and actuators from the code (2026-09-24)

User: check in the repo whether G 01's unit-transformer tap is an actuator;
tie lines are not measured variables (only TS/DS voltages and interface Q).

Code facts (`experiments/runners/multi_tso_dso.py`, `configs/config.py`):
- G 01 = `net.gen[9]` `G1_bus38`, terminal bus 40, `slack = True` (no ext_grid).
- AVR: slack/equivalent V-ref withheld unless `dispatch_slack_gen_v_ref`
  (default False, not overridden in `configs/paramsets/*`), L710–726. The
  comment's reason "the PowerFactory RMS model has no AVR block for it" is
  outdated (AVR 01 exists since 2026-07-23).
- OLTC: "Exclude the slack gen's machine trafo from the controllable OLTC set",
  L766–798 -> `MT_g9_t12` is not dispatched (`tso_oltc_mode = "ofo"`). The
  snapshot's `actuators` inventory lists it anyway (and gen 9 under
  `avr_gen_indices`): the inventory is broader than the controller's set.
- Measured: TSO V = TN buses minus PV/slack (`v_bus_indices_z`, L947–950; 30);
  DSO V = `hv.bus_indices` (L1351; 40 incl. PCC, NOT the 12 tertiary buses
  that `meta.dn_bus_indices` adds); interface Q = `res_trafo3w.q_hv_mvar`.

Changes (`analysis/modal_observability.py`): outputs from
`controlled_voltage_names` (the runner's logic) + coupler HV Q, tie lines
dropped (`include_ties=False`); `slack_machine_taps` excludes the slack
machine's OLTC; G 01's AVR is computed as a separate class
`AVR_G01_equivalent` (not in the verdict, not in the per-mode worst case).
Reruns W1–W6 offline -> `<window>/observability_measured/`; bundle rebuilt.

Result: **0 of 53 300 pairs above threshold** (W1–W6, both cases). Max T_s:
AVR 12.4 s (margin 7.6), DER Q 13.9 s (6.1), MT taps 12.7 s from contact
(2.3), coupler/other taps ≤ 0.5 s. For information, G 01's V-ref with the
RMS model's AVR 01: up to 33.7 s at TN_bus38, 264 of 984 pairs > 20 s,
dominated by the G 01 excitation mode (τ 15 s). The author's idea of an
ideal pilot-node control behind the equivalent would remove that path; it is
not what the RMS model contains and was not evaluated.
