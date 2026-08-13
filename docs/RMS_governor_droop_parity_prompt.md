# Task prompt — prove or refute governor-droop parity between pandapower and PowerFactory RMS

> Hand this to a Claude Code session running **on the PowerFactory machine**
> (licence seat available, GUI closed). Written 2026-07-22.

---

## 0. Context you need before starting

This repo is the plant model for a PhD on hierarchical multi-zone reactive-power
control. The pandapower plant and a PowerFactory RMS twin of the same IEEE-39
network are kept in parity through the snapshot exporter
(`export/dynamic_snapshot.py`) and the sync/parity scripts in `pf/`. Gates A–C
(static load-flow parity) are closed; Phases 5–6 (RMS) are in progress. Read
`docs/RMS_IEEE39_PowerFactory_Build_Plan.md` and `docs/pf_api_notes.md` first.

**Established model facts (verified, do not re-derive):**

- The pandapower plant has **no `ext_grid`**. Bus 38 hosts a `slack=True` gen
  (G1); every machine carries `slack_weight = sn_mva`
  (`network/ieee39/build.py:171`, `network/ieee39/helpers.py:140`).
- Every plant power flow runs `distributed_slack=True, enforce_q_lims=True`
  (`export/make_snapshots.py:91`, `configs/baseline_002.yaml:68`,
  `core/plant.py:238`).
- Machine ratings (`network/ieee39/constants.py:65`, `GEN_NAMEPLATE`):

  | label | bus | S_n [MVA] | type | PF `loc_name` |
  |---|---|---|---|---|
  | G1  | 38 | 10000 | Equivalent (NY system) | `G 01` |
  | G2  | 30 |   700 | Nuclear    | `G 02` |
  | G3  | 31 |   800 | Nuclear    | `G 03` |
  | G4  | 32 |   800 | Coal       | `G 04` |
  | G5  | 33 |   600 | Coal       | `G 05` (`ngnum = 2`) |
  | G6  | 34 |   800 | Nuclear    | `G 06` |
  | G7  | 35 |   700 | Coal       | `G 07` |
  | G8  | 36 |   700 | Nuclear    | `G 08` |
  | G9  | 37 |  1000 | Nuclear    | `G 09` |
  | G10 | 29 |  1000 | Hydro      | `G 10` |

  Σ S_n = **17 100 MVA**, so G1 alone carries **58.5 %** of the distributed-slack
  weight in the `base` model. Name map: `pf/naming.py::TEMPLATE_MACHINE_NAMES`.

**The mathematical claim under test.** pandapower's distributed slack allocates
the post-contingency active-power mismatch as

```
ΔP_i = (S_n,i / Σ_j S_n,j) · (ΔP_out + ΔP_loss)
```

Classical primary-control (droop) sharing allocates it as

```
ΔP_i = ΔP_out · (1/R_i) / (Σ_j 1/R_j + D),      1/R_i [MW/Hz] = S_n,i / (R_pu,i · f_n)
```

These two coincide **iff** (a) every machine has the same per-unit droop
`R_pu` on its own machine base, (b) load frequency self-regulation `D = 0`,
(c) no governor saturates on `P_max`, (d) no secondary control (AGC) acts.

So pandapower's `slack_weight = sn_mva` silently encodes a **uniform per-unit
droop** assumption. Nothing in the repo states a droop value; there is no
`R_pu` / `droop` field anywhere in the pandapower model.

**The unverified claim you must settle:**

> PF-RMS and pandapower will diverge in post-contingency active-power sharing
> unless the PowerFactory governor droops are parameterised so that the
> settled primary-control sharing is ∝ S_n,i. The static parity gates (A–C)
> cannot detect this, because they compare a single operating point at which
> the snapshot already stores the *converged, post-slack-allocation*
> `res_gen.p_mw` — the allocation rule itself is never exercised.

Note also (relevant to condition (c)): pandapower's distributed slack is a
single **unconstrained** scalar folded into the NR Jacobian
(`pandapower/pypower/newtonpf.py:93,275-292`) — `min_p_mw` / `max_p_mw` are
**not** enforced by `runpp`. PF governors *do* saturate. This is a predicted
divergence mode, not a bug to fix silently.

---

## 1. What to produce

A verdict — **confirmed**, **refuted**, or **confirmed with quantified
magnitude** — backed by a numerical comparison, plus a daily log at
`docs/daily_log/07_2026/2026-07-22_governor_droop_parity.md` per the repo convention
(what changed, method, timestamp, reason).

Do **not** change the pandapower plant model. If the finding implies a PF-side
parameterisation change, propose it and stop — Manuel decides architectural
changes.

---

## 2. Step A — inventory the PF primary-control layer (cheap, do this first)

Before any simulation, establish **whether governors exist at all**. The
DIgSILENT "39 Bus New England System" template ships primarily with AVR
models; a missing turbine-governor slot would make the whole comparison moot
in a different and more important way.

For each in-service `ElmSym`, via `pf/session.py` in external engine mode
(follow the pattern in `pf/probes/probe_tap_avr.py`, which already walks
`ElmComp` per machine):

1. `mach.GetAttribute("c_pmod")` → composite model (`ElmComp`) or `None`.
2. Enumerate its slots (`pblk` / contents) and classify each DSL model: AVR,
   PSS, turbine-governor, PLL, etc. Report the frame name and every slot.
3. For any governor-class block, dump **all** parameters. You are looking for
   the droop and its base. Attribute names are model-dependent — likely
   candidates are `R`, `bp`, `Kdroop`, `sigma`, `droop`, plus a rating
   (`Pnom`, `Pgnn`, `sgn`) — **probe, do not assume**. `pf/probes/probe_api.py`
   already has the attribute-dump pattern.
4. Record `ElmSym.ip_ctrl` (reference-machine flag) and `av_mode` per machine.
5. Check for any active **secondary** controller (`ElmSecctrl`) or station
   controller that would act on P within the simulation window.
6. Check the load frequency dependence: `TypLod` attributes `kpf` / `kqf`
   (or equivalent) — these set `D`. Record whether they are zero.

**Decision point:**

- *No governor blocks on any machine* → the claim is confirmed in its
  strongest form, and the correct statement is stronger than the original:
  PF-RMS has **constant mechanical power** and cannot reproduce droop
  sharing at all. Report this, skip Step C, and propose what would need to be
  added (which governor models, what `R_pu`). This is a plausible outcome —
  say so plainly if you find it.
- *Governors exist* → continue.

**Analytic pre-check (no simulation needed).** From the collected droops
compute, per machine, `β_i = S_n,i / (R_pu,i · f_n)` on a common base, then

```
share_droop_i  = β_i / Σ_j β_j
share_pp_i     = S_n,i / Σ_j S_n,j
```

Tabulate both plus the difference in percentage points. If every `R_pu,i` is
identical, the two columns must agree to floating-point. **This table alone
largely settles the claim** — Step C then just confirms the analytics survive
contact with a real simulation.

---

## 3. Step B — the pandapower reference (run on either machine)

Model: **`base` phase at `t0`** — all ten machines present, parity closed at
this operating point (`export/snapshots/base_t0_20160105-0800.json`).
`wind_replace` is unsuitable: G2/G5/G6/G8 are removed there, and it does not
converge at `peakres` (build plan item 9).

Two contingencies, run each independently from the same snapshot:

- **C1 — generator trip:** `G9_bus37` (1000 MVA, mid-size, not the slack
  anchor, not a machine removed in later layers) set `in_service = False`.
- **C2 — load step:** a step of comparable magnitude (aim for the same ΔP as
  C1's dispatched output) applied to a single 345 kV load, so that the
  frequency response is exercised without changing the machine set.

For each: rebuild the net from the snapshot with
`export/dynamic_snapshot.py`, solve the base case with the stored
`solver_options`, apply the contingency, re-solve with the **same** options,
and record per machine

```
ΔP_i = res_gen.p_mw[after] − res_gen.p_mw[before]
```

Also record `Σ_i ΔP_i`, the loss delta, and — importantly — **whether any
machine's post-contingency `res_gen.p_mw` exceeds `max_p_mw`**. Report that
explicitly; it is the condition-(c) check and pandapower will not warn you.

---

## 4. Step C — the PowerFactory RMS reference

Use study case `02_RMS_CoSim` (not `01_LDF_Parity`). Follow the existing
plant wrapper (`pf/plant.py`) and event handling — note the build plan's
warning that the simulation-events folder is purged by the wrapper, and the
recording-stage ordering caveat in `docs/pf_api_notes.md` §3 (activate the
`base` variation only).

Setup requirements, each of which must be **verified and reported**, not
assumed:

1. **Same operating point.** Sync from the same snapshot; confirm the
   pre-contingency `res_gen.p_mw` match to the Gate-A/C tolerance before
   trusting any delta.
2. **AGC off.** No `ElmSecctrl` active. If one is active, `Δf → 0` and the
   settled sharing follows participation factors, not droop — a different
   experiment.
3. **Automatic tap controllers off** (already the `02_RMS_CoSim` convention).
4. **Simulate long enough to settle.** Run to at least `t = 120 s` after the
   event. Confirm settling by checking `|df/dt| ≈ 0` over the final 10 s —
   report the actual residual, do not just assert convergence.
5. Record per-machine electrical power at `t_end` and the settled `Δf`.

Then apply C1 (trip `G 09` by event) and C2 (the same load step), and record
`ΔP_i` per machine.

---

## 5. Step D — the comparison and the verdict

Produce one table per contingency:

| machine | S_n | ΔP_pp [MW] | share_pp [%] | ΔP_PF [MW] | share_PF [%] | Δshare [pp] |
|---|---|---|---|---|---|---|

**Compare normalised shares, not absolute MW.** The two models are not
directly comparable in absolute terms for two structural reasons, and your
report must state both:

- pandapower's static power flow has **no frequency variable**, so `D = 0`
  by construction and `Σ_i ΔP_i = ΔP_out + ΔP_loss`. PF-RMS settles at
  `Δf ≠ 0`; if PF loads carry non-zero `kpf`, then
  `Σ_i ΔP_i = ΔP_out · β/(β+D) < ΔP_out`, with the remainder absorbed by
  load self-regulation. Quantify `D` from Step A.6 and apply the correction
  before comparing, or zero `kpf` for the test and say that you did.
- The angle reference differs (template reference is at **Bus 31 / G 02**;
  the move to G 01 is a scripted pf_sync Phase-2 action). In RMS with active
  governors the reference machine sets only the angle datum and should not
  affect P sharing — **verify this rather than assuming it**, e.g. by
  repeating C1 with `ip_ctrl` on a different machine and confirming the
  shares are unchanged.

**Verdict criteria:**

- `max_i |Δshare_i| < 1 pp` → refuted; the models agree and PF governors
  already imply uniform per-unit droop.
- `1–5 pp` → confirmed, minor; report the implied per-machine `R_pu` spread.
- `> 5 pp`, or governors absent, or a saturation event in either tool →
  confirmed, material. Report which of conditions (a)–(d) breaks, and by how
  much.

Report the failing condition specifically. "They disagree" is not a result;
"G 05's governor droop is 4 % while G 10's is 6 %, so hydro under-contributes
by 8 pp relative to `slack_weight = sn_mva`" is.

---

## 6. Constraints and cautions

- **Do not modify the pandapower plant.** Read-only on `network/`, `core/`,
  `configs/`. PF-side changes must go through `pf/pf_sync.py` (scripted, never
  a GUI edit) and should be *proposed*, not applied, if they alter parity.
- PF engine mode consumes a licence seat and `GetApplication*` may be called
  only once per process (`docs/pf_api_notes.md` §1). Close the GUI first.
- If `TEMPLATE_NAMES_VERIFIED` or any Gate is found broken, stop and report —
  do not repair parity as a side quest.
- Report negative and partial results faithfully. If PF-RMS will not settle,
  or a governor model is unidentifiable, say so with the evidence rather than
  substituting an assumption.
- Output in the repo's standard format: (1) short but comprehensive answer,
  (2) assumptions used, (3) reasoning, (4) risks / unresolved points.
