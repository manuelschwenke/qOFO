# 2026-07-20 — Phase 5: WECC DER models — DER Q now RMS-controllable

**The Phase-5/6 blocker is cleared.** All 44 DER static generators now carry
a WECC RMS converter model whose reactive power tracks an external
reference in RMS — the handle the OFO needs, which `qsetp` could not provide.

## Method (collaborative → then fully scripted)

The user exposed the GUI via `App.Show()` (held open by a long-lived host
process; note `App.Hide()` segfaults on exit but releases the session
cleanly). That let me read what the headless engine hides. The decisive
discovery came from the library template internals, not the GUI:

- The ready template `Lib/Templ/TemplPv/WECC Large-scale PV Plant 110MVA
  60Hz` contains a **fully parameterised** composite `WECC Large-scale PV
  Plant` = REGC_C + REEC_D + Protection + Weak-Grid + **StaPqmea / StaVmea /
  StaImea measurement devices**. Those measurement devices were exactly what
  my earlier from-scratch build left empty, causing the silent ComInc
  failure.

**Working recipe** (`pf/wecc_apply.py`, idempotent):

1. `grid.AddCopy(template_composite)` into the park's own `ElmNet`.
2. Re-point Generator slot `pelm[0]` from the template's `ElmPvsys` to our
   `ElmGenstat`; re-point the three measurement devices
   (`StaVmea.pbusbar` = terminal, `StaPqmea/StaImea.pcubic` = cubicle).
3. Put REEC_D in **reactive-power control** (`PfFlag=0`, `VFlag=0`) so
   `REEC_D.Qext` (pu of the park's rated S) is the reactive reference.

The `ElmPvsys`-vs-`ElmGenstat` concern was a non-issue — the frame accepts
`ElmGenstat`.

## Verified (live, full model, `02_RMS_CoSim`)

- **ComInc green** with all 44 WECC composites active.
- **Q tracks Qext**: WP_TSO_s0_b18 (508 MVA) — `Qext=0.1` → Q = 50.8 Mvar;
  `Qext=1.0` → Q = 507.8 Mvar (full ±S_n, correctly scaled — no rescaling
  needed).
- **Init holds the load-flow operating point**; stepping one park leaves the
  others stable (DSO park drift 0.000).

## Architecture note

No separate REPC (plant controller) is needed: the OFO *is* the plant
reactive-power controller — it writes `REEC_D.Qext` directly. This matches
the project's cascade design (OFO dispatches Q setpoints to DER).

## New artefacts

- `pf/wecc_apply.py` — rollout script (all DER, or `--only <prefix>`,
  `--verify`).
- `pf/wecc_introspect.py` — dumps a composite's structure/parameters.
- `docs/pf_wecc_gui_build.md` — the GUI build spec (superseded by the
  scripted route, kept for reference).

## Remaining Phase-5/6 work

1. **Re-verify Gate C** (`01_LDF_Parity`) still passes with the WECC
   composites present — they are dynamic-only + passive measurements, so the
   load flow should be unaffected, but confirm.
2. **Rebuild the step battery** (`pf/screening.py steps`): target
   `REEC_D.Qext` for the DER Q steps (60 Mvar → in pu of each park's S_n);
   switch trajectory reading to `ComRes` CSV export (the per-cell reads
   timed out).
3. **Re-run modal** on the full model *with* the WECC DER dynamics (the
   earlier modal predated them; the converters add modes).
4. **Machine AVR step** still needs the correct DSL V-ref signal (minor).
5. **Phase 6:** `PowerFactoryPlant.apply_u` writes `REEC_D.Qext` (was
   `qsetp`); AVR `usetp`; tap events.

Model left with the 44 WECC composites in place (full variation), ComInc
green.
