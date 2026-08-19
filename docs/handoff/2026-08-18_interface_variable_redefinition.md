# Handoff — define the DSO's tracked variable as the interface flow *net of TSO-owned tertiary compensation*

**Date:** 2026-08-18 · **Scope:** dissertation prose + notation only · **Code change required: none.**

The cascade already computes the right numbers. What it lacks is a *name* for
what the subordinate controller tracks, and the absence of that name is why the
setpoint feedforward and the capability-band shift read as two independent
bookkeeping rules that must be kept in sync by hand — the failure mode measured
on 2026-08-13 (`00_daily_log/2026-08-13_capability_open_loop.md`).

Stating the definition makes the band shift a *consequence* of the definition
rather than a rule that can drift away from it. The closed loop is bit-identical.

---

## 1 — The definition to introduce

Let interface transformer `t` host a TSO-owned tertiary bank. Define the
**estimated contribution of TSO-owned tertiary compensation to the measured
primary-port reactive flow**, accumulated over the commits up to instant `k`:

```
  Qsh_t(k) = SUM over commits c at t up to k of  (dQ_hv,t / dQ_sh,c) * dQ_sh,c
```

and the **delegated interface variable**

```
  qtilde_t = q_hv,t  -  Qsh_t                                            (DEF)
```

The subordinate controller tracks `qtilde_t` against the supervisory decision
variable `u_pcc,t`. The metering point does not move: `q_hv,t` is still
`res_trafo3w.q_hv_mvar` at the EHV port, and settlement is untouched.

**Sign check — do not get this wrong.** `Qsh` is the contribution *to* the
measured flow, so it is **subtracted**. The code accumulates exactly this
quantity in `q_itf_sh_offset` and *adds* it to the setpoint; subtracting it from
the measurement is the same equation rearranged. (An earlier draft of this
proposal wrote `qtilde = q_hv + Qsh`; that sign is wrong.)

Suggested framing in the text: this is ordinary **disturbance feedforward** —
the subordinate layer regulates the boundary flow net of a known,
supervisory-owned exogenous input. Naming it that way gives the mechanism a
standard pedigree instead of presenting it as an ad-hoc correction.

---

## 2 — Why it is exact (three identities, each with its code anchor)

The redefinition touches exactly three objects. All three are identities, so
nothing numerical changes.

**(I1) The tracking error.** Today: `e = q_hv - (u_pcc + Qsh)`. Redefined:
`e = qtilde - u_pcc`. Same expression.

- `controller/dso_controller.py:1209` — `q_error = q_interface - self.q_setpoint_mvar`
- `experiments/runners/multi_tso_dso.py:4002` — `q_adj[_ii] += q_itf_sh_offset...`

**(I2) The capability band.** The DSO reports deltas `[dmin, dmax]` measured on
`q_hv`; the runner hands the TSO `[dmin - Qsh, dmax - Qsh]`, and the TSO anchors
the band at the raw `q_now`. So the band in force is
`[q_now - Qsh + dmin, ...]` = `[qtilde_now + dmin, qtilde_now + dmax]` — i.e.
the DSO's own unshifted envelope, anchored on `qtilde`.

- `experiments/runners/multi_tso_dso.py:4089-4091` — the `-= _off` shift
- `controller/tso_controller.py:1266-1279` — band anchored at `q_iface_now`

**(I3) Every sensitivity is unchanged.** `Qsh` depends only on the bank levels,
which are *not* decision variables of either optimiser in the shipped
configuration: with `shunt_dispatch="integrator"` the per-zone shunt lists are
left empty, so `TSOControllerConfig.shunt_bus_indices == []` and `H` carries no
shunt column at all. Hence `d(qtilde)/du = d(q_hv)/du` for every input `u` on
either side, including the supervisory identity column `dQ_PCC/dQ_PCC,set = 1`.

- `experiments/runners/multi_tso_dso.py:899-913` — empty shunt lists in integrator mode
- `controller/tso_controller.py:1925-1935` — the identity column
- `controller/dso_controller.py:1101-1105` — the DSO places no hard bound on the
  interface row (`+/-1e6`), so no output constraint needs restating.

**Conclusion to state in the thesis:** the supervisory layer commands, and the
subordinate layer tracks, the boundary flow *net of supervisory-owned tertiary
compensation*. The capability envelope is reported on the same variable. One
definition; the feedforward and the band shift are its two consequences.

---

## 3 — Concrete edit list

The chapter sources are not in this repo. Locate the targets by searching for
the setpoint-dispatch and capability equations (the E1 work touched
`Chapters/Chapter06.tex`, label `ch:architectures:cascade:aggregation`).

1. **Add (DEF)** where the interface variable is first introduced, with the
   sentence that the metering point is unchanged and only the *tracked* variable
   is net of compensation.
2. **Restate the DSO tracking objective** on `qtilde`, and delete any wording
   that presents the feedforward as a correction *added to the setpoint*.
3. **Restate the capability report** as an envelope on `qtilde`; the `-Qsh`
   shift then needs no separate justification — remove any text that argues for
   it as a separate rule.
4. **Keep one explicit sentence** that `Qsh` is an estimate and not a meter
   (see §4), so the definition is not read as exact bookkeeping.
5. **Do not** change any result, figure or number. Nothing was re-run.

---

## 4 — What must NOT be claimed

These are the honest limits. Each is measured; do not soften them.

1. **`qtilde` is a meter minus a model, not a measurement.** `Qsh` is a
   first-order estimate from `compute_dQtrafo3w_hv_dQ_shunt`, evaluated at the
   pre-switch operating point, accumulated and never reconciled against
   measurement. Measured per-commit error (2026-08-13): estimate −23.31 vs
   metered −18.74 Mvar on DSO_2's bank (~22 % over-statement); −20.47 vs −21.33
   on DSO_3's (essentially exact). The redefinition **relocates** that error
   into the definition of the tracked variable; it does not remove it.
2. **Host transformer only.** The offset is booked to the transformer hosting
   the bank (`experiments/runners/multi_tso_dso.py:3947`). Siblings in the same
   DSO area take a real share of the step — measured 2026-08-18: −6.3 and −5.6
   Mvar of a −65.8 Mvar area total, ~9 % each — and are **not** compensated. So
   `qtilde` is net of the host's own bank, not of all supervisory compensation.
   Sibling compensation was measured (26–48 % at-commit improvement) and
   deliberately reverted.
3. **Known MSR bias.** A reactor depresses the voltage and a constant-susceptance
   device then delivers `B*V^2`, while the linear estimate is symmetric —
   over-statement up to 36 %.
4. **Incremental, not absolute.** `q_itf_sh_offset` starts at zero
   (`experiments/runners/multi_tso_dso.py:3034`) and accumulates only changes.
   This coincides with the absolute definition **because the banks start
   de-energised** (`controller/shunt_integrator.py:197`). If any run warm-starts
   a bank, (DEF) must read "relative to the initial bank state".
5. **The disturbance channel does not disappear.** A bank step still moves the
   DSO's own 110 kV buses by 0.7–3.3 pu-% (measured 2026-08-18), so the
   subordinate voltage problem is disturbed either way and
   `ShuntDisturbanceMessage` remains necessary. The redefinition concerns the
   *tracked variable* only — say so explicitly, or a reader will conclude the
   layers are decoupled.

---

## 5 — Numbers measured 2026-08-18 (available to quote)

Plant: IEEE 39 + four 110 kV sub-networks, `rural_700`, 2016-01-05 08:00,
4.76 GW load, `run_control=False`, central finite differences. Couplers are
lightly loaded at this point (S/Sn 0.03–0.24).

| quantity | value |
|---|---|
| coupler reactive consumption | 0.14–1.79 Mvar/transformer, median 0.41; 2.2 Mvar per DSO area |
| marginal port gain `dq_hv/dq_mv` | 0.989–0.992, near-invariant to ×7 DS load |
| MSC step (≈54 Mvar), host transformer | `dq_hv` −53.9, `dq_mv` −11.8 Mvar |
| MSC step, sibling transformers | `dq_hv` −6.3 / −5.6 Mvar |
| MSC step, netted over the DSO's three interfaces | `dq_hv` −65.8, `dq_mv` +0.18 Mvar |
| coupler tap ±1 | `dq_hv` −4.2 … −9.6 Mvar (`dq_mv` equal within 2 %) |
| bank step → DSO's own 110 kV buses | 0.7–3.3 pu-% |

Context for why the metering point stays at the EHV port: the SBX-V settlement
plane is defined on `q_hv_mvar` at the EHV port — DP3, confirmed 2026-07-09,
`docs/status/STATUS_SBXV.md:485`.

---

## 6 — Verification the writing session can run

No numerical check is needed for the redefinition itself; §2 is algebra. To
confirm the anchors still say what this document claims:

```bash
grep -n "q_itf_sh_offset" experiments/runners/multi_tso_dso.py
```

Expect: init at `:2996` / `:3034`, accumulation at `:3947`, setpoint add at
`:4002`, band shift at `:4089-4091`.

---

## 7 — Open

- Whether to *also* compensate the siblings is a live design question, not
  settled by this redefinition. Under (DEF) the uncompensated sibling share
  becomes visible as a definitional gap rather than as a tracking error, which
  is an argument for revisiting the 2026-08-13 revert — but that is a code
  change and out of scope here.
- The measurements in §5 are one operating point with no controllers in the
  loop. The coupler-consumption figures scale as (S/Sn)², so at rated loading
  the term is `vk_mv * Sn = 24 Mvar` per transformer. Quote the light-loading
  numbers only with that caveat.
