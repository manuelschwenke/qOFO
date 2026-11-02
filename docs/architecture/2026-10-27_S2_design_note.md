# S2 — classical SVR with sub-transmission access: design note for review

2026-10-27, Claude Code for M. Schwenke. Status: **proposal, not implemented**.
It implements the thesis definition `eq:svr-alignment-sts` (Ch 7, `ch:setup:references:svr`,
paragraph "Classical reference with sub-transmission access") and the campaign note
`latex_diss_clone/00_daily_log/2026-10-06_part3_case_design.md` §3.

## 1. What S2 is

S2 = S1 (integral RVR + RPR, safeguard on, T_RVR = 300 s, taps held, no banks) plus:

- **the couplers of the area's child STS join the RVR alignment** as participants;
- **each STS runs the STS-OFO + DAD exactly as in O3** and tracks the interface reference
  Q_j^ref the RVR assigns.

So S1 → S2 isolates the access under the classical law, and S2 → O3 changes only the
transmission law (with its objective and discrete devices, as Case 1 decomposes).

## 2. Quantities and sign convention

The CAIR message (`core/message.py::CapabilityMessage`) carries **changes**
[Δq_min, Δq_max] from the current flow, in **load convention at the HV port**: positive Q
flows from the TS bus into the coupler. Supporting the TS voltage means *lowering* that flow.

Per coupler j (load convention, absolute values):

| Symbol | Meaning | Source |
|---|---|---|
| Q_j | measured interface flow | STS measurement (noisy, minimum class) |
| [Q̲_j, Q̄_j] = Q_j + [Δq_min, Δq_max] | CAIR interval | existing CAIR message |
| Q_0,j = Q_j − Σ_i s_ji Q_IBG,i | **neutral point**: the flow with every STS unit at Q = 0 | **new scalar in the CAIR message**, computed by the STS-OFO from its own CAIR snapshot `_H_iface_der_open_bus` and its measured unit Q |

Q_0,j lies inside [Q̲_j, Q̄_j], because zero output lies inside every unit envelope
(VDE-AR-N 4120 and STATCOM), and CAIR is the linear projection of those envelopes.

**Alignment.** The level q_a ∈ [−1, 1] is the S1 level; positive means raise the voltage, as
for the machines (`eq:svr-alignment`):

    q_a ≥ 0:  Q_j^ref = Q_0,j − q_a · (Q_0,j − Q̲_j)    (toward more injection into the TS)
    q_a < 0:  Q_j^ref = Q_0,j + |q_a| · (Q̄_j − Q_0,j)

This is `eq:svr-alignment-sts` written in load convention. It gives the STS the same relative
utilisation as the machines of the area, measured from the STS's neutral point instead of
from zero output.

## 3. Pilot sensitivity

`eq:svr-rvr` normalises the RVR by s_p,a = Σ_i ∂v_p/∂Q_i · Q̄_i. With the couplers:

    s_p,a = Σ_i ∂v_p/∂Q_i · Q̄_i  +  Σ_j ∂v_p/∂Q_inj,j · (Q_0,j − Q̲_j)

- ∂v_p/∂Q_inj,j is a finite difference of the pilot voltage to a reactive injection at the
  coupler's TS-side bus, in the same `calibrate()` on a copy of the plant as for the converters.
- The headroom (Q_0,j − Q̲_j) is taken from the **first** CAIR message and **held**, like the
  machine headroom inside s_p,a. The participation structure is fixed, as in classical SVR.
- Effect: the per-dispatch gain stays b = T_TS/T_RVR = 0.6. Without the coupler term the extra
  participants would raise S2's loop gain above S1's, and S1 → S2 would mix access with tuning.

## 4. Data flow and timing (runner)

| Step | O3 today | S2 (proposed) |
|---|---|---|
| STS → TS, every 20 s | DSO `generate_capability_message` → `TSOController.receive_capability` | the same message, also delivered to the area's `SVRZoneController` (new `receive_capability`), carrying Q_0,j |
| TS → STS, every 180 s | `TSOController.generate_setpoint_messages` → `DSOController.receive_setpoint` | the SVR fills the **PCC block** of its `u` (held at 0 today) with Q_j^ref at each RVR update. The runner sends it as a `SetpointMessage` through the same `receive_setpoint` (RUN:4425-4450 path) |
| STS inner loop | STS-OFO + DAD, N_inner = 9 | identical: same weights, DAD, OLTC treatment |
| bank feed-forward offset | `q_itf_sh_offset` | none: S2 has no TS banks (like S1) |

Config: new flag `svr_sts_access: bool = False`, with `tso_mode="svr"` and `dso_mode="ofo"`.

Pair guard:
- (S1, S2) may change {`dso_mode`, `svr_sts_access`, the three STS-label fields};
- (S2, O3) may change the SVR fields ∪ {`tso_oltc_mode`, `install_tso_tertiary_shunts`,
  `shunt_dispatch`, `svr_sts_access`}.

## 5. Tests

1. Unit: alignment with synthetic headrooms. Signs: q_a > 0 lowers Q_j^ref below Q_0,j, and
   q_a = ±1 reaches the interval edges.
2. Unit: a zero-width interval (Q̲ = Q̄ = Q_0) leaves every machine and TS-IBG output identical
   to S1 at the same level, and the PCC block equals Q_0. The coupler drops out of the alignment.
3. Unit: Q_0,j from the DSO equals Q_j − Σ s_ji Q_i on a synthetic sensitivity.
4. Smoke: S2 for 10 min. Outcome `ok`, set-points inside the CAIR interval at every dispatch.
5. Guard: `assert_case2_controlled` with (S1, S2) and (S2, O3).

## 6. Points for the author

1. **Neutral point as an extra CAIR field** (one scalar per coupler). The thesis already says
   the STS "discloses one additional scalar per coupling transformer". Confirm.
2. **s_p,a with the coupler term, headroom frozen at the first CAIR message** (§3). The
   alternative, current headroom at every RVR update, makes the normalisation time-varying,
   which classical SVR does not do.
3. **Q_j^ref is issued at the RVR rate (180 s)**, not at the RPR rate (20 s). The STS-OFO
   then has the same N_inner = 9 as in O3. Issuing it at 20 s would make the coupler a
   faster participant than the machines' RVR level allows.
4. If noise or linearisation puts Q_0,j outside [Q̲_j, Q̄_j], it is clipped to the interval
   and the clip is counted in the record.
