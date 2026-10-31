# New base case: merit-order unit commitment, limited network equivalent

Timestamp: 2026-10-31, ~00:45–04:00 Europe/Berlin.
Branch `res/commitment-base-2026-10-31`, from `weekend/scenario-E` at c375de8. Worktree
`F:\qOFO_res`.
Session: side session of the Case-2 main session, at the author's direct instruction (31 Oct,
~00:40). The author:
- replaces the base case everywhere;
- uses a merit order;
- limits G1;
- wants REF and MC started on 20 workers per host (STEP left for now);
- asks to ensure convergence first and to re-tune if needed.

## Why

- The previous dispatch spread each zone's residual load over all machines in proportion to S_n,
  with P_min = 0, and never decommitted a unit.
- Every synchronous machine therefore stayed on line all year with its full Q range:
  - G3 below 20 % of rating in 25 % of hours, often at 0 MW;
  - G1, the 10-GVA equivalent, the reference bus with unlimited Q in the plant.
- The renewable-dominated hours in the profile year (RES share > 50 % in 24 % of hours) were thus
  Q-rich by construction. That hid the condition in which a TSO needs reactive power from the STS.
- Analysis and figures: `docs/analysis/2026-10-31_res_dominated_dispatch_design.md` in the
  `case2-ref` worktree, commit dc893c9.

## What changed (key structures)

| file | change |
|---|---|
| `configs/config.py` | **new fields:** `unit_commitment` = True; `gen_min_stable_frac` = {Nuclear 0.5, Coal 0.4, Hydro 0.1}; `gen_merit_order` = (Hydro, Nuclear, Coal); `slack_q_limit_mvar` = 500 |
| `network/ieee39/zonal_balancing.py` | new `compute_merit_order_dispatch` and `_zone_residual_series` (details below); the old `compute_zonal_gen_dispatch` is untouched |
| `controller/slack_q_limit.py` (new) | `SlackQLimit` (details below) and `install_slack_q_limit` |
| `experiments/runners/multi_tso_dso.py` | step [9]: commitment branch (details below). Step [10.3]: installs the G1 limit |
| `sensitivity/network_reduction.py` | `build_tso_local_net`: a zone with no machine in service takes its angle reference from its first boundary source (WARD_TIE_TH/PV). Before, it failed with "no reference bus" |
| `core/plant.py` | new `runpp_holding_taps` (details below), used by `PandapowerStaticPlant.advance` and the runner's post-contingency and fallback solves |
| `experiments/ch_11_case2_multisystem/ref/scenarios.py` | `start_time(name)`; new scenarios **RA** and **RE** (details below) |
| `experiments/ch_11_case2_multisystem/ref/run_ref.py` | applies the scenario start; the manifest records start, `unit_commitment` and the G1 limit |

**`compute_merit_order_dispatch`**
- Per zone, units are taken in merit order. A unit is committed while the zone's peak residual
  over the run window, less the caps of the units before it, is at least its technical minimum.
- Committed units run between their minimum and 0.85 S_n.
- A zone deficit goes to the headroom of committed units elsewhere, in merit order. The remainder
  (positive or negative) is the import of the equivalent.

**`SlackQLimit`**
- A plant-side pandapower controller.
- While G1's Q is inside ±limit, its terminal voltage is held at the set-point. At a limit, a
  secant Newton step on vm_pu holds Q on the limit and the voltage floats (PV → PQ), as
  `enforce_q_lims` does for other machines.

**Runner, step [9] commitment branch**
- Decommitted units and their machine transformers go out of service via `_apply_contingency`.
- TS STATCOM Q is reset to 0 before the post-profile PF; Phase 1 recomputes it anyway.
- The outage masks of the coordinator and of CEN are updated, all before the post-Phase-2
  Jacobian rebuild.

**`runpp_holding_taps`**
- Used only if the plant control loop raises `ControllerNotConverged`: it re-solves once with the
  local `DiscreteTapControl` relays held.
- Cause: with few machines on line, a coupler relay hunts inside one control loop against the DER
  Q(V) loops (each converges alone).

**Scenarios RA and RE**
- Rule: the earliest 6-h window of the profile year, outside `run_mc.EXCLUDED_DAYS`, starting
  between 06:00 and 12:00, in which one synchronous machine is committed.
- Result: 2016-03-13 09:00, with only G10 (hydro) and the equivalent on line.
- RA carries A's events, RE carries E's events.

`unit_commitment=False` and `slack_q_limit_mvar=None` reproduce the previous base. Measured: OFO,
scenario A, 4 min against the weekend code: max |ΔV| = 0.0, max |ΔQ_DER| = 0.0.

## Checks

- **Smoke**, every Case-2 variant, 10 min, scenario A, new base: all six ok. G3 is decommitted at
  the reference period (zone-2 peak residual 123 MW < 400 MW minimum).
- **Year scan** (`year_commit.py`, shared notes folder), 3-h windows as in the MC:
  - 5 machines committed in 59 % of windows, 4 in 17.5 %, 3 in 6.5 %, 2 in 14.7 %, 1 in 2.5 %;
  - G3 committed in 75 %.
- **Convergence before the campaign:** **not met** in the renewable-dominated regime. Details,
  diagnosis and the decisions needed are in `docs/analysis/2026-10-31_commitment_base_status.md`.
  - At the reference period (all five machines committed over 08:00–14:00), REF A/B/D/E converge,
    and so do their post-event states.
  - LOC loses voltage stability without events at the W1 evening peak and in W6 (G1 at its limit;
    W6 also at ±1000 Mvar).
  - After N-1 events in two- and one-machine periods, the plant has no solution. On 8 Jan this is
    plant numerics: the state is solvable with ideal TS-DER voltage control.
- **Found and fixed on the way:**
  - (i) a zone net without a machine (boundary source as reference);
  - (ii) the post-profile PF from the build-state STATCOM Q;
  - (iii) relay hunting (relays held once, also at the plant seam);
  - (iv) G1-limit chatter (10-Mvar tolerance, 30-Mvar hysteresis, c05da69).
  - A PV warm start for post-event solves was tried and removed (it did not help).

## Not changed / open

- The weights stay c08b1d3d33ab.
  - The re-validation on the new base is not launched, since W1 and W6 do not converge under LOC
    with G1 at ±500. It needs the G1 decision first.
  - The full Ch 9 procedure stays planned.
- STEP is not rerun.
- The running old-base jobs (STEP on e5wimi, MC on e5power, the queued old-base E) could not be
  stopped from the session: the action was denied in auto mode.
- Queued on e5wimi: REF A, E (after the old E) and B, D (after the old STEP); chain
  `results/res_logs/chain_res.py`, pid 146552.
- Not launched: the e5power re-validation and MC.
- An outage event on a decommitted machine (B/D/E's G3, MC gen trips) changes nothing.
