# 2026-08-02 — N-1 generator outages now run on the RMS plant

**Reason.** The dead-band experiment needs a disturbance that is both credible
and strong enough to engage a wide dead zone. A load step is neither: the only
amplitude that settles (+400 MW at EHV bus 41) moves the worst park 0.0035 pu,
and 400–1100 MW at one node is not a credible single contingency. A generator
outage is a textbook N-1 and moves the worst park **0.0104 pu (gen 7) to
0.1025 pu (gen 2)** — measured statically at 2016-01-05 08:00.

The impact is driven by the **lost AVR voltage support**, not by the machine's
active power: gen 7 is the largest unit (830 MW) and produces the *weakest*
disturbance, while gen 0 (250 MW) moves the TS parks 0.0539 pu. That is the
right physics for a reactive-power study.

## Why it did not work before

`experiments/runners/multi_tso_dso.py` refused **all** contingencies for any
non-static plant. The refusal was correct in substance — PowerFactory reads
element input attributes only at initialisation, so writing `in_service` on the
mirror leaves the simulator on the pre-contingency topology — but it was
wholesale, and the RMS half of the machinery already existed:

- `pf/screening.py::add_outage_event` creates an `EvtOutage`, verified
  2026-07-26 on machine `G 10`;
- `pf/plant.py::_read_m` already tolerated the disappearance of an element's
  `m:` result variables after an outage, with a comment stating that "any N-1
  study therefore needs this path".

Only the wiring between the runner and those two pieces was missing.

## Defects found while checking

**1. `add_outage_event` did not fold the event time into PF's 60 s window.**
`add_tap_event` does (`EVENT_WINDOW_S`, established over 30+ measurements);
`add_outage_event` wrote the absolute time. Its 2026-07-26 verification armed
the event at t = 5 s — inside the first window, where the fold is a no-op — so
the bug could not show. An outage armed at a calculation clock ≥ 60 s would
have fired `60·floor(clock/60)` s late, or never if the run ended first, i.e.
**the run would silently contain no contingency at all.** Fixed.

**2. Every Gate E report asserted "profiles and contingencies are disabled".**
Hard-coded, and false for every profiled run since 2026-07-21. Now generated
from the actual config (profiles on/off, the contingency list, any load step).

## Change

| file | change |
|---|---|
| `pf/screening.py` | `add_outage_event` folds into the current event window |
| `pf/plant.py` | `apply_contingency()` + `supports_contingency()`; trips the ElmSym **and** its machine transformer, matching what `_apply_contingency` does to the mirror |
| `core/plant.py` | `apply_contingency` on the `Plant` protocol; no-op on `PandapowerStaticPlant` (there `net` *is* the plant) |
| `experiments/runners/multi_tso_dso.py` | calls `plant.apply_contingency`; gate now refuses only event **types** the plant cannot deliver; the post-contingency `pp.runpp` recovery ladder is restricted to the static plant, because on a non-static plant `net` is the measurement mirror and re-solving it would overwrite the PF measurements with a static solution |
| `experiments/helpers/rms_cosim_config.py` | `--trip-gen`, `--trip-time` |
| `experiments/run_comparison_rms_cosim_qss.py` | Gate E summary states the real exogenous drive |

Only `gen`/`trip` is supported. Line trips, load connect/shed and every
`restore` still raise: `EvtOutage` could express them, but none is verified
against this adapter, and an unverified topology change is worse than an
explicit refusal — it would leave the mirror and the simulator on different
networks with no visible symptom.

## Event timing convention

The outage is armed at `plant.t + 0.5 s`, which for a trip nominally at
t = 100 s reads as **80.5 s** in the log. This is correct, not an off-by-one:
at the dispatch step labelled T the RMS plant has simulated only to T − dt and
advances to T at the end of the step, so an event armed at `self.t + EPS` fires
inside the interval whose measurements are reported for step T — exactly when
the static plant's mutate-and-resolve makes the outage visible. It is the same
convention `apply_exogenous` uses for profiles. Arming at the nominal absolute
time instead would push the outage into the *next* step's measurements and put
the two legs one interval apart. The log message states both times.

## Verification

Static plant, unchanged (regression, 4 configurations incl. δ_TS ≠ δ_DS):

```
no outage (baseline)      OK  36.1 s   records=15
gen 7 trip @100s          OK  73.2 s   *** CONTINGENCY t=100s: TRIP gen 7 (G9_bus37) + machine trafo 10
gen 1 trip @100s          OK 101.3 s
gen 1 trip, dTS=0.02 dDS=0 OK 74.3 s
```

RMS co-simulation (`--trip-gen 7`), previously `NotImplementedError`:

```
*** CONTINGENCY t=100s: TRIP gen 7 (G9_bus37) + machine trafo 10
[rms-contingency] EvtOutage ... armed at RMS t=80.5s ... nominal t=100s
('gen', 7) left service (m:P:bus1 unavailable); reading m:P:bus1 as 0.0 until it returns
```

The third line is the proof that PF actually tripped the machine: PowerFactory
deletes an element's `m:` result variables when it leaves service, and that is
the only reliable detector — `outserv` is **not** updated by `EvtOutage`, the
same trap as tap positions under `EvtTap`.

## Open

- Tripping the machine transformer in PF alongside the machine is implemented
  for mirror parity but has not been verified independently of the machine
  trip.
- QSS/RMS agreement under an N-1 is unproven; a severe trip may break Gate E
  legitimately, since Gate E validates equivalence and the two plants have
  genuinely different dynamics through a topology change.
- `gen 9` (1000 MW) diverges in the static outage scan and must be excluded
  from the candidate set.
- Run cost: QSS 36–101 s against RMS 13–17 min, so a full factorial belongs on
  the QSS plant with RMS confirming selected cells.
