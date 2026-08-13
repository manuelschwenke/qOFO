# 2026-08-02 — Per-level Q(V) dead band: the RMS plant could not represent δ_TS ≠ δ_DS

**Reason.** The next experiment varies the DER Q(V) dead-zone half-width
independently at the two voltage levels — δ_TS × δ_DS — under a load step on the
transmission side, and reads off the Pareto front over interface-Q tracking and
maximum voltage deviation. Before this change that sweep was not expressible.

## Defect

`configs/config.py` has carried three separate fields for a while:

| field | consumed by |
|---|---|
| `tso_qv_deadband_pu` | static plant, via `tag_der_q_modes` → `net.sgen.qv_deadband_pu` |
| `dso_qv_deadband_pu` | same |
| `der_qv_deadband_override_pu` | **blanket scalar**, both plants |

The static plant was never the problem: `tag_der_q_modes` has always written
per-level values into `net.sgen.qv_deadband_pu`.

The RMS plant was. `PFPlant._anchor_qv_precontrollers` (`pf/plant.py`) reads each
park's dead band from the **exported snapshot**, and
`export/make_snapshots.py:407` builds that snapshot from `MultiTSOConfig()`
*defaults* (`cfg = _cfg_defaults`), not from the config of the run being
executed. So the RMS parks were anchored at the default 0.01 pu regardless of
what the run asked for, and the only channel that reached PowerFactory was the
blanket scalar — **one number for every park at both levels**.

Two consequences:

1. δ_TS ≠ δ_DS was not representable in the RMS plant at all.
2. `experiments/helpers/rms_cosim_config.py` set all three fields from the
   single `--der-deadband` flag, which is why this never surfaced: every run so
   far pinned the three to the same value, and the blanket override — which
   takes precedence — happened to carry the intended number.

This is a correctness defect in the co-simulation's plant parity, not only a
missing feature: any run that set `tso_`/`dso_qv_deadband_pu` *without* the
blanket override had the two plants silently running different dead bands.

## Change

**`core/actuator_bounds.py`** — added `DER_QV_DEADBAND_BY_SGEN_PU`, a per-sgen
map, and `set_der_qv_deadband_by_sgen()`. Precedence in the RMS plant is now
`blanket scalar > per-sgen map > snapshot value`; the blanket is retained as an
explicit diagnostic.

**`experiments/runners/multi_tso_dso.py`** — step [4a] publishes the map from
`net.sgen.qv_deadband_pu` *after* `tag_der_q_modes` has tagged it, so both
plants are driven from **one column** rather than from two independently derived
numbers. Cleared at the top of every run so a stale map cannot leak between runs
in one process. Prints the TS and DS value sets.

**`pf/plant.py`** — anchor pass consults the map; prints `[qvpre] ... deadbands
applied: [...] pu`, recording what the RMS plant actually used. `print`, not
`logger.info`: **no logging handler is configured in this pipeline**, so the
pre-existing `logger.info("anchored %d Q(V) pre-controllers")` line has never
appeared in any run log.

**`experiments/helpers/rms_cosim_config.py`** — new `--tso-deadband` /
`--dso-deadband`. Either one clears `der_qv_deadband_override_pu`, because the
blanket would otherwise override the per-level values and collapse the matrix
onto its diagonal without any visible sign. `--der-deadband` alone is unchanged,
so every previous invocation reproduces exactly.

**`experiments/run_deadband_2d.ps1`** (new) — the 2D sweep driver.

**`analysis/deadband_2d.py`** (new) — collection, δ_TS × δ_DS cross-tabs and the
Pareto front. Its `ADMIT` requires `der_qv_deadband_override_pu: None`, which is
what keeps the 1D and 2D studies from contaminating each other: all 277 existing
runs carry the blanket and are rejected (verified).

## Behaviour change

A run that sets `tso_`/`dso_qv_deadband_pu` away from the default *without* the
blanket override now applies those values in the RMS plant, where previously it
applied the snapshot default. This is the fix. It is a no-op for every run
executed to date, because all of them set the blanket.

## Verification

Static plant, three configurations (`scratchpad/validate_split_deadband.py`):

```
TS=0.02 DS=0.0      map=[0.0, 0.02]   blanket=None   [4a] TS=[0.02] DS=[0.0] pu (44 parks)
TS=0.005 DS=0.02    map=[0.005, 0.02] blanket=None   [4a] TS=[0.005] DS=[0.02] pu (44 parks)
blanket 0.01 wins   map=[0.0, 0.02]   blanket=0.01   (map published, scalar takes precedence)
```

CLI resolution, including backward compatibility:

```
--der-deadband 0.005                    -> TS=0.005 DS=0.005 blanket=0.005
--tso-deadband 0.02 --dso-deadband 0.0  -> TS=0.02  DS=0.0   blanket=None
--der-deadband 0.01 --tso-deadband 0.005-> TS=0.005 DS=0.01  blanket=None
```

RMS leg, co-simulation smoke run (run 0278, `--tso-deadband 0.02
--dso-deadband 0.0`, localised +1100 MW step at bus 41):

```
[4a]    Q(V) deadband published to both plants: TS=[0.02] DS=[0.0] pu (44 parks)
[qvpre] anchored 44 Q(V) pre-controllers; deadbands applied: [0.0, 0.02] pu
```

The RMS plant applies both values. Before this change it would have applied
0.01 pu to all 44 parks — the snapshot default — with no indication in any log.

Contamination guard, both directions:

- 2D runs into the 1D studies: `analysis/deadband_selection.uniform_deadband()`
  now gates `deadband_selection`, `deadband_disturbance` (which additionally
  rejects `load_step_bus is not None`, since its disturbance is the uniform
  multiplicative step) and `deadband_threshold` (collection *and* twin lookup).
  Verified a no-op: of 278 runs with a `runner_static` block, the only
  off-diagonal ones are the two smoke runs above. Admitted counts unchanged at
  47 undisturbed / 41 stepped.
- 1D runs into the 2D study: `ADMIT` requires `der_qv_deadband_override_pu:
  None`, which all 277 pre-existing runs carry. Verified: 0 admitted.
- Short validation runs into the 2D study: `ADMIT` requires `n_total_s: 300.0`,
  so the 160 s smoke runs cannot pair with a full-length twin.

## Open

- The snapshot remains exported from `MultiTSOConfig()` defaults. The map now
  covers the dead band; **any other droop parameter set per-run still reaches
  the RMS plant only as the snapshot default.** `qv_slope_pu` has the same
  structure and is not currently swept, but a slope study would hit this.
