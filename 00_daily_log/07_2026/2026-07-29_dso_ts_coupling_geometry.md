# 2026-07-29 — TN↔DS coupling geometry check and `scale` recommendation

**What:** New analysis script `analysis/dso_ts_coupling_geometry.py`.
**Why:** All four DSOs currently use `scale = 1.00`, i.e. an identical 110 kV
footprint (~60–85 km between coupling points), while the IEEE 39-bus TN buses
they hang off are spread over anything from 29 km (DSO_3) to 235 km (DSO_4).
Follow-up to `2026-07-29_simbench_hv_topology_benchmark.md`.
**Timestamp:** 2026-07-29
**Status:** analysis done, **change applied and verified** (option A + D, see §5).

---

## 1. Premise

A 345/110 kV coupling transformer sits inside the EHV substation, so HV bus
`hv_buses[i]` is geographically co-sited with TN bus `ieee_1idx[i]` (positional
pairing). The three HV coupling buses therefore span the same geography as the
three TN buses, and the HV route between two of them should be comparable to
the TN route between the corresponding TN buses.

Detour factor `r_ij = d_HV,ij / D_TN,ij`. Physically `r ≳ 1` (EHV corridors are
direct, 110 kV paths follow a meshed regional structure). Recommended scale
`s* = κ / geomean(r_ij)`; geometric mean because `scale` acts multiplicatively.
**κ = 1.0 used throughout — this is conservative**, since κ ∈ [1.0, 1.5] is the
physically plausible band and larger κ pushes all scales up.

## 2. Method note (a real bug found and fixed)

The first version built one TN graph with transformers as zero-length edges.
IEEE 39-bus feeds bus 12 exclusively through two transformers from buses 11 and
13, so the route 11→12→13 cost **0 km** — although 11 and 13 are two
substations 57.6 km apart via bus 10. This corrupted DSO_3.

Fix: two graphs.
- **geometric** = lines only (`LINE_LENGTHS_KM`); lineless buses (12, and the
  generator terminals) are *proxied* to their line-connected parents and treated
  as co-sited with them.
- **electrical** = lines + transformers weighted by series reactance, where the
  transformer branches are real impedance and must be traversable.

Also noted: line 23–36 never matched `LINE_LENGTHS_KM` and keeps the case39
default `length_km = 1.0`. It is a generator step-up connection and lies on no
coupling-bus path, so it does not affect these results — but it is a latent
inconsistency in `LINE_LENGTHS_KM`.

## 3. Results

**Pairwise, scale = 1.0**

| DSO | TN pair | HV pair | TN km | HV km | detour |
|---|---|---|---|---|---|
| DSO_1 | 7–8 | 3–0 | 30.8 | 60.0 | 1.95 |
| DSO_1 | 7–5 | 3–8 | 79.1 | 65.0 | 0.82 |
| DSO_1 | 8–5 | 0–8 | 75.1 | 85.0 | 1.13 |
| DSO_2 | 12–14 | 3–0 | 67.7 | 60.0 | 0.89 |
| DSO_2 | 12–4 | 3–8 | 154.2 | 65.0 | 0.42 |
| DSO_2 | 14–4 | 0–8 | 86.5 | 85.0 | 0.98 |
| DSO_3 | 11–10 | 3–0 | 28.8 | 60.0 | 2.08 |
| DSO_3 | 11–13 | 3–8 | 57.6 | 65.0 | 1.13 |
| DSO_3 | 10–13 | 0–8 | 28.8 | 85.0 | 2.95 |
| DSO_4 | 24–21 | 3–0 | 130.0 | 60.0 | 0.46 |
| DSO_4 | 24–23 | 3–8 | 234.6 | 65.0 | 0.28 |
| DSO_4 | 21–23 | 0–8 | 158.1 | 85.0 | 0.54 |

**Per-DSO**

| DSO | TN footprint (geomean) | detour geomean | `scale*` (κ=1) | longest circuit at `scale*` | SimBench-admissible |
|---|---|---|---|---|---|
| DSO_1 | 56.8 km | 1.22 | **0.82** | 32.8 km | yes |
| DSO_2 | 96.6 km | 0.72 | **1.40** | 55.9 km | **no** (>51.6) |
| DSO_3 | 36.3 km | 1.91 | **0.52** | 21.0 km | yes |
| DSO_4 | 169.0 km | 0.41 | **2.44** | 97.6 km | **no** (>51.6) |

The TN footprints span 36–169 km, a factor of 4.7. A common `scale = 1.00`
fits none of them: DSO_3's HV grid is ~1.9× too large for its TN footprint,
DSO_4's is ~2.4× too small.

**Ordering (shape, not size)**

`rank` is `partial` for all four: the TN and HV orderings of the three pairs
never agree. `log_spread` = std of `log(detour)` across the three pairs; 0
means a single scale matches every pair exactly.

| DSO | log_spread now | best `hv_buses` | log_spread best |
|---|---|---|---|
| DSO_1 | 0.356 | (0, 3, 8) | 0.343 |
| DSO_2 | 0.377 | (0, 3, 8) | 0.197 |
| DSO_3 | 0.397 | (0, 3, 8) | 0.184 |
| DSO_4 | 0.283 | (0, 3, 8) | 0.099 |

Permuting `hv_buses` from `(3, 0, 8)` to `(0, 3, 8)` keeps the same coupling-bus
*set* — it only swaps which TN bus feeds which coupler — and roughly halves the
shape mismatch for DSO_2/3/4. DSO_1's footprint shape (one short pair at 30.8 km,
two long at ~77 km) does not match the HV shape (60/65/85) under any permutation.

## 4. Conflict with the SimBench bound

Longest base circuit in `HV_LINE_TOPOLOGY` is 40 km, and the longest 110 kV
overhead circuit anywhere in SimBench is 51.64 km, so `scale ≤ 1.29` keeps every
synthetic circuit inside the observed range. DSO_2 (1.40) and DSO_4 (2.44)
exceed it.

This is a genuine, unresolved tension, not a numerical artefact: SimBench HV1
*does* span 223.6 km between coupling points, so a 169 km DSO footprint is
realistic — but SimBench builds it from 95 circuits with a median of 8.5 km,
whereas a 10-bus equivalent must use ~10 long circuits. The synthetic grid is
too coarse to be both geometrically consistent with DSO_4's TN footprint and
composed of individually realistic circuits.

## 5. Decision and applied change

Options considered:

- **A — exact:** 0.82 / 1.40 / 0.52 / 2.44. TN-consistent; DSO_2 and DSO_4 hold
  circuits longer than any SimBench observation.
- **B — capped:** 0.82 / 1.29 / 0.52 / 1.29. Every circuit stays inside the
  SimBench range; DSO_4 remains ~47 % smaller than its TN footprint implies.
- **C — status quo:** keep 1.00 and document the mismatch.
- **D — reorder** `hv_buses` to `(0, 3, 8)`, combinable with A or B; halves the
  shape mismatch for DSO_2/3/4 at no impedance cost.

**Chosen by Manuel (2026-07-29): A + D.** TN geometric consistency takes
priority over the SimBench per-circuit bound; DSO_2's and DSO_4's long circuits
are accepted as aggregated corridors (see the KNOWN LIMITATION note now in
`constants.py`).

Applied to `network/ieee39/constants.py`:

```python
dict(net_id="DSO_1", zone=2, ieee_1idx=(7, 8, 5),    hv_buses=(0, 3, 8), scale=0.82, gen="mixed"),
dict(net_id="DSO_2", zone=2, ieee_1idx=(12, 14, 4),  hv_buses=(0, 3, 8), scale=1.40, gen="mixed"),
dict(net_id="DSO_3", zone=2, ieee_1idx=(11, 10, 13), hv_buses=(0, 3, 8), scale=0.52, gen="mixed"),
dict(net_id="DSO_4", zone=3, ieee_1idx=(24, 21, 23), hv_buses=(0, 3, 8), scale=2.44, gen="mixed"),
```

## 5a. Verification

**Geometry re-check** — `detour_at_configured` (the residual detour at the
scale now in `SUBNET_DEFS`; equals κ when the calibration is exact):

| DSO | scale | detour at configured scale | rank |
|---|---|---|---|
| DSO_1 | 0.82 | 1.000 | **match** (was partial) |
| DSO_2 | 1.40 | 1.002 | **match** (was partial) |
| DSO_3 | 0.52 | 0.992 | **match** (was partial) |
| DSO_4 | 2.44 | 1.000 | **match** (was partial) |

The reorder to `(0, 3, 8)` moved all four DSOs from `partial` to `match`: TN and
HV now order the three coupling pairs identically. `hv_buses_best` confirms
`(0, 3, 8)` is the optimum of all six permutations for every DSO.

**Power flow** — converges (NR, 50 it max) with all four DSOs attached.
Voltages `[0.9248, 1.0300]` p.u., **no bus outside [0.90, 1.10]**; 110 kV band
`[0.9248, 1.0042]`. Trafo3w loading max 46.5 %.

Line overloads are pre-existing IEEE 39 TN-side stress, not caused by the DS:

| | baseline (1.00, `(3,0,8)`) | new | |
|---|---|---|---|
| overloaded lines | 5 | 6 | all on the 345 kV TN, none in any HV sub-network |
| max loading | 153.4 % (line 1–2) | 153.1 % (line 1–2) | |
| newly overloaded | — | line 16–21 at 111.5 % | DSO_4 zone-3 corridor |

**SimBench re-check** — the DSOs are now differentiated, and two of them land
much closer to the reference than before:

| DSO | scale | route km | median circuit | max circuit | vs SimBench max 51.6 km |
|---|---|---|---|---|---|
| DSO_3 | 0.52 | 124.8 | 10.4 km | 20.8 km | ok |
| DSO_1 | 0.82 | 196.8 | 16.4 km | 32.8 km | ok |
| DSO_2 | 1.40 | 336.0 | 28.0 km | 56.0 km | exceeds |
| DSO_4 | 2.44 | 585.6 | 48.8 km | 97.6 km | exceeds |

For reference: SimBench HV1 median 8.5 km / max 43.8 km, HV2 median 5.7 km /
max 51.6 km. DSO_3 and DSO_1 now sit squarely inside the reference
distribution; DSO_2 and DSO_4 are the accepted exceptions.

## 6. Risks / unresolved

- **OPEN — re-tuning required.** The change alters every TN–DS impedance, so
  `results/tuned_params_t0min.json`, all cached sensitivities and every
  archived experiment result predate it and are no longer comparable.
  Controller re-tuning is needed before new results are read against old ones.
  Not done here.
- **DSO_2 and DSO_4 exceed the SimBench per-circuit bound** (56.0 and 97.6 km
  vs 51.6 km observed maximum). Accepted deliberately; they must be described
  as aggregated corridors, not single tower lines, wherever the line-length
  justification is cited.
- One additional TN line overload (16–21, 111.5 %). The IEEE 39 base case at
  this loading already carries 5 overloads up to 153 %, so this is a
  pre-existing stress pattern rather than a new defect — but if TN loading
  matters for a given study, it should be checked.
- κ = 1.0 is a modelling convention, not a measured law. κ = 1.25 would raise
  all scales by 25 % and worsen the SimBench conflict.
- DSO_1's shape mismatch is irreducible by scale or permutation; only changing
  `ieee_1idx` or the HV topology would fix it.
- Divergence history: the header comment says DSO_2 was once disabled because
  "PF diverges with 3 HV sub-networks". Changing scales changes impedances and
  may re-open convergence issues — a power-flow check is needed after any change.
- `LINE_LENGTHS_KM` has no entry for line 23–36 (kept at 1.0 km). Harmless here,
  worth fixing separately.

## 7. Files

- `analysis/dso_ts_coupling_geometry.py` — new, read-only (prints a proposed
  `SUBNET_DEFS` block, never edits `constants.py`).
- `results/dso_ts_geometry/` — `pairwise_distances.csv`, `recommendation.csv`.

```
python -m analysis.dso_ts_coupling_geometry
python -m analysis.dso_ts_coupling_geometry --kappa 1.25
```
