# 2026-07-29 — SimBench 110 kV benchmark for the synthetic DSO topology

**What:** New analysis script `analysis/simbench_hv_benchmark.py`.
**Why:** The synthetic 110 kV DS attached to the IEEE 39-bus case
(`HV_LINE_TOPOLOGY`, 10 buses / 11 circuits, 3 EHV/HV couplers) was so far
justified only internally. A published reference was needed for the thesis to
defend (a) the 10–40 km circuit lengths and (b) the ~60–85 km spacing between
coupling transformers.
**Timestamp:** 2026-07-29
**Status:** implemented, both code paths verified against each other.

---

## 1. Reference dataset

SimBench v1.0 (Meinecke et al., *Energies* 13(12):3290, 2020; simbench.de),
package `simbench` 1.6.2 already in `qOFO_clean` — data ships with the package,
no download.

Established fact for the selected scenario: SimBench scenario 0 contains
**two HV reference grids**.

| Grid | Code | Character | Circuits | Route km | EHV/HV trafos | Coupling substations |
|---|---|---|---|---|---|---|
| HV1 | `1-HV-mixed--0-sw` | mixed rural/semi-urban | 95 | 1083.6 | 6 | 3 |
| HV2 | `1-HV-urban--0-sw` | urban | 113 | 751.6 | 3 | 1 |

Within scenario 0, `1-HVMV-*`, `1-EHVHV-*`, `1-EHVHVMVLV-*` embed the same
base HV1/HV2 grids. A later generation audit found that scenarios 1 and 2 also
add generator-interconnection buses and lines (HV1: 95 → 125 → 145 lines;
HV2: 113 → 131 → 151), so topology is not scenario-independent. This
benchmark deliberately fixes scenario 0. Its topological sample is N = 2
grids, and for coupling-point spacing N = 3 pairs (HV2 contributes
none — all three of its transformers sit in one substation).

## 2. Method (key structural decision)

SimBench models substations at node-breaker detail (auxiliary nodes, busbars,
bus-bus switches): 294 raw 110 kV buses in HV1 for ~61 real stations. Raw bus
and line counts are therefore **not** comparable to a 10-bus equivalent.

Reduction applied before any statistic:

1. `pandapower.topology.create_nxgraph(respect_switches=True)` → normal
   operational state (open switches removed).
2. Contract every zero-length-connected cluster into one **station node**.
   A cluster counts as a station if it carries a busbar, an injection or a
   transformer terminal.
3. Coupling transformers = 2W with `vn_lv_kv=110`, `vn_hv_kv ∈ {220,380}`, or
   3W with `vn_mv_kv=110` (the qOFO convention). Transformers landing in the
   same station node = one **coupling point** (parallel bank).
   Cross-checked against SimBench's own `substation` column → **consistent**
   for both grids (3↔3 clusters in HV1, 1↔1 in HV2).
4. Distances = shortest path *inside* the 110 kV graph, reported both
   geometrically (km) and electrically (`X = Σ x_i ℓ_i`, and `X_pu` on
   `Z_base = 110²/100 = 121 Ω`).

The qOFO auxiliary ZIP-load buses/links (`AUX_LOAD*`, r = x = 0.01 Ω over 1 km)
are excluded — load-model artefact, not stations.

A secondary **corridor** sample is produced by additionally contracting
degree-2 non-coupling stations (lengths summed), as a like-for-like reference
for an aggregated equivalent. It turned out to be the *weaker* comparison —
see §4.

## 3. Results

**Circuit lengths [km]**

| Sample | n | median | mean | p90 | max |
|---|---|---|---|---|---|
| SimBench HV1, overhead | 94 | 8.5 | 11.5 | 24.6 | 43.8 |
| SimBench HV2, overhead | 75 | 5.7 | 8.4 | 18.2 | 51.6 |
| SimBench pooled, overhead | 169 | 6.3 | 10.1 | 24.6 | 51.6 |
| **Synthetic DS** | 11 | 20.0 | 21.8 | 30.0 | 40.0 |

**Coupling-point spacing [km / Ω / p.u.]**

| Grid | pairs |
|---|---|
| SimBench HV1 | 70.6 (20.9 Ω, 0.173 pu), 152.9 (45.3 Ω, 0.374 pu), 223.6 (66.2 Ω, 0.547 pu) |
| **Synthetic DS** | 60.0 (24.0 Ω, 0.198 pu), 65.0 (26.0 Ω, 0.215 pu), 85.0 (34.0 Ω, 0.281 pu) |

**Depth to nearest coupling point [km]**: HV1 median 20.9 / max 78.3;
HV2 median 22.4 / max 114.3; synthetic median 17.5 / max 55.0.

**Density**: route-km per station 17.8 (HV1), 9.3 (HV2), 24.0 (synthetic);
circuits per station 1.56 / 1.40 / 1.10.

**Conductors**: synthetic `184-AL1/30-ST1A` (x = 0.400 Ω/km) vs SimBench
`Al/St 265/35` (x = 0.296 Ω/km) and `1x630_RM/50` cable (x = 0.123 Ω/km).

## 4. Assessment (what the thesis can claim)

- **Defensible:** every synthetic circuit length lies inside the empirical
  SimBench overhead range [0.54, 51.6] km, and 34 % of SimBench's 169 overhead
  circuits fall inside the synthetic band [10, 40] km. The lengths are
  *realisable*, not extrapolated.
- **Honest qualifier:** the synthetic median sits at the 87th percentile of
  the SimBench circuit distribution — the upper decile. This is the expected
  direction for a 10-node equivalent of an area that SimBench resolves with
  61–81 stations, but it should be stated, not hidden.
- **Coupling spacing is the strongest result:** synthetic 60/65/85 km against
  HV1's 70.6/152.9/223.6 km. The synthetic spacings are *conservative*
  (shorter than the reference), and in electrical terms 0.198–0.281 pu vs
  0.173–0.547 pu — squarely inside the reference range.
- **Corridor comparison is the weakest and is labelled as such:** SimBench HV
  grids are strongly meshed, so series reduction removes little (182 of 208
  circuits survive) while the sparser synthetic grid contracts 11 → 6. It
  therefore overstates the synthetic lengths (92nd percentile) and is reported
  for completeness only.

## 5. Risks / unresolved

- **Sample size for coupling spacing is N = 3 pairs, from a single grid.**
  This is an order-of-magnitude check, not a distribution. If a stronger claim
  is needed, extend to the SciGRID_de or the ENTSO-E-derived 110 kV datasets.
- **Conductor mismatch is unresolved and matters for the control results.**
  At equal route length the synthetic grid is ~35 % electrically longer than
  the SimBench reference (0.400 vs 0.296 Ω/km), which inflates ∂V/∂Q
  sensitivities. Options: (i) switch `HV_LINE_TOPOLOGY`'s std_type to an
  Al/St 265/35-equivalent, (ii) keep it and state it as a deliberately
  conservative (weak-grid) choice. **Not changed here** — this is a model
  change and needs a decision.
- The corridor reduction contracts stations that carry load/DER; it is valid
  for length comparison only, never as an electrical equivalent.
- Statistics use the `-0-` scenario and `sw` (switched) variants at their
  normal operational state. Scenarios 1 and 2 add generator-interconnection
  assets, so the scenario must be retained in every topology citation.

## 6. Files

- `analysis/simbench_hv_benchmark.py` — new.
- `results/simbench_hv_benchmark/` — `report.md`, `simbench_hv_benchmark.png`,
  `lines.csv`, `corridors.csv`, `line_length_stats.csv`, `coupling_pairs.csv`,
  `station_depth.csv`, `topology_summary.csv`, `conductors.csv`.

Run:

```
python -m analysis.simbench_hv_benchmark
python -m analysis.simbench_hv_benchmark --from-built-net   # cross-check
```

The `--from-built-net` path builds the real IEEE39 + DSO network and slices
each `DSO_i` sub-network out by bus-name prefix. Verified to reproduce the
constants-based reconstruction exactly: 10 stations, 11 circuits, 240 route-km,
spacings 60/65/85 km for all four DSOs.
