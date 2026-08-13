# 2026-07-29 — SimBench HV generation benchmark

**What:** New reproducible analysis
`analysis/simbench_hv_generation_benchmark.py`.

**Why:** Follow-up to
`2026-07-29_simbench_hv_topology_benchmark.md` after differentiating the four
synthetic DSO footprints. The question is whether the generation installed in
each IEEE39-underlaid synthetic 110 kV network is of a plausible magnitude
relative to the two public SimBench HV references.

**Timestamp:** 2026-07-29

**Status:** implemented and verified against the fully built IEEE39 + four-DSO
network. Analysis only; no network or controller parameter changed.

---

## 1. Scope and conventions

Reference data:

- SimBench package 1.6.2, grids `1-HV-mixed--{0,1,2}-sw` (HV1) and
  `1-HV-urban--{0,1,2}-sw` (HV2).
- Scenario 0 is the primary comparison because the topology benchmark uses
  `1-HV-*-0-sw`. Scenarios 1 and 2 are retained as expansion-scenario
  sensitivity cases.
- The official SimBench code defines the scenario as its fifth component:
  <https://simbench.de/en/download/datasets/>.
- Dataset/paper reference: Meinecke et al., “SimBench—A Benchmark Dataset of
  Electric Power Systems to Compare Innovative Solutions Based on Power Flow
  Analysis,” *Energies* 13(12):3290, 2020,
  <https://doi.org/10.3390/en13123290>.

Capacity convention:

1. Installed active generation is `sum(p_mw)` over in-service, non-zero
   `gen` and `sgen` rows represented on the selected 110 kV component.
2. `ext_grid` is excluded because it is the upstream system, not generation
   installed inside the studied HV grid.
3. Storage is excluded. In SimBench scenarios 1/2, `storage.p_mw` is a signed
   dispatch point and is negative while charging; it is not generation
   nameplate. A storage comparison would require separate MW/MWh metrics.
4. `sum(sn_mva)` is reported separately.
5. SimBench generation is split into:
   - `direct_110kV`: plants represented as directly connected to the HV grid;
   - `equivalent_below_110kV`: aggregate generation of equivalent MV grids,
     injected at a 110 kV station in the HV-only model.
6. For the synthetic side, the script builds the actual IEEE39 + DSO network
   and parses the `sgen` and load tables by `DSO_i|...` prefix. This captures
   the three 40 MW coupling-bus wind parks in addition to the base wind/PV
   portfolio.

The direct/equivalent distinction is essential: all synthetic generators are
modelled as `sgen` elements on physical 110 kV buses, whereas SimBench's
HV-only grids also contain explicit equivalents of downstream MV generation.

## 2. Scenario-0 result

| Grid | units | installed P | nominal S | direct at 110 kV | equivalent below 110 kV | reference load | Pgen/Pload |
|---|---:|---:|---:|---:|---:|---:|---:|
| SimBench HV1 | 57 | 1410.5 MW | 1516.7 MVA | 1077.2 MW | 333.3 MW | 521.6 MW | 2.70 |
| SimBench HV2 | 42 | 586.2 MW | 630.3 MVA | 309.9 MW | 276.3 MW | 490.1 MW | 1.20 |
| Synthetic DSO_1 | 10 | 410.0 MW | 410.0 MVA | 410.0 MW | 0 | 261.8 MW | 1.57 |
| Synthetic DSO_2 | 10 | 410.0 MW | 410.0 MVA | 410.0 MW | 0 | 261.8 MW | 1.57 |
| Synthetic DSO_3 | 10 | 410.0 MW | 410.0 MVA | 410.0 MW | 0 | 261.8 MW | 1.57 |
| Synthetic DSO_4 | 10 | 410.0 MW | 410.0 MVA | 410.0 MW | 0 | 261.8 MW | 1.57 |

Synthetic carrier split per DSO:

- wind: 270 MW in 6 plants;
  - base wind parks: 40 + 60 + 50 = 150 MW;
  - coupling-bus wind parks: 3 × 40 = 120 MW;
- PV: 140 MW in 4 plants;
- total: 410 MW in 10 continuously Q-controllable `sgen` elements.

The synthetic direct-HV mix is therefore 65.9 % wind / 34.1 % PV. In
SimBench scenario 0, HV1's direct-HV plants are all wind (1077.2 MW), while
HV2 has 299.9 MW wind and 10 MW PV directly at HV. The synthetic model has a
materially larger HV-level PV share than either reference.

## 3. Topology-normalised result

| Grid | reduced stations | route km | MW/station | MW/route-km |
|---|---:|---:|---:|---:|
| SimBench HV1, scenario 0 | 61 | 1083.6 | 23.12 | 1.302 |
| SimBench HV2, scenario 0 | 81 | 751.6 | 7.24 | 0.780 |
| Synthetic DSO_1 | 10 | 196.8 | 41.00 | 2.083 |
| Synthetic DSO_2 | 10 | 336.0 | 41.00 | 1.220 |
| Synthetic DSO_3 | 10 | 124.8 | 41.00 | 3.285 |
| Synthetic DSO_4 | 10 | 585.6 | 41.00 | 0.700 |

Interpretation:

- Per reduced station, the synthetic system is generation-dense: 1.77 times
  HV1 and 5.67 times HV2 when total represented generation is used.
- Per route-km, DSO_2 is close to HV1 (0.94 times), and DSO_4 is close to HV2
  (0.90 times).
- DSO_1 is high (1.60 times HV1; 2.67 times HV2).
- DSO_3 is the clearest outlier (2.52 times HV1; 4.21 times HV2).
- `MW/route-km` is a geometric density indicator, not a loading or electrical
  strength metric. Its variation is caused by retaining the identical 410 MW
  portfolio while changing only line-length scale from 0.52 to 2.44.

## 4. SimBench scenario sensitivity

| Grid | scenario | units | total P | direct 110 kV | equivalent below 110 kV |
|---|---:|---:|---:|---:|---:|
| HV1 | 0 | 57 | 1410.5 MW | 1077.2 MW | 333.3 MW |
| HV1 | 1 | 63 | 1873.0 MW | 1295.0 MW | 578.0 MW |
| HV1 | 2 | 78 | 2611.8 MW | 1932.2 MW | 679.5 MW |
| HV2 | 0 | 42 | 586.2 MW | 309.9 MW | 276.3 MW |
| HV2 | 1 | 45 | 1047.9 MW | 558.1 MW | 489.9 MW |
| HV2 | 2 | 62 | 2025.5 MW | 1443.9 MW | 581.6 MW |

Established result: the scenario suffix materially changes installed
generation and generator-interconnection assets; it is not valid to quote one
SimBench capacity or topology without the scenario. Relative to scenario 0:

- HV1 adds 30 lines / 49 buses in scenario 1 and 50 lines / 71 buses in
  scenario 2;
- HV2 adds 18 lines / 30 buses in scenario 1 and 38 lines / 56 buses in
  scenario 2.

Scenario 0 remains the consistent comparison for the existing topology note.

## 5. Assessment for the synthetic DS

**Short conclusion:** 410 MW per synthetic DSO is defensible as an aggregate
installed-generation magnitude. Its generation/reference-load ratio of 1.57
lies between SimBench HV1 (2.70) and HV2 (1.20) in scenario 0. The result does
not support treating all four DSOs as equally generation-dense after their
geographic footprints were differentiated.

Controller-relevant interpretation:

- **Actuators:** all 10 synthetic `sgen` elements are DSO continuous-Q
  actuators with VDE-AR-N-4120 operating diagrams. SimBench generator counts
  are an asset benchmark only; they are not equivalent actuator sets.
- **Controlled outputs:** EHV/HV interface reactive-power tracking and 110 kV
  nodal voltages.
- **Constraint/model fact:** controllers do not see the plant directly; they
  act through cached sensitivities. Generation placement and available Q
  headroom affect those sensitivities and the reachable interface-Q interval.
- **No Q-capability conclusion:** installed MW/MVA alone does not establish
  comparable Q flexibility. The synthetic operating diagrams and the
  SimBench rows' capability metadata differ.

## 6. Risks and unresolved points

1. **Open modelling decision:** keep 410 MW identical across DSO_1--DSO_4, or
   scale generation using footprint, reference demand, coupling capacity, or
   a chosen regional scenario? This is an architectural/data-calibration
   choice and was not implemented.
2. **DSO_3 is the geometric-density outlier.** If geographic density matters
   to the intended study, its 410 MW portfolio deserves review first.
3. **Equivalent-MV carrier resolution:** the HV-only SimBench nets expose the
   downstream equivalents via aggregate `mv_*` profiles. Carrier-resolved
   downstream comparison requires parsing HVMV grids.
4. **Time coincidence not assessed:** installed generation/reference load is
   not simultaneous export. Wind, PV, and load profiles must be applied to
   compare interface-P operating distributions.
5. **Reactive capability not assessed:** a follow-up should compare
   time-dependent Q capability intervals at the EHV/HV interfaces, not just
   installed P and S.

## 7. Files

- `analysis/simbench_hv_generation_benchmark.py`
- `results/simbench_hv_generation_benchmark/generation_units.csv`
- `results/simbench_hv_generation_benchmark/generation_by_scope_carrier.csv`
- `results/simbench_hv_generation_benchmark/generation_summary.csv`
- `results/simbench_hv_generation_benchmark/scenario0_comparison.csv`
- `results/simbench_hv_generation_benchmark/report.md`
- `results/simbench_hv_generation_benchmark/simbench_hv_generation_benchmark.png`

Run:

```powershell
python -m analysis.simbench_hv_generation_benchmark
```
