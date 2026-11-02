# Reactive-reserve map on the single-line diagram (GA, new base)

Timestamp: 2026-11-02, ~08:30–10:00 Europe/Berlin. Branch `res/commitment-base-2026-10-31`,
worktree `F:\qOFO_res`. Separate session, at the author's request: overlay the reactive-power
scarcity of the new base (merit-order commitment) on the single-line diagram of
`fig:ieee39_wind_replace` (thesis Fig 7.2).

## Author decisions

- **Encoding.** Of three proposed concepts, the author chose A: reserve gauges, one ring per
  reactive source. Rejected:
  - B, a level/shape split;
  - C, reach with and without the STS. In the sketch, its two strokes looked almost the same.
- **Period.** The author chose GA (08 Jan 06:00, G3 offline by commitment, A's events) instead
  of GE. GE is "too critical": 5 of 6 variants fail the load flow after 212 min.
- **Panels.** Three: (a) the start of the period, (b) OFO, (c) OFO+ at the same instant.

## Why GA (measured before building)

Own upward Q reserve of the TS's own sources, LOC runs, c08b REF runs 0004–0009:
- per area, Σ(Q_max − Q) over the in-service machines without G1, plus the TS wind parks (VDE
  box at the recorded P);
- area-2 values: A mean 681 / min 465 Mvar; B and D ≈ 236 / ≈ 120; E 194 / 0; GA 143 / 51;
  GE 142 / 0 (LOC fails at 313 min).

GA at 150 min:
- OFO: area-2 own reserve 0 Mvar, V_min 0.972;
- OFO+: area-2 own reserve 65 Mvar, V_min 1.002. STS 1–3 inject ≈ 233 Mvar, against reported
  capabilities of 153 / 241 / 332 Mvar.

Scripts: `reserve_rank.py`, `ga_instants.py` and `ga_o2.py` in the session scratchpad. They are
superseded by the producer below.

## What changed

**New file `experiments/ch_11_case2_multisystem/ref/figure_scarcity_sld.py`**
- Producer of `case3_scarcity_sld`. Same conventions as `figures_ref.py`:
  - output to `<run>/figures/` and `figures/data/`;
  - `--thesis` copies to `graphics/Ch10/` and `graphics/Ch10/data/<date>_case3_scarcity_sld/`
    with a README;
  - `ThesisRun.record_export`.
- **Panel instant.** The rule is declared in the module docstring before drawing: t* is the
  first record of the minimum area-2 own reserve under O2. For GA, t* = 138 min.
  - (a) is the first L2 record (t = 20 s).
  - (b) is O2 and (c) is O3 at t*.
- **Ring encoding.**
  - Ring radius ∝ √(Q_max − Q_min), so the circle area is proportional to the capability. The
    dark arc is the upward reserve share (Q_max − Q)/(Q_max − Q_min).
  - A machine out at the first record is drawn dashed (decommitted). A machine that goes out
    later is drawn crossed (tripped).
  - G1 has no ring and is not counted.
  - STS ring: the sum of the DER Q capability inside the STS. This is the resource, not the
    interface capability. The interface injection and the O3 reported interval go to the CSV
    only.
- **Area tint.** α = 0.4 · max(0, 1 − R⁺_z / 500 Mvar).
- **Geometry.**
  - Bus coordinates, line routing and STS couplings are copied from
    `graphics/Ch7/ieee39_wind_replace_body.tex`.
  - Two corrections to match the code's network (`network/ieee39/scenarios/wind_replace.py`,
    `constants.py`):
    - IEEE bus 30 carries the synchronous hydro unit **G10**, not a wind park "W1";
    - the area-2 wind park (ex-G2, bus 31) couples at **bus 6**, not bus 5.
- **Checks inside the producer:**
  - the gen index → label map is checked against `zone_gen_idx`;
  - the wind-park S_n (540 / 500 / 508 / 650 MVA) is checked against the recorded
    `tso_der_q_reserve` to 1e-6.
  - Both pass on run 0008.

## Outputs

- **Run 0008** (`results/ch_11_case2_multisystem/ref/0008_2026-10-31_125725_run_ref`):
  - `figures/case3_scarcity_sld.{pdf,png}`;
  - `figures/data/case3_scarcity_sld.csv`;
  - two `exports` entries in `run_manifest.json`.
- **Thesis:**
  - `graphics/Ch10/case3_scarcity_sld.pdf`;
  - `graphics/Ch10/data/2026-10-31_case3_scarcity_sld/` (CSV and README).
  - Not referenced in any chapter yet.

## Fig 7.2 corrected (~10:30, author: "Yes, fix Fig 7.2")

- **Thesis `graphics/Ch7/ieee39_wind_replace_body.tex`:**
  - G10 is drawn at B02 through a machine transformer and terminal bus B30;
  - the former W1 is removed;
  - W2 is coupled at B06;
  - the former W5 is renamed W1, so the parks are W1–W4;
  - the "STS 1" label is moved.
- Details and checks are in the thesis log
  `00_daily_log/2026-11-02_fig72_g10_and_wind_parks_corrected.md`. The LuaLaTeX pass had
  0 errors, and the figure is still on p. 101.
- **The producer follows the corrected figure:**
  - the area-1 park is labelled W1 (was W5);
  - W2 sits below B06;
  - the docstring is updated.
- The producer was re-run with `--thesis`, refreshing the PDF, the CSV and the export records.

## Voltage colouring and IBG wording (~11:00–11:40, after commit 44eafd4)

**Author request:**
- the area tint shows the median voltage instead of the own reserve;
- the busbars use the same colour code;
- a colour bar goes below the panels;
- "IBG", not "DER".

**What changed in `figure_scarcity_sld.py`:**
- **Colour scale.**
  - Diverging ColorBrewer PuOr, with low voltage in orange.
  - It is centred on the run's TS voltage set point V_set. `_v_set` reads it from `config.json`
    and checks that every zone set point equals `v_setpoint_pu` (1.03 pu).
  - The half range is V_HALF = 0.06 pu, so the scale runs from 0.97 to 1.09. It is symmetric
    and covers the extremes of the three GA panels (0.976 to 1.054).
- **Busbars.**
  - Each bar is a rectangle filled with the logged voltage `bus_vm_pu` (pandapower index = IEEE
    number − 1, `BUS_IDX`).
  - The machine terminal buses B32, B33 and B38 are not logged. They stay as black bars and have
    a legend entry.
  - B36 is logged; in the model it is a 345-kV bus, with G7's terminal behind it.
- **Areas.**
  - Each area is filled with the median over its logged buses, at 50 % opacity.
  - Area membership comes from `network.zone_partition._FIXED_ZONES_IEEE39`.
  - New check `_bus_voltages`: every logged TS bus must be drawn, and per area the min, mean and
    max of the logged buses must equal the recorded `zone_v_min/mean/max` to 1e-9. Passes on all
    three panels.
- **Panel text.** Three lines: R⁺ per area, median V per area, and area-2 V_min with its bus and
  the STS 1–3 injection.
- **Colour bar.** Under panel (b), with side labels. The legend's tint entry is replaced by the
  terminal-bus entry.
- **Layout.** Explicit placement with `fig.add_axes`; the figure is now 8.6 cm high (was 7.5 cm).
- **CSV.** New rows of kind `bus` (logged voltage per bus and panel). Area rows gain `v_median`
  and `v_min_bus`.
- **IBG wording.** "STS (IBG inside)" in the legend, plus the docstring and the README text.
  Record field names (`dso_der_*`) are unchanged.

**GA values at t* = 138 min.** Median V per area:

| panel | A1 | A2 | A3 |
|---|---|---|---|
| (a) start of the period | 1.040 | 1.036 | 1.032 |
| (b) OFO | 1.032 | 1.022 | 1.028 |
| (c) OFO+ | 1.031 | 1.036 | 1.033 |

- The bus with the area-2 V_min is B12 in all panels: 1.032, 0.976 and 1.001 pu.
- In (b), B10, B11, B13 and B14 are at 1.013–1.019 pu. In (c) they are at 1.031–1.036 pu.

**Outputs:**
- Run 0008 `figures/` was re-rendered.
- The first thesis export failed because `graphics/Ch10/case3_scarcity_sld.pdf` was open in
  Acrobat (WinError 32).
- Re-run with `--thesis` at 11:44, after the author closed it. The thesis PDF is byte-identical to
  the run copy, the README says IBG, and the CSV has 90 bus rows (30 × 3). The manifest still
  holds two export entries (no duplicates).

## Open

- **Weights.** Panels (b) and (c) use c08b, which is not final on the commitment base (Ch 9
  rerun). Redraw after the switch: rerun the producer on the new GA run.
- **Commits.** The first version is commit 44eafd4. The voltage-colouring change is not committed
  yet. The worktree also holds the Ch 9 session's untracked `stage1_v2_uc_2026_11_pq*.json`,
  which were left alone.
- **Median tint is weak.** The area medians lie within ±0.014 pu of V_set, so the area fills are
  faint; the information sits in the busbars (B12 and its neighbours). The area min, or the share
  of buses below a threshold, would separate (b) and (c) more strongly. Left as is, because the
  author asked for the median.
