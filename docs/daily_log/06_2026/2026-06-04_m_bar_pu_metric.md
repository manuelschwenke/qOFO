# 2026-06-04 — Add size-comparable generator reserve metric `m_bar_pu`

**Timestamp:** 2026-06-04 (local)
**Author:** Manuel Schwenke / Claude Code

## Reason / motivation
`m_bar_mvar` (fleet-mean generator Q headroom in Mvar) is **size-weighted**: a
mean of absolute Mvar is dominated by the single largest machine. In the
`wind_replace` system zone 1 carries a **10000 MVA** machine next to a 1000 MVA
one (verified via `gen_s_rated_by_zone`), so `m_bar_mvar` mostly reflects that
one unit and is not comparable across machines of very different ratings. Add a
per-unit counterpart that normalises each machine's headroom by its own MVA
rating and **equal-weights** machines, so the fleet reserve is no longer
dominated by the largest unit.

Note: normalising headroom by `q_max` would have been redundant — since the
record stores `headroom = q_max − |Q|` ([records.py:290]), `headroom/q_max =
1 − res_util`. The non-redundant, size-comparable base is the **nameplate
`S_n`**, hence `m_bar_pu = mean_g (headroom_g / S_n,g)`.

## What was changed

### `experiments/helpers/comparison_metrics.py`
- **New helper `gen_s_rated_by_zone(scenario)`** → `{zone: array of S_n [MVA]}`
  aligned to `record.zone_q_gen` / `gen_q_headroom_mvar` via the `k_in_zone`
  ordering from `plot_cigre._gen_info_with_k` (lazy import to avoid the circular
  import, since plot_cigre imports comparison_metrics). This is the same index
  convention the Fig. 4 capability plot already relies on, so alignment is
  guaranteed.
- **`cigre_summary_table(...)`** gained optional arg
  `gen_s_rated_mva: Optional[Dict[int, NDArray]] = None` and a new output column
  **`m_bar_pu`** computed in the existing per-step gen loop (mirrors the
  `res_util` masking exactly): per step, `mean_g (headroom_g / S_n,g)` over all
  machines, then time-averaged. **`NaN` when ratings are not supplied** (fully
  backward-compatible). Column placed right after `m_bar_mvar`.

### `experiments/005_CIGRE_MULTI.py`
- `write_tables` now builds the ratings map (`gen_s_rated_by_zone(cfg.scenario)`,
  guarded) and passes it to `cigre_summary_table`, so `cigre_summary.csv` /
  the printed `tab:summary` gain the `m_bar_pu` column automatically.

### `experiments/006_CIGRE_MONTECARLO.py`
- Added `m_bar_pu` to `METRIC_COLS`, `LOWER_BETTER` (`False` = higher better),
  `METRIC_LABELS` (`"gen Q reserve [p.u. $S_n$]"`), and to
  `BOX_METRICS_DEFAULT` (now 5 panels: rms_v_ts, **m_bar_pu**, res_util,
  tie-Q, n_sw).
- New cached getter `_get_gen_srated()` (per-process, spawn-safe like the other
  caches) feeding the per-run `cigre_summary_table` call.
- **Backward-compat guards** so a `--replot` of the pre-existing 100-run staging
  (whose `scenario_*.json` predate this column) does not crash: `aggregate_table3`
  aggregates only metric columns present in the CSV (warns about the rest), and
  `plot_table3_boxplots` drops any selected metric missing from the CSV.

## Validation
- `py_compile` clean on all three files.
- `gen_s_rated_by_zone('wind_replace')` → zone1 [1000, 10000], zone2 [800],
  zone3 [800, 700] MVA (5 sync machines; matches `zone_q_gen` cardinality).
- Synthetic unit test (2 machines, S_n=100/1000, Q=40/40, headroom=10/410):
  `m_bar_mvar=210.0` (big-unit-dominated), `m_bar_pu=0.255` (= mean(0.10,0.41)),
  `res_util=0.4444`; `m_bar_pu=NaN` when ratings omitted. All as hand-computed.

## Open / next
- **The existing 100-run batch must be re-simulated to populate `m_bar_pu`**:
  metrics are computed at sim time and staged into `scenario_*.json`; the full
  per-step gen headroom is not kept in the `timeseries/*.npz`, so `--replot`
  cannot back-fill it. Re-run **without** `--resume` (resume reuses the old
  staged JSON). 005 `--replot` *can* back-fill (it reloads full `log.pkl`).
- Interpretation for the paper: report `m_bar_pu` (size-comparable reserve)
  alongside / instead of `m_bar_mvar`; pairs with `res_util` (utilisation).
