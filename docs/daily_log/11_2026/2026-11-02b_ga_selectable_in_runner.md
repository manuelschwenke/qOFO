# Case-2 scenarios (e.g. GA) selectable in run_multi_system_ofo.py

Timestamp: 2026-11-02, ~10:30–10:50 Europe/Berlin. New worktree `F:\qOFO_ga`, branch
`res/ga-runner-2026-11-02`, created from `res/commitment-base-2026-10-31` at e69cc1e.
`F:\qOFO_res` was left untouched because Ch 9 run 0004 (pid 97020) runs from it.

## Why

The author asked to run GA from `experiments/run_multi_system_ofo.py` and to be able to select it.
GA is not a window of the tuning set. It is a Case-2 REF scenario, defined as a period plus an
event schedule (`experiments/ch_11_case2_multisystem/ref/scenarios.py`):
- period 2016-01-08 06:00, 360 min;
- A's events: G4 trip 30:40, 250 Mvar at bus 12 (load bus 11) 121:40, tie 17–27 trip 212:20.

G3's outage is not an event. It follows from the merit-order unit commitment over the window
(`MultiTSOConfig.unit_commitment`, default True on this branch). GA therefore needs the
commitment base. The author's working copy `Z:\Python_Projekte\qOFO` (minimal/thesis-core) does
not have it.

## What changed (`experiments/run_multi_system_ofo.py`)

- **Module constants:**
  - `SCENARIO: Optional[str] = None`: a key of `ref/scenarios.py`, e.g. `"GA"`. `None` keeps the
    previous behaviour.
  - `VARIANT: str = "O3"`: a key of `ch_11_case2_multisystem/variants.py`.
  - `SCENARIO_MINUTES = 360` and `SCENARIO_NOISE_SEED = 20261027`, the REF values. They are
    copied rather than imported, because `run_ref` sets `MPLBACKEND=Agg` on import, which would
    disable the live plots.
- **New `scenario_config(name, variant, minutes, noise_seed)`.** It builds exactly the config
  that `run_ref._run_variant` builds:
  - `variants.resolved(variant, minutes, scenarios.events(name), noise_seed)`;
  - plus the scenario's `start_time`, when it has one;
  - plus `verbose=1`;
  - only change: the `live_plot_*` fields of `make_config()` are switched back on.
- **`main()`:** if `SCENARIO` is set, it uses `scenario_config(SCENARIO, VARIANT)` and prints the
  scenario, variant, start, horizon and event count. Otherwise it runs as before (make_config +
  SELECTED_PARAMSET + the `dso_gamma_oltc_q=1.0` TEST line).
- `make_config()` is unchanged. Ch 9 (`build_selected_config`) and the Case-1/2 bases build on
  it, so their output is unaffected.
- The module and `main()` docstrings name the switch.

## Check

The scratchpad script `check_ga_cfg.py` compares `scenario_config("GA", "O3")` field by field with
`results/ch_11_case2_multisystem/ref/0008_2026-10-31_125725_run_ref/O3/config_repr.txt`:
- start 2016-01-08 06:00, 360 min, UC on, slack Q limit 1200 Mvar, three events at 1840 / 7300 /
  12740 s;
- **2 differing fields**, `live_plot_controller` and `live_plot_cascade` (True here, False in the
  REF). These are visualisation only; `run_case1.base_config` documents them as having no effect
  on the control path.

No simulation was run.

## Use

Set `SCENARIO = "GA"` (and, if wanted, `VARIANT = "O2"`) in the file. Then, from `F:\qOFO_ga`:

    F:\python_environments\qOFO_clean\python.exe experiments\run_multi_system_ofo.py

## Open

- Committed on `res/ga-runner-2026-11-02` (author, ~10:55: "bring everything to Z, by
  committing and fast forwarding"). Integration into `minimal/thesis-core` (Z:) is held:
  the session "Live plotter redesign" was editing the Z: working tree at that time, with
  26 modified and 6 new files, uncommitted.
- The paramset is c08b (`ch9_selected_band_2026_10_a1`), which is not final on the commitment
  base (Ch 9 run 0004).
- Code note found while checking: `network/ieee39/constants.py` labels G3 **Nuclear** (800 MVA,
  technical minimum 0.5·S_n = 400 MW). The coal units are G4 and G7 (area 3).
