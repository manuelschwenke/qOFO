# Interactive saved-result comparison notebook

**Timestamp:** 2026-07-17 13:10:25 +02:00  
**Reason:** Provide a reusable, interactive way to compare two or more uniquely identified experiment runs without writing a dedicated plotting script for every comparison.

## Changes

- Added `analysis/compare_results.ipynb` as the user-facing notebook.
- Added `analysis/result_comparison.py` so result discovery, nested-record extraction, plotting, and widget construction are testable outside notebook state.
- The notebook recursively scans `results/` for directories containing `records.pkl` and uses the result path relative to `results/` as the unique selector label, for example `run_multi_system_ofo/0011_2026-07-17_104337`.
- Added curated selectors for:
  - TSO and DSO voltage envelopes;
  - TSO and DSO DER reactive-power infeed;
  - synchronous-generator reactive power;
  - TSO shunt and TSO/DSO OLTC positions;
  - generator AVR voltage setpoints;
  - DSO interface and transformer setpoint/actual Q;
  - inter-zone tie-line Q;
  - TSO/DSO loading envelopes and system losses.
- All numeric leaves in the saved record schema are also discovered dynamically and exposed as `Record field: ...` quantities.
- Nested dictionaries and NumPy arrays are flattened into selectable physical/group channels. Because the records do not persist element names/indices parallel to every actuator array, array entries are labelled by controller ordering (`item 0`, `item 1`, ...).
- Added overlay and small-multiple layouts, seconds/minutes/hours time axes, optional difference-to-first-run comparison, and a per-trace summary table (minimum, sample mean, maximum, final value).
- Differences are evaluated after interpolation onto the union of timestamps, restricted to the common supported time interval.

## Assumptions and constraints

- Result pickles are trusted local project output; Python pickle must not be used with untrusted files.
- The comparison reads recorded plant/controller quantities only and does not rerun the plant, cached sensitivity models, TSO/DSO controllers, or actuators.
- Controlled-output interpretation remains unchanged: TSO/DSO voltage and interface/tie reactive-power tracking can be compared alongside AVR/DER, OLTC, and shunt actions.
- Run schemas may differ. Missing fields/channels are skipped per run, while the available-channel selector uses the union across selected runs.
- `ipywidgets` is required for the interactive controls. It was not installed in `qOFO_clean` at implementation time; the notebook displays the one-time `%pip install ipywidgets` instruction rather than mutating the environment automatically.

## Verification

- Validated notebook JSON and compiled every transformed code cell with IPython.
- Byte-compiled `analysis/result_comparison.py`.
- Loaded the two latest saved runs (`0010` and `0011`) and successfully extracted all requested core quantities.
- Confirmed representative channel counts: TSO voltage 9, DSO voltage 12, TSO DER Q 4, DSO DER Q 4, TSO shunts 8, TSO OLTCs 7, DSO OLTCs 12, and AVR setpoints 6.
- Rendered a headless small-multiple comparison and generated its summary table successfully.

## Open points

- Persisting physical actuator indices/names alongside arrays in future result records would allow labels such as generator or transformer names instead of `item N`.
- A later extension could add event markers, voltage/thermal limits, figure/CSV export buttons, and configuration-difference tables.
