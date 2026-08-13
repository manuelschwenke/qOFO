# RMS replay: force UTF-8 console output

- **Timestamp:** 2026-07-23
- **Changed:** `experiments/run_rms_phase6_replay.py`
- **Method:** Reconfigured `stdout` and `stderr` to UTF-8 before importing project modules that may wrap the streams with Colorama.
- **Reason:** The 18,000 s replay stopped during the static reference at step 818/900 because Windows `cp1252` could not encode the Unicode level symbol `ℓ`.
- **Scope:** Console encoding only. No controller, plant, event-pool, configuration, or simulation-run behavior was changed.
