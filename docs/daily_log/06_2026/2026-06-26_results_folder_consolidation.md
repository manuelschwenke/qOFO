# 2026-06-26 Results Folder Consolidation

- Timestamp: 2026-06-26 00:00 Europe/Berlin
- Changed: moved experiment result storage to the top-level results directory and removed the nested experiments/results location.
- Method: added experiments.paths.results_path(), changed experiment output roots 002-007 to use that helper, and anchored MultiTSOConfig.result_dir to the project-root results path for scripts that rely on the config default.
- Data migration: removed the old top-level results directory as requested, then moved/merged experiments/results into top-level results; experiments/results no longer exists.
- Reason: avoid ambiguous relative output paths when scripts are launched from different working directories and keep all experiment artifacts in one repository-level results folder.
- Constraints: controller logic, actuator definitions, measurements, and objective formulations were not changed.
