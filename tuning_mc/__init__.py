"""
tuning_mc
=========
Monte-Carlo based controller tuning campaign (2026-08).

Stages, each a runnable module:

* ``stage_0_preconditioning`` — analytic weight design from the cached
  sensitivities: the curvature rule for continuous actuator classes (existing)
  **and** a commit-threshold rule for the integer classes (new).  Produces the
  starting point for everything downstream, plus the diagnostic that reads the
  currently shipped integer weights back as physical commit thresholds.

Later stages (scenario/excitation design, metric definition, the Monte-Carlo
campaign itself) are added as they are agreed.
"""

__all__: list[str] = []
