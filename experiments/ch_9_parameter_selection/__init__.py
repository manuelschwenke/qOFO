"""Entry points producing the timescale-separation results of the thesis.

One module per section, named after the section it fills:

* ``ch_9_1_timescale_seperation`` -- the open-loop RMS step battery
  (\\cref{ch:timescales}). **A historical instrument**: it no longer fixes
  any period. Its table, ``tab:param:timescales:settling``, was retired
  with the step tests at the 2026-09-18 colloquium and is archived in
  ``Chapters/archive/Ch9_sec1_step_tests_CUT.tex``. The subordinate period
  now follows from the modal bound of Experiment A
  (:math:`T_{STS} \\geq 17.7` s, set to 20 s), reported in
  ``tab:param:timescales:actuators`` and produced by a different pipeline
  (see below). Kept because its PowerFactory infrastructure in
  ``pf/screening.py`` still serves the dead-band study.

Numbering note (2026-10-23): the chapter this folder serves is
``ch:timescales``, renumbered to **Ch 9** in the 2026-10-23 chapter split
(it was Ch 8 before that, and Ch 9 before 2026-10-06). The folder name and
the dead ``ch:param:timescales`` label both date from the chapter's older
title, *Parameterisation of the Control Hierarchy*. See
``docs/architecture/chapter_folder_map.md``.

The PowerFactory driver infrastructure these use (``ScreeningContext``, the
step catalogues, the settling metric) stays in ``pf/screening.py``: it is
shared with the RMS build and the dead-band study, and duplicating it here
would let the two drift apart.

The 2026-10-22 minimal-codebase pass removed ``pf/`` and briefly took these
two modules with it.  Both were restored, with the subset of ``pf/`` and
``export/`` they need, once it became clear that §9.1 Table 9.1 and §9.2
``dU_db`` have no non-PowerFactory producer: see ``README.md`` for the
section-to-script map and ``docs/daily_log/10_2026/`` for the reasoning.
"""
