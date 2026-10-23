"""Entry points producing the timescale-separation results of the thesis.

One module per section, named after the section it fills:

* ``ch_9_1_timescale_seperation`` -- the settling battery that fixes
  ``T_DS`` and ``T_TS`` (\\cref{ch:timescales}).

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
