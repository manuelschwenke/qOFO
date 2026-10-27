"""Entry points producing the timescale-separation results of the thesis.

One module per section, named after the section it fills:

* ``ch_9_1_timescale_seperation`` -- the open-loop RMS step battery
  (\\cref{ch:timescales}). **A historical instrument**: it fixes no period.
  Its table, ``tab:param:timescales:settling``, was retired with the step
  tests at the 2026-09-18 colloquium and is archived in
  ``Chapters/archive/Ch9_sec1_step_tests_CUT.tex``. Kept because its
  PowerFactory infrastructure in ``pf/screening.py`` still serves the
  dead-band study.

  Because this folder's name says otherwise, the record: the two periods are
  fixed as follows in the live chapter, verified 2026-10-23 against
  ``Chapters/Chapter09.tex``.

  * ``T_TS`` = 180 s **by definition**, not measured -- "the supervisory
    period is set by definition rather than derived from an experiment"
    (l. 218) -- chosen because three minutes lies inside the classical
    secondary-voltage-control range of tens of seconds to minutes
    (l. 225, paul.1987, corsi.2004a).
  * ``T_STS`` = 20 s from the modal bound of Experiment A,
    :math:`T_{STS} \\geq 17.7` s (l. 168), reported in
  ``tab:param:timescales:actuators`` and produced by the modal pipeline, not
    by this battery.
  * ``N_inner`` = 9 follows as ``T_TS / T_STS``
    (``eq:param:timescales:ninner``); Experiment C only *checks* that the
    selected weights settle inside that budget (:math:`n_{s,95}` = 23.05
    <= 27) and does not move the period.

  The thesis names this package's counterpart at ll. 216-217:
  ``tuning/parameters.py`` ``TIMING_FIELDS``, with ``refine_evented``
  candidates ``tso_period_s`` 180 and ``dso_period_s`` 20.

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
``export/`` they need, once it became clear that the settling table and the
dead-band half-width ``dU_db`` have no non-PowerFactory producer: see
``README.md`` for the section-to-script map and ``docs/daily_log/10_2026/``
for the reasoning.

Thesis references in this package cite **labels, not numbers**. The chapter has
been renumbered three times (2026-10-06, and the 2026-09-29 and 2026-10-23
restructures), and each renumbering silently invalidated every hard-coded
number while leaving the labels untouched.  Where an old number does appear it
is describing superseded material on purpose, and says so.
"""
