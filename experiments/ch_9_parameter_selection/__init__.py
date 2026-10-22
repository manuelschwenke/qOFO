"""Entry points producing the parameter-selection results of thesis Ch. 9.

One module per section, named after the section it fills:

* ``ch_9_1_timescale_seperation`` -- Table 9.1, the settling battery that
  fixes ``T_DS`` and ``T_TS`` (\\cref{ch:param:timescales}).

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
