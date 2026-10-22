"""Entry points producing the parameter-selection results of thesis Ch. 9.

One module per section, named after the section it fills.

Table 9.1 was produced by ``ch_9_1_timescale_seperation``, the settling battery
that fixes ``T_DS`` and ``T_TS`` (\\cref{ch:param:timescales}), and the
actuator-location sweep by ``ch_9_1_actuator_location_sweep``.  Both drove
PowerFactory through ``pf/screening.py`` and were removed on 2026-10-22 with
the rest of the PowerFactory adapter, together with that driver
infrastructure (``ScreeningContext``, the step catalogues, the settling
metric).  Their results stand as recorded; the code remains in git history.
"""
