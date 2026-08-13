r"""
pf.probes
=========
Standalone, read-only PowerFactory probes.

Each module here is a *script*, not a library: it connects to the live PF
project, answers one narrow question about PF's API or RMS semantics, prints
its findings, and restores whatever state it touched.  Probes exist so that
:mod:`pf.pf_sync`, :mod:`pf.screening` and :mod:`pf.plant` are written against
verified attribute spellings and verified solver behaviour instead of assumed
ones; the answers are recorded in ``docs/pf_api_notes.md`` and the daily logs.

Run from the repository root on the PF machine, e.g.::

    python pf\probes\probe_api.py

Families
--------
``probe_api``              -- attribute-reality check per PF class.
``probe_event_*``          -- persistent simulation-event semantics
                              (re-arm, double buffering, admission barriers,
                              pre-allocation volume, cross-``ComInc`` reuse).
                              ``probe_event_rearm`` is the base module the
                              others import their helpers from.
``probe_live_*``           -- writing DSL parameters / ``IntMat`` contents
                              while an RMS calculation is active.
``probe_rms_elmfile_profile`` -- ``ElmFile``-based profile playback.
``probe_tap_avr``          -- ``EvtTap`` / AVR setpoint handles.
``probe_g01_avr_plan``     -- G01 AVR retrofit reconnaissance.
``probe_wecc_frame_connections`` -- WECC composite frame slot namespace.

Nothing in this subpackage can run on the development server (no PF
installation / licence).  Every probe body is behind an ``if __name__ ==
"__main__"`` guard, so importing a probe module (as the ``probe_event_*``
family does, to reuse ``probe_event_rearm``'s helpers) executes only imports.
"""
