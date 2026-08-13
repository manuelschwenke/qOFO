"""
pf
==
DIgSILENT PowerFactory interface layer for the RMS co-simulation
(docs/RMS_IEEE39_PowerFactory_Build_Plan.md).

Modules
-------
``pf.session``  -- engine/embedded API session helpers (stdlib only; must
                   run standalone on the PF machine).
``pf.naming``   -- pandapower-index-embedding loc_name convention shared by
                   the sync/parity scripts (see docs/pf_naming.md).
``pf.hello_pf`` -- Phase-1 manual smoke test (see docs/pf_api_notes.md).
``pf.probes``   -- standalone read-only probes that establish PF API and RMS
                   semantics for the modules above (see that subpackage's
                   docstring for the families).

None of this package can be executed on the development server (no
PowerFactory installation / licence); everything is written defensively and
verified on the PF machine per the notes in docs/pf_api_notes.md.
"""
