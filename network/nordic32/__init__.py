"""
network/nordic32/
=================
Nordic-32 ("Nordic SM") test-system package.

Currently contains only the Phase-A converter probe
(:mod:`convert_from_pf`).  When the conversion path is confirmed, this
package will mirror the :mod:`network.ieee39` layout:

* ``build.py``     — ``build_nordic32_net()`` loads the converted net + fixes.
* ``constants.py`` — scalar constants, capability tables, line ratings.
* ``meta.py``      — ``Nordic32NetworkMeta`` dataclass.
* ``helpers.py``   — post-conversion parameter polish.
* ``scenarios/``   — scenario overrides.
"""
