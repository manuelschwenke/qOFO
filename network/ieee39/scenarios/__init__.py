"""
Scenario registry for the IEEE 39-bus network.

Each scenario is a function ``(net, meta, **kwargs) -> (net, meta)`` that
modifies the base network in-place and returns updated metadata.
"""
from functools import partial

from network.ieee39.scenarios.scarcity import apply_scarcity
from network.ieee39.scenarios.wind_replace import apply_wind_replace

# Both study scenarios share the validated transmission-side wind replacement.
# Their DSO installed capacities are selected later by ``add_hv_networks``
# from ``network.ieee39.constants.DSO_DER_CAPACITY_SCENARIOS``.
#
# Scarcity redesign candidates (2026-11-08, screened before adoption):
#   ``*_north`` -- transmission wind layout "north" (W1, W2, W4 concentrated
#                  at IEEE bus 26; see wind_replace.py);
#   ``*_res``   -- coupling-bus parks of every STS with reserved capability
#                  (constants.RESERVED_OP_DIAGRAM);
#   ``*_no939`` -- line 9-39 not built (scarcity.py).
_north = partial(apply_wind_replace, ts_wind_layout="north")
_no939 = partial(apply_scarcity, lines_not_built=((9, 39),))
_north_no939 = partial(apply_scarcity, ts_wind_layout="north",
                       lines_not_built=((9, 39),))
SCENARIO_REGISTRY = {
    "base_410": apply_wind_replace,
    "rural_700": apply_wind_replace,
    "rural_700_res": apply_wind_replace,
    "rural_700_north": _north,
    "rural_700_north_res": _north,
    "rural_700_res_no939": _no939,
    "rural_700_north_res_no939": _north_no939,
}

#: Non-study build target: the unmodified IEEE 39-bus case, with the full
#: 10-machine fleet and no transmission-side wind replacement (see
#: ``base.md``).  It is the PowerFactory Gate-0 parity baseline -- the empty
#: variation stack in ``pf.pf_parity._PHASE_VARIATIONS`` and the ``base``
#: phase of ``export.make_snapshots`` / ``pf.pf_sync``.
#:
#: Deliberately NOT a member of :data:`SCENARIO_REGISTRY`.  That registry is
#: the set of *study* scenarios: ``tuning.scenarios.VALID_NETWORK_SCENARIOS``
#: is ``frozenset(SCENARIO_REGISTRY)`` and ``probe_dso_boundary_conditions``
#: offers ``sorted(SCENARIO_REGISTRY)`` as its CLI choices, so a no-op entry
#: here would re-admit a non-study network into tuning draws and study CLIs.
#: That is the 2026-07-31 defect: ``tuning.scenarios`` drew the then-unknown
#: name ``"base"`` for ~20 % of its validation specs, ``build_ieee39_net``
#: raised, ``run_one`` swallowed each into a sentinel cost, and a fifth of
#: every campaign was silently scored as a power-flow failure.
#:
#: ``build_ieee39_net`` resolves this name before the registry check and
#: writes it to ``net["ieee39_scenario"]``, which keeps ``add_hv_networks``
#: fail-fast: building HV underlays on the bare case raises rather than
#: silently picking a DER capacity it was never given.
BARE_SCENARIO = "bare"
