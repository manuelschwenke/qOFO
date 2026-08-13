"""
Scenario registry for the IEEE 39-bus network.

Each scenario is a function ``(net, meta, **kwargs) -> (net, meta)`` that
modifies the base network in-place and returns updated metadata.
"""
from network.ieee39.scenarios.wind_replace import apply_wind_replace

# Both study scenarios share the validated transmission-side wind replacement.
# Their DSO installed capacities are selected later by ``add_hv_networks``
# from ``network.ieee39.constants.DSO_DER_CAPACITY_SCENARIOS``.
SCENARIO_REGISTRY = {
    "base_410": apply_wind_replace,
    "rural_700": apply_wind_replace,
}
