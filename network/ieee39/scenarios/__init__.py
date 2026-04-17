"""
Scenario registry for the IEEE 39-bus network.

Each scenario is a function ``(net, meta, **kwargs) -> (net, meta)`` that
modifies the base network in-place and returns updated metadata.
"""
from network.ieee39.scenarios.reduced_gen_z2 import apply_reduced_gen_z2
from network.ieee39.scenarios.wind_replace import apply_wind_replace

SCENARIO_REGISTRY = {
    "base": lambda net, meta, **kw: (net, meta),
    "reduced_gen_z2": apply_reduced_gen_z2,
    "wind_replace": apply_wind_replace,
}
