"""
Scenario: reduced_gen_z2
========================
Remove the ex-slack synchronous generator at bus 30 from Zone 2.

After the slack is relocated from bus 30 to bus 38 (the IEEE standard
location), the generator that replaced the old ext_grid at bus 30 is
removed entirely.  This leaves Zone 2 with only Gen 1
(bus 31, grid_bus 9, ~650 MW) as its sole synchronous machine, producing
a low-inertia zone suitable for studying voltage-control stress.

Author: Manuel Schwenke / Claude Code
"""
from __future__ import annotations

import pandapower as pp

from network.ieee39.helpers import remove_generators


def apply_reduced_gen_z2(net, meta, *, new_gen_bus30_idx, **kwargs):
    """Apply the *reduced_gen_z2* scenario.

    Parameters
    ----------
    net : pp.pandapowerNet
        IEEE 39-bus network (modified in-place).
    meta : IEEE39NetworkMeta
        Current metadata catalogue.
    new_gen_bus30_idx : int
        Pandapower ``net.gen`` index of the generator that replaced the
        original ext_grid at bus 30 (returned by ``_swap_slack_to_bus38``).

    Returns
    -------
    (net, meta)
        The modified network and updated metadata.
    """
    meta = remove_generators(net, meta, [new_gen_bus30_idx])
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)
    return net, meta
