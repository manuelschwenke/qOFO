"""Controller-intent schedule tests for the active SBX-H adapter."""

from types import SimpleNamespace

import numpy as np

from sbx_h.adapter import controller_intent_schedule


def test_controller_intent_not_plant_snapshot(plant) -> None:
    net, zone_map, registry, *_ = plant
    controllers = {}
    intended_by_area = {1: 1.031, 2: 1.032, 3: 1.033}
    for area, buses in zone_map.items():
        bus_list = [int(bus) for bus in buses]
        controllers[area] = SimpleNamespace(
            config=SimpleNamespace(
                voltage_bus_indices=bus_list,
                v_setpoints_pu=np.full(
                    len(bus_list), intended_by_area[area], dtype=float
                ),
            )
        )

    corridor = registry[(2, 3)]
    schedule = controller_intent_schedule(corridor, controllers)
    _, v_a, v_b = schedule[0]

    assert v_a == (intended_by_area[2],) * len(corridor.lines)
    assert v_b == (intended_by_area[3],) * len(corridor.lines)
    measured = tuple(
        float(net.res_bus.at[line.bus_b, "vm_pu"])
        for line in corridor.lines
    )
    assert not np.allclose(v_b, measured, atol=1e-6)
