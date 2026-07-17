"""Ex-post voltage-tracking equity metrics for SBX-H.

The metric is diagnostic only and never feeds the controller.  For each
area, all monitored-bus voltage errors in an evaluation window are reduced
to one RMSE burden in mpu.  Areas then receive equal weight, irrespective of
their number of monitored buses.  A normalized Gini coefficient quantifies
how unequally that burden is distributed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from sbx_h.fail import rep1


@dataclass(frozen=True)
class VoltageTrackingEquity:
    """Voltage-tracking burden and equality across control areas.

    ``gini`` is normalized to ``[0, 1]``: zero means equal area RMSE and
    one is the limiting case where one area carries the complete burden.
    ``fairness`` is the presentation-equivalent ``1 - gini``.
    """

    area_rmse_mpu: Dict[int, float]
    mean_rmse_mpu: float
    worst_rmse_mpu: float
    worst_area: int
    gini: float
    fairness: float


def voltage_tracking_equity(
    errors_by_area_pu: Mapping[int, Sequence[float]],
) -> VoltageTrackingEquity:
    """Calculate per-area voltage RMSE and normalized burden inequality.

    Parameters
    ----------
    errors_by_area_pu:
        ``{area: [V_meas - V_ref, ...]}`` over all monitored buses and
        samples in the chosen evaluation window.  Every value within an
        area receives equal weight; every area receives equal weight in
        the cross-area statistics.
    """
    if not errors_by_area_pu:
        rep1("voltage-tracking equity needs at least one area")

    area_rmse: Dict[int, float] = {}
    for area, raw_errors in sorted(errors_by_area_pu.items()):
        errors = tuple(float(value) for value in raw_errors)
        if not errors:
            rep1(
                "voltage-tracking equity needs at least one error per area",
                area=area,
            )
        if not all(math.isfinite(value) for value in errors):
            rep1(
                "voltage-tracking equity received a non-finite error",
                area=area,
            )
        mean_square = sum(value * value for value in errors) / len(errors)
        area_rmse[int(area)] = 1000.0 * math.sqrt(mean_square)

    burdens = list(area_rmse.values())
    n_areas = len(burdens)
    mean_rmse = sum(burdens) / n_areas
    worst_area = min(
        area_rmse,
        key=lambda area: (-area_rmse[area], area),
    )
    worst_rmse = area_rmse[worst_area]

    total = sum(burdens)
    if n_areas == 1 or total <= 1e-12:
        gini = 0.0
    else:
        pairwise_distance = sum(
            abs(left - right)
            for left in burdens
            for right in burdens
        )
        gini = pairwise_distance / (2.0 * (n_areas - 1) * total)
        gini = min(1.0, max(0.0, gini))

    return VoltageTrackingEquity(
        area_rmse_mpu=area_rmse,
        mean_rmse_mpu=mean_rmse,
        worst_rmse_mpu=worst_rmse,
        worst_area=worst_area,
        gini=gini,
        fairness=1.0 - gini,
    )


__all__ = ["VoltageTrackingEquity", "voltage_tracking_equity"]
