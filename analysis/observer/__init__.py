"""
Runtime stability observer sub-package.

Provides the passive spectral-gap observer that records per-block
``g_w^min`` at every cross-sensitivity refresh during a normal simulation
run.  Complementary to :mod:`analysis.stability_analysis` (Theorem 3.3
post-hoc framework).

The "spectral-gap" floor evaluated here is
``g_w >= ||M||_op - lam_min(M)`` with ``M = g_v M_V + g_q M_Q``.  This
condition was previously labelled "Bianchi" but does NOT actually appear
in Bianchi & Doerfler (2025); see the docstring of :mod:`stability_tuning`
for the naming-history note.
"""

from analysis.observer.stability_integration_ieee39 import (
    attach_observer,
    observer_record_fresh,
    write_observer_results_alongside_report,
    derive_tuned_gw,
    ObserverTunedGw,
)
from analysis.observer.stability_observer import (
    StabilityObserver,
    ObservationRecord,
    ZoneTrajectory,
    write_trajectory_report,
    plot_trajectory,
)
from analysis.observer.stability_tuning import (
    BlockLayout,
    StabilityResult,
    compute_min_gw_per_block,
    compute_min_gw_lmi,
    run_monte_carlo,
    aggregate_monte_carlo,
)

__all__ = [
    "attach_observer",
    "observer_record_fresh",
    "write_observer_results_alongside_report",
    "derive_tuned_gw",
    "ObserverTunedGw",
    "StabilityObserver",
    "ObservationRecord",
    "ZoneTrajectory",
    "write_trajectory_report",
    "plot_trajectory",
    "BlockLayout",
    "StabilityResult",
    "compute_min_gw_per_block",
    "compute_min_gw_lmi",
    "run_monte_carlo",
    "aggregate_monte_carlo",
]
