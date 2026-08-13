"""
tuning/bisect_switching.py
==========================
Calibrate an OLTC proximal weight against an operational switching budget.

Why bisection rather than BO
----------------------------
``g_w_tso_oltc`` (machine transformers) and ``g_w_dso_oltc`` (network coupling
transformers) price *integer* tap moves.  Two structural facts make them a poor
fit for a Bayesian search and a good fit for a 1-D root find.

**They are monotone.**  ``G_w`` is strictly diagonal
(``optimisation/miqp_solver.py:880``) and each OLTC block is a single scalar
replicated across that class
(``controller/multi_tso_coordinator.py:289``, ``multi_tso_dso.py:1276``),
appearing nowhere else in the problem.  So with

    V(g) = min_{(w,z) in S} [ q(w_c, z) + grad_f^T w + g * ||w_i||^2 ]

the objective is **affine in g** for each fixed feasible point, hence ``V`` is a
pointwise minimum of affine functions and therefore **concave**.  Any minimiser
supplies ``||w_i*(g)||^2`` as a supergradient of ``V`` at ``g``, and
supergradients of a concave function are non-increasing.  With
``int_max_step = 1`` the integer moves are ``w_i in {-1, 0, +1}``, so
``||w_i*||^2`` is exactly *the number of taps that move in that solve* —
**monotone non-increasing in g**.

**They have two exactly-flat tails.**  Let ``C = sup |q + grad_f^T w|`` over the
(compact) feasible set.  For ``g > C`` any non-zero ``w_i`` costs at least
``g - C > 0``, so ``w_i = 0`` strictly dominates: there is a finite ``g`` above
which *no tap ever moves* and the objective is exactly constant.  Below, the
switching rate is capped by ``int_cooldown`` and the wall-clock
``oltc_cooldown_s_*``, giving a second flat tail.  A density-ratio sampler such
as TPE cannot represent a plateau — every point in it looks equally good — which
is a large part of why these two coordinates showed ``|rho| <= 0.27`` against the
objective across 1555 recorded runs.

**Caveat, stated rather than assumed.**  The monotonicity above is *per solve*.
It does not chain into per-trajectory monotonicity, because a larger ``g`` shifts
the whole state sequence and can delay a tap into a worse operating point that
later forces two.  Empirically the total-switch curve is monotone in expectation
with local roughness, so this module brackets first, bisects to a *tolerance
band* rather than a point, and takes the median across the scenario set.

What it returns
---------------
A *specified* value — "``g_w_dso_oltc = X`` gives a median 9.4 tap operations per
day per transformer across the design envelope, within the +/-20 % band" — rather
than an arbitrary optimiser output.  Since EHV maintenance budgets are real
operational constraints, that converts a free tuning parameter into a stated
requirement, which is a far stronger position than a number BO happened to pick.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np

from configs.config import MultiTSOConfig
from tuning.metrics import CostWeights, NoiseFloors
from tuning.parameters import params_from_config
from tuning.runner import run_one
from tuning.scenarios import ScenarioSpec

__all__ = [
    "BisectionResult",
    "SwitchingProbe",
    "calibrate_switching_price",
]

#: Which recorded wear metric a field is calibrated against.
_FIELD_METRIC = {
    "g_w_tso_oltc": "tap_ops_per_h_tso",
    "g_w_dso_oltc": "tap_ops_per_h_dso",
}


@dataclass(frozen=True)
class SwitchingProbe:
    """One evaluation of the switching rate at a given weight."""

    g_w: float
    ops_per_day: float
    n_scenarios: int
    n_failed: int


@dataclass(frozen=True)
class BisectionResult:
    """Outcome of :func:`calibrate_switching_price`.

    Attributes
    ----------
    field
        Config field calibrated (``g_w_tso_oltc`` / ``g_w_dso_oltc``).
    g_w
        Recommended value.  Meaningful only when ``status == "bracketed"``.
    achieved_ops_per_day
        Median switching rate at :attr:`g_w`, per transformer.
    status
        * ``"bracketed"`` — the target lies strictly inside ``[lo, hi]`` and the
          bisection converged into the tolerance band.
        * ``"plateau_high"`` — even at ``lo`` the rate is at or below target:
          the actuator is already quieter than the budget, so the budget does
          not bind and any ``g_w >= lo`` satisfies it.  **Do not read the
          returned value as "the tuned value"** — the constraint is slack.
        * ``"plateau_low"`` — even at ``hi`` the rate exceeds target: no weight
          in the bracket buys enough quiet.  The binding constraint is elsewhere
          (cooldowns, scenario severity, or the loop gain), not this weight.
        * ``"no_signal"`` — every probe failed to simulate.
    ladder
        Every probe, in evaluation order.  Publish it: it is the evidence that
        the response really is monotone over the bracket, and it is a thesis
        figure on its own.
    """

    field: str
    g_w: float
    achieved_ops_per_day: float
    target_ops_per_day: float
    status: Literal["bracketed", "plateau_high", "plateau_low", "no_signal"]
    n_evaluations: int
    ladder: tuple[SwitchingProbe, ...] = ()
    #: The acceptance band the search was driven at, so
    #: :attr:`within_tolerance` judges against the requested tolerance rather
    #: than a hardcoded one.  Defaults to the historical 0.2.
    tol_rel: float = 0.2

    @property
    def within_tolerance(self) -> bool:
        """Did the search actually reach the band it was *asked* for?

        Checks :attr:`tol_rel`, not a hardcoded 0.2.  With the literal in place
        a run driven at ``--tol-rel 0.1`` reported ``within_tolerance=True`` for
        a point 19.6 % off target: the bisection loop honoured the tighter band
        and exhausted ``max_iter`` without entering it, but the reported flag
        still compared against 0.2 and called the failure a success.
        """
        if self.status != "bracketed" or self.target_ops_per_day <= 0:
            return False
        rel = abs(self.achieved_ops_per_day - self.target_ops_per_day)
        return rel / self.target_ops_per_day <= self.tol_rel


def _probe(
    field: str,
    g_w: float,
    baseline_cfg: MultiTSOConfig,
    scenarios: Sequence[ScenarioSpec],
    cost_weights: CostWeights | None,
    noise_floors: NoiseFloors | None,
    params: dict[str, float] | None,
    runner: Callable[..., object] | None,
) -> SwitchingProbe:
    """Median switching rate across the scenario set at one weight."""
    metric = _FIELD_METRIC[field]
    cfg = dataclasses.replace(baseline_cfg, **{field: float(g_w)})
    # ``run_one`` overlays a *complete* BO parameter set onto ``cfg`` through
    # ``apply_to_config``, which rejects a partial dict -- so the historical
    # ``params or {}`` raised ``KeyError`` on all 8 dims the moment the real
    # runner was used.  Every test here injects a ``runner`` stub, so this path
    # was never exercised.
    #
    # Seed from ``cfg`` (already carrying the swept weight) and let ``field``
    # win explicitly: the swept weight is *itself* a BO dimension, so seeding
    # from ``baseline_cfg`` -- or letting a caller's ``params`` through last --
    # would overwrite it, and every rung of the ladder would quietly run at the
    # baseline weight.  That failure is silent: the rate would be constant in
    # ``g_w`` and the bisection would report a bogus ``plateau_*``.
    probe_params = params_from_config(cfg)
    if params:
        probe_params.update(params)
    probe_params[field] = float(g_w)
    rates: list[float] = []
    n_failed = 0
    run = runner or run_one
    for sc in scenarios:
        res = run(probe_params, sc, cfg, cost_weights, noise_floors)
        m = res.metrics
        if not m.feasible:
            n_failed += 1
            continue
        value = float(getattr(m, metric))
        if math.isfinite(value):
            rates.append(value)
    # Median, not mean: one pathological scenario should not drag the
    # calibration, and the response is rough at the per-trajectory level.
    ops = float(np.median(rates)) if rates else float("nan")
    return SwitchingProbe(float(g_w), ops, len(scenarios), n_failed)


def calibrate_switching_price(
    field: str,
    target_ops_per_day: float,
    baseline_cfg: MultiTSOConfig,
    scenarios: Sequence[ScenarioSpec],
    *,
    lo: float = 1.0,
    hi: float = 1.0e5,
    tol_rel: float = 0.2,
    max_iter: int = 8,
    cost_weights: CostWeights | None = None,
    noise_floors: NoiseFloors | None = None,
    params: dict[str, float] | None = None,
    runner: Callable[..., object] | None = None,
    verbose: bool = True,
) -> BisectionResult:
    """Find the OLTC weight meeting a switching budget, by log-space bisection.

    Parameters
    ----------
    field
        ``"g_w_tso_oltc"`` or ``"g_w_dso_oltc"``.
    target_ops_per_day
        Tap operations per day **per transformer**, worst transformer in the
        class.  Aggregating over the fleet would let one hunting changer hide
        behind a quiet fleet.
    lo, hi
        Bracket in ``g_w``.  Both ends are probed first: outside the bracket the
        response is exactly flat, so a bisection started blind would converge to
        a meaningless interior point.
    tol_rel
        Relative half-width of the acceptance band around the target.  A point
        target is not meaningful given the per-trajectory roughness.
    max_iter
        Cap on bisection steps (each is one full scenario-set evaluation).
        8 steps resolves ~3 decades.

    Notes
    -----
    Cost is ``(2 + max_iter) * len(scenarios)`` simulations — about 10 x 5 at
    ~105 s each, i.e. under 20 minutes per class. That is far cheaper than
    handing the coordinate to a sampler that cannot represent its plateaus.
    """
    if field not in _FIELD_METRIC:
        raise ValueError(
            f"field must be one of {sorted(_FIELD_METRIC)}, got {field!r}"
        )
    if target_ops_per_day <= 0.0:
        raise ValueError("target_ops_per_day must be positive")
    if not (0.0 < lo < hi):
        raise ValueError(f"need 0 < lo < hi, got lo={lo}, hi={hi}")

    ladder: list[SwitchingProbe] = []

    def probe(g: float) -> SwitchingProbe:
        p = _probe(field, g, baseline_cfg, scenarios, cost_weights,
                   noise_floors, params, runner)
        ladder.append(p)
        if verbose:
            print(
                f"  [bisect:{field}] g_w={p.g_w:11.4g} -> "
                f"{p.ops_per_day:8.3f} ops/day/trafo"
                f"{'  (all scenarios failed)' if p.n_failed == p.n_scenarios else ''}",
                flush=True,
            )
        return p

    # ── Bracket first.  The response has exactly-flat tails on both sides, so
    #    a blind bisection would happily converge inside one of them.
    p_lo = probe(lo)     # smallest weight  -> most switching
    p_hi = probe(hi)     # largest weight   -> least switching

    if not math.isfinite(p_lo.ops_per_day) and not math.isfinite(p_hi.ops_per_day):
        return BisectionResult(field, float("nan"), float("nan"),
                               target_ops_per_day, "no_signal", len(ladder),
                               tuple(ladder), tol_rel=tol_rel)

    if math.isfinite(p_lo.ops_per_day) and p_lo.ops_per_day <= target_ops_per_day:
        # Even the cheapest weight is quieter than the budget: the constraint
        # is slack and this weight is not what limits switching.
        return BisectionResult(field, lo, p_lo.ops_per_day, target_ops_per_day,
                               "plateau_high", len(ladder), tuple(ladder),
                               tol_rel=tol_rel)

    if math.isfinite(p_hi.ops_per_day) and p_hi.ops_per_day > target_ops_per_day:
        # Even the most expensive weight cannot buy enough quiet.
        return BisectionResult(field, hi, p_hi.ops_per_day, target_ops_per_day,
                               "plateau_low", len(ladder), tuple(ladder),
                               tol_rel=tol_rel)

    a, b = float(lo), float(hi)
    best = p_hi
    for _ in range(max_iter):
        mid = math.sqrt(a * b)          # bisect in log space
        p = probe(mid)
        if not math.isfinite(p.ops_per_day):
            # Treat an unsimulable point as "too aggressive" and damp further.
            a = mid
            continue
        best = p
        rel = abs(p.ops_per_day - target_ops_per_day) / target_ops_per_day
        if rel <= tol_rel:
            break
        if p.ops_per_day > target_ops_per_day:
            a = mid                     # too much switching -> raise the price
        else:
            b = mid                     # too little -> lower it

    return BisectionResult(field, best.g_w, best.ops_per_day,
                           target_ops_per_day, "bracketed", len(ladder),
                           tuple(ladder), tol_rel=tol_rel)
