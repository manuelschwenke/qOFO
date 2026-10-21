"""Joint OFO shunt decision with independent slow tracking/reserve memories.

Bank levels are genuine additional MIQP integer variables. The adapter returns
the original control slice and an explicit output contribution to the caller.
The runner applies the selected bank step and feedforward once, after the solve.
"""
from dataclasses import dataclass, replace
import math
from time import perf_counter
import numpy as np

from controller.shunt_support import (STSShuntSupportConfig, STSShuntRequest,
                                      fleet_margin_cost, fleet_reserve_direction,
                                      support_request)


@dataclass
class DemandMemory:
    """Signed accumulated demand in nameplate Mvar, with sampled persistence."""
    value: float = 0.
    direction: int = 0
    since: float | None = None
    last_t: float | None = None

    def clear(self):
        self.value, self.direction, self.since = 0., 0, None

    def observe(self, gradient, t, bank):
        cfg = bank.config
        if self.last_t is not None and t < self.last_t:
            raise ValueError("Demand observation time moved backwards")
        gap = self.last_t is not None and t-self.last_t > cfg.max_sample_gap_s
        self.last_t = t
        direction = (-1 if gradient > cfg.gradient_deadband else
                     1 if gradient < -cfg.gradient_deadband else 0)
        if not direction or not 0 <= bank.level+direction <= cfg.n_levels:
            self.clear()
            return
        if gap or direction != self.direction:
            self.clear()
            self.direction, self.since = direction, t
        self.value = float(np.clip(self.value-gradient/(2*cfg.g_w),
                                   -cfg.q_step_mvar, cfg.q_step_mvar))

    def ready(self, t, bank):
        c = bank.config
        return (self.direction if self.since is not None
                and t-self.since >= c.t_persistence_s
                and abs(self.value) >= .5*c.q_step_mvar+c.delta else 0)


@dataclass
class BankModel:
    bank: object
    output_step: np.ndarray       # all TS outputs, per positive bank level
    interface_indices: np.ndarray
    pcc_control_indices: np.ndarray
    interface_step: np.ndarray    # all affected STS interfaces, per level
    interface_offset: np.ndarray
    feedback: object


@dataclass
class HybridContext:
    time_s: float
    models: list[BankModel]
    output_gradient: np.ndarray
    voltage_indices: np.ndarray
    voltage_targets: np.ndarray
    # TS fleet reserve, supervisory only. ``fleet`` carries the directional
    # margins of the machines and the TS DER; ``n_gen`` is the length of the
    # Q_gen output tail and ``n_der`` the length of the DER control head.
    fleet: object = None
    n_gen: int = 0
    n_der: int = 0


class HybridShuntSolver:
    """A shared MIQP decision and an optional, bounded slow support proposal.

    Regular OFO has first choice. If it holds every bank, confirmed support
    candidates are solved with their bank step fixed and all ordinary controls
    re-optimised. A support candidate pins affected *total* interface references
    to the regular solution by reducing the delegated requests by its bank
    effect. Tracking has priority over reserve. No plant model is used here.

    A confirmed candidate is admitted by one of two gates, selected by
    ``support_gate``. ``"voltage_allowance"`` bounds the predicted increase of
    the worst observed bus's voltage tracking error by
    ``voltage_error_allowance_pu``. ``"shared_objective"`` instead requires the
    first-order increase of the shared TS objective, ``grad_y f @ dy`` between
    the candidate's and the regular solution's predicted outputs, not to exceed
    ``support_exchange`` times the support cost it removes, so the exchange is
    priced in the weights the controller is tuned on rather than in a pu bound.
    That measure excludes the MIQP's quadratic terms, which regularise the step
    rather than express ``f``, and it covers the output part of the objective
    only, as ``grad_y f`` does. Both gates keep the hard predicted voltage
    corridor and refuse any candidate that increases a predicted output-limit
    violation.
    """
    def __init__(self, solver, context_provider, *, movement_weight,
                 support_config: STSShuntSupportConfig,
                 voltage_error_allowance_pu=.001,
                 support_gate="voltage_allowance", support_exchange=0.,
                 ts_reserve_weight=0., ts_reserve_margin_steps=1.,
                 regular_confirmations=1):
        if not math.isfinite(movement_weight) or movement_weight <= 0:
            raise ValueError("Hybrid shunt movement weight must be positive")
        if (not math.isfinite(voltage_error_allowance_pu)
                or voltage_error_allowance_pu < 0):
            raise ValueError("Voltage error allowance must be finite and non-negative")
        if support_gate not in ("voltage_allowance", "shared_objective"):
            raise ValueError(f"Unknown support gate: {support_gate}")
        if not math.isfinite(support_exchange) or support_exchange < 0:
            raise ValueError("Support exchange weight must be finite and non-negative")
        if not math.isfinite(ts_reserve_weight) or ts_reserve_weight < 0:
            raise ValueError("TS fleet reserve weight must be finite and non-negative")
        if (not math.isfinite(ts_reserve_margin_steps)
                or ts_reserve_margin_steps <= 0):
            raise ValueError("TS fleet reserve margin must be a positive step count")
        if int(regular_confirmations) != regular_confirmations or regular_confirmations < 1:
            raise ValueError("regular_confirmations must be an integer >= 1")
        self.solver, self.context_provider = solver, context_provider
        self.movement_weight = movement_weight
        self.support_config = support_config
        self.voltage_error_allowance_pu = voltage_error_allowance_pu
        self.support_gate = support_gate
        self.support_exchange = support_exchange
        self.ts_reserve_weight = ts_reserve_weight
        self.ts_reserve_margin_steps = ts_reserve_margin_steps
        self.regular_confirmations = int(regular_confirmations)
        self._regular_pending = {}
        self.memories = {}
        self.selected = None
        self.last_context = None
        self.diagnostics = {}

    def __getattr__(self, name):
        return getattr(self.solver, name)

    @staticmethod
    def _increment(problem, result):
        # OFO solver ordering is [continuous | integer].
        return np.r_[problem.alpha*result.w_continuous, result.w_integer]

    def _augment(self, p, ctx):
        n, m = p.n_total, len(ctx.models)
        columns = np.column_stack([x.output_step for x in ctx.models])
        if (columns.shape != (p.n_outputs, m)
                or not np.all(np.isfinite(columns))
                or ctx.output_gradient.shape != (p.n_outputs,)
                or not np.all(np.isfinite(ctx.output_gradient))):
            raise ValueError("Invalid hybrid bank model")
        levels = np.array([x.bank.level for x in ctx.models], dtype=float)
        lo, hi = levels.copy(), levels.copy()
        for j, x in enumerate(ctx.models):
            directions = x.bank.available_directions(ctx.time_s)
            lo[j] += min([0]+directions)
            hi[j] += max([0]+directions)
        G = np.zeros((n+m, n+m))
        G[:n, :n] = p.G_w
        G[n:, n:] = np.eye(m)*self.movement_weight
        yl, yu = p.y_lower.copy(), p.y_upper.copy()
        rows = ctx.voltage_indices
        yl[rows] = np.maximum(yl[rows], max(x.bank.config.y_h_min for x in ctx.models))
        yu[rows] = np.minimum(yu[rows], min(x.bank.config.y_h_max for x in ctx.models))
        return replace(p, n_integer=p.n_integer+m, G_w=G,
            grad_f=np.r_[p.grad_f, ctx.output_gradient @ columns],
            H_tilde=np.column_stack([p.H_tilde, columns]),
            u_current=np.r_[p.u_current, levels],
            u_lower=np.r_[p.u_lower, lo], u_upper=np.r_[p.u_upper, hi],
            y_lower=yl, y_upper=yu,
            integer_indices=p.integer_indices+list(range(n, n+m)),
            one_step_indices=tuple(range(n, n+m)),
            hard_output_indices=tuple(int(i) for i in rows))

    def _regular_confirmed(self, ctx, j, direction):
        """Whether the regular path has selected this bank and direction for
        ``regular_confirmations`` consecutive TS observations.

        A count of 1, the default, is the immediate behaviour. Higher counts
        require the proposal to survive the ordinary actuators' attempt to
        recover the deviation: a changed bank, a changed direction, an
        observation gap, or an instant in which the MIQP holds every bank all
        reset it. The deviation must therefore be both large enough to clear
        the movement weight and persistent enough to outlast the continuous
        response.
        """
        bank = ctx.models[j].bank
        key = bank.config.shunt_idx
        previous = self._regular_pending.get(key)
        gap = (previous is not None
               and ctx.time_s - previous[2] > bank.config.max_sample_gap_s)
        if previous is None or previous[0] != direction or gap:
            count = 1
        else:
            count = previous[1] + 1
        # Only one proposal is tracked: a different bank means the previous
        # proposal was not sustained.
        self._regular_pending = {key: (direction, count, ctx.time_s)}
        self.diagnostics["regular_pending"] = {
            "shunt_idx": key, "direction": direction, "count": count,
            "required": self.regular_confirmations}
        return count >= self.regular_confirmations

    def _fleet_enabled(self, ctx):
        return self.ts_reserve_weight > 0.0 and ctx.fleet is not None

    def _fleet_margin_mvar(self, bank):
        return self.ts_reserve_margin_steps*bank.config.q_step_mvar

    def _fleet_observation(self, ctx, model):
        """Direction the fleet margins demand, and the gradient that drives the
        confirmation memory.

        The fleet condition is a **threshold on a directional Mvar margin**, not
        a gradient: the DER half of the fleet is a control the candidate
        re-optimises, so no ``dQ/dlevel`` sensitivity exists for it and no
        gradient can be formed. The memory is therefore a pure confirmation
        timer — one observation saturates the accumulator, so readiness is
        governed by ``t_persistence_s`` alone. The weight no longer decides
        whether the path fires; the margin threshold does. It only prices the
        benefit in the acceptance test.
        """
        if not self._fleet_enabled(ctx):
            return 0, 0.
        bank = model.bank
        direction = fleet_reserve_direction(ctx.fleet,
            margin_mvar=self._fleet_margin_mvar(bank),
            injects=1 if bank.config.kind == "MSC" else -1,
            level=bank.level, n_levels=bank.config.n_levels)
        if not direction:
            return 0, 0.
        saturating = 2.*bank.config.g_w*bank.config.q_step_mvar
        return direction, -direction*saturating

    def _fleet_cost(self, ctx, bank, u=None, y=None):
        """Fleet margin cost at a solution: machines from the predicted outputs,
        DER from the chosen controls."""
        return fleet_margin_cost(ctx.fleet,
            weight=self.ts_reserve_weight,
            margin_mvar=self._fleet_margin_mvar(bank),
            q_gen=(None if not ctx.n_gen or y is None
                   else np.asarray(y)[len(y)-ctx.n_gen:]),
            q_der=None if not ctx.n_der or u is None else np.asarray(u)[:ctx.n_der])

    def _observe(self, ctx):
        observations = []
        for model in ctx.models:
            b, f = model.bank, model.feedback
            requests = {}
            common = dict(h_q=model.interface_step/b.config.q_step_mvar,
                q_step_mvar=b.config.q_step_mvar, level=b.level,
                n_levels=b.config.n_levels, t_now=ctx.time_s)
            requests["sts_tracking"] = support_request(f, **common,
                config=replace(self.support_config, reserve_weight=0))
            # Observe actual near-limit loading independently of tracking.
            # Commit eligibility below still requires tracking within tolerance.
            measured = replace(f, q_requested=f.q_measured) if f is not None else None
            requests["sts_reserve"] = support_request(measured, **common,
                config=replace(self.support_config, tracking_weight=0))
            gradients = {k: (r.gradient if r else 0.) for k, r in requests.items()}
            if self._fleet_enabled(ctx):
                # Supervisory only: absent from the shared objective, so the
                # continuous actuators keep their voltage task.
                gradients["ts_reserve"] = self._fleet_observation(ctx, model)[1]
            for kind, gradient in gradients.items():
                mem = self.memories.setdefault((b.config.shunt_idx, kind), DemandMemory())
                mem.observe(gradient, ctx.time_s, b)
                observations.append(dict(shunt_idx=b.config.shunt_idx, kind=kind,
                    gradient=gradient, accumulated_mvar=mem.value,
                    ready_direction=mem.ready(ctx.time_s, b)))
        return observations

    def solve(self, problem):
        started = perf_counter()
        if self.selected is not None:
            raise RuntimeError("Previous hybrid shunt decision was not applied")
        ctx = self.context_provider()
        self.last_context = ctx
        observations = self._observe(ctx)
        self.diagnostics = {"memory": observations, "candidate_solves": 1,
                            "selected_reason": None, "rejected_support": 0,
                            "support_candidates": []}
        if self._fleet_enabled(ctx):
            # Directional margins in Mvar, recorded whether or not either is
            # short: without them an absent request cannot be told from a fleet
            # that simply has room.
            up, down = ctx.fleet.directional()
            self.diagnostics["fleet_margin_up_mvar"] = up
            self.diagnostics["fleet_margin_down_mvar"] = down
        joint = self._augment(problem, ctx)
        regular = self.solver.solve(joint)
        if not regular.is_feasible:
            # An unavoidable voltage violation must not prevent the ordinary
            # OFO recovery action. No bank moves in this fallback.
            result = self.solver.solve(problem)
            self.diagnostics["candidate_solves"] += 1
            self.diagnostics["regular_fallback"] = True
            return replace(result, solve_time_s=perf_counter()-started)
        moves = regular.w_integer[problem.n_integer:]
        if np.count_nonzero(moves):
            j = int(np.flatnonzero(moves)[0])
            if self._regular_confirmed(ctx, j, int(moves[j])):
                return self._finish(problem, joint, regular, ctx, j, "objective", started)
            # Not yet confirmed. Re-solve with every bank held, so the ordinary
            # actuators are dispatched consistently and get another instant to
            # recover the deviation themselves. If they do, the MIQP stops
            # selecting the bank and the count resets: that is what makes the
            # criterion "large AND not recovered by the other actuators".
            lo, hi = joint.u_lower.copy(), joint.u_upper.copy()
            lo[problem.n_total:] = hi[problem.n_total:] = joint.u_current[problem.n_total:]
            regular = self.solver.solve(replace(joint, u_lower=lo, u_upper=hi))
            self.diagnostics["candidate_solves"] += 1
            if not regular.is_feasible:
                result = self.solver.solve(problem)
                self.diagnostics["candidate_solves"] += 1
                self.diagnostics["regular_fallback"] = True
                return replace(result, solve_time_s=perf_counter()-started)
        else:
            self._regular_pending.clear()

        regular_u = joint.u_current+self._increment(joint, regular)
        regular_y = joint.y_current+joint.H_tilde @ self._increment(joint, regular)
        viable = []
        for j, model in enumerate(ctx.models):
            b = model.bank
            for priority, kind in enumerate(("sts_tracking", "sts_reserve",
                                             "ts_reserve")):
                mem = self.memories.get((b.config.shunt_idx, kind))
                if mem is None:
                    continue
                direction = mem.ready(ctx.time_s, b)
                if direction not in b.available_directions(ctx.time_s):
                    continue
                indices = model.pcc_control_indices
                if kind == "ts_reserve":
                    # The condition must still hold now: a memory can be ready
                    # from earlier observations while the margin has recovered.
                    now, _ = self._fleet_observation(ctx, model)
                    cost = self._fleet_cost(ctx, b) if now == direction else 0.
                    request = (STSShuntRequest("ts_reserve", -float(direction),
                                               np.zeros(0), cost, cost)
                               if now == direction else None)
                else:
                    cfg = replace(self.support_config,
                        reserve_weight=0 if kind == "sts_tracking" else self.support_config.reserve_weight,
                        tracking_weight=0 if kind == "sts_reserve" else self.support_config.tracking_weight)
                    request = support_request(model.feedback,
                        h_q=model.interface_step/b.config.q_step_mvar,
                        q_step_mvar=b.config.q_step_mvar, level=b.level,
                        n_levels=b.config.n_levels, t_now=ctx.time_s, config=cfg,
                        current_q_requested=regular_u[indices]+model.interface_offset)
                if request is None or request.kind != kind or -np.sign(request.gradient) != direction:
                    continue
                diagnostic = dict(shunt_idx=b.config.shunt_idx, kind=kind,
                                  direction=direction, rejection=None)
                self.diagnostics["support_candidates"].append(diagnostic)
                lo, hi = joint.u_lower.copy(), joint.u_upper.copy()
                lo[problem.n_total:] = hi[problem.n_total:] = joint.u_current[problem.n_total:]
                lo[problem.n_total+j] = hi[problem.n_total+j] = b.level+direction
                if kind != "ts_reserve":
                    # An STS request transfers its bank contribution out of the
                    # delegated request, holding the total subordinate reference
                    # fixed. A TS fleet request must NOT: leaving the request
                    # free keeps the feedforward offset-only convention, whose
                    # purpose is that the subordinate controller does not
                    # counteract the switch (runner, "q_adj" / offset pair).
                    target = regular_u[indices]-direction*model.interface_step
                    if np.any(target < lo[indices]-1e-8) or np.any(target > hi[indices]+1e-8):
                        self.diagnostics["rejected_support"] += 1
                        diagnostic["rejection"] = "delegated_bounds"
                        continue
                    lo[indices] = hi[indices] = target
                candidate_problem = replace(joint, u_lower=lo, u_upper=hi)
                candidate = self.solver.solve(candidate_problem)
                self.diagnostics["candidate_solves"] += 1
                if not candidate.is_feasible:
                    self.diagnostics["rejected_support"] += 1
                    diagnostic["rejection"] = "joint_infeasible"
                    continue
                y = joint.y_current+joint.H_tilde @ self._increment(joint, candidate)
                v = ctx.voltage_indices
                v_error = np.abs(y[v]-ctx.voltage_targets)
                base_error = np.abs(regular_y[v]-ctx.voltage_targets)
                excess = lambda a: np.maximum(joint.y_lower-a, 0)+np.maximum(a-joint.y_upper, 0)
                dv_error = float(np.max(v_error-base_error, initial=0.))
                violation_increase = float(np.max(excess(y)-excess(regular_y), initial=0.))
                # First-order change of the shared objective between the two
                # predicted operating points. The MIQP's quadratic G_w terms are
                # trust-region regularisation rather than part of f, and a
                # candidate is forced to move its delegated request by a whole
                # bank step, so comparing objective values would be dominated
                # by that step-size price and not by the exchange itself.
                shared_increase = float(ctx.output_gradient @ (y-regular_y))
                reduction = float(request.cost_before-request.cost_after)
                if kind == "ts_reserve":
                    # The candidate re-optimised every ordinary control around
                    # the forced step, so an AVR or DER adjustment can cancel the
                    # relief. Measure the benefit on the COMPLETE candidate,
                    # against the regular solution, so the bank is not credited
                    # with a recovery the ordinary controller achieves anyway.
                    candidate_u = joint.u_current+self._increment(joint, candidate)
                    before = self._fleet_cost(ctx, b, u=regular_u, y=regular_y)
                    after = self._fleet_cost(ctx, b, u=candidate_u, y=y)
                    reduction = before-after
                    diagnostic["fleet_cost_regular"] = before
                    diagnostic["fleet_cost_candidate"] = after
                    diagnostic["fleet_reduction_full_candidate"] = reduction
                    if reduction <= 1e-12:
                        self.diagnostics["rejected_support"] += 1
                        diagnostic["rejection"] = "no_fleet_relief"
                        continue
                diagnostic["max_voltage_error_increase_pu"] = dv_error
                diagnostic["max_output_violation_increase"] = violation_increase
                diagnostic["shared_objective_increase"] = shared_increase
                diagnostic["support_cost_reduction"] = reduction
                if self.support_gate == "shared_objective":
                    too_costly = shared_increase > self.support_exchange*reduction+1e-9
                else:
                    too_costly = dv_error > self.voltage_error_allowance_pu+1e-8
                if too_costly or violation_increase > 1e-7:
                    self.diagnostics["rejected_support"] += 1
                    diagnostic["rejection"] = (
                        ("shared_objective" if self.support_gate == "shared_objective"
                         else "voltage_tracking") if too_costly else "output_violation")
                    continue
                benefit = reduction/max(request.cost_before, 1e-12)
                viable.append((priority, -benefit, candidate.objective_value, j, kind, candidate))
        if viable:
            chosen = min(viable, key=lambda a: a[:4])
            return self._finish(problem, joint, chosen[5], ctx, chosen[3], chosen[4], started)
        return self._finish(problem, joint, regular, ctx, None, None, started)

    def _finish(self, base, joint, result, ctx, j, kind, started):
        moves = result.w_integer[base.n_integer:]
        if np.sum(np.abs(moves)) > 1:
            raise RuntimeError("Hybrid MIQP returned more than one bank step")
        if j is not None:
            self.selected = (ctx.models[j], int(moves[j]), kind, ctx.time_s)
        self.diagnostics["selected_reason"] = kind
        self.diagnostics["bank_moves"] = moves.tolist()
        extra = joint.H_tilde[:, base.n_total:] @ moves
        return replace(result, w_integer=result.w_integer[:base.n_integer].copy(),
            additional_output_change=extra, solve_time_s=perf_counter()-started)

    def take_commit(self):
        if self.selected is None:
            return None
        model, direction, kind, t = self.selected
        commit = model.bank.commit_selected(direction, t, kind)
        self.selected = None
        self._regular_pending.clear()
        # A bank changes the shared operating point. All slow requests in this
        # zone must be observed again before another slow commit.
        for memory in self.memories.values():
            memory.clear()
        return commit, model.interface_indices, direction*model.interface_step
