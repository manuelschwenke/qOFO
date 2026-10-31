"""
Reactive limit of the network equivalent (plant side).
=======================================================

pandapower treats the reference machine (``net.gen.slack``) as a slack bus and
never enforces its ``[min_q_mvar, max_q_mvar]``: ``enforce_q_lims`` converts
only PV machines to PQ.  The IEEE 39-bus equivalent G1 (10 GVA) therefore acted
as an infinite reactive source.  :class:`SlackQLimit` gives it the PV -> PQ
behaviour of every other machine: while its Q lies inside the limits the
terminal voltage is held at the set-point; at a limit the voltage is moved
until Q sits on the limit, i.e. Q is held and the voltage floats.  The machine
stays the angle reference and keeps its distributed-slack weight.

The controller acts inside the plant's ``pp.runpp(run_control=True)`` loop,
like the DER Q(V) loops.  No OFO dispatches this machine's voltage
(``dispatch_slack_gen_v_ref = False``), so the set-point is fixed at
construction.

Author: Manuel Schwenke / Claude Code (2026-10-31)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandapower as pp
from pandapower.control.basic_controller import Controller


class SlackQLimit(Controller):
    """Hold the reference machine's Q inside ``[q_min, q_max]`` by its voltage.

    Each control step estimates ``k = dQ/dV`` at the terminal by a secant over
    the iterations (nominal ``k0_per_sn * sn_mva``; measured for G1 about
    2000-3000 Mvar/pu with the DER loops responding) and moves ``vm_pu`` by a
    damped Newton step towards the violated limit, or back towards the
    set-point once Q has left the hysteresis band ``band_mvar`` at the limit.
    """

    def __init__(
        self,
        net: pp.pandapowerNet,
        gen_idx: int,
        q_min_mvar: float,
        q_max_mvar: float,
        *,
        tol_mvar: float = 10.0,
        band_mvar: float = 30.0,
        damping: float = 0.5,
        max_dv_pu: float = 0.02,
        k0_per_sn: float = 0.25,
        in_service: bool = True,
        order: int = 1,
        level: int = 0,
        index: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(net, in_service=in_service, order=order, level=level,
                         index=index, **kwargs)
        if not q_min_mvar < q_max_mvar:
            raise ValueError("q_min_mvar must be below q_max_mvar")
        self.gen_idx = int(gen_idx)
        self.q_min = float(q_min_mvar)
        self.q_max = float(q_max_mvar)
        self.v_set = float(net.gen.at[self.gen_idx, "vm_pu"])
        self.tol = float(tol_mvar)
        self.band = float(band_mvar)
        self.damping = float(damping)
        self.max_dv = float(max_dv_pu)
        self.k0 = float(k0_per_sn) * float(net.gen.at[self.gen_idx, "sn_mva"])
        self._prev: Optional[tuple] = None
        self.applied = False

    # ------------------------------------------------------------------
    def _state(self, net: pp.pandapowerNet) -> tuple:
        return (float(net.gen.at[self.gen_idx, "vm_pu"]),
                float(net.res_gen.at[self.gen_idx, "q_mvar"]))

    def _target(self, vm: float, q: float) -> Optional[float]:
        """Q the machine must move to; NaN = back towards the set-point; None =
        nothing to do.  Off the set-point the machine holds while Q lies in the
        hysteresis band next to the limit that moved it there, so the voltage
        does not chatter against the DER Q(V) loops."""
        if q > self.q_max + self.tol:
            return self.q_max
        if q < self.q_min - self.tol:
            return self.q_min
        if abs(vm - self.v_set) < 1e-7:
            return None
        if vm < self.v_set:            # lowered by the upper limit
            return float("nan") if q < self.q_max - self.band else None
        return float("nan") if q > self.q_min + self.band else None

    # ------------------------------------------------------------------
    def initialize_control(self, net: pp.pandapowerNet) -> None:
        self.applied = False
        self._prev = None

    def is_converged(self, net: pp.pandapowerNet) -> bool:
        if not bool(net.gen.at[self.gen_idx, "in_service"]):
            return True
        if not self.applied:
            # First call of a PF: converged only if nothing needs doing.
            return self._target(*self._state(net)) is None
        return self._target(*self._state(net)) is None

    def control_step(self, net: pp.pandapowerNet) -> None:
        vm, q = self._state(net)
        k = self.k0
        if self._prev is not None:
            dv, dq = vm - self._prev[0], q - self._prev[1]
            # The other plant loops move Q too, so the secant is noisy: accept
            # it only within a decade of the nominal slope.
            if abs(dv) > 1e-7 and 0.1 * self.k0 < dq / dv < 10.0 * self.k0:
                k = dq / dv
        target = self._target(vm, q)
        if target is None:
            self.applied = True
            return
        if np.isnan(target):
            # Back towards the set-point, but no further than Q allows.
            dv = self.v_set - vm
            q_pred = q + k * dv
            half = 0.5 * self.band
            if q_pred > self.q_max - half:
                dv = (self.q_max - half - q) / k
            elif q_pred < self.q_min + half:
                dv = (self.q_min + half - q) / k
            dv *= self.damping
        else:
            dv = self.damping * (target - q) / k
        dv = float(np.clip(dv, -self.max_dv, self.max_dv))
        self._prev = (vm, q)
        net.gen.at[self.gen_idx, "vm_pu"] = vm + dv
        self.applied = True


def install_slack_q_limit(net: pp.pandapowerNet, q_limit_mvar: float) -> Optional[int]:
    """Attach one :class:`SlackQLimit` to the (single) reference machine and
    write the same box into its ``min/max_q_mvar``.  Returns the controller
    index, or None if the net has no slack machine."""
    if "slack" not in net.gen.columns:
        return None
    slack = net.gen.index[net.gen["slack"].astype(bool)].tolist()
    if not slack:
        return None
    if len(slack) > 1:
        raise ValueError(f"more than one slack machine: {slack}")
    g = int(slack[0])
    q = abs(float(q_limit_mvar))
    net.gen.at[g, "min_q_mvar"] = -q
    net.gen.at[g, "max_q_mvar"] = q
    return int(SlackQLimit(net, g, -q, q).index)
