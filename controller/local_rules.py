"""
Local rules for the discrete transmission devices of the classical variants
=========================================================================

Author decision 2026-10-30: every discrete device that no MIQP dispatches is
under a classical local rule, with the same minimum holding time the MIQP
variants observe.  Two rules:

* :class:`TapRelay` -- a TS 2-winding OLTC (machine transformer, or one of the
  two network OLTCs at IEEE 12) holds the voltage on its HV side within
  ``v_ref ± tol`` (1.03 ± 0.01 pu, as the coupler relays), one tap per action,
  at most one action per holding time (900 s = the TS-OFO lock).
* :class:`BankRelay` -- a tertiary MSC or MSR reads the voltage on the primary
  (EHV) side of its own coupler.  The MSC engages one level below
  ``v_ref - dv`` (1.01 pu) and is released above ``v_ref + dv``; the MSR
  engages above ``v_ref + dv`` (1.05 pu) and is released below ``v_ref - dv``.
  A bank acts only after its condition has held for ``confirm_steps``
  consecutive steps (3 x 20 s, author decision 2026-10-30; applied to the
  release as well).  Holding time and MSC/MSR exclusion per STS are those of
  :class:`controller.shunt_banks.ShuntBank`, i.e. the MIQP's.

The TS tap relay is evaluated explicitly once per step, outside the power
flow, instead of as a pandapower ``DiscreteTapControl``: inside
``runpp(run_control=True)`` that controller iterates many taps within one call,
which made the plant diverge after the gen-2 trip under SVR when the AVR
regulated the LV side of the same transformer (runner note, 2026-09-01).  One
step per action is also what the coupler relays amount to after the runner's
``±1`` per-step clamp.

The rules read plant truth, like the coupler ``DiscreteTapControl`` relays,
and act on ``net`` directly; no controller model is informed.

A third rule was added on 2026-11-10 (author decision, Ch 9 v5 analysis):

* :class:`SetpointTapRelay` -- a machine transformer re-centres the AVR
  set-point of its machine.  It reads the set-point, not a voltage: at or above
  ``v_high_pu`` (1.04) for ``confirm_steps`` consecutive steps, one tap step
  raises the grid side against the terminal, after which the controller above
  (RPR or TS-OFO) can lower the set-point again; at or below ``v_low_pu`` (0.96)
  the opposite step.  Same holding time as :class:`TapRelay`.  It has no voltage
  target of its own, so unlike :class:`TapRelay` on a machine transformer it does
  not undo the set-point changes of the AVR loop.  An optional reserve condition
  blocks a step in the direction in which the machine is already beyond
  ``q_util_max`` of its reactive capability.

Author: Manuel Schwenke / Claude Code (2026-10-30; SetpointTapRelay 2026-11-10)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandapower as pp

from controller.shunt_banks import ShuntBank, opposite_engaged


@dataclass
class TapRelay:
    """Local voltage relay of one TS 2-winding OLTC (``net.trafo``)."""

    trafo_idx: int
    v_ref_pu: float
    tol_pu: float
    hold_s: float
    _last_move_s: float = -math.inf

    def __post_init__(self) -> None:
        if self.tol_pu <= 0.0 or self.hold_s < 0.0:
            raise ValueError("TapRelay needs tol_pu > 0 and hold_s >= 0")

    def direction(self, net: pp.pandapowerNet) -> int:
        """Tap direction that raises the regulated (HV-side) voltage.

        A higher ``tap_pos`` raises the rated voltage of the tap side, so with
        the tap on the HV side it raises the HV voltage against a held LV
        side, and lowers it with the tap on the LV side.
        """
        side = str(net.trafo.at[self.trafo_idx, "tap_side"])
        step = float(net.trafo.at[self.trafo_idx, "tap_step_percent"])
        sign = 1 if side == "hv" else -1
        return sign if step > 0.0 else -sign

    def step(self, net: pp.pandapowerNet, t_s: float) -> int:
        """Move at most one tap; returns the applied tap change (0, ±1)."""
        t = net.trafo
        if not bool(t.at[self.trafo_idx, "in_service"]):
            return 0
        if (t_s - self._last_move_s) < self.hold_s:
            return 0
        v = float(net.res_bus.at[int(t.at[self.trafo_idx, "hv_bus"]), "vm_pu"])
        if not math.isfinite(v):
            return 0
        if v < self.v_ref_pu - self.tol_pu:
            d = self.direction(net)
        elif v > self.v_ref_pu + self.tol_pu:
            d = -self.direction(net)
        else:
            return 0
        pos = int(t.at[self.trafo_idx, "tap_pos"])
        new = pos + d
        if not int(t.at[self.trafo_idx, "tap_min"]) <= new <= int(t.at[self.trafo_idx, "tap_max"]):
            return 0
        t.at[self.trafo_idx, "tap_pos"] = new
        self._last_move_s = t_s
        return d


@dataclass
class SetpointTapRelay:
    """Re-centring rule of one machine transformer on the AVR set-point of its machine.

    ``q_limits(net) -> (Q_min, Q_max)`` [Mvar] is the machine's capability at the
    current operating point (the controllers' capability curve); it is only used
    by the reserve condition and may be None when ``q_util_max`` is None.
    """

    trafo_idx: int
    gen_idx: int
    v_low_pu: float
    v_high_pu: float
    hold_s: float
    confirm_steps: int = 3
    q_util_max: Optional[float] = None
    q_limits: Optional[Callable[[pp.pandapowerNet], Tuple[float, float]]] = None
    _last_move_s: float = -math.inf
    _last_want: int = 0
    _count: int = 0

    def __post_init__(self) -> None:
        if not self.v_low_pu < self.v_high_pu:
            raise ValueError("SetpointTapRelay needs v_low_pu < v_high_pu")
        if self.confirm_steps < 1 or self.hold_s < 0.0:
            raise ValueError("SetpointTapRelay needs confirm_steps >= 1 and hold_s >= 0")
        if self.q_util_max is not None and (self.q_limits is None
                                            or not 0.0 < self.q_util_max <= 1.0):
            raise ValueError("SetpointTapRelay: q_util_max in (0, 1] needs q_limits")

    def direction(self, net: pp.pandapowerNet) -> int:
        """Tap direction that raises the grid (HV) side against a held terminal."""
        return TapRelay.direction(self, net)  # same convention, see TapRelay

    def wanted(self, v_set: float) -> int:
        """+1 raise the grid side (set-point high), -1 lower it, 0 hold."""
        if v_set >= self.v_high_pu:
            return 1
        if v_set <= self.v_low_pu:
            return -1
        return 0

    def _reserve_blocks(self, net: pp.pandapowerNet, want: int) -> bool:
        if self.q_util_max is None:
            return False
        q_lo, q_hi = self.q_limits(net)
        q = float(net.res_gen.at[self.gen_idx, "q_mvar"])
        if not (math.isfinite(q) and math.isfinite(q_lo) and math.isfinite(q_hi)):
            return True
        if want > 0:
            return q >= self.q_util_max * q_hi
        return q <= self.q_util_max * q_lo

    def step(self, net: pp.pandapowerNet, t_s: float) -> int:
        """Move at most one tap; returns the applied tap change (0, +-1)."""
        t = net.trafo
        if not (bool(t.at[self.trafo_idx, "in_service"])
                and bool(net.gen.at[self.gen_idx, "in_service"])):
            self._last_want, self._count = 0, 0
            return 0
        w = self.wanted(float(net.gen.at[self.gen_idx, "vm_pu"]))
        self._count = self._count + 1 if (w != 0 and w == self._last_want) else int(w != 0)
        self._last_want = w
        if w == 0 or self._count < self.confirm_steps:
            return 0
        if (t_s - self._last_move_s) < self.hold_s:
            return 0
        if self._reserve_blocks(net, w):
            return 0
        d = w * self.direction(net)
        new = int(t.at[self.trafo_idx, "tap_pos"]) + d
        if not int(t.at[self.trafo_idx, "tap_min"]) <= new <= int(t.at[self.trafo_idx, "tap_max"]):
            return 0
        t.at[self.trafo_idx, "tap_pos"] = new
        self._last_move_s = t_s
        self._count = 0                      # the next step needs a new confirmation
        return d


@dataclass
class BankRelay:
    """Local voltage rule of one tertiary MSC/MSR (state in ``bank``)."""

    bank: ShuntBank
    primary_bus: int
    v_low_pu: float
    v_high_pu: float
    confirm_steps: int = 3
    _last_want: int = 0
    _count: int = 0

    def __post_init__(self) -> None:
        if not self.v_low_pu < self.v_high_pu:
            raise ValueError("BankRelay needs v_low_pu < v_high_pu")
        if self.confirm_steps < 1:
            raise ValueError("BankRelay needs confirm_steps >= 1")
        if self.bank.config.kind not in ("MSC", "MSR"):
            raise ValueError(f"unknown bank kind {self.bank.config.kind!r}")

    def wanted(self, v: float) -> int:
        """+1 engage, -1 release, 0 hold, from the primary-side voltage."""
        msc = self.bank.config.kind == "MSC"
        if v < self.v_low_pu:
            return 1 if msc else -1
        if v > self.v_high_pu:
            return -1 if msc else 1
        return 0

    def confirmed(self, v: float) -> int:
        """:meth:`wanted`, once it has held for ``confirm_steps`` consecutive
        calls (one call per step); 0 before that."""
        w = self.wanted(v)
        self._count = self._count + 1 if (w != 0 and w == self._last_want) else int(w != 0)
        self._last_want = w
        return w if self._count >= self.confirm_steps else 0


class LocalDiscreteRules:
    """All local tap and bank rules of one run, evaluated once per step."""

    def __init__(self, taps: Sequence[TapRelay], banks: Sequence[BankRelay],
                 mutual_exclusion: bool,
                 setpoint_taps: Sequence[SetpointTapRelay] = ()) -> None:
        self.taps: List[TapRelay] = list(taps)
        self.banks: List[BankRelay] = list(banks)
        self.setpoint_taps: List[SetpointTapRelay] = list(setpoint_taps)
        self.mutual_exclusion = bool(mutual_exclusion)
        idx = [r.trafo_idx for r in self.taps] + [r.trafo_idx for r in self.setpoint_taps]
        dup = sorted({i for i in idx if idx.count(i) > 1})
        if dup:
            raise ValueError(f"trafo(s) {dup} under two tap rules")

    def __bool__(self) -> bool:
        return bool(self.taps or self.banks or self.setpoint_taps)

    def step(self, net: pp.pandapowerNet, t_s: float) -> List[Dict[str, Any]]:
        """Apply every rule on the latest power flow; returns the actions.

        Releases are committed before engagements, so a bank released in this
        step no longer blocks the opposite device of its STS.
        """
        actions: List[Dict[str, Any]] = []
        for r in self.taps:
            d = r.step(net, t_s)
            if d:
                actions.append({"device": "tap", "trafo_idx": r.trafo_idx,
                                "direction": d,
                                "new_pos": int(net.trafo.at[r.trafo_idx, "tap_pos"])})
        for r in self.setpoint_taps:
            d = r.step(net, t_s)
            if d:
                actions.append({"device": "tap", "rule": "setpoint",
                                "trafo_idx": r.trafo_idx, "gen_idx": r.gen_idx,
                                "direction": d,
                                "new_pos": int(net.trafo.at[r.trafo_idx, "tap_pos"]),
                                "v_set_pu": float(net.gen.at[r.gen_idx, "vm_pu"])})
        wanted =[(r, r.confirmed(float(net.res_bus.at[r.primary_bus, "vm_pu"])))
                  for r in self.banks]
        all_banks = [r.bank for r in self.banks]
        for want in (-1, 1):
            for r, w in wanted:
                if w != want:
                    continue
                opp = self.mutual_exclusion and opposite_engaged(r.bank, all_banks)
                if w not in r.bank.available_directions(t_s, opposite_engaged=opp):
                    continue
                c = r.bank.commit_selected(w, t_s, opposite_engaged=opp)
                r._count = 0                  # the next level needs a new confirmation
                net.shunt.at[c.shunt_idx, "step"] = int(c.pp_step_new)
                actions.append({"device": "bank", "zone": "local",
                                "shunt_idx": c.shunt_idx, "dso_id": c.dso_id,
                                "kind": c.kind, "direction": c.direction,
                                "new_level": c.new_level})
        return actions
