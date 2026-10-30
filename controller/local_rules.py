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
  Holding time and MSC/MSR exclusion per STS are those of
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

Author: Manuel Schwenke / Claude Code (2026-10-30)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

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
class BankRelay:
    """Local voltage rule of one tertiary MSC/MSR (state in ``bank``)."""

    bank: ShuntBank
    primary_bus: int
    v_low_pu: float
    v_high_pu: float

    def __post_init__(self) -> None:
        if not self.v_low_pu < self.v_high_pu:
            raise ValueError("BankRelay needs v_low_pu < v_high_pu")
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


class LocalDiscreteRules:
    """All local tap and bank rules of one run, evaluated once per step."""

    def __init__(self, taps: Sequence[TapRelay], banks: Sequence[BankRelay],
                 mutual_exclusion: bool) -> None:
        self.taps: List[TapRelay] = list(taps)
        self.banks: List[BankRelay] = list(banks)
        self.mutual_exclusion = bool(mutual_exclusion)

    def __bool__(self) -> bool:
        return bool(self.taps or self.banks)

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
        wanted = [(r, r.wanted(float(net.res_bus.at[r.primary_bus, "vm_pu"])))
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
                net.shunt.at[c.shunt_idx, "step"] = int(c.pp_step_new)
                actions.append({"device": "bank", "zone": "local",
                                "shunt_idx": c.shunt_idx, "dso_id": c.dso_id,
                                "kind": c.kind, "direction": c.direction,
                                "new_level": c.new_level})
        return actions
