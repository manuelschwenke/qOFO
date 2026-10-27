"""
experiments/ch_09_parameter_selection/_droop_probe.py
=====================================================

Per-converter record of the dispatch-anchored droop (DAD) layer at every
plant solve -- the measurement behind Experiments D1 and D2
(``ch:timescales:deadband``).

The runner records totals only.  D1 needs, per converter and solution instant,
the anchor held since the last dispatch and the voltage entering the dead-zone
function; D2 needs the droop increment and proof that each disturbance reached
the plant.  :class:`DroopProbePlant` collects them without touching the runner:
it is the quasi-static plant itself (a subclass of
:class:`core.plant.PandapowerStaticPlant`, passed as ``plant_factory``), and
records after every ``advance``.  For the static plant every regular solve of
the main loop goes through ``advance``:

* ``phase = "profile"`` -- the first solve after the profile step, BEFORE any
  controller writes: the anchors are those of the last dispatch (held);
* ``phase = "end"``     -- the end-of-step solve, run when a controller acted
  or an event fired this step: new anchors at a dispatch, held anchors after
  an event between dispatches.

Not recorded: the runner's fallback re-solves (``pp.runpp`` with a flat start
after a non-converged end-of-step PF, and the post-contingency PF, which the
end-of-step solve of the same step repeats).  ``n_fallback`` is not observable
from here; the runner prints those cases.

The droop law is evaluated exactly as in
:meth:`controller.der_qv_local_loop.QVLocalLoop._compute_target` (same
columns, same capability helper), so ``dq_unsat`` is the incremental command
before the capability clip and ``q_mvar - q_set_mvar`` the delivered one.

Author: Manuel Schwenke / Claude Code (2026-10-27)
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import pandapower as pp

from controller.der_qv_local_loop import _qv_capability
from core.plant import PandapowerStaticPlant

COLUMNS = ("t_s", "solve", "phase", "sgen", "level", "bus", "p_mw", "q_mvar",
           "q_set_mvar", "anchor_pu", "v_pu", "deadband_pu", "slope_pu",
           "sn_mva", "q_min_mvar", "q_max_mvar", "v_eff_pu", "dq_unsat_mvar",
           "dz_active")


class DroopProbePlant(PandapowerStaticPlant):
    """Static plant that records the DAD layer after every solve."""

    def __init__(self, net: pp.pandapowerNet, *, start_time: datetime,
                 der_levels: Dict[int, str], watch_buses: Iterable[int] = (),
                 **kwargs) -> None:
        super().__init__(net, **kwargs)
        self.start_time = start_time
        self.der = [s for s in sorted(der_levels)
                    if str(net.sgen.at[s, "q_mode"]) == "qv"]
        self.level = np.array([der_levels[s] for s in self.der])
        self.bus = np.array([int(net.sgen.at[s, "bus"]) for s in self.der])
        self.sn = np.array([float(net.sgen.at[s, "sn_mva"]) for s in self.der])
        od = (net.sgen["op_diagram"] if "op_diagram" in net.sgen.columns
              else pd.Series("VDE-AR-N-4120-v2", index=net.sgen.index))
        self.op = [str(od.at[s]) if isinstance(od.at[s], str)
                   else "VDE-AR-N-4120-v2" for s in self.der]
        self.watch_buses = [int(b) for b in watch_buses]
        self.t_s: Optional[float] = None
        self.solve = 0
        self._blocks: List[pd.DataFrame] = []
        self._watch: List[dict] = []

    # -- Plant protocol -------------------------------------------------------
    def apply_exogenous(self, profiles, t) -> None:
        super().apply_exogenous(profiles, t)
        self.t_s = (t - self.start_time).total_seconds()
        self.solve = 0

    def advance(self, duration_s: float = 0.0) -> None:
        super().advance(duration_s)
        if self.t_s is not None:
            self._record()
            self.solve += 1

    # -- recording --------------------------------------------------------------
    def _col(self, name: str, default: float) -> np.ndarray:
        sg = self.net.sgen
        if name not in sg.columns:
            return np.full(len(self.der), default)
        return sg.loc[self.der, name].astype(float).to_numpy()

    def _record(self) -> None:
        net = self.net
        p = net.res_sgen.loc[self.der, "p_mw"].to_numpy(dtype=float)
        q = net.res_sgen.loc[self.der, "q_mvar"].to_numpy(dtype=float)
        v = net.res_bus.loc[self.bus, "vm_pu"].to_numpy(dtype=float)
        anchor = self._col("qv_vref_anchor_pu", np.nan)
        vref = self._col("qv_vref_pu", 1.0)
        anchor = np.where(np.isnan(anchor), vref, anchor)   # cold start, as the loop
        q_set = self._col("q_set_mvar", 0.0)
        db = self._col("qv_deadband_pu", 0.0)
        slope = self._col("qv_slope_pu", np.nan)
        r = np.where(slope > 0.0, self.sn / slope, 0.0)
        v_eff = v - anchor
        dq = np.where(v_eff > db, -r * (v_eff - db),
                      np.where(v_eff < -db, -r * (v_eff + db), 0.0))
        cap = np.array([_qv_capability(sn, op, pm)
                        for sn, op, pm in zip(self.sn, self.op, p)])
        self._blocks.append(pd.DataFrame({
            "t_s": self.t_s, "solve": self.solve,
            "phase": "profile" if self.solve == 0 else "end",
            "sgen": self.der, "level": self.level, "bus": self.bus,
            "p_mw": p, "q_mvar": q, "q_set_mvar": q_set, "anchor_pu": anchor,
            "v_pu": v, "deadband_pu": db, "slope_pu": slope, "sn_mva": self.sn,
            "q_min_mvar": cap[:, 0], "q_max_mvar": cap[:, 1],
            "v_eff_pu": v_eff, "dq_unsat_mvar": dq,
            "dz_active": np.abs(v_eff) > db,
        }))
        if self.watch_buses:
            ld = net.load[net.load.in_service & net.load.bus.isin(self.watch_buses)]
            row = {"t_s": self.t_s, "solve": self.solve}
            for b in self.watch_buses:
                row[f"load_q_bus{b}"] = float(ld.loc[ld.bus == b, "q_mvar"].sum())
                row[f"v_bus{b}"] = float(net.res_bus.at[b, "vm_pu"])
            self._watch.append(row)

    def frame(self) -> pd.DataFrame:
        if not self._blocks:
            return pd.DataFrame(columns=list(COLUMNS))
        return pd.concat(self._blocks, ignore_index=True)[list(COLUMNS)]

    def watch_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._watch)


class DroopProbeFactory:
    """``plant_factory`` for :func:`run_multi_tso_dso`; keeps the plant it made.

    Mirrors the runner's default plant construction (distributed slack and the
    plant's Q-limit flag from the config), so the probe changes nothing but
    what is recorded.
    """

    def __init__(self, config, watch_buses: Iterable[int] = ()) -> None:
        self.config = config
        self.watch_buses = tuple(watch_buses)
        self.plant: Optional[DroopProbePlant] = None

    def __call__(self, net, *, meta, zone_map=None) -> DroopProbePlant:
        levels = {int(s): "TS" for s in meta.tso_der_indices}
        levels.update({int(s): "STS" for s in meta.dso_der_indices})
        self.plant = DroopProbePlant(
            net, start_time=self.config.start_time, der_levels=levels,
            watch_buses=self.watch_buses,
            distributed_slack=self.config.distributed_slack,
            enforce_q_lims=self.config.enforce_q_lims_plant)
        return self.plant
