"""
visualisation/live_sts.py
=========================
Live window 2 — SUB-TRANSMISSION SYSTEMS (Layer 2, one MIQP per DSO).

Metered view: the noisy pre-control samples the DSO controllers received,
sampled on DSO ticks.  One row per DSO, small multiples on a shared time
axis:

    |   CONTROLLED OUTPUTS            |   ACTUATORS                     |
    | interface Q    | HV voltages    | DER Q (sum)    | coupler taps   |
    | (set / metered |  mean, min-max |                |  per coupler   |
    |  / capability) |  + band limits |                |                |
    +---------------------------------------------------------------- +
    | interface-Q tracking error, RMS over the couplers of each DSO     |

Coupler colours repeat per DSO row; the DSO colour is used in the
tracking-error strip.  The edges of the reported capability interval are
faint dotted lines; the y-range follows setpoint and metered value, so the
edges appear only when the interval is close to binding.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from visualisation.live_base import (
    COLOUR_ACT_BAND, COLOUR_MEAS_BAND, LIMIT_RED, REF_GREY, LiveWindow, Trace,
    _c, empty_note, fill_section_band, legend, proxy, set_band, set_ylim_from,
    tile_title, value_tag,
)
from visualisation.style import TITLE_BAR_HEIGHT_FRAC

if TYPE_CHECKING:
    from experiments.helpers import ContingencyEvent, MultiTSOIterationRecord


def _short(trafo_key: str) -> str:
    """``"DSO_1|trafo_37"`` -> ``"T37"``."""
    return "T" + trafo_key.rsplit("|trafo_", 1)[1] if "|trafo_" in trafo_key else trafo_key


class STSLiveWindow(LiveWindow):
    """Layer-2 window: interface tracking, voltages and actuators per DSO."""

    _COLS = ("iface", "v", "der", "tap")

    def __init__(
        self,
        dso_ids: Sequence[str],
        *,
        dso_zone: Dict[str, int],
        v_setpoint_pu: float = 1.03,
        v_min_pu: float = 0.90,
        v_max_pu: float = 1.10,
        events: Iterable["ContingencyEvent"] = (),
        slot_idx: int = 1,
        layout: str = "dual_screen",
        use_tex: bool = False,
        refresh_s: float = 2.0,
    ) -> None:
        super().__init__(
            "STS - Layer 2 (DSO MIQPs) - metered view", figsize=(8.5, 10.0),
            slot_idx=slot_idx, layout=layout, use_tex=use_tex,
            refresh_s=refresh_s, events=events,
        )
        self._d = list(dso_ids)
        self._dc = {d: _c(5 + i) for i, d in enumerate(self._d)}
        self._dso_zone = dict(dso_zone)
        self._v_set = v_setpoint_pu
        self._v_lim = (v_min_pu, v_max_pu)
        # Coupler keys are discovered from the first metered sample.
        self._keys: Dict[str, List[str]] = {}
        self._q: Dict[str, Trace] = {}      # per DSO: [set | meas | cap_min | cap_max] x n
        self._tap: Dict[str, Trace] = {}
        self._v = {d: Trace(3) for d in self._d}
        self._der = {d: Trace(1) for d in self._d}
        self._err = Trace(len(self._d))

        n = len(self._d)
        gs = GridSpec(
            n + 2, 4, figure=self.fig,
            height_ratios=[0.028 * (n + 0.85)] + [1.0] * n + [0.85],
            hspace=0.40, wspace=0.34, left=0.085, right=0.985, bottom=0.05,
            top=1.0 - TITLE_BAR_HEIGHT_FRAC - 0.008,
        )
        fill_section_band(self.fig.add_subplot(gs[0, 0:2]), "Controlled outputs", COLOUR_MEAS_BAND)
        fill_section_band(self.fig.add_subplot(gs[0, 2:4]), "Actuators", COLOUR_ACT_BAND)
        self.ax: Dict[str, Dict[str, plt.Axes]] = {
            d: {name: self.fig.add_subplot(gs[1 + r, c]) for c, name in enumerate(self._COLS)}
            for r, d in enumerate(self._d)
        }
        self.ax_err = self.fig.add_subplot(gs[n + 1, :])
        axes = [ax for row in self.ax.values() for ax in row.values()] + [self.ax_err]
        self._time_axes(axes, bottom=[self.ax_err])
        self._label_events([self.ax[self._d[0]]["iface"]])
        self._build_static()
        self._show()

    # ── artists ─────────────────────────────────────────────────────────────

    def _build_static(self) -> None:
        titles = {"iface": "Interface Q", "v": "HV voltages", "der": "DER Q (sum)",
                  "tap": "Coupler taps"}
        nan = np.array([np.nan])
        self._a_v, self._a_der, self._notes, self._arts = {}, {}, {}, {}
        for r, d in enumerate(self._d):
            row = self.ax[d]
            z = self._dso_zone.get(d)
            if r == 0:
                for name, ax in row.items():
                    tile_title(ax, titles[name])
            row["iface"].set_ylabel(f"{d}" + (f" (Z{z})" if z is not None else "") + "\nQ / Mvar",
                                    fontweight="bold")
            row["v"].set_ylabel("V / p.u.")
            row["der"].set_ylabel("Q / Mvar")
            row["tap"].set_ylabel("tap")
            ax = row["v"]
            ax.axhline(self._v_set, color=REF_GREY, ls=":", lw=0.8)
            for lim in self._v_lim:
                ax.axhline(lim, color=LIMIT_RED, ls="--", lw=0.6, alpha=0.7)
            self._a_v[d] = (ax.fill_between(nan, nan, nan, color=self._dc[d], alpha=0.18, lw=0),
                            ax.plot([], [], color=self._dc[d], lw=1.2)[0])
            row["der"].axhline(0.0, color=REF_GREY, lw=0.6)
            self._a_der[d] = row["der"].plot([], [], color=self._dc[d], lw=1.2)[0]
            self._notes[d] = empty_note(row["iface"], "waiting for interface metering")

        tile_title(self.ax_err, "Interface-Q tracking error |Q - Q_set| (RMS over couplers)")
        self.ax_err.set_ylabel("Mvar")
        self._a_err = [self.ax_err.plot([], [], color=self._dc[d], lw=1.2, label=d)[0]
                       for d in self._d]
        legend(self.ax_err, self._a_err, ncol=len(self._d))
        self._tag_err = value_tag(self.ax_err)

    def _discover(self, rec: "MultiTSOIterationRecord") -> None:
        keys = set(rec.dso_trafo_q_meas_mvar) | set(rec.dso_trafo_tap_pos)
        for d in self._d:
            if d in self._keys:
                continue
            mine = sorted((k for k in keys if k.split("|", 1)[0] == d),
                          key=lambda k: int(k.rsplit("_", 1)[1]) if k.rsplit("_", 1)[1].isdigit() else 0)
            if not mine:
                continue
            self._keys[d] = mine
            n = len(mine)
            self._q[d] = Trace(4 * n)
            self._tap[d] = Trace(n)
            self._notes[d].set_visible(False)
            ax_q, ax_t = self.ax[d]["iface"], self.ax[d]["tap"]
            arts = []
            for k, key in enumerate(mine):
                c = _c(3 + k)
                # Capability edges as faint dotted lines: three overlapping
                # filled bands covered the whole tile and hid the traces.
                (cap_lo,) = ax_q.plot([], [], color=c, lw=0.7, ls=":", alpha=0.7,
                                      drawstyle="steps-post")
                (cap_hi,) = ax_q.plot([], [], color=c, lw=0.7, ls=":", alpha=0.7,
                                      drawstyle="steps-post")
                (meas,) = ax_q.plot([], [], color=c, lw=1.1, label=_short(key))
                (sp,) = ax_q.plot([], [], color=c, lw=0.9, ls="--", drawstyle="steps-post")
                (tap,) = ax_t.plot([], [], color=c, lw=1.1, drawstyle="steps-post")
                arts.append((cap_lo, cap_hi, meas, sp, tap))
            self._arts[d] = arts
            legend(ax_q, [a[2] for a in arts] + [
                proxy("0.3", ls="--", lw=0.9, label="set"),
                proxy("0.3", ls=":", lw=0.8, label="capability")], ncol=n + 2, fontsize=6)

    # ── ingest ──────────────────────────────────────────────────────────────

    def _ingest(self, rec: "MultiTSOIterationRecord", t: float) -> None:
        if not (rec.dso_trafo_q_meas_mvar or rec.dso_group_v_meas_mean_pu):
            return  # no metered STS sample at this step
        self._discover(rec)
        nan = float("nan")
        errs = []
        for d in self._d:
            g = rec.dso_controller_group.get(d, d)
            self._v[d].append(t, (rec.dso_group_v_meas_min_pu.get(g, nan),
                                  rec.dso_group_v_meas_mean_pu.get(g, nan),
                                  rec.dso_group_v_meas_max_pu.get(g, nan)))
            self._der[d].append(t, (rec.dso_group_q_der_meas_mvar.get(g, nan),))
            keys = self._keys.get(d)
            if not keys:
                errs.append(nan)
                continue
            q_set = np.array([rec.dso_trafo_q_set_mvar.get(k, nan) for k in keys])
            q_meas = np.array([rec.dso_trafo_q_meas_mvar.get(k, nan) for k in keys])
            self._q[d].append(t, np.concatenate([
                q_set, q_meas,
                [rec.dso_trafo_q_cap_min_mvar.get(k, nan) for k in keys],
                [rec.dso_trafo_q_cap_max_mvar.get(k, nan) for k in keys],
            ]))
            self._tap[d].append(t, [rec.dso_trafo_tap_pos.get(k, nan) for k in keys])
            e = q_meas - q_set
            e = e[np.isfinite(e)]
            errs.append(float(np.sqrt(np.mean(e ** 2))) if e.size else nan)
        self._err.append(t, errs)

    # ── render ──────────────────────────────────────────────────────────────

    def _render(self) -> None:
        for d in self._d:
            band, line = self._a_v[d]
            tr = self._v[d]
            set_band(band, tr.t, tr.col(0), tr.col(2))
            line.set_data(tr.t, tr.col(1))
            set_ylim_from(self.ax[d]["v"], tr.y, include=[self._v_set], min_span=0.01)
            tr = self._der[d]
            self._a_der[d].set_data(tr.t, tr.col(0))
            set_ylim_from(self.ax[d]["der"], tr.y, include=[0.0], min_span=5.0)
            keys = self._keys.get(d)
            if not keys:
                continue
            n = len(keys)
            q, tp = self._q[d], self._tap[d]
            for k, (cap_lo, cap_hi, meas, sp, tap) in enumerate(self._arts[d]):
                cap_lo.set_data(q.t, q.col(2 * n + k))
                cap_hi.set_data(q.t, q.col(3 * n + k))
                meas.set_data(q.t, q.col(n + k))
                sp.set_data(q.t, q.col(k))
                tap.set_data(tp.t, tp.col(k))
            set_ylim_from(self.ax[d]["iface"], q.y[:, :2 * n], min_span=5.0)
            set_ylim_from(self.ax[d]["tap"], tp.y, min_span=2.0, pad=0.15)
        for k, line in enumerate(self._a_err):
            line.set_data(self._err.t, self._err.col(k))
        set_ylim_from(self.ax_err, self._err.y, include=[0.0], min_span=1.0)
        self._tag_err.set_text("now  " + "   ".join(
            f"{d} {v:.1f} Mvar" for d, v in zip(self._d, self._err.last)))
