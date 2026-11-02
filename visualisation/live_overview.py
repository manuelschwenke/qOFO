"""
visualisation/live_overview.py
==============================
Live window 3 — SYSTEM OVERVIEW (plant view, every step).

    +----------------------------------------------------------------+
    | active-power infeed by generation type, stacked in merit order |
    +--------------------+--------------------+----------------------+
    | Q injection into   |   ... per zone     |                      |
    | the TS by source   |                    |                      |
    +--------------------+--------------------+----------------------+
    | Q reserve per zone, up (+) / down (-), stacked by source;      |
    | the STS share is the capability interval the DSOs reported     |
    +--------------------+--------------------+----------------------+
    | Φ_TS per zone      | Φ_STS per DSO      | Φ_TS by term (share) |
    | and |J*| (dotted)  | and |J*| (dotted)  | Φ_STS by term        |
    +--------------------+--------------------+----------------------+

The G1 network equivalent is a separate reactive source: stacked outermost
in the reserve tiles and excluded from their y-range (value in the header).

Φ is the plant-level objective each controller descends, evaluated at the
step's linearisation point (``*_cost_terms``); J* is the optimum of the
step QP, i.e. the predicted change of the local model, which tends to 0 at
a fixed point.  A decreasing Φ is actual improvement; a small |J*| is
stationarity.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Sequence

import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from visualisation.live_base import (
    REF_GREY, LiveWindow, Trace, _c, empty_note, legend, patch, proxy,
    set_band, set_ylim_from, tile_title, value_tag,
)
from visualisation.style import TITLE_BAR_HEIGHT_FRAC, slot_geometry

if TYPE_CHECKING:
    from experiments.helpers import ContingencyEvent, MultiTSOIterationRecord

#: Generation types: (colour, alpha, hatch).  STS-connected units share the
#: hue of their TS counterpart and are lighter.
GEN_STYLE = {
    "Wind STS": ("#008877", 0.40, None),
    "PV STS": ("#D7AC00", 0.45, None),
    "Wind TS": ("#008877", 0.85, None),
    "PV TS": ("#D7AC00", 0.90, None),
    "Hydro": ("#00689D", 0.85, None),
    "Nuclear": ("#611C73", 0.80, None),
    "Coal": ("#4A4A4A", 0.80, None),
    "Other": ("#9A9A9A", 0.70, None),
    "Equivalent": ("#B9B9B9", 0.70, "///"),
}
#: Reactive sources: colour; the STS reserve is hatched (reported, not owned).
#: "Equivalent" is the G1 network anchor (10 GVA): stacked outermost in the
#: reserve view and left out of its y-range, so it cannot hide the zone.
Q_SOURCES = ("SG", "TS-DER", "MSC/MSR", "STS", "Ties", "Equivalent")
Q_COLOUR = {"SG": "#004E8A", "TS-DER": "#008877", "MSC/MSR": "#B90F22",
            "STS": "#951169", "Ties": "#D28700", "Equivalent": "#8C8C8C"}
Q_STYLE = {"Ties": "--", "Equivalent": "-."}
RESERVE_SOURCES = ("SG", "TS-DER", "MSC/MSR", "STS", "Equivalent")
RESERVE_HATCH = {"STS": "///", "Equivalent": "xx"}
TS_TERMS = ("v", "q_pcc", "loss", "res_sg", "res_der", "u_reg")
STS_TERMS = ("q_iface", "v", "u_reg")
TERM_COLOUR = {"v": "#004E8A", "q_pcc": "#951169", "loss": "#4A4A4A",
               "res_sg": "#CC4C03", "res_der": "#008877", "u_reg": "#B1BD00",
               "q_iface": "#951169"}
TERM_LABEL = {"v": "V tracking", "q_pcc": "Q_PCC tracking", "loss": "losses",
              "res_sg": "SG reserve", "res_der": "DER reserve", "u_reg": "usage",
              "q_iface": "interface-Q tracking"}


class OverviewLiveWindow(LiveWindow):
    """System view: generation mix, reactive balance and reserves, objectives."""

    def __init__(
        self,
        zone_ids: Sequence[int],
        dso_ids: Sequence[str],
        *,
        events: Iterable["ContingencyEvent"] = (),
        slot_idx: int = 2,
        layout: str = "dual_screen",
        use_tex: bool = False,
        refresh_s: float = 2.0,
    ) -> None:
        # Full second screen -> landscape grid; half screen or a third ->
        # portrait grid with one row per zone.  Without a Qt application
        # (headless) the layout name decides.
        geom = slot_geometry(slot_idx, layout=layout)
        self._portrait = (
            geom[3] > 0.8 * geom[2] if geom is not None
            else layout in ("thirds", "dual_screen_half")
        )
        super().__init__(
            "System overview - plant view",
            figsize=(8.5, 10.0) if self._portrait else (15.0, 9.5),
            slot_idx=slot_idx, layout=layout, use_tex=use_tex,
            refresh_s=refresh_s, events=events,
        )
        self._z = list(zone_ids)
        self._zc = {z: _c(i + 1) for i, z in enumerate(self._z)}
        self._d = list(dso_ids)
        self._dc = {d: _c(5 + i) for i, d in enumerate(self._d)}
        self._mix_keys: List[str] = []
        self._mix: Trace = None          # created when the types are known
        self._load = Trace(1)
        self._qsrc = {z: Trace(len(Q_SOURCES)) for z in self._z}
        self._qres = {z: Trace(2 * len(RESERVE_SOURCES)) for z in self._z}
        self._phi_ts = {z: Trace(2) for z in self._z}               # Φ, |J*|
        self._phi_sts = {d: Trace(2) for d in self._d}
        self._terms_ts = Trace(len(TS_TERMS))
        self._terms_sts = Trace(len(STS_TERMS))

        if self._portrait:
            self._grid_portrait()
        else:
            self._grid_landscape()
        self._label_events([self.ax_mix])
        self._build_static()
        self._show()

    def _grid_landscape(self) -> None:
        """Rows: infeed | Q sources per zone | Q reserve per zone | objectives."""
        nz = len(self._z)
        add = self.fig.add_subplot
        outer = GridSpec(
            4, 1, figure=self.fig, height_ratios=[1.15, 1.0, 1.1, 1.0], hspace=0.42,
            left=0.05, right=0.985, bottom=0.06, top=1.0 - TITLE_BAR_HEIGHT_FRAC - 0.03,
        )
        self.ax_mix = add(outer[0])
        g_src = GridSpecFromSubplotSpec(1, nz, subplot_spec=outer[1], wspace=0.18)
        g_res = GridSpecFromSubplotSpec(1, nz, subplot_spec=outer[2], wspace=0.18)
        g_obj = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3], wspace=0.22)
        g_cmp = GridSpecFromSubplotSpec(2, 1, subplot_spec=g_obj[0, 2], hspace=0.55)
        self.ax_src = {z: add(g_src[0, i]) for i, z in enumerate(self._z)}
        self.ax_res = {z: add(g_res[0, i]) for i, z in enumerate(self._z)}
        self.ax_phi_ts, self.ax_phi_sts = add(g_obj[0, 0]), add(g_obj[0, 1])
        self.ax_cmp_ts, self.ax_cmp_sts = add(g_cmp[0, 0]), add(g_cmp[1, 0])
        axes = self._all_axes()
        self._time_axes(axes, bottom=[self.ax_phi_ts, self.ax_phi_sts, self.ax_cmp_sts])
        for ax in axes:  # each row keeps its own x labels in this grid
            if ax is not self.ax_cmp_ts:
                ax.tick_params(labelbottom=True)

    def _grid_portrait(self) -> None:
        """Rows: infeed | one row per zone (Q sources, Q reserve) | objectives
        | term shares."""
        nz = len(self._z)
        add = self.fig.add_subplot
        outer = GridSpec(
            nz + 3, 2, figure=self.fig, height_ratios=[1.3] + [1.0] * nz + [1.1, 0.7],
            hspace=0.55, wspace=0.26, left=0.09, right=0.985, bottom=0.05,
            top=1.0 - TITLE_BAR_HEIGHT_FRAC - 0.03,
        )
        self.ax_mix = add(outer[0, :])
        self.ax_src = {z: add(outer[1 + i, 0]) for i, z in enumerate(self._z)}
        self.ax_res = {z: add(outer[1 + i, 1]) for i, z in enumerate(self._z)}
        self.ax_phi_ts, self.ax_phi_sts = add(outer[nz + 1, 0]), add(outer[nz + 1, 1])
        self.ax_cmp_ts, self.ax_cmp_sts = add(outer[nz + 2, 0]), add(outer[nz + 2, 1])
        self._time_axes(self._all_axes(), bottom=[self.ax_cmp_ts, self.ax_cmp_sts])

    def _all_axes(self) -> List:
        return ([self.ax_mix] + list(self.ax_src.values()) + list(self.ax_res.values())
                + [self.ax_phi_ts, self.ax_phi_sts, self.ax_cmp_ts, self.ax_cmp_sts])

    # ── artists ─────────────────────────────────────────────────────────────

    def _build_static(self) -> None:
        nan = np.array([np.nan])
        tile_title(self.ax_mix, "Active-power infeed by type (merit order, bottom to top)")
        self.ax_mix.set_ylabel("P / MW")
        self._mix_note = empty_note(self.ax_mix, "waiting for first sample")
        (self._a_load,) = self.ax_mix.plot([], [], color="black", lw=1.3, label="load")
        self._tag_mix = value_tag(self.ax_mix)

        self._a_src, self._a_res = {}, {}
        for z in self._z:
            ax = self.ax_src[z]
            tile_title(ax, f"Z{z}: Q injection into TS by source")
            ax.axhline(0.0, color=REF_GREY, lw=0.6)
            self._a_src[z] = [ax.plot([], [], color=Q_COLOUR[s], lw=1.2,
                                      ls=Q_STYLE.get(s, "-"), label=s)[0]
                              for s in Q_SOURCES]
            ax = self.ax_res[z]
            tile_title(ax, f"Z{z}: Q reserve +up / -down")
            ax.axhline(0.0, color="0.2", lw=0.7)
            fills = [
                [ax.fill_between(nan, nan, nan, color=Q_COLOUR[s], lw=0,
                                 alpha=0.35 if s in RESERVE_HATCH else 0.6,
                                 hatch=RESERVE_HATCH.get(s))
                 for s in RESERVE_SOURCES]
                for _ in ("up", "dn")
            ]
            self._a_res[z] = (fills[0], fills[1], value_tag(ax))
        first = self._z[0]
        for z in (self._z if self._portrait else [first]):
            self.ax_src[z].set_ylabel("Q / Mvar")
            self.ax_res[z].set_ylabel("Q / Mvar")
        legend(self.ax_src[first], self._a_src[first], ncol=len(Q_SOURCES))
        legend(self.ax_res[first], [patch(Q_COLOUR[s], s, hatch=RESERVE_HATCH.get(s))
                                    for s in RESERVE_SOURCES], ncol=len(RESERVE_SOURCES))

        ax = self.ax_phi_ts
        tile_title(ax, "Phi_TS per zone (|J*| dotted)")
        ax.set_yscale("log")
        self._a_phi_ts = {z: (ax.plot([], [], color=self._zc[z], lw=1.3, marker="o", ms=2)[0],
                              ax.plot([], [], color=self._zc[z], lw=0.9, ls=":")[0])
                          for z in self._z}
        legend(ax, [proxy(self._zc[z], label=f"Z{z}") for z in self._z])
        self._tag_phi_ts = value_tag(ax)

        ax = self.ax_phi_sts
        tile_title(ax, "Phi_STS per DSO (|J*| dotted)")
        ax.set_yscale("log")
        self._a_phi_sts = {d: (ax.plot([], [], color=self._dc[d], lw=1.2)[0],
                               ax.plot([], [], color=self._dc[d], lw=0.8, ls=":")[0])
                           for d in self._d}
        legend(ax, [proxy(self._dc[d], label=d.replace("DSO_", "D")) for d in self._d])
        if not self._d:
            empty_note(ax, "no STS controllers")
        self._tag_phi_sts = value_tag(ax)

        # name -> (axes, terms, fills); the legend lists the active terms.
        self._a_cmp, self._cmp_active = {}, {}
        for ax, terms, name in ((self.ax_cmp_ts, TS_TERMS, "Phi_TS"),
                                (self.ax_cmp_sts, STS_TERMS, "Phi_STS")):
            tile_title(ax, f"{name} by term (share)")
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0.0, 0.5, 1.0])
            self._a_cmp[name] = (ax, terms, [
                ax.fill_between(nan, nan, nan, color=TERM_COLOUR[k], alpha=0.7, lw=0)
                for k in terms])

    def _init_mix(self, keys: List[str]) -> None:
        self._mix_keys = keys
        self._mix = Trace(len(keys))
        nan = np.array([np.nan])
        self._a_mix, handles = [], []
        for k in keys:
            colour, alpha, hatch = GEN_STYLE.get(k, GEN_STYLE["Other"])
            self._a_mix.append((
                self.ax_mix.fill_between(nan, nan, nan, color=colour, alpha=alpha, lw=0, hatch=hatch),
                self.ax_mix.fill_between(nan, nan, nan, color=colour, alpha=alpha, lw=0, hatch=hatch),
            ))
            handles.append(patch(colour, k, alpha=alpha, hatch=hatch))
        legend(self.ax_mix, handles[::-1] + [self._a_load], loc="upper left",
               ncol=len(handles) + 1)
        self._mix_note.set_visible(False)

    # ── ingest ──────────────────────────────────────────────────────────────

    def _ingest(self, rec: "MultiTSOIterationRecord", t: float) -> None:
        nan = float("nan")
        if rec.gen_mix_p_mw:
            if self._mix is None:
                self._init_mix(list(rec.gen_mix_p_mw))
            self._mix.append(t, [rec.gen_mix_p_mw.get(k, 0.0) for k in self._mix_keys])
            self._load.append(t, (rec.total_load_res_p_mw,))
        for z in self._z:
            src = rec.zone_q_source_mvar.get(z)
            if src is not None:
                self._qsrc[z].append(t, [src.get(s, nan) for s in Q_SOURCES])
            up, dn = rec.zone_q_reserve_up_mvar.get(z), rec.zone_q_reserve_dn_mvar.get(z)
            if up is not None and dn is not None:
                self._qres[z].append(t, [up.get(s, nan) for s in RESERVE_SOURCES]
                                     + [dn.get(s, nan) for s in RESERVE_SOURCES])
        if rec.zone_tso_cost_terms:
            total = np.zeros(len(TS_TERMS))
            for z in self._z:
                terms = rec.zone_tso_cost_terms.get(z)
                if terms is None:
                    continue
                j = rec.zone_tso_objective.get(z)
                self._phi_ts[z].append(t, (sum(terms.values()),
                                           abs(j) if j is not None else nan))
                total += [terms.get(k, 0.0) for k in TS_TERMS]
            self._terms_ts.append(t, total)
        if rec.dso_cost_terms:
            total = np.zeros(len(STS_TERMS))
            for d in self._d:
                terms = rec.dso_cost_terms.get(d)
                if terms is None:
                    continue
                j = rec.dso_objective.get(d)
                self._phi_sts[d].append(t, (sum(terms.values()),
                                            abs(j) if j is not None else nan))
                total += [terms.get(k, 0.0) for k in STS_TERMS]
            self._terms_sts.append(t, total)

    # ── render ──────────────────────────────────────────────────────────────

    def _render(self) -> None:
        if self._mix is not None and self._mix.n:
            self._render_stack(self._mix, self._a_mix)
            self._a_load.set_data(self._load.t, self._load.col(0))
            pos = np.clip(self._mix.y, 0.0, None).sum(axis=1)
            neg = np.clip(self._mix.y, None, 0.0).sum(axis=1)
            set_ylim_from(self.ax_mix, pos, neg, self._load.y, include=[0.0], pad=0.04)
            last = dict(zip(self._mix_keys, self._mix.last))
            res = sum(v for k, v in last.items() if "Wind" in k or "PV" in k)
            gen = sum(max(v, 0.0) for v in last.values())
            self._tag_mix.set_text(
                f"load {self._load.last[0]:.0f} MW   generation {gen:.0f} MW   "
                f"RES share {100 * res / gen:.0f} %" if gen > 0 else "")

        n_eq = RESERVE_SOURCES.index("Equivalent")
        i_sts = RESERVE_SOURCES.index("STS")
        for z in self._z:
            tr = self._qsrc[z]
            for k, line in enumerate(self._a_src[z]):
                line.set_data(tr.t, tr.col(k))
            set_ylim_from(self.ax_src[z], tr.y, include=[0.0], min_span=20.0)

            tr = self._qres[z]
            up_arts, dn_arts, tag = self._a_res[z]
            if not tr.n:
                continue
            ns = len(RESERVE_SOURCES)
            up = np.nan_to_num(tr.y[:, :ns])
            dn = np.nan_to_num(tr.y[:, ns:])
            cu, cd = np.zeros(tr.n), np.zeros(tr.n)
            for k in range(ns):
                if k == n_eq:  # y-range set by the zone's own resources + STS
                    set_ylim_from(self.ax_res[z], cu, -cd, include=[0.0],
                                  min_span=20.0, pad=0.12)
                set_band(up_arts[k], tr.t, cu, cu + up[:, k])
                set_band(dn_arts[k], tr.t, -cd - dn[:, k], -cd)
                cu, cd = cu + up[:, k], cd + dn[:, k]
            text = f"+{cu[-1] - up[-1, n_eq]:.0f} / -{cd[-1] - dn[-1, n_eq]:.0f} Mvar"
            if up[-1, i_sts] or dn[-1, i_sts]:
                text += f"  (STS +{up[-1, i_sts]:.0f} / -{dn[-1, i_sts]:.0f})"
            if up[-1, n_eq] or dn[-1, n_eq]:
                text += f"  Eq. +{up[-1, n_eq]:.0f} / -{dn[-1, n_eq]:.0f}"
            tag.set_text(text)

        for traces, arts, ax, tag, labels in (
            (self._phi_ts, self._a_phi_ts, self.ax_phi_ts, self._tag_phi_ts,
             {z: f"Z{z}" for z in self._z}),
            (self._phi_sts, self._a_phi_sts, self.ax_phi_sts, self._tag_phi_sts,
             {d: d.replace("DSO_", "D") for d in self._d}),
        ):
            arrs = []
            for key, (phi, jstar) in arts.items():
                tr = traces[key]
                phi.set_data(tr.t, tr.col(0))
                jstar.set_data(tr.t, tr.col(1))
                arrs.append(tr.y)
            _log_ylim(ax, *arrs)
            tag.set_text("  ".join(f"{labels[k]} {traces[k].last[0]:.3g}"
                                   for k in arts if traces[k].n))

        for name, tr in (("Phi_TS", self._terms_ts), ("Phi_STS", self._terms_sts)):
            if not tr.n:
                continue
            tot = tr.y.sum(axis=1)
            share = np.divide(tr.y, tot[:, None], out=np.zeros_like(tr.y),
                              where=tot[:, None] > 0)
            ax, terms, fills = self._a_cmp[name]
            cum = np.zeros(tr.n)
            for k, coll in enumerate(fills):
                set_band(coll, tr.t, cum, cum + share[:, k])
                cum = cum + share[:, k]
            active = [k for i, k in enumerate(terms) if np.any(share[:, i] > 0)]
            if active != self._cmp_active.get(name):
                self._cmp_active[name] = active
                legend(ax, [patch(TERM_COLOUR[k], TERM_LABEL[k], alpha=0.7) for k in active],
                       ncol=len(active), fontsize=6, loc="lower left")

    def _render_stack(self, tr: Trace, arts) -> None:
        """Positive parts stacked upward, negative parts downward."""
        pos = np.clip(np.nan_to_num(tr.y), 0.0, None)
        neg = np.clip(np.nan_to_num(tr.y), None, 0.0)
        cp, cn = np.zeros(tr.n), np.zeros(tr.n)
        for k, (up, dn) in enumerate(arts):
            set_band(up, tr.t, cp, cp + pos[:, k])
            set_band(dn, tr.t, cn + neg[:, k], cn)
            cp, cn = cp + pos[:, k], cn + neg[:, k]


def _log_ylim(ax, *arrays: np.ndarray) -> None:
    vals = [a[np.isfinite(a) & (a > 0)] for a in arrays if a.size]
    vals = [v for v in vals if v.size]
    if not vals:
        return
    lo = min(float(v.min()) for v in vals)
    hi = max(float(v.max()) for v in vals)
    ax.set_ylim(lo / 2.0, hi * 2.0 if hi > lo else lo * 10.0)
