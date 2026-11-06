"""
visualisation/live_ts.py
========================
Live window 1 — TRANSMISSION SYSTEM (Layer 1, one MIQP per zone).

Metered view: every analogue quantity is the noisy pre-control sample the
zone controllers received (``*_meas`` record fields), sampled on TSO ticks.
Setpoints and discrete states are the commanded / read-back values.

    CONTROLLED OUTPUTS (left)          ACTUATORS (right)
      voltages per zone                  AVR setpoints
      inter-zone tie-line Q              TS-DER Q
      synchronous-machine Q              Q_PCC: setpoint to STS vs metered
      normalised Q reserve               machine-transformer taps (lanes)
      line loading        [optional]     MSC / MSR levels (lanes)

Tiles without content in the network at hand are not created.  Colours
encode the zone throughout; raster lanes use a diverging map centred on 0
(taps: neutral position; banks: capacitive < 0 < inductive).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from visualisation.live_base import (
    COLOUR_ACT_BAND, COLOUR_MEAS_BAND, LIMIT_RED, REF_GREY, LiveWindow, Trace,
    _c, fill_section_band, legend, proxy, set_band, set_ylim_from,
    tile_title, value_tag,
)
from visualisation.style import TITLE_BAR_HEIGHT_FRAC

if TYPE_CHECKING:
    from experiments.helpers import ContingencyEvent, MultiTSOIterationRecord

#: One raster lane: (zone, label).
Lane = Tuple[int, str]


class TSLiveWindow(LiveWindow):
    """Layer-1 window: controlled outputs and actuators of every TS zone."""

    def __init__(
        self,
        zone_ids: Sequence[int],
        *,
        tie_line_pairs: Sequence[Tuple[int, int]],
        n_gen_per_zone: Dict[int, int],
        n_der_per_zone: Dict[int, int],
        oltc_lanes: Sequence[Lane],
        shunt_lanes: Sequence[Tuple[int, int, str, str]],
        dso_zone: Dict[str, int],
        v_setpoint_pu: Dict[int, float],
        v_min_pu: float = 0.90,
        v_max_pu: float = 1.10,
        show_line_currents: bool = False,
        show_reserves: bool = True,
        show_tie_flows: bool = True,
        events: Iterable["ContingencyEvent"] = (),
        slot_idx: int = 0,
        layout: str = "dual_screen",
        use_tex: bool = False,
        refresh_s: float = 2.0,
    ) -> None:
        super().__init__(
            "TS - Layer 1 (zone MIQPs) - metered view", figsize=(8.5, 10.0),
            slot_idx=slot_idx, layout=layout, use_tex=use_tex,
            refresh_s=refresh_s, events=events,
        )
        self._z = list(zone_ids)
        self._zc = {z: _c(i + 1) for i, z in enumerate(self._z)}
        self._pairs = list(tie_line_pairs) if show_tie_flows else []
        self._n_gen = {z: int(n_gen_per_zone.get(z, 0)) for z in self._z}
        self._n_der = {z: int(n_der_per_zone.get(z, 0)) for z in self._z}
        self._oltc_lanes = list(oltc_lanes)
        # (shunt_idx, zone, kind, label); MSC levels are drawn negative.
        self._shunt_lanes = list(shunt_lanes)
        self._dso_zone = dict(dso_zone)
        self._pcc_zones = sorted({z for z in self._dso_zone.values() if z in self._zc})
        self._v_set = dict(v_setpoint_pu)
        self._v_lim = (v_min_pu, v_max_pu)

        # ── buffers ─────────────────────────────────────────────────────────
        self._v = {z: Trace(3) for z in self._z}                       # min/mean/max
        self._tie = Trace(len(self._pairs)) if self._pairs else None
        self._qgen = {z: Trace(self._n_gen[z]) for z in self._z if self._n_gen[z]}
        self._res = {z: Trace(3) for z in self._z} if show_reserves else {}  # SG mean/min, DER mean
        self._iline = {z: Trace(3) for z in self._z} if show_line_currents else {}
        self._avr = {z: Trace(self._n_gen[z]) for z in self._z if self._n_gen[z]}
        self._qder = {z: Trace(self._n_der[z]) for z in self._z if self._n_der[z]}
        self._pcc = {z: Trace(2) for z in self._pcc_zones}             # set, metered
        self._taps = Trace(len(self._oltc_lanes)) if self._oltc_lanes else None
        self._banks = Trace(len(self._shunt_lanes)) if self._shunt_lanes else None

        # ── layout ──────────────────────────────────────────────────────────
        left = ["v"] + (["tie"] if self._pairs else []) + (["qgen"] if self._qgen else []) \
            + (["res"] if self._res else []) + (["iline"] if self._iline else [])
        right = (["avr"] if self._avr else []) + (["qder"] if self._qder else []) \
            + (["pcc"] if self._pcc else []) + (["taps"] if self._taps else []) \
            + (["banks"] if self._banks else [])
        outer = GridSpec(
            2, 2, figure=self.fig, height_ratios=[0.028, 1.0], hspace=0.035,
            wspace=0.30, left=0.085, right=0.915, bottom=0.05,
            top=1.0 - TITLE_BAR_HEIGHT_FRAC - 0.008,
        )
        fill_section_band(self.fig.add_subplot(outer[0, 0]), "Controlled outputs", COLOUR_MEAS_BAND)
        fill_section_band(self.fig.add_subplot(outer[0, 1]), "Actuators", COLOUR_ACT_BAND)
        gl = GridSpecFromSubplotSpec(len(left), 1, subplot_spec=outer[1, 0], hspace=0.42)
        gr = GridSpecFromSubplotSpec(max(len(right), 1), 1, subplot_spec=outer[1, 1], hspace=0.42)
        self.ax: Dict[str, plt.Axes] = {}
        for names, gs in ((left, gl), (right, gr)):
            for r, name in enumerate(names):
                self.ax[name] = self.fig.add_subplot(gs[r, 0])
        bottoms = [self.ax[left[-1]]] + ([self.ax[right[-1]]] if right else [])
        self._time_axes(list(self.ax.values()), bottom=bottoms)
        self._label_events([self.ax["v"]] + ([self.ax[right[0]]] if right else []))
        self._build_artists()
        self._show()

    # ── artists ─────────────────────────────────────────────────────────────

    def _zone_handles(self, zones: Optional[Sequence[int]] = None) -> List:
        return [proxy(self._zc[z], label=f"Z{z}") for z in (zones or self._z)]

    def _build_artists(self) -> None:
        nan = np.array([np.nan])
        ax = self.ax["v"]
        tile_title(ax, "Voltages (band: min-max)")
        ax.set_ylabel("V / p.u.")
        for v in sorted(set(self._v_set.values())):
            ax.axhline(v, color=REF_GREY, ls=":", lw=0.8)
        for lim in self._v_lim:
            ax.axhline(lim, color=LIMIT_RED, ls="--", lw=0.6, alpha=0.6)
        self._a_v = {}
        for z in self._z:
            band = ax.fill_between(nan, nan, nan, color=self._zc[z], alpha=0.16, lw=0)
            (line,) = ax.plot([], [], color=self._zc[z], lw=1.3)
            self._a_v[z] = (band, line)
        legend(ax, self._zone_handles())
        self._tag_v = value_tag(ax)

        if self._pairs:
            ax = self.ax["tie"]
            tile_title(ax, "Tie-line Q")
            ax.set_ylabel("Q / Mvar")
            ax.axhline(0.0, color=REF_GREY, lw=0.6)
            # Pair colours avoid the zone colours (_c(1.._n_zones)).
            pair_c = [_c(k) for k in (4, 5, 10, 7, 9, 6)]
            self._a_tie = [ax.plot([], [], color=pair_c[k % len(pair_c)], lw=1.2,
                                   label=f"Z{i}>Z{j}")[0]
                           for k, (i, j) in enumerate(self._pairs)]
            legend(ax, self._a_tie)
            self._tag_tie = value_tag(ax)

        if self._qgen:
            ax = self.ax["qgen"]
            tile_title(ax, "Synchronous-machine Q")
            ax.set_ylabel("Q / Mvar")
            ax.axhline(0.0, color=REF_GREY, lw=0.6)
            self._a_qgen = {z: [ax.plot([], [], color=self._zc[z], lw=0.9, alpha=0.85)[0]
                                for _ in range(tr.width)] for z, tr in self._qgen.items()}
            legend(ax, self._zone_handles(list(self._qgen)))

        if self._res:
            ax = self.ax["res"]
            tile_title(ax, "Q reserve (normalised)")
            ax.set_ylabel("r (0 = at limit)")
            ax.set_ylim(-0.01, 0.52)
            self._a_res = {}
            for z in self._z:
                c = self._zc[z]
                self._a_res[z] = (
                    ax.plot([], [], color=c, lw=1.3)[0],
                    ax.plot([], [], color=c, lw=0.8, ls=":")[0],
                    ax.plot([], [], color=c, lw=1.1, ls="--")[0],
                )
            legend(ax, self._zone_handles() + [
                proxy("0.3", label="SG mean"), proxy("0.3", ls=":", lw=0.9, label="SG min"),
                proxy("0.3", ls="--", label="IBG mean")], ncol=3)
            self._tag_res = value_tag(ax)

        if self._iline:
            ax = self.ax["iline"]
            tile_title(ax, "Line loading")
            ax.set_ylabel("I / %")
            ax.axhline(100.0, color=LIMIT_RED, ls="--", lw=0.6)
            self._a_iline = {z: (ax.fill_between(nan, nan, nan, color=self._zc[z], alpha=0.12, lw=0),
                                 ax.plot([], [], color=self._zc[z], lw=1.1)[0])
                             for z in self._z}

        if self._avr:
            ax = self.ax["avr"]
            tile_title(ax, "AVR setpoints")
            ax.set_ylabel("V_set / p.u.")
            for v in sorted(set(self._v_set.values())):
                ax.axhline(v, color=REF_GREY, ls=":", lw=0.8)
            self._a_avr = {z: [ax.plot([], [], color=self._zc[z], lw=1.0, drawstyle="steps-post")[0]
                               for _ in range(tr.width)] for z, tr in self._avr.items()}
            legend(ax, self._zone_handles(list(self._avr)))

        if self._qder:
            ax = self.ax["qder"]
            tile_title(ax, "TS-IBG Q")
            ax.set_ylabel("Q / Mvar")
            ax.axhline(0.0, color=REF_GREY, lw=0.6)
            self._a_qder = {z: [ax.plot([], [], color=self._zc[z], lw=0.9, alpha=0.85)[0]
                                for _ in range(tr.width)] for z, tr in self._qder.items()}
            legend(ax, self._zone_handles(list(self._qder)))

        if self._pcc:
            ax = self.ax["pcc"]
            tile_title(ax, "Q_PCC to STS")
            ax.set_ylabel("sum Q_PCC / Mvar")
            self._a_pcc = {z: (ax.plot([], [], color=self._zc[z], lw=1.0, ls="--", drawstyle="steps-post")[0],
                               ax.plot([], [], color=self._zc[z], lw=1.3)[0])
                           for z in self._pcc}
            legend(ax, self._zone_handles(self._pcc_zones) + [
                proxy("0.3", ls="--", lw=1.0, label="setpoint"), proxy("0.3", label="metered")])
            self._tag_pcc = value_tag(ax)

        if self._taps:
            ax = self.ax["taps"]
            tile_title(ax, "Machine-transformer taps")
            self._setup_lanes(ax, [lab for _, lab in self._oltc_lanes])
            self._mesh_taps = None
            self._lane_txt_taps = self._lane_texts(ax, len(self._oltc_lanes))
        if self._banks:
            ax = self.ax["banks"]
            tile_title(ax, "MSC / MSR levels (C < 0 < R)")
            self._setup_lanes(ax, [lab for *_, lab in self._shunt_lanes])
            self._mesh_banks = None
            self._lane_txt_banks = self._lane_texts(ax, len(self._shunt_lanes))

    @staticmethod
    def _setup_lanes(ax: plt.Axes, labels: Sequence[str]) -> None:
        """Lane k at y = k, first lane on top."""
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.grid(False)
        ax.tick_params(axis="y", length=0)

    @staticmethod
    def _lane_texts(ax: plt.Axes, n: int) -> List:
        """Current value and switch count, right of each lane."""
        return [ax.text(1.01, 1.0 - (k + 0.5) / n, "", transform=ax.transAxes, fontsize=6.5,
                        va="center", ha="left", color="0.25") for k in range(n)]

    # ── ingest ──────────────────────────────────────────────────────────────

    def _ingest(self, rec: "MultiTSOIterationRecord", t: float) -> None:
        if not rec.zone_v_meas_mean:
            return  # no metered TS sample at this step
        nan = float("nan")
        for z in self._z:
            self._v[z].append(t, (rec.zone_v_meas_min.get(z, nan),
                                  rec.zone_v_meas_mean.get(z, nan),
                                  rec.zone_v_meas_max.get(z, nan)))
            if z in self._qgen:
                self._qgen[z].append(t, rec.zone_q_gen_meas.get(z, ()))
            if z in self._avr:
                self._avr[z].append(t, rec.zone_v_gen.get(z, ()))
            if z in self._qder:
                self._qder[z].append(t, rec.zone_q_der_meas.get(z, ()))
            if self._res:
                sg = np.asarray(rec.gen_q_reserve_meas.get(z, ()), dtype=float)
                der = np.asarray(rec.tso_der_q_reserve_meas.get(z, ()), dtype=float)
                self._res[z].append(t, (_nanmean(sg), _nanmin(sg), _nanmean(der)))
            if self._iline:
                self._iline[z].append(t, (rec.zone_line_loading_meas_min_pct.get(z, nan),
                                          rec.zone_line_loading_meas_mean_pct.get(z, nan),
                                          rec.zone_line_loading_meas_max_pct.get(z, nan)))
        if self._tie is not None:
            self._tie.append(t, [rec.zone_tie_q_meas_mvar.get(p, nan) for p in self._pairs])
        if self._pcc:
            meas = {z: 0.0 for z in self._pcc}
            seen = set()
            for key, q in rec.dso_trafo_q_meas_mvar.items():
                z = self._dso_zone.get(key.split("|", 1)[0])
                if z in meas:
                    meas[z] += q
                    seen.add(z)
            for z in self._pcc:
                # Plot what the STS is sent: u_pcc plus the switched-shunt
                # feedforward, which is what the metered flow is tracking.
                q_set = rec.zone_q_pcc_set.get(z)
                q_sent = (float(np.sum(q_set)) + float(np.sum(rec.zone_q_pcc_sh_offset.get(z, 0.0)))
                          if q_set is not None and len(q_set) else nan)
                self._pcc[z].append(t, (q_sent,
                                        meas[z] if z in seen else nan))
        if self._taps is not None:
            self._taps.append(t, np.concatenate(
                [np.asarray(rec.zone_oltc_taps.get(z, ()), dtype=float) for z in self._z]))
        if self._banks is not None:
            self._banks.append(t, [
                (-1.0 if kind == "MSC" else 1.0) * rec.tso_shunt_step.get(sh, nan)
                for sh, _z, kind, _lab in self._shunt_lanes])

    # ── render ──────────────────────────────────────────────────────────────

    def _render(self) -> None:
        v_all = []
        for z, (band, line) in self._a_v.items():
            tr = self._v[z]
            set_band(band, tr.t, tr.col(0), tr.col(2))
            line.set_data(tr.t, tr.col(1))
            v_all.append(tr.y)
        set_ylim_from(self.ax["v"], *v_all, include=list(self._v_set.values()), min_span=0.01)
        self._tag_v.set_text("   ".join(f"Z{z} {self._v[z].last[1]:.3f}" for z in self._z))

        if self._pairs:
            for k, line in enumerate(self._a_tie):
                line.set_data(self._tie.t, self._tie.col(k))
            set_ylim_from(self.ax["tie"], self._tie.y, include=[0.0], min_span=10.0)
            self._tag_tie.set_text("   ".join(
                f"{line.get_label()} {v:+.0f}" for line, v in zip(self._a_tie, self._tie.last)))

        if self._qgen:
            self._render_multi(self.ax["qgen"], self._qgen, self._a_qgen, include=[0.0])
        if self._res:
            for z, (sg, sg_min, der) in self._a_res.items():
                tr = self._res[z]
                sg.set_data(tr.t, tr.col(0))
                sg_min.set_data(tr.t, tr.col(1))
                der.set_data(tr.t, tr.col(2))
            self._tag_res.set_text("SG min " + "  ".join(
                f"Z{z} {self._res[z].last[1]:.2f}" for z in self._z))
        if self._iline:
            arrs = []
            for z, (band, line) in self._a_iline.items():
                tr = self._iline[z]
                set_band(band, tr.t, tr.col(0), tr.col(2))
                line.set_data(tr.t, tr.col(1))
                arrs.append(tr.y)
            set_ylim_from(self.ax["iline"], *arrs, min_span=10.0)
        if self._avr:
            self._render_multi(self.ax["avr"], self._avr, self._a_avr,
                               include=list(self._v_set.values()), min_span=0.01)
        if self._qder:
            self._render_multi(self.ax["qder"], self._qder, self._a_qder, include=[0.0])
        if self._pcc:
            arrs = []
            for z, (l_set, l_meas) in self._a_pcc.items():
                tr = self._pcc[z]
                l_set.set_data(tr.t, tr.col(0))
                l_meas.set_data(tr.t, tr.col(1))
                arrs.append(tr.y)
            set_ylim_from(self.ax["pcc"], *arrs, min_span=10.0)
            self._tag_pcc.set_text("   ".join(
                f"Z{z} {self._pcc[z].last[0]:+.0f}/{self._pcc[z].last[1]:+.0f}" for z in self._pcc))
        if self._taps is not None:
            self._mesh_taps = self._render_lanes(
                self.ax["taps"], self._taps, self._mesh_taps, self._lane_txt_taps,
                lambda v: f"{v:+.0f}")
        if self._banks is not None:
            self._mesh_banks = self._render_lanes(
                self.ax["banks"], self._banks, self._mesh_banks, self._lane_txt_banks,
                lambda v: f"{abs(v):.0f}")

    @staticmethod
    def _render_multi(ax, traces: Dict[int, Trace], lines: Dict[int, List],
                      *, include=(), min_span: float = 10.0) -> None:
        arrs = []
        for z, tr in traces.items():
            for k, line in enumerate(lines[z]):
                line.set_data(tr.t, tr.col(k))
            arrs.append(tr.y)
        set_ylim_from(ax, *arrs, include=include, min_span=min_span)

    @staticmethod
    def _render_lanes(ax, tr: Trace, mesh, texts, fmt):
        """Steps-post raster: lane k holds value y[i, k] on [t_i, t_{i+1})."""
        if tr.n == 0:
            return mesh
        t = tr.t
        dt = t[-1] - t[-2] if tr.n > 1 else 1.0
        edges = np.append(t, t[-1] + max(dt, 1e-6))
        z = tr.y.T
        vmax = max(1.0, float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0)
        if mesh is not None:
            mesh.remove()
        mesh = ax.pcolormesh(edges, np.arange(tr.width + 1) - 0.5, z, cmap="RdBu_r",
                             norm=TwoSlopeNorm(0.0, -vmax, vmax), shading="flat",
                             rasterized=True, zorder=0)
        for k, txt in enumerate(texts):
            col = tr.col(k)
            col = col[np.isfinite(col)]
            n_sw = int(np.count_nonzero(np.diff(col))) if col.size > 1 else 0
            txt.set_text(f"{fmt(col[-1])}  {n_sw}x" if col.size else "")
        return mesh


def _nanmean(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _nanmin(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(a.min()) if a.size else float("nan")
