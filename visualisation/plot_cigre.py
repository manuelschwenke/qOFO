#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualisation/plot_cigre.py
===========================
Paper-ready figures for the CIGRE Energy Forum 2026 case study
(``experiments/005_CIGRE_MULTI.py``).  Three figures, each rendered as a
vector PDF in the CIGRE paper font (Times New Roman / STIX maths) and written
to both the paper's ``Figures/`` directory and the repo ``results/`` tree:

* **Fig. 3a** ``Fig3a_voltage_tracking.pdf`` -- single-panel system-wide TS
  voltage tracking error (RMS over all EHV buses), variants overlaid.
* **Fig. 3b** ``Fig3b_iface_tracking.pdf`` -- per-STS-group interface
  reactive-power tracking, one subplot per DSO group, drawn like the cascade
  live plot: per coupling transformer a solid (measured Q) + dashed
  (dispatched setpoint) trace.  Shows the proposed V4 (and the V5 centralized
  reference when ``show_v5=True``), not the local-only baselines which dispatch
  no interface setpoint.
* **Fig. 4** ``Fig4_capability.pdf`` (+ ``Fig4_capability_all.pdf``) --
  synchronous-generator P-Q operating-point clouds against the Milano §12.2
  capability envelope, variants overlaid; ``_all`` shows every machine so the
  user can pick the 2-3 to keep via ``GEN_SELECT``.
* **Fig. 5** ``Fig5_tieflow.pdf`` -- inter-zone tie-line reactive-power flow,
  one panel per zone pair, variants overlaid.

The metrics are computed by ``experiments.helpers.comparison_metrics``; the
generator-capability geometry reuses
``visualisation.plot_compare_scenarios._build_gen_info_for_scenario`` and
``core.actuator_bounds``.

Author: Manuel Schwenke / Claude Code
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from experiments.helpers.records import MultiTSOIterationRecord
from experiments.helpers.comparison_metrics import (
    V_SET_DEFAULT,
    q_iface_per_trafo,
    tie_q_per_pair,
    voltage_rms_err_all,
    voltage_rms_err_per_zone,
)
from visualisation.style import TU_COLOURS


# ---------------------------------------------------------------------------
#  Variant palette / labels (V1..V4; proposed = V4)
# ---------------------------------------------------------------------------

CIGRE_PALETTE: Dict[str, str] = {
    "V1": TU_COLOURS[8],   # #B90F22 red    -- lower bound
    "V2": TU_COLOURS[9],   # #D28700 amber  -- local-only
    "V3": TU_COLOURS[1],   # #004E8A blue   -- one-sided OFO
    "V4": TU_COLOURS[4],   # #008877 teal   -- proposed (cascaded OFO)
    "V5": "#000000",       # black (dashed) -- centralized upper-bound reference
}
#: The proposed variant is drawn slightly heavier so it reads as the headline.
_PROPOSED = "V4"
#: The single centralized controller is the best-case upper-bound reference;
#: drawn as a heavier black dashed line so it reads as the benchmark envelope.
_REFERENCE = "V5"


def _variant_style(name: str) -> dict:
    col = CIGRE_PALETTE.get(name, TU_COLOURS[hash(name) % len(TU_COLOURS)])
    if name == _REFERENCE:
        return dict(color=col, linewidth=1.0, zorder=6, linestyle=(0, (5, 2)))
    lw = 1.2 if name == _PROPOSED else 1.0
    z = 5 if name == _PROPOSED else 3
    return dict(color=col, linewidth=lw, zorder=z)


# ---------------------------------------------------------------------------
#  Styling + saving
# ---------------------------------------------------------------------------


def apply_cigre_style() -> None:
    """Apply matplotlib rcParams matching the CIGRE paper template.

    Main text in Times New Roman (with Times / DejaVu Serif fallbacks) and
    maths in STIX, which is metric-compatible with Times and visually close to
    the paper's XITS Math.  No LaTeX dependency (kept off for speed and
    portability).  Idempotent.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Times", "XITS", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size":        9.0,
        "axes.titlesize":   9.0,
        "axes.titleweight": "bold",
        "axes.labelsize":   9.0,
        "xtick.labelsize":  8.0,
        "ytick.labelsize":  8.0,
        "legend.fontsize":  8.0,
        "axes.grid":        True,
        "grid.alpha":       0.35,
        "grid.linewidth":   0.5,
        "lines.linewidth":  1.0,
        "axes.linewidth":   0.7,
        "savefig.dpi":      300,
        "pdf.fonttype":     42,   # embed TrueType (editable, no Type-3)
        "ps.fonttype":      42,
        "text.usetex":      False,
    })


def _save_pdf(fig, basename: str, out_dirs: Sequence[str],
              also_png_dir: Optional[str] = None,
              tight: bool = True) -> None:
    """Save *fig* as ``<basename>.pdf`` into every directory in *out_dirs*.

    Locked / unwritable targets are reported but never abort the run (mirrors
    ``plot_compare_scenarios._save``).  Optionally also drop a PNG preview into
    *also_png_dir*.

    ``tight=True`` crops surrounding whitespace (``bbox_inches="tight"``).  Set
    ``tight=False`` to keep the full canvas at its exact ``figsize`` -- required
    when several figures must share an identical plotting-box position so they
    align when each is included at ``width=\\linewidth`` in LaTeX.
    """
    bbox = "tight" if tight else None
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{basename}.pdf")
        try:
            fig.savefig(path, bbox_inches=bbox)
            print(f"  [plot_cigre] wrote {path}")
        except (PermissionError, OSError) as exc:
            print(f"  [plot_cigre] WARNING: could not write {path}: {exc} "
                  f"-- close it in any viewer and re-run with --replot")
    if also_png_dir:
        os.makedirs(also_png_dir, exist_ok=True)
        path = os.path.join(also_png_dir, f"{basename}.png")
        try:
            fig.savefig(path, dpi=200, bbox_inches=bbox)
        except (PermissionError, OSError):
            pass


def _ordered_variants(logs: Dict[str, List[MultiTSOIterationRecord]],
                      excluded_variants: List[str] = None) -> List[str]:
    """Variant names with data, in canonical V1..V4 order then any extras."""
    excluded = set(excluded_variants) if excluded_variants else set()
    canonical = [v for v in ("V1", "V2", "V3", "V4") if logs.get(v) and v not in excluded]
    extra = [k for k in logs if k not in canonical and logs.get(k) and k not in excluded]
    return canonical + extra


def _draw_event_markers(axes, events, *, label_ax_index: int = 0) -> None:
    """Mark contingency events with thin vertical rules + rotated labels.

    *events* is a list of ``(t_min, label)`` pairs.  A faint vertical line is
    drawn on every axes; the label is annotated only on ``axes[label_ax_index]``
    (just inside its top edge) so stacked panels are not over-printed.
    """
    if not events:
        return
    axes = np.atleast_1d(axes)
    for ax in axes:
        for t_min, _lab in events:
            ax.axvline(t_min, color="0.25", linewidth=0.45, linestyle="-",
                       alpha=0.7, zorder=2)
    lab_ax = axes[label_ax_index]
    for t_min, lab in events:
        lab_ax.annotate(
            lab, xy=(t_min, 0.985), xycoords=("data", "axes fraction"),
            xytext=(1.5, 0.0), textcoords="offset points",
            rotation=90, ha="left", va="top", fontsize=6, color="0.25",
            zorder=7, annotation_clip=False,
        )


# ---------------------------------------------------------------------------
#  Fig. 3a -- system-wide TS voltage tracking error (RMS over all EHV buses)
# ---------------------------------------------------------------------------

#: Shared geometry for Figs 3a/3b so their plotting boxes coincide when both
#: are included at ``width=\linewidth``.  Same figure width and same left/right
#: axes margins (figure fraction); both saved un-cropped (``tight=False``).
_FIG3_WIDTH_IN = 6.9
_FIG3_LEFT = 0.115
_FIG3_RIGHT = 0.985


def plot_fig3a_voltage(
    logs: Dict[str, List[MultiTSOIterationRecord]],
    out_dirs: Sequence[str],
    *,
    v_set: float = V_SET_DEFAULT,
    per_zone: bool = False,
    events: Optional[Sequence] = None,
    png_dir: Optional[str] = None,
) -> None:
    """TS voltage tracking-error time series (Fig. 3a).

    If *per_zone* is false, plot the system-wide RMS voltage tracking error over
    all EHV buses. If *per_zone* is true, plot one RMS trace per TSO zone in a
    stacked layout (one subplot per zone). The per-zone series are read from
    ``record.zone_v_rms_err_pu`` via :func:`voltage_rms_err_per_zone` (with the
    helper's documented fallback to ``|zone_v_mean - v_set|`` for older pickles).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    variants = _ordered_variants(logs)
    if not variants:
        print("  [plot_cigre] Fig.3a: no variants with data -- skipped")
        return

    if not per_zone:
        fig, ax = plt.subplots(1, 1, figsize=(_FIG3_WIDTH_IN, 2.7))

        for name in variants:
            d = voltage_rms_err_all(logs[name], v_set)
            y = d["rms_err_pu"] * 1e3  # p.u. -> mp.u. for readability
            ax.plot(d["t_min"], y, label=name, **_variant_style(name))

        ax.set_ylabel(r"$\bar e_\mathrm{v}$ / mp.u.")
        ax.set_xlabel("Time / min")
        ax.set_ylim(bottom=0.0)
        ax.margins(x=0.0)
        ax.set_title(
            "TS voltage tracking error (RMS over all EHV buses, to $V_{\\mathrm{ref}}$)",
            loc="left",
        )

        _draw_event_markers(ax, events)

        handles = [
            Line2D(
                [],
                [],
                **{
                    k: v
                    for k, v in _variant_style(n).items()
                    if k in ("color", "linewidth", "linestyle")
                },
            )
            for n in variants
        ]
        ax.legend(
            handles,
            variants,
            loc="upper right",
            ncol=1,
            frameon=True,
            framealpha=0.9,
            fontsize=7,
            handlelength=1.4,
            labelspacing=0.3,
            borderaxespad=0.4,
        )

        fig.subplots_adjust(left=_FIG3_LEFT, right=_FIG3_RIGHT, top=0.86, bottom=0.18)

        _save_pdf(
            fig,
            "Fig3a_voltage_tracking",
            out_dirs,
            also_png_dir=png_dir,
            tight=False,
        )
        plt.close(fig)
        return

    # Per-zone RMS voltage tracking error, precomputed once per variant from
    # ``record.zone_v_rms_err_pu`` (the records store the per-zone spatial RMS,
    # not the per-bus voltage arrays the old code assumed via ``record.zone_v``).
    per_zone_by_variant: Dict[str, Dict[str, object]] = {
        name: voltage_rms_err_per_zone(logs[name], v_set) for name in variants
    }
    zone_ids = sorted({
        z
        for pz in per_zone_by_variant.values()
        for z in pz["zones"]
    })
    if not zone_ids:
        print("  [plot_cigre] Fig.3a (per-zone): no zone data -- skipped")
        return

    fig, axes = plt.subplots(
        len(zone_ids),
        1,
        sharex=True,
        figsize=(_FIG3_WIDTH_IN, 3.35),
        gridspec_kw=dict(hspace=0.12),
    )
    axes = np.atleast_1d(axes)

    for row, zone in enumerate(zone_ids):
        ax = axes[row]

        for name in variants:
            pz = per_zone_by_variant[name]
            if zone not in pz["rms_err_pu"]:
                continue
            t_min = np.asarray(pz["t_min"], dtype=float)
            y = pz["rms_err_pu"][zone] * 1e3  # p.u. -> mp.u.
            ax.plot(t_min, y, label=name, **_variant_style(name))

        ax.set_ylabel(rf"$\bar e_{{\mathrm{{v}},{zone}}}$ / mp.u.")
        ax.set_ylim(bottom=0.0)
        ax.margins(x=0.0)

        if row == 0:
            ax.set_title(
                "TS voltage tracking error per zone (RMS to $V_{\\mathrm{ref}}$)",
                loc="left",
            )

    axes[-1].set_xlabel("Time / min")

    _draw_event_markers(axes, events, label_ax_index=0)

    handles = [
        Line2D(
            [],
            [],
            **{
                k: v
                for k, v in _variant_style(n).items()
                if k in ("color", "linewidth", "linestyle")
            },
        )
        for n in variants
    ]
    axes[0].legend(
        handles,
        variants,
        loc="upper right",
        ncol=1,
        frameon=True,
        framealpha=0.9,
        fontsize=7,
        handlelength=1.4,
        labelspacing=0.3,
        borderaxespad=0.4,
    )

    fig.subplots_adjust(left=_FIG3_LEFT, right=_FIG3_RIGHT, top=0.88, bottom=0.14, hspace=0.12)
    fig.align_ylabels(axes)

    _save_pdf(
        fig,
        "Fig3a_voltage_tracking_zones",
        out_dirs,
        also_png_dir=png_dir,
        tight=False,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Fig. 3b -- per-STS interface Q: dispatched setpoint vs measured flow
# ---------------------------------------------------------------------------

#: Variants drawn in Fig. 3b (only those that dispatch an interface setpoint).
#: V4 is the proposed cascade; V5 the centralized reference (toggle via
#: ``show_v5``).  The local-only baselines (V1/V2/V3) issue no STS setpoint.
_IFACE_VARIANTS = ("V4", "V5")


def _trafo_short(trafo_key: str) -> str:
    """``"DSO_1|trafo_37"`` -> ``"T37"``; falls back to the raw key."""
    if "|trafo_" in trafo_key:
        return "T" + trafo_key.rsplit("|trafo_", 1)[1]
    return trafo_key


def plot_fig3b_iface(
    logs: Dict[str, List[MultiTSOIterationRecord]],
    out_dirs: Sequence[str],
    *,
    show_v5: bool = False,
    events: Optional[Sequence] = None,
    png_dir: Optional[str] = None,
) -> None:
    """Per-STS interface reactive-power tracking (Fig. 3b).

    Drawn for V4 by default: per coupling transformer, measured interface
    reactive power is shown against the dispatched setpoint. If *show_v5* is
    true, the centralised reference V5 is overlaid as an additional measured
    trace. The 2x2 layout assumes exactly four STS groups.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sel = [v for v in _IFACE_VARIANTS if logs.get(v)]
    if not show_v5 and "V5" in sel:
        sel.remove("V5")
    if not sel:
        print("  [plot_cigre] Fig.3b: no setpoint-dispatching variant with data -- skipped")
        return

    data = {v: q_iface_per_trafo(logs[v]) for v in sel}
    groups = sorted({g for v in sel for g in data[v]["groups"]})
    if not groups:
        print("  [plot_cigre] Fig.3b: no interface transformers found -- skipped")
        return

    n = len(groups)
    assert n == 4, f"plot_fig3b_iface requires exactly 4 STS groups, got {n}"

    fig, axes_2d = plt.subplots(
        2,
        2,
        sharex=True,
        figsize=(_FIG3_WIDTH_IN, 0.95 * 2 + 0.8),
        gridspec_kw=dict(hspace=0.18),
    )
    axes = axes_2d.flatten()

    v4_lw = _variant_style("V4")["linewidth"]

    for idx, group in enumerate(groups):
        ax = axes[idx]
        trafo_keys = sorted({k for v in sel for k in data[v]["trafos"].get(group, {})})

        group_idx = int(group.split("_")[1])
        y_label = rf"$q_\mathrm{{STS}}$ / Mvar ($\mathcal{{A}}_\mathrm{{S,{group_idx}}}$)"

        for ti, key in enumerate(trafo_keys):
            colour = TU_COLOURS[ti % len(TU_COLOURS)]

            for variant in sel:
                d = data[variant]
                if key not in d["actual_mvar"]:
                    continue

                t_min = d["t_min"]
                q_meas = d["actual_mvar"][key]

                if variant == "V5":
                    ax.plot(
                        t_min,
                        q_meas,
                        color=colour,
                        linewidth=1.0,
                        linestyle=(0, (1, 1)),
                        alpha=0.85,
                        zorder=4,
                    )
                else:
                    ax.plot(
                        t_min,
                        q_meas,
                        color=colour,
                        linewidth=v4_lw,
                        zorder=5,
                        label=_trafo_short(key),
                    )
                    if d["has_setpoint"].get(key):
                        ax.plot(
                            t_min,
                            d["set_mvar"][key],
                            color=colour,
                            linewidth=0.9,
                            linestyle="--",
                            drawstyle="steps-post",
                            alpha=0.9,
                            zorder=5,
                        )

        ax.axhline(0.0, color="#444", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.margins(x=0.0)

        if idx % 2 == 0:
            ax.set_ylabel(y_label, labelpad=6)
        else:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(y_label, rotation=270, labelpad=10)

        if ax.get_legend_handles_labels()[1]:
            ax.legend(
                loc="upper left",
                fontsize=6,
                frameon=False,
                ncol=min(len(trafo_keys), 3),
                handlelength=1.4,
                columnspacing=0.9,
                borderaxespad=0.2,
            )

    for ax in axes_2d[1, :]:
        ax.set_xlabel("Time / min")

    _draw_event_markers(axes, events, label_ax_index=0)

    style_handles = [
        Line2D([], [], color="0.3", linewidth=v4_lw, label="V4 measured"),
        Line2D([], [], color="0.3", linewidth=0.9, linestyle="--", label="V4 setpoint"),
    ]
    if "V5" in sel:
        style_handles.append(
            Line2D(
                [],
                [],
                color="0.3",
                linewidth=1.0,
                linestyle=(0, (1, 1)),
                alpha=0.85,
                label="V5 measured",
            )
        )

    axes[0].legend(
        style_handles,
        [h.get_label() for h in style_handles],
        loc="lower right",
        ncol=len(style_handles),
        frameon=True,
        fontsize=7,
        handlelength=1.6,
        columnspacing=1.0,
        bbox_to_anchor=(1.0, 1.0),
        bbox_transform=axes_2d[0, 1].transAxes,
    )

    fig_height = fig.get_figheight()
    top = 1.0 - 0.32 / fig_height
    bottom = 0.40 / fig_height

    w_gap = 0.05
    col_width = (_FIG3_RIGHT - _FIG3_LEFT - w_gap) / 2.0

    fig.subplots_adjust(
        left=_FIG3_LEFT,
        right=_FIG3_RIGHT,
        top=top,
        bottom=bottom,
        wspace=w_gap / col_width,
        hspace=0.18,
    )

    fig.align_ylabels(axes)

    _save_pdf(
        fig,
        "Fig3b_iface_tracking",
        out_dirs,
        also_png_dir=png_dir,
        tight=False,
    )
    plt.close(fig)

# def plot_fig3b_iface(
#     logs: Dict[str, List[MultiTSOIterationRecord]],
#     out_dirs: Sequence[str],
#     *,
#     show_v5: bool = False,
#     events: Optional[Sequence] = None,
#     png_dir: Optional[str] = None,
# ) -> None:
#     """Per-STS interface reactive-power tracking (Fig. 3b).
#
#     Drawn for V4 (proposed) only by default; the per-coupling-transformer
#     measured Q (solid) is shown against its dispatched setpoint (dashed step).
#     Set ``show_v5=True`` to overlay the centralized reference as dotted traces.
#
#     One subplot per DSO group; within each, one colour per coupling
#     transformer.  For V4 (proposed) the measured interface Q is drawn solid and
#     its dispatched setpoint dashed (step); for V5 (centralized reference, if
#     ``show_v5``) the measured flow is drawn dotted.  Mirrors the cascade live
#     plot's "TSO-DSO Interface Q" panel.  *events* is an optional list of
#     ``(t_min, label)`` contingency markers.
#     """
#     import matplotlib.pyplot as plt
#     from matplotlib.lines import Line2D
#
#     sel = [v for v in _IFACE_VARIANTS if logs.get(v)]
#     if not show_v5 and "V5" in sel:
#         sel.remove("V5")
#     if not sel:
#         print("  [plot_cigre] Fig.3b: no setpoint-dispatching variant with "
#               "data -- skipped")
#         return
#
#     data = {v: q_iface_per_trafo(logs[v]) for v in sel}
#     groups = sorted({g for v in sel for g in data[v]["groups"]})
#     if not groups:
#         print("  [plot_cigre] Fig.3b: no interface trafos found -- skipped")
#         return
#
#     n = len(groups)
#     fig, axes = plt.subplots(n, 1, sharex=True,
#                              figsize=(_FIG3_WIDTH_IN, 0.95 * n + 0.8),
#                              gridspec_kw=dict(hspace=0.12))
#     axes = np.atleast_1d(axes)
#
#     # Per-variant line style: V4 actual solid / setpoint dashed; V5 actual dotted.
#     v4_lw = _variant_style("V4")["linewidth"]
#     for row, g in enumerate(groups):
#         ax = axes[row]
#         trafo_keys = sorted({k for v in sel for k in data[v]["trafos"].get(g, [])})
#         for ti, key in enumerate(trafo_keys):
#             c = TU_COLOURS[ti % len(TU_COLOURS)]
#             for v in sel:
#                 d = data[v]
#                 if key not in d["actual_mvar"]:
#                     continue
#                 t = d["t_min"]
#                 act = d["actual_mvar"][key]
#                 if v == "V5":
#                     ax.plot(t, act, color=c, linewidth=1.0, linestyle=":",
#                             alpha=0.85, zorder=4)
#                 else:  # V4
#                     ax.plot(t, act, color=c, linewidth=v4_lw, zorder=5,
#                             label=_trafo_short(key))
#                     if d["has_setpoint"].get(key):
#                         ax.plot(t, d["set_mvar"][key], color=c, linewidth=0.9,
#                                 linestyle="--", drawstyle="steps-post",
#                                 alpha=0.9, zorder=5)
#         ax.axhline(0.0, color="#444", linewidth=0.6, linestyle="--", alpha=0.6)
#         ax.set_ylabel("%s\n$q_{\\mathrm{STS}}$ / Mvar" % g)
#         ax.margins(x=0.0)
#         # Per-group legend identifying the interface trafos (colours).
#         if ax.get_legend_handles_labels()[1]:
#             ax.legend(loc="upper left", fontsize=6, frameon=False,
#                       ncol=min(len(trafo_keys), 3), handlelength=1.4,
#                       columnspacing=0.9, borderaxespad=0.2)
#     axes[-1].set_xlabel("Time / min")
#
#     # Contingency markers (labelled on the top panel only).
#     _draw_event_markers(axes, events)
#
#     # Title on the top-left; shared style legend at the top-right next to it.
#     axes[0].set_title("STS interface reactive power: measured vs dispatched",
#                       loc="left")
#     style_handles = [
#         Line2D([], [], color="0.3", linewidth=v4_lw, label="V4 measured"),
#         Line2D([], [], color="0.3", linewidth=0.9, linestyle="--",
#                label="V4 setpoint"),
#     ]
#     if "V5" in sel:
#         style_handles.append(
#             Line2D([], [], color="0.3", linewidth=1.0, linestyle=":",
#                    alpha=0.85, label="V5 measured"))
#     axes[0].legend(style_handles, [h.get_label() for h in style_handles],
#                    loc="lower right", ncol=len(style_handles), frameon=True,
#                    fontsize=7, handlelength=1.6, columnspacing=1.0,
#                    bbox_to_anchor=(1.0, 1.0), bbox_transform=axes[0].transAxes)
#
#     # Fixed margins (shared left/right with Fig 3a) + un-cropped save so the
#     # plotting box lands at the same place when both are included at
#     # width=\linewidth.  top/bottom are converted from fixed inch reserves so
#     # the panel band is independent of the figure height.
#     fig_h = 0.95 * n + 0.8
#     top = 1.0 - 0.42 / fig_h   # ~0.42 in reserved for title + legend strip
#     bot = 0.52 / fig_h         # ~0.52 in reserved for the x-axis label
#     fig.subplots_adjust(left=_FIG3_LEFT, right=_FIG3_RIGHT, top=top, bottom=bot,
#                         hspace=0.12)
#     fig.align_ylabels(axes)
#     _save_pdf(fig, "Fig3b_iface_tracking", out_dirs, also_png_dir=png_dir,
#               tight=False)
#     plt.close(fig)


# ---------------------------------------------------------------------------
#  Fig. 4 -- generator P-Q capability clouds
# ---------------------------------------------------------------------------


def _draw_capability_tile(ax, gi: Dict, logs, variants,
                          filter_near_zero: bool = True,
                          filter_outside_envelope: bool = True,
                          p_zero_tolerance_mw: float = 20.0) -> None:
    """Draw one generator's capability envelope + per-variant P-Q scatter.

    Parameters
    ----------
    filter_near_zero : bool
        If True (default), exclude operating points with |P| < p_zero_tolerance_mw.
    filter_outside_envelope : bool
        If True (default), exclude operating points outside the Q capability envelope.
    p_zero_tolerance_mw : float
        Tolerance in MW below which operating points are considered near-zero.
    """
    from core.actuator_bounds import GeneratorParameters, compute_generator_q_limits

    gp = GeneratorParameters(
        s_rated_mva=gi["s_rated_mva"], p_max_mw=gi["p_max_mw"],
        xd_pu=gi["xd_pu"], i_f_max_pu=gi["i_f_max_pu"],
        beta=gi["beta"], q0_pu=gi["q0_pu"],
    )
    z = gi["zone"]
    k = gi["k_in_zone"]
    v = 1.03

    # Composite feasible region (P-dependent Q limits).
    p_min = gi.get("p_min_mw", 0.0)
    p_sweep = np.linspace(p_min, gi["p_max_mw"], 300)
    q_lo = np.empty_like(p_sweep)
    q_hi = np.empty_like(p_sweep)
    for ip, ppv in enumerate(p_sweep):
        q_lo[ip], q_hi[ip] = compute_generator_q_limits(gp, p_mw=ppv, v_pu=v)
    ax.fill_between(p_sweep, q_lo, q_hi, color="0.85", alpha=0.6, zorder=1,
                    label="Capability")
    ax.plot(p_sweep, q_hi, color="0.35", linewidth=0.9, zorder=1)
    ax.plot(p_sweep, q_lo, color="0.35", linewidth=0.9, zorder=1)
    ax.axvline(gi["p_max_mw"], color="0.35", linewidth=0.6, linestyle="--",
               alpha=0.6, zorder=1)

    # Per-variant operating-point scatter.
    for name in variants:
        p_vals, q_vals = [], []
        for r in logs[name]:
            pv = r.zone_p_gen.get(z)
            qv = r.zone_q_gen.get(z)
            if pv is not None and qv is not None and k < len(pv):
                p = float(pv[k])
                q = float(qv[k])
                if filter_near_zero and abs(p) < p_zero_tolerance_mw:
                    continue
                if filter_outside_envelope:
                    q_min, q_max = compute_generator_q_limits(gp, p_mw=p, v_pu=v)
                    if q < q_min or q > q_max:
                        continue
                p_vals.append(p)
                q_vals.append(q)
        if p_vals:
            ax.scatter(p_vals, q_vals, color=CIGRE_PALETTE.get(name, "0.5"),
                       alpha=0.30, s=8, zorder=3, edgecolors="none", label=name)

    ax.axhline(0.0, color="black", linewidth=0.3, linestyle=":")
    # "G3_bus31" -> "G3@B31"; zone z -> calligraphic area label A_{T,z}.
    gen_label = str(gi["name"]).replace("_bus", "@B")
    ax.set_title("%s ($\\mathcal{A}_{\\mathrm{T,%d}}$)" % (gen_label, z),
                 loc="left")
    ax.set_xlabel("$P$ / MW")
    ax.set_ylabel("$Q$ / Mvar")


def _gen_info_with_k(scenario: str) -> List[Dict]:
    """gen_info list (sorted by zone, gen_idx) annotated with the per-zone
    index ``k_in_zone`` used to index ``zone_p_gen`` / ``zone_q_gen``."""
    from visualisation.plot_compare_scenarios import _build_gen_info_for_scenario

    gen_info = _build_gen_info_for_scenario(scenario)
    counter: Dict[int, int] = {}
    for gi in gen_info:
        z = gi["zone"]
        gi["k_in_zone"] = counter.get(z, 0)
        counter[z] = gi["k_in_zone"] + 1
    return gen_info


def _select_gens(gen_info: List[Dict], gen_select) -> List[Dict]:
    """Resolve ``GEN_SELECT`` to a subset of gen_info.

    ``gen_select`` may be a list of generator names (str) or gen indices (int).
    ``None`` -> one representative generator per zone (first in each zone).
    """
    if gen_select:
        wanted = set(gen_select)
        sub = [gi for gi in gen_info
               if gi["name"] in wanted or gi["gen_idx"] in wanted]
        if sub:
            return sub
        print(f"  [plot_cigre] GEN_SELECT={gen_select} matched nothing; "
              f"falling back to one gen per zone")
    seen = set()
    sub = []
    for gi in gen_info:
        if gi["zone"] not in seen:
            sub.append(gi)
            seen.add(gi["zone"])
    return sub


def plot_fig4_capability(
    logs: Dict[str, List[MultiTSOIterationRecord]],
    out_dirs: Sequence[str],
    *,
    scenario: str = "wind_replace",
    excluded_variants: List[str] = None,
    gen_select=None,
    png_dir: Optional[str] = None,
) -> None:
    """Generator P-Q capability clouds (Fig. 4) -- subset + full overview."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    variants = _ordered_variants(logs, excluded_variants=excluded_variants)
    if not variants:
        print("  [plot_cigre] Fig.4: no variants with data -- skipped")
        return
    try:
        gen_info = _gen_info_with_k(scenario)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  [plot_cigre] Fig.4: could not build gen_info: {exc}")
        return
    if not gen_info:
        return

    def _legend_handles():
        h = [Line2D([], [], marker="o", linestyle="none", color=CIGRE_PALETTE.get(n, "0.5"),
                    markersize=5, alpha=0.6) for n in variants]
        h.append(Line2D([], [], color="0.6", linewidth=6, alpha=0.6))
        return h, variants + ["Capability"]

    # ── Subset figure (the one for the paper) ───────────────────────────────
    sub = _select_gens(gen_info, gen_select)
    ncol = len(sub)
    fig, axes = plt.subplots(1, ncol, figsize=(2.55 * ncol - 0.8, 2.5),
                             squeeze=False)
    for ax, gi in zip(axes[0], sub):
        _draw_capability_tile(ax, gi, logs, variants)
    h, lbl = _legend_handles()
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.legend(h, lbl, loc="lower center", ncol=len(lbl), frameon=True,
               bbox_to_anchor=(0.5, 0.01))
    _save_pdf(fig, "Fig4_capability", out_dirs, also_png_dir=png_dir)
    plt.close(fig)

    # ── Full overview (all machines, to choose GEN_SELECT from) ─────────────
    n = len(gen_info)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    figA, axesA = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.7 * nrows),
                               squeeze=False)
    for idx, gi in enumerate(gen_info):
        r, c = divmod(idx, ncols)
        _draw_capability_tile(axesA[r, c], gi, logs, variants)
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axesA[r, c].set_visible(False)
    h, lbl = _legend_handles()
    figA.legend(h, lbl, loc="lower center", ncol=len(lbl), frameon=True)
    figA.tight_layout(rect=(0, 0.04, 1, 1))
    _save_pdf(figA, "Fig4_capability_all", out_dirs, also_png_dir=png_dir)
    plt.close(figA)


# ---------------------------------------------------------------------------
#  Fig. 5 -- inter-zone tie-line reactive-power flow
# ---------------------------------------------------------------------------

#: Preferred panel order for the 3-zone partition.
_TIE_PAIR_ORDER = [(1, 2), (2, 3), (1, 3)]


def plot_fig5_tie(
    logs: Dict[str, List[MultiTSOIterationRecord]],
    out_dirs: Sequence[str],
    *,
    png_dir: Optional[str] = None,
) -> None:
    """Inter-zone tie-line Q, one panel per zone pair (Fig. 5)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    variants = _ordered_variants(logs)
    if not variants:
        print("  [plot_cigre] Fig.5: no variants with data -- skipped")
        return

    all_pairs = sorted({p for v in variants for p in tie_q_per_pair(logs[v])["pairs"]})
    pairs = [p for p in _TIE_PAIR_ORDER if p in all_pairs]
    pairs += [p for p in all_pairs if p not in pairs]
    if not pairs:
        print("  [plot_cigre] Fig.5: no tie-line pairs found -- skipped")
        return

    n = len(pairs)
    fig, axes = plt.subplots(n, 1, sharex=True,
                             figsize=(6.9, 1.25 * n + 0.8),
                             gridspec_kw=dict(hspace=0.28))
    axes = np.atleast_1d(axes)
    for row, p in enumerate(pairs):
        ax = axes[row]
        for name in variants:
            d = tie_q_per_pair(logs[name])
            if p not in d["q_mvar"]:
                continue
            ax.plot(d["t_min"], d["q_mvar"][p], label=name, **_variant_style(name))
        ax.axhline(0.0, color="#444", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.set_ylabel("Z%d$\\rightarrow$Z%d\n$q_{\\mathrm{tie}}$ [Mvar]" % p)
        ax.margins(x=0.0)
        if row == 0:
            ax.set_title("Inter-zone tie-line reactive-power flow", loc="left")
    axes[-1].set_xlabel("Time [min]")

    handles = [Line2D([], [], **{k: v for k, v in _variant_style(nm).items()
                                 if k in ("color", "linewidth")})
               for nm in variants]
    fig.legend(handles, variants, loc="upper center", ncol=len(variants),
               frameon=True, bbox_to_anchor=(0.5, 1.01))
    fig.align_ylabels(axes)
    _save_pdf(fig, "Fig5_tieflow", out_dirs, also_png_dir=png_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Orchestrator
# ---------------------------------------------------------------------------


def make_cigre_figures(
    logs: Dict[str, List[MultiTSOIterationRecord]],
    out_dirs: Sequence[str],
    *,
    scenario: str = "wind_replace",
    gen_select=None,
    v_set: float = V_SET_DEFAULT,
    iface_show_v5: bool = False,
    events: Optional[Sequence] = None,
    png_dir: Optional[str] = None,
) -> None:
    """Render Figs 3a/3b-5 into every directory in *out_dirs* (PDF) and *png_dir*.

    *events* is an optional list of ``(t_min, label)`` contingency markers
    overlaid on the two tracking figures (3a, 3b).
    """
    apply_cigre_style()
    plot_fig3a_voltage(logs, out_dirs, v_set=v_set, events=events, png_dir=png_dir)
    plot_fig3b_iface(logs, out_dirs, show_v5=iface_show_v5, events=events,
                     png_dir=png_dir)
    plot_fig4_capability(logs, out_dirs, scenario=scenario, excluded_variants=['V1'],
                         gen_select=gen_select, png_dir=png_dir)
    plot_fig5_tie(logs, out_dirs, png_dir=png_dir)
