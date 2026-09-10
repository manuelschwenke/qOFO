"""
ch_10_2_figures.py
==================

Thesis Ch 10 figures and the single-run metrics table, from the variant ladder
in ``results/THESIS_ch10_variants_single_run``.

Produces, into ``graphics/Ch10`` of the dissertation repo:

* ``ch10_voltage_tracking.pdf``  -- thesis Fig. 10.1: system-wide RMS voltage
  tracking error over the horizon, one line per variant.
* ``ch10_iface_tracking.pdf``    -- thesis Fig. 10.2: per-DSO interface reactive
  power against its dispatched setpoint under the proposed cascade O3.
* ``ch10_ladder_metrics.tex``    -- the single-run metrics table, ready to
  \\input.

Why not ``visualisation/plot_cigre.make_cigre_figures``: that module hardcodes
``_PROPOSED = "V4"``, ``_REFERENCE = "V5"`` and ``_IFACE_VARIANTS = ("V4","V5")``
for the paper's variant names, which do not exist in the thesis ladder.  Its
*style* is reused (``apply_cigre_style``, ``CIGRE_PALETTE``, ``_variant_style``,
``_draw_event_markers``) so the figures sit in the same visual language, but the
axes are built here against the thesis names.  Editing plot_cigre instead would
have put the paper's figures at risk for no gain.

Run:
    python -m experiments.ch_10_case_study.ch_10_2_figures
    python -m experiments.ch_10_case_study.ch_10_2_figures --run-dir <dir>
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiments.helpers.comparison_metrics import (
    q_iface_per_trafo,
    voltage_rms_err_all,
)
from experiments.paths import RESULTS_ROOT

RESULT_DIR = Path(RESULTS_ROOT) / "THESIS_ch10_variants_single_run"
THESIS_GFX = Path(
    r"C:\Users\Manuel Schwenke\Desktop\Daten\01_Forschung\12_Dissertation"
    r"\latex_diss_ms\graphics\Ch10"
)

LADDER = ["L1", "L2", "S1", "O1", "O2", "O3", "M"]
PROPOSED = "O3"
PILOT_BUSES = {1: 25, 2: 7, 3: 20}
V_SET = 1.03


#: Figure palette, keyed by THESIS variant name.
#:
#: visualisation.plot_cigre.CIGRE_PALETTE is keyed by the PAPER names (V1..V5);
#: looking L1/L2/O2/O3/M up in it silently returns the fallback and renders five
#: of seven series in identical grey.  Hence a palette of our own.
#:
#: Only FOUR categorical hues are used, for the four schemes that carry the
#: argument.  Six TU hues were tried and none passed: the brand palette cannot
#: separate that many series (worst adjacent pair reached normal-vision dE 10-14,
#: against a floor of 15).  Rather than force it, the contextual rungs drop to
#: grey and are separated by dash pattern instead -- composite encoding, and
#: honest, since L1/L2 overlap almost exactly anyway and M is a bound rather
#: than a candidate.
#:
#: Validated with the dataviz validator, light surface:
#:   CVD separation      PASS  worst adjacent dE 12.8 (protan)
#:   normal-vision floor PASS  worst adjacent dE 19.2
#:   chroma floor        PASS
#:   lightness band      #004E8A at L=0.418 sits just under the 0.43 band --
#:                       accepted, because the alternative (#00689D) fails the
#:                       HARD normal-vision check against the teal at dE 13.0.
#:   contrast            #D28700 at 2.84:1 -- carries a legend entry, as required.
PALETTE = {
    "S1": "#B90F22",   # classical reference
    "O1": "#D28700",   # control-law isolation
    "O2": "#008877",   # + discrete devices
    "O3": "#004E8A",   # proposed cascade
    "L1": "#767676",   # contextual floor, separated by dash
    "L2": "#767676",
    "M":  "#1a1a1a",   # bound, separated by dot
}
#: Within-panel hues for Fig. 10.2 -- three of the four validated categorical
#: hues above, so the two figures speak one colour language and the triple
#: inherits the validation (worst adjacent normal-vision dE 19.2).
IFACE_COLOURS = ["#004E8A", "#D28700", "#008877"]

STYLE = {
    "L1": dict(ls=(0, (5, 2)),   lw=1.0),
    "L2": dict(ls=(0, (1.5, 1.5)), lw=1.0),
    "S1": dict(ls="-",           lw=1.3),
    "O1": dict(ls="-",           lw=1.3),
    "O2": dict(ls="-",           lw=1.3),
    "O3": dict(ls="-",           lw=2.1),
    "M":  dict(ls=(0, (1, 1.8)), lw=1.2),
}

#: Contingencies of the validated config, for the event markers.
EVENTS = [
    (30, "gen trip"), (120, "load"), (180, "gen restore"),
    (210, "line trip"), (300, "line restore"), (330, "load off"),
]


def load(run_dir: Path) -> Dict[str, List]:
    logs = {}
    for n in LADDER:
        p = run_dir / n / "log.pkl"
        if p.exists():
            with open(p, "rb") as f:
                logs[n] = pickle.load(f)
    return logs


# ---------------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------------


def fig_voltage_tracking(logs, out: Path) -> None:
    """Thesis Fig. 10.1 -- one line per variant, single axis, legend always.

    Seven overlaid series is at the edge of what a single axis carries.  They
    are kept on one axis rather than split into small multiples because the
    point of the figure is the ORDERING of the variants at each instant, which
    a facet destroys.  Legibility is bought instead with the ladder's own
    structure: local variants dashed, the reference dotted, the proposed
    controller heavier than the rest.
    """
    import matplotlib.pyplot as plt
    from visualisation.plot_cigre import apply_cigre_style

    apply_cigre_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.0))

    for name in LADDER:
        if name not in logs:
            continue
        d = voltage_rms_err_all(logs[name], v_set=V_SET)
        t, y = np.asarray(d["t_min"], float), np.asarray(d["rms_err_pu"], float)
        ax.plot(
            t, y,
            color=PALETTE[name], **STYLE[name],
            label=name, zorder=3 if name == PROPOSED else 2,
        )

    for t_min, lbl in EVENTS:
        ax.axvline(t_min, color="0.75", lw=0.6, zorder=1)
        ax.text(t_min, ax.get_ylim()[1], f" {lbl}", rotation=90,
                va="top", ha="left", fontsize=5.5, color="0.45")

    ax.set_xlabel(r"$t$ / min")
    ax.set_ylabel(r"$e_{\mathrm{v}}$ / p.u.")
    ax.set_xlim(0, max(t))
    ax.set_ylim(bottom=0)
    ax.grid(True, lw=0.4, color="0.9")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(ncol=7, frameon=False, fontsize=6.5,
              loc="upper center", bbox_to_anchor=(0.5, 1.18),
              columnspacing=1.1, handlelength=1.8)

    fig.tight_layout()
    fig.savefig(out / "ch10_voltage_tracking.pdf", bbox_inches="tight")
    fig.savefig(out / "ch10_voltage_tracking.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out/'ch10_voltage_tracking.pdf'}")


def fig_iface_tracking(logs, out: Path) -> None:
    """Thesis Fig. 10.2 -- interface Q against setpoint under the cascade."""
    import matplotlib.pyplot as plt
    from visualisation.plot_cigre import apply_cigre_style

    if PROPOSED not in logs:
        print(f"  !! {PROPOSED} missing; skipping interface figure")
        return

    apply_cigre_style()
    d = q_iface_per_trafo(logs[PROPOSED])
    t = np.asarray(d["t_min"], float)
    groups = list(d["groups"])

    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(7.0, 1.5 * n), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, g in zip(axes, groups):
        for k, key in enumerate(d["trafos"][g]):
            act = d["actual_mvar"][key]
            setp = d["set_mvar"][key]
            ax.plot(t, np.asarray(setp, float), lw=0.9, ls=(0, (3, 2)),
                    color="0.55", zorder=2)
            ax.plot(t, np.asarray(act, float), lw=1.2, zorder=3,
                    color=IFACE_COLOURS[k % len(IFACE_COLOURS)],
                    label=f"interface {k + 1}")
        ax.set_ylabel(r"$q_{\mathrm{STS}}$ / Mvar")
        ax.text(0.012, 0.06, g, transform=ax.transAxes,
                fontsize=7, va="bottom", color="0.3")
        ax.grid(True, lw=0.4, color="0.9")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[-1].set_xlabel(r"$t$ / min")
    # Legend above the first panel, not inside it: at "upper right" the third
    # entry clipped the axes edge and the panel label sat under it.
    axes[0].legend(ncol=4, frameon=False, fontsize=6.5,
                   loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(out / "ch10_iface_tracking.pdf", bbox_inches="tight")
    fig.savefig(out / "ch10_iface_tracking.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out/'ch10_iface_tracking.pdf'}")


# ---------------------------------------------------------------------------
#  Metrics table
# ---------------------------------------------------------------------------


def _rms(v) -> float:
    a = np.asarray([x for x in v if x is not None], float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")


def metrics(logs) -> Dict[str, Dict[str, float]]:
    out = {}
    for n in LADDER:
        if n not in logs:
            continue
        lg = logs[n]
        ev = [_rms([r.zone_v_rms_err_pu.get(z, np.nan) for r in lg
                    if getattr(r, "zone_v_rms_err_pu", None)]) for z in (1, 2, 3)]
        ep = [_rms([float(r.bus_vm_pu[PILOT_BUSES[z]]) - V_SET for r in lg
                    if getattr(r, "bus_vm_pu", None) and PILOT_BUSES[z] in r.bus_vm_pu])
              for z in (1, 2, 3)]
        vmin = min(float(r.zone_v_min[z]) for r in lg
                   for z in (1, 2, 3) if z in getattr(r, "zone_v_min", {}))
        vmax = max(float(r.zone_v_max[z]) for r in lg
                   for z in (1, 2, 3) if z in getattr(r, "zone_v_max", {}))
        taps, prev = 0, {}
        for r in lg:
            for z, tp in (getattr(r, "zone_oltc_taps", {}) or {}).items():
                tp = np.asarray(tp, float)
                if z in prev and prev[z].shape == tp.shape:
                    taps += int(np.sum(np.abs(tp - prev[z]) > 0.5))
                prev[z] = tp
        out[n] = dict(e_v=_rms(ev), e_p=_rms(ep), v_min=vmin, v_max=vmax, n_sw=taps)
    return out


def write_table(m: Dict[str, Dict[str, float]], out: Path) -> None:
    rows = []
    for n in LADDER:
        if n not in m:
            continue
        d = m[n]
        bold = r"\textbf{%s}" % n if n == PROPOSED else r"\textsf{%s}" % n
        rows.append(
            f"\t\t{bold} & {d['e_v']:.5f} & {d['e_p']:.5f} & "
            f"{d['v_min']:.3f} & {d['v_max']:.3f} & {d['n_sw']:d} \\\\")
    body = "\n".join(rows)
    tex = (
        "% GENERATED by experiments/ch_10_case_study/ch_10_2_figures.py -- do not\n"
        "% hand-edit; regenerate instead.  Source: the single-run ladder in\n"
        "% results/THESIS_ch10_variants_single_run (validated weights, 360 min,\n"
        "% all six contingencies, ONE run per configuration).\n"
        "\\begin{table}[htb]\n\t\\centering\\small\n"
        "\t\\caption[Single-run metrics for the variant ladder.]{Single-run "
        "metrics over the \\SI{360}{\\minute} horizon. $\\bar{e}_{\\mathrm{v}}$ "
        "is the voltage-tracking error over each area's reference set, "
        "aggregated across areas; $\\bar{e}_{\\mathrm{v},p}$ the same on the "
        "pilot nodes alone; $v^{\\min}/v^{\\max}$ the extreme \\gls{TSO} bus "
        "voltages; $N_{\\mathrm{sw}}$ the machine-transformer tap operations. "
        "One run per variant: no confidence statement is implied.}\n"
        "\t\\label{tab:case3:ladder}\n"
        "\t\\renewcommand{\\arraystretch}{1.2}\n"
        "\t\\begin{tabular}{l c c c c c}\n\t\t\\toprule\n"
        "\t\t\\textbf{Variant} & $\\bar{e}_{\\mathrm{v}}$ / \\si{\\perunit} & "
        "$\\bar{e}_{\\mathrm{v},p}$ / \\si{\\perunit} & "
        "$v^{\\min}$ / \\si{\\perunit} & $v^{\\max}$ / \\si{\\perunit} & "
        "$N_{\\mathrm{sw}}$ \\\\\n\t\t\\midrule\n"
        f"{body}\n"
        "\t\t\\bottomrule\n\t\\end{tabular}\n\\end{table}\n"
    )
    p = out / "ch10_ladder_metrics.tex"
    p.write_text(tex, encoding="utf-8")
    print(f"  wrote {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(RESULT_DIR / "g_w_gen_1e9"))
    ap.add_argument("--out", default=str(THESIS_GFX))
    args = ap.parse_args()

    run_dir, out = Path(args.run_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logs = load(run_dir)
    print(f"read {len(logs)} variants from {run_dir}: {', '.join(logs)}")
    if not logs:
        raise SystemExit("no logs found")

    fig_voltage_tracking(logs, out)
    fig_iface_tracking(logs, out)
    m = metrics(logs)
    write_table(m, out)

    print(f"\n{'':>4}{'e_v':>10}{'e_v,p':>10}{'v_min':>8}{'v_max':>8}{'N_sw':>7}")
    for n in LADDER:
        if n in m:
            d = m[n]
            print(f"{n:>4}{d['e_v']:>10.5f}{d['e_p']:>10.5f}"
                  f"{d['v_min']:>8.3f}{d['v_max']:>8.3f}{d['n_sw']:>7d}")


if __name__ == "__main__":
    main()
