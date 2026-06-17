#!/usr/bin/env python3
"""Plot the DSO-2 grid as a classic single-line diagram (Einlinienschaltbild).

Style reference: IEC / VDE standard single-line diagrams with
  - thick horizontal busbars
  - overlapping-circle transformer symbols
  - triangle load symbols
  - circle-with-~ generator symbols
  - km labels on lines
"""
import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches, matplotlib.lines as mlines
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D
from matplotlib import collections as mc

_here = os.path.dirname(os.path.abspath(__file__))
# Adjust this if needed:
_root = os.path.dirname(_here)
# os.chdir(_root); sys.path.insert(0, _root)

# ═══════════════════════════════════════════════════════════════════════════
# LAYOUT — bus positions  (x, y)  in arbitrary units
# Voltage tiers at different y-heights:
#   y ≈ 8.0   380 kV upper row  (EHV 1, 6, 4, 5)
#   y ≈ 6.0   380 kV lower row  (EHV 0, 2, 3)
#   y ≈ 3.5   110 kV upper row  (HV 3, 2, 1, 0, 7, 8)
#   y ≈ 1.5   110 kV lower row  (HV 4, 5, 6, 9)
# ═══════════════════════════════════════════════════════════════════════════
BUS_XY = {
    # 380 kV  — upper row
    1:  ( 1.0, 8.0),   # EHV 1
    6:  ( 5.0, 8.0),   # EHV 6 (slack / ext grid)
    4:  ( 9.0, 8.0),   # EHV 4
    5:  (13.0, 8.0),   # EHV 5
    # 380 kV — lower row
    0:  ( 1.0, 6.0),   # EHV 0
    2:  ( 4.0, 6.0),   # EHV 2
    3:  ( 9.0, 6.0),   # EHV 3
    # 15 kV generator terminal
    20: (10.5, 9.0),   # GEN terminal
    # 20 kV tertiary (drawn as small taps, not full busbars)
    17: ( 1.0, 4.8),   # Tertiary TN0-DN3
    18: ( 9.0, 4.8),   # Tertiary TN3-DN0
    19: (13.0, 4.8),   # Tertiary TN5-DN8
    # 110 kV — upper row
    10: ( 1.0, 3.5),   # HV 3  (DN Bus_3)
    12: ( 3.5, 3.5),   # HV 5  (DN Bus_5)
    9:  ( 5.5, 3.5),   # HV 2  (DN Bus_2)
    8:  ( 7.0, 3.5),   # HV 1  (DN Bus_1)
    7:  ( 9.0, 3.5),   # HV 0  (DN Bus_0)
    14: (11.0, 3.5),   # HV 7  (DN Bus_7)
    15: (13.0, 3.5),   # HV 8  (DN Bus_8)
    # 110 kV — lower row
    11: ( 1.0, 1.5),   # HV 4  (DN Bus_4)
    13: ( 4.0, 1.5),   # HV 6  (DN Bus_6)
    16: (11.0, 1.5),   # HV 9  (DN Bus_9)
}

# Busbar half-width (visual length of the thick bar)
BAR_HW = {
    380.0: 0.6,
    110.0: 0.5,
     20.0: 0.25,
     15.0: 0.2,
}

# ═══════════════════════════════════════════════════════════════════════════
# STYLE constants
# ═══════════════════════════════════════════════════════════════════════════
EHV_COLOR   = "#8B4513"   # brown for 380 kV busbars
HV_COLOR    = "#000000"   # black for 110 kV busbars
LINE_CLR    = "#000000"
TRAFO_CLR   = "#000000"
R_TRAFO     = 0.22        # radius of each transformer circle
FONT        = "serif"
BG_COLOR    = "#FFFFFF"


def _bar_color(vn_kv):
    if vn_kv >= 300:
        return EHV_COLOR
    return HV_COLOR


def _bar_lw(vn_kv):
    if vn_kv >= 300:
        return 5
    if vn_kv >= 100:
        return 4
    return 2.5


# ─── drawing primitives ────────────────────────────────────────────────────

def draw_busbar(ax, bus_id, vn_kv, label=None):
    """Thick horizontal bar."""
    x, y = BUS_XY[bus_id]
    hw = BAR_HW.get(vn_kv, 0.3)
    color = _bar_color(vn_kv)
    lw = _bar_lw(vn_kv)
    ax.plot([x - hw, x + hw], [y, y], color=color, lw=lw,
            solid_capstyle="butt", zorder=4)
    if label:
        col = EHV_COLOR if vn_kv >= 300 else HV_COLOR
        ax.text(x - hw - 0.1, y, label, ha="right", va="center",
                fontsize=7, fontfamily=FONT, color=col, zorder=5)


def draw_line(ax, bus_a, bus_b, length_km=None):
    """Simple line between two buses, with optional km label."""
    xa, ya = BUS_XY[bus_a]
    xb, yb = BUS_XY[bus_b]
    ax.plot([xa, xb], [ya, yb], color=LINE_CLR, lw=1.0, zorder=2)
    if length_km is not None:
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        txt = f"{length_km:.0f} km"
        angle = np.degrees(np.arctan2(yb - ya, xb - xa))
        offset_x, offset_y = 0, 0.15
        if abs(xb - xa) < 0.01:  # vertical
            offset_x, offset_y = 0.15, 0
            ax.text(mx + offset_x, my + offset_y, txt, ha="left", va="center",
                    fontsize=6, fontfamily=FONT, rotation=90, zorder=5)
        else:
            ax.text(mx + offset_x, my + offset_y, txt, ha="center", va="bottom",
                    fontsize=6, fontfamily=FONT, zorder=5)


def draw_trafo_2w(ax, x, y_top, y_bot, label=None):
    """Two overlapping circles for a 2-winding transformer."""
    r = R_TRAFO
    gap = r * 0.6
    cy_upper = (y_top + y_bot) / 2 + gap
    cy_lower = (y_top + y_bot) / 2 - gap
    # vertical leads
    ax.plot([x, x], [y_top, cy_upper + r], color=LINE_CLR, lw=1.0, zorder=2)
    ax.plot([x, x], [cy_lower - r, y_bot], color=LINE_CLR, lw=1.0, zorder=2)
    # circles
    c1 = Circle((x, cy_upper), r, fill=False, ec=TRAFO_CLR, lw=1.2, zorder=3)
    c2 = Circle((x, cy_lower), r, fill=False, ec=TRAFO_CLR, lw=1.2, zorder=3)
    ax.add_patch(c1)
    ax.add_patch(c2)
    if label:
        ax.text(x + r + 0.1, (cy_upper + cy_lower) / 2, label,
                ha="left", va="center", fontsize=7, fontfamily=FONT, zorder=5)


def draw_trafo_3w(ax, x, y_hv, y_mv, y_lv=None, label=None):
    """Three-winding transformer: two overlapping circles + optional tertiary tap."""
    r = R_TRAFO
    gap = r * 0.6
    cy_upper = (y_hv + y_mv) / 2 + gap
    cy_lower = (y_hv + y_mv) / 2 - gap
    # HV lead
    ax.plot([x, x], [y_hv, cy_upper + r], color=LINE_CLR, lw=1.0, zorder=2)
    # MV lead
    ax.plot([x, x], [cy_lower - r, y_mv], color=LINE_CLR, lw=1.0, zorder=2)
    # circles
    c1 = Circle((x, cy_upper), r, fill=False, ec=TRAFO_CLR, lw=1.2, zorder=3)
    c2 = Circle((x, cy_lower), r, fill=False, ec=TRAFO_CLR, lw=1.2, zorder=3)
    ax.add_patch(c1)
    ax.add_patch(c2)
    # tertiary tap (small line to the side)
    if y_lv is not None:
        tap_x = x + r * 0.7
        tap_y = (cy_upper + cy_lower) / 2
        ax.plot([x + r, tap_x + 0.15], [tap_y, tap_y], color=LINE_CLR, lw=1.0, zorder=2)
        ax.plot([tap_x + 0.15, tap_x + 0.15], [tap_y, y_lv], color=LINE_CLR, lw=1.0, zorder=2)
    if label:
        ax.text(x - r - 0.1, (cy_upper + cy_lower) / 2, label,
                ha="right", va="center", fontsize=7, fontfamily=FONT,
                fontweight="bold", zorder=5)


def draw_load(ax, x, y, offset_x=0.0, offset_y=-0.5):
    """Downward-pointing triangle for a load."""
    tx, ty = x + offset_x, y + offset_y
    size = 0.15
    tri = Polygon([
        (tx - size, ty),
        (tx + size, ty),
        (tx, ty - size * 1.3)
    ], closed=True, fill=False, ec=LINE_CLR, lw=1.0, zorder=3)
    ax.add_patch(tri)
    # lead from bus to triangle
    ax.plot([x + offset_x, tx], [y, ty], color=LINE_CLR, lw=0.8, zorder=2)


def draw_gen(ax, x, y, label="G\n~"):
    """Circle with ~ inside for a synchronous generator."""
    r = 0.25
    c = Circle((x, y), r, fill=True, fc="white", ec=TRAFO_CLR, lw=1.2, zorder=4)
    ax.add_patch(c)
    ax.text(x, y, label, ha="center", va="center", fontsize=6,
            fontfamily=FONT, zorder=5)


def draw_wind(ax, x, y, offset_x=0.0, offset_y=-0.5):
    """Wind DER symbol: circle with three-pronged rotor."""
    cx, cy = x + offset_x, y + offset_y
    r = 0.18
    c = Circle((cx, cy), r, fill=True, fc="white", ec=LINE_CLR, lw=1.0, zorder=4)
    ax.add_patch(c)
    # three rotor blades at 120° spacing
    for angle_deg in [90, 210, 330]:
        a = np.radians(angle_deg)
        ax.plot([cx, cx + r * 0.75 * np.cos(a)],
                [cy, cy + r * 0.75 * np.sin(a)],
                color=LINE_CLR, lw=1.0, zorder=5)
    # lead
    ax.plot([x + offset_x, cx], [y, cy + r], color=LINE_CLR, lw=0.8, zorder=2)


def draw_pv(ax, x, y, offset_x=0.2, offset_y=-0.5):
    """PV DER symbol: small rectangle with diagonal arrow."""
    cx, cy = x + offset_x, y + offset_y
    size = 0.15
    rect = plt.Rectangle((cx - size, cy - size), 2 * size, 2 * size,
                          fill=False, ec=LINE_CLR, lw=1.0, zorder=4)
    ax.add_patch(rect)
    # arrow (sunlight)
    ax.annotate("", xy=(cx + size * 0.6, cy + size * 0.6),
                xytext=(cx - size * 0.6, cy - size * 0.6),
                arrowprops=dict(arrowstyle="->", color=LINE_CLR, lw=0.8),
                zorder=5)
    # lead
    ax.plot([x + offset_x, cx], [y, cy + size], color=LINE_CLR, lw=0.8, zorder=2)


def draw_ext_grid(ax, x, y):
    """External grid / slack: cross-hatched rectangle above bus."""
    size = 0.3
    rect = plt.Rectangle((x - size, y + 0.3), 2 * size, 2 * size,
                          fill=False, ec=LINE_CLR, lw=1.2, zorder=4,
                          hatch="xxxx")
    ax.add_patch(rect)
    ax.plot([x, x], [y, y + 0.3], color=LINE_CLR, lw=1.0, zorder=2)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PLOT — hard-coded topology for DSO-2
# ═══════════════════════════════════════════════════════════════════════════

def plot_dso2_sld(net, *, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # ── 1) Draw busbars ────────────────────────────────────────────────────
    bus_labels = {}
    for idx, row in net.bus.iterrows():
        if idx not in BUS_XY:
            continue
        vn = float(row["vn_kv"])
        name = str(row["name"])
        # Build short label
        short = name.split("|")[-1].replace("Bus_", "")
        if vn >= 300:
            lbl = f"EHV {short}"
        elif vn >= 100:
            lbl = f"HV {short}"
        else:
            lbl = None  # tertiary / gen: no bar label
        bus_labels[idx] = lbl
        draw_busbar(ax, idx, vn, label=lbl)

    # ── 2) Draw lines with km labels ───────────────────────────────────────
    for _, row in net.line.iterrows():
        fb, tb = int(row.from_bus), int(row.to_bus)
        if fb not in BUS_XY or tb not in BUS_XY:
            continue
        length = float(row.length_km) if "length_km" in row.index else None
        draw_line(ax, fb, tb, length_km=length)

    # ── 3) Draw 3-winding transformers ─────────────────────────────────────
    for _, row in net.trafo3w.iterrows():
        hb, mb, lb = int(row.hv_bus), int(row.mv_bus), int(row.lv_bus)
        if hb not in BUS_XY or mb not in BUS_XY:
            continue
        x_hv, y_hv = BUS_XY[hb]
        x_mv, y_mv = BUS_XY[mb]
        y_lv = BUS_XY[lb][1] if lb in BUS_XY else None
        name = str(row.get("name", ""))
        short = name.split("|")[-1] if "|" in name else name
        # For 3W, HV and MV should be vertically aligned
        draw_trafo_3w(ax, x_hv, y_hv, y_mv, y_lv=y_lv, label=short[:6])

    # ── 4) Draw 2-winding transformers ─────────────────────────────────────
    for _, row in net.trafo.iterrows():
        hb, lb = int(row.hv_bus), int(row.lv_bus)
        if hb not in BUS_XY or lb not in BUS_XY:
            continue
        x_hv, y_hv = BUS_XY[hb]
        x_lv, y_lv = BUS_XY[lb]
        name = str(row.get("name", ""))
        draw_trafo_2w(ax, x_hv, y_hv, y_lv, label=name.split("|")[-1][:6])

    # ── 5) Draw loads ──────────────────────────────────────────────────────
    load_count = {}
    for _, row in net.load.iterrows():
        bus_idx = int(row.bus)
        if bus_idx not in BUS_XY:
            continue
        n = load_count.get(bus_idx, 0)
        load_count[bus_idx] = n + 1
        ox = -0.2 + n * 0.35
        draw_load(ax, *BUS_XY[bus_idx], offset_x=ox, offset_y=-0.55)

    # ── 6) Draw static generators (DER) ────────────────────────────────────
    sgen_count = {}
    for _, row in net.sgen.iterrows():
        bus_idx = int(row.bus)
        if bus_idx not in BUS_XY:
            continue
        n = sgen_count.get(bus_idx, 0)
        sgen_count[bus_idx] = n + 1
        stype = str(row.get("type", ""))
        ox = 0.2 + n * 0.4
        if stype in ("WP", "Wind", "wind"):
            draw_wind(ax, *BUS_XY[bus_idx], offset_x=ox, offset_y=-0.55)
        else:
            draw_pv(ax, *BUS_XY[bus_idx], offset_x=ox, offset_y=-0.55)

    # ── 7) Draw generators (gen) ───────────────────────────────────────────
    for _, row in net.gen.iterrows():
        bus_idx = int(row.bus)
        if bus_idx not in BUS_XY:
            continue
        x, y = BUS_XY[bus_idx]
        draw_gen(ax, x, y + 0.5, label="GS\n~")
        ax.plot([x, x], [y, y + 0.25], color=LINE_CLR, lw=1.0, zorder=2)

    # ── 8) Draw external grid ──────────────────────────────────────────────
    for _, row in net.ext_grid.iterrows():
        bus_idx = int(row.bus)
        if bus_idx not in BUS_XY:
            continue
        draw_ext_grid(ax, *BUS_XY[bus_idx])

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color=EHV_COLOR, lw=4, label="380 kV busbar"),
        Line2D([0], [0], color=HV_COLOR,  lw=3, label="110 kV busbar"),
        Line2D([0], [0], color=LINE_CLR,  lw=1, label="Transmission line"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="w",
               markeredgecolor=LINE_CLR, ms=10, label="Transformer"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="w",
               markeredgecolor=LINE_CLR, ms=8, label="Load"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8,
              framealpha=0.95, edgecolor="#ccc", fancybox=False)

    ax.set_xlim(-1, 15)
    ax.set_ylim(-0.5, 10.5)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="dso2_sld.png")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    args.save = os.path.join(_root, "experiments", "results",
                             "003_cigre_2026", "figures", "dso2_grid_sld.png")
    # ── Import your network builder ──
    sys.path.insert(0, os.path.dirname(_here))
    from network.build_tuda_net import build_tuda_net
    net, meta = build_tuda_net()

    plot_dso2_sld(net, save_path=args.save, show=not args.no_show)