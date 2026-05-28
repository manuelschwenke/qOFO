"""
Shared visualisation styling (classicthesis look).

Used by all live-plotter modules in ``visualisation/`` to give the
MULTI-TSO CONTROLLER, CASCADE-DSO CONTROLLER, and SYSTEM POWER FLOW
figures a consistent appearance:

* Palatino-family serif fonts via rcParams (no LaTeX required).
* Optional ``text.usetex=True`` with a mathpazo + eulervm preamble
  that matches the classicthesis thesis template.
* TU Darmstadt PANTONE colour palette for data series, with the
  olive / dark-orange / dark-blue colours repurposed as header
  bar, measurements band, and actuators band colours.
* Left-aligned UPPERCASE subplot titles (``tile_title``).
* Coloured figure header (``draw_figure_header``) and section
  bands (``draw_section_band``).
* Three-pane side-by-side window tiling on Qt5Agg
  (``position_figure_in_slot``).
"""
from __future__ import annotations

import os
from typing import List

import matplotlib as mpl

# Force Qt5Agg as the default backend for all live plots.  Use
# ``force=True`` so the switch takes effect even if matplotlib.pyplot
# was already imported by an earlier module (imports from this file
# typically arrive in the middle of another module's imports).  Respect
# an explicit MPLBACKEND override so headless tests (MPLBACKEND=Agg)
# still work.
if not os.environ.get("MPLBACKEND"):
    os.environ["QT_API"] = "pyqt5"
    try:
        mpl.use("Qt5Agg", force=True)
    except (ImportError, ValueError):
        pass  # PyQt5 missing or backend unavailable — fall back silently.

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator


# ─── TU Darmstadt PANTONE palette ───────────────────────────────────────────

#: Ordered colour sequence for all data series
#: (5c, 1c, 8c, 6c, 3c, 10c, 4c, 2c, 9c, 7c, 11c).
TU_COLOURS: List[str] = [
    "#B1BD00",  # 0  –  5c  Yellow-green   (PANTONE 390)
    "#004E8A",  # 1  –  1c  Dark blue      (PANTONE 2945)
    "#CC4C03",  # 2  –  8c  Dark orange    (PANTONE 173)
    "#D7AC00",  # 3  –  6c  Gold           (PANTONE 110)
    "#008877",  # 4  –  3c  Teal           (PANTONE 3285)
    "#951169",  # 5  –  10c Magenta        (PANTONE 249)
    "#7FAB16",  # 6  –  4c  Olive green    (PANTONE 376)
    "#00689D",  # 7  –  2c  Mid blue       (PANTONE 3015)
    "#B90F22",  # 8  –  9c  Red            (PANTONE 193)
    "#D28700",  # 9  –  7c  Amber          (PANTONE 124)
    "#611C73",  # 10 –  11c Purple         (PANTONE 268)
]

#: Olive header bar.
COLOUR_TITLE_BAR: str = TU_COLOURS[0]
#: Dark-orange band for MEASUREMENTS sections.
COLOUR_MEAS_BAND: str = TU_COLOURS[2]
#: Dark-blue band for ACTUATORS sections.
COLOUR_ACT_BAND:  str = TU_COLOURS[1]


def _c(index: int) -> str:
    """Return the TU Darmstadt colour for a zero-based series index."""
    return TU_COLOURS[index % len(TU_COLOURS)]


# ─── classicthesis font setup ───────────────────────────────────────────────

#: LaTeX preamble matching the classicthesis thesis template
#: (Palatino text, Euler maths).
LATEX_PREAMBLE: str = "\n".join([
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{mathpazo}",
    r"\usepackage[scaled=0.95]{helvet}",
    r"\usepackage[varg]{txfonts}",
    r"\usepackage{eulervm}",
])


def apply_serif_style(use_tex: bool = False) -> None:
    """Apply the classicthesis-style rcParams.

    Idempotent.  Safe to call from every plotter's ``__init__``.

    Parameters
    ----------
    use_tex : bool
        When ``True``, enable ``text.usetex`` with the classicthesis
        mathpazo + eulervm preamble.  Requires a working LaTeX install
        and noticeably slows every redraw, so the default is ``False``.
        In serif-only mode the font is chosen from a Palatino-family
        fallback chain; maths is rendered by matplotlib's own mathtext.
    """
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Palatino Linotype", "Palatino", "Book Antiqua",
            "URW Palladio L", "DejaVu Serif",
        ],
        "font.size":         9.0,
        "axes.titlesize":    10.0,
        "axes.titleweight":  "bold",
        "axes.labelsize":    9.0,
        "xtick.labelsize":   8.0,
        "ytick.labelsize":   8.0,
        "legend.fontsize":   7.0,
        "mathtext.fontset":  "cm",
        "mathtext.rm":       "serif",
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linewidth":    0.6,
        "lines.linewidth":   1.1,
    })
    if use_tex:
        mpl.rcParams.update({
            "text.usetex":          True,
            "text.latex.preamble":  LATEX_PREAMBLE,
        })
    else:
        mpl.rcParams["text.usetex"] = False


# ─── Figure decoration (header bar, section bands, tile titles) ─────────────

#: Figure-fraction height of the main title bar.
TITLE_BAR_HEIGHT_FRAC: float = 0.035


def draw_figure_header(
    fig: Figure,
    title: str,
    color: str = COLOUR_TITLE_BAR,
    *,
    height_frac: float = TITLE_BAR_HEIGHT_FRAC,
) -> None:
    """Draw an olive header bar with left-aligned UPPERCASE title.

    Caller is expected to have reserved space for the bar, either by
    using an empty GridSpec spacer row for it or by calling
    :func:`plt.subplots_adjust(top=1 - height_frac - 0.005)`.
    """
    rect = Rectangle(
        (0.0, 1.0 - height_frac), 1.0, height_frac,
        transform=fig.transFigure, facecolor=color, edgecolor="none",
        zorder=5, clip_on=False,
    )
    fig.patches.append(rect)
    fig.text(
        0.012, 1.0 - height_frac / 2.0, title.upper(),
        color="white", fontsize=11.5, fontweight="bold",
        ha="left", va="center", zorder=6,
    )


def fill_section_band(ax: plt.Axes, label: str, color: str) -> None:
    """Turn a spacer GridSpec axes into a coloured section band.

    The axes keeps its GridSpec slot (so it participates in layout) but
    renders only a solid-coloured background with a left-aligned
    UPPERCASE label in white.  Plotters create a thin spacer row in
    their GridSpec (``height_ratios`` entry ~0.15) and hand that axes
    to this function.
    """
    ax.set_facecolor(color)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.text(
        0.012, 0.5, label.upper(),
        color="white", fontsize=9.5, fontweight="bold",
        transform=ax.transAxes, ha="left", va="center", zorder=100
    )


def tile_title(ax: plt.Axes, text: str) -> None:
    """Set a left-aligned UPPERCASE subplot title."""
    ax.set_title(text.upper(), loc="left", pad=4, fontweight="bold")


# ─── Adaptive x-axis tick formatting ────────────────────────────────────────


def apply_x_fmt(ax: plt.Axes, sub_minute: bool = False) -> None:
    """Minute-resolution tick formatter that adapts to the time range."""
    x_min, x_max = ax.get_xlim()
    duration_min = (x_max - x_min) / 60.0 if sub_minute else (x_max - x_min)

    if   duration_min > 240: spacing_min = 60
    elif duration_min > 120: spacing_min = 30
    elif duration_min >  60: spacing_min = 15
    else:                    spacing_min = 3

    if sub_minute:
        ax.xaxis.set_major_locator(MultipleLocator(spacing_min * 60))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _pos: f"{int(round(x / 60))}")
        )
    else:
        ax.xaxis.set_major_locator(MultipleLocator(spacing_min))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _pos: f"{int(round(x))}")
        )


# ─── Window placement (three-pane side-by-side tiling) ──────────────────────


def position_figure_in_slot(
    fig: Figure,
    slot_idx: int,
    *,
    layout: str = "dual_screen",
    n_slots: int = 3,
) -> None:
    """Place ``fig`` in the requested slot of the chosen layout.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure whose window will be moved.
    slot_idx : int
        Which slot: 0, 1 or 2 for the three live plotters in order
        (TSO controller · Cascade-DSO controller · System power flow).
    layout : {"thirds", "dual_screen"}
        * ``"thirds"`` — three panes side-by-side across the primary
          screen, each spanning ``1/n_slots`` of the screen width and
          the full available height.
        * ``"dual_screen"`` — slots 0 and 1 share the primary screen
          half-and-half; slot 2 takes the full secondary screen.  If
          only one screen is available, silently falls back to
          ``"thirds"``.
    n_slots : int
        Only used by the ``"thirds"`` layout.

    Silently no-ops on non-Qt backends or when ``PyQt5`` is unavailable
    (e.g. on the ``Agg`` headless backend).
    """
    backend = mpl.get_backend()
    if "Qt" not in backend:
        return
    try:
        from PyQt5.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError):
        return
    try:
        app = QApplication.instance()
        if app is None:
            return
        screens = app.screens()
        primary = app.primaryScreen()
        primary_geom = primary.availableGeometry()

        if layout == "dual_screen" and len(screens) >= 2:
            # Find a secondary screen that differs from the primary.
            secondary = next(
                (s for s in screens if s is not primary), primary
            )
            sec_geom = secondary.availableGeometry()
            if slot_idx == 0:
                # Primary screen, left half
                x = primary_geom.x()
                y = primary_geom.y()
                w = primary_geom.width() // 2
                h = primary_geom.height()
            elif slot_idx == 1:
                # Primary screen, right half
                half_w = primary_geom.width() // 2
                x = primary_geom.x() + half_w
                y = primary_geom.y()
                w = primary_geom.width() - half_w
                h = primary_geom.height()
            else:
                # Secondary screen, full
                x = sec_geom.x()
                y = sec_geom.y()
                w = sec_geom.width()
                h = sec_geom.height()
            fig.canvas.manager.window.setGeometry(x, y, w, h)
            return

        # Default "thirds" layout on primary screen
        x0, y0 = primary_geom.x(), primary_geom.y()
        w_slot = primary_geom.width() // n_slots
        h_slot = primary_geom.height()
        x_slot = x0 + slot_idx * w_slot
        fig.canvas.manager.window.setGeometry(x_slot, y0, w_slot, h_slot)
    except Exception:
        return


def raise_figure_to_front(fig: Figure) -> None:
    """Raise the figure's window to the top of the Z-order and focus it.

    On Windows this requires a brief ``WindowStaysOnTopHint`` flip,
    otherwise the taskbar merely flashes rather than focusing the
    window.  Silent no-op on non-Qt backends.
    """
    backend = mpl.get_backend()
    if "Qt" not in backend:
        return
    try:
        from PyQt5 import QtCore
    except (ImportError, ModuleNotFoundError):
        return
    try:
        window = fig.canvas.manager.window
        window.setWindowState(
            (window.windowState() & ~QtCore.Qt.WindowMinimized)
            | QtCore.Qt.WindowActive
        )
        window.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        window.show()
        window.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, False)
        window.show()
        window.raise_()
        window.activateWindow()
    except Exception:
        return


