"""
Shared visualisation styling (classicthesis look).

Used by the live windows in ``visualisation/`` (TS, STS, system
overview, SBX-V) and the post-run figures for a consistent appearance:

* Palatino-family serif fonts via rcParams (no LaTeX required).
* Optional ``text.usetex=True`` with a mathpazo + eulervm preamble
  that matches the classicthesis thesis template.
* TU Darmstadt PANTONE colour palette for data series, with the
  olive / dark-orange / dark-blue colours repurposed as header
  bar, measurements band, and actuators band colours.
* Left-aligned UPPERCASE subplot titles (``tile_title``).
* Coloured figure header (``draw_figure_header``) and section
  bands (``draw_section_band``).
* Window tiling on Qt5Agg across one or two screens
  (``slot_geometry`` / ``position_figure_in_slot``).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import matplotlib as mpl

from configs.color_config import TU_COLOURS, TU_PRIMARY

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

# TU Darmstadt PANTONE palette roles

#: TU 5c yellow-green primary header bar.
COLOUR_TITLE_BAR: str = TU_PRIMARY
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


def slot_geometry(
    slot_idx: int,
    *,
    layout: str = "dual_screen",
    n_slots: int = 3,
) -> Optional[Tuple[int, int, int, int]]:
    """Screen rectangle ``(x, y, w, h)`` of a slot, or ``None``.

    Layouts:

    * ``"thirds"`` — slots side by side on the primary screen, each
      ``1/n_slots`` of its width and the full height.
    * ``"dual_screen"`` — slots 0 and 1 share the primary screen half and
      half; slot 2 takes the full secondary screen.
    * ``"dual_screen_half"`` — as ``"dual_screen"``, but slot 2 takes the
      left half of the secondary screen.

    Both dual layouts fall back to ``"thirds"`` with a single screen.
    ``None`` on non-Qt backends, without PyQt5, or before a Qt application
    exists (e.g. on the headless ``Agg`` backend).
    """
    if "Qt" not in mpl.get_backend():
        return None
    try:
        from PyQt5.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError):
        return None
    try:
        app = QApplication.instance()
        if app is None:
            return None
        screens = app.screens()
        primary = app.primaryScreen()
        pg = primary.availableGeometry()
        if layout in ("dual_screen", "dual_screen_half") and len(screens) >= 2:
            if slot_idx in (0, 1):
                half_w = pg.width() // 2
                x = pg.x() + slot_idx * half_w
                w = half_w if slot_idx == 0 else pg.width() - half_w
                return x, pg.y(), w, pg.height()
            secondary = next((s for s in screens if s is not primary), primary)
            sg = secondary.availableGeometry()
            w = sg.width() // 2 if layout == "dual_screen_half" else sg.width()
            return sg.x(), sg.y(), w, sg.height()
        w_slot = pg.width() // n_slots
        return pg.x() + slot_idx * w_slot, pg.y(), w_slot, pg.height()
    except Exception:
        return None


def position_figure_in_slot(
    fig: Figure,
    slot_idx: int,
    *,
    layout: str = "dual_screen",
    n_slots: int = 3,
) -> None:
    """Place ``fig`` in the requested slot of the chosen layout.

    ``slot_idx`` 0, 1, 2 are the TS, STS and overview windows; see
    :func:`slot_geometry` for the layouts.  Silently no-ops on non-Qt
    backends or when ``PyQt5`` is unavailable.
    """
    geom = slot_geometry(slot_idx, layout=layout, n_slots=n_slots)
    if geom is None:
        return
    try:
        fig.canvas.manager.window.setGeometry(*geom)
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


