"""
visualisation/live_base.py
==========================
Shared machinery of the three live windows (:mod:`visualisation.live_ts`,
:mod:`visualisation.live_sts`, :mod:`visualisation.live_overview`).

Cost model (why this replaces the per-step ``ax.clear()`` plotters):

* Samples go into :class:`Trace` buffers -- preallocated numpy arrays with
  amortised O(1) append, read back as zero-copy views.
* Every artist is created once.  A redraw only hands the views to
  ``Line2D.set_data`` / ``FillBetweenPolyCollection.set_data`` and sets the
  axis limits from the buffers, so its Python cost does not grow with the
  number of tiles' history and nothing is re-laid out.
* Titles and axis labels sit at fixed positions and tick labels use plain
  formatters.  Matplotlib otherwise measures every tick label several times
  per frame to place them, which was most of the frame time (profiled
  2026-11-02).  Axes do not share x; the limits are set on each per frame.
* Redraws are throttled by wall clock: the next redraw waits at least
  ``refresh_s`` and at least ``last draw time / DRAW_SHARE``, so a window
  never spends more than ``DRAW_SHARE`` of the wall time drawing, however
  slow its canvas.  Between redraws a step costs the buffer appends plus one
  Qt event flush.  :meth:`finalize` forces the last frame.

The windows only read :class:`MultiTSOIterationRecord` fields; they hold no
reference to the plant or the controllers.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from visualisation.style import (  # noqa: F401  (style sets the Qt backend)
    COLOUR_ACT_BAND,
    COLOUR_MEAS_BAND,
    TITLE_BAR_HEIGHT_FRAC,
    _c,
    apply_serif_style,
    draw_figure_header,
    fill_section_band,
    position_figure_in_slot,
    raise_figure_to_front,
)

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
from matplotlib.transforms import offset_copy

if TYPE_CHECKING:
    from experiments.helpers import ContingencyEvent, MultiTSOIterationRecord

#: Upper bound on the share of wall time one window spends drawing.
DRAW_SHARE = 0.05

#: Muted colours for reference lines, limits and placeholder text.
REF_GREY = "0.35"
LIMIT_RED = "#B90F22"
EMPTY_GREY = "0.55"


# ─── Buffers ────────────────────────────────────────────────────────────────


class Trace:
    """Append-only time series ``(t, y)`` with ``y`` of shape ``(n, width)``.

    Rows shorter than ``width`` are NaN-padded, longer ones truncated, so a
    vector whose length changes mid-run (an element switched out) cannot
    break the buffer.  ``t`` and ``y`` are views valid until the next append.
    """

    __slots__ = ("_t", "_y", "n", "width")

    def __init__(self, width: int = 1, capacity: int = 512) -> None:
        self.width = max(1, int(width))
        self._t = np.empty(capacity)
        self._y = np.full((capacity, self.width), np.nan)
        self.n = 0

    def append(self, t: float, values) -> None:
        if self.n == self._t.size:
            cap = 2 * self._t.size
            self._t = np.resize(self._t, cap)
            y = np.full((cap, self.width), np.nan)
            y[:self.n] = self._y[:self.n]
            self._y = y
        self._t[self.n] = t
        v = np.asarray(values, dtype=float).ravel()
        k = min(v.size, self.width)
        self._y[self.n, :k] = v[:k]
        self.n += 1

    @property
    def t(self) -> np.ndarray:
        return self._t[:self.n]

    @property
    def y(self) -> np.ndarray:
        return self._y[:self.n]

    def col(self, k: int) -> np.ndarray:
        return self._y[:self.n, k]

    @property
    def last(self) -> np.ndarray:
        return self._y[self.n - 1] if self.n else np.full(self.width, np.nan)


def finite_range(*arrays: np.ndarray) -> Optional[Tuple[float, float]]:
    """``(min, max)`` over the finite entries of all arrays, else ``None``."""
    lo, hi = np.inf, -np.inf
    for a in arrays:
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            continue
        f = a[np.isfinite(a)]
        if f.size:
            lo = min(lo, float(f.min()))
            hi = max(hi, float(f.max()))
    return (lo, hi) if np.isfinite(lo) else None


def set_ylim_from(ax, *arrays: np.ndarray, pad: float = 0.08,
                  min_span: float = 1e-3, include: Sequence[float] = ()) -> None:
    """Fit the y-axis to the finite data (plus ``include`` values)."""
    rng = finite_range(*arrays, np.asarray(include, dtype=float))
    if rng is None:
        return
    lo, hi = rng
    span = max(hi - lo, min_span)
    mid = 0.5 * (lo + hi)
    lo, hi = mid - span / 2, mid + span / 2
    ax.set_ylim(lo - pad * span, hi + pad * span)


def set_band(coll, t: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> None:
    """Update a ``fill_between`` collection in place.

    Finite bands get their polygon assigned directly; ``set_data``
    (matplotlib ≥ 3.10), which splits at NaN gaps, is several times slower
    and only used when a gap exists.
    """
    if not t.size:
        return
    if np.isfinite(lo).all() and np.isfinite(hi).all():
        coll.set_verts([np.concatenate([np.column_stack((t, lo)),
                                        np.column_stack((t[::-1], hi[::-1]))])])
    else:
        coll.set_data(t, lo, hi)


def event_label(ev: "ContingencyEvent") -> str:
    """Short marker text for a contingency, e.g. ``G2 trip`` / ``Ld b11 +``."""
    act = {"trip": "trip", "restore": "rest.", "connect": "+",
           "setpoint_change": "set"}.get(ev.action, ev.action)
    kind = ev.element_type
    if kind == "load":
        what = ev.name or (f"Ld b{ev.bus}" if ev.bus is not None else f"Ld{ev.element_index}")
    else:
        what = {"gen": "G", "line": "L", "ext_grid": "X"}.get(kind, kind) + str(ev.element_index)
    return f"{what} {act}"


# ─── Window base ────────────────────────────────────────────────────────────


class LiveWindow:
    """One live figure: header, throttled redraws, shared time axis.

    Subclasses build their axes in ``__init__`` (after calling this one),
    register every time axis with :meth:`_time_axes`, implement
    :meth:`_ingest` (append one record's samples) and :meth:`_render`
    (push buffers into the artists).
    """

    def __init__(
        self,
        title: str,
        *,
        figsize: Tuple[float, float],
        slot_idx: int,
        layout: str,
        use_tex: bool,
        refresh_s: float,
        events: Iterable["ContingencyEvent"] = (),
    ) -> None:
        apply_serif_style(use_tex=use_tex)
        plt.ion()
        self._refresh_s = max(0.0, float(refresh_s))
        self._next_draw = -np.inf
        self._wall0 = time.perf_counter()
        self._n_updates = 0
        self._t_now = np.nan
        self._t0 = np.nan
        self._axes_t: List[plt.Axes] = []
        self._events = [(ev.effective_time_s / 60.0, event_label(ev)) for ev in events]

        self.fig = plt.figure(figsize=figsize, facecolor="white")
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        draw_figure_header(self.fig, title)
        self._status = self.fig.text(
            0.988, 1.0 - TITLE_BAR_HEIGHT_FRAC / 2.0, "", color="white",
            fontsize=9, ha="right", va="center", zorder=6,
        )
        self._slot_idx = slot_idx
        self._layout = layout

    # -- construction helpers -------------------------------------------------

    def _show(self) -> None:
        """Place the window and bring it up (call at the end of ``__init__``)."""
        position_figure_in_slot(self.fig, self._slot_idx, layout=self._layout, n_slots=3)
        plt.pause(0.01)
        raise_figure_to_front(self.fig)

    def _time_axes(self, axes: Sequence[plt.Axes], *, bottom: Sequence[plt.Axes]) -> None:
        """Register time axes: light styling, event markers, x labels only on
        the ``bottom`` axes of each column."""
        for ax in axes:
            self._axes_t.append(ax)
            style_axes(ax)
            for t_ev, _ in self._events:
                ax.axvline(t_ev, color=REF_GREY, lw=0.6, ls=(0, (2, 3)), alpha=0.6, zorder=0.5)
            if ax not in bottom:
                ax.tick_params(labelbottom=False)
        for ax in bottom:
            ax.set_xlabel("Time / min")

    def _label_events(self, axes: Sequence[plt.Axes]) -> None:
        """Write the event names along the top edge of ``axes``."""
        for ax in axes:
            for t_ev, text in self._events:
                ax.annotate(
                    text, (t_ev, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(2, -2), textcoords="offset points",
                    fontsize=6, color=REF_GREY, ha="left", va="top",
                    rotation=90, annotation_clip=True,
                )

    # -- public API -------------------------------------------------------------

    def update(self, rec: "MultiTSOIterationRecord") -> None:
        """Ingest one record; redraw if ``refresh_s`` wall time has passed."""
        self._n_updates += 1
        t = rec.time_s / 60.0
        if not np.isfinite(self._t0):
            self._t0 = t
        self._t_now = t
        self._ingest(rec, t)
        start = time.perf_counter()
        if start >= self._next_draw:
            self._draw()
            end = time.perf_counter()
            self._next_draw = end + max(self._refresh_s, (end - start) / DRAW_SHARE)
        elif plt.fignum_exists(self.fig.number):
            self.fig.canvas.flush_events()

    def finalize(self) -> None:
        """Draw the final state regardless of the refresh throttle."""
        self._draw()

    def save(self, path, **kwargs) -> None:
        self._render_all()
        self.fig.savefig(path, **kwargs)

    # -- internals ----------------------------------------------------------------

    def _draw(self) -> None:
        if not plt.fignum_exists(self.fig.number):
            return
        self._render_all()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _render_all(self) -> None:
        self._render()
        if self._axes_t and np.isfinite(self._t0):
            t1 = max(self._t_now, self._t0 + 1.0)
            spacing = _tick_spacing_min(t1 - self._t0)
            for ax in self._axes_t:
                ax.set_xlim(self._t0, t1)
                ax.xaxis.set_major_locator(MultipleLocator(spacing))
        wall = time.perf_counter() - self._wall0
        rate = self._n_updates / wall if wall > 0 else 0.0
        self._status.set_text(f"t = {self._t_now:.0f} min   ·   {rate:.1f} steps/s")

    def _ingest(self, rec: "MultiTSOIterationRecord", t: float) -> None:
        raise NotImplementedError

    def _render(self) -> None:
        raise NotImplementedError


def show_until_closed() -> None:
    """Keep the live windows usable after the run: run the GUI event loop
    until every open figure is closed (``plt.ioff(); plt.show(block=True)``).

    During a run the windows are pumped only by :meth:`LiveWindow.update`;
    after the last step nothing does.  In PyCharm's Python console the idle
    console did not take over either (run 0025, 2026-11-06: console idle,
    window not responding for 96 s after the records were written), so the
    windows froze.  Blocking here works the same in the console, under Run
    and in plain ``python``; the prompt returns once the windows are closed.

    No-op without open figures or on a non-interactive backend (``Agg`` in
    tests and batch runs).
    """
    from matplotlib.backends import BackendFilter, backend_registry

    interactive = backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    if not plt.get_fignums() or plt.get_backend().lower() not in interactive:
        return
    plt.ioff()
    plt.show(block=True)


def _tick_spacing_min(duration_min: float) -> float:
    for limit, spacing in ((600, 120), (240, 60), (120, 30), (60, 15), (20, 5)):
        if duration_min > limit:
            return spacing
    return 2


def _plain(v: float, _pos=None) -> str:
    return f"{v:g}"


def style_axes(ax: plt.Axes) -> None:
    """Open, light axes: no top/right spines, faint grid, small ticks; plain
    tick formatters and axis labels at fixed point offsets (no per-frame
    tick-label measurement)."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.6")
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(axis="both", labelsize=7.5, length=2.5, width=0.6, colors="0.25")
    ax.grid(True, color="0.85", lw=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FuncFormatter(_plain))
    fig = ax.get_figure()
    ax.yaxis.set_label_coords(0.0, 0.5, transform=offset_copy(
        ax.transAxes, fig=fig, x=-30, units="points"))
    ax.xaxis.set_label_coords(0.5, 0.0, transform=offset_copy(
        ax.transAxes, fig=fig, y=-14, units="points"))


def tile_title(ax: plt.Axes, text: str) -> None:
    """Left-aligned UPPERCASE title at a fixed height (``y`` given, so
    matplotlib skips its per-frame title placement)."""
    ax.set_title(text.upper(), loc="left", pad=4, fontweight="bold", fontsize=8.5, y=1.0)


def empty_note(ax: plt.Axes, text: str):
    """Muted centred placeholder; returned so it can be hidden once data arrive."""
    return ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center",
                   va="center", color=EMPTY_GREY, fontsize=8, style="italic")


def value_tag(ax: plt.Axes):
    """Right-aligned title slot for the latest values of a tile."""
    return ax.set_title("", loc="right", fontsize=7, color="0.3", pad=4, y=1.0)


def proxy(color: str, *, ls: str = "-", lw: float = 1.4, label: str = "",
          marker: Optional[str] = None, alpha: float = 1.0) -> plt.Line2D:
    return plt.Line2D([0], [0], color=color, ls=ls, lw=lw, label=label,
                      marker=marker, alpha=alpha)


def patch(color: str, label: str, alpha: float = 0.6, hatch: Optional[str] = None):
    from matplotlib.patches import Patch
    return Patch(facecolor=color, edgecolor=color if hatch else "none",
                 alpha=alpha, label=label, hatch=hatch, lw=0.0)


def legend(
    ax: plt.Axes,
    handles,
    *,
    ncol: Optional[int] = None,
    loc: str = "upper left",
    fontsize: float = 6.5,
    columnspacing: float = 1.0,
    handlelength: float = 1.6,
    handletextpad: float = 0.8,
) -> None:
    if not handles:
        return

    ax.legend(
        handles=handles,
        loc=loc,
        fontsize=fontsize,
        frameon=True,
        framealpha=0.75,
        edgecolor="none",
        ncol=ncol or min(len(handles), 4),
        handlelength=handlelength,
        columnspacing=columnspacing,
        handletextpad=handletextpad,
        borderaxespad=0.3,
    )


__all__ = [
    "COLOUR_ACT_BAND", "COLOUR_MEAS_BAND", "EMPTY_GREY", "LIMIT_RED",
    "LiveWindow", "REF_GREY", "Trace", "_c", "empty_note", "event_label",
    "fill_section_band", "finite_range", "legend", "patch", "proxy",
    "set_band", "set_ylim_from", "show_until_closed", "style_axes",
    "tile_title", "value_tag",
]
