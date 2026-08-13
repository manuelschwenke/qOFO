"""PowerFactory replay construction at the plant boundary.

This module owns the PowerFactory-specific part of a closed-loop replay:

1. copy the runner's converged post-initialisation pandapower state,
2. remove controller objects from that copy and export a reproducible snapshot,
3. synchronize the layered PowerFactory model to the snapshot, and
4. construct a PowerFactoryPlant on the runner-owned mirror net.

The experiment runner and the pure replay-analysis helpers do not import or
manipulate PowerFactory objects.  This keeps the plant substitution boundary
explicit and one-directional.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional

from export.dynamic_snapshot import (
    dump_dynamic_snapshot,
    load_snapshot,
    verify_roundtrip,
)
from pf.pf_sync import (
    FULL_MODEL_VARIATION,
    WIND_REPLACE_VARIATION,
    SyncContext,
    sync_full,
)
from pf.plant import PowerFactoryPlant
from pf.screening import RMS_STEP_MS, RMS_STUDY_CASE
from pf.session import (
    DEFAULT_PROJECT_PATH,
    connect,
    deactivate_variations_except,
    ensure_variation,
    set_variation_active,
)


def controller_free_snapshot_net(net):
    """Deep-copy net and remove controller rows from only the copy."""

    snapshot_net = copy.deepcopy(net)
    if hasattr(snapshot_net, "controller") and len(snapshot_net.controller):
        snapshot_net.controller.drop(
            index=snapshot_net.controller.index,
            inplace=True,
        )
    return snapshot_net


def snapshot_solver_options(
    *,
    distributed_slack: bool,
    enforce_q_lims: bool,
) -> Dict[str, Any]:
    """Exact options used by the runner's final controller-free re-solve."""

    return {
        "run_control": False,
        "calculate_voltage_angles": True,
        "init": "auto",
        "max_iteration": 50,
        "distributed_slack": bool(distributed_slack),
        "enforce_q_lims": bool(enforce_q_lims),
    }


def _find_page(app, page_name: str):
    """Graphics-board page by name, or None."""
    board = app.GetFromStudyCase("SetDesktop")
    return next((o for o in board.GetContents()
                 if o.loc_name == page_name), None)


def write_gui_control_scripts(flag_path: Path, refresh_every: int = 1) -> None:
    """Write the live GUI-control helpers beside the sentinel.

    Three files, all read by the running plant each dispatch interval so a
    long run's plots can be steered without stopping it:

    * ``plots_off.bat`` / ``plots_on.bat`` -- create/delete ``flag_path``,
      which pauses/resumes the live *refresh* (window stays open).  ``%~dp0``
      makes each script act on its own folder wherever it is launched.
    * ``hide_gui.bat`` / ``show_gui.bat`` -- create/delete ``HIDE_GUI``, which
      hides the whole PowerFactory desktop AND skips the redraw.  This is the
      stronger off-switch (verified to speed the run up the most), toggled
      live both ways.  NB: PF may segfault at process *exit* once the window
      has been hidden -- harmless, results are written before exit.
    * ``GUI_EVERY`` -- a plain-text integer; the plot refreshes every Nth
      interval.  Pre-filled with the launch value; edit the number to retune
      the cadence live (a profile run's per-interval cost is dominated by the
      simulation, not the redraw, so 1--2 is usually fine for watching).
    """
    flag = flag_path.name
    folder = flag_path.parent
    (folder / "plots_off.bat").write_text(
        "@echo off\r\n"
        f'type nul > "%~dp0{flag}"\r\n'
        "echo Live plots PAUSED (window stays). Run plots_on.bat to resume.\r\n",
        encoding="ascii",
    )
    (folder / "plots_on.bat").write_text(
        "@echo off\r\n"
        f'if exist "%~dp0{flag}" del "%~dp0{flag}"\r\n'
        "echo Live plots RESUMED.\r\n",
        encoding="ascii",
    )
    (folder / "hide_gui.bat").write_text(
        "@echo off\r\n"
        'type nul > "%~dp0HIDE_GUI"\r\n'
        "echo PowerFactory window HIDDEN (max speed). "
        "Run show_gui.bat to bring it back.\r\n",
        encoding="ascii",
    )
    (folder / "show_gui.bat").write_text(
        "@echo off\r\n"
        'if exist "%~dp0HIDE_GUI" del "%~dp0HIDE_GUI"\r\n'
        "echo PowerFactory window SHOWN again.\r\n",
        encoding="ascii",
    )
    (folder / "GUI_EVERY").write_text(f"{max(1, int(refresh_every))}\r\n",
                                      encoding="ascii")


def show_desktop(app, page_name: str = "TS Bus Voltages (qOFO)"):
    """Make the PowerFactory desktop visible and front the given plot page.

    Returns the page object (or ``None``), so the caller can hand it to the
    plant for a redraw after every ``advance`` -- an API-driven ``ComSim``
    writes results but never asks the desktop to repaint, so without that
    the curves stay empty until the run ends.

    ``App.Show()`` takes ~20-30 s to bring the window up, and the desktop
    cannot paint while the engine is busy.  Call this *before* any heavy PF
    work (sync / ComInc), otherwise the window may not appear until long
    after -- which is why an earlier attempt to show it immediately before
    the RMS build looked like ``Show()`` had silently failed.

    ``App.Hide()`` is deliberately never called: it segfaults in this setup,
    and process exit frees the session anyway (2026-07-20 finding).
    """
    app.Show()
    board = app.GetFromStudyCase("SetDesktop")
    page = next((o for o in board.GetContents()
                 if o.loc_name == page_name), None)
    if page is None:
        print(f"  [warn] graphics page {page_name!r} not found; desktop "
              f"shown without fronting a page")
        return None
    page.Show()
    # Bind the simulation result file explicitly.  ``autoSearchResultFile``
    # finds it for an interactive run, but an API-driven ComSim does not
    # announce itself to the desktop.
    try:
        page.SetResults(app.GetFromStudyCase("ElmRes"))
    except Exception as exc:                       # noqa: BLE001
        print(f"  [warn] could not bind result file to the plot page: {exc}")
    print(f"  [gui] PowerFactory desktop visible; page {page_name!r} fronted")
    return page


def activate_full_layers(app) -> None:
    """Activate the established base -> wind_replace -> full stack."""

    deactivate_variations_except(app, keep=None)
    ensure_variation(app, WIND_REPLACE_VARIATION)
    set_variation_active(app, WIND_REPLACE_VARIATION, True)
    ensure_variation(app, FULL_MODEL_VARIATION)
    set_variation_active(app, FULL_MODEL_VARIATION, True)


@dataclass
class PowerFactoryReplayFactory:
    """One-shot plant_factory for run_multi_tso_dso.

    The instance retains the constructed plant, parsed snapshot and sync
    reports so the orchestration layer can harvest trajectories and persist
    provenance after the runner returns.
    """

    out_dir: Path
    project: str = DEFAULT_PROJECT_PATH
    study_case: str = RMS_STUDY_CASE
    on_missing_avr: str = "skip"
    snapshot_label: str = "gate_e_post_init"
    distributed_slack: bool = True
    enforce_q_lims: bool = True
    verify_snapshot: bool = True
    event_pool_slots: int = 1
    """One-shot EvtParam/EvtLod slots reserved per target before ComInc."""

    preallocate_profiles: bool = False
    """Reserve load and DER-active-power slots in addition to controls."""

    rms_step_ms: float = RMS_STEP_MS
    """RMS integration step [ms]; the SMALLEST step when ``adaptive_step``."""

    rms_step_max_ms: Optional[float] = None
    """Largest step [ms] with ``adaptive_step`` (default: ``RMS_STEP_MS``)."""

    adaptive_step: bool = False
    """PF automatic step-size adaptation.  Off reproduces every run before
    2026-08-06."""

    qv_deadband_at_contingency: Optional[float] = None
    """Dead-band x droop study only: Q(V) dead band [pu] installed on the RMS
    parks at the instant of the contingency; the run-up keeps the configured
    one.

    Lets two legs share config, controllers, static plant and RMS run-up
    exactly and differ only in what the local layer does after the
    disturbance.  Separating the legs via ``--tso-deadband`` instead does NOT
    work: that value also feeds the controllers and the static plant, so the
    closed loops diverge from t = 0 (measured 2026-08-06: 1.33e-2 pu of
    run-up divergence, against 6.7e-7 pu when the configured dead band
    matches).

    The STATIC plant is deliberately left alone -- in every other experiment
    both plants run Q(V) throughout -- so Gate E cannot certify such a run."""

    profile_delivery: str = "events"
    """``events`` (legacy) or pre-ComInc ``elmfile`` profile delivery.

    Online controller outputs remain one-shot events in both modes."""

    start_hidden: bool = True
    """Leave the PowerFactory desktop hidden at startup.

    The GUI *machinery* (result-bound plot pages, the live sentinels and the
    control scripts) is installed either way, so the window can be raised
    mid-run with ``show_gui.bat`` and hidden again with ``hide_gui.bat``.  This
    only decides the initial state.

    Default True: an engine-mode session is headless unless ``App.Show()`` is
    called, the redraw is the dominant per-interval cost once results
    accumulate, and ``App.Show()`` itself takes 20-30 s.  A run that nobody is
    watching should not pay any of that.  ``PowerFactoryPlant`` seeds its
    hidden-state bookkeeping from the pre-created ``HIDE_GUI`` sentinel, so the
    first ``show_gui.bat`` reveals the window rather than being a no-op."""

    show_gui: bool = False
    """Install the GUI machinery (plot pages, sentinels, control scripts).

    Note this no longer implies the window is *visible*: see ``start_hidden``.

    The engine-mode session normally runs headless.  ``App.Show()`` makes the
    desktop visible so its plot pages update live while ``ComSim`` advances --
    useful for watching an RMS run rather than only its exported CSVs.  The
    window belongs to this process and closes when it exits; ``App.Hide()`` is
    deliberately NOT called on teardown because it segfaults here (the session
    is freed by process exit anyway, 2026-07-20 finding)."""

    gui_page: str = "TS Bus Voltages (qOFO)"
    """Graphics-board page to bring to front when ``show_gui`` is set."""

    gui_off_flag: Any = None
    """Path to a sentinel file; if it appears mid-run the live refresh stops
    (the only way to disable the GUI on a running engine-mode session)."""

    gui_refresh_every: int = 1
    """Refresh the live plot every Nth dispatch interval (1 = every one)."""

    live_plot: bool = False
    """Open a Python-side live plot of the TS bus voltages during the run.

    Engine mode gives no live PowerFactory plot (see ``pf/live_view.py``),
    so this is the only way to watch an RMS run in progress."""

    app_handle: Any = None
    """Reuse an already-connected application instead of opening a session.

    Engine mode allows one session at a time -- a second ``connect()``
    terminates the first ("User session has been terminated").  So a caller
    that has already connected (e.g. to bring the GUI up early and let it
    paint while the static run proceeds) must hand its handle in rather than
    letting the factory make its own."""

    plant: Optional[PowerFactoryPlant] = field(default=None, init=False)
    app: Any = field(default=None, init=False)
    snapshot_path: Optional[Path] = field(default=None, init=False)
    snapshot_doc: Optional[Dict[str, Any]] = field(default=None, init=False)
    sync_context: Optional[SyncContext] = field(default=None, init=False)
    _gui_pages: Any = field(default_factory=list, init=False)
    _profile_playback_config: Optional[Dict[str, Any]] = field(
        default=None, init=False)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if int(self.event_pool_slots) < 1:
            raise ValueError("event_pool_slots must be >= 1")
        self.event_pool_slots = int(self.event_pool_slots)
        self.profile_delivery = str(self.profile_delivery).lower()
        if self.profile_delivery not in ("events", "elmfile"):
            raise ValueError(
                "profile_delivery must be 'events' or 'elmfile'")
        if self.profile_delivery == "elmfile" and self.preallocate_profiles:
            raise ValueError(
                "ElmFile delivery and profile-event preallocation are "
                "mutually exclusive")

    def configure_exogenous_profiles(
        self,
        profiles,
        *,
        start_time,
        dt_s: float,
        duration_s: float,
    ) -> None:
        """Receive the runner's already-clipped authoritative profile table."""

        if self.plant is not None:
            raise RuntimeError(
                "profiles must be configured before plant construction")
        if self.profile_delivery != "elmfile":
            return
        self._profile_playback_config = {
            "profiles": profiles.copy(deep=True),
            "start_time": start_time,
            "dt_s": float(dt_s),
            "duration_s": float(duration_s),
            "rms_step_s": float(RMS_STEP_MS) / 1000.0,
            "file_path": self.out_dir / "rms_profiles_elmfile.txt",
        }

    def __call__(self, net, *, meta, zone_map) -> PowerFactoryPlant:
        if self.plant is not None:
            raise RuntimeError(
                "PowerFactoryReplayFactory is one-shot; a second runner "
                "would require a fresh RMS initialisation"
            )

        if (self.profile_delivery == "elmfile"
                and self._profile_playback_config is None):
            raise RuntimeError(
                "ElmFile profile delivery was selected but the runner did "
                "not configure a profile trajectory")
        snapshot_net = controller_free_snapshot_net(net)
        options = snapshot_solver_options(
            distributed_slack=self.distributed_slack,
            enforce_q_lims=self.enforce_q_lims,
        )
        self.snapshot_path = dump_dynamic_snapshot(
            snapshot_net,
            meta,
            zone_map,
            self.snapshot_label,
            self.out_dir,
            solver_options=options,
            phase="full",
            notes=(
                "Phase 6 Gate E: exact runner post-initialisation state; "
                + ("known profiles delivered by pre-ComInc ElmFile sources"
                   if self._profile_playback_config is not None
                   else "fixed exogenous injections, no profiles")
                + "; no contingencies"
            ),
        )
        self.snapshot_doc = load_snapshot(self.snapshot_path)

        if self.verify_snapshot:
            roundtrip = verify_roundtrip(self.snapshot_path)
            (self.out_dir / "snapshot_roundtrip.txt").write_text(
                roundtrip.summary() + "\n",
                encoding="utf-8",
            )
            with (self.out_dir / "snapshot_roundtrip.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        "ok": roundtrip.ok,
                        "max_dev": roundtrip.max_dev,
                        "worst": roundtrip.worst,
                    },
                    handle,
                    indent=2,
                    allow_nan=False,
                )
            if not roundtrip.ok:
                raise RuntimeError(
                    "post-initialisation snapshot failed round-trip "
                    "verification; refusing to start RMS replay"
                )

        if self.app_handle is not None:
            self.app = self.app_handle
        else:
            self.app = connect(self.project, study_case=self.study_case)
            if self.show_gui and not self.start_hidden:
                show_desktop(self.app, self.gui_page)
        if self.show_gui:
            # Pre-create HIDE_GUI BEFORE the plant is constructed: the plant
            # seeds its hidden-state flag from this file, and its toggle only
            # fires when the wanted state differs from that seed.  Without it a
            # hidden-start run would believe the desktop is already up and
            # show_gui.bat would do nothing.
            if self.start_hidden and self.gui_off_flag is not None:
                _hide_flag = Path(self.gui_off_flag).parent / "HIDE_GUI"
                _hide_flag.write_text("", encoding="ascii")
            # Every qOFO page, not just the fronted one: each needs its own
            # result-file binding, and the plant refreshes all of them so a
            # tab switch mid-run shows current curves.
            from pf.plot_pages import find_pages
            pages = find_pages(self.app)
            self._gui_pages = pages
            res = self.app.GetFromStudyCase("ElmRes")
            for _pg in pages:
                try:
                    _pg.SetResults(res)
                except Exception as exc:           # noqa: BLE001
                    print(f"  [warn] SetResults failed on "
                          f"{_pg.loc_name!r}: {exc}")
            print(f"  [gui] result file bound on {len(pages)} page(s): "
                  f"{[p.loc_name for p in pages]}; ALL refresh every "
                  f"{self.gui_refresh_every} dispatch interval(s)")
            if self.gui_off_flag is not None:
                write_gui_control_scripts(Path(self.gui_off_flag),
                                          self.gui_refresh_every)
                if self.start_hidden:
                    print(f"  [gui] desktop HIDDEN at start (max speed). To "
                          f"watch this run, launch\n"
                          f"        "
                          f"{Path(self.gui_off_flag).parent / 'show_gui.bat'}\n"
                          f"        (hide_gui.bat puts it back; both act "
                          f"within one dispatch interval)")
        # A previous screening/replay calculation may still own events or
        # stale result state.  Release it before mutating synchronized inputs.
        self.app.ResetCalculation()
        activate_full_layers(self.app)
        self.sync_context = SyncContext(
            self.app,
            self.snapshot_doc,
            dry_run=False,
        )
        sync_full(self.sync_context)
        report = self.sync_context.report
        (self.out_dir / "powerfactory_sync.txt").write_text(
            report.summary() + "\n",
            encoding="utf-8",
        )
        with (self.out_dir / "powerfactory_sync.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    key: list(getattr(report, key))
                    for key in (
                        "created",
                        "renamed",
                        "updated",
                        "deleted",
                        "unchanged",
                    )
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

        live_view = None
        if self.live_plot:
            from pf.live_view import ts_bus_voltage_view
            live_view = ts_bus_voltage_view()
            print("  [live] TS bus voltage window open; it redraws once per "
                  "dispatch interval")

        self.plant = PowerFactoryPlant(
            self.snapshot_doc,
            app=self.app,
            mirror_net=net,
            on_missing_avr=self.on_missing_avr,
            gui_pages=self._gui_pages,
            gui_off_flag=self.gui_off_flag,
            gui_refresh_every=self.gui_refresh_every,
            live_view=live_view,
            event_pool_slots=self.event_pool_slots,
            preallocate_profiles=self.preallocate_profiles,
            qv_deadband_at_contingency=self.qv_deadband_at_contingency,
            rms_step_ms=self.rms_step_ms,
            rms_step_max_ms=self.rms_step_max_ms,
            adaptive_step=self.adaptive_step,
            profile_playback_config=self._profile_playback_config,
        )
        return self.plant
