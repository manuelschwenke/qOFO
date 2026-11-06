# 2026-11-06 — Interface-Q spikes at 452 min: PV capability edge at P/Sₙ = 0.1; live windows after the run

**Questions (author, 2026-11-06).**
1. Random large spikes in the STS interface Q between 420 and 480 min of the last run
   (0023, 2026-11-05 16:23; e.g. DSO_4 T10).
2. The live windows freeze as soon as a `run_multi_system_ofo.py` run ends.

**Code changed for question 2 only** (see "Change, 2026-11-06" below); question 1 is diagnosis only.

## 1. Spikes (established from run 0023 records)

Run 0023: `make_config()` (uncommitted: start 2016-05-01 10:00, load ±350 Mvar at bus 9
90–330 min, line 25 out 300–420 min), `trial_061d_v4_run0008`, frozen doubled-DSO_3
network, γ_OLTC,Q = 0, `sts_coupler_hold_after_bank` = 1.

- **Event-driven spikes** at 90 and 330 min (bus-9 load plus MSC commits) and 30 min (G4 trip).
  These are expected. The metered sample of that instant is taken after the event and before
  the controllers act.
- **The spike at 451.67–452.67 min (17:32 local time)** has no event, no commit and no tap
  or bank move. It hits DSO_1, DSO_2, DSO_3 and DSO_4. Examples:
  - DSO_4 T10, metered: −32.3 → −41.4 Mvar
  - DSO_1 T1, metered: −50.6 → −62.1 Mvar
  - DSO_3 T7, post-control: −20.0 → −9.6 Mvar

**Cause.**
- Each STS has four PV units. Their S_n is 80 / 70 / 50 / 40 MVA, doubled in DSO_3. All of them
  follow one PV profile, so they cross P/Sₙ = 0.1 in the same minute of the evening ramp-down.
- The modelled VDE-AR-N 4120 v2 diagram is discontinuous there:
  - Q ∈ [−0.10, +0.10]·Sₙ at P/Sₙ = 0.1;
  - Q = 0 below 0.1.
  - Defined in `core/actuator_bounds.py` and mirrored in the plant, `der_qv_local_loop._qv_capability`.
- The PV units sat at these ±0.1·Sₙ rails. Before the edge:
  - DSO_1: −8.1, −7.0, −5.1, −4.0 Mvar (all absorbing)
  - DSO_3: +16.2, −14.2, +10.2, −8.2 Mvar (alternating rails: 49 Mvar moved for a net +4 Mvar)
  - DSO_4: −8.1, −7.1, +5.1, −4.0 Mvar
- At the edge the input bound collapses to [0, 0]. The step sets q_set = 0, so each unit's Q is
  released in one step.
- The release is staggered unit by unit, because the DSO computes the bounds from the
  **metered** P (`DSOController._extract_der_active_power` → `measurement.der_p_mw`).
  - DSO_3 unit 3 is cut at step 1355 with plant P = 16.05 MW, which is still above 0.1·160.
  - The plant clip uses true P (`res_sgen.p_mw`) and would have allowed 16.15 Mvar. The cut
    therefore came from the controller's bound.
- The forced move is not compensated in the same step. The step QP
  J = wᵀG_w w + ∇Φ_kᵀ w + g_z‖z‖² sees the outputs only through the gradient at the present
  measurement, so it does not price the predicted output change of a bound-forced move.
  The other IBGs correct it from the next measurement on, within about 3 steps.

## 2. Live windows after the run

**Established.**
- Runs execute in PyCharm's Python console (`pydevconsole.py --mode=client`).
  - It is the plain pydev console: IPython is installed in `qOFO_clean` but fails to import
    (`pure_eval` missing).
  - The live windows are PyQt5 windows. Only Qt5 DLLs are loaded; `QT_API=pyqt5`.
- During a run, the windows are pumped only by the runner: `LiveWindow.update` →
  `canvas.flush_events()` once per step. Nothing in `visualisation/` or the runner keeps a Qt
  event loop running after the last step. `finalize()` draws once and returns.
- Measured on run 0025 (console PID 47984), sampled every ~5 s for 96 s after `records.pkl`
  appeared:
  - the console was idle (CPU +0.6 s);
  - the window reported `Responding = False` throughout.
  - So the main thread is not blocked; nobody pumps the window messages. The freeze is the
    absence of an event loop after the run, not a hang.

**Not established.** PyCharm's console is meant to pump GUI events while idle.
- A replay of the import sequence under PyCharm's own import hook
  (`helpers/pydev/pydev_ipython/matplotlibtools.py`) requests `enable_gui('qt')`, then
  `'qt5'`. The resulting `inputhook_qt5` processes Qt events in an offscreen test.
- Why the live console does not end up pumping is not reproduced.
- Side findings:
  - In plain `python`, matplotlib resolves to PyQt6 before `style.py` sets `QT_API`, and
    `mpl.use("Qt5Agg", force=True)` fails silently.
  - The pre-2026-11-02 plotter had a blocking `finish()` (`plt.ioff(); plt.show()`), but the
    runner never called it.

### Change, 2026-11-06 (author decision: block at the end of the run)
- `visualisation/live_base.py`: new `show_until_closed()`, exported.
  - It runs `plt.ioff(); plt.show(block=True)`, so the GUI event loop runs until every open
    figure is closed.
  - It is a no-op without open figures or on a non-interactive backend (checked against
    `backend_registry.list_builtin(BackendFilter.INTERACTIVE)`).
- `experiments/run_multi_system_ofo.py`, `main()`: after the records are pickled and the
  result path is printed, if any `live_plot_*` flag is set, it prints a one-line note and
  calls `show_until_closed()`.
  - The console prompt returns once the windows are closed.
  - The records are already on disk at that point.
  - `run_multi_tso_dso` and the batch drivers are unchanged.
- Checks:
  - Agg with an open figure: returns at once, with no "cannot be shown" warning (warnings
    set to error).
  - Qt5Agg, offscreen: blocks 1.50 s until a timer closes the figures; 48 canvas-timer events
    are processed meanwhile.
  - `py_compile` of both files.
  - `tests/test_live_plot_measurement_perspective.py`: 2 passed.
  - Not yet checked in the PyCharm console with a full run.

## Open (for decision, not implemented)
- Capability edge, three options:
  - (a) Make the modelled diagram continuous below P/Sₙ = 0.1. This is a plant-model change.
  - (b) Taper the controller's bound when the metered P/Sₙ is within a margin above 0.1 and
    falling, so the unit is unloaded over several steps.
  - (c) Price the bound-forced move in the step objective. This is an architectural change.
- In DSO_3 the four PV units sit at alternating rails. This looks like voltage-objective-driven
  circulation (dso_g_v = 10⁴, g_w,DSO-IBG = 1.5). It is an observation, not checked further.
