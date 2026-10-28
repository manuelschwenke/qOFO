# Amendment 1 (STS coupler engagement bound): run 0008, re-selection and engagement probe

Timestamp: 2026-10-28, ~10:15–11:50 Europe/Berlin.

**Reason:** the author chose option (c1) of
`docs/tuning/2026-10-28_tap_engagement_at_selected_weights.md`. The stated
intent: the tap changers should engage when many STS voltages (the area
median in the live plot) are far from 1.03 pu. If the re-selection does not
show this, the next step is a full rerun with another strategy.

## Changed / added

- `docs/tuning/2026-10-28_amendment1_sts_engagement_bound.md`: the rule
  θ_STS,min ≤ 1.08 − 1.03 = 0.05 pu, declared and committed (dddfa45) before
  the run, with its execution record.
- `experiments/ch_09_parameter_selection/ch_9_3_amendment1.py` (new
  producer, `start_thesis_run`). It runs, in one invocation:
  - **select:** θ_STS,min per candidate via `stage1_screen.engage_deviation`,
    then the unchanged `ch_9_3_select.py` on the admitted candidates;
  - **settle:** Experiment C in rank order, reusing run 0001's measured
    results;
  - **probe:** W1–W4, W6 re-simulated with a read-only wrapper on each
    controller's `_compute_objective_gradient` and `step`, installed through
    the runner's `pre_loop_hook`. It logs the measured voltages and the commit
    ratio |c_i + 2 g_u u_i| / (g_w,i + g_u) of every tap and bank;
  - **paramset:** writes `ch9_selected_band_2026_10_a1` on a settling pass.
    `SELECTED_PARAMSET` is not switched.
- `analysis/tap_engagement/engagement_eval.py`: the engagement criteria,
  fixed before the results were seen:
  - minutes with |median dev| ≥ 0.010/0.015/0.020/0.025 pu;
  - far episodes, |median dev| ≥ 0.02 pu for ≥ 10 min, and whether a
    coupler served each;
  - unprompted commits.

  After the first evaluation one bug was fixed: a commit was mapped to the
  STS iteration before the decision instead of the decision iteration. This
  understated the deviation at a commit that coincided with a voltage jump.
  The commit ratio is now printed with each commit.
- `analysis/tap_engagement/replay_path.py`: checks whether the amended rule
  would have changed the Stage-II-b centres.
- Provenance: `thesis_runs` calls plain `git`, which fails on this host
  (safe.directory). The run was started with an env-scoped
  `GIT_CONFIG_COUNT/KEY/VALUE` setting, so it records commit dddfa45.
  Run 0007 of the D1/D2 session recorded "Code | unavailable".

## Results (run 0008)

- **Selection:** 140 / 242 admitted, 51 feasible. Rank 1 is 301c285ee52b
  (θ_STS,min 0.0147 pu), rank 2 c08b1d3d33ab.
- **Replay:** incumbents after II-a and rounds 1–3 are identical under the
  amended rule. A from-scratch application would evaluate the same
  242 candidates.
- **Probe:** reproduction is bit-exact, 20/20.
  - 301c285ee52b: the area median is ≥ 0.020 pu from 1.03 for 1 min in
    240 area-hours, with no far episode.
  - As frozen: 782 min, 16 far episodes, none served by a tap.
  - All non-disturbance commits of 301c are gradient-driven, at median
    deviation 0.009–0.015 pu. There are no reversals.
- **Settling** of 301c: running, 24/120 steps at 11:45; finish expected in
  the afternoon.

## Open

- **Paramset switch:** whether to switch `SELECTED_PARAMSET` after settling.
  It invalidates D1/D2 and the `ch_9_1_*` producers.
- **Thesis route:** how the thesis presents this. Two options:
  - the amendment as is;
  - a frozen "procedure 2026-10-28" applied once, with run 0001 as a
    development campaign. Replay mode reuses the identical units; a verbatim
    rerun takes ≈ 38 h.
