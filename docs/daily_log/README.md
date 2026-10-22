# Daily log — the single location

The log was kept in two places, both tracked in this repository:
`docs/daily_log/` (212 entries in `MM_YYYY/` folders, with `INDEX.md` and
`_build_index.py`) and `00_daily_log/` at the repo root (10 entries, flat).
The global convention named the second and the project `.claude/CLAUDE.md` the
first, so entries landed in whichever the writing session had read.

Author decision 2026-10-22: **`docs/daily_log/` is the single location**, which
is what `.claude/CLAUDE.md` already instructs. The ten entries that had been
filed flat in `00_daily_log/` were moved into their month folders and that
folder is gone. Nothing was deleted.

Write new entries as `docs/daily_log/<MM_YYYY>/<YYYY-MM-DD>_<slug>.md` and
regenerate the index:

```
python docs\daily_log\_build_index.py
```

Two things a reader of the older entries should know:

- One filename existed in both places with different content — a short summary
  and the detailed change log of the same name. Both are kept:
  `10_2026/2026-10-20_hybrid_shunt_dispatch.md` is the detail and
  `..._summary.md` the summary beside it.
- Historical entries still refer to `00_daily_log/...` paths in places. Those
  are left as written: an entry records what was true when it was written.
