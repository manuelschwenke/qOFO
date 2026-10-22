# Moved: the daily log now lives in `00_daily_log/`

The daily log was kept in two places — this folder (`docs/daily_log/`, 214
entries in `MM_YYYY/` folders) and `00_daily_log/` at the repo root (10
entries, flat). Author decision 2026-10-22: **`00_daily_log/` is the single
location.** Everything that was here moved there on that date, keeping the
`MM_YYYY/` structure, `INDEX.md` and `_build_index.py`.

Nothing was deleted. The ten entries that had been filed flat in
`00_daily_log/` were sorted into their month folders in the same move. One
filename existed in both places with different content — a short summary in
`00_daily_log/` and the detailed change log here — so the summary is now
`00_daily_log/10_2026/2026-10-20_hybrid_shunt_dispatch_summary.md` beside the
detail it points to.

Write new entries to `00_daily_log/<MM_YYYY>/<YYYY-MM-DD>_<slug>.md` and
regenerate the index with:

```
python 00_daily_log\_build_index.py
```

This file is a signpost for links and notes that still point here; it is not a
second log location.
