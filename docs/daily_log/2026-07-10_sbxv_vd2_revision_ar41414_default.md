# 2026-07-10 — SBX-V: V-D2 revised — Normalbereich = AR 4141-4 preset

**What:** Per Manuel's decision, the operative Normalbereich is now the E VDE-AR-N 4141-4
default band (5 % raising / 10 % lowering of contracted P per AggregationArea, [AR §5.2.1])
instead of the fixed ±50 Mvar box.

**Changes:**

- `sbxv/config.py`: `band_preset` default flipped to `"ar41414_default"` (documented as the
  V-D2 revision); explicit edges remain for `"fixed"`.
- `experiments/019_SBXV_E2.py`: the symmetric sweep arms now set `band_preset="fixed"`
  EXPLICITLY (they would otherwise silently ignore their widths under the new default).
- Runner `[sbxv]` startup print + `adapter.finalise()["bands"]` + E1 summary now report the
  RESOLVED per-area band edges (under the preset the config's `band_q_*` values are inert and
  printing them would mislead).
- Tests updated (default-preset assertion; explicit-fixed variants). Suite 102 green.

**Context (STATUS §5.6/§5.7):** the preset fits the ungesichert-centred thesis scope — the
band is the standard's own planning product (derived from the connection agreement /
contracted P), so no ORPF pre-run is needed; contracted P is proxied by the rated interface
capacity Σ `sn_hv_mva`. E1 re-run with the preset band launched (results follow).

**Why:** V-D2 was a locked decision ("fixed ±50, preset prepared but not default"); the plan
requires deviations to be recorded — done here and in STATUS §5.7.
