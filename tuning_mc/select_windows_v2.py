"""
tuning_mc/select_windows_v2.py
==============================
Pick the Tier-1 / Tier-2 windows of the 0815 campaign **from the Screen-1
measurement**, not by hand, and print a paste-ready block for
``scenarios_mc_v2.py``.

Why this exists
---------------
The 0814 bank was five hand-picked operating points.  Two defects followed
from that and both are measurable rather than matters of taste:

1. one window (``mc_undervolt_ramp_winter``) contributed ~60 % of the
   aggregate ``f_ts``, so the "aggregate over five windows" was in effect a
   single window;
2. that same window has **exactly zero** DER reactive capability, the stratum
   in which ``tau``, ``lambda_dso`` and ``dso_g_v_ratio`` are structurally
   inert -- so the dominant window was also the one carrying no signal about
   half the search space.

Screen 1 (``stage_1a_excitation --screen1``) already measures DER reactive
capability for the whole profile year at 2-h stride, exactly and without
simulation.  This module turns that measurement into a *stratified* design:
the role x season cells are declared here, and which timestamp fills each cell
is decided by the recorded capability, deterministically.

Strata
------
The capability distribution over the year is not smooth -- the VDE curve
saturates -- so the strata are cut at its own structure rather than at
quantiles of convenience:

    S0 "none"     q_range_total == 0                 (18.6 % of the year)
    S1 "partial"  0 < q_range_total < SATURATION
    S2 "full"     q_range_total >= SATURATION

``SATURATION`` is read off the data as the modal plateau (the value at which
every DSO DER is above the dead zone).

Parity
------
Design windows sit in **odd** ISO weeks, confirmation windows in **even** ones,
which is the convention the campaign has used throughout.  The Tier-2 audit
windows are placed in **odd** weeks: they measure wear on candidates already
selected from the design bank, so keeping them off the even weeks leaves the
confirmation claim untouched by anything the audit does.

Usage::

    python -m tuning_mc.select_windows_v2
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCREEN1 = _REPO_ROOT / "results" / "tuning_mc" / "stage1a" / "screen1.json"

SEASON_MONTHS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}

#: ``(role, season) -> (hour band, target stratum)``.
#:
#: The hour band is a physical requirement of the role (an over-voltage ramp is
#: only meaningful where PV is producing); the target stratum is the design
#: variable this module exists to control.
DESIGN_CELLS: list[tuple[str, str, tuple[int, int], str]] = [
    # role            season     hours       stratum
    ("quiet",        "summer",  (2, 4),     "none"),
    ("quiet",        "spring",  (10, 14),   "full"),
    ("gen_trip",     "spring",  (10, 14),   "full"),
    ("gen_trip",     "summer",  (10, 14),   "full"),
    ("gen_trip",     "winter",  (8, 12),    "partial"),
    ("ramp_up",      "winter",  (17, 20),   "none"),
    ("ramp_up",      "autumn",  (10, 14),   "partial"),
    ("ramp_up",      "spring",  (10, 14),   "full"),
    ("ramp_down",    "summer",  (11, 14),   "full"),
    ("ramp_down",    "spring",  (10, 14),   "partial"),
    ("reversal",     "spring",  (10, 14),   "full"),
    ("reversal",     "autumn",  (10, 14),   "partial"),
]

#: The confirmation set is a *subset of the same cells*, so the two banks are
#: drawn from one distribution and their costs are comparable.
#:
#: **Nine cells, at least two per stratum.**  The first version had six, with
#: ``partial`` represented by a single window -- and measured 2026-08-15 that one
#: window (`c_gen_trip_winter`, f_ts 7.21 against a design-bank ``partial`` mean
#: of 1.12) contributed **49 % of the whole confirmation aggregate**.  With n = 1
#: there is no averaging, so one extreme draw becomes half the answer and the
#: design/confirmation comparison is unusable in aggregate.  The design bank has
#: n = 4 there and is insulated.  Three ``partial`` cells is the minimum that
#: gives the stratum any tolerance to an outlier.
CONFIRM_CELLS = [
    ("quiet",     "summer", (2, 4),   "none"),
    ("ramp_up",   "winter", (17, 20), "none"),
    ("gen_trip",  "winter", (8, 12),  "partial"),
    ("ramp_up",   "autumn", (10, 14), "partial"),
    ("reversal",  "autumn", (10, 14), "partial"),
    ("gen_trip",  "spring", (10, 14), "full"),
    ("ramp_down", "summer", (11, 14), "full"),
    ("reversal",  "spring", (10, 14), "full"),
    ("ramp_up",   "spring", (10, 14), "full"),
]

#: Tier 2: 12-h profile-driven audit windows.  Start hours are early so the
#: window covers the daily PV rise and the evening load peak -- the two
#: transitions a tap changer actually responds to over a real day.
AUDIT_CELLS = [
    ("audit_quiet",    "winter", (5, 7),  "partial"),
    ("audit_quiet",    "summer", (5, 7),  "full"),
    ("audit_gen_trip", "spring", (5, 7),  "full"),
    ("audit_ramp",     "autumn", (5, 7),  "partial"),
]


def _season(ts: datetime) -> str:
    for name, months in SEASON_MONTHS.items():
        if ts.month in months:
            return name
    raise AssertionError(ts)


def _stratum(q: float, saturation: float) -> str:
    if q <= 1e-9:
        return "none"
    return "partial" if q < saturation - 1e-6 else "full"


def load_rows() -> tuple[list[dict], float]:
    rows = json.loads(SCREEN1.read_text(encoding="utf-8"))
    for r in rows:
        r["_ts"] = datetime.fromisoformat(r["timestamp"])
        r["_season"] = _season(r["_ts"])
    q = np.array([r["q_range_total_mvar"] for r in rows])
    nz = q[q > 1e-9]
    # The plateau: the most frequent non-zero value, to 0.1 Mvar.  Every DSO DER
    # above the dead zone and saturated is one number, and it is by far the
    # commonest, so the mode is a robust read of "full".
    vals, counts = np.unique(np.round(nz, 1), return_counts=True)
    saturation = float(vals[int(np.argmax(counts))])
    for r in rows:
        r["_stratum"] = _stratum(r["q_range_total_mvar"], saturation)
    return rows, saturation


def pick(rows: list[dict], *, season: str, hours: tuple[int, int],
         stratum: str, parity: str, taken: list[datetime]) -> dict:
    """The window filling one cell.

    Two criteria, in this order:

    1. **Representativeness** -- keep only candidates whose capability is in
       the closest quartile to the stratum's own median.  Selecting the
       stratum's tail would make the bank harder than the year it claims to
       sample; selecting its median makes it typical of that stratum.
    2. **Temporal spread** -- among those, take the window whose nearest
       already-selected neighbour is furthest away (maximin in days).

    The second criterion is not cosmetic.  The "full" stratum is an exact
    saturation *plateau*: hundreds of windows carry bit-identical capability,
    so criterion 1 alone leaves a large tie set, and any deterministic
    tie-break on timestamp collapses the whole bank onto the first few days of
    each season.  A first pass did exactly that -- four of twelve design
    windows landed in ISO week 9, three of them on one calendar day at 10:00,
    12:00 and 14:00.  Windows two hours apart on the same day are very nearly
    the same operating point, which is the opposite of what a design bank is
    for.
    """
    lo, hi = hours
    want_odd = parity == "odd"
    cand = [r for r in rows
            if r["_season"] == season
            and lo <= r["hour"] <= hi
            and (r["iso_week"] % 2 == 1) == want_odd
            and r["_stratum"] == stratum
            and all(r["_ts"].date() != t.date() for t in taken)]
    if not cand:
        raise SystemExit(
            f"[select] no window for season={season} hours={hours} "
            f"stratum={stratum} parity={parity}")
    med = float(np.median([r["q_range_total_mvar"] for r in cand]))
    cand.sort(key=lambda r: abs(r["q_range_total_mvar"] - med))
    keep = max(1, len(cand) // 4)
    near = cand[:keep]
    cutoff = abs(near[-1]["q_range_total_mvar"] - med)
    near = [r for r in cand if abs(r["q_range_total_mvar"] - med) <= cutoff]

    def spread(r: dict) -> float:
        if not taken:
            return float("inf")
        return min(abs((r["_ts"] - t).total_seconds()) for t in taken)

    near.sort(key=lambda r: (-spread(r), r["timestamp"]))
    taken.append(near[0]["_ts"])
    return near[0]


def report(title: str, cells, rows, parity: str,
           taken: list[datetime] | None = None) -> list[dict]:
    # ``taken`` carries across banks when passed: the Tier-2 audit must not
    # land on a calendar day the Tier-1 bank already uses, or the "authoritative
    # wear measurement" would be partly a re-run of an operating point the
    # search had already seen.
    taken = [] if taken is None else taken
    print(f"\n### {title}  (ISO-week parity: {parity})")
    print(f"{'role':<14}{'season':<9}{'stratum':<9}{'timestamp':<20}"
          f"{'wk':>4}{'load MW':>9}{'Qrange':>9}{'TSO Q':>8}{'DSO Q':>8}")
    out = []
    for role, season, hours, stratum in cells:
        r = pick(rows, season=season, hours=hours, stratum=stratum,
                 parity=parity, taken=taken)
        tso = sum(r[f"TSO-z{i}.q_range_mvar"] for i in (1, 2, 3))
        dso = sum(r[f"DSO-DSO_{i}.q_range_mvar"] for i in (1, 2, 3, 4))
        print(f"{role:<14}{season:<9}{r['_stratum']:<9}{r['timestamp']:<20}"
              f"{r['iso_week']:>4}{r['load_p_mw']:>9.0f}"
              f"{r['q_range_total_mvar']:>9.0f}{tso:>8.0f}{dso:>8.0f}")
        out.append({"role": role, "season": season, "stratum": r["_stratum"],
                    "timestamp": r["timestamp"], "iso_week": r["iso_week"],
                    "q_range_total_mvar": r["q_range_total_mvar"],
                    "tso_q_range_mvar": tso, "dso_q_range_mvar": dso,
                    "load_p_mw": r["load_p_mw"]})
    counts: dict[str, int] = {}
    for e in out:
        counts[e["stratum"]] = counts.get(e["stratum"], 0) + 1
    print(f"  strata: {counts}   ({len(out)} windows)")
    return out


def main() -> int:
    rows, saturation = load_rows()
    q = np.array([r["q_range_total_mvar"] for r in rows])
    strata = {s: sum(1 for r in rows if r["_stratum"] == s)
              for s in ("none", "partial", "full")}
    print(f"[select] screen1: {len(rows)} windows, stride 2 h, 2016")
    print(f"[select] saturation plateau = {saturation:.1f} Mvar")
    print(f"[select] year strata: " + "  ".join(
        f"{k}={v} ({100 * v / len(rows):.1f} %)" for k, v in strata.items()))

    used: list[datetime] = []
    design = report("Tier 1 -- design bank (12 x 90 min)", DESIGN_CELLS,
                    rows, "odd", used)
    confirm = report("Tier 1 -- confirmation (9 x 90 min)", CONFIRM_CELLS,
                     rows, "even")
    audit = report("Tier 2 -- audit (4 x 12 h)", AUDIT_CELLS, rows, "odd",
                   list(used))

    out = _REPO_ROOT / "results" / "tuning_mc" / "campaign_0815"
    out.mkdir(parents=True, exist_ok=True)
    (out / "window_selection.json").write_text(json.dumps(
        {"saturation_mvar": saturation, "year_strata": strata,
         "design": design, "confirmation": confirm, "audit": audit},
        indent=1), encoding="utf-8")
    print(f"\n[select] wrote {out / 'window_selection.json'}")

    print("\n# --- paste into scenarios_mc_v2.py -----------------------------")
    for tag, entries in (("DESIGN", design), ("CONFIRM", confirm),
                         ("AUDIT", audit)):
        print(f"{tag}_WINDOWS = [")
        for e in entries:
            print(f"    ({e['role']!r}, {e['season']!r}, "
                  f"datetime.fromisoformat({e['timestamp']!r}), "
                  f"{e['stratum']!r}),   # wk {e['iso_week']}, "
                  f"Q {e['q_range_total_mvar']:.0f} Mvar")
        print("]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
