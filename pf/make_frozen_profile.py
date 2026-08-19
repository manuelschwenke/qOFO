r"""Build a CONSTANT ``ElmFile`` profile from one row of a recorded profile.

**Why.** The Ch. 9.1 open-loop settling battery measures how long the plant
takes to re-settle after a single actuator step, from a fixed operating point.
Its preflight refuses to measure anything unless a 60 s flat run drifts less
than ``1e-4`` pu. A recorded replay profile is a *trajectory*: repointing the
study case's ``ElmFile`` sources at one makes the DER parks ramp their active
power along that trajectory, the synchronous machines take up the difference,
and voltages move ~1.4e-2 pu -- correctly refused by the preflight. Measured
2026-08-19: the four TSO parks each shed 0.43-0.56 MW over 60 s while G 01
picked up 5.3 MW.

Freezing every channel at its value at one instant gives the static operating
point the battery is specified on -- which is what the label ``full_t0_wecc``
means and what the run of record's ``1.41e-10`` pu drift reflects.

**Durability.** The output goes under ``pf/profiles/`` rather than ``results/``.
``results/`` is gitignored and pruned, and a study case pointing into it is
exactly how ``02_RMS_CoSim`` came to be broken: every replay run writes an
absolute path into all 97 sources, so the case dies with the next cleanup.

Format (from ``profile_playback``): line 1 is the channel count ``n``; every
later line is ``time`` followed by ``n`` values, whitespace separated.

Usage::

    python pf\make_frozen_profile.py --source <recorded.txt> --row 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "pf" / "profiles" / "rms_profile_t0_frozen.txt"

#: Horizon of the generated file [s]. Must outlast the longest case the
#: battery runs (600 s disturbance horizon) with margin; PowerFactory holds
#: the last row beyond the end, but relying on that is a silent dependency.
DEFAULT_HORIZON_S = 4000.0
DEFAULT_STEP_S = 20.0


def parse_profile(path: Path) -> tuple[int, List[List[float]]]:
    """``(n_channels, rows)`` where each row is ``[time, v1..vn]``."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    n_ch = int(float(lines[0].split()[0]))
    rows = [[float(x) for x in ln.split()] for ln in lines[1:]]
    bad = [i for i, r in enumerate(rows) if len(r) != n_ch + 1]
    if bad:
        raise ValueError(
            f"{path}: header declares {n_ch} channels, but row(s) {bad[:5]} "
            f"carry {len(rows[bad[0]]) - 1}; refusing to guess")
    return n_ch, rows


def freeze(n_ch: int, values: List[float], horizon_s: float,
           step_s: float) -> str:
    """Render a constant profile holding ``values`` across the horizon."""
    out = [str(n_ch)]
    t = 0.0
    while t <= horizon_s + 1e-9:
        out.append(" ".join([f"{t:g}"] + [f"{v:.12g}" for v in values]))
        t += step_s
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, required=True,
                    help="recorded ElmFile profile to take the frozen row from")
    ap.add_argument("--row", type=int, default=0,
                    help="index of the row to freeze (0 = first / t0)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--horizon-s", type=float, default=DEFAULT_HORIZON_S)
    ap.add_argument("--step-s", type=float, default=DEFAULT_STEP_S)
    a = ap.parse_args(argv)

    n_ch, rows = parse_profile(a.source)
    if not -len(rows) <= a.row < len(rows):
        print(f"[frozen] row {a.row} out of range; source has {len(rows)} rows")
        return 1
    row = rows[a.row]
    values = row[1:]

    # How far the source moves overall, so the reader can see what was frozen
    # away rather than having to trust that it was small.
    spans = [max(r[i + 1] for r in rows) - min(r[i + 1] for r in rows)
             for i in range(n_ch)]
    print(f"[frozen] source {a.source}")
    print(f"[frozen] {n_ch} channels, {len(rows)} rows, "
          f"t {rows[0][0]:g} -> {rows[-1][0]:g} s")
    print(f"[frozen] freezing row {a.row} (t = {row[0]:g} s)")
    print(f"[frozen] channel span over the source trajectory:")
    for i, (v, sp) in enumerate(zip(values, spans)):
        print(f"    ch{i:<2d} value {v:14.9g}   span over source {sp:.6g}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(freeze(n_ch, values, a.horizon_s, a.step_s),
                     encoding="utf-8")
    n_rows = int(a.horizon_s / a.step_s) + 1
    print(f"[frozen] wrote {a.out}  ({n_rows} rows, 0 -> {a.horizon_s:g} s)")
    print(f"[frozen] repoint with:\n"
          f"    python pf\\deactivate_stale_elmfiles.py --repoint {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
