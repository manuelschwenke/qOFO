#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_11_case2_multisystem/ref/run_ref_remote.py
=========================================================
One ``run_ref`` invocation on e5power, pulled back here: the ``--ref-remote`` path
of ``ch_9_16_ref_after_v4`` for a single scenario with its own tag and note, and
without that queue's campaign gates (the selected parameter set must already be
on e5power).  Written for scenario F (author 2026-11-05: "Maybe this would also
fit in the night simulation plan").

This checkout's code is copied over e5power's commit and hash-checked
(``ch_9_11_procedure_v3.sync_remote_code``); e5power gets the run number this
host would give (``ch_9_16_ref_after_v4.reserve_number``); a start that leaves
variants unfinished is completed with ``--run-dir``, as in ``ref_remote_one``.
The driver log goes to ``results/ch_11_case2_multisystem/ref/_remote/<tag>/``.

Usage (from the repository root, on e5wimi with the author's logon)::

    python -X utf8 -m experiments.ch_11_case2_multisystem.ref.run_ref_remote \
        --scenario F --tag v4_F --note "..."

Author: Manuel Schwenke / Claude Code (2026-11-05)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P  # noqa: E402
from experiments.ch_09_parameter_selection import ch_9_11_procedure_v3 as V3  # noqa: E402
from experiments.ch_09_parameter_selection import ch_9_16_ref_after_v4 as R  # noqa: E402

REL = R.F.REF_REL


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="run_ref_remote")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--variants", default=R.VARIANTS)
    ap.add_argument("--minutes", type=int, default=360)
    ap.add_argument("--jobs", type=int, default=R.REMOTE_REF_JOBS)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--note", required=True)
    a = ap.parse_args(argv)
    names = [n.strip() for n in a.variants.split(",") if n.strip()]
    out = ROOT / REL / "_remote" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    P.RUN = out
    P.log(f"{P.REMOTE} at {P.remote_head()[:7]}; scenario {a.scenario}, {names}, "
          f"{a.minutes} min, --jobs {a.jobs}, tag {a.tag}")
    V3.sync_remote_code()
    name: Optional[str] = None
    local: Optional[Path] = None
    starts = fails = 0
    while starts < R.MAX_STARTS and fails <= R.RETRIES:
        miss = names if local is None else R._missing(local, names)
        if not miss:
            break
        cmd = ["-m", "experiments.ch_11_case2_multisystem.ref.run_ref",
               "--variants", ",".join(miss), "--jobs", str(min(a.jobs, len(miss)))]
        cmd += (["--scenario", a.scenario, "--minutes", str(a.minutes), "--tag", a.tag,
                 "--note", a.note] if name is None else ["--run-dir", str(REL / name)])
        if name is None:
            P.log(f"{P.REMOTE} numbers the run {R.reserve_number(REL):04d}")
        before = set(R.remote_names(REL))
        starts += 1
        t0 = time.time()
        try:
            rc: Optional[int] = P.run_remote(f"ref_{a.tag}_{time.strftime('%m%d_%H%M%S')}",
                                             cmd, out, [])
        except RuntimeError as exc:              # the shard died 3 times, or no remoting
            rc = None
            P.log(f"start {starts}: {exc}")
        fails += rc is None or time.time() - t0 >= R.START_FAIL_S
        new = sorted(n for n in set(R.remote_names(REL)) - before
                     if n.endswith(f"_run_ref_{a.tag}"))
        name = name or (new[-1] if new else None)
        if name is not None:
            local = R._pull(name, rc, synced=True, rel=REL, what=f"REF {a.scenario}", need=None)
        left = names if local is None else R._missing(local, names)
        if left:
            P.log(f"start {starts} ended with rc {rc}; "
                  + ("no run folder" if name is None else f"missing {left}"))
    status = {} if local is None else {
        n: json.loads((local / n / "status.json").read_text("utf-8"))["status"]
        for n in names if (local / n / "status.json").exists()}
    P.log(f"REF {a.scenario}: {P.rel(local) if local else 'no run folder'}; {status}")
    return 0 if local is not None and not R._missing(local, names) else 1


if __name__ == "__main__":
    sys.exit(main())
