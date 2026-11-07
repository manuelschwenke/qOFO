#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_11_case2_multisystem/mc/run_mc_remote.py
=======================================================
One ``run_mc`` campaign share on e5power, pulled back here: the pattern of
``ref/run_ref_remote.py`` for the Monte-Carlo (plan
``docs/architecture/2026-11-07_case2_mc_new_base_plan.md``: the variants are
split between e5wimi and e5power, 30 workers each, and merged afterwards with
``mc/merge_mc.py``).

This checkout's code is copied over e5power's commit and hash-checked
(``ch_9_11_procedure_v3.sync_remote_code``); e5power gets the run number this
host would give (``ch_9_16_ref_after_v4.reserve_number``), so launch the local
share first.  The remote call has no ``--new``: the first start opens the
campaign of ``--tag``, a relaunch of a dead shard resumes it (``run_mc`` skips
every run that has its JSON).  The driver log goes to
``results/ch_11_case2_multisystem/mc/_remote/<tag>/``.

Usage (from the repository root, on e5wimi with the author's logon)::

    python -X utf8 -m experiments.ch_11_case2_multisystem.mc.run_mc_remote \
        --variants S1,M,O2 --jobs 30 --buffer 12 --tag newbase_e5power --note "..."

Author: Manuel Schwenke / Claude Code (2026-11-07)
"""

from __future__ import annotations

import argparse
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

REL = Path("results") / "ch_11_case2_multisystem" / "mc"
MAX_STARTS = 4


def remote_argv(a: argparse.Namespace) -> List[str]:
    """The ``run_mc`` call on e5power (no ``--new``, see the module docstring)."""
    return ["-m", "experiments.ch_11_case2_multisystem.mc.run_mc",
            "--variants", a.variants, "--jobs", str(a.jobs), "--buffer", str(a.buffer),
            "--tag", a.tag, "--note", a.note]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="run_mc_remote")
    ap.add_argument("--variants", required=True)
    ap.add_argument("--jobs", type=int, default=30)
    ap.add_argument("--buffer", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--note", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="sync and print the remote call, launch nothing")
    a = ap.parse_args(argv)
    if "O3" in a.variants.split(","):
        raise SystemExit("O3 stays on the host that runs the full prescreen (n_o3)")
    out = ROOT / REL / "_remote" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    P.RUN = out
    P.log(f"{P.REMOTE} at {P.remote_head()[:7]}; MC {a.variants}, --jobs {a.jobs}, "
          f"--buffer {a.buffer}, tag {a.tag}")
    V3.sync_remote_code()
    cmd = remote_argv(a)
    P.log(f"remote call: {' '.join(cmd)}")
    if a.dry_run:
        return 0
    P.log(f"{P.REMOTE} numbers the run {R.reserve_number(REL):04d}")
    name: Optional[str] = None
    rc: Optional[int] = None
    for start in range(1, MAX_STARTS + 1):
        try:
            rc = P.run_remote(f"mc_{a.tag}_{time.strftime('%m%d_%H%M%S')}", cmd, out, [])
        except RuntimeError as exc:              # the shard died 3 times, or no remoting
            rc = None
            P.log(f"start {start}: {exc}")
        names = sorted(n for n in R.remote_names(REL) if n.endswith(f"_run_mc_{a.tag}"))
        name = names[-1] if names else None
        if rc == 0:
            break
    if name is None:
        P.log("no remote campaign folder")
        return 1
    local = R._pull(name, rc, synced=True, rel=REL, what=f"MC {a.tag}", need=None)
    P.log(f"MC {a.tag}: {P.rel(local)}, rc {rc}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
