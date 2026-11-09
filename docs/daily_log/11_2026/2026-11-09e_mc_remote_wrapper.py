"""run_mc_remote for the REF checkout: e5power root F:\\qOFO_runs\\qOFO_ref (the module default
is qOFO_res) and an explicit ``--n`` (the module's remote call has none, so run_mc's default
N = 100 would apply).  No repository code is changed.

Usage: python -X utf8 mc_remote_wrapper.py N <run_mc_remote args...>
"""
import sys
from pathlib import Path

sys.path.insert(0, r"F:\qOFO_ref")
from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P  # noqa: E402
from experiments.ch_11_case2_multisystem.mc import run_mc_remote as RM  # noqa: E402

P.REMOTE_ROOT = r"F:\qOFO_runs\qOFO_ref"
N = sys.argv[1]
_orig = RM.remote_argv


def remote_argv(a):
    return _orig(a) + ["--n", N]


RM.remote_argv = remote_argv
sys.exit(RM.main(sys.argv[2:]))
