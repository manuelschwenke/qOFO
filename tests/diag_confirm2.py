#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/diag_confirm2.py
============================
Round 2 (Claude Code, 2026-06-24). Closes two gaps from diag_confirm.py:
  * Proper T' test on V4 (both layers OFO -> DSO+TSO DER columns exist, so T'
    actually transforms something). tprimeOFF vs tprimeON.
  * Tie-flow lever probe: V4 with g_v lowered 1e7 -> 1e6 (less aggressive
    voltage tracking). g_w_pcc x10 did NOT damp the tie flows; test whether the
    voltage-tracking weight does.
Reuses run()/summarize() from diag_confirm.py. Writes results/_diag_confirm/.
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tests.diag_confirm import run, summarize

RUNS = [
    ("V4_tprimeOFF", "V4", {"apply_qv_h_transform": False}),
    ("V4_tprimeON",  "V4", {"apply_qv_h_transform": True}),
    ("V4_gv_lo",     "V4", {"g_v": 1.0e6}),
]


def main():
    results = {}
    for tag, base, extra in RUNS:
        print("\n" + "=" * 72 + f"\n  RUN {tag} (extra {extra})\n" + "=" * 72,
              flush=True)
        results[tag] = summarize(tag, run(tag, base, extra))

    print("\n" + "#" * 72 + "\n  ROUND-2 SUMMARY\n" + "#" * 72)
    hdr = f"{'run':<16}{'family':<12}{'rev_rate':>9}{'tv_ratio':>9}{'step_rms':>10}"
    print(hdr)
    for tag in results:
        for fam, (rr, tv, sr) in results[tag].items():
            print(f"{tag:<16}{fam:<12}{rr:>9.2f}{tv:>9.1f}{sr:>10.3g}")
        print("-" * len(hdr))


if __name__ == "__main__":
    main()
