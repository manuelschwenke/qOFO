#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick Vdbg-vs-V3 e_v comparison on the TRUE-config fresh pickles."""
from __future__ import annotations
import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.diag_voltage import ev_series, teeth

FRESH = os.path.join("results", "_diag_fresh")
out = {}
for tag in ["Vdbg", "V3"]:
    with open(os.path.join(FRESH, tag, "log.pkl"), "rb") as f:
        out[tag] = ev_series(pickle.load(f))

for tag in ["Vdbg", "V3"]:
    t, e, a = out[tag]
    print(f"\n{tag}: {len(t)} rec, TSO steps {a.sum()}, "
          f"median TSO spacing {np.median(np.diff(t[a])):.1f} min, "
          f"e_v overall [{e.min():.1f}, {e.max():.1f}] mp.u.")
    for w in [(65, 175), (125, 175), (200, 255)]:
        th = teeth(t, e, *w)
        print(f"   win{w}: peaks={th.get('n_peaks')} "
              f"spacing={th.get('mean_spacing_min', float('nan')):.1f}min "
              f"p2t={th.get('mean_p2t', float('nan')):.2f} mp.u. "
              f"[{th.get('emin', float('nan')):.1f},{th.get('emax', float('nan')):.1f}]")

# raw trace 60-96 min (around gen trip) at full 20s resolution, TSO marked
print("\n   t[min]   Vdbg    V3    (*=TSO)")
tV, eV, aV = out["Vdbg"]; t3, e3, a3 = out["V3"]
m = (t3 >= 60) & (t3 <= 96)
for i in np.where(m)[0]:
    star = "*" if a3[i] else " "
    j = np.argmin(np.abs(tV - t3[i]))
    print(f"   {t3[i]:6.1f} {eV[j]:6.1f} {e3[i]:6.1f}  {star}")

fig, ax = plt.subplots(figsize=(9, 4))
for tag, col in (("Vdbg", "tab:orange"), ("V3", "tab:blue")):
    t, e, a = out[tag]
    m = (t >= 55) & (t <= 130)
    ax.plot(t[m], e[m], color=col, lw=1.2, label=tag)
    ax.plot(t[m][a[m]], e[m][a[m]], "o", color=col, ms=3)
for x in (60, 120):
    ax.axvline(x, color="0.6", lw=0.8, ls="--")
ax.set_xlabel("time / min"); ax.set_ylabel("e_v / mp.u.")
ax.set_title("TRUE config (TSO 180s): Vdbg (TS-OFO+cosphi) vs V3 (TS-OFO+Q(V))")
ax.legend()
png = os.path.join(FRESH, "ev_vdbg_vs_v3.png")
fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)
print(f"\n[saved] {png}")
