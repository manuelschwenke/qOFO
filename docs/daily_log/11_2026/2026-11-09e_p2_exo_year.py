"""Exogenous year series (15 min) of the Case-2 network for the Period II score, per network.

Writes <out_dir>/exo_year.pkl exactly as the input of the declared score script
(Z:\\latex_diss_clone\\00_daily_log\\2026-11-07_ch10_period2_score.py.txt), built with
characterise_reference_period.build/classify_buses/exogenous_year from F:\\qOFO_ref.
Also prints the installed TS IBG per area and profile.

Usage: python -I p2_exo_year.py <network> <out_dir>
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import sys
sys.dont_write_bytecode = True
if os.environ.get("SCREEN_CPUS"):
    import psutil
    psutil.Process().cpu_affinity([int(c) for c in os.environ["SCREEN_CPUS"].split(",")])
sys.path.insert(0, r"F:\qOFO_ref")
os.chdir(r"F:\qOFO_ref")
import dataclasses, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

from experiments.ch_11_case2_multisystem import variants as v
from experiments.ch_08_setup.reference_period import characterise_reference_period as crp

NETWORK, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
cfg = dataclasses.replace(v.resolved("L2", 360, [], 20261027), scenario=NETWORK)
net, meta, bus_zone = crp.build(cfg)
level = crp.classify_buses(net, meta, bus_zone)
for s in net.sgen.index:
    if level[int(net.sgen.at[s, "bus"])].startswith("TS:") and net.sgen.at[s, "in_service"]:
        print(f"TS IBG {net.sgen.at[s, 'name']}: bus {int(net.sgen.at[s, 'bus'])}, "
              f"{level[int(net.sgen.at[s, 'bus'])]}, {float(net.sgen.at[s, 'base_p_mw']):.0f} MW, "
              f"profile {net.sgen.at[s, 'profile']}")
df = crp.exogenous_year(net, level)
df.to_pickle(OUT / "exo_year.pkl")
print(NETWORK, "->", OUT / "exo_year.pkl", df.shape)
