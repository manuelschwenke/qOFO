"""Per-area residual load of the six windows (profile-turn draft, read-only).

Builds the run-0008 configuration of each window, captures net/meta/zone_defs at
the pre-loop hook (no simulation), and writes the residual load
sum(loads) - sum(DER P) [MW] per TS zone (with its HV networks) and per HV
network at 60-s resolution over the 12-h window.
"""
import ctypes
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
k32 = ctypes.windll.kernel32
k32.GetCurrentProcess.restype = ctypes.c_void_p
k32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
assert k32.SetProcessAffinityMask(k32.GetCurrentProcess(), 0x55555)

ROOT = Path(r"F:\qOFO_res")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import contextlib, dataclasses, io  # noqa: E401,E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from configs.paramsets import apply_paramset  # noqa: E402
from core.profiles import load_profiles  # noqa: E402
from experiments.ch_09_parameter_selection._ch9_selected_design import (  # noqa: E402
    FROZEN_DSO_DER_SCALE, FROZEN_DSO_LOAD_P_SCALE)
from experiments.run_multi_system_ofo import make_config  # noqa: E402
from tuning._sim_loader import get_run_multi_tso_dso  # noqa: E402
from tuning.scenarios_mc_v2 import condition_set_evented  # noqa: E402

OUT = Path(__file__).parent / "pt_residuals.json"
base = apply_paramset(make_config(), "stage1_v3_uc_2026_11_pq10", verbose=False)
base = dataclasses.replace(base, dso_der_scale=dict(FROZEN_DSO_DER_SCALE),
                           dso_load_p_scale=dict(FROZEN_DSO_LOAD_P_SCALE),
                           sts_coupler_hold_after_bank=1)
res = {}
for sc in condition_set_evented(hours=12):
    cfg = sc.overlay_on(base)
    cap = {}

    def hook(state):
        cap.update(state)
        return True

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        get_run_multi_tso_dso()(cfg, pre_loop_hook=hook)
    net, meta, zdefs = cap["net"], cap["meta"], cap["zone_defs"]
    if sc.name == "W1":
        z0 = next(iter(zdefs.values()))
        print("ZoneDefinition fields:", [f for f in dir(z0) if not f.startswith("_")][:40])
    csv = cfg.profiles_csv or os.path.join("data", "profiles.csv")
    prof = load_profiles(csv, timestep_s=60.0)
    t0 = pd.Timestamp(cfg.start_time)
    idx = pd.date_range(t0, t0 + timedelta(hours=12), freq="60s")
    prof = prof.reindex(prof.index.union(idx)).interpolate("time").loc[idx]

    def residual(buses):
        b = set(int(x) for x in buses)
        r = np.zeros(len(idx))
        cap_mw = 0.0
        for tab, sign, pcol in ((net.load, 1.0, "profile_p"), (net.sgen, -1.0, "profile"),
                                (net.gen, -1.0, "profile")):
            if "base_p_mw" not in tab.columns or pcol not in tab.columns:
                continue
            for i in tab.index:
                if int(tab.at[i, "bus"]) not in b:
                    continue
                if tab is net.sgen and str(tab.at[i, "name"]).startswith("BOUND_"):
                    continue
                p = tab.at[i, pcol]
                if tab is net.gen and pd.isna(p):
                    continue          # dispatched machine, not exogenous
                s = prof[p].to_numpy(float) if (isinstance(p, str) and p in prof.columns) else 1.0
                r += sign * float(tab.at[i, "base_p_mw"]) * s
                if sign < 0:
                    cap_mw += float(tab.at[i, "base_p_mw"])
        return r, cap_mw

    areas = {}
    for zid, zd in zdefs.items():
        buses = getattr(zd, "bus_indices", None) or getattr(zd, "buses", None)
        areas[f"zone{zid}"] = residual(buses)
    for hv in meta.hv_networks:
        areas[str(hv.net_id)] = residual(hv.bus_indices)
    res[sc.name] = {"start": str(t0), "events_s": list(sc.event_times_s),
                    "areas": {a: {"r_mw": [round(x, 3) for x in r], "der_cap_mw": round(c, 1)}
                              for a, (r, c) in areas.items()}}
    hvz = {str(hv.net_id): [z for z, zd in zdefs.items()
                            if set(hv.bus_indices) & set(getattr(zd, "bus_indices", []) or [])]
           for hv in meta.hv_networks}
    res[sc.name]["hv_zone"] = hvz
    print(sc.name, t0, {a: (round(min(r), 1), round(max(r), 1), round(c, 1)) for a, (r, c) in areas.items()}, hvz)
OUT.write_text(json.dumps(res), "utf-8")
print("written", OUT)
