"""Usage: python -X utf8 analysis/tap_engagement/capture_H.py <out_dir>  (one controller build, ~1 min)

Capture H_V of every integer column at W1 (one controller build, no time loop).

Same base configuration as tuning/stage1_screen main(): make_config() with the
selected weights applied, frozen network (DSO_3 x2), start time of W1, and the
replacements of load_or_build_operating_points. Read-only use of make_config.
Writes the voltage rows of each tap/bank column, the device indices and the
transformer data, to check (a) sign structure of the columns and (b) the
column order assumed for the record keys.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import dataclasses as dc
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from experiments.run_multi_system_ofo import make_config  # noqa: E402
from experiments.ch_09_parameter_selection._ch9_selected_design import (  # noqa: E402
    FROZEN_DSO_DER_SCALE, FROZEN_DSO_LOAD_P_SCALE)
from tuning.parameters import apply_to_config  # noqa: E402
from tuning.scenarios_mc_v2 import CONDITIONS  # noqa: E402
from tuning._sim_loader import get_run_multi_tso_dso  # noqa: E402

w = json.loads((ROOT / "configs/paramsets/ch9_selected_band_2026_10.json").read_text("utf-8"))["fields"]
base = apply_to_config(make_config(), w)
base = dc.replace(base, dso_der_scale=dict(FROZEN_DSO_DER_SCALE),
                  dso_load_p_scale=dict(FROZEN_DSO_LOAD_P_SCALE))
start = dict((c[0], c[1]) for c in CONDITIONS)["W1"]
cfg = dc.replace(base, start_time=start, n_total_s=60.0, contingencies=[], verbose=0,
                 live_plot_controller=False, live_plot_cascade=False,
                 live_plot_overview=False, run_stability_analysis=False,
                 precondition_g_w=False)
cap = {}
get_run_multi_tso_dso()(cfg, pre_loop_hook=lambda s: cap.update(s) or True)
net = cap["net"]
out = {"tso": {}, "dso": {}}
for zid, ctrl in sorted(cap["tso_controllers"].items()):
    c = ctrl.config
    H = np.asarray(ctrl._expand_H_to_der_level(ctrl._build_sensitivity_matrix()), float)
    n_v = len(c.voltage_bus_indices)
    mapping = getattr(c, "der_mapping", None)
    n_der = mapping.n_der if mapping is not None else len(c.der_indices)
    n_pcc, n_gen, n_oltc = len(c.pcc_trafo_indices), len(c.gen_indices), len(c.oltc_trafo_indices)
    o0 = n_der + n_pcc + n_gen
    tr = []
    for t in c.oltc_trafo_indices:
        row = net.trafo.loc[int(t)]
        hv, lv = int(row.hv_bus), int(row.lv_bus)
        gens_lv = [int(g) for g in net.gen.index if int(net.gen.at[g, "bus"]) == lv]
        gens_hv = [int(g) for g in net.gen.index if int(net.gen.at[g, "bus"]) == hv]
        tr.append(dict(trafo=int(t), name=str(row.get("name", "")), hv_bus=hv, lv_bus=lv,
                       hv_bus_in_v=hv in list(map(int, c.voltage_bus_indices)),
                       gens_on_lv=gens_lv, gens_on_hv=gens_hv,
                       tap_min=int(row.tap_min), tap_max=int(row.tap_max),
                       tap_step_percent=float(row.tap_step_percent)))
    out["tso"][str(zid)] = dict(
        v_buses=list(map(int, c.voltage_bus_indices)), oltc=tr,
        gen_indices=list(map(int, c.gen_indices)),
        H_V_oltc=H[:n_v, o0:o0 + n_oltc].tolist(),
        H_V_shunt=H[:n_v, o0 + n_oltc:o0 + n_oltc + len(c.shunt_bus_indices)].tolist(),
        shunt_buses=list(map(int, c.shunt_bus_indices)))
for area, ctrl in sorted(cap["dso_controllers"].items()):
    c = ctrl.config
    H = np.asarray(ctrl._expand_H_to_der_level(ctrl._build_sensitivity_matrix()), float)
    n_if, n_v = len(c.interface_trafo_indices), len(c.voltage_bus_indices)
    mapping = getattr(c, "der_mapping", None)
    n_der = mapping.n_der if mapping is not None else len(c.der_indices)
    n_oltc = len(c.oltc_trafo_indices)
    tr = []
    for t in c.oltc_trafo_indices:
        row = net.trafo3w.loc[int(t)]
        tr.append(dict(trafo=int(t), name=str(row.get("name", "")),
                       tap_side=str(row.get("tap_side", "")), tap_pos=float(row.tap_pos),
                       tap_min=int(row.tap_min), tap_max=int(row.tap_max),
                       tap_step_percent=float(row.tap_step_percent),
                       hv_bus=int(row.hv_bus), mv_bus=int(row.mv_bus), lv_bus=int(row.lv_bus)))
    out["dso"][str(area)] = dict(
        v_buses=list(map(int, c.voltage_bus_indices)), oltc=tr,
        v_set=list(map(float, c.v_setpoints_pu)), v_min=float(c.v_min_pu), v_max=float(c.v_max_pu),
        g_v=float(c.g_v), g_q=float(c.g_q), gamma_oltc_q=float(c.gamma_oltc_q),
        H_V_oltc=H[n_if:n_if + n_v, n_der:n_der + n_oltc].tolist(),
        H_Q_oltc=H[:n_if, n_der:n_der + n_oltc].tolist(),
        H_V_der=H[n_if:n_if + n_v, :n_der].tolist())
(Path(sys.argv[1]) / "H_W1.json").write_text(json.dumps(out, indent=1), "utf-8")
print("ok")
