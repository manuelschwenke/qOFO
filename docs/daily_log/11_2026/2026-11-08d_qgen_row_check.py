"""Q_gen rows of the TSO sensitivity matrix H versus central finite differences.

Question (2026-11-08): on ``rural_700_north_res`` the TS-OFO's cached model
gives dQ_G3/dV_set,G3 = 22 (zone 2, W1).  For a transmission machine the
self-sensitivity should be of the order of hundreds to thousands of Mvar/pu.
Is the Q_gen row of H right, and if not, why?

Method
------
Per operating point (W1 = 2016-01-05 08:00, 12 h; MC20261108 = 2016-01-12
09:00, 3 h, G3 committed) the runner is built up to its pre-loop hook exactly
as in ``2026-11-08c_nr_capture.py`` (base_config of ch_9_6, weights v4 -- the
weights do not enter H).  Then, per TS zone:

* H is taken the way the controller takes it:
  ``ctrl._expand_H_to_der_level(ctrl._build_sensitivity_matrix())``.
* Central finite differences (FD) of a pandapower power flow on three nets:
  - MODEL-PQ: the MODEL net below with every load frozen to constant P/Q at
    its solved value (pandapower's NR Jacobian has no dS_load/dV, so this is
    the like-for-like reference for the analytical formulas);
  - MODEL: a copy of ``ctrl.sensitivities.net`` -- the reduced zone net the
    controller's Jacobian is linearised on (local_sensitivities_tso=True,
    Thevenin tie boundaries), with its ZIP loads.  FD on these nets isolates
    formula / unit / index errors of the analytical H.
  - PLANT: a copy of the full plant net, with the same preparation as
    ``JacobianSensitivities.__init__`` (non-slack P frozen at res_gen,
    single slack).  FD on this net adds the reduction mismatch.
  Power flow: calculate_voltage_angles=True, run_control=False (DER Q and
  DSO taps held), distributed_slack=False, enforce_q_lims=False (PV buses
  hold V; the analytical H assumes the same).
* Steps: V_set of every in-service, dispatchable machine +-0.005 pu (all
  other PV setpoints held); one Q_PCC set point (a Q load at the coupler HV
  bus, load convention, +-10 Mvar); one TS-IBG (sgen q_mvar, generator
  convention, +-10 Mvar); one OLTC (+-1 tap step).
* Recorded per step: Q of every machine (res_gen.q_mvar, generator
  convention), Q of the slack, Q_PCC (res_trafo3w.q_hv_mvar, load
  convention), |V| at the zone's monitored buses.

Further sections: the full-chain dQ_HV/dx . dx/dV_k for the Q_PCC x V_set
entries; the DSO controllers' interface-Q x OLTC entries against FD (DSO
reduced net and plant); the stylised relief of a 10-Mvar Q_gen violation with
the row as built and corrected; and the stage-1 STS lambda on the 35 captures
with the DSO interface-Q x OLTC entries scaled by sn_mva (no power flow).

Units: H rows are meant to be [Mvar per unit input] (V_gen in pu, Q in Mvar,
tap in steps) because the controller compares them against measured Q_gen in
Mvar and capability bounds in Mvar.  FD gives exactly those units.

Run from the worktree root:
    set PYTHONPATH=F:\\qOFO_qgen
    python docs\\daily_log\\11_2026\\2026-11-08d_qgen_row_check.py > ...\\2026-11-08d_qgen_row_check.out
"""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, io, copy, json, pickle, contextlib, dataclasses, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

POINTS = [("W1", "2016-01-05T08:00", 12), ("MC20261108", "2016-01-12T09:00", 3)]
CAPTURES = Path(r"C:\Users\MSCHWE~1\AppData\Local\Temp\259\claude\Z--Python-Projekte-qOFO"
                r"\56eb5879-34eb-40a0-aeee-65e3879c4433\scratchpad\nr_ops")
DV = 0.005     # pu, V_set step
DQ = 10.0      # Mvar, Q_PCC / IBG step
PF_KW = dict(calculate_voltage_angles=True, run_control=False, distributed_slack=False,
             enforce_q_lims=False, tolerance_mva=1e-9, max_iteration=50)
#: weight sets for the stylised consequence calculation (continuous classes)
WEIGHTS = {
    "v4 (ch9_selected_uc_2026_11_v4)": dict(der=0.014211188776637435, pcc=0.8729582741905181,
                                            gen=1701774.6318173753),
    "v5 start (stage1_v5_nr_2026_11_pq10)": dict(der=0.08969523650650978, pcc=0.1765811691675626,
                                                 gen=5000000.0),
}
N_PF = {"model": 0, "plant": 0}


def runpp(net, tag):
    N_PF[tag] += 1
    try:
        pp.runpp(net, init="results", **PF_KW)
    except Exception:  # noqa: BLE001
        pp.runpp(net, init="auto", **PF_KW)


# --------------------------------------------------------------------------
#  Operating point
# --------------------------------------------------------------------------

def build_state(iso, hours):
    from experiments.ch_09_parameter_selection.ch_9_6_full_procedure import base_config
    from configs.paramsets import load_paramset
    from tuning.parameters import apply_to_config
    from tuning._sim_loader import get_run_multi_tso_dso
    w = {k: float(v) for k, v in load_paramset("ch9_selected_uc_2026_11_v4")["fields"].items()
         if k != "sts_coupler_hold_after_bank"}
    cfg = apply_to_config(base_config(), w)
    assert cfg.scenario == "rural_700_north_res", cfg.scenario
    cfg = dataclasses.replace(cfg, start_time=datetime.fromisoformat(iso), n_total_s=3600.0 * hours,
                              contingencies=[], verbose=0, live_plot_controller=False,
                              live_plot_cascade=False, live_plot_overview=False,
                              run_stability_analysis=False, precondition_g_w=False)
    st = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        get_run_multi_tso_dso()(cfg, pre_loop_hook=lambda s: (st.update(s), True)[1])
    return st, cfg


def prepare_plant(net):
    """Copy + the JacobianSensitivities preparation (P frozen, single slack)."""
    w = copy.deepcopy(net)
    if hasattr(w, "controller"):
        w.controller = w.controller.iloc[0:0]
    has_slack = "slack" in w.gen.columns
    for g in w.gen.index:
        if has_slack and bool(w.gen.at[g, "slack"]):
            continue
        if g in w.res_gen.index and np.isfinite(w.res_gen.at[g, "p_mw"]):
            w.gen.at[g, "p_mw"] = float(w.res_gen.at[g, "p_mw"])
    v_before = w.res_bus.vm_pu.copy()
    runpp(w, "plant")
    dv = float(np.nanmax(np.abs(w.res_bus.vm_pu - v_before)))
    return w, dv


def prepare_model(sens):
    w = copy.deepcopy(sens.net)
    if hasattr(w, "controller"):
        w.controller = w.controller.iloc[0:0]
    v_before = w.res_bus.vm_pu.copy()
    runpp(w, "model")
    dv = float(np.nanmax(np.abs(w.res_bus.vm_pu - v_before)))
    return w, dv


def freeze_loads(w):
    """Copy with every load turned into constant P/Q at its solved value.

    The network's loads are voltage dependent (network/ieee39/load_model.py:
    P constant current, Q constant impedance).  pandapower's Newton-Raphson
    Jacobian -- the J that JacobianSensitivities inverts -- does NOT contain
    dS_load/dV (newtonpf only re-evaluates Sbus between iterations), so the
    analytical H is the sensitivity of a constant-power-load network.  FD on
    this frozen copy is the like-for-like reference for the analytical
    formulas; the difference to FD on the ZIP model is the load-model term.
    """
    f = copy.deepcopy(w)
    for col in ("const_z_p_percent", "const_i_p_percent", "const_z_q_percent",
                "const_i_q_percent", "const_z_percent", "const_i_percent"):
        if col in f.load.columns:
            f.load[col] = 0.0
    f.load["p_mw"] = w.res_load.p_mw.reindex(f.load.index).fillna(0.0).values
    f.load["q_mvar"] = w.res_load.q_mvar.reindex(f.load.index).fillna(0.0).values
    if "scaling" in f.load.columns:
        f.load["scaling"] = 1.0
    v_before = f.res_bus.vm_pu.copy()
    runpp(f, "model")
    dv = float(np.nanmax(np.abs(f.res_bus.vm_pu - v_before)))
    return f, dv


def snapshot(w):
    s = {"q_gen": w.res_gen.q_mvar.copy(), "vm": w.res_bus.vm_pu.copy(),
         "q_t3w": (w.res_trafo3w.q_hv_mvar.copy() if len(w.trafo3w) else pd.Series(dtype=float))}
    q_sl = 0.0
    if len(w.ext_grid):
        q_sl += float(w.res_ext_grid.q_mvar.sum())
    if "slack" in w.gen.columns:
        sl = w.gen.index[w.gen.slack.astype(bool) & w.gen.in_service.astype(bool)]
        q_sl += float(w.res_gen.q_mvar.reindex(sl).sum())
        s["slack_gens"] = list(map(int, sl))
    s["q_slack"] = q_sl
    return s


def fd(w, tag, kind, idx, extra=None):
    """Central difference; returns (snap_plus, snap_minus, denominator) or None."""
    if kind == "gen":
        v0 = float(w.gen.at[idx, "vm_pu"])
        w.gen.at[idx, "vm_pu"] = v0 + DV; runpp(w, tag); sp = snapshot(w)
        w.gen.at[idx, "vm_pu"] = v0 - DV; runpp(w, tag); sm = snapshot(w)
        w.gen.at[idx, "vm_pu"] = v0
        return sp, sm, 2 * DV
    if kind == "pcc":   # idx = trafo3w; load convention at the HV bus
        hv = int(w.trafo3w.at[idx, "hv_bus"])
        ld = pp.create_load(w, bus=hv, p_mw=0.0, q_mvar=DQ, name="_FD_PCC")
        runpp(w, tag); sp = snapshot(w)
        w.load.at[ld, "q_mvar"] = -DQ; runpp(w, tag); sm = snapshot(w)
        w.load.drop(index=ld, inplace=True)
        return sp, sm, 2 * DQ
    if kind == "der":   # idx = sgen; generator convention
        q0 = float(w.sgen.at[idx, "q_mvar"])
        w.sgen.at[idx, "q_mvar"] = q0 + DQ; runpp(w, tag); sp = snapshot(w)
        w.sgen.at[idx, "q_mvar"] = q0 - DQ; runpp(w, tag); sm = snapshot(w)
        w.sgen.at[idx, "q_mvar"] = q0
        return sp, sm, 2 * DQ
    if kind == "tap":
        t0 = int(w.trafo.at[idx, "tap_pos"])
        tmin, tmax = int(w.trafo.at[idx, "tap_min"]), int(w.trafo.at[idx, "tap_max"])
        tp, tm = min(t0 + 1, tmax), max(t0 - 1, tmin)
        if tp == tm:
            return None
        w.trafo.at[idx, "tap_pos"] = tp; runpp(w, tag); sp = snapshot(w)
        w.trafo.at[idx, "tap_pos"] = tm; runpp(w, tag); sm = snapshot(w)
        w.trafo.at[idx, "tap_pos"] = t0
        return sp, sm, float(tp - tm)
    if kind == "tap3w":
        t0 = int(w.trafo3w.at[idx, "tap_pos"])
        tmin, tmax = int(w.trafo3w.at[idx, "tap_min"]), int(w.trafo3w.at[idx, "tap_max"])
        tp, tm = min(t0 + 1, tmax), max(t0 - 1, tmin)
        if tp == tm:
            return None
        w.trafo3w.at[idx, "tap_pos"] = tp; runpp(w, tag); sp = snapshot(w)
        w.trafo3w.at[idx, "tap_pos"] = tm; runpp(w, tag); sm = snapshot(w)
        w.trafo3w.at[idx, "tap_pos"] = t0
        return sp, sm, float(tp - tm)
    raise ValueError(kind)


def d(res, key, i):
    sp, sm, den = res
    a, b = sp[key], sm[key]
    if key == "q_slack":
        return (a - b) / den
    if i not in a.index or i not in b.index:
        return np.nan
    return (float(a[i]) - float(b[i])) / den


# --------------------------------------------------------------------------
#  Zone layout
# --------------------------------------------------------------------------

def layout(ctrl):
    c = ctrl.config
    mp = getattr(c, "der_mapping", None)
    L = dict(n_v=len(c.voltage_bus_indices), n_pcc=len(c.pcc_trafo_indices),
             n_i=len(c.current_line_indices), n_gen=len(c.gen_indices),
             n_der=mp.n_der if mp is not None else len(c.der_indices),
             n_oltc=len(c.oltc_trafo_indices), n_shunt=len(c.shunt_bus_indices))
    L["row_qgen"] = L["n_v"] + L["n_pcc"] + L["n_i"]
    L["row_qpcc"] = L["n_v"]
    L["col_pcc"] = L["n_der"]
    L["col_gen"] = L["n_der"] + L["n_pcc"]
    L["col_oltc"] = L["n_der"] + L["n_pcc"] + L["n_gen"]
    L["col_shunt"] = L["col_oltc"] + L["n_oltc"]
    return L


def fmt(x, w=10, p=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return f"{'n/a':>{w}}"
    return f"{x:>{w}.{p}f}"


def ratio(a, b):
    if b is None or not np.isfinite(b) or abs(b) < 1e-9 or a is None or not np.isfinite(a):
        return np.nan
    return a / b


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------

def analyse_point(name, iso, hours, summary):
    t0 = time.time()
    st, cfg = build_state(iso, hours)
    net = st["net"]
    print(f"\n{'=' * 100}\n{name}: start {iso}, {hours} h  (runner built in {time.time() - t0:.0f} s)")
    print(f"plant net: {len(net.bus)} buses, sn_mva = {float(net.sn_mva):g}, "
          f"local_sensitivities_tso = {cfg.local_sensitivities_tso}, "
          f"tie_boundary_equivalent = {cfg.tie_boundary_equivalent!r}")
    print(f"config: g_z_q_gen = {cfg.g_z_q_gen:g}, tso_g_res_sg = {cfg.tso_g_res_sg:g}, "
          f"zone_tso_g_res_sg = {cfg.zone_tso_g_res_sg}, g_z_q_pcc = {cfg.g_z_q_pcc:g}, "
          f"enable_avr_saturation_mode = {getattr(cfg, 'enable_avr_saturation_mode', None)}, "
          f"enforce_q_lims_plant = {getattr(cfg, 'enforce_q_lims_plant', None)}")

    plant, dv_plant = prepare_plant(net)
    print(f"plant FD base: max |dV| vs. the runner's plant solution after P freeze = {dv_plant:.2e} pu")
    ctrls = st["tso_controllers"]

    cap = None
    if (CAPTURES / f"{name}.pkl").exists():
        with open(CAPTURES / f"{name}.pkl", "rb") as fh:
            cap = pickle.load(fh)["extra"]["zones"]

    plant_cache = {}

    def plant_fd(kind, idx):
        key = (kind, idx)
        if key not in plant_cache:
            plant_cache[key] = fd(plant, "plant", kind, idx)
        return plant_cache[key]

    for z in sorted(ctrls):
        ctrl = ctrls[z]
        c = ctrl.config
        L = layout(ctrl)
        ctrl.invalidate_sensitivity_cache()
        H = np.asarray(ctrl._expand_H_to_der_level(ctrl._build_sensitivity_matrix()), float)
        sens = ctrl.sensitivities
        mnet = sens.net
        sn = float(mnet.sn_mva)
        model, dv_model = prepare_model(sens)
        model_pq, dv_pq = freeze_loads(model)
        gens = list(map(int, c.gen_indices))
        oos = np.asarray(ctrl._oos_gen_mask, bool)
        fixed = np.asarray(getattr(ctrl, "_fixed_v_gen_mask", np.zeros(len(gens), bool)), bool)
        sat = np.asarray(getattr(ctrl, "_sat_mode", np.zeros(len(gens), int)))
        oos_oltc = np.asarray(ctrl._oos_oltc_mask, bool)
        print(f"\n--- {name} zone {z} " + "-" * 70)
        print(f"model net: {len(mnet.bus)} buses (plant {len(net.bus)}), sn_mva = {sn:g}; "
              f"model slack gens = {snapshot(model).get('slack_gens')}, ext_grids = {len(model.ext_grid)}; "
              f"re-solve max |dV| = {dv_model:.2e} pu; constant-PQ copy max |dV| = {dv_pq:.2e} pu; "
              f"loads: const_i_p = {sorted(set(model.load.get('const_i_p_percent', pd.Series([0])).round(1)))}, "
              f"const_z_q = {sorted(set(model.load.get('const_z_q_percent', pd.Series([0])).round(1)))} %")
        print(f"layout: {L}")
        print(f"gens {gens}: in_service(plant) = {[bool(net.gen.at[g, 'in_service']) for g in gens]}, "
              f"oos_mask = {oos.tolist()}, fixed_v_mask = {fixed.tolist()}, sat_mode = {sat.tolist()}")
        print(f"oltc {list(map(int, c.oltc_trafo_indices))}: oos_mask = {oos_oltc.tolist()}")
        gz = np.broadcast_to(np.asarray(ctrl.params.g_z, float), (H.shape[0],))
        print(f"g_z on Q_gen rows = {gz[L['row_qgen']:L['row_qgen'] + L['n_gen']].tolist()}, "
              f"alpha = {ctrl.params.alpha}, g_res_sg = {c.g_res_sg}, g_q_tso = {c.g_q_tso}")
        if cap is not None and z in cap and cap[z]["H"].shape == H.shape:
            r0 = L["row_qgen"]
            print(f"capture pickle vs recomputed H: max |diff| whole H = {np.max(np.abs(cap[z]['H'] - H)):.2e}, "
                  f"Q_gen block = {np.max(np.abs(cap[z]['H'][r0:] - H[r0:])):.2e}")

        # machine data
        for k, g in enumerate(gens):
            if not bool(net.gen.at[g, "in_service"]):
                continue
            gb = int(net.gen.at[g, "bus"])
            mt = net.trafo.index[(net.trafo.lv_bus == gb) | (net.trafo.hv_bus == gb)].tolist()
            info = ""
            for t in mt:
                sn_t = float(net.trafo.at[t, "sn_mva"]); vk = float(net.trafo.at[t, "vk_percent"])
                x_sys = vk / 100.0 * float(net.sn_mva) / sn_t
                info += (f" machine trafo {t}: {sn_t:g} MVA, vk {vk:g} %, x = {x_sys:.4f} pu(sys), "
                         f"V/x bound = {float(net.res_bus.at[gb, 'vm_pu']) / x_sys * float(net.sn_mva):.0f} Mvar/pu;")
            gp = ctrl.actuator_bounds.compute_gen_q_bounds(
                np.array([float(net.res_gen.at[gg, "p_mw"]) for gg in gens]),
                np.array([float(net.res_gen.at[gg, "vm_pu"]) for gg in gens]))
            print(f"  gen {g} ({net.gen.at[g, 'name']}): bus {gb} ({float(net.bus.at[gb, 'vn_kv']):g} kV), "
                  f"P = {float(net.res_gen.at[g, 'p_mw']):.0f} MW, Q = {float(net.res_gen.at[g, 'q_mvar']):.1f} Mvar, "
                  f"V = {float(net.res_gen.at[g, 'vm_pu']):.4f} pu, capability [{gp[0][k]:.0f}, {gp[1][k]:.0f}] Mvar, "
                  f"table [{float(net.gen.at[g, 'min_q_mvar']):.0f}, {float(net.gen.at[g, 'max_q_mvar']):.0f}];{info}")

        # actuators to step
        acts = []
        for k, g in enumerate(gens):
            if oos[k] or not bool(net.gen.at[g, "in_service"]):
                continue
            acts.append(("gen", g, L["col_gen"] + k, f"V_set gen {g}", "pu"))
        rows_q = [(k, g) for k, g in enumerate(gens) if not oos[k] and bool(net.gen.at[g, "in_service"])]
        if L["n_pcc"] and rows_q:
            kq = rows_q[0][0]
            cols = [L["col_pcc"] + j for j in range(L["n_pcc"])]
            j = int(np.argmax(np.abs(H[L["row_qgen"] + kq, cols])))
            t = int(c.pcc_trafo_indices[j])
            acts.append(("pcc", t, L["col_pcc"] + j, f"Q_PCC,set trafo3w {t} (load conv.)", "Mvar"))
        if L["n_der"]:
            s = int(c.der_indices[0])
            acts.append(("der", s, 0, f"Q IBG sgen {s} (gen. conv.)", "Mvar"))
        for k, t in enumerate(c.oltc_trafo_indices):
            colk = L["col_oltc"] + k
            if not oos_oltc[k] and np.any(H[:, colk] != 0.0):
                t = int(t)
                acts.append(("tap", t, colk,
                             f"tap trafo {t} (hv {int(net.trafo.at[t, 'hv_bus'])}, lv {int(net.trafo.at[t, 'lv_bus'])},"
                             f" side {net.trafo.at[t, 'tap_side']}, {float(net.trafo.at[t, 'tap_step_percent']):g} %/step)",
                             "step"))
                break

        print(f"\n  rows: H as built | H x sn_mva (V_set/tap columns only) | FD model-PQ | FD model (ZIP) | "
              f"FD plant | FD_model-PQ/H | FD_plant/H   [Mvar per unit input]")
        for kind, idx, col, label, unit in acts:
            has = kind != "der" or idx in model.sgen.index
            rq = fd(model_pq, "model", kind, idx) if has else None
            rm = fd(model, "model", kind, idx) if has else None
            rp = plant_fd(kind, idx)
            scaled = kind in ("gen", "tap")
            print(f"  column {col:2d}: {label} [{unit}]")

            def line(lab, h, fq, fm, fp):
                hs = fmt(h * sn, 11, 1) if scaled else f"{'-':>11s}"
                print(f"     {lab:<20s}: {fmt(h)} {hs} {fmt(fq, 11, 1)} {fmt(fm, 11, 1)} {fmt(fp, 11, 1)} "
                      f"{fmt(ratio(fq, h), 9, 2)} {fmt(ratio(fp, h), 9, 2)}")

            for k, g in rows_q:
                h = float(H[L["row_qgen"] + k, col])
                fq = d(rq, "q_gen", g) if rq else np.nan
                fm = d(rm, "q_gen", g) if rm else np.nan
                fp = d(rp, "q_gen", g) if rp else np.nan
                line(f"dQ_gen{g} (row {L['row_qgen'] + k})", h, fq, fm, fp)
                summary.append(dict(point=name, zone=z, column=label, kind=kind, row=f"Q_gen{g}",
                                    H=h, H_x_sn=h * sn if scaled else h, fd_model_pq=fq,
                                    fd_model=fm, fd_plant=fp))
            fq_s = d(rq, "q_slack", None) if rq else np.nan
            fm_s = d(rm, "q_slack", None) if rm else np.nan
            fp_s = d(rp, "q_slack", None) if rp else np.nan
            print(f"     {'dQ_slack':<20s}: {'(no row)':>10s} {'':>11s} {fmt(fq_s, 11, 1)} {fmt(fm_s, 11, 1)} "
                  f"{fmt(fp_s, 11, 1)}")
            # Q_PCC rows (same unit question), two largest |FD model-PQ| entries
            if L["n_pcc"]:
                qp = []
                for j, t in enumerate(c.pcc_trafo_indices):
                    if kind == "pcc" and int(t) == idx:
                        continue  # identity entry by design (closed-loop DSO), not a physical sensitivity
                    h = float(H[L["row_qpcc"] + j, col])
                    fq = d(rq, "q_t3w", int(t)) if rq else np.nan
                    fm = d(rm, "q_t3w", int(t)) if rm else np.nan
                    fp = d(rp, "q_t3w", int(t)) if rp else np.nan
                    qp.append((abs(fq) if np.isfinite(fq) else 0.0, j, t, h, fq, fm, fp))
                qp.sort(key=lambda r: r[0], reverse=True)
                for _, j, t, h, fq, fm, fp in qp[:2]:
                    line(f"dQ_PCC{int(t)} (row {L['row_qpcc'] + j})", h, fq, fm, fp)
                    summary.append(dict(point=name, zone=z, column=label, kind=kind, row=f"Q_PCC{int(t)}",
                                        H=h, H_x_sn=h * sn if scaled else h, fd_model_pq=fq,
                                        fd_model=fm, fd_plant=fp))
            # V rows: sanity (pu rows, no base factor expected)
            hv = H[:L["n_v"], col]
            buses = [int(b) for b in c.voltage_bus_indices]
            out = []
            for lab, r in (("model-PQ", rq), ("model", rm), ("plant", rp)):
                fv = np.array([d(r, "vm", b) if r else np.nan for b in buses])
                ok = np.isfinite(fv)
                if ok.any():
                    e = np.abs(hv - fv); e[~ok] = -1
                    i = int(np.argmax(e))
                    out.append(f"{lab} {e[i]:.2e} (bus {buses[i]}: H {hv[i]:.4g}, FD {fv[i]:.4g})")
            print(f"     V rows: max|H| = {np.max(np.abs(hv)):.4g}; max|H - FD|: " + "; ".join(out))
            if kind == "gen":
                gb = int(net.gen.at[idx, "bus"])
                print(f"     dV_bus(gen terminal)/dV_set: model {d(rm, 'vm', gb):.4f}, plant {d(rp, 'vm', gb):.4f}")

        # analytical builder called directly (bypasses the controller assembly)
        is_buses = [int(net.gen.at[g, "bus"]) for k, g in enumerate(gens) if not oos[k]]
        if is_buses:
            M, _, _ = sens.compute_dQgen_dVgen_matrix(is_buses, is_buses)
            print(f"  compute_dQgen_dVgen_matrix (direct call) = {np.round(M, 3).tolist()}  "
                  f"(x sn_mva = {np.round(M * sn, 1).tolist()})")

        # Q_PCC x V_set: compute_dQtrafo3w_hv_dVgen keeps only the two branch
        # voltage terms (dQ/dV_hv, dQ/dV_star); the full chain dQ_HV/dx . dx/dV_k
        # (_compute_dQhv_dx_3w also carries the angle terms) is the reference.
        if L["n_pcc"] and is_buses:
            from sensitivity.index_helper import _get_trafo3w_hv_branch_data, pp_bus_to_ppc_bus
            gb = is_buses[0]
            dg, _ = sens._compute_dg_dVk(pp_bus_to_ppc_bus(mnet, gb))
            dx = -sens.J_inv @ dg
            rows = []
            for t in c.pcc_trafo_indices[:9]:
                dd = _get_trafo3w_hv_branch_data(mnet, int(t))
                full = float(sens._compute_dQhv_dx_3w(dd) @ dx) * sn
                asb = float(sens.compute_dQtrafo3w_hv_dVgen(int(t), pp_bus_to_ppc_bus(mnet, gb))) * sn
                rows.append(f"{int(t)}: {asb:+.2f} / {full:+.2f}")
            print(f"  Q_PCC x V_set(bus {gb}) [Mvar/pu]  as-built x sn_mva / full chain x sn_mva: "
                  + ", ".join(rows))

        # stylised consequence: relief of a Q_gen upper-limit violation z0
        g_w_vec, g_u_vec = ctrl._get_per_variable_weights()
        n_u = H.shape[1]
        g_w_vec = (np.broadcast_to(np.asarray(ctrl.params.g_w, float), (n_u,)) if g_w_vec is None
                   else np.asarray(g_w_vec, float))
        alpha = float(ctrl.params.alpha)
        classes = (["der"] * L["n_der"] + ["pcc"] * L["n_pcc"] + ["gen"] * L["n_gen"])
        n_c = len(classes)
        for k, g in rows_q:
            h_wrong = H[L["row_qgen"] + k, :n_c].copy()
            h_fix = h_wrong.copy()
            h_fix[L["col_gen"]:L["col_gen"] + L["n_gen"]] *= sn
            gzq = float(gz[L["row_qgen"] + k])
            for wname, wc in WEIGHTS.items():
                gw = np.array([wc[cl] for cl in classes])
                gw = np.where(gw > 0, gw, np.inf)
                z0 = 10.0
                out = []
                for lab, h in (("H as built", h_wrong), ("H fixed", h_fix)):
                    a = alpha * h
                    s = float(np.sum(a * a / gw))
                    wv = -gzq * z0 * (a / gw) / (1.0 + gzq * s)
                    pred = float(a @ wv)
                    act = float(alpha * h_fix @ wv)
                    du_gen = alpha * wv[L["col_gen"]:L["col_gen"] + L["n_gen"]]
                    rel = {cl: float(np.sum((alpha * h_fix * wv)[np.array(classes) == cl])) for cl in ("der", "pcc", "gen")}
                    out.append(f"{lab}: predicted dQ = {pred:+.2f}, actual (fixed row) = {act:+.2f} Mvar, "
                               f"dV_set = {np.round(du_gen, 5).tolist()} pu, actual relief by class "
                               f"{ {kk: round(v, 2) for kk, v in rel.items()} }")
                print(f"  [z0 = {z0:g} Mvar over Q_max of gen {g}; g_z = {gzq:g}; {wname}]")
                for o in out:
                    print(f"     {o}")

    # ---- Layer 2: DSO interface-Q x OLTC tap (compute_dQtrafo3w_hv_ds, same
    #      builder family as the TS Q rows; checked for the base question only)
    print(f"\n--- {name} DSO interface-Q x OLTC tap (Layer 2, same base question) " + "-" * 30)
    print("  rows: H as built | H x sn_mva | FD DSO model | FD plant | FD_DSO-model/H | FD_plant/H  [Mvar/step]")
    for area, dctrl in sorted(st["dso_controllers"].items()):
        dc = dctrl.config
        dctrl.invalidate_sensitivity_cache()
        Hd = np.asarray(dctrl._build_sensitivity_matrix(), float)
        n_if, n_ol, n_sh = (len(dc.interface_trafo_indices), len(dc.oltc_trafo_indices),
                            len(dc.shunt_bus_indices))
        n_b = Hd.shape[1] - n_ol - n_sh
        if not n_ol:
            continue
        t = int(dc.oltc_trafo_indices[0])
        kind = "tap3w" if t in net.trafo3w.index else "tap"
        dmodel, _ = prepare_model(dctrl.sensitivities)
        rm = fd(dmodel, "model", kind, t)
        rp = plant_fd(kind, t)
        tab = net.trafo3w if kind == "tap3w" else net.trafo
        print(f"  {area}: OLTC {kind} {t} ({float(tab.at[t, 'tap_step_percent']):g} %/step, "
              f"side {tab.at[t, 'tap_side']}); interface trafo3w {list(map(int, dc.interface_trafo_indices))}; "
              f"sn_mva(model) = {float(dctrl.sensitivities.net.sn_mva):g}")
        for i, ti in enumerate(dc.interface_trafo_indices):
            h = float(Hd[i, n_b])
            fm = d(rm, "q_t3w", int(ti)) if rm else np.nan
            fp = d(rp, "q_t3w", int(ti)) if rp else np.nan
            print(f"     dQ_iface{int(ti):<3d}: {fmt(h)} {fmt(h * float(net.sn_mva), 11, 1)} {fmt(fm, 11, 1)} "
                  f"{fmt(fp, 11, 1)} {fmt(ratio(fm, h), 9, 2)} {fmt(ratio(fp, h), 9, 2)}")
            summary.append(dict(point=name, zone=area, column=f"{kind} {t}", kind="dso_" + kind,
                                row=f"Q_iface{int(ti)}", H=h, H_x_sn=h * float(net.sn_mva),
                                fd_model_pq=np.nan, fd_model=fm, fd_plant=fp))

    print(f"\n{name}: power flows so far: model {N_PF['model']}, plant {N_PF['plant']}; "
          f"elapsed {time.time() - t0:.0f} s")


def lambda_consequence():
    """STS loop gain of the stage-1 screen with the DSO Q_iface x OLTC entries
    scaled by sn_mva (the fix the FD above supports), on the 35 captures.

    S_Q = H_Q^T H_Q, so scaling the dso_oltc columns of H_Q by 100 scales the
    (oltc, oltc) block of S_Q by 1e4 and the (oltc, other) blocks by 100.
    No power flow.  TS layers are unaffected (their S_Q enters with
    tso_g_q_pcc = 0).
    """
    from tuning import stage1_screen as s1
    from configs.paramsets import load_paramset
    caps = sorted(CAPTURES.glob("*.pkl"))
    if not caps:
        print("\n(no captures found; lambda consequence skipped)")
        return
    w4 = {k: float(v) for k, v in load_paramset("ch9_selected_uc_2026_11_v4")["fields"].items()
          if k != "sts_coupler_hold_after_bank"}
    # stage1_v5_nr_2026_11_pq10 is untracked in the v5 campaign checkout; its fields, read 2026-11-08:
    w5 = dict(g_v=40000.0, g_q=0.4, dso_g_v=10000.0, g_w_der=0.08969523650650978,
              g_w_pcc=0.1765811691675626, g_w_gen=5000000.0, g_w_tso_oltc=21.307839984260788,
              g_w_tso_shunt=14.9233182123854, g_w_dso_der=1.5280510783923393,
              g_w_dso_oltc=11.342714141101698)
    ops = [pickle.load(open(f, "rb"))["op"] for f in caps]
    print(f"\nCONSEQUENCE for the stage-1 lambda screen (STS layers, banks excluded as in v4/v5), "
          f"{len(ops)} captures; ceiling {s1.select_ceiling():.3f}")
    print("  layer       | weights  | lambda as built: median / max | DSO Q_iface x OLTC scaled by sn_mva: "
          "median / max | diag share of g_q S_Q on the OLTC column (corrected)")
    for wname, w in (("v4", w4), ("v5 start", w5)):
        res = {}
        for op in ops:
            for lc in op.layers:
                if lc.layer != "STS":
                    continue
                g_v, g_q = s1._layer_weights(lc, w)
                try:
                    g_w = lc.g_w_vector(w)
                except KeyError:
                    continue
                lam0 = s1.lambda_max_B(lc, g_v, g_q, g_w, exclude_classes=("shunt",))
                D = np.array([float(DQ_SCALE) if cl == "dso_oltc" else 1.0 for cl in lc.classes])
                lc2 = dataclasses.replace(lc, S_Q=D[:, None] * lc.S_Q * D[None, :])
                lam1 = s1.lambda_max_B(lc2, g_v, g_q, g_w, exclude_classes=("shunt",))
                io = [i for i, cl in enumerate(lc.classes) if cl == "dso_oltc"]
                share = (float(np.mean([g_q * lc2.S_Q[i, i] / (g_q * lc2.S_Q[i, i] + g_v * lc.S_V[i, i])
                                        for i in io])) if io else np.nan)
                res.setdefault(lc.key, []).append((lam0, lam1, share))
        for key in sorted(res):
            a = np.array(res[key])
            print(f"  {key:<11s} | {wname:<8s} | {np.median(a[:, 0]):6.3f} / {np.max(a[:, 0]):6.3f}"
                  f"              | {np.median(a[:, 1]):6.3f} / {np.max(a[:, 1]):6.3f}"
                  f"                         | {np.nanmedian(a[:, 2]):.2f}")


DQ_SCALE = 100.0   # sn_mva of rural_700_north_res (printed above as plant sn_mva)


def main():
    print(f"2026-11-08d Q_gen row check -- {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"code: {ROOT}; pandapower {pp.__version__}; DV = {DV} pu, DQ = {DQ} Mvar")
    summary = []
    for name, iso, hours in POINTS:
        analyse_point(name, iso, hours, summary)
    df = pd.DataFrame(summary)
    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.float_format", "{:.3f}".format):
        print("\nSUMMARY (Mvar per unit input)")
        print(df.to_string(index=False))
        for prefix, title in (("Q_gen", "Q_gen rows"), ("Q_PCC", "Q_PCC rows (physical entries only)"),
                              ("Q_iface", "DSO interface-Q rows x OLTC")):
            q = df[df.row.str.startswith(prefix)].copy()
            if not len(q):
                continue
            q = q[np.abs(q.H) > 1e-6]
            for col in ("fd_model_pq", "fd_model", "fd_plant"):
                q[col + "/H"] = q[col] / q.H
            print(f"\nratio FD/H by input kind, {title} (entries with |H| > 1e-6):")
            print(q.groupby("kind")[["fd_model_pq/H", "fd_model/H", "fd_plant/H"]]
                  .agg(["median", "min", "max", "count"]).to_string())
    print(f"\ntotal power flows: model {N_PF['model']}, plant {N_PF['plant']}")
    lambda_consequence()


if __name__ == "__main__":
    main()
