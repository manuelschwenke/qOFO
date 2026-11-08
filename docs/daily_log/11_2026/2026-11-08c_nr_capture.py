"""Capture TS/STS controller sensitivities on rural_700_north_res (no simulation).

Per condition: the stage-1 OperatingPoint (lambda screen) plus, per TS zone, the
full H (rows [V | Q_PCC | I | Q_gen]), column counts, actuator Q capabilities, and
per STS the IBG Q capability.  Run with the F:\\qOFO_net code base.
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import sys, json, io, pickle, contextlib, dataclasses, time, traceback
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

CODE = r"F:\qOFO_net"
OUT = Path(r"C:\Users\MSCHWE~1\AppData\Local\Temp\259\claude\Z--Python-Projekte-qOFO\56eb5879-34eb-40a0-aeee-65e3879c4433\scratchpad\nr_ops")
MINI = Path(r"F:\qOFO_net\results\ch_08_setup\network_scarcity\_remote_e5power\0007_2026-11-08_160221_mini_mc_north_res\prescreen")


def jobs():
    from_w = [("W1", "2016-01-05T08:00", 12), ("W2", "2016-02-22T13:00", 12), ("W3", "2016-01-04T08:00", 12),
              ("W4", "2016-11-26T10:00", 12), ("W5", "2016-02-03T20:00", 12), ("W6", "2016-07-18T14:00", 12)]
    off, on = [], []
    for f in sorted(MINI.glob("*.json")):
        d = json.loads(f.read_text())
        if not d.get("valid"):
            continue
        (off if 1 in d.get("uc_offline", []) else on).append((f"MC{d['seed']}", d["start_time"][:16], 3))
    return from_w + off + on


def capture(job):
    name, iso, hours = job
    sys.path.insert(0, CODE)
    import warnings; warnings.filterwarnings("ignore")
    out = OUT / f"{name}.pkl"
    if out.exists():
        return name, "cached", 0.0
    t0 = time.time()
    try:
        from experiments.ch_09_parameter_selection.ch_9_6_full_procedure import base_config
        from configs.paramsets import load_paramset
        from tuning.parameters import apply_to_config
        from tuning import stage1_screen as s1
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
        net, meta = st["net"], st["meta"]
        bank_sts = {int(lv): str(hv.net_id) for hv in meta.hv_networks for lv in hv.coupling_lv_bus_indices}
        op = s1.OperatingPoint(condition=name)
        extra = {"name": name, "start": iso, "hours": hours, "zones": {}, "sts": {}}
        for zid, ctrl in sorted(st["tso_controllers"].items()):
            op.layers.append(s1._tso_curvature(zid, ctrl, bank_sts))
            c = ctrl.config
            H = np.asarray(ctrl._expand_H_to_der_level(ctrl._build_sensitivity_matrix()), float)
            mp = getattr(c, "der_mapping", None)
            n_der = mp.n_der if mp is not None else len(c.der_indices)
            ab = ctrl.actuator_bounds
            der_p = np.array([float(net.sgen.at[s, "p_mw"]) for s in c.der_indices]) if len(c.der_indices) else np.zeros(0)
            dq = ab.compute_der_q_bounds(der_p) if len(der_p) == ab.n_ders and len(der_p) else (np.zeros(0), np.zeros(0))
            gq = (np.zeros(0), np.zeros(0))
            if len(c.gen_indices) and ab.gen_params is not None:
                gp = np.array([float(net.res_gen.at[g, "p_mw"]) for g in c.gen_indices])
                gv = np.array([float(net.res_gen.at[g, "vm_pu"]) for g in c.gen_indices])
                gq = ab.compute_gen_q_bounds(gp, gv)
            extra["zones"][int(zid)] = dict(
                H=H, n_v=len(c.voltage_bus_indices), n_pcc=len(c.pcc_trafo_indices),
                n_i=len(c.current_line_indices), n_gen=len(c.gen_indices), n_der=n_der,
                n_oltc=len(c.oltc_trafo_indices), n_shunt=len(c.shunt_bus_indices),
                gen_indices=list(map(int, c.gen_indices)),
                gen_in_service=[bool(net.gen.at[g, "in_service"]) for g in c.gen_indices],
                der_q=(np.asarray(dq[0]), np.asarray(dq[1])), gen_q=(np.asarray(gq[0]), np.asarray(gq[1])),
                pcc_dso=list(c.pcc_dso_controller_ids),
                v_meas=np.array([float(net.res_bus.at[b, "vm_pu"]) for b in c.voltage_bus_indices]))
        for area, ctrl in sorted(st["dso_controllers"].items()):
            op.layers.append(s1._dso_curvature(area, ctrl))
            c, ab = ctrl.config, ctrl.actuator_bounds
            p = np.array([float(net.sgen.at[s, "p_mw"]) for s in c.der_indices])
            try:
                qmin, qmax = ab.compute_der_q_bounds(p)
                extra["sts"][area] = (float(np.sum(qmin)), float(np.sum(qmax)))
            except Exception as e:  # noqa: BLE001
                extra["sts"][area] = repr(e)
        with out.open("wb") as fh:
            pickle.dump({"op": op, "extra": extra}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return name, "ok", time.time() - t0
    except Exception:  # noqa: BLE001
        return name, traceback.format_exc()[-1500:], time.time() - t0


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    J = jobs()
    only = sys.argv[1:]
    if only:
        J = [j for j in J if j[0] in only]
    print(f"{len(J)} captures", flush=True)
    with ProcessPoolExecutor(max_workers=min(10, len(J))) as ex:
        for name, status, dt in ex.map(capture, J):
            print(f"{name}: {status if status in ('ok', 'cached') else 'FAILED'} {dt:.0f}s", flush=True)
            if status not in ("ok", "cached"):
                print(status, flush=True)
