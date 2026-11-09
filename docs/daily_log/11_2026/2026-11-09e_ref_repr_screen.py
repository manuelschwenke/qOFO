"""Representativeness screen of the Ch 10 reference scenarios on the new network (plant only).

Re-applies the plant-only criteria of the E2 declaration (docs/analysis/2026-11-06_scenario_E2_
declaration.md, "Plant-only bounds" and "Choice of the line") to rural_700_north_res, and the same
code to rural_700 for a like-for-like comparison (the declaration's own scripts candE.py /
cand_mismatch.py are not available; this follows scarcity_E_bounds{,2}.py of session 29f1fa99).

Start point: the runner's plant at t = 0 of the period (pre_loop_hook, 360-min config so that the
unit commitment is the REF's), code F:\\qOFO_ref.

Per state:
  frozen : events applied; only the plant's IBG Q(V) loops react (controllers frozen)
  TS-a2  : area 2's own TS resources at their raising limit: G3's AVR at 1.05 pu if in service,
           the area-2 MSCs fully in, MSRs out, area-2 TS IBG at Q_max (none on north_res);
           neighbours as at t = 0; STS IBG frozen at their t = 0 Q (loop bypassed)
  TS-all : TS-a2 plus every in-service AVR at 1.05 pu and every TS IBG at Q_max
  STS a  : TS-a2 plus the area-2 STS IBG at a common fraction a of their upward capability
           (plant op diagram: 4120 v2, reserved box on north_res), a such that the area-2 level
           returns to 1.03 pu; coupler taps frozen
"Level" = mean deviation of IEEE 3..14 from 1.03 pu [mpu].

Usage: python -I ref_repr_screen.py <network> <case> [out.pkl]
  case E2  : E2 stages 0-4 + end states of the line candidates
  case P2  : Period II t = 0, P2R hold state, P2R_B03B05 hold state
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
import contextlib, copy, dataclasses, io, pickle, time, warnings
from datetime import datetime
import numpy as np
import pandapower as pp
warnings.filterwarnings("ignore")

from experiments.ch_11_case2_multisystem import variants as v
from experiments.ch_11_case2_multisystem.ref import scenarios as sc
from experiments.runners.multi_tso_dso import run_multi_tso_dso
from controller.der_qv_local_loop import _qv_capability

NETWORK, CASE = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None
A2 = list(range(2, 14))                 # pp buses of IEEE 3..14
REF = 1.03
LINES = {"3-4": 4, "2-3": 2, "4-14": 7, "3-18": 5, "17-27": 25, "14-15": 18, "9-39": 14}


def capture(scenario, start=None):
    cfg = v.resolved("O3", 360, sc.events(scenario), 20261027)
    cfg = dataclasses.replace(cfg, result_dir="", verbose=0, scenario=NETWORK)
    st = start or sc.start_time(scenario)
    if st is not None:
        cfg = dataclasses.replace(cfg, start_time=st)
    out = {}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        run_multi_tso_dso(cfg, pre_loop_hook=lambda s: out.update(net=copy.deepcopy(s["net"])) or True)
    return out["net"], cfg


def solve(n):
    pp.runpp(n, numba=False, enforce_q_lims=True, distributed_slack=True,
             calculate_voltage_angles=True, max_iteration=50, max_iter=300, run_control=True)


def sts_of(name):
    return str(name).split("|")[0]


def ev_gen(i, on=False):
    def f(n):
        n.gen.at[i, "in_service"] = on
        for t in n.trafo.index:
            if str(n.trafo.at[t, "name"]).startswith(f"MachineTrf|gen{i}_"):
                n.trafo.at[t, "in_service"] = on
    return f


def ev_line(name):
    i = LINES[name]
    return lambda n: n.line.__setitem__("in_service", n.line.in_service.where(n.line.index != i, False))


def ev_load(bus, q, p=0.0):
    return lambda n: pp.create_load(n, bus=bus, p_mw=p, q_mvar=q, name=f"EV|{bus}|{q}")


class Screen:
    def __init__(self, net0):
        self.net0 = net0
        n = copy.deepcopy(net0)
        solve(n)
        self.ts = set(int(b) for b in net0.bus.index[net0.bus.vn_kv >= 300])
        sg = net0.sgen
        self.tsibg = [i for i in sg.index if int(sg.at[i, "bus"]) in self.ts and sg.at[i, "in_service"]]
        self.ds = [i for i in sg.index if int(sg.at[i, "bus"]) not in self.ts]
        # area-2 STS: those whose coupling transformers sit at an IEEE 3..14 bus
        self.sts2 = sorted({sts_of(net0.trafo3w.at[t, "name"]) for t in net0.trafo3w.index
                            if int(net0.trafo3w.at[t, "hv_bus"]) in A2})
        self.ds2 = [i for i in self.ds if sts_of(sg.at[i, "name"]) in self.sts2]
        self.q0 = n.res_sgen.q_mvar.loc[self.ds].to_dict()
        self.gen_on = {str(net0.gen.at[g, "name"]): bool(net0.gen.at[g, "in_service"]) for g in net0.gen.index}
        self.tsibg_a2 = [i for i in self.tsibg if int(sg.at[i, "bus"]) in A2]
        self.diag_col = "op_diagram" if "op_diagram" in sg.columns else None

    def qmax(self, n, i):
        od = str(n.sgen.at[i, self.diag_col]) if self.diag_col else ""
        return _qv_capability(float(n.sgen.at[i, "sn_mva"]), od, float(n.sgen.at[i, "p_mw"]))[1]

    def ts_a2(self, n):
        for i in self.ds:
            n.sgen.at[i, "q_set_mvar"] = self.q0[i]
            n.sgen.at[i, "qv_deadband_pu"] = 1.0
        for i in self.tsibg_a2:
            n.sgen.at[i, "q_set_mvar"] = 1e4
        for s in n.shunt.index:
            nm = str(n.shunt.at[s, "name"])
            if sts_of(nm) in self.sts2:
                n.shunt.at[s, "step"] = int(n.shunt.at[s, "max_step"]) if "MSC" in nm else 0
        if bool(n.gen.at[1, "in_service"]):
            n.gen.at[1, "vm_pu"] = 1.05

    def ts_all(self, n):
        self.ts_a2(n)
        n.gen.loc[n.gen.in_service, "vm_pu"] = 1.05
        for i in self.tsibg:
            n.sgen.at[i, "q_set_mvar"] = 1e4

    def with_sts(self, n, a):
        for i in self.ds2:
            n.sgen.at[i, "q_set_mvar"] = a * self.qmax(n, i)
            n.sgen.at[i, "qv_deadband_pu"] = 1.0

    def pf(self, events, mode, a=None):
        n = copy.deepcopy(self.net0)
        for e in events:
            e(n)
        if mode == "TS-a2":
            self.ts_a2(n)
        elif mode == "TS-all":
            self.ts_all(n)
        if a is not None:
            self.ts_a2(n)
            self.with_sts(n, a)
        try:
            solve(n)
        except Exception as exc:  # noqa: BLE001
            return dict(ok=False, err=type(exc).__name__)
        vm = n.res_bus.vm_pu.loc[A2].to_numpy()
        return dict(ok=True, level=(vm.mean() - REF) * 1e3, e=np.sqrt(np.mean((vm - REF) ** 2)) * 1e3,
                    vmin=vm.min(), ieee_min=A2[int(np.argmin(vm))] + 1,
                    q_g1=float(n.res_ext_grid.q_mvar.sum()) if len(n.ext_grid) else np.nan,
                    q_ds2=float(n.res_sgen.q_mvar.loc[self.ds2].sum()),
                    qcap_ds2=float(sum(self.qmax(n, i) for i in self.ds2)))

    def alpha(self, events):
        rl, rh = self.pf(events, None, 0.0), self.pf(events, None, 1.0)
        if not (rl["ok"] and rh["ok"]):
            return None, None
        if rl["level"] >= 0:
            return 0.0, rl
        if rh["level"] < 0:
            return float("inf"), rh
        lo, hi, ra, a = 0.0, 1.0, rh, 1.0
        for _ in range(8):
            a = lo + (hi - lo) * (-rl["level"]) / (rh["level"] - rl["level"])
            a = min(max(a, lo + 0.05 * (hi - lo)), hi - 0.05 * (hi - lo))
            ra = self.pf(events, None, a)
            if not ra["ok"] or abs(ra["level"]) < 0.3:
                break
            if ra["level"] < 0:
                lo, rl = a, ra
            else:
                hi, rh = a, ra
        return a, ra


def row(scr, tag, events, with_alpha=True):
    r = {"tag": tag}
    for mode in ("frozen", "TS-a2", "TS-all"):
        r[mode] = scr.pf(events, None if mode == "frozen" else mode)
    if with_alpha and r["TS-a2"].get("ok") and r["TS-a2"]["level"] < 0:
        r["alpha"], r["alpha_pf"] = scr.alpha(events)
    else:
        r["alpha"], r["alpha_pf"] = (0.0, None) if r["TS-a2"].get("ok") else (None, None)
    f, a2, al = r["frozen"], r["TS-a2"], r["TS-all"]
    fmt = lambda x: f"{x['level']:+6.1f}" if x.get("ok") else f"PF-{x.get('err', '?')[:8]}"
    fmin = f"{f['vmin']:.3f} (IEEE {f['ieee_min']})" if f.get("ok") else "--"
    a = r["alpha"]
    astr = "--" if a is None else ("not scarce" if a == 0 else ("> 100 %" if a == float("inf") else f"{100 * a:.0f} %"))
    print(f"{tag:34s} | frozen {fmt(f)} min {fmin:16s} | TS-a2 {fmt(a2)} | TS-all {fmt(al)} | STS share {astr}",
          flush=True)
    return r


if __name__ == "__main__":
    t0 = time.time()
    res = {"network": NETWORK, "case": CASE, "rows": []}
    if CASE == "E2":
        net0, cfg = capture("E2")
        scr = Screen(net0)
        print(f"network {NETWORK} | start {cfg.start_time} | committed: "
              f"{[k for k, on in scr.gen_on.items() if on]} | off: {[k for k, on in scr.gen_on.items() if not on]}")
        print(f"area-2 STS {scr.sts2} ({len(scr.ds2)} IBG) | TS IBG {len(scr.tsibg)} (area 2: {len(scr.tsibg_a2)}) "
              f"| op_diagram column: {scr.diag_col}", flush=True)
        G3o, G3b, L = ev_gen(1), ev_gen(1, on=True), ev_load(3, 600.0)
        stages = [("0: t = 0", []), ("1: G3 out", [G3o]), ("2: G3 out + 600", [G3o, L]),
                  ("3: G3 back + 600", [G3o, L, G3b]), ("4: + 3-4 open", [G3o, L, G3b, ev_line("3-4")])]
        for tag, evs in stages:
            res["rows"].append(row(scr, tag, evs))
        print("-- end states of the line candidates (G3 back + 600, line open)", flush=True)
        for ln in ("2-3", "4-14", "3-18", "17-27", "14-15", "9-39"):
            res["rows"].append(row(scr, f"end: {ln} open", [G3o, L, G3b, ev_line(ln)]))
    elif CASE == "P2":
        for tag, start, evs in (
                ("P2 t = 0 (6 Nov 08:00)", None, []),
                ("P2R hold (10:35, +200/+100 @ 12)", datetime(2016, 11, 6, 10, 35), [ev_load(11, 100.0, 200.0)]),
                ("B03B05 hold (10:35, 2x300/300)", datetime(2016, 11, 6, 10, 35),
                 [ev_load(2, 300.0, 300.0), ev_load(4, 300.0, 300.0)])):
            net0, cfg = capture("P2", start)
            scr = Screen(net0)
            print(f"network {NETWORK} | start {cfg.start_time} | committed: "
                  f"{[k for k, on in scr.gen_on.items() if on]} | off: {[k for k, on in scr.gen_on.items() if not on]}",
                  flush=True)
            res["rows"].append(row(scr, tag, evs))
    print(f"[{time.time() - t0:.0f} s]")
    if OUT:
        pickle.dump(res, open(OUT, "wb"))
