"""Init-only diagnostic (no loop): which machine-transformer taps / AVRs does each TS-OFO of O3 hold, and why;
CEN's masks; per-switch curvature c_j of CEN vs owner. E2, new base."""
import contextlib, dataclasses, io, sys
import numpy as np
from experiments.ch_11_case2_multisystem import variants as v
from experiments.ch_11_case2_multisystem.ref import scenarios as sc
from experiments.ch_11_case2_multisystem.ref.run_ref import NOISE_SEED
from experiments.runners.multi_tso_dso import run_multi_tso_dso

OUT = {}


def c_diag(Hv, gv, gw):
    return (np.asarray(gv, float)[:, None] * np.asarray(Hv, float) ** 2).sum(axis=0) / np.asarray(gw, float)


def hook_o3(st):
    net = st["net"]
    cfg = st["config"]
    print("O3 local_sensitivities_tso", cfg.local_sensitivities_tso, "g_q_tso?",
          {z: getattr(c.config, "g_q_tso", None) for z, c in st["tso_controllers"].items()})
    for z, ctrl in st["tso_controllers"].items():
        c = ctrl.config
        ol = [int(t) for t in c.oltc_trafo_indices]
        print(f"zone {z}: gens {list(c.gen_indices)} oos_gen {ctrl._oos_gen_mask.tolist()} "
              f"fixed_v {getattr(ctrl, '_fixed_v_gen_mask', None)}")
        for k, t in enumerate(ol):
            hv, lv = int(net.trafo.at[t, 'hv_bus']), int(net.trafo.at[t, 'lv_bus'])
            g = [int(i) for i in net.gen.index if int(net.gen.at[i, 'bus']) in (hv, lv)]
            gn = [(str(net.gen.at[i, 'name']), bool(net.gen.at[i, 'in_service'])) for i in g]
            print(f"   trafo {t}: hv {hv} lv {lv} gens {gn} tap {net.trafo.at[t, 'tap_pos']} "
                  f"oos_mask {bool(ctrl._oos_oltc_mask[k])} in_service {bool(net.trafo.at[t, 'in_service'])}")
        Hv, gv = ctrl.objective_curvature_inputs()
        gw, _ = ctrl._get_per_variable_weights()
        if gw is None:
            gw = np.broadcast_to(np.asarray(ctrl.params.g_w, float), (Hv.shape[1],))
        cc = c_diag(Hv, gv, gw)
        sh = [int(b) for b in c.shunt_bus_indices]
        n0 = Hv.shape[1] - len(ol) - len(sh)
        for k, t in enumerate(ol):
            OUT[("tap", t)] = (cc[n0 + k], gw[n0 + k], z)
        for k, b in enumerate(sh):
            OUT[("bank", b)] = (cc[n0 + len(ol) + k], gw[n0 + len(ol) + k], z)
        print(f"   Hv rows {Hv.shape[0]} (n_v {len(c.voltage_bus_indices)})")
    return True


def hook_m(st):
    cc, cfg = st["central_controller"], st["central_cfg"]
    print("CEN oos_gen", cc._oos_gen_mask.tolist(), "oos_oltc", cc._oos_oltc_mask.tolist(),
          "oltc", list(cfg.oltc_trafo_indices), "gens", list(cfg.gen_indices))
    H = cc._expand_H_to_der_level(cc._build_sensitivity_matrix())
    Hv = H[:len(cfg.voltage_bus_indices), :]
    gw, _ = cc._get_per_variable_weights()
    c = c_diag(Hv, cc.g_v_per_bus, gw)
    n0 = len(cfg.der_indices) + len(cfg.gen_indices)
    ol = [int(t) for t in cfg.oltc_trafo_indices]
    sh = [int(b) for b in cfg.shunt_bus_indices]
    print(f"{'device':10s} {'c_CEN':>9s} {'c_own':>9s} {'ratio':>6s}  gw_CEN gw_own zone")
    for k, t in enumerate(ol):
        o = OUT.get(("tap", t))
        print(f"tap {t:4d}  {c[n0+k]:9.4g} {o[0]:9.4g} {c[n0+k]/o[0] if o[0] else float('inf'):6.2f}  "
              f"{gw[n0+k]:.4g} {o[1]:.4g} z{o[2]}")
    for k, b in enumerate(sh):
        o = OUT.get(("bank", b))
        j = n0 + len(ol) + k
        print(f"bank {b:4d} {c[j]:9.4g} {o[0]:9.4g} {c[j]/o[0]:6.2f}  {gw[j]:.4g} {o[1]:.4g} z{o[2]}")
    return True


for name, hk in (("O3", hook_o3), ("M", hook_m)):
    cfg = dataclasses.replace(v.resolved(name, 10, sc.events("E2"), NOISE_SEED), verbose=0, unit_commitment_window_s=21600)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_multi_tso_dso(cfg, pre_loop_hook=hk)
    txt = buf.getvalue()
    keep = [ln for ln in txt.splitlines() if ln.startswith(("O3 ", "zone", "   ", "CEN", "device", "tap ", "bank ",
                                                            "  [central]", "  [9]"))]
    print(f"===== {name}")
    print("\n".join(keep), flush=True)
