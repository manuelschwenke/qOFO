#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLOSED-LOOP CIGRE MV comparison for the paper: the OFO voltage controller uses
the estimator's H online (like DSO_2), so we can show BOTH (a) H-error and
(b) voltage-tracking improvement of Kalman vs constant-H, over the max-load/no-gen
-> max-gen/no-load sweep (H varies ~250-286%).  Open ring, taps fixed, DER Q = u.

Three passes over the SAME sweep:
  A 'oracle' (control with true H)  -> calibrate KF Q,R from H_true + residuals
  B 'kf'     (control with online KF H, biased init)
  C 'const'  (control with frozen biased-init H)
Plots: H-error(KF vs const) and tracking ||V-Vref||(KF vs const).  .venv312.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandapower as pp
import pandapower.networks as ppn
from pandapower.auxiliary import LoadflowNotConverged
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datetime import datetime

N, PE, ALPHA, DELTA, VREF, BIAS = 200, 0.2, 0.3, 0.1, 1.0, 0.10
STEP_CLIP = 0.5    # per-step |Δu| limit [Mvar] (stability vs a wrong H)
rng_master = np.random.default_rng(0)
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "003_cigre_2026")

net = ppn.create_cigre_network_mv(with_der="pv_wind")
der = net.sgen.index.tolist(); der_bus = net.sgen.bus.tolist()
net.sgen.loc[der, "p_mw"] = 2.0; net.sgen.loc[der, "sn_mva"] = 3.0
p_full = net.sgen.p_mw.values.copy(); sn = net.sgen.sn_mva.values.copy()
L0p = net.load.p_mw.values.copy(); L0q = net.load.q_mvar.values.copy()
n_der, n_v = len(der), len(der_bus)
qlim = lambda p: np.sqrt(np.maximum(sn ** 2 - p ** 2, 0.0))
Vof = lambda: net.res_bus.vm_pu.values[der_bus].copy()

def fd_HV(u):
    V0 = Vof(); H = np.zeros((n_v, n_der))
    for i in range(n_der):
        net.sgen.at[der[i], "q_mvar"] = u[i] + DELTA; pp.runpp(net)
        H[:, i] = (Vof() - V0) / DELTA; net.sgen.at[der[i], "q_mvar"] = u[i]
    pp.runpp(net); return H

def run_pass(mode, h_init=None, Q=None, R=None):
    rng = np.random.default_rng(1)                       # same PE stream across passes
    u = np.zeros(n_der); net.sgen.loc[der, "q_mvar"] = 0.0   # clean start (no leftover Q)
    ns = n_v * n_der
    h = (h_init.flatten().copy() if h_init is not None else None)
    P = (0.10 * np.sqrt(np.mean(h_init ** 2))) ** 2 * np.eye(ns) if h is not None else None
    Vt, He, Ht, resid, prevV, prevu = [], [], [], [], None, None
    for k in range(N):
        g = k / (N - 1); ls = (1 - g) * 0.95 + 0.05; gs = 0.05 + g * 0.95
        net.load.loc[:, "p_mw"] = ls * L0p; net.load.loc[:, "q_mvar"] = ls * L0q
        net.sgen.loc[der, "p_mw"] = gs * p_full; ql = qlim(gs * p_full)
        try:
            pp.runpp(net); V = Vof(); HVt = fd_HV(u)
            Hc = HVt if mode == "oracle" else (h.reshape(n_v, n_der) if mode == "kf" else h_init)
            step = np.clip(-ALPHA * (np.linalg.pinv(Hc, rcond=0.05) @ (V - VREF)), -STEP_CLIP, STEP_CLIP)
            u = np.clip(u + step + rng.normal(0, PE, n_der), -ql, ql)
            net.sgen.loc[der, "q_mvar"] = u; pp.runpp(net); Vp = Vof()
        except LoadflowNotConverged:
            print(f"  [{mode}] non-conv at step {k}; stop"); break
        if mode == "kf" and prevV is not None:
            dV, du = Vp - prevV, u - prevu
            C = np.kron(np.eye(n_v), du.reshape(1, -1)); Pp = P / 0.99 + Q
            K = np.linalg.solve(C @ Pp @ C.T + R, C @ Pp).T
            h = h + K @ (dV - C @ h); P = (np.eye(ns) - K @ C) @ Pp
        if prevV is not None:                            # one-step residual for R calib
            resid.append((Vp - prevV) - HVt @ (u - prevu))
        Vt.append(np.linalg.norm(Vp - VREF)); He.append(Hc); Ht.append(HVt)
        prevV, prevu = Vp, u.copy()
    return dict(Vt=np.array(Vt), Hc=np.stack(He), Ht=np.stack(Ht), resid=np.array(resid))

# Pass A: oracle -> calibrate Q,R
A = run_pass("oracle")
dH = np.diff(A["Ht"].reshape(len(A["Ht"]), -1), axis=0)
Q = np.diag(np.var(dH, axis=0) + 1e-12); R = np.diag(np.var(A["resid"], axis=0) + 1e-9)
h0 = A["Ht"][0] * (1 + BIAS * rng_master.standard_normal(A["Ht"][0].shape))   # shared biased init
B = run_pass("kf", h_init=h0, Q=Q, R=R)
Cc = run_pass("const", h_init=h0)

rel = lambda P_, T_: np.array([np.linalg.norm(P_[k] - T_[k]) / (np.linalg.norm(T_[k]) + 1e-12) for k in range(len(T_))])
L = min(len(B["Vt"]), len(Cc["Vt"]))
eB, eC = rel(B["Hc"], B["Ht"])[:L], rel(Cc["Hc"], Cc["Ht"])[:L]
vB, vC = B["Vt"][:L], Cc["Vt"][:L]
print("=" * 60)
print("  CIGRE MV CLOSED LOOP (KF-in-loop vs constant-H driving OFO)")
print("=" * 60)
print(f"  steps completed   : {L}/{N}")
print(f"  H-error mean      : KF {eB.mean():.3f}   const {eC.mean():.3f}")
print(f"  H-error last20    : KF {eB[-20:].mean():.3f}   const {eC[-20:].mean():.3f}   (max-gen extreme)")
print(f"  V-track mean ||V-Vref|| : KF {vB.mean():.4f}   const {vC.mean():.4f}   "
      f"improvement {100*(1-vB.mean()/vC.mean()):.0f}%")
print(f"  V-track last20    : KF {vB[-20:].mean():.4f}   const {vC[-20:].mean():.4f}   "
      f"improvement {100*(1-vB[-20:].mean()/vC[-20:].mean()):.0f}%   (max-gen extreme)")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
ax[0].plot(eB, color="tab:blue", lw=2, label="Kalman (online, drives control)")
ax[0].plot(eC, color="tab:orange", lw=2, ls="--", label="constant-H (biased init)")
ax[0].set_title("H-error  ||Ĥ-H||/||H||"); ax[0].set_xlabel("sweep step"); ax[0].set_ylabel("relative H-error")
ax[1].plot(vB, color="tab:blue", lw=2, label="Kalman-driven OFO")
ax[1].plot(vC, color="tab:orange", lw=2, ls="--", label="constant-H-driven OFO")
ax[1].set_title("voltage tracking  ||V - V_ref||"); ax[1].set_xlabel("sweep step"); ax[1].set_ylabel("Mvar-bus voltage dev [pu]")
for a in ax:
    a.grid(alpha=0.3); a.legend()
fig.suptitle("CIGRE MV (open ring), closed loop: Kalman vs constant-H over max-load→max-gen sweep", fontsize=11)
ts = datetime.now().strftime("%Y-%m-%d--%H-%M-%S"); png = os.path.join(RES, f"cigre_closedloop_{ts}.png")
fig.savefig(png, dpi=150); print(f"  figure -> {png}")
print("=" * 60)
try:
    os.startfile(png)
except AttributeError:
    import subprocess; subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", png], check=False)
