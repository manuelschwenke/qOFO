#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLOSED-LOOP CIGRE MV, Q_sub TRACKING (the faithful DSO_2 analog).  The OFO
controller drives the DER-feeder substation reactive Q_sub (Trafo 0-1, the
HV/MV interface = DSO_2's q_trafo analog) to a setpoint Q_REF using DER Q, with
the estimator's H = ∂Q_sub/∂Q_DER used online.  Compares Kalman-in-loop vs
constant-H on H-error and Q-tracking |Q_sub - Q_REF| over the max-load/no-gen ->
max-gen/no-load sweep (H varies a lot).  .venv312.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandapower as pp
import pandapower.networks as ppn
from pandapower.auxiliary import LoadflowNotConverged
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datetime import datetime

N, PE, ALPHA, DELTA, Q_REF, BIAS, STEP_CLIP = 200, 0.2, 0.3, 0.1, 0.0, 0.10, 0.5
rng_master = np.random.default_rng(0)
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "003_cigre_2026")

net = ppn.create_cigre_network_mv(with_der="pv_wind")
der = net.sgen.index.tolist()
net.sgen.loc[der, "p_mw"] = 2.0; net.sgen.loc[der, "sn_mva"] = 3.0
p_full = net.sgen.p_mw.values.copy(); sn = net.sgen.sn_mva.values.copy()
L0p = net.load.p_mw.values.copy(); L0q = net.load.q_mvar.values.copy()
n_der, n_out = len(der), 1                          # 1 controlled output: feeder-1 Q_sub
qlim = lambda p: np.sqrt(np.maximum(sn ** 2 - p ** 2, 0.0))
Qsub = lambda: float(net.res_trafo.q_lv_mvar.values[0])    # Trafo 0-1 (DER feeder) lv-side Q

def fd_HQ(u):
    q0 = Qsub(); H = np.zeros((1, n_der))
    for i in range(n_der):
        net.sgen.at[der[i], "q_mvar"] = u[i] + DELTA; pp.runpp(net)
        H[0, i] = (Qsub() - q0) / DELTA; net.sgen.at[der[i], "q_mvar"] = u[i]
    pp.runpp(net); return H

def run_pass(mode, h_init=None, Q=None, R=None):
    rng = np.random.default_rng(1)
    u = np.zeros(n_der); net.sgen.loc[der, "q_mvar"] = 0.0
    ns = n_out * n_der
    h = (h_init.flatten().copy() if h_init is not None else None)
    P = (0.10 * np.sqrt(np.mean(h_init ** 2))) ** 2 * np.eye(ns) if h is not None else None
    Qt, He, Ht, resid, prevQ, prevu = [], [], [], [], None, None
    for k in range(N):
        g = k / (N - 1); ls = (1 - g) * 0.95 + 0.05; gs = 0.05 + g * 0.95
        net.load.loc[:, "p_mw"] = ls * L0p; net.load.loc[:, "q_mvar"] = ls * L0q
        net.sgen.loc[der, "p_mw"] = gs * p_full; ql = qlim(gs * p_full)
        try:
            pp.runpp(net); q1 = Qsub(); HQt = fd_HQ(u)
            Hc = HQt if mode == "oracle" else (h.reshape(n_out, n_der) if mode == "kf" else h_init)
            step = np.clip(-ALPHA * (np.linalg.pinv(Hc, rcond=0.05) @ np.array([q1 - Q_REF])), -STEP_CLIP, STEP_CLIP)
            u = np.clip(u + step + rng.normal(0, PE, n_der), -ql, ql)
            net.sgen.loc[der, "q_mvar"] = u; pp.runpp(net); q1p = Qsub()
        except LoadflowNotConverged:
            print(f"  [{mode}] non-conv at step {k}; stop"); break
        if mode == "kf" and prevQ is not None:
            dQ, du = np.array([q1p - prevQ]), u - prevu
            C = np.kron(np.eye(n_out), du.reshape(1, -1)); Pp = P / 0.99 + Q
            K = np.linalg.solve(C @ Pp @ C.T + R, C @ Pp).T
            h = h + K @ (dQ - C @ h); P = (np.eye(ns) - K @ C) @ Pp
        if prevQ is not None:
            resid.append((q1p - prevQ) - float(HQt @ (u - prevu)))
        Qt.append(abs(q1p - Q_REF)); He.append(Hc.copy()); Ht.append(HQt.copy())
        prevQ, prevu = q1p, u.copy()
    return dict(Qt=np.array(Qt), Hc=np.stack(He), Ht=np.stack(Ht), resid=np.array(resid))

A = run_pass("oracle")
dH = np.diff(A["Ht"].reshape(len(A["Ht"]), -1), axis=0)
Q = np.diag(np.var(dH, axis=0) + 1e-12); R = np.atleast_2d(np.var(A["resid"]) + 1e-9)
h0 = A["Ht"][0] * (1 + BIAS * rng_master.standard_normal(A["Ht"][0].shape))
B = run_pass("kf", h_init=h0, Q=Q, R=R); Cc = run_pass("const", h_init=h0)

rel = lambda P_, T_: np.array([np.linalg.norm(P_[k] - T_[k]) / (np.linalg.norm(T_[k]) + 1e-12) for k in range(len(T_))])
L = min(len(B["Qt"]), len(Cc["Qt"]))
eB, eC = rel(B["Hc"], B["Ht"])[:L], rel(Cc["Hc"], Cc["Ht"])[:L]
qB, qC = B["Qt"][:L], Cc["Qt"][:L]
np.savez(os.path.join(RES, "cigre_cl_q_traj.npz"), eB=eB, eC=eC, qB=qB, qC=qC)
print("=" * 60)
print("  CIGRE MV CLOSED LOOP — Q_sub TRACKING (DSO_2 analog)")
print("=" * 60)
print(f"  steps completed   : {L}/{N}   (Q_REF = {Q_REF} Mvar)")
print(f"  H-error mean      : KF {eB.mean():.3f}   const {eC.mean():.3f}")
print(f"  H-error last20    : KF {eB[-20:].mean():.3f}   const {eC[-20:].mean():.3f}   (max-gen extreme)")
print(f"  Q-track mean |Q_sub-Q_REF| : KF {qB.mean():.4f}   const {qC.mean():.4f}   "
      f"improvement {100*(1-qB.mean()/qC.mean()):.0f}%")
print(f"  Q-track last20    : KF {qB[-20:].mean():.4f}   const {qC[-20:].mean():.4f}   "
      f"improvement {100*(1-qB[-20:].mean()/qC[-20:].mean()):.0f}%   (max-gen extreme)")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
ax[0].plot(eB, color="tab:blue", lw=2, label="Kalman (online)")
ax[0].plot(eC, color="tab:orange", lw=2, ls="--", label="constant-H")
ax[0].set_title("H-error  ||Ĥ-H||/||H||  (∂Q_sub/∂Q)"); ax[0].set_xlabel("sweep step"); ax[0].set_ylabel("relative H-error")
ax[1].plot(qB, color="tab:blue", lw=2, label="Kalman-driven OFO")
ax[1].plot(qC, color="tab:orange", lw=2, ls="--", label="constant-H-driven OFO")
ax[1].set_title("Q_sub tracking  |Q_sub - Q_REF|"); ax[1].set_xlabel("sweep step"); ax[1].set_ylabel("Mvar")
for a in ax:
    a.grid(alpha=0.3); a.legend()
fig.suptitle("CIGRE MV (open ring), closed loop, Q_sub tracking: Kalman vs constant-H", fontsize=11)
ts = datetime.now().strftime("%Y-%m-%d--%H-%M-%S"); png = os.path.join(RES, f"cigre_cl_q_{ts}.png")
fig.savefig(png, dpi=150); print(f"  figure -> {png}")
print("=" * 60)
try:
    os.startfile(png)
except AttributeError:
    import subprocess; subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", png], check=False)
