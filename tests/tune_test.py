#!/usr/bin/env python3
"""Quick tuning test."""
import numpy as np, time, sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
from run_cascade import run_cascade

def metrics(log, label):
    dso_recs = [r for r in log if r.dso_q_actual_mvar is not None]
    if not dso_recs: return {}
    q_actual = np.array([r.dso_q_actual_mvar for r in dso_recs])
    q_set = np.array([r.dso_q_setpoint_mvar for r in dso_recs])
    q_err = q_actual - q_set
    q_mae, q_max_err = np.mean(np.abs(q_err)), np.max(np.abs(q_err))
    q_jitter = np.mean(np.abs(np.diff(q_actual, axis=0)))
    v_tn = np.array([r.plant_tn_voltages_pu for r in log if r.plant_tn_voltages_pu is not None])
    v_dn = np.array([r.plant_dn_voltages_pu for r in log if r.plant_dn_voltages_pu is not None])
    v_tn_err, v_dn_err = np.max(np.abs(v_tn-1.05)), np.max(np.abs(v_dn-1.05))
    v_dn_jitter = np.mean(np.abs(np.diff(v_dn, axis=0)))
    tr = [r for r in log if r.tso_oltc_taps is not None]
    tso_sw = sum(int(np.sum(tr[i].tso_oltc_taps != tr[i-1].tso_oltc_taps)) for i in range(1, len(tr)))
    do = [r for r in log if r.dso_oltc_taps is not None]
    dso_sw = sum(int(np.sum(do[i].dso_oltc_taps != do[i-1].dso_oltc_taps)) for i in range(1, len(do)))
    ds = [r for r in log if r.dso_shunt_states is not None]
    sh_sw = sum(int(np.sum(ds[i].dso_shunt_states != ds[i-1].dso_shunt_states)) for i in range(1, len(ds)))
    dd = [r for r in log if r.dso_q_der_mvar is not None]
    der_j = np.mean(np.abs(np.diff(np.array([r.dso_q_der_mvar for r in dd]), axis=0))) if len(dd)>1 else 0
    lr = [r for r in log if r.plant_tn_voltages_pu is not None][-1]
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    print(f"  Q MAE={q_mae:.2f} max={q_max_err:.2f} jitter={q_jitter:.2f}")
    print(f"  V_TN err={v_tn_err:.4f}  V_DN err={v_dn_err:.4f}  V_DN jit={v_dn_jitter:.5f}")
    print(f"  DER jit={der_j:.2f}  TSO_OLTC={tso_sw}  DSO_OLTC={dso_sw}  Shunt={sh_sw}")
    print(f"  TN V=[{np.min(lr.plant_tn_voltages_pu):.4f},{np.max(lr.plant_tn_voltages_pu):.4f}]"
          f"  DN V=[{np.min(lr.plant_dn_voltages_pu):.4f},{np.max(lr.plant_dn_voltages_pu):.4f}]")
    print(f"{'='*60}")
    return dict(q_mae=q_mae, q_jitter=q_jitter, v_tn_err=v_tn_err, v_dn_err=v_dn_err,
                tso_oltc=tso_sw, dso_oltc=dso_sw, dso_shunt=sh_sw, der_j=der_j)

def run_test(label, g_v, g_q, n_min=90):
    print(f"\n>>> {label} ..."); sys.stdout.flush()
    t0 = time.time()
    r = run_cascade(v_setpoint_pu=1.05, n_minutes=n_min, tso_period_min=3, dso_period_min=1,
                    verbose=False, live_plot=False, g_v=g_v, g_q=g_q, use_profiles=False)
    print(f"  Runtime: {time.time()-t0:.1f}s")
    return metrics(r.log, label)

if __name__ == "__main__":
    # Fine-tuning around the TEST5 sweet spot
    m1 = run_test("TEST7: g_v=200000, g_q=1", g_v=200000, g_q=1)
    m2 = run_test("TEST8: g_v=100000, g_q=1.5", g_v=100000, g_q=1.5)
