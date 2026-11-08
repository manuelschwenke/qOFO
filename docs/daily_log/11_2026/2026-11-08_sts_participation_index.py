"""Analytic STS reactive participation of the run-0008 feasible candidates.

First unconstrained TS-OFO step under a uniform TS voltage deviation eps (the
theta commit-rule condition): du_i = (g_v / g_w,i) * c_i * eps, c_i = |sum_j H_V[j,i]|.
  sigma_STS = sum_pcc du / (sum_pcc du + sum_der du)                (Mvar share)
  r         = (sum_pcc du / Qcap_STS) / (sum_der du / Qcap_IBG)     (participation-level ratio)
"""
import sys, json, os, pickle, numpy as np
sys.path.insert(0, r"F:\qOFO_res")
D = r"F:\qOFO_res\results\ch_09_parameter_selection\0008_2026-11-04_133847_ch_9_14_procedure_v4"
ops = [pickle.load(open(os.path.join(D, "operating_points", f"op_rural_700_netf8c419bf_W{i}.npz"), "rb")) for i in range(1, 7)]
# capability (up+dn, Mvar) per zone from run 0031 records (time means)
QCAP = {"2": {"der": 370.0, "pcc": 1356.0}, "3": {"der": 857.0, "pcc": 343.0}}

print("column sensitivities c_i = |sum_j H_V[j,i]| [pu/Mvar], range over W1..W6")
C = {}
for z in ("2", "3"):
    der = np.array([[lc.colsum_V[i] for i, c in enumerate(lc.classes) if c == "der"]
                    for op in ops for lc in op.layers if lc.key == f"TS_{z}"])
    pcc = np.array([[lc.colsum_V[i] for i, c in enumerate(lc.classes) if c == "pcc"]
                    for op in ops for lc in op.layers if lc.key == f"TS_{z}"])
    C[z] = (der, pcc)
    print(f"  zone {z}: IBG cols {der.shape[1]} c = {der.min():.2e}..{der.max():.2e};"
          f"  PCC cols {pcc.shape[1]} c = {pcc.min():.2e}..{pcc.max():.2e};"
          f"  mean c_pcc/c_ibg = {pcc.mean()/der.mean():.2f}")

def part(w, z):
    der, pcc = C[z]
    a_der = (der / w["g_w_der"]).sum(axis=1)      # per window
    a_pcc = (pcc / w["g_w_pcc"]).sum(axis=1)
    sigma = a_pcc / (a_pcc + a_der)
    r = (a_pcc / QCAP[z]["pcc"]) / (a_der / QCAP[z]["der"])
    return sigma.min(), sigma.max(), r.min(), r.max()

sel = json.load(open(os.path.join(D, "selection", "selection_complete_sts+bank.json")))
print("\nfeasible candidates (5-window completion), ranked as selected:")
print("  key           g_w_pcc  g_w_der  ratio  e_v_ts   e_v_ds   | z2 sigma_STS   z2 r        | z3 sigma_STS  z3 r")
rows = []
for i, c in enumerate(sel["ranked"]):
    w, a = c["weights"], c["aggregate"]
    p2, p3 = part(w, "2"), part(w, "3")
    rows.append((c["key"], w, a, p2, p3))
    print(f"  {i+1:2d} {c['key']}  {w['g_w_pcc']:.3f}  {w['g_w_der']:.4f}  {w['g_w_pcc']/w['g_w_der']:5.1f}  {a['e_v_ts']:.5f}  {a['e_v_ds']:.5f}"
          f" | {p2[0]:.2f}-{p2[1]:.2f}  {p2[2]:.2f}-{p2[3]:.2f} | {p3[0]:.2f}-{p3[1]:.2f}  {p3[2]:.2f}-{p3[3]:.2f}")

w0 = {"g_w_der": 0.014211188776637435}
for z in ("2", "3"):
    der, pcc = C[z]
    # g_w_pcc for r = 1 (equal participation level) at g_w_der of 061d, worst window
    k = (pcc.sum(axis=1) / QCAP[z]["pcc"]) / ((der / w0["g_w_der"]).sum(axis=1) / QCAP[z]["der"])
    print(f"zone {z}: g_w_pcc giving r = 1 at g_w_der 0.0142: {k.min():.4f}..{k.max():.4f}; r = 0.5: {k.min()/0.5:.4f}..{k.max()/0.5:.4f}")
for lab, w in [("0030 (g_w_pcc 0.10)", {"g_w_der": 0.014211188776637435, "g_w_pcc": 0.1}),
               ("Stage I (0.0897/0.0897)", {"g_w_der": 0.0897, "g_w_pcc": 0.0897})]:
    print(lab, "z2", np.round(part(w, "2"), 2), "z3", np.round(part(w, "3"), 2))
