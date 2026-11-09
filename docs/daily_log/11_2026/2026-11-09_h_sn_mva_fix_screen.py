"""Does the sn_mva fix change procedure v5's lambda screen?  (2026-11-09)

Read-only on run 0002's lambda bank (6 design + 186 grid points, captured with
the as-built H).  The fix scales the STS interface-Q x OLTC entries by sn_mva
(x100) and, with the companion change in ``_dso_curvature``, weights them by
``gamma_oltc_q`` (0 in every campaign config).  On the cached Grams this is

    S_Q' = D S_Q D,  D = diag(1, ..., 100*gamma on the dso_oltc columns, ..., 1)

V rows (S_V) are unchanged by the fix; TS layers read V and Q_PCC rows only and
weight Q_PCC with tso_g_q_pcc = 0, so the TS lambda is unchanged as well.
"""
import dataclasses
import glob
import json
import pickle
import sys

import numpy as np

sys.path.insert(0, r"F:\qOFO_qgen")
from tuning import stage1_screen as s1  # noqa: E402

RUN = r"F:\qOFO_v5\results\ch_09_parameter_selection\0002_2026-11-08_185554_ch_9_17_procedure_v5"
EXCLUDE = ("shunt", "dso_shunt")


def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        with open(f, "rb") as fh:
            out.append(pickle.load(fh))
    return out


def weights(path, key="fields"):
    d = json.load(open(path))
    return {k: float(v) for k, v in d[key].items() if k != "sts_coupler_hold_after_bank"}


def rescale(ops, factor):
    """Copy of ops with every STS layer's dso_oltc columns of S_Q scaled by factor."""
    out = []
    for op in ops:
        layers = []
        for lc in op.layers:
            if lc.layer == "STS":
                d = np.array([factor if c == "dso_oltc" else 1.0 for c in lc.classes])
                lc = dataclasses.replace(lc, S_Q=d[:, None] * lc.S_Q * d[None, :])
            layers.append(lc)
        out.append(dataclasses.replace(op, layers=layers))
    return out


ops = load(RUN + r"\operating_points\op_*.npz") + load(RUN + r"\operating_points_lambda\op_*.npz")
sel = json.load(open(RUN + r"\selection\selection_2a_sts+bank.json"))
W = {
    "v4 (061d)": weights(r"F:\qOFO_v5\configs\paramsets\ch9_selected_uc_2026_11_v4.json"),
    "v5 Stage I": weights(r"F:\qOFO_v5\configs\paramsets\stage1_v5_nr_2026_11_pq10.json"),
    "v5 II-a leader": {k: float(v) for k, v in sel["selected"]["weights"].items()},
}
variants = {"as built": 1.0, "fixed, gamma 0": 0.0, "fixed, gamma 1": 100.0}
banks = {name: rescale(ops, f) for name, f in variants.items()}
print(f"{len(ops)} operating points; ceiling {s1.select_ceiling():.4f}; excluded {EXCLUDE}")
print(f"{'weights':16s} {'variant':16s} {'worst TS':>9s} {'worst STS':>10s} {'STS layer, condition':>32s}")
for wn, w in W.items():
    base = None
    for vn, bank in banks.items():
        rows = s1.screen_weights(bank, w, exclude_classes=EXCLUDE)
        ts = s1.worst([r for r in rows if r.layer == "TS"])
        sts = s1.worst([r for r in rows if r.layer == "STS"])
        if base is None:
            base = sts.lambda_max
        print(f"{wn:16s} {vn:16s} {ts.lambda_max:9.4f} {sts.lambda_max:10.4f} "
              f"{sts.key + ' ' + sts.condition:>32s}   (STS vs as built {sts.lambda_max - base:+.2e})")
