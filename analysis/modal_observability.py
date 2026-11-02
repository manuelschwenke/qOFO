#!/usr/bin/env python3
r"""Exact residues of the linearised RMS plant in the controlled outputs.

Option 1 of CHECKPOINT R1 (``ch:timescales:plant``, author's go 2026-09-23):
an observability argument.  The declared rule bounds the *modes*; this module
asks how much of each slow mode actually reaches a **controlled output** after
a **dispatch** step, and when every (input, output) step response has settled.

Model, from PowerFactory's own export (``pf/modal_window.py`` runs):

    E x' = J z,  z = [x; y]   (Jacobian.mtl, M.mtl; states first)
    x' = A11 x + A12 y + b_x u,   0 = A21 x + A22 y + b_y u
    A = A11 - A12 A22^-1 A21          (reproduces Amat.mtl to ~1e-13)
    B = b_x - A12 A22^-1 b_y,  C = -c_y^T A22^-1 A21,  D = -c_y^T A22^-1 b_y

Step response of output i to input j from equilibrium:

    y(t) = D + sum_k R_k / lambda_k (e^{lambda_k t} - 1),   R_k = C P_k B

with ``P_k`` the spectral projector of eigenvalue cluster ``k`` (repeated
eigenvalues of identical decoupled blocks are handled as clusters, not as
individually normalised modes).  The structural zeros are treated apart: a
non-zero ``C P_0 B`` would be a ramp and is reported, never divided by
``lambda ~ 1e-12``.

Settling time ``T_s`` = the last time ``|y(t) - y(inf)|`` exceeds the band,
evaluated on a grid to 600 s (not a bound).  Bands and step sizes are the
Gate-D ones already in the repo (``pf/screening.py``): 1e-3 pu voltage, 1 Mvar
reactive power; DER ``min(60 Mvar, 0.5 S_n, upward headroom)``, AVR set-point
0.02 pu, one tap.  A relative settling time (band ``e^-kappa |dy_inf|``) is
reported alongside.  Tap responses are timed from contact: the 5 s
command-to-contact dead time is not part of the linear model.

Inputs: DER ``qset`` via the QVPRE ``Qext`` equation (``g = h - Qext``);
AVR ``usetp`` via the IEEET1 ``xa`` row (``v_err = usetp - ut``); taps via
``nntapin`` of the taps-in-service run.  Outputs: TN and PCC voltage
magnitudes from ``ur/ui``; coupler HV-side and TSO tie-line Q from the
pandapower branch model evaluated at PF's voltages (validated against PF's own
flows).

Shunt banks (MSC/MSR, added 2026-11-02, :func:`run_shunts`): PF's shunt has
no RMS input signal, so a bank step enters through the admittance of its
terminal.  PF writes each terminal's network equations as the real and
imaginary current balance (rows ``ur``, ``ui``) with the admittances on a
1 MVA base, i.e. ``baseMVA`` times pandapower's Ybus (established on all 12
coupler tertiaries, ratio 100.00002, and re-checked per terminal).  A step
``dB`` adds ``j dB`` to ``Y_kk``, i.e. ``dB (-ui_k, ur_k)`` on the two rows.

Usage::

    python -m analysis.modal_observability results/screening/modal_campaign/W1/<stamp>
    python -m analysis.modal_observability <window dir> --shunts --out <dir>

Author: Manuel Schwenke / Claude Code (2026-09-23)
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pf import modal_participation as mp  # noqa: E402

KAPPA = 3.0
T_STS = 20.0
T_TAP = 5.0
BAND = {"V": 1e-3, "Q": 1.0}
DER_STEP_MVAR = 60.0
#: Largest AVR set-point step (author's decision 2026-09-23; the Gate-D
#: catalogue used 0.02 pu).
AVR_STEP_PU = 0.01
#: |lambda| below this x max|lambda| is a structural zero.
ZERO_REL = 1e-10
#: Relative distance under which eigenvalues form one cluster.
CLUSTER_REL = 1e-7
#: Settling is resolved from T_MIN on: a defective fast cluster (e.g. 84
#: identical 0.01 s lags, lambda = -100) cannot be decomposed reliably, but its
#: contribution is below e^-50 by then.  A reported T_s = T_MIN means "<= T_MIN".
T_MIN = 0.5
T_GRID = np.concatenate([np.arange(T_MIN, 60.0, 0.02), np.arange(60.0, 600.001, 0.5)])
#: Base of the admittances in PF's network equations (see the module doc).
PF_ADMITTANCE_BASE_MVA = 1.0
#: Relative tolerance of the per-terminal check of that base.
SHUNT_SCALE_RTOL = 1e-4

_VAR_RE = re.compile(r'^\s*(\d+)\s+(.+?)\s+"([^"]+)"\s*$')


# ── DAE from the export ────────────────────────────────────────────────────

@dataclass
class Dae:
    V: pd.DataFrame                    # i, path, parent, leaf, var
    J: csr_matrix
    n: int                             # number of states (first n variables)
    index: Dict[Tuple[str, str, str], int]

    @property
    def N(self) -> int:
        return len(self.V)


def load_dae(run_dir: Path) -> Dae:
    matl = Path(run_dir) / "matl"
    rows = []
    for line in (matl / "VariableToIdx_Jacobian.txt").read_text(
            encoding="utf-8", errors="replace").splitlines():
        m = _VAR_RE.match(line)
        if m:
            path = m.group(2).strip()
            parts = path.split("\\")
            rows.append((int(m.group(1)) - 1, path,
                         parts[-2] if len(parts) > 1 else "", parts[-1], m.group(3)))
    V = pd.DataFrame(rows, columns=["i", "path", "parent", "leaf", "var"])
    if (V["i"].to_numpy() != np.arange(len(V))).any():
        raise ValueError(f"{run_dir}: Jacobian variable indices are not 0..N-1")
    n = len(mp.read_varidx(matl))
    x = pd.read_csv(matl / "Jacobian.mtl", sep=r"\s+", header=None).to_numpy()
    J = csr_matrix((x[:, 2], (x[:, 0].astype(int) - 1, x[:, 1].astype(int) - 1)),
                   shape=(len(V), len(V)))
    index = {(r.parent, r.leaf, r.var): int(r.i) for r in V.itertuples()}
    return Dae(V=V, J=J, n=n, index=index)


class Reduced:
    """Kron reduction of the DAE onto its states, with input/output maps."""

    def __init__(self, dae: Dae):
        n = dae.n
        J = dae.J
        self.dae = dae
        self.A11 = J[:n, :n].toarray()
        self.A12 = J[:n, n:].tocsr()
        self.A21 = J[n:, :n].tocsc()
        self.lu = splu(csc_matrix(J[n:, n:]))
        self.A = self.A11 - self.A12 @ self.lu.solve(self.A21.toarray())

    def B(self, b_full: np.ndarray) -> np.ndarray:
        n = self.dae.n
        return b_full[:n] - self.A12 @ self.lu.solve(b_full[n:])

    def CD(self, cz: np.ndarray, bz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``cz`` (n_out x N_alg), ``bz`` (N_alg x n_in) -> (C, D)."""
        Yt = self.lu.solve(cz.T.astype(float), trans="T").T      # c^T A22^-1
        return -(self.A21.T @ Yt.T).T, -(Yt @ bz)


# ── spectral decomposition with clusters ──────────────────────────────────

def clustered_modes(A: np.ndarray):
    """Eigen-decomposition grouped into clusters of (numerically) equal
    eigenvalues, each with a proper projector ``Vc (Wc^T Vc)^-1 Wc^T``."""
    lam, vl, vr = sla.eig(A, left=True, right=True)
    order = np.argsort(np.abs(lam))
    lam, vl, vr = lam[order], vl[:, order], vr[:, order]
    scale = np.abs(lam).max()
    used = np.zeros(len(lam), dtype=bool)
    clusters = []
    for i in range(len(lam)):
        if used[i]:
            continue
        tol = max(CLUSTER_REL * abs(lam[i]), ZERO_REL * scale)
        idx = np.where(~used & (np.abs(lam - lam[i]) <= tol))[0]
        used[idx] = True
        Vc = vr[:, idx]
        Wc = vl[:, idx].conj()
        M = Wc.T @ Vc
        cond = np.linalg.cond(M)
        clusters.append({"lam": complex(np.mean(lam[idx])), "idx": idx, "Vc": Vc,
                         "Wt": np.linalg.solve(M, Wc.T), "cond": cond,
                         "zero": abs(lam[i]) <= ZERO_REL * scale})
    return lam, clusters


# ── inputs ─────────────────────────────────────────────────────────────────

def der_inputs(dae: Dae, op: dict) -> List[dict]:
    out = []
    for p in op["parks"]:
        parent = p["pre_key"][0]
        r = dae.index.get((parent, "QVPRE.ElmDsl", "Qext"))
        if r is None:
            raise KeyError(f"{parent}: no QVPRE Qext variable in the Jacobian")
        if abs(dae.J[r, r] + 1.0) > 1e-9:
            raise ValueError(f"{parent}: Qext row is not of the form h - Qext")
        head = (p["qmax"] - p["qset"]) * p["sn"]
        unclipped = p["qmin"] + 1e-9 < p["qset"] < p["qmax"] - 1e-9
        step_mvar = min(DER_STEP_MVAR, 0.5 * p["sn"], head)
        if not unclipped or step_mvar <= 1e-6:
            out.append({"name": f"DER_Q_{p['name']}", "cls": "DER_Q", "skip":
                        "clipped or no upward headroom (zero-width box)"})
            continue
        b = np.zeros(dae.N)
        b[r] = 1.0
        out.append({"name": f"DER_Q_{p['name']}", "cls": "DER_Q", "b": b,
                    "step": step_mvar / p["sn"], "unit": "pu(S_n)",
                    "step_label": f"+{step_mvar:.1f} Mvar",
                    "tso": parent.startswith("WECC_WP_TSO")})
    return out


def avr_inputs(dae: Dae) -> List[dict]:
    out = []
    ut_cols = {int(r.i): r.leaf.split(".")[0] for r in dae.V.itertuples()
               if r.leaf.endswith(".ElmSym") and r.var == "ut"}
    Jc = dae.J.tocsc()
    for c, machine in ut_cols.items():
        col = Jc.getcol(c)
        for r, val in zip(col.indices, col.data):
            if r < dae.n and dae.V.leaf[r].startswith("AVR") and dae.V["var"][r] == "xa":
                b = np.zeros(dae.N)
                b[r] = -val                       # v_err = usetp - ut
                entry = {"name": f"AVR_{machine}", "cls": "AVR", "b": b,
                         "step": AVR_STEP_PU, "unit": "pu",
                         "step_label": f"+{AVR_STEP_PU:g} pu", "machine": machine}
                if machine == "G 01":
                    # The slack/equivalent machine: its V-ref is withheld by the
                    # thesis configuration (multi_tso_dso.py,
                    # dispatch_slack_gen_v_ref = False).  Computed as a separate,
                    # labelled class so the consequence of dispatching it is
                    # visible; it is not part of the verdict.
                    entry["cls"] = "AVR_G01_equivalent"
                out.append(entry)
    return out


def tapsin_system(dae_t: Dae) -> Tuple[Dae, List[dict]]:
    """The taps-in-service DAE with the TAPCTRL blocks cut out and each
    ``nntapin`` turned into an exogenous input (a tap position after
    contact).  With a dynamic tap PF formulates the tapped transformers with
    explicit branch-current variables, so this system is NOT variable-for-
    variable the taps-out one; it is used in its own coordinates and checked
    by its spectrum."""
    tap_mask = (dae_t.V.parent.str.startswith("TAPC_")
                & (dae_t.V.leaf == "TAPCTRL.ElmDsl")).to_numpy()
    keep = np.where(~tap_mask)[0]
    n_new = int((~tap_mask[:dae_t.n]).sum())
    if tap_mask[:dae_t.n].sum() + (~tap_mask[:dae_t.n]).sum() != dae_t.n or \
            (keep[:n_new] >= dae_t.n).any():
        raise ValueError("taps-in DAE: states are not first after the cut")
    Vk = dae_t.V.iloc[keep].reset_index(drop=True)
    Vk["i"] = np.arange(len(Vk))
    sysk = Dae(V=Vk, J=dae_t.J[keep][:, keep].tocsr(), n=n_new,
               index={(r.parent, r.leaf, r.var): int(r.i) for r in Vk.itertuples()})
    Jtc = dae_t.J.tocsc()
    out = []
    for r in dae_t.V[tap_mask & (dae_t.V["var"] == "nntapin").to_numpy()].itertuples():
        b = Jtc.getcol(int(r.i)).toarray().ravel()[keep]
        trafo = r.parent[len("TAPC_"):]
        cls = ("TAP_MT" if trafo.startswith("MT_") else
               "TAP_NC3W" if trafo.startswith("NC3W_") else "TAP_other")
        out.append({"name": f"TAP_{trafo}", "cls": cls, "b": b, "step": 1.0,
                    "unit": "tap", "step_label": "+1 tap"})
    return sysk, out


def shunt_placement() -> Dict[str, dict]:
    """Bank placement and step size of the thesis configuration
    (``MultiTSOConfig`` defaults: MSR on the tertiary of the first coupler,
    MSC on that of the second, two steps of 25 Mvar each, since 2026-10-15)."""
    from configs.config import MultiTSOConfig
    d = {f.name: f.default for f in dataclasses.fields(MultiTSOConfig)}
    if d["tso_shunt_kind"] != "msc_msr":
        raise ValueError(f"tso_shunt_kind {d['tso_shunt_kind']!r}: no MSC/MSR banks")
    return {k: {"pos": int(d[f"tso_shunt_{k.lower()}_coupling_pos"]),
                "q_step_mvar": float(d[f"tso_shunt_{k.lower()}_q_step_mvar"]),
                "n_levels": int(d[f"tso_shunt_{k.lower()}_n_levels"]),
                "sign": +1 if k == "MSC" else -1}
            for k in ("MSC", "MSR")}


def shunt_inputs(dae: Dae, op: dict, snapshot: Path) -> Tuple[List[dict], pd.DataFrame]:
    """One step of each TSO bank at its terminal in the thesis placement.

    The PF model of the modal campaign predates the placement of 2026-10-15
    (both banks on the first tertiary).  Every bank is at step 0 in every
    window, so the operating point does not depend on where a bank is
    installed, and the step is applied at the thesis terminal.  Returns the
    inputs and a parity table PF bank vs thesis bank (step size, levels,
    rated voltage, terminal) with the measured admittance scale."""
    net, doc, names, ppc, base, Ybus, *_, bl, _ = _pp_model(snapshot)
    term = {t["name"]: t for t in op["terminals"]}
    if any(float(s["ncapa"]) != 0.0 for s in op["shunts"]):
        raise ValueError("a bank is switched in at the operating point: the "
                         "linearisation then depends on its location")
    J = dae.J.tocsr()
    out, parity = [], []
    for hv in doc["meta"]["hv_networks"]:
        net_id = hv["net_id"]
        for kind, p in shunt_placement().items():
            bus = int(hv["coupling_lv_bus_indices"][p["pos"]])
            tname = names[("bus", bus)]
            t = term[tname]
            iu = dae.index[(t["key"][0], t["key"][1], "ur")]
            iv = dae.index[(t["key"][0], t["key"][1], "ui")]
            if min(iu, iv) < dae.n:
                raise ValueError(f"{tname}: ur/ui are not algebraic")
            # the tertiary's only neighbour in pandapower's Ybus is the
            # trafo3w star bus; PF eliminates it, so compare the reduced
            # diagonal
            kb = int(bl[bus])
            row = Ybus.getrow(kb)
            nb = [int(c) for c in row.indices if int(c) != kb]
            if len(nb) != 1:
                raise ValueError(f"{tname}: expected one neighbour (the star bus), got {nb}")
            s = nb[0]
            y_pp = Ybus[kb, kb] - Ybus[kb, s] * Ybus[s, kb] / Ybus[s, s]
            y_pf = complex(J[iu, iu], J[iv, iu])
            ratio = y_pf / y_pp
            expect = base / PF_ADMITTANCE_BASE_MVA
            rows_ok = (abs(J[iu, iu] - J[iv, iv]) + abs(J[iu, iv] + J[iv, iu])) / abs(y_pf)
            if abs(ratio / expect - 1.0) > SHUNT_SCALE_RTOL or rows_ok > SHUNT_SCALE_RTOL:
                raise ValueError(f"{tname}: PF admittance scale {ratio:.6g} (expected "
                                 f"{expect:g}), row structure residual {rows_ok:.2e}")
            pf = [x for x in op["shunts"] if x["name"].startswith(f"SH_{kind}_{net_id}_")]
            if len(pf) != 1:
                raise ValueError(f"{net_id}: {len(pf)} PF {kind} banks")
            pf = pf[0]
            q_pf = float(pf["qcapn"] if kind == "MSC" else pf["qrean"])
            vn = float(net.bus.at[bus, "vn_kv"])
            parity.append({"bank": f"{kind}_{net_id}", "terminal_thesis": tname,
                           "terminal_pf_model": pf["bus"], "q_step_thesis": p["q_step_mvar"],
                           "q_step_pf": q_pf, "levels_thesis": p["n_levels"],
                           "levels_pf": int(pf["ncapx"]), "u_rated_pf_kv": float(pf["ushnm"]),
                           "u_n_terminal_kv": vn, "admittance_scale": ratio.real,
                           "admittance_scale_imag": ratio.imag, "row_structure_residual": rows_ok})
            if (abs(q_pf - p["q_step_mvar"]) > 1e-9 or int(pf["ncapx"]) != p["n_levels"]
                    or abs(float(pf["ushnm"]) - vn) > 1e-9):
                raise ValueError(f"{kind}_{net_id}: PF bank differs from the thesis bank "
                                 f"{parity[-1]}")
            # per Mvar of rated bank power: dB = 1 Mvar / (1 MVA) at 1 pu
            dB = p["sign"] / PF_ADMITTANCE_BASE_MVA
            b = np.zeros(dae.N)
            b[iu], b[iv] = -dB * t["ui"], dB * t["ur"]
            out.append({"name": f"SHUNT_{kind}_{net_id}", "cls": f"SHUNT_{kind}", "b": b,
                        "step": p["q_step_mvar"], "unit": "Mvar (rated)",
                        "step_label": f"{'+' if p['sign'] > 0 else '-'}{p['q_step_mvar']:g} Mvar",
                        "terminal": tname})
    return out, pd.DataFrame(parity)


def shunt_lf_check(df: pd.DataFrame, inputs: List[dict], snapshot: Path) -> pd.DataFrame:
    """Plausibility of the shunt columns: final value ``y_inf`` of the
    linearisation against pandapower load flows with the bank step in.
    ``dy_loadflow`` is the one-step difference; ``dy_loadflow_central`` the
    central difference over +-1 step, which removes the second-order term
    ``dB dV`` and compares the linear parts.  Not an equality: pandapower
    holds machine terminal voltages ideally, PF's AVRs have finite gain;
    meaningful in the droop-inert case only (DER Q fixed in both)."""
    import pandapower as pp
    net0, doc, names, *_ = _pp_model(snapshot)
    bus_of = {v: k[1] for k, v in names.items() if k[0] == "bus"}
    tr3_of = {v: k[1] for k, v in names.items() if k[0] == "trafo3w"}

    def stepped(bus: int, q_mvar: float):
        net = copy.deepcopy(net0)
        pp.create_shunt(net, bus=bus, q_mvar=q_mvar, p_mw=0.0,
                        vn_kv=float(net.bus.vn_kv[bus]), step=1, max_step=1)
        pp.runpp(net, **doc["solver_options"])
        return net

    def value(net, o: str) -> float:
        if o.startswith("V_"):
            return float(net.res_bus.vm_pu[bus_of[o[2:]]])
        return float(net.res_trafo3w.q_hv_mvar[tr3_of[o[2:]]])

    rows = []
    for u in inputs:
        kind = u["cls"].split("_")[1]
        q = u["step"] * (-1.0 if kind == "MSC" else 1.0)     # pandapower load sign
        bus = bus_of[u["terminal"]]
        up, down = stepped(bus, q), stepped(bus, -q)
        for r in df[df["input"] == u["name"]].itertuples():
            y0 = value(net0, r.output)
            d1 = value(up, r.output) - y0
            dc = 0.5 * (value(up, r.output) - value(down, r.output))
            rows.append({"input": u["name"], "output": r.output, "kind": r.kind,
                         "dy_inf_linear": r.dy_inf, "dy_loadflow": d1,
                         "dy_loadflow_central": dc, "diff": r.dy_inf - d1,
                         "diff_central": r.dy_inf - dc, "band": r.band})
    return pd.DataFrame(rows)


def spectrum_distance(A1: np.ndarray, A2: np.ndarray) -> float:
    """Max relative distance between matched eigenvalues of two matrices."""
    from scipy.optimize import linear_sum_assignment
    l1, l2 = np.linalg.eigvals(A1), np.linalg.eigvals(A2)
    if len(l1) != len(l2):
        return math.inf
    C = np.abs(l1[:, None] - l2[None, :])
    r, c = linear_sum_assignment(C)
    return float((C[r, c] / np.maximum(np.abs(l1[r]), 1.0)).max())


# ── outputs ────────────────────────────────────────────────────────────────

def controlled_voltage_names(snapshot: Path) -> Tuple[set, set, dict]:
    """PF names of the voltages the controllers measure, following
    ``experiments/runners/multi_tso_dso.py``: TSO = TN buses minus PV/slack
    buses (``v_bus_indices_z``); DSO = ``hv.bus_indices`` (110 kV incl. PCC,
    no tertiary buses).  Tie lines are not measured."""
    net, doc, names, *_ = _pp_model(snapshot)
    meta = doc["meta"]
    pv = {int(b) for b in net.gen.bus} | {int(b) for b in net.ext_grid.bus}
    ts = [int(b) for b in meta["tn_bus_indices"] if int(b) in net.bus.index and int(b) not in pv]
    ds = [int(b) for hv in meta["hv_networks"] for b in hv["bus_indices"]]
    ts_n = {names[("bus", b)] for b in ts if ("bus", b) in names}
    ds_n = {names[("bus", b)] for b in ds if ("bus", b) in names}
    info = {"ts_buses": len(ts), "ts_unmapped": [b for b in ts if ("bus", b) not in names],
            "ds_buses": len(ds), "ds_unmapped": [b for b in ds if ("bus", b) not in names]}
    return ts_n, ds_n, info


def slack_machine_taps(snapshot: Path) -> set:
    """Tap inputs the TSO never dispatches: the slack generator's machine
    transformer (``multi_tso_dso.py``: "Exclude the slack gen's machine trafo
    from the controllable OLTC set")."""
    net, doc, names, *_ = _pp_model(snapshot)
    slack_bus = {int(net.gen.at[g, "bus"]) for g in net.gen.index
                 if bool(net.gen.get("slack", pd.Series(dtype=bool)).get(g, False))}
    return {f"TAP_{names[('trafo', int(t))]}" for t in net.trafo.index
            if int(net.trafo.at[t, "lv_bus"]) in slack_bus and ("trafo", int(t)) in names}


def voltage_outputs(dae: Dae, op: dict, ts_names: set, ds_names: set) -> List[dict]:
    pcc = {t["mv"] for t in op["tr3"] if t["name"].startswith("NC3W_") and t["mv"]}
    out = []
    for t in op["terminals"]:
        name = t["name"]
        if not (name in ts_names or name in ds_names):
            continue
        parent, leaf = t["key"]
        iu, iv = dae.index.get((parent, leaf, "ur")), dae.index.get((parent, leaf, "ui"))
        if iu is None or iv is None:
            continue
        u = math.hypot(t["ur"], t["ui"])
        c = np.zeros(dae.N)
        c[iu], c[iv] = t["ur"] / u, t["ui"] / u
        family = ("V_TS" if name in ts_names else
                  "V_PCC" if name in pcc else "V_DS")
        out.append({"name": f"V_{name}", "family": family, "kind": "V", "c": c, "y0": u})
    return out


_PP_CACHE: Dict[str, tuple] = {}


def _pp_model(snapshot: Path) -> tuple:
    """pandapower rebuild of the snapshot with its branch admittances (cached)."""
    key = str(Path(snapshot).resolve())
    if key not in _PP_CACHE:
        import pandapower as pp
        from pandapower.pypower.makeYbus import makeYbus
        from export.dynamic_snapshot import load_snapshot_to_pandapower
        from pf.naming import build_name_map

        net, doc = load_snapshot_to_pandapower(snapshot)
        pp.runpp(net, **doc["solver_options"])
        ppc = net._ppc
        base = float(ppc["baseMVA"])
        Ybus, Yf, _ = makeYbus(base, ppc["bus"], ppc["branch"])
        _PP_CACHE[key] = (net, doc, build_name_map(doc), ppc, base, Ybus.tocsr(),
                          Yf.tocsr(), net._pd2ppc_lookups["bus"],
                          net._pd2ppc_lookups["branch"])
    return _PP_CACHE[key]


def branch_q_outputs(dae: Dae, op: dict, snapshot: Path,
                     include_ties: bool = False) -> Tuple[List[dict], pd.DataFrame]:
    """Coupler HV-side and TSO tie-line Q as functions of PF's node voltages,
    through pandapower's branch model (the PF/pandapower LF parity model)."""
    net, doc, names, ppc, base, Ybus, Yf, bl, br = _pp_model(snapshot)
    term = {t["name"]: t for t in op["terminals"]}
    pp2pf = {int(b): names[("bus", int(b))] for b in net.bus.index}

    def v_pf(pf_name):
        t = term[pf_name]
        return complex(t["ur"], t["ui"]), t

    def q_row(row: int, from_bus: int, star: Optional[int] = None):
        """(coefficients a_k on PF buses, from-bus PF name) of I_from."""
        yrow = Yf.getrow(row)
        coef: Dict[int, complex] = {}
        for c, val in zip(yrow.indices, yrow.data):
            coef[int(c)] = coef.get(int(c), 0) + val
        if star is not None and star in coef:
            ys = Ybus.getrow(star)
            yss = ys[0, star]
            a_s = coef.pop(star)
            for c, val in zip(ys.indices, ys.data):
                if c != star:
                    coef[int(c)] = coef.get(int(c), 0) - a_s * val / yss
        return coef

    # pandapower may fuse buses joined by closed bus-bus switches into one
    # ppc bus; their voltages are equal, so any of the PF names serves.
    ppc2pf = {}
    for b, pfn in pp2pf.items():
        ppc2pf.setdefault(int(bl[b]), pfn)

    outs, val = [], []

    def build(label, family, coef, from_pf, q_pf):
        vh, th = v_pf(from_pf)
        vec = {ppc2pf[k]: a for k, a in coef.items()}
        I = sum(a * v_pf(n)[0] for n, a in vec.items())
        q_lin = base * (vh * np.conj(I)).imag
        c = np.zeros(dae.N)
        for nm, a in vec.items():
            t = term[nm]
            iu = dae.index[(t["key"][0], t["key"][1], "ur")]
            iv = dae.index[(t["key"][0], t["key"][1], "ui")]
            d_ur = (vh * np.conj(a)).imag       # d/d ur_k of Im(V_h conj(I))
            d_ui = (vh * np.conj(1j * a)).imag
            if nm == from_pf:
                d_ur += (np.conj(I)).imag
                d_ui += (1j * np.conj(I)).imag
            c[iu] += base * d_ur
            c[iv] += base * d_ui
        outs.append({"name": label, "family": family, "kind": "Q", "c": c, "y0": q_lin})
        val.append({"output": label, "q_pp_model_mvar": q_lin, "q_pf_mvar": q_pf,
                    "diff_mvar": q_lin - q_pf if q_pf is not None else float("nan")})

    tr3_pf = {t["name"]: t for t in op["tr3"]}
    lo, hi = br["trafo3w"]
    n3 = len(net.trafo3w)
    for pos, idx in enumerate(net.trafo3w.index):
        pfn = names[("trafo3w", int(idx))]
        if not pfn.startswith("NC3W_") or pfn not in tr3_pf:
            continue
        row = lo + pos                                   # hv branch block first
        fb = int(ppc["branch"][row, 0].real)
        tb = int(ppc["branch"][row, 1].real)
        coef = q_row(row, fb, star=tb)
        build(f"Q_{pfn}", "Q_STS", coef, tr3_pf[pfn]["hv"], tr3_pf[pfn]["Q_hv"])
    assert hi - lo == 3 * n3

    if not include_ties:            # tie-line Q is not a measured variable
        return outs, pd.DataFrame(val)
    zone_of = {int(b): int(z) for z, bs in doc["zone_map"].items() for b in bs}
    lines_pf = {l["name"]: l for l in op["lines"]}
    lo, _ = br["line"]
    for pos, idx in enumerate(net.line.index):
        fb, tb = int(net.line.from_bus[idx]), int(net.line.to_bus[idx])
        zf, zt = zone_of.get(fb), zone_of.get(tb)
        if zf is None or zt is None or zf == zt or not net.line.in_service[idx]:
            continue
        pfn = names[("line", int(idx))]
        lp = lines_pf.get(pfn)
        if lp is None:
            continue
        row = lo + pos
        coef = q_row(row, int(ppc["branch"][row, 0].real))
        from_pf = pp2pf[fb]
        q_pf = lp["Q_1"] if lp["bus1"] == from_pf else lp["Q_2"]
        build(f"Q_{pfn}_z{zf}z{zt}", "Q_tie", coef, from_pf, q_pf)
    return outs, pd.DataFrame(val)


# ── responses and settling ─────────────────────────────────────────────────

def settle_time(tail: np.ndarray, band: float) -> float:
    over = np.where(np.abs(tail) > band)[0]
    if len(over) == 0:
        return T_MIN
    k = over[-1]
    return float(T_GRID[min(k + 1, len(T_GRID) - 1)]) if k + 1 < len(T_GRID) else math.inf


def analyse(red: Reduced, inputs: List[dict], outputs: List[dict],
            groups: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, dict]:
    ins = [u for u in inputs if "b" in u and "skip" not in u]
    n = red.dae.n
    Bf = np.column_stack([u["b"] for u in ins])
    B = np.column_stack([red.B(Bf[:, j]) for j in range(Bf.shape[1])])
    cz = np.vstack([o["c"][n:] for o in outputs])
    if any(np.abs(o["c"][:n]).max() > 0 for o in outputs):
        raise ValueError("outputs must be algebraic")
    C, D = red.CD(cz, Bf[n:, :])
    lam, clusters = clustered_modes(red.A)
    step = np.array([u["step"] for u in ins])
    band = np.array([BAND[o["kind"]] for o in outputs])

    live = [c for c in clusters if not c["zero"]]
    zero = [c for c in clusters if c["zero"]]
    R0 = sum((C @ c["Vc"]) @ (c["Wt"] @ B) for c in zero) if zero else np.zeros((len(outputs), len(ins)))
    lam_live = np.array([c["lam"] for c in live])
    Rk = np.stack([(C @ c["Vc"]) @ (c["Wt"] @ B) for c in live])    # (K, out, in)
    Rl = Rk / lam_live[:, None, None]
    y_inf = D - Rl.sum(axis=0)                                        # per unit input
    tau = np.where(lam_live.real != 0, 1 / np.abs(lam_live.real), np.inf)
    slow = tau > T_STS / KAPPA

    # tail on the grid, per pair: sum_k Rl_k e^{lam_k t}; clusters that are
    # below 1e-16 at T_MIN are dropped from the tail (kept in y_inf)
    act = np.exp(lam_live.real * T_MIN) > 1e-16
    E = np.exp(np.outer(lam_live[act], T_GRID))                       # (K', T)
    nO, nI = len(outputs), len(ins)
    Rflat = Rl[act].reshape(int(act.sum()), nO * nI).T                # (pairs, K')
    Tc = Rflat @ E
    tail = Tc.real.reshape(nO, nI, len(T_GRID))
    imag_resid = float(np.abs(Tc.imag).max())

    t20 = np.searchsorted(T_GRID, T_STS)
    t15 = np.searchsorted(T_GRID, T_STS - T_TAP)
    E20 = np.exp(lam_live * T_STS)
    rows = []
    for i, o in enumerate(outputs):
        for j, u in enumerate(ins):
            tl = tail[i, j] * step[j]
            dy = y_inf[i, j].real * step[j]
            ts_abs = settle_time(tl, band[i])
            ts_rel = settle_time(tl, math.exp(-KAPPA) * abs(dy)) if abs(dy) > 0 else 0.0
            contrib = np.abs(Rl[:, i, j] * E20) * step[j]
            k = int(np.argmax(contrib))
            slow_tail20 = float(np.abs((Rl[slow, i, j] * E20[slow]).sum()) * step[j])
            rows.append({
                "input": u["name"], "input_class": u["cls"], "step": u["step_label"],
                "output": o["name"], "family": o["family"], "kind": o["kind"],
                "dy_inf": dy, "D_instant": D[i, j].real * step[j],
                "Ts_abs_s": ts_abs, "Ts_rel_s": ts_rel,
                "tail_at_20s": float(tl[t20]), "tail_at_15s": float(tl[t15]),
                "slow_modes_tail_at_20s": slow_tail20,
                "band": band[i],
                "dom20_re": lam_live[k].real, "dom20_im": lam_live[k].imag,
                "dom20_tau_s": float(tau[k]),
                "ramp_from_zero_modes": float(np.abs(R0[i, j]) * step[j]),
            })
    df = pd.DataFrame(rows)

    # Per-mode view, directly from the modal data: a mode of amplitude
    # A = |R/lambda| * du in an output (x2 for a conjugate pair) stays outside
    # the band b until  T = tau * ln(A / b).  Worst case over all pairs.
    scale = step[None, :] / band[:, None]                              # (out, in)
    # only dispatch inputs enter the per-mode worst case (the G 01 V-ref
    # class is reported separately and is not dispatched)
    scale[:, [u["cls"] == "AVR_G01_equivalent" for u in ins]] = 0.0
    mrows = []
    for k, lamk in enumerate(lam_live):
        if lamk.imag < -1e-9:
            continue
        amp = np.abs(Rl[k]) * scale * (2.0 if lamk.imag > 1e-9 else 1.0)
        i, j = np.unravel_index(int(np.argmax(amp)), amp.shape)
        a = float(amp[i, j])
        mrows.append({"re": lamk.real, "im": lamk.imag, "tau_s": float(tau[k]),
                      "amp_over_band": a, "pair_input": ins[j]["name"],
                      "pair_output": outputs[i]["name"],
                      "Ts_mode_s": float(tau[k] * math.log(a)) if a > 1 else 0.0})
    modal = pd.DataFrame(mrows)
    info = {"n_states": n, "n_inputs": nI, "n_outputs": nO,
            "clusters": len(clusters), "zero_clusters": len(zero),
            "zero_modes": int(sum(len(c["idx"]) for c in zero)),
            "max_cluster_cond": float(max(c["cond"] for c in clusters)),
            "max_abs_ramp_from_zero_modes": float(df["ramp_from_zero_modes"].max()),
            "max_imag_residual_of_tail": imag_resid,
            "unstable": int((lam.real > 1e-6).sum())}
    info["modal"] = modal
    return df, info


def time_domain_check(red: Reduced, inputs: List[dict], outputs: List[dict],
                      df: pd.DataFrame, n_pairs: int = 4) -> pd.DataFrame:
    """Independent check of the modal step response: integrate x' = A x + B
    with a stiff solver and compare y(t) at a few times."""
    from scipy.integrate import solve_ivp
    ins = {u["name"]: u for u in inputs if "b" in u and "skip" not in u}
    outs = {o["name"]: o for o in outputs}
    n = red.dae.n
    worst = df.sort_values("Ts_abs_s", ascending=False).drop_duplicates("input_class").head(n_pairs)
    rows = []
    for r in worst.itertuples():
        u, o = ins[r.input], outs[r.output]
        b = red.B(u["b"])
        C, D = red.CD(o["c"][n:][None, :], u["b"][n:][:, None])
        sol = solve_ivp(lambda t, x: red.A @ x + b, (0, T_STS), np.zeros(n), method="BDF",
                        jac=red.A, t_eval=[T_STS - T_TAP, T_STS], rtol=1e-7, atol=1e-10)
        y = ((C @ sol.y).ravel() + D[0, 0]) * u["step"]
        for t, yy, modal in zip(sol.t, y, (r.tail_at_15s, r.tail_at_20s)):
            ode_tail = yy - r.dy_inf
            rows.append({"input": r.input, "output": r.output, "t": t,
                         "tail_modal": modal, "tail_ode": ode_tail,
                         "abs_diff": abs(ode_tail - modal), "band": r.band})
    return pd.DataFrame(rows)


# ── driver ─────────────────────────────────────────────────────────────────

def modal_settling(m: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    """Attach each eigenvalue cluster to its mode in the run's participation
    table and keep, per mode, the worst amplitude over both input systems."""
    modes = pd.read_csv(Path(run_dir) / "modal_modes.csv")
    z = modes["re"].to_numpy() + 1j * modes["im"].to_numpy()
    idx = [int(np.argmin(np.abs(z - complex(r, i)))) for r, i in zip(m["re"], m["im"])]
    keep = ["mode_idx", "f_hz", "zeta", "group", "dominant_state", "dominant_object",
            "in_L_VQ"] + [c for c in modes.columns if c.startswith("p_") and c != "p_abs_sum"
                          and c != "p_dominant"]
    m = m.reset_index(drop=True).join(modes.iloc[idx][keep].reset_index(drop=True))
    m = m.sort_values("amp_over_band", ascending=False).drop_duplicates("mode_idx")
    return m.sort_values("Ts_mode_s", ascending=False).reset_index(drop=True)


def run(window_dir: Path, out: Optional[Path] = None, ode_check: bool = True) -> Path:
    window_dir = Path(window_dir)
    state = json.loads((window_dir / "window_state.json").read_text(encoding="utf-8"))
    op = json.loads((window_dir / "op.json").read_text(encoding="utf-8"))
    snapshot = PROJECT_ROOT / state["snapshot"] if not Path(state["snapshot"]).is_absolute() \
        else Path(state["snapshot"])
    out = out or window_dir / "observability"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    all_rows = []
    for case in ("db_wide", "db_zero"):
        run_dir = Path(state["cases"][case]["run_dir"])
        run_t = Path(state["cases_tapsin"][case]["run_dir"])
        dae, dae_t = load_dae(run_dir), load_dae(run_t)
        red = Reduced(dae)
        A_pf = mp.read_amat(run_dir / "matl", dae.n)
        check = {"A_vs_Amat_rel": float(np.abs(red.A - A_pf).max() / np.abs(A_pf).max())}
        inputs = der_inputs(dae, op) + avr_inputs(dae)
        ts_n, ds_n, vinfo = controlled_voltage_names(snapshot)
        check["measured_voltage_sets"] = vinfo
        outputs = voltage_outputs(dae, op, ts_n, ds_n)
        qouts, qval = branch_q_outputs(dae, op, snapshot)
        outputs += qouts
        qval.to_csv(out / f"branch_q_validation_{case}.csv", index=False)
        check["branch_q_max_abs_diff_mvar"] = float(qval["diff_mvar"].abs().max())
        df, info = analyse(red, inputs, outputs)
        tdc = (time_domain_check(red, inputs, outputs, df, n_pairs=2) if ode_check
               else pd.DataFrame(columns=["abs_diff", "band"]))

        # taps: in the taps-in system's own coordinates
        sys_t, taps = tapsin_system(dae_t)
        not_dispatched = slack_machine_taps(snapshot)
        for u in taps:
            if u["name"] in not_dispatched:
                u["skip"] = ("slack/equivalent machine (G 01) OLTC: excluded from the "
                             "TSO OLTC set in experiments/runners/multi_tso_dso.py")
        inputs_skipped_taps = [u for u in taps if "skip" in u]
        red_t = Reduced(sys_t)
        check["taps_n_inputs"] = len(taps) - len(inputs_skipped_taps)
        check["taps_spectrum_rel_dist"] = spectrum_distance(red.A, red_t.A)
        outputs_t = (voltage_outputs(sys_t, op, ts_n, ds_n)
                     + branch_q_outputs(sys_t, op, snapshot)[0])
        if [o["name"] for o in outputs_t] != [o["name"] for o in outputs]:
            raise ValueError("taps-in system: output set differs")
        df_t, info_t = analyse(red_t, taps, outputs_t)
        if ode_check:
            tdc = pd.concat([tdc, time_domain_check(red_t, taps, outputs_t, df_t, n_pairs=1)])
            tdc.to_csv(out / f"time_domain_check_{case}.csv", index=False)
            check["time_domain_max_abs_diff_over_band"] = float(
                (tdc["abs_diff"] / tdc["band"]).max())
        modal = modal_settling(pd.concat([info.pop("modal"), info_t.pop("modal")]),
                               run_dir)
        modal.to_csv(out / f"modal_settling_{case}.csv", index=False)
        summary.setdefault("_modal", {})[case] = modal
        info["taps_system"] = info_t
        df = pd.concat([df, df_t], ignore_index=True)
        df.insert(0, "droop_case", case)
        all_rows.append(df)
        summary[case] = {"checks": check, "info": info,
                         "skipped_inputs": [(u["name"], u["skip"]) for u in
                                            inputs + inputs_skipped_taps if "skip" in u],
                         "n_outputs_by_family": pd.Series([o["family"] for o in outputs]).value_counts().to_dict()}
        print(f"[obs] {case}: {info}; checks {check}")
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(out / "pairs.csv", index=False)
    modal = summary.pop("_modal", {})
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    write_summary_md(df, summary, out / "summary.md", state["window"], modal)
    return out


def run_shunts(window_dir: Path, out: Optional[Path] = None, ode_check: bool = True) -> Path:
    """The MSC/MSR bank steps on the taps-out linearisation of a window run
    (same matrices, outputs and bands as :func:`run`; the responses are linear
    per input, so they extend that run's pairs without recomputing them)."""
    window_dir = Path(window_dir)
    state = json.loads((window_dir / "window_state.json").read_text(encoding="utf-8"))
    op = json.loads((window_dir / "op.json").read_text(encoding="utf-8"))
    snapshot = PROJECT_ROOT / state["snapshot"] if not Path(state["snapshot"]).is_absolute() \
        else Path(state["snapshot"])
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != state["sha256"]:
        raise ValueError(f"{snapshot}: not the snapshot of the window run")
    out = out or window_dir / "observability_shunt"
    out.mkdir(parents=True, exist_ok=True)
    summary, all_rows, modal = {}, [], {}
    for case in ("db_wide", "db_zero"):
        run_dir = Path(state["cases"][case]["run_dir"])
        dae = load_dae(run_dir)
        red = Reduced(dae)
        A_pf = mp.read_amat(run_dir / "matl", dae.n)
        check = {"A_vs_Amat_rel": float(np.abs(red.A - A_pf).max() / np.abs(A_pf).max())}
        inputs, parity = shunt_inputs(dae, op, snapshot)
        parity.to_csv(out / f"shunt_parity_{case}.csv", index=False)
        expect = _pp_model(snapshot)[4] / PF_ADMITTANCE_BASE_MVA
        check["admittance_scale_max_rel_dev"] = float(
            (parity["admittance_scale"] / expect - 1.0).abs().max())
        ts_n, ds_n, vinfo = controlled_voltage_names(snapshot)
        check["measured_voltage_sets"] = vinfo
        outputs = voltage_outputs(dae, op, ts_n, ds_n) + branch_q_outputs(dae, op, snapshot)[0]
        df, info = analyse(red, inputs, outputs)
        if ode_check:
            tdc = time_domain_check(red, inputs, outputs, df, n_pairs=2)
            tdc.to_csv(out / f"time_domain_check_{case}.csv", index=False)
            check["time_domain_max_abs_diff_over_band"] = float(
                (tdc["abs_diff"] / tdc["band"]).max())
        if case == "db_wide":
            lf = shunt_lf_check(df, inputs, snapshot)
            lf.to_csv(out / "shunt_lf_check_db_wide.csv", index=False)
            for kind, x in lf.groupby("kind"):
                check[f"lf_check_max_abs_diff_{kind}"] = float(x["diff"].abs().max())
                check[f"lf_check_max_abs_diff_central_{kind}"] = float(
                    x["diff_central"].abs().max())
        modal[case] = modal_settling(info.pop("modal"), run_dir)
        modal[case].to_csv(out / f"modal_settling_{case}.csv", index=False)
        df.insert(0, "droop_case", case)
        all_rows.append(df)
        summary[case] = {"checks": check, "info": info, "skipped_inputs": [],
                         "n_outputs_by_family": pd.Series([o["family"] for o in outputs]).value_counts().to_dict()}
        print(f"[obs/shunt] {case}: {info}; checks {check}")
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(out / "pairs.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    write_summary_md(df, summary, out / "summary.md", state["window"], modal)
    return out


def write_summary_md(df: pd.DataFrame, summary: dict, path: Path, window: str,
                     modal: Optional[Dict[str, pd.DataFrame]] = None) -> None:
    L = [f"# Residue-based settling, {window}", "",
         f"Bands V {BAND['V']} pu, Q {BAND['Q']} Mvar; steps DER min({DER_STEP_MVAR:g} Mvar, "
         f"0.5 S_n, headroom), AVR {AVR_STEP_PU} pu, 1 tap. T_s = last time |y - y_inf| > band "
         f"(taps timed from contact). Relative: band e^-{KAPPA:g}|dy_inf|.", ""]
    for case, s in summary.items():
        L += [f"## {case}", "", f"- checks: {s['checks']}", f"- info: {s['info']}",
              f"- outputs: {s['n_outputs_by_family']}",
              f"- skipped inputs: {len(s['skipped_inputs'])}", ""]
        d = df[df.droop_case == case]
        L += ["| input class | max T_s abs [s] | threshold [s] | met | margin [s] |",
              "|:--|--:|--:|:--|--:|"]
        for cls, x in d.groupby("input_class"):
            ts = x["Ts_abs_s"].max()
            thr = T_STS - T_TAP if cls.startswith("TAP") else T_STS
            note = " (not dispatched in the thesis configuration; outside the verdict)" \
                if cls == "AVR_G01_equivalent" else ""
            L.append(f"| {cls}{note} | {ts:.2f} | {thr:g} | {'yes' if ts <= thr else 'NO'} | "
                     f"{thr - ts:.2f} |")
        L.append("")
        g = d.groupby(["input_class", "family"])
        tab = g.apply(lambda x: x.loc[x["Ts_abs_s"].idxmax()]).reset_index(drop=True)
        L += ["| input class | output family | max T_s abs [s] | pair | tail @20 s | "
              "slow-mode tail @20 s | band | dominant mode @20 s (tau) | max T_s rel [s] |",
              "|:--|:--|--:|:--|--:|--:|--:|--:|--:|"]
        for r in tab.itertuples():
            rel = d[(d.input_class == r.input_class) & (d.family == r.family)]["Ts_rel_s"].max()
            L.append(f"| {r.input_class} | {r.family} | {r.Ts_abs_s:.2f} | {r.input} -> "
                     f"{r.output} | {r.tail_at_20s:.2e} | {r.slow_modes_tail_at_20s:.2e} | "
                     f"{r.band:g} | {r.dom20_re:.4f}{r.dom20_im:+.4f}j ({r.dom20_tau_s:.2f} s) "
                     f"| {rel:.2f} |")
        L.append("")
        if modal and case in modal:
            m = modal[case]
            m = m[m["Ts_mode_s"] > 0].head(15)
            L += ["Per-mode settling, directly from the modal data: T = tau ln(A/band), "
                  "A = the mode's largest amplitude over all dispatch -> output pairs.", "",
                  "| mode | f [Hz] | zeta | tau [s] | group | dominant | A/band | T = tau ln(A/band) [s] "
                  "| worst pair |", "|--:|--:|--:|--:|:--|:--|--:|--:|:--|"]
            for r in m.itertuples():
                L.append(f"| {r.mode_idx} | {r.f_hz:.3f} | {r.zeta:.3f} | {r.tau_s:.2f} | "
                         f"{r.group} | {r.dominant_state} @ {r.dominant_object.split(chr(92))[-1]} | "
                         f"{r.amp_over_band:.2f} | {r.Ts_mode_s:.2f} | {r.pair_input} -> "
                         f"{r.pair_output} |")
            L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("window_dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-ode-check", action="store_true",
                    help="skip the stiff-ODE cross-check (already passed on the same "
                         "matrices; a changed step size scales responses linearly)")
    ap.add_argument("--shunts", action="store_true",
                    help="only the MSC/MSR bank steps (default out: <window>/observability_shunt)")
    args = ap.parse_args(argv)
    out = (run_shunts if args.shunts else run)(args.window_dir, args.out,
                                               ode_check=not args.no_ode_check)
    print((out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
