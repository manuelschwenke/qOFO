"""
pf/modal_participation.py
=========================
Participation factors of the PowerFactory modal analysis, read from ComMod's
Matlab export, reduced to the per-mode table of ``ch:timescales:plant``.

No ``powerfactory`` import: everything here works on the exported files, so
it is testable offline.  ``pf/screening.py::cmd_modal`` supplies the object
metadata (model class, DSL type, parent) that the files do not carry.

Export layout (PF 2025 SP4, established 2026-09-22 against the live export):

* ``EVals.mtl``           ``k 1 Re Im`` per eigenvalue, 1-based ``k``.
* ``PartFacs.mtl``        sparse ``i j Re Im``: row ``i`` = STATE, column
                          ``j`` = MODE.  For simple real modes it equals
                          ``w_ij v_ij / (w_j^T v_j)`` exactly.  The exported
                          ``lEV_Amat``/``rEV_Amat`` columns are NOT
                          consistently paired for complex or repeated
                          eigenvalues, so participation is taken from this
                          file and never recomputed from the eigenvectors.
* ``VariableToIdx_Amat.txt``  ``idx  <grid>\\<parent>\\<obj>.<Class>  "state"``.
* ``Amat.mtl``            sparse ``i j a_ij``; ``eig(A)`` reproduces
                          ``EVals`` to ~2e-11 relative.

Normalisation: ``p_kj = |P_kj| / sum_k |P_kj|`` (magnitudes, per mode, sum
to 1) -- the "normalised participation sum" of the declared rule.

Author: Manuel Schwenke / Claude Code (2026-09-22)
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.modal_reduce import (  # noqa: E402
    COLUMNS, GROUPS, P_COLS, in_l_vq,
)

#: One object's metadata as cmd_modal resolves it in PF:
#: ``(model class, DSL type or "", parent loc_name, parent class)``.
ObjMeta = Tuple[str, str, str, str]

_VARIDX_RE = re.compile(r'^\s*(\d+)\s+(.+?)\s+"([^"]+)"\s*$')
_CONST_RE = re.compile(r"^\s*(.+?) \[(.+?)\]: (.+?)\s*$")


# ── file readers ───────────────────────────────────────────────────────────

def _triplets(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None, engine="c").to_numpy()


def read_evals(export_dir: Path) -> np.ndarray:
    x = _triplets(Path(export_dir) / "EVals.mtl")
    order = np.argsort(x[:, 0])
    return (x[order, 2] + 1j * x[order, 3]).astype(complex)


def read_partfacs(export_dir: Path, n: int) -> np.ndarray:
    """Complex participation matrix, states x modes."""
    x = _triplets(Path(export_dir) / "PartFacs.mtl")
    P = np.zeros((n, n), dtype=complex)
    P[x[:, 0].astype(int) - 1, x[:, 1].astype(int) - 1] = x[:, 2] + 1j * x[:, 3]
    return P


def read_amat(export_dir: Path, n: int) -> np.ndarray:
    x = _triplets(Path(export_dir) / "Amat.mtl")
    A = np.zeros((n, n))
    A[x[:, 0].astype(int) - 1, x[:, 1].astype(int) - 1] = x[:, 2]
    return A


def read_varidx(export_dir: Path) -> pd.DataFrame:
    """State index table: ``idx, path, parent, obj, obj_class, state``."""
    rows = []
    text = (Path(export_dir) / "VariableToIdx_Amat.txt").read_text(
        encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = _VARIDX_RE.match(line)
        if not m:
            continue
        idx, path, state = int(m.group(1)), m.group(2).strip(), m.group(3)
        parts = path.split("\\")
        leaf = parts[-1]
        obj, _, cls = leaf.rpartition(".")
        rows.append({"idx": idx, "path": path,
                     "parent": parts[-2] if len(parts) > 1 else "",
                     "obj": obj, "obj_class": cls, "state": state})
    df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    if len(df) and (df["idx"].to_numpy() != np.arange(1, len(df) + 1)).any():
        raise ValueError("VariableToIdx_Amat.txt: state indices are not 1..n")
    return df


def parse_constant_states(lines: Iterable[str]) -> Set[Tuple[str, str, str]]:
    """``(parent, object loc_name, state)`` PF reports as constant at the OP.

    Parses the block after "The state variables of the following models were
    assumed to be constant at current operating point:" in the ComMod output
    window, lines of the form ``AVR 03 [Power Plant 03]: xr`` or
    ``REEC_D Electrical Control Model [WECC_...]: x2, x3``.
    """
    out: Set[Tuple[str, str, str]] = set()
    active = False
    # GetContent() returns one element per MESSAGE; a message may span lines.
    flat = [ln for raw in lines for ln in str(raw).splitlines()]
    for raw in flat:
        s = raw.rstrip()
        if "assumed to be constant at current operating point:" in s:
            active = True
            continue
        if not active:
            continue
        m = _CONST_RE.match(s)
        if not m or s.lstrip().startswith(("info", "warn", "error", "---")):
            if s.strip():
                active = False
            continue
        obj, parent, states = m.group(1).strip(), m.group(2).strip(), m.group(3)
        for st in states.split(","):
            out.add((parent, obj, st.strip()))
    return out


# ── grouping ───────────────────────────────────────────────────────────────

def group_of(obj_class: str, dsl_type: str, parent: str) -> str:
    """Declared group of a state from its owning object's model class.

    ElmSym -> rotor; AVR/exciter DSL -> exc; governor/turbine DSL -> gov;
    REGC/REEC/REPC/QVPRE/WTG DSL, and any block inside a ``WECC_`` composite
    (StaVmea, protection, weak-grid option) -> conv; everything else (loads,
    tap controllers, unresolved) -> other, listed explicitly downstream.
    """
    t = (dsl_type or "").lower()
    if obj_class == "ElmSym":
        return "rotor"
    if obj_class == "ElmDsl":
        if t.startswith(("avr", "ieeet", "sexs", "esst", "esac", "esdc", "exc")):
            return "exc"
        if t.startswith(("gov", "ieeeg", "turb", "hygov", "tgov")):
            return "gov"
        if t.startswith(("regc", "reec", "repc", "qvpre", "wtg")):
            return "conv"
    if str(parent).startswith("WECC_"):
        return "conv"
    return "other"


def state_table(varidx: pd.DataFrame, meta: Mapping[Tuple[str, str], ObjMeta],
                constant: Set[Tuple[str, str, str]]) -> pd.DataFrame:
    """Attach model class, DSL type, group and the constant flag per state.

    ``meta`` is keyed ``(parent loc_name, "<obj>.<Class>")`` -- the last two
    components of the index path, which is what cmd_modal can build from PF.
    """
    st = varidx.copy()
    cls, typ, grp, found = [], [], [], []
    for _, r in st.iterrows():
        m = meta.get((r["parent"], f"{r['obj']}.{r['obj_class']}"))
        if m is None:
            c, t, par = r["obj_class"], "", r["parent"]
            found.append(False)
        else:
            c, t, par, _ = m
            found.append(True)
        cls.append(c if not t else f"{c}:{t}")
        typ.append(t)
        grp.append(group_of(c, t, par))
    st["model_class"] = cls
    st["dsl_type"] = typ
    st["group"] = grp
    st["resolved"] = found
    st["constant_at_op"] = [(r["parent"], r["obj"], r["state"]) in constant
                            for _, r in st.iterrows()]
    return st


# ── per-mode table ─────────────────────────────────────────────────────────

def normalised(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``|P|`` normalised per mode column; also the raw column sums."""
    mag = np.abs(P)
    s = mag.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pn = np.where(s > 0, mag / np.where(s > 0, s, 1.0), np.nan)
    return pn, s


def mode_table(lam: np.ndarray, P: np.ndarray, states: pd.DataFrame, *,
               window: str, droop_case: str, other_is_vq: bool = True,
               top_k: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(modes, top_states) for every eigenvalue with Im >= 0.

    ``modes`` carries the ``modal_modes.csv`` columns plus diagnostics
    (``dominant_group``, ``dominant_constant``, ``p_dominant``,
    ``p_abs_sum``); ``top_states`` the ``top_k`` states per mode.
    """
    n = len(lam)
    if P.shape != (n, n) or len(states) != n:
        raise ValueError(f"shape mismatch: lam {n}, P {P.shape}, states {len(states)}")
    pn, psum = normalised(P)
    gidx = {g: np.where(states["group"].to_numpy() == g)[0] for g in GROUPS}
    rows, tops = [], []
    for j in range(n):
        re_, im_ = float(lam[j].real), float(lam[j].imag)
        if im_ < -1e-9:
            continue
        col = pn[:, j]
        gp = {g: float(np.nansum(col[ix])) for g, ix in gidx.items()}
        group = max(GROUPS, key=lambda g: gp[g])
        k = int(np.nanargmax(col)) if np.isfinite(col).any() else 0
        s = states.iloc[k]
        wn = math.hypot(re_, im_)
        rows.append({
            "window": window, "droop_case": droop_case, "mode_idx": j + 1,
            "re": re_, "im": im_, "f_hz": abs(im_) / (2 * math.pi),
            "zeta": (-re_ / wn) if wn > 1e-12 else 1.0,
            "tau_s": 1.0 / abs(re_) if abs(re_) > 0 else math.inf,
            "ts_s": 4.0 / abs(re_) if abs(re_) > 0 else math.inf,
            "group": group,
            "dominant_state": s["state"],
            "dominant_object": f"{s['parent']}\\{s['obj']}",
            "dominant_class": s["model_class"],
            **{f"p_{g}": gp[g] for g in GROUPS},
            "in_L_VQ": in_l_vq(group, gp, other_is_vq),
            "dominant_group": s["group"],
            "dominant_constant": bool(s["constant_at_op"]),
            "p_dominant": float(col[k]) if np.isfinite(col[k]) else float("nan"),
            "p_abs_sum": float(psum[j]),
        })
        order = np.argsort(-np.nan_to_num(col, nan=-1.0))[:top_k]
        for rank, kk in enumerate(order, start=1):
            ss = states.iloc[int(kk)]
            tops.append({"window": window, "droop_case": droop_case,
                         "mode_idx": j + 1, "rank": rank, "p": float(col[kk]),
                         "state": ss["state"],
                         "object": f"{ss['parent']}\\{ss['obj']}",
                         "model_class": ss["model_class"], "group": ss["group"],
                         "constant_at_op": bool(ss["constant_at_op"])})
    modes = pd.DataFrame(rows)
    extra = [c for c in modes.columns if c not in COLUMNS]
    return modes[list(COLUMNS) + extra], pd.DataFrame(tops)


def degenerate_clusters(modes: pd.DataFrame, rel_tol: float = 1e-8) -> pd.DataFrame:
    """Clusters of (numerically) repeated eigenvalues and their groups.

    Per-state participation inside a repeated eigenvalue's eigenspace is not
    unique, so a group label is only robust if the whole cluster falls into
    one group.  Returns one row per cluster of size > 1.
    """
    out = []
    z = modes["re"].to_numpy() + 1j * modes["im"].to_numpy()
    used = np.zeros(len(z), dtype=bool)
    for i in range(len(z)):
        if used[i]:
            continue
        tol = rel_tol * max(abs(z[i]), 1e-12)
        same = np.where(np.abs(z - z[i]) <= max(tol, 1e-12))[0]
        used[same] = True
        if len(same) > 1:
            g = modes.iloc[same]["group"]
            out.append({"re": z[i].real, "im": z[i].imag, "size": len(same),
                        "groups": ",".join(sorted(g.unique())),
                        "single_group": g.nunique() == 1})
    return pd.DataFrame(out, columns=["re", "im", "size", "groups", "single_group"])
