#!/usr/bin/env python3
r"""Reduction of the PowerFactory modal screen for the plant-time-constant section.

Reads one or more ``modal_modes.csv`` files written by
``pf/screening.py modal`` (one row per mode, conjugate pairs counted once),
applies the **declared** selection rule of ``ch:timescales:plant`` and writes
the bundle the dissertation copies in::

    modal_modes.csv       all rows, all windows, both droop cases (merged)
    modal_groups.tex      tabular BODY, one row per group, ranges over windows,
                          both droop cases pooled; ``modal_groups_<case>.tex``
                          per droop case
    modal_spectrum.pdf    sigma vs omega, coloured by group, boundary
                          sigma = -kappa / T_STS
    near_zero_modes.md    stop rule 2: every |Re| < 1e-3, with dominant state
    modal_summary.md      the numbers the section's [TBD]s need, per droop case

Declared rule (author-confirmed 2026-09-20; NOT tuned here)::

    T_STS >= max( kappa * max_{l in L_VQ} tau_l ,  T_tap + kappa * tau_exc )
    tau_l = 1/|Re lambda_l|,  kappa = 3,  T_tap = 5 s,  T_STS = 20 s

Group of a mode = the group with the largest normalised participation.
Material participation: p >= 0.10.  L_VQ = rotor, excitation and converter
groups, plus any mode with material participation in a V/Q-relevant group.

Usage::

    python -m analysis.modal_reduce RUN1/modal_modes.csv RUN2/modal_modes.csv \
        --out results/THESIS_ch8_modal_<date>
    python -m analysis.modal_reduce --synthetic <dir>   # offline dry run

Author: Manuel Schwenke / Claude Code (2026-09-22)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── declared constants (do not tune to the result) ─────────────────────────
KAPPA = 3.0
P_MATERIAL = 0.10
T_STS = 20.0
T_TAP = 5.0
#: |Re| below this is a near-zero mode (stop rule 2): tau is not meaningful.
NEAR_ZERO_RE = 1e-3
#: Re above this counts as unstable; the same numerical-zero tolerance as
#: ``cmd_modal``'s existing unstable count.
UNSTABLE_RE = 1e-6

GROUPS = ("rotor", "exc", "gov", "conv", "other")
GROUP_LABEL = {
    "rotor": "Rotor angle and speed",
    "exc": "Excitation",
    "gov": "Governor and turbine",
    "conv": "Converter control and droop",
    "other": "Other",
}
P_COLS = tuple(f"p_{g}" for g in GROUPS)
#: Groups whose states act on voltage or reactive power.  ``other`` is
#: counted as V/Q-relevant by default (conservative: it can only enlarge
#: L_VQ); every ``other`` state is listed so this can be revisited.
VQ_GROUPS = ("rotor", "exc", "conv")

COLUMNS = ("window", "droop_case", "mode_idx", "re", "im", "f_hz", "zeta",
           "tau_s", "ts_s", "group", "dominant_state", "dominant_object",
           "dominant_class") + P_COLS + ("in_L_VQ",)
#: Diagnostics ``pf/modal_participation.py`` adds; carried through if present.
OPTIONAL_COLUMNS = ("dominant_group", "dominant_constant", "p_dominant", "p_abs_sum")


def vq_groups(other_is_vq: bool = True) -> tuple:
    return VQ_GROUPS + (("other",) if other_is_vq else ())


def in_l_vq(group: str, p: Dict[str, float], other_is_vq: bool = True) -> bool:
    """Declared L_VQ membership of one mode.

    ``p`` maps group -> normalised participation (sums to 1).  A mode is in
    L_VQ if its own group is V/Q-relevant, or if any V/Q-relevant group holds
    material participation -- so a governor mode is excluded only when every
    V/Q-relevant group stays below ``P_MATERIAL``.
    """
    vq = vq_groups(other_is_vq)
    if group in vq:
        return True
    return any(float(p.get(g, 0.0)) >= P_MATERIAL for g in vq)


def tau_limit() -> float:
    """Largest admissible tau in L_VQ: T_STS / kappa."""
    return T_STS / KAPPA


# ── loading and derived columns ────────────────────────────────────────────

def load(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{p}: missing columns {missing}")
        frames.append(df[list(COLUMNS) + [c for c in OPTIONAL_COLUMNS if c in df.columns]])
    if not frames:
        raise ValueError("no modal_modes.csv given")
    df = pd.concat(frames, ignore_index=True)
    dup = df.duplicated(["window", "droop_case", "mode_idx"])
    if dup.any():
        raise ValueError(f"duplicate (window, droop_case, mode_idx): "
                         f"{df.loc[dup, ['window', 'droop_case', 'mode_idx']].values.tolist()}")
    return df


def derive(df: pd.DataFrame, other_is_vq: bool = True) -> pd.DataFrame:
    """Recompute tau and L_VQ from Re and the participation columns.

    The recomputed membership is authoritative; ``in_L_VQ_csv`` keeps what the
    emitter wrote so any disagreement is visible rather than silently fixed.
    """
    out = df.copy()
    re = out["re"].astype(float)
    out["near_zero"] = re.abs() < NEAR_ZERO_RE
    out["unstable"] = re > UNSTABLE_RE
    out["tau_s"] = np.where(re.abs() > 0, 1.0 / re.abs(), np.inf)
    out["ts_s"] = 4.0 * out["tau_s"]
    psum = out[list(P_COLS)].sum(axis=1)
    out["p_sum"] = psum
    out["in_L_VQ_csv"] = out["in_L_VQ"].astype(str).str.lower().isin(("1", "true", "yes"))
    out["in_L_VQ"] = [
        in_l_vq(g, {k: r[f"p_{k}"] for k in GROUPS}, other_is_vq)
        for g, (_, r) in zip(out["group"], out.iterrows())
    ]
    out["p_vq"] = out[[f"p_{g}" for g in vq_groups(other_is_vq)]].max(axis=1)
    out["p_vq_sum"] = out[[f"p_{g}" for g in vq_groups(other_is_vq)]].sum(axis=1)
    return out


# ── rule evaluation ────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame) -> Dict[str, dict]:
    """Per droop case: the quantities of the declared rule and stop rules."""
    res: Dict[str, dict] = {}
    lim = tau_limit()
    for case, d in df.groupby("droop_case", sort=True):
        finite = d[~d["near_zero"]]
        lvq = finite[finite["in_L_VQ"]]
        exc = finite[finite["group"] == "exc"]
        r: dict = {"case": case}
        r["modes_per_window"] = d.groupby("window").size().to_dict()
        r["unstable_per_window"] = d.groupby("window")["unstable"].sum().astype(int).to_dict()
        r["near_zero"] = int(d["near_zero"].sum())
        r["slowest_lvq"] = _row(lvq.loc[lvq["tau_s"].idxmax()]) if len(lvq) else None
        r["tau_lvq_max"] = float(lvq["tau_s"].max()) if len(lvq) else float("nan")
        r["slowest_exc"] = _row(exc.loc[exc["tau_s"].idxmax()]) if len(exc) else None
        r["tau_exc"] = float(exc["tau_s"].max()) if len(exc) else float("nan")
        r["tap_term"] = T_TAP + KAPPA * r["tau_exc"]
        r["lvq_term"] = KAPPA * r["tau_lvq_max"]
        r["required"] = float(np.nanmax([r["lvq_term"], r["tap_term"]]))
        r["margin"] = T_STS - r["required"]
        r["slowest_overall"] = (_row(finite.loc[finite["tau_s"].idxmax()])
                                if len(finite) else None)
        # CHECKPOINT R1: slow L_VQ modes, and governor modes with material
        # V/Q participation (whatever their tau).
        r1a = lvq[lvq["tau_s"] > lim]
        r1b = finite[(finite["group"] == "gov") & finite["in_L_VQ"]]
        r["r1_slow_lvq"] = [_row(x) for _, x in r1a.iterrows()]
        r["r1_gov_vq"] = [_row(x) for _, x in r1b.iterrows()]
        # near-zero modes inside L_VQ would trigger R1 unless explained
        nz = d[d["near_zero"] & d["in_L_VQ"]]
        r["r1_pending_near_zero"] = [_row(x) for _, x in nz.iterrows()]
        r["r1_triggered"] = bool(len(r1a) or len(r1b))
        # informative: L_VQ modes where the governor holds material share, and
        # modes kept out of L_VQ only by the per-group (not summed) test
        r["lvq_with_gov_share"] = [_row(x) for _, x in
                                   finite[finite["in_L_VQ"] & (finite["p_gov"] >= P_MATERIAL)].iterrows()]
        r["borderline_excluded"] = [_row(x) for _, x in
                                    finite[~finite["in_L_VQ"] & (finite["p_vq_sum"] >= P_MATERIAL)].iterrows()]
        r["csv_lvq_disagreements"] = int((d["in_L_VQ"] != d["in_L_VQ_csv"]).sum())
        res[case] = r
    return res


def _row(x: pd.Series) -> dict:
    keys = ("window", "droop_case", "mode_idx", "re", "im", "f_hz", "zeta",
            "tau_s", "group", "dominant_state", "dominant_object",
            "dominant_class", "in_L_VQ") + P_COLS
    return {k: (x[k].item() if hasattr(x[k], "item") else x[k]) for k in keys}


# ── LaTeX table body ───────────────────────────────────────────────────────

def _num(x: float, sig: int = 2) -> str:
    if not math.isfinite(x):
        return r"$\infty$"
    if x == 0:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return s


def _rng(values: Sequence[float], sig: int = 2) -> str:
    v = [float(x) for x in values if math.isfinite(float(x))]
    if not v:
        return "--"
    lo, hi = _num(min(v), sig), _num(max(v), sig)
    return lo if lo == hi else f"{lo}--{hi}"


def groups_tex(df: pd.DataFrame) -> str:
    """Rows ``Group & Modes & f/Hz & zeta & tau/s & In L_VQ \\``.

    Ranges run over every (window, droop case) present.  Near-zero modes are
    counted in *Modes* but excluded from the f, zeta and tau ranges; they are
    listed in ``near_zero_modes.md``.
    """
    lines = [f"% generated by analysis/modal_reduce.py; windows: "
             f"{', '.join(sorted(df['window'].unique()))}; droop cases: "
             f"{', '.join(sorted(df['droop_case'].unique()))}"]
    nz = int(df["near_zero"].sum())
    if nz:
        lines.append(f"% {nz} near-zero mode row(s) excluded from f/zeta/tau "
                     f"ranges, see near_zero_modes.md")
    combos = df.groupby(["window", "droop_case"]).size().index
    for g in GROUPS:
        d = df[df["group"] == g]
        if g == "other" and d.empty:
            continue
        counts = [int(((df["window"] == w) & (df["droop_case"] == c)
                       & (df["group"] == g)).sum()) for w, c in combos]
        cnt = str(min(counts)) if min(counts) == max(counts) else f"{min(counts)}--{max(counts)}"
        fin = d[~d["near_zero"]]
        n_in, n = int(d["in_L_VQ"].sum()), len(d)
        lvq = "yes" if n and n_in == n else ("no" if n_in == 0 else f"partly ({n_in}/{n})")
        lines.append(f"{GROUP_LABEL[g]} & {cnt} & {_rng(fin['f_hz'])} & "
                     f"{_rng(fin['zeta'])} & {_rng(fin['tau_s'])} & {lvq} \\\\")
    return "\n".join(lines) + "\n"


# ── near-zero list ─────────────────────────────────────────────────────────

def _candidate_explanation(r: pd.Series) -> str:
    """A *candidate* only; every entry needs the author's or the session's
    confirmation before the mode is set aside."""
    obj = str(r["dominant_object"])
    cls = str(r["dominant_class"])
    st = str(r["dominant_state"]).lower()
    if str(r.get("dominant_constant", "")).lower() in ("true", "1"):
        return ("structural: PF reports this state constant at the operating "
                "point (repConstantStates)")
    if cls.startswith("ElmSym") and obj.split("\\")[-1].strip() == "G 01" \
            and ("phi" in st or "ang" in st):
        return "candidate: angle reference (G 01) -- structural zero"
    if "REGC" in cls and "phi" in st:
        return ("candidate: converter PLL angle integrator -- invariance to "
                "the absolute angle reference")
    if cls.startswith("ElmSym") and ("phi" in st or "ang" in st):
        return "candidate: rotor-angle common mode -- check reference"
    if r["group"] == "gov":
        return "candidate: governor integrator state -- check block"
    return "UNEXPLAINED -- to be explained"


def near_zero_md(df: pd.DataFrame) -> str:
    nz = df[df["near_zero"]].sort_values(["droop_case", "window", "mode_idx"])
    lines = ["# Near-zero modes (|Re| < 1e-3)", "",
             f"Stop rule 2. {len(nz)} row(s). `tau` is not meaningful here "
             "(structural zeros have tau = inf). Explanations marked "
             "*candidate* are heuristics from the dominant state and are NOT "
             "established until confirmed.", "",
             "| window | case | idx | Re | Im | group | dominant state | object | class "
             "| p_rotor | p_exc | p_gov | p_conv | p_other | in L_VQ | explanation |",
             "|:--|:--|--:|--:|--:|:--|:--|:--|:--|--:|--:|--:|--:|--:|:--|:--|"]
    for _, r in nz.iterrows():
        lines.append(
            f"| {r['window']} | {r['droop_case']} | {r['mode_idx']} | {r['re']:.3e} "
            f"| {r['im']:.4f} | {r['group']} | {r['dominant_state']} | {r['dominant_object']} "
            f"| {r['dominant_class']} | " + " | ".join(f"{r[c]:.3f}" for c in P_COLS)
            + f" | {'yes' if r['in_L_VQ'] else 'no'} | {_candidate_explanation(r)} |")
    return "\n".join(lines) + "\n"


# ── spectrum plot ──────────────────────────────────────────────────────────

def spectrum_pdf(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from configs.color_config import TU_COLOURS

    plt.rcParams.update({
        "figure.figsize": (5.2, 3.4), "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.fontsize": 7.5,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    colour = {"rotor": TU_COLOURS[1], "exc": TU_COLOURS[2], "gov": TU_COLOURS[3],
              "conv": TU_COLOURS[4], "other": TU_COLOURS[8]}
    markers = ["o", "x", "s", "^"]
    cases = sorted(df["droop_case"].unique())
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots()
    handles = []
    for g in GROUPS:
        if not (df["group"] == g).any():
            continue
        handles.append(Line2D([], [], ls="none", marker="s", color=colour[g],
                              label=GROUP_LABEL[g]))
        for i, c in enumerate(cases):
            d = df[(df["group"] == g) & (df["droop_case"] == c)]
            if d.empty:
                continue
            mk = markers[i % len(markers)]
            style = ({"facecolors": "none", "edgecolors": colour[g]} if mk != "x"
                     else {"c": colour[g]})
            ax.scatter(d["re"], d["im"], s=12, marker=mk, linewidths=0.8, **style)
    for i, c in enumerate(cases):
        mk = markers[i % len(markers)]
        handles.append(Line2D([], [], ls="none", marker=mk, color="0.3",
                              mfc="none" if mk != "x" else None, label=c))
    sig_b = -KAPPA / T_STS
    ax.axvline(sig_b, color="0.3", lw=0.8, ls="--")
    ax.text(sig_b, 0.98, rf"$\sigma=-\kappa/T_{{STS}}$ ", transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=7)
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_yscale("symlog", linthresh=1.0)
    re_max = float(df["re"].max())
    ax.set_xlim(right=max(1e-2, 3 * re_max))
    ax.set_ylim(bottom=0, top=max(10.0, 2 * float(df["im"].max())))
    ax.set_xlabel(r"$\sigma = \mathrm{Re}\,\lambda$ / s$^{-1}$")
    ax.set_ylabel(r"$\omega = \mathrm{Im}\,\lambda$ / rad s$^{-1}$")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3)
    fig.savefig(path)
    plt.close(fig)


# ── summary ────────────────────────────────────────────────────────────────

def _mode_line(m: Optional[dict]) -> str:
    if not m:
        return "none"
    p = ", ".join(f"{c[2:]} {m[c]:.2f}" for c in P_COLS)
    return (f"tau = {m['tau_s']:.3g} s, group {m['group']}, {m['window']} / "
            f"{m['droop_case']}, mode {m['mode_idx']}, f = {m['f_hz']:.3g} Hz, "
            f"zeta = {m['zeta']:.3f}, dominant {m['dominant_state']} @ "
            f"{m['dominant_object']} ({m['dominant_class']}), in L_VQ = "
            f"{'yes' if m['in_L_VQ'] else 'no'}; p: {p}")


def summary_md(ev: Dict[str, dict], label: str = "") -> str:
    lim = tau_limit()
    lines = [f"# Modal reduction summary{(' -- ' + label) if label else ''}", "",
             f"Declared: kappa = {KAPPA:g}, p = {P_MATERIAL:.2f}, T_STS = {T_STS:g} s, "
             f"T_tap = {T_TAP:g} s => tau_max(L_VQ) = {lim:.2f} s; near-zero "
             f"|Re| < {NEAR_ZERO_RE:g}; unstable Re > {UNSTABLE_RE:g}.", ""]
    for case, r in ev.items():
        lines += [f"## Droop case `{case}`", "",
                  f"- modes per window: {r['modes_per_window']}",
                  f"- unstable per window: {r['unstable_per_window']}",
                  f"- near-zero rows: {r['near_zero']} (see near_zero_modes.md)",
                  f"- slowest in L_VQ: {_mode_line(r['slowest_lvq'])}",
                  f"- tau_exc = {r['tau_exc']:.3g} s ({_mode_line(r['slowest_exc'])}); "
                  f"tap term 5 s + 3 tau_exc = {r['tap_term']:.3g} s",
                  f"- 3 * max tau(L_VQ) = {r['lvq_term']:.3g} s",
                  f"- required T_STS = {r['required']:.3g} s; margin = {r['margin']:.3g} s",
                  f"- slowest overall: {_mode_line(r['slowest_overall'])}",
                  f"- CHECKPOINT R1: {'**TRIGGERED**' if r['r1_triggered'] else 'not triggered'}"
                  f" ({len(r['r1_slow_lvq'])} slow L_VQ mode(s), "
                  f"{len(r['r1_gov_vq'])} governor mode(s) with material V/Q participation)"]
        for m in r["r1_slow_lvq"]:
            lines.append(f"    - R1 slow L_VQ: {_mode_line(m)}")
        for m in r["r1_gov_vq"]:
            lines.append(f"    - R1 governor V/Q: {_mode_line(m)}")
        if r["r1_pending_near_zero"]:
            lines.append(f"- near-zero modes in L_VQ pending explanation: "
                         f"{len(r['r1_pending_near_zero'])}")
        if r["lvq_with_gov_share"]:
            lines.append(f"- info: {len(r['lvq_with_gov_share'])} L_VQ mode(s) with "
                         f"p_gov >= {P_MATERIAL:.2f}")
        if r["borderline_excluded"]:
            lines.append(f"- info: {len(r['borderline_excluded'])} mode(s) outside L_VQ "
                         f"whose SUMMED V/Q participation reaches {P_MATERIAL:.2f} "
                         f"(per-group test is the declared one)")
            for m in r["borderline_excluded"]:
                lines.append(f"    - {_mode_line(m)}")
        if r["csv_lvq_disagreements"]:
            lines.append(f"- WARNING: {r['csv_lvq_disagreements']} row(s) where the "
                         f"emitter's in_L_VQ differs from the recomputed one")
        lines.append("")
    return "\n".join(lines)


# ── synthetic data for the offline dry run ─────────────────────────────────

def make_synthetic() -> pd.DataFrame:
    """A dozen invented modes per (window, case). NOT model data."""
    base = [
        # re, im, group, state, object, class, p (rotor, exc, gov, conv, other)
        (-0.30, 6.0, "rotor", "speed", "G 04", "ElmSym", (0.70, 0.20, 0.05, 0.05, 0)),
        (-0.50, 8.4, "rotor", "phi", "G 07", "ElmSym", (0.80, 0.10, 0.05, 0.05, 0)),
        (-1.20, 0.0, "exc", "vr", "AVR 03", "ElmDsl:avr_IEEET1", (0.05, 0.90, 0, 0.05, 0)),
        (-0.40, 0.5, "exc", "efd", "AVR 06", "ElmDsl:avr_IEEET1", (0.20, 0.75, 0.05, 0, 0)),
        (-0.08, 0.0, "gov", "x1", "GOV 05", "ElmDsl:gov_IEEEG1", (0.02, 0.02, 0.95, 0.01, 0)),
        (-0.25, 0.3, "gov", "x2", "GOV 10", "ElmDsl:gov_IEEEG3", (0.12, 0.03, 0.80, 0.05, 0)),
        (-10.0, 0.0, "conv", "Qflt", "QVPRE_s3_01", "ElmDsl:QVPRE", (0, 0.02, 0, 0.98, 0)),
        (-1e4, 0.0, "conv", "Iq", "REGC_s1_02", "ElmDsl:REGC_C", (0, 0, 0, 1.0, 0)),
        (-30.0, 300.0, "conv", "pll", "REGC_s2_01", "ElmDsl:REGC_C", (0.05, 0, 0, 0.95, 0)),
        (0.0, 0.0, "rotor", "phi", "G 01", "ElmSym", (1.0, 0, 0, 0, 0)),
        (-2e-4, 0.0, "gov", "xint", "GOV 01", "ElmDsl:gov_IEEEG1", (0.01, 0, 0.99, 0, 0)),
        (-2.0, 0.0, "other", "xtap", "TAPCTRL_T3", "ElmDsl:TAPCTRL_qOFO", (0, 0, 0, 0.1, 0.9)),
    ]
    rows = []
    for w, scale in (("W1", 1.0), ("W2", 1.1)):
        for case in ("db_wide", "db_zero"):
            for i, (re, im, g, st, ob, cl, p) in enumerate(base):
                re_ = re * scale
                if case == "db_zero" and st == "Qflt":
                    re_ = -55.0 * scale        # droop on its slope
                wn = math.hypot(re_, im)
                rows.append({
                    "window": w, "droop_case": case, "mode_idx": i,
                    "re": re_, "im": im, "f_hz": abs(im) / (2 * math.pi),
                    "zeta": (-re_ / wn) if wn > 1e-9 else 1.0,
                    "tau_s": 1 / abs(re_) if re_ else math.inf,
                    "ts_s": 4 / abs(re_) if re_ else math.inf,
                    "group": g, "dominant_state": st, "dominant_object": ob,
                    "dominant_class": cl,
                    **{c: v for c, v in zip(P_COLS, p)},
                    "in_L_VQ": in_l_vq(g, dict(zip(GROUPS, p))),
                })
    return pd.DataFrame(rows, columns=list(COLUMNS))


# ── driver ─────────────────────────────────────────────────────────────────

def reduce(paths: Sequence[Path], out: Path, *, other_is_vq: bool = True,
           label: str = "", plot: bool = True) -> Dict[str, dict]:
    out.mkdir(parents=True, exist_ok=True)
    df = derive(load(paths), other_is_vq=other_is_vq)
    df[list(COLUMNS) + [c for c in OPTIONAL_COLUMNS if c in df.columns]].to_csv(
        out / "modal_modes.csv", index=False)
    (out / "modal_groups.tex").write_text(groups_tex(df), encoding="utf-8")
    for case, d in df.groupby("droop_case"):
        (out / f"modal_groups_{case}.tex").write_text(groups_tex(d), encoding="utf-8")
    (out / "near_zero_modes.md").write_text(near_zero_md(df), encoding="utf-8")
    ev = evaluate(df)
    (out / "modal_summary.md").write_text(summary_md(ev, label), encoding="utf-8")
    if plot:
        spectrum_pdf(df, out / "modal_spectrum.pdf")
    return ev


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="*", type=Path, help="modal_modes.csv files")
    ap.add_argument("--out", type=Path, help="bundle directory")
    ap.add_argument("--synthetic", type=Path, metavar="DIR",
                    help="write a synthetic modal_modes.csv to DIR and reduce it")
    ap.add_argument("--other-not-vq", action="store_true",
                    help="do NOT count the 'other' group as V/Q-relevant")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    if args.synthetic:
        args.synthetic.mkdir(parents=True, exist_ok=True)
        src = args.synthetic / "synthetic_modal_modes.csv"
        make_synthetic().to_csv(src, index=False)
        paths, out, label = [src], args.out or args.synthetic / "bundle", "SYNTHETIC"
    else:
        if not args.csv or not args.out:
            ap.error("give modal_modes.csv file(s) and --out, or --synthetic DIR")
        paths, out, label = args.csv, args.out, ""
    ev = reduce(paths, out, other_is_vq=not args.other_not_vq, label=label,
                plot=not args.no_plot)
    print((out / "modal_summary.md").read_text(encoding="utf-8"))
    print(f"[modal_reduce] bundle -> {out}")
    return 1 if any(r["r1_triggered"] for r in ev.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
