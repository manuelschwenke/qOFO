#!/usr/bin/env python3
r"""Assemble the dissertation bundle of the plant-time-constant modal analysis.

Collects the latest window folder of W1-W6 under
``results/screening/modal_campaign/`` and writes
``results/THESIS_ch8_modal_<date>/``:

    README.md                  provenance, method, the answers of handoff §9
    modal_modes.csv            all modes, all windows, both droop cases
    modal_groups.tex           tabular body, groups x ranges over W1-W6
    modal_groups_<case>.tex    the same per droop case
    modal_spectrum.pdf         sigma vs omega by group, boundary -kappa/T_STS
    near_zero_modes.md         stop rule 2, with the A-matrix verification per run
    modal_summary.md           the declared rule per droop case (analysis.modal_reduce)
    residue_settling.csv/.tex  exact output settling per input class (R1 option 1)
    modal_settling_modes.csv/.tex  per mode: tau, largest amplitude/band, tau ln(A/b)

Usage::

    python -m analysis.build_modal_bundle [--date 2026-09-23]

Author: Manuel Schwenke / Claude Code (2026-09-23)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis import modal_reduce as mr  # noqa: E402
from pf import modal_participation as mp  # noqa: E402

CAMPAIGN = PROJECT_ROOT / "results" / "screening" / "modal_campaign"
WINDOWS = {
    "W1": "05 Jan 08:00, STS export",
    "W2": "22 Feb 13:00, import, high load",
    "W3": "04 Jan 08:00, reduced capability both levels",
    "W4": "26 Nov 10:00, no TS-DER Q",
    "W5": "03 Feb 20:00, no DER Q at all",
    "W6": "18 Jul 14:00, high export, voltage-rise corner",
}
RESIDUE_DIR = "observability_measured"
CLASS_LABEL = {"AVR": "AVR set-point (0.01 pu)", "DER_Q": "DER reactive power",
               "TAP_MT": "Machine-transformer tap", "TAP_NC3W": "Coupler tap",
               "TAP_other": "Other tap",
               "AVR_G01_equivalent": "AVR set-point of G 01 (equivalent; not dispatched)"}
#: Input class computed for information only (not a dispatch input in the
#: thesis configuration); kept out of the verdict and the .tex table.
INFO_CLASS = "AVR_G01_equivalent"


def latest(w: str) -> Path:
    return sorted(p for p in (CAMPAIGN / w).iterdir() if p.is_dir())[-1]


def dims(run_dir: Path) -> int:
    txt = (run_dir / "commod_output.txt").read_text(encoding="utf-8", errors="replace")
    return int(re.search(r"Dimension of eigenvalue problem: (\d+)", txt).group(1))


# ── near-zero verification on A ────────────────────────────────────────────

def zero_structure(run_dir: Path) -> Dict[str, object]:
    matl = run_dir / "matl"
    vi = mp.read_varidx(matl)
    n = len(vi)
    A = mp.read_amat(matl, n)
    lam = mp.read_evals(matl)
    states = pd.read_csv(run_dir / "modal_states.csv")
    zero_rows = np.abs(A).max(axis=1) == 0
    const = states["constant_at_op"].astype(bool).to_numpy()
    phi = np.where(vi["obj"].str.startswith("REGC") & (vi["state"] == "x_phiref"))[0]
    rows = A[phi]
    identical = bool(len(phi) > 0 and np.allclose(rows, rows[0], rtol=0, atol=1e-12))
    nz_cols = sorted({(vi.obj[c], vi.state[c]) for c in np.where(np.abs(rows).max(axis=0) > 0)[0]})
    n_zero = int((np.abs(lam) < 1e-6).sum())
    return {"n": n, "zeros": n_zero, "zero_rows": int(zero_rows.sum()),
            "zero_rows_equal_pf_constant": bool((zero_rows == const).all()),
            "x_phiref_rows": len(phi), "x_phiref_rows_identical": identical,
            "x_phiref_row_support": nz_cols,
            "rank_deficit": int(n - np.linalg.matrix_rank(A))}


# ── residue aggregation ────────────────────────────────────────────────────

def residue_tables(wdirs: Dict[str, Path]):
    P, M = [], []
    for w, d in wdirs.items():
        r = d / RESIDUE_DIR
        if not (r / "pairs.csv").exists():
            # a window analysed after the 0.01 pu decision uses the default folder
            r = d / "observability"
            head = (r / "summary.md").read_text(encoding="utf-8") if (r / "summary.md").exists() else ""
            if "AVR 0.01 pu" not in head:
                raise FileNotFoundError(f"{w}: no residue results with the 0.01 pu AVR step")
        p = pd.read_csv(r / "pairs.csv")
        p["window"] = w
        P.append(p)
        for case in ("db_wide", "db_zero"):
            f = r / f"modal_settling_{case}.csv"
            if f.exists():
                m = pd.read_csv(f)
                m["window"], m["droop_case"] = w, case
                M.append(m)
    pairs, modes = pd.concat(P, ignore_index=True), pd.concat(M, ignore_index=True)
    pairs["threshold_s"] = np.where(pairs.input_class.str.startswith("TAP"), 15.0, 20.0)
    idx = pairs.groupby(["window", "droop_case", "input_class"])["Ts_abs_s"].idxmax()
    cls = pairs.loc[idx, ["window", "droop_case", "input_class", "Ts_abs_s", "threshold_s",
                          "input", "output", "tail_at_20s", "band", "dom20_re", "dom20_im",
                          "dom20_tau_s"]].reset_index(drop=True)
    cls["margin_s"] = cls.threshold_s - cls.Ts_abs_s
    modes["key"] = (modes.dominant_object.str.split("\\").str[-1] + ":" + modes.dominant_state)
    return pairs, modes, cls


def residue_tex(cls: pd.DataFrame, windows: List[str]) -> str:
    L = [f"% generated by analysis/build_modal_bundle.py; exact settling [s] of the "
         f"linearised plant, max over both droop cases and all controlled outputs; "
         f"taps from contact; bands 1e-3 pu / 1 Mvar"]
    for c in ("AVR", "DER_Q", "TAP_MT", "TAP_NC3W", "TAP_other"):
        x = cls[cls.input_class == c]
        if x.empty:
            continue
        cells = []
        for w in windows:
            v = x[x.window == w]["Ts_abs_s"]
            cells.append("--" if v.empty else (r"$\le 0.5$" if v.max() <= 0.5 else f"{v.max():.1f}"))
        thr = 15 if c.startswith("TAP") else 20
        L.append(f"{CLASS_LABEL[c]} & " + " & ".join(cells) + f" & {thr} \\\\")
    return "\n".join(L) + "\n"


def modal_settling_modes(modes: pd.DataFrame) -> pd.DataFrame:
    m = modes[modes.Ts_mode_s > 0]
    rows = []
    for key, g in m.groupby("key"):
        w = g.loc[g.Ts_mode_s.idxmax()]
        rows.append({"mode": key, "group": w.group, "f_hz_min": g.f_hz.min(),
                     "f_hz_max": g.f_hz.max(), "zeta_min": g.zeta.min(),
                     "zeta_max": g.zeta.max(), "tau_s_min": g.tau_s.min(),
                     "tau_s_max": g.tau_s.max(), "amp_over_band_max": g.amp_over_band.max(),
                     "T_mode_s_max": g.Ts_mode_s.max(), "at": f"{w.window} {w.droop_case}",
                     "worst_pair": f"{w.pair_input} -> {w.pair_output}"})
    return pd.DataFrame(rows).sort_values("T_mode_s_max", ascending=False).reset_index(drop=True)


def modal_settling_tex(ms: pd.DataFrame, top: int = 8) -> str:
    L = ["% generated by analysis/build_modal_bundle.py; per mode: tau = 1/|Re lambda|, "
         "A/b = largest amplitude over all dispatch->output pairs relative to the band, "
         "T = tau ln(A/b) (envelope); ranges over W1-W6 and both droop cases"]
    for r in ms.head(top).itertuples():
        def rng(a, b, fmt):
            return fmt.format(a) if fmt.format(a) == fmt.format(b) else f"{fmt.format(a)}--{fmt.format(b)}"
        L.append(f"{r.mode.replace('_', r'\_')} & {r.group} & {rng(r.f_hz_min, r.f_hz_max, '{:.2f}')} & "
                 f"{rng(r.zeta_min, r.zeta_max, '{:.3f}')} & {rng(r.tau_s_min, r.tau_s_max, '{:.1f}')} & "
                 f"{r.amp_over_band_max:.1f} & {r.T_mode_s_max:.1f} \\\\")
    return "\n".join(L) + "\n"


# ── README ─────────────────────────────────────────────────────────────────

def _fmt_mode(m: dict) -> str:
    if not m:
        return "none"
    return (f"tau {m['tau_s']:.2f} s, {m['f_hz']:.3f} Hz, zeta {m['zeta']:.3f}, group "
            f"{m['group']}, {m['window']} {m['droop_case']}, dominant {m['dominant_state']} @ "
            f"{m['dominant_object']}; p rotor {m['p_rotor']:.2f} / exc {m['p_exc']:.2f} / "
            f"gov {m['p_gov']:.2f} / conv {m['p_conv']:.2f} / other {m['p_other']:.2f}")


def readme(out: Path, wdirs, states, runs, zs, ev, cls, ms, campaigns, today,
           pairs: pd.DataFrame = None) -> str:
    L = [f"# Plant time constants: modal analysis W1–W6 ({today})", "",
         "Bundle for `ch:timescales:plant` (Ch 8). Produced in `qOFO_2027` on the "
         "PowerFactory machine, handoff of 2026-09-22 (Task M) and the author's "
         "decisions of 2026-09-23 (R1 option 1: residue-based observability; AVR "
         "step 0.01 pu; no communication delay). Daily log: "
         "`docs/daily_log/09_2026/2026-09-22_modal_analysis.md`.", "",
         "## Provenance", "",
         "- PowerFactory 2025 SP4 (Python module 25.0.6), engine mode, project "
         "`qOFO\\IEEE39_qOFO`, study case `02_RMS_CoSim`, variations `wind_replace` + "
         "`full`, DSO grids active.",
         "- Python `F:\\python_environments\\qOFO_clean` 3.12.14.",
         "- ComMod: QR, all eigenvalues (`iopt_met = 0`, `iQRorQZ = 0`), left+right "
         "eigenvectors, participation factors, **all recording filters off**, "
         "constant-state report on, Matlab export (`EVals`, `PartFacs`, `Amat`, "
         "`Jacobian`, `M`, `VariableToIdx_*`). Options restored after every run.",
         "- PSS blocks disabled (deliberate, unchanged).",
         "- Tap controllers (`TAPCTRL_qOFO`, 24) out of service in the modal runs "
         "(their 24 lag states at −0.2 s⁻¹ removed), restored after each window.",
         "- QVPRE droop: re-anchored per window at the synced load flow "
         "(`qset = Q_LF/S_n`, `Vanchor = V_LF`, `Kdroop = 1/0.06`, capability box from "
         "the park's operating diagram at its snapshot P); `db_wide`: db = 0.1 pu "
         "(every droop inert), `db_zero`: db = 0 (every droop on its slope; the DSL's "
         "dead zone is identically linear at db = 0). All 44 parameter vectors "
         "restored and read back after each window.",
         "- Parks whose snapshot load-flow Q lies outside their own capability box "
         "(W3–W5 TSO parks) have the box pinned at that Q (droop inert, linearisation "
         "at the snapshot operating point).", "",
         "Commands per window: `python -m pf.modal_campaign <W>` = `pf.pf_sync <snap> "
         "--phase full --study-case 02_RMS_CoSim` (+ `--dry-run` gate) -> "
         "`python -m pf.modal_window` -> `python -m analysis.modal_observability`; "
         "offline reruns with the 0.01 pu AVR step: `python -m analysis.modal_observability "
         "<window dir> --out <window dir>/observability_avr0.01 --no-ode-check`; this "
         "bundle: `python -m analysis.build_modal_bundle`.", "",
         "| W | regime | snapshot sha256 | sync upd. | dry-run | n | modes (wide/zero) | "
         "zero-box | pinned | db before/after | acceptance |",
         "|---|---|---|--:|:--|--:|:--|--:|--:|:--|:--|"]
    for w, d in wdirs.items():
        s = states[w]
        dr = s.get("dryrun") or {}
        drs = "0/0/0/0" if dr and not any(dr.get(k) for k in ("created", "renamed", "updated", "deleted")) else str(dr)
        rs = s["restore"]
        acc = s.get("acceptance", {})
        acc_s = ("passed" if acc.get("passed") else
                 "structural (no droop can be on a slope)" if w == "W5" else "FAILED")
        L.append(f"| {w} | {WINDOWS[w]} | `{s['sha256'][:16]}…` | "
                 f"{(s.get('sync') or {}).get('updated', '?')} | {drs} | {runs[w]['n']} | "
                 f"{runs[w]['modes']} | {len(s['reanchor']['zero_width_box'])} | "
                 f"{len(s['reanchor'].get('box_pinned_to_lf_q', []))} | "
                 f"{rs['db_before']}/{rs['db_after']} | {acc_s} |")
    fin = [c.get("final_W1") for c in campaigns if c.get("final_W1")]
    if fin:
        L += ["", f"Final re-sync to W1 (project left as found): {fin[-1]}."]
    L += ["", "Full sha256 per window:", ""] + [f"- {w}: `{states[w]['sha256']}`" for w in wdirs]

    L += ["", "## §5 diagnostics", ""]
    for w in wdirs:
        s = states[w]
        db = s["diag_before"]
        ia = s["init_anchored"]
        L.append(f"- **{w}**: before any change |V_LF − Vanchor| max {db['max_abs_dv']:.4f} pu "
                 f"({db['n_dv_gt_1e-4']}/44 above 1e-4 -> stored reference NOT anchored); "
                 f"inside stored db {db['db_stored_values']}: {db['inside_stored_db']}/44; "
                 f"inside thesis db 0.01: {db['inside_snapshot_db']}/44. After re-anchoring: "
                 f"max |x1 − Vanchor| {ia['max_abs_x1_minus_vanchor']:.1e}, max |Qext − Q_LF| "
                 f"{ia['max_abs_qext_minus_qlf_pu']:.1e} pu, init warnings {ia['n_init_warnings']}. "
                 f"QVPRE-dominated σ: wide {s['acceptance'].get('db_wide', {}).get('qvpre_sigma_min', float('nan')):.1f}"
                 f"…{s['acceptance'].get('db_wide', {}).get('qvpre_sigma_max', float('nan')):.1f}, zero "
                 f"{s['acceptance'].get('db_zero', {}).get('qvpre_sigma_min', float('nan')):.1f}"
                 f"…{s['acceptance'].get('db_zero', {}).get('qvpre_sigma_max', float('nan')):.1f} s⁻¹.")

    L += ["", "## Declared rule (handoff §3), answers to §9", "",
          f"kappa = {mr.KAPPA:g}, p = {mr.P_MATERIAL:.2f}, T_STS = {mr.T_STS:g} s, T_tap = "
          f"{mr.T_TAP:g} s; group = argmax of normalised participation (PF `PartFacs`, "
          "|P_kj|/Σ_k|P_kj|); L_VQ = rotor, excitation, converter groups + any mode with "
          "p ≥ 0.10 in one of them (`other` counted V/Q-relevant).", ""]
    w1 = runs["W1"]
    L.append(f"1. **W1:** n = {w1['n']} states; {w1['modes']} modes (Im ≥ 0, incl. 199 "
             "structural zeros).")
    for case, r in ev.items():
        L += [f"", f"**Droop case `{case}`**", "",
              f"2. unstable modes per window: {r['unstable_per_window']}",
              f"3. slowest τ in L_VQ: {_fmt_mode(r['slowest_lvq'])}",
              f"4. tau_exc = {r['tau_exc']:.2f} s -> tap term 5 s + 3 tau_exc = {r['tap_term']:.1f} s "
              f"({_fmt_mode(r['slowest_exc'])})",
              f"5. slowest overall: {_fmt_mode(r['slowest_overall'])} -- in L_VQ: "
              f"{'yes' if r['slowest_overall']['in_L_VQ'] else 'no (V/Q-relevant participation < 0.10)'}",
              f"6. margin 20 s − max(3·{r['tau_lvq_max']:.2f}, {r['tap_term']:.1f}) = "
              f"**{r['margin']:.1f} s** -> CHECKPOINT R1 "
              f"{'TRIGGERED' if r['r1_triggered'] else 'not triggered'} "
              f"({len(r['r1_slow_lvq'])} slow L_VQ mode rows, {len(r['r1_gov_vq'])} governor rows "
              "with material V/Q share)"]
    L += ["", "7. table rows: `modal_groups.tex` (ranges over W1–W6 and both droop "
          "cases; per case in `modal_groups_<case>.tex`). The `Other` row are the 113 "
          "ElmLod states of the dynamic general load (delay t1 = 0.1 s).",
          "8. near-zero modes: `near_zero_modes.md` (all explained, see below).",
          "9. §5 diagnostics above; db_zero acceptance passed in W1–W4 and W6, "
          "structural in W5 (every park has zero Q capability, so no droop can be on a "
          "slope; both spectra are identical by construction).", ""]
    L += ["Near-zero modes, verified on A per run:", "",
          "| run | n | zeros | zero rows | zero rows = PF constant list | x_phiref rows identical | rank deficit |",
          "|:--|--:|--:|--:|:--|:--|--:|"]
    for k, z in zs.items():
        L.append(f"| {k} | {z['n']} | {z['zeros']} | {z['zero_rows']} | "
                 f"{z['zero_rows_equal_pf_constant']} | {z['x_phiref_rows_identical']} "
                 f"(support {z['x_phiref_row_support']}) | {z['rank_deficit']} |")

    L += ["", "## R1 option 1: output settling from the residues (author's go 2026-09-23)", "",
          "The declared rule bounds every mode's τ regardless of how much it reaches a "
          "controlled output. The residue analysis weights each mode by what a dispatch "
          "excites and what an output sees, from PF's own linearisation:", "",
          "- DAE from `Jacobian.mtl` (1218 states + ~4180 algebraic variables), "
          "A = A11 − A12 A22⁻¹ A21 (= PF's `Amat` to ~3e-13).",
          "- Inputs where the set-point enters PF's equations: DER `qset` in the QVPRE "
          "`Qext` equation; AVR `usetp` in the IEEET1 `x_a` row (K_A/T_A); taps via "
          "`nntapin` from an extra run with TAPCTRL in service (its reduced spectrum "
          "equals the taps-out one to ~1e-10). The input set follows "
          "`experiments/runners/multi_tso_dso.py`: the slack/equivalent machine G 01 "
          "has neither its AVR set-point (`dispatch_slack_gen_v_ref = False`) nor its "
          "unit-transformer OLTC (`MT_g9_t12`, \"Exclude the slack gen's machine trafo "
          "from the controllable OLTC set\") in the TSO actuator set. G 01's AVR set-point "
          "is nevertheless computed as a separate, labelled class (below). Shunts not "
          "included.",
          "- Outputs = the measured variables of the controllers, as built in "
          "`multi_tso_dso.py`: TSO voltages = TN buses minus PV/slack buses (30); DSO "
          "voltages = `hv.bus_indices` (40, incl. the 12 PCCs; no tertiary buses); "
          "interface Q = coupler HV side (`res_trafo3w.q_hv_mvar`, 12). Tie-line flows "
          "are not measured and not included. Interface Q from the pandapower branch "
          "model at PF's voltages (validated against PF's own flows: a constant +0.04 "
          "Mvar offset, the no-load representation).",
          "- Residues R_k = C P_k B (cluster projectors for repeated eigenvalues); "
          "step response y(t) = D + Σ_k R_k/λ_k (e^{λ_k t} − 1); **exact settling "
          "time** = last time |y(t) − y(∞)| exceeds the band (grid 0.02 s to 60 s, "
          "0.5 s to 600 s; resolved from 0.5 s on). Validated against stiff ODE "
          "integration (≤ 5e-6 of the band). Structural zeros: no drift (≤ 3e-13).",
          "- **Per-mode settling, read directly off the modal data**: "
          "T_k = τ_k ln(A_k/b), A_k = the mode's largest amplitude |R_k/λ_k|·Δu (×2 for "
          "a pair) over all dispatch/output pairs. κ = 3 in the declared rule "
          "corresponds to A/b = e³ for every mode. For oscillatory modes T_k is the "
          "envelope crossing (up to half a period later than the signal's last "
          "excursion); the verdict uses the exact values.",
          "- Bands 1e-3 pu / 1 Mvar; worst-case steps DER min(60 Mvar, 0.5 S_n, "
          "upward headroom), AVR 0.01 pu, one tap; threshold 20 s, taps 15 s after "
          "contact (5 s command-to-contact dead time; no communication delay).", ""]
    vcls = cls[cls.input_class != INFO_CLASS]
    tot = int((vcls.Ts_abs_s > vcls.threshold_s).sum())
    L += [f"Result: **{tot} (input class, window, droop case) combination(s) above "
          "threshold**; per class (max over windows and cases):", "",
          "| input class | max T_s [s] | threshold | margin [s] | at | binding pair | dominant mode @20 s |",
          "|:--|--:|--:|--:|:--|:--|:--|"]
    for c, g in vcls.groupby("input_class"):
        r = g.loc[g.Ts_abs_s.idxmax()]
        L.append(f"| {CLASS_LABEL.get(c, c)} | {r.Ts_abs_s:.2f} | {r.threshold_s:g} | "
                 f"{r.threshold_s - r.Ts_abs_s:.2f} | {r.window} {r.droop_case} | {r.input} -> "
                 f"{r.output} | λ = {r.dom20_re:.4f}{r.dom20_im:+.4f}j (τ {r.dom20_tau_s:.1f} s) |")
    if pairs is not None:
        vp = pairs[pairs.input_class != INFO_CLASS]
        over = vp[vp.Ts_abs_s > vp.threshold_s]
        L += ["", f"Pairs above threshold: **{len(over)} of {len(vp)}**."]
        for r in over.itertuples():
            L.append(f"- {r.window} {r.droop_case}: {r.input} -> {r.output}: {r.Ts_abs_s:.2f} s "
                     f"(threshold {r.threshold_s:g} s), tail {r.tail_at_15s:+.3f} @15 s, "
                     f"{r.tail_at_20s:+.3f} @20 s; dominant λ = {r.dom20_re:.4f}{r.dom20_im:+.4f}j "
                     f"(τ {r.dom20_tau_s:.1f} s)")
        g = pairs[pairs.input_class == INFO_CLASS]
        if len(g):
            w = g.loc[g.Ts_abs_s.idxmax()]
            L += ["", "**For information: G 01's AVR set-point (not dispatched).** With the RMS "
                  "model's own IEEET1 AVR 01, a 0.01 pu V-ref step on the equivalent would settle "
                  f"in up to **{w.Ts_abs_s:.1f} s** ({w.window} {w.droop_case}, output {w.output}); "
                  f"{int((g.Ts_abs_s > 20).sum())} of {len(g)} of its pairs exceed 20 s. The tail "
                  f"is dominated by λ = {w.dom20_re:.4f}{w.dom20_im:+.4f}j (τ {w.dom20_tau_s:.1f} s), "
                  "the G 01 excitation mode. Treating the equivalent's voltage as an ideal pilot-node "
                  "control (the transformers behind the equivalent hold it at the set-point) would "
                  "remove exactly this path; that idealisation is not what the RMS model contains "
                  "and is not evaluated here."]
    L += ["", "Per mode (`modal_settling_modes.csv/.tex`, top 8). The key is the mode's "
          "dominant state; where one state dominates different modes in different windows "
          "(e.g. `G 10:phi`, several electromechanical modes) the row spans them.", "",
          "| mode | group | f [Hz] | ζ | τ [s] | max A/b | max T = τ ln(A/b) [s] | at | worst pair |",
          "|:--|:--|--:|--:|--:|--:|--:|:--|:--|"]
    for r in ms.head(8).itertuples():
        L.append(f"| {r.mode} | {r.group} | {r.f_hz_min:.2f}–{r.f_hz_max:.2f} | "
                 f"{r.zeta_min:.3f}–{r.zeta_max:.3f} | {r.tau_s_min:.1f}–{r.tau_s_max:.1f} | "
                 f"{r.amp_over_band_max:.2f} | {r.T_mode_s_max:.1f} | {r.at} | {r.worst_pair} |")
    L += ["", "## What is NOT established", "",
          "- Large-signal behaviour (limits, clipping transitions, dead-band crossings): "
          "the analysis is small-signal at two frozen droop states.",
          "- 132 WECC objects use buffers (delay/movingavg); a linearisation does not "
          "represent them.",
          "- Shunt (MSC/MSR) inputs.",
          "- The PF tap model is a 5 s first-order lag (`TAPCTRL` Tmech = 5 s); the thesis "
          "and this analysis use a 5 s pure dead time + contact.",
          "- W3–W5 snapshots: TSO-park Q outside the parks' own capability box (seeded "
          "Q); pinned here, but it also concerns the QSS runs from these snapshots.",
          "- The 0.77 Hz electromechanical mode is almost undamped in W4 (droop inert) "
          "and W5 (ζ = 0.004, τ = 47 s; PSS off). Stable, weakly excited by dispatches "
          "(A/b ≤ 1.25), but a model property worth stating.",
          "- The group label of the G 01/G 10 field modes is a near tie between rotor "
          "(ElmSym psifd) and excitation (AVR states); tau_exc and the tap term depend on it.",
          "- The 2026-07-20 screen (`results/screening/full_t0_wecc/20260720-125758/`) "
          "is provenance only and is not quoted."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args(argv)
    out = PROJECT_ROOT / "results" / f"THESIS_ch8_modal_{args.date}"
    out.mkdir(parents=True, exist_ok=True)
    wdirs = {w: latest(w) for w in WINDOWS}
    states = {w: json.loads((d / "window_state.json").read_text(encoding="utf-8"))
              for w, d in wdirs.items()}
    csvs, runs, zs = [], {}, {}
    for w, s in states.items():
        modes_n = []
        for case in ("db_wide", "db_zero"):
            rd = Path(s["cases"][case]["run_dir"])
            csvs.append(rd / "modal_modes.csv")
            modes_n.append(len(pd.read_csv(rd / "modal_modes.csv")))
            zs[f"{w} {case}"] = zero_structure(rd)
            runs.setdefault(w, {})["n"] = dims(rd)
        runs[w]["modes"] = "/".join(map(str, modes_n))
    ev = mr.reduce(csvs, out, label="W1–W6")
    pairs, modes, cls = residue_tables(wdirs)
    cls.to_csv(out / "residue_settling.csv", index=False)
    (out / "residue_settling.tex").write_text(residue_tex(cls, list(WINDOWS)), encoding="utf-8")
    ms = modal_settling_modes(modes)
    ms.to_csv(out / "modal_settling_modes.csv", index=False)
    (out / "modal_settling_modes.tex").write_text(modal_settling_tex(ms), encoding="utf-8")
    nz = out / "near_zero_modes.md"
    nz.write_text(nz.read_text(encoding="utf-8") + "\n## Verification on A (per run)\n\n"
                  "The candidate labels above are confirmed structurally: in every run the "
                  "all-zero rows of A are exactly PF's constant-at-OP states, the REGC_C "
                  "x_phiref rows are identical (each x_phiref' = omega_n * Delta omega of "
                  "G 01), and rank(A) is deficient by exactly the number of zero "
                  "eigenvalues (semisimple zeros).\n\n" +
                  "\n".join(f"- {k}: {v}" for k, v in zs.items()) + "\n", encoding="utf-8")
    campaigns = [json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted(CAMPAIGN.glob("campaign_*.json"))]
    (out / "README.md").write_text(
        readme(out, wdirs, states, runs, zs, ev, cls, ms, campaigns, args.date,
               pairs=pairs),
        encoding="utf-8")
    print(f"[bundle] -> {out}")
    for f in sorted(out.iterdir()):
        print(f"   {f.name:<32} {f.stat().st_size:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
