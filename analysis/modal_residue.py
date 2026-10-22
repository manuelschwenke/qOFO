r"""
analysis/modal_residue.py
=========================
Residue-weighted **output** settling from step-battery trajectories.

Why this exists
---------------
``pf/screening.py modal`` reports, per mode, the 2 %-band settling time

    T_s(mode) = 4 / |Re lambda|                                        (1)

which is the settling of a *unit-amplitude* excitation of that mode.  The
full-model table carries a 107.3 s mode; taken at face value (1) forbids
``T_DS = 20 s``, while the Gate-D battery measures the worst controlled
output settling at 13.2 s.  The two are not in conflict: (1) says nothing
about whether a mode is *visible* in the quantity the premise is about.

Mode ``l`` reaches output ``y_i`` under dispatch ``u_j`` with residue
``R_ijl``, contributing ``R_ijl e^{lambda_l t}``.  Replacing the unit
amplitude in (1) by the measured one gives the quantity actually needed:

    T_l(i,j) = ln(|A_ijl| / tol) / |Re lambda_l|,   T_l := 0 if |A| <= tol
                                                                       (2)

and the residue-weighted output settling is ``max_l T_l(i,j)``, i.e. the
instant after which no mode contributes more than the tolerance band.  A
poorly damped mode with negligible amplitude in the interface Q and the
constrained voltages yields ``T_l = 0`` and is correctly excluded.  This is
the "missing half of the screen" of
``00_daily_log/2026-07-22_ch12_rms_purpose_analysis.md`` (Tier 2).

Method
------
Amplitudes and eigenvalues are identified from the recorded trajectories by
the **matrix-pencil** method (Hua & Sarkar), preferred over classical Prony
for its noise robustness: the model order is selected from the singular-value
spectrum rather than assumed.  This is Path B of
``pf/probes/probe_modal_residue.py`` -- it needs no PF internals and measures
the modal content of the *algebraic* controlled outputs directly, which is
exactly what PowerFactory's state-space modal result cannot supply.

Path A (exact residues from an exported C matrix) is preferred where the
probe finds it available; this module then serves as its cross-check.

Accuracy (validated 2026-08-03 on synthetic sums of the five slowest modes of
``modal.md``, 40 s at 50 ms)
---------------------------------------------------------------------------
Eigenvalues and amplitudes are recovered to 4 significant figures at
negligible noise.  With additive noise at 1 % / 5 % of the settling band the
worst residue-weighted settling is under-reported by 0.50 s / 0.80 s, the
error growing with noise because a noisier fit slightly over-damps the
slowest visible mode.  **The bias is optimistic**, so a margin below ~1 s
must not be claimed from this screen alone -- the time-domain battery
remains the instrument of record, and this module is the coverage argument.
A mode whose amplitude is genuinely large but slow (the 35 s, Re = -0.1134
case) is retained and dominates correctly; only sub-band modes are zeroed.

Input
-----
Per-signal trajectories written by ``pf/screening.py steps
--save-trajectories`` into ``traj_<step-name>.csv`` (long format:
``signal,t,y``).  The Gate-D run of 2026-07-20 persisted summary statistics
only, so its modal content is unrecoverable and the battery must be re-run.

Usage
-----
    python -m analysis.modal_residue results/screening/<label>/<stamp>/ \
        --t-event 5.0 --modal results/screening/<label>/<stamp>/modal.md

Author: Manuel Schwenke / Claude Code (2026-08-03)
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

#: Settling tolerance per quantity, matching ``screening.settling_metrics``
#: so the residue-weighted number is comparable with the time-domain one.
TOL_VOLTAGE_PU = 1e-3
TOL_Q_MVAR = 1.0
TOL_DEFAULT = 1e-4

#: Singular values below ``SV_RATIO`` times the largest are noise, not modes.
SV_RATIO = 1e-3

#: Hard cap on identified model order.  The battery records 45 s at a 50 ms
#: effective stride; orders far above this fit measurement noise.
MAX_ORDER = 24


def tolerance_for(signal: str) -> float:
    """Absolute settling band for a monitored signal, by name."""
    if signal.startswith("u_") or signal.startswith("uGEN_") or signal.startswith("uDER_"):
        return TOL_VOLTAGE_PU
    if signal.startswith("qSTS_") or signal.startswith("qGEN_") or signal.startswith("qDER_"):
        return TOL_Q_MVAR
    return TOL_DEFAULT


@dataclass
class Mode:
    """One identified mode of one signal under one step."""
    lam: complex          # continuous-time eigenvalue [1/s]
    amp: complex          # complex amplitude (residue) in this signal's unit
    signal: str

    @property
    def magnitude(self) -> float:
        """Peak physical contribution: 2|A| for a conjugate pair, |A| real."""
        return 2.0 * abs(self.amp) if abs(self.lam.imag) > 1e-9 else abs(self.amp)

    @property
    def freq_hz(self) -> float:
        return abs(self.lam.imag) / (2.0 * math.pi)

    @property
    def zeta(self) -> float:
        wn = abs(self.lam)
        return float(-self.lam.real / wn) if wn > 1e-12 else 1.0

    @property
    def t_s_unit(self) -> float:
        """Classical per-mode settling 4/|Re lambda| -- what modal.md reports."""
        re = abs(self.lam.real)
        return 4.0 / re if re > 1e-9 else float("inf")

    def t_s_weighted(self, tol: float) -> float:
        """Residue-weighted settling, equation (2).

        Zero when the mode never leaves the band in this signal: that is the
        whole point -- an invisible mode imposes no bound on ``T_DS``.
        """
        mag = self.magnitude
        if mag <= tol:
            return 0.0
        re = abs(self.lam.real)
        if re < 1e-9:
            return float("inf")       # undamped and visible: a genuine failure
        return float(math.log(mag / tol) / re)


# =====================================================================
#  Matrix pencil
# =====================================================================

def matrix_pencil(y: NDArray[np.float64], dt: float,
                  pencil: Optional[int] = None,
                  sv_ratio: float = SV_RATIO,
                  max_order: int = MAX_ORDER,
                  ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Identify ``lambda`` and complex amplitudes of ``y(t) = sum A_l e^{lam_l t}``.

    ``y`` must already have its final value removed.  Returns
    ``(lambdas [1/s], amplitudes)``, both complex, conjugate pairs included
    once each (they are returned as identified, not folded).

    The pencil parameter defaults to ``N/3``, the Hua & Sarkar recommendation
    for variance; the model order comes from the singular-value spectrum, so
    no prior guess of the number of modes is required.
    """
    n = y.size
    if n < 8:
        return np.empty(0, complex), np.empty(0, complex)
    L = pencil if pencil is not None else max(2, n // 3)
    L = min(L, n - 2)

    # Hankel matrix, (n-L) x (L+1).
    rows = n - L
    Y = np.empty((rows, L + 1), float)
    for i in range(rows):
        Y[i, :] = y[i:i + L + 1]

    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    if s.size == 0 or s[0] <= 0.0:
        return np.empty(0, complex), np.empty(0, complex)
    order = int(np.sum(s > sv_ratio * s[0]))
    order = max(1, min(order, max_order, L - 1))

    V = Vh[:order, :].conj().T          # (L+1) x order
    V1, V2 = V[:-1, :], V[1:, :]        # drop last / first row
    A = np.linalg.pinv(V1) @ V2
    z = np.linalg.eigvals(A)

    # Discrete -> continuous.  Discard the numerically dead and the growing:
    # a z on or outside the unit circle in an open-loop step response of a
    # stable plant is a fit artefact, and keeping it would manufacture an
    # infinite settling time.
    keep = (np.abs(z) > 1e-12) & (np.abs(z) < 1.0 - 1e-9)
    z = z[keep]
    if z.size == 0:
        return np.empty(0, complex), np.empty(0, complex)
    lam = np.log(z) / dt

    # Amplitudes by least squares on the Vandermonde system.
    t_idx = np.arange(n)
    Vand = z[None, :] ** t_idx[:, None]
    amp, *_ = np.linalg.lstsq(Vand, y.astype(complex), rcond=None)
    return lam, amp


# =====================================================================
#  Trajectory I/O
# =====================================================================

def load_trajectories(path: Path) -> Dict[str, Tuple[NDArray, NDArray]]:
    """Read a long-format ``traj_*.csv`` into ``{signal: (t, y)}``."""
    buckets: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                buckets[row["signal"]].append((float(row["t"]), float(row["y"])))
            except (KeyError, ValueError):
                continue
    out: Dict[str, Tuple[NDArray, NDArray]] = {}
    for sig, pairs in buckets.items():
        pairs.sort()
        t = np.array([p[0] for p in pairs], float)
        y = np.array([p[1] for p in pairs], float)
        if t.size >= 8:
            out[sig] = (t, y)
    return out


def parse_modal_table(path: Path) -> List[Tuple[float, float, float, float]]:
    """Read ``modal.md`` -> ``[(T_s, Re, Im, zeta)]`` for mode labelling."""
    modes: List[Tuple[float, float, float, float]] = []
    if not path.exists():
        return modes
    row = re.compile(r"^\|\s*([\d.]+|inf)\s*\|\s*(-?[\d.eE+-]+)\s*\|"
                     r"\s*(-?[\d.eE+-]+)\s*\|\s*(-?[\d.eE+-]+)\s*\|"
                     r"\s*(-?[\d.eE+-]+)\s*\|")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = row.match(line.strip())
        if not m:
            continue
        ts = float("inf") if m.group(1) == "inf" else float(m.group(1))
        modes.append((ts, float(m.group(2)), float(m.group(3)), float(m.group(5))))
    return modes


#: Controlled-output variables in a ComRes export, mapped to the label prefix
#: ``pf/screening.py`` uses, so both sources feed the same analysis.
COMRES_VARS: Dict[str, str] = {
    "m:u": "u_",              # bus voltage         [pu]
    "m:Q:bushv": "qSTS_",     # coupler interface Q [Mvar]
    "s:xspeed": "spd_",       # machine speed       [pu]  (diagnostic)
}


def load_comres(path: Path, t0: float = 0.0, t1: float = float("inf"),
                want: Tuple[str, ...] = ("m:u", "m:Q:bushv"),
                ) -> Dict[str, Tuple[NDArray, NDArray]]:
    r"""Read a PowerFactory ``ComRes`` export into ``{signal: (t, y)}``.

    Format, established 2026-08-03 on
    ``results/rms_phase6_replay/0301_*/csv/rms_comres_full.csv``: semicolon
    separated, **decimal comma** (German locale), two header rows -- row 1 the
    object paths (``Grid\TN_bus18.ElmTerm``), row 2 the variables
    (``"m:u in p.u."``) -- and column 0 the time axis ``b:tnow in s`` at a
    10 ms step.

    Only the requested variables and the ``[t0, t1]`` window are parsed: the
    files are ~130 MB and usually live on a network share, and a modal fit
    needs one inter-dispatch window, not the whole run.

    This is what lets the **existing** N-1 corpus answer the observability
    question with no PowerFactory seat: those runs already record every
    controlled output at full RMS resolution.
    """
    with path.open(encoding="utf-8", errors="replace") as fh:
        objs = next(fh).rstrip("\n").split(";")
        vars_ = next(fh).rstrip("\n").split(";")
        keep: List[Tuple[int, str]] = []
        for i, (o, v) in enumerate(zip(objs, vars_)):
            if i == 0:
                continue
            var = v.strip().strip('"').split(" in ")[0].strip()
            if var not in want:
                continue
            name = o.split("\\")[-1].split(".")[0].strip()
            keep.append((i, COMRES_VARS.get(var, "") + name))
        if not keep:
            return {}
        times: List[float] = []
        cols: Dict[str, List[float]] = {lab: [] for _i, lab in keep}
        max_i = keep[-1][0]
        for line in fh:
            parts = line.rstrip("\n").split(";")
            if len(parts) <= max_i:
                continue
            try:
                t = float(parts[0].replace(",", "."))
            except ValueError:
                continue
            if t < t0:
                continue
            if t > t1:
                break
            times.append(t)
            for i, lab in keep:
                try:
                    cols[lab].append(float(parts[i].replace(",", ".")))
                except ValueError:
                    cols[lab].append(float("nan"))
    t_arr = np.asarray(times, float)
    out: Dict[str, Tuple[NDArray, NDArray]] = {}
    for lab, vals in cols.items():
        y = np.asarray(vals, float)
        if y.size == t_arr.size and y.size >= 8 and np.all(np.isfinite(y)):
            out[lab] = (t_arr, y)
    return out


def match_mode(lam: complex,
               table: Sequence[Tuple[float, float, float, float]],
               tol_hz: float = 0.05, tol_re: float = 0.15) -> Optional[int]:
    """Index of the ``modal.md`` row this fitted eigenvalue corresponds to."""
    best, best_d = None, float("inf")
    for i, (_ts, re, im, _z) in enumerate(table):
        d_hz = abs(abs(lam.imag) - abs(im)) / (2.0 * math.pi)
        d_re = abs(lam.real - re)
        if d_hz <= tol_hz and d_re <= tol_re and (d_hz + d_re) < best_d:
            best, best_d = i, d_hz + d_re
    return best


# =====================================================================
#  Analysis
# =====================================================================

def analyse_step(traj: Dict[str, Tuple[NDArray, NDArray]], t_event: float,
                 controlled_only: bool = True) -> Tuple[List[Mode], Dict[str, float]]:
    """Fit every signal of one step; return modes and per-signal settling."""
    modes: List[Mode] = []
    settling: Dict[str, float] = {}
    for sig, (t, y) in sorted(traj.items()):
        if controlled_only and not (sig.startswith("u_") or sig.startswith("qSTS_")):
            continue
        post = t >= t_event
        if post.sum() < 8:
            continue
        tp, yp = t[post], y[post]
        dt = float(np.median(np.diff(tp)))
        if dt <= 0:
            continue
        # Final value: mean of the last 0.5 s, as settling_metrics uses.
        tail = tp >= tp[-1] - 0.5
        y_inf = float(np.mean(yp[tail]))
        lam, amp = matrix_pencil(yp - y_inf, dt)
        tol = tolerance_for(sig)
        worst = 0.0
        for l_, a_ in zip(lam, amp):
            if l_.imag < -1e-9:          # keep one of each conjugate pair
                continue
            mode = Mode(lam=l_, amp=a_, signal=sig)
            modes.append(mode)
            worst = max(worst, mode.t_s_weighted(tol))
        settling[sig] = worst
    return modes, settling


def report(run_dir: Path, t_event: float, modal_md: Optional[Path],
           top: int = 12) -> int:
    traj_files = sorted(run_dir.glob("traj_*.csv"))
    if not traj_files:
        print(f"[residue] no traj_*.csv in {run_dir}\n"
              f"[residue] the battery must be re-run with --save-trajectories;\n"
              f"[residue] the 2026-07-20 Gate-D run persisted summary rows only.",
              file=sys.stderr)
        return 2
    table = parse_modal_table(modal_md) if modal_md else []
    lines = ["# Residue-weighted output settling", "",
             "Per-mode settling weighted by the amplitude the mode actually "
             "reaches in each controlled output, equation (2) of "
             "`analysis/modal_residue.py`. `T_s unit` is the classical "
             "`4/|Re lambda|` that `modal.md` reports; `T_s weighted` is the "
             "same mode measured in the output the premise is about. A mode "
             "whose amplitude never leaves the settling band scores 0 and "
             "imposes no bound on `T_DS`.", ""]
    overall = 0.0
    overall_where = ""
    for tf in traj_files:
        step_name = tf.stem[len("traj_"):]
        traj = load_trajectories(tf)
        modes, settling = analyse_step(traj, t_event)
        if not settling:
            continue
        worst_sig = max(settling, key=lambda s: settling[s])
        worst_val = settling[worst_sig]
        if worst_val > overall:
            overall, overall_where = worst_val, f"{step_name} / {worst_sig}"
        lines += [f"## {step_name}", "",
                  f"Worst controlled output: **{worst_sig}**, "
                  f"residue-weighted settling **{worst_val:.2f} s**", "",
                  "| signal | f [Hz] | zeta | Re lambda | \\|A\\| | "
                  "T_s unit [s] | T_s weighted [s] | modal.md row |",
                  "|:--|--:|--:|--:|--:|--:|--:|--:|"]
        ranked = sorted(modes, key=lambda m: -m.t_s_weighted(tolerance_for(m.signal)))
        for m in ranked[:top]:
            tol = tolerance_for(m.signal)
            row = match_mode(m.lam, table)
            lines.append(
                f"| {m.signal} | {m.freq_hz:.3f} | {m.zeta:.3f} | "
                f"{m.lam.real:.4f} | {m.magnitude:.3e} | {m.t_s_unit:.2f} | "
                f"{m.t_s_weighted(tol):.2f} | "
                f"{row if row is not None else '--'} |")
        lines.append("")
    lines += ["## Screen result", "",
              f"Worst residue-weighted controlled-output settling over the "
              f"battery: **{overall:.2f} s** ({overall_where}).", "",
              "Compare against the time-domain `T_s ctrl` of `steps.md`: the "
              "screen is only usable envelope-wide if the two agree at the "
              "points where both exist."]
    out = run_dir / "residue.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[residue] worst residue-weighted settling {overall:.2f} s "
          f"({overall_where}) -> {out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Residue-weighted output settling from step trajectories.")
    ap.add_argument("run_dir", type=Path,
                    help="results/screening/<label>/<stamp>/")
    ap.add_argument("--t-event", type=float, default=5.0)
    ap.add_argument("--modal", type=Path, default=None,
                    help="modal.md to label fitted modes against")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args(argv)
    modal = a.modal
    if modal is None and (a.run_dir / "modal.md").exists():
        modal = a.run_dir / "modal.md"
    return report(a.run_dir, a.t_event, modal, a.top)


if __name__ == "__main__":
    sys.exit(main())
