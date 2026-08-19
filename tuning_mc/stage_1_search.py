"""
tuning_mc/stage_1_search.py
===========================
Stage 1 — refine the Stage-0 design point by searching *the design rule's
inputs*, not the weights it emits.

Why the knobs and not the weights
---------------------------------
Stage 0 emits dozens of numbers (per class, per area, continuous and integer).
Searching those directly would re-introduce everything the reparameterisation
was built to avoid: the exact scaling redundancy, per-column coordinates no
affordable ensemble can identify, and steps with no engineering meaning.  The
rule's *inputs* are six numbers, each of which has units and a defensible
interpretation, and the rule regenerates the whole weight set from them:

======================  ====================================================
knob                    what it moves
======================  ====================================================
``lambda_tso``          TSO loop gain: target ``lambda_max(M)`` over the
                        continuous columns.  ~0.9 well damped, 2.0 the bound.
``lambda_dso``          same, DSO layer.
``tau``                 DER vs PCC allocation.  Gauge-fixed to geometric
                        mean 1, so it rotates the two class weights against
                        each other while ``kappa`` holds the loop gain --
                        (lambda, tau) spans every (g_w_der, g_w_pcc) pair.
``engage_tso_pu``       TSO tap commit threshold [pu], under a systematic
                        offset across the zone.
``engage_dso_pu``       same, DSO taps.
``dso_g_v_ratio``       DSO objective trade-off, as a multiple of the
                        baseline ``dso_g_v``.  Applied directly: the Stage-0
                        rules do not set the objective weights.
======================  ====================================================

Search
------
**Phase A — one-at-a-time probe.**  Each knob at x{1/4, 1/2, 2, 4} with the
rest held at the design point.  This is the step both BO campaigns lacked: it
measures, before any optimisation, which directions carry signal at all.  On
the finished Thevenin study three of five coordinates turned out to carry none
(``tso_lambda`` rho=+0.046 p=0.58, ``shunt_int_gain`` rho=+0.064 p=0.44), and
they consumed budget anyway.  A knob whose four probes move ``f_ts`` by less
than ``--dead-threshold`` is dropped from Phase B and reported as such.

**Phase B — compass search with a filter.**  From the incumbent, poll ``x +/-
delta`` in every live direction (complete polling: the whole batch is
evaluated, which turns idle workers into information), accept the best
filter-admissible improvement, halve ``delta`` when a poll fails.  Steps are
in log10 space because every knob is a scale parameter.  Deterministic, with
the usual convergence argument for generalised pattern search, and every
accepted move is reportable as "we moved X by a factor Y and gained Z".

Acceptance uses :mod:`tuning_mc.metrics`: hard constraints are an extreme
barrier, while TS voltage and interface-Q form a two-criterion filter -- the
encoding of "Q may be violated, but the least violation is best".

Parallelism
-----------
One subprocess per *candidate*, each evaluating its scenarios serially; this is
the pattern validated for this simulator (per-scenario joblib has been seen to
interfere with pandapower's solver setup).  Throughput on this class of machine
was measured at 2.14x for 6 workers, 2.02x for 8 and regressing past 10 -- the
bottleneck is memory bandwidth on the sparse Newton power flow, not cores -- so
``--workers`` defaults to 6 and BLAS threads are pinned to 1 in every child.

Usage::

    python -m tuning_mc.stage_1_search --phase a --workers 6
    python -m tuning_mc.stage_1_search --phase b --workers 6
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.config import apply_dso_v_relief

DEFAULT_BASELINE = (_REPO_ROOT / "tuning" / "scripts" / "configs"
                    / "baseline_ieee39_thevenin.yaml")
DEFAULT_OUT = _REPO_ROOT / "results" / "tuning_mc" / "stage1"

#: Design point: Stage 0's analytic values.  Every search coordinate is a
#: multiplicative offset from here, so the design point is the origin and is
#: representable by construction.
X0: dict[str, float] = {
    "lambda_tso": 0.9,
    "lambda_dso": 0.9,
    "tau": 1.0,
    "engage_tso_pu": 0.015,
    "engage_dso_pu": 0.025,
    "dso_g_v_ratio": 1.0,
}

#: Hard bounds.  ``lambda`` is bounded by OFO stability (<2); the engage
#: thresholds by the plant corridor; the rest generously.
BOUNDS: dict[str, tuple[float, float]] = {
    "lambda_tso": (0.05, 1.90),
    "lambda_dso": (0.05, 1.90),
    "tau": (1.0 / 64.0, 64.0),
    "engage_tso_pu": (0.002, 0.08),
    "engage_dso_pu": (0.002, 0.08),
    "dso_g_v_ratio": (1e-3, 1e3),
    # Per-zone TSO loop gain -- Phase 2 of the 0815 campaign.  Deliberately
    # NOT in X0: this is a gated hypothesis, not a seventh search coordinate.
    # Absent from a knob dict, nothing changes and the global path is used;
    # present, Stage 0 designs that zone at its own lambda and the result is
    # applied through ``zone_g_w_class`` (see :func:`build_config`).  Promote
    # into X0 only if Phase 2 pays.
    #
    # Upper bound 1.90, the same as the global coordinate, and NOT the ~4.7 the
    # analytic per-zone contraction says zone 1's own ceiling would permit.
    # ``controller/gw_precondition.py:458`` raises for any
    # ``lambda_target >= 2`` -- a module-level invariant written for the global
    # coordinate -- and that module may not be edited, since changing it would
    # alter the meaning of every existing study.  So the per-zone hypothesis is
    # testable over 0.05-1.90 only.  That is still a 38x span, and if zone 1
    # improves all the way to the bound then the *guard*, not the plant, is what
    # limits it -- which is a reportable result and a well-posed follow-up, not
    # a reason to bypass a stability check at 02:00.
    "lambda_tso_z1": (0.05, 1.90),
    "lambda_tso_z2": (0.05, 1.90),
    "lambda_tso_z3": (0.05, 1.90),
    # Per-DSO voltage authority -- same gating as the per-zone lambdas above:
    # in BOUNDS so it is addressable, deliberately NOT in X0 so nothing changes
    # unless it is asked for (``--search-dso-v-authority``).  Absent from a knob
    # dict, DSO_V_RELIEF_FACTORS is used; present, it overrides that for every
    # area in DSO_V_RELIEF_AREAS.
    #
    # ONE shared coordinate, not one per area.  Phase B is a compass search:
    # every live direction costs two evaluations (+/- delta) on every poll, so a
    # per-area split doubles the marginal cost to distinguish two areas that
    # both measured a factor around 20.  If they need to differ, the ratio
    # between them is better derived from the reach x |Q_PCC| predictor
    # (daily log 2026-08-18 section 8) than searched.
    #
    # Lower bound 1.0 = "no relief", so the incumbent can always walk back to
    # the unrelieved plant; upper 100 is ~5x the measured operating point and
    # well inside the DER's remaining Q-tracking dominance (at x20 the voltage
    # gradient is still ~18:1 below the Q-tracking gradient, so the DSO stays a
    # Q tracker that also shapes voltage; at x100 that margin is ~3.6:1 and the
    # role would start to inverert -- the bound is where the coordinate stops
    # meaning what it says).
    "dso_v_authority": (1.0, 100.0),
}

#: Areas the ``dso_v_authority`` coordinate acts on -- the spread-limited ones.
#: Measured 2026-08-18: internal spread 0.117 / 0.147 p.u. for DSO_2 / DSO_4
#: against 0.015 / 0.037 for DSO_1 / DSO_3, and giving the latter two the factor
#: bought ~0.001 p.u. of V_max while costing DSO_3 +53 % interface-Q RMSE.
DSO_V_RELIEF_AREAS: tuple[str, ...] = ("DSO_2", "DSO_4")

#: The knobs above that are per-zone.  Their presence in a knob dict switches
#: the design on to the per-area path.
PER_ZONE_KNOBS = ("lambda_tso_z1", "lambda_tso_z2", "lambda_tso_z3")

#: Actuator classes the loop gain actually generates.  Measured 2026-08-14:
#: lambda scales the *continuous* block only -- ``g_w_tso_oltc`` was constant in
#: every row of both lambda scans -- so the per-zone override must cover exactly
#: these and leave the tap price on the global scalar.  Overriding more would
#: confound "zone 1 at a higher gain" with "zone 1 at a different tap price".
PER_ZONE_CLASSES = ("der", "pcc")

#: Which Stage-0 ``config_block`` fields are applied to the config.  ``g_w_gen``
#: is deliberately excluded: it is part of the pinned gauge, and applying a
#: designed value would move the numeraire the whole comparison rests on.
APPLIED_FIELDS = ("g_w_der", "g_w_pcc", "g_w_dso_der",
                  "g_w_tso_oltc", "g_w_dso_oltc")


# ---------------------------------------------------------------------------
# Knob -> weights -> config
# ---------------------------------------------------------------------------

def load_limits(path: Path | None):
    """Constraint limits, explicit or the package defaults."""
    from tuning.objectives_v2 import ConstraintLimits
    if path is None:
        return ConstraintLimits()
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    known = {f.name for f in dataclasses.fields(ConstraintLimits)}
    unknown = set(data) - known
    if unknown:
        raise SystemExit(f"[stage1] unknown limit fields: {sorted(unknown)}")
    return ConstraintLimits(**data)


def knob_key(knobs: dict[str, float]) -> str:
    blob = json.dumps({k: round(float(v), 10) for k, v in sorted(knobs.items())},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def bank_fingerprint() -> str:
    """Short hash of the scenario-bank source, stamped on every evaluation.

    The evaluation cache is keyed on ``(scenario_set, knob hash)``, which is not
    enough: ``--scenario-set confirm`` names a *function*, and the windows that
    function returns are edited between campaigns.  Measured 2026-08-16 --
    extending the confirmation bank from six windows to nine left eight cached
    ``confirm_*.json`` results on disk whose knob hashes still matched, so a
    re-run would have silently reported six-window numbers as nine-window ones.
    ``_design_is_current`` cannot catch this: it compares *weights*, and the
    weights are correct; it is the ensemble that changed.

    Same remedy as :func:`stage0_fingerprint`: stamp the source, and treat a
    mismatch as stale rather than as a cache hit.
    """
    src = (Path(__file__).with_name("scenarios_mc_v2.py")).read_bytes()
    return hashlib.sha256(src).hexdigest()[:12]


def scoring_fingerprint(ds_criterion: str, limits=None) -> str:
    """Short hash of everything that decides what a stored score *means*.

    Third member of the same family as :func:`bank_fingerprint` and
    :func:`stage0_fingerprint`, and it exists for the same measured reason.
    The evaluation cache is keyed on ``(scenario_set, knob hash)`` and validated
    against the bank and the design rule -- neither of which notices a change to
    the *objective*.  Measured 2026-08-18: adding the ``guard`` DS criterion
    touched only ``tuning_mc/metrics.py`` and ``tuning/metrics.py``, so every
    cached tier-1 result still matched on bank and weights and would have been
    replayed with its old ``f_ds`` silently mixed into a new front.

    Hashes the two metric sources *and* the criterion selection, so a re-run
    under a different objective invalidates rather than half-reuses. Broad on
    purpose: an unrelated edit to those files also invalidates, which costs a
    re-run and never costs correctness.

    **The constraint limits are in here too**, measured 2026-08-18 the hard way:
    a tier-1 run launched without ``--limits`` silently used
    ``ConstraintLimits()`` defaults (``rho_emp_p95 = 1.0`` against the tier-1
    file's 1.5), so every candidate came back ``feasible=False`` on g3 at a
    measured rho of 1.4357.  ``filter_accepts`` rejects infeasible candidates
    outright, so the filter would have stayed empty for the whole campaign.
    Worse, limits sat in *no* fingerprint: re-launching with the right file
    would have replayed the cached rows and kept their stale ``feasible``.
    ``_evaluate`` already stamps ``limits`` into the payload with the note
    "a silently-defaulted limit set changes what 'feasible' means without
    changing anything visible in the result" -- this makes the cache honour it.
    """
    payload = [
        (Path(__file__).with_name("metrics.py")).read_bytes(),
        (Path(__file__).parent.parent / "tuning" / "metrics.py").read_bytes(),
        ds_criterion.encode(),
    ]
    if limits is not None:
        payload.append(
            json.dumps(dataclasses.asdict(limits), sort_keys=True).encode())
    return hashlib.sha256(b"".join(payload)).hexdigest()[:12]


def stage0_fingerprint() -> str:
    """Short hash of the Stage-0 source, stamped on every design it produces.

    The design cache is keyed on the *knobs*, which is not enough: Stage 0 is
    under active development, so the same knobs can emit different weights on
    different days.  Measured 2026-08-14 -- a design cached at 13:30 and reused
    at 15:20 differed from a fresh one by 1.8-6 % on every applied field, which
    put one point of a lambda-calibration curve on a different rule from the
    other six with nothing in the output to say so.  Stamping the source lets
    the cache detect that instead of hiding it.
    """
    src = (Path(__file__).with_name("stage_0_preconditioning.py")).read_bytes()
    return hashlib.sha256(src).hexdigest()[:12]


def _zone_lambda_spec(knobs: dict[str, float]) -> str | None:
    """``--lambda-tso-zone`` string, or ``None`` when no per-zone knob is set."""
    present = {k: knobs[k] for k in PER_ZONE_KNOBS if k in knobs}
    if not present:
        return None
    base = float(knobs["lambda_tso"])
    # Every zone is listed explicitly, including the ones left at the global
    # value: the control row of the Phase-2 scan is "all three zones at the
    # Phase-1 lambda, applied through the per-area path", and it must go down
    # the same code path as the rows it is the control for.
    return ",".join(
        f"{z}={float(present.get(f'lambda_tso_z{z}', base))!r}"
        for z in (1, 2, 3))


def design_payload(knobs: dict[str, float], *, baseline: Path,
                   design_scenario: str, workdir: Path) -> dict[str, Any]:
    """Run Stage 0 with these knobs and return its whole JSON payload.

    Stage 0 is invoked as a subprocess against its CLI + JSON contract rather
    than by importing its internals, so this keeps working across refactors of
    that module (it is under active development in parallel).

    A cached design is reused only when it carries the current Stage-0
    fingerprint; otherwise it is regenerated.  Regeneration is ~60 s and the
    result is usually identical (most edits to Stage 0 are reporting, not rule),
    so this is cheap insurance rather than a recompute-everything switch.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    fp = stage0_fingerprint()
    out = workdir / f"stage0_{knob_key(knobs)}.json"
    if out.exists():
        try:
            cached = json.loads(out.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            cached = {}
        if cached.get("_stage0_fingerprint") != fp:
            # Keep the superseded design rather than overwrite it: it is the
            # only record of what an already-cached evaluation actually ran.
            out.replace(out.with_suffix(f".json.fp_{cached.get('_stage0_fingerprint', 'unstamped')}"))
    if not out.exists():
        cmd = [
            sys.executable, "-m", "tuning_mc.stage_0_preconditioning",
            "--baseline", str(baseline),
            "--scenario", design_scenario,
            "--lambda-tso", repr(float(knobs["lambda_tso"])),
            "--lambda-dso", repr(float(knobs["lambda_dso"])),
            "--tau", repr(float(knobs["tau"])),
            "--engage-tso-pu", repr(float(knobs["engage_tso_pu"])),
            "--engage-dso-pu", repr(float(knobs["engage_dso_pu"])),
            "--out", str(out),
        ]
        zone_spec = _zone_lambda_spec(knobs)
        if zone_spec is not None:
            # --per-area is required, not cosmetic: the per-zone design is only
            # expressible through the per-area block, and Stage 0 populates
            # ``per_area`` only when asked.
            cmd += ["--lambda-tso-zone", zone_spec, "--per-area"]
        env = dict(os.environ)
        env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "PYTHONPATH": str(_REPO_ROOT)})
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"stage_0 failed (rc={proc.returncode}):\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    payload = json.loads(out.read_text(encoding="utf-8-sig"))
    if payload.get("_stage0_fingerprint") != fp:
        payload["_stage0_fingerprint"] = fp
        # Atomic, and best-effort.  Two workers can hold the same design: the
        # cache is keyed on the knobs, so the same candidate appearing in two
        # concurrently-running phases (a Tier-1 confirmation and a Tier-2 audit,
        # say) has both of them re-stamping one file.  Measured 2026-08-15 --
        # a plain write_text raced and one candidate died with
        # ``PermissionError: [Errno 13]`` on the SMB share, losing a 2.5-hour
        # evaluation slot for a bookkeeping write.
        #
        # The stamp is only a cache annotation; ``payload`` in memory is already
        # correct, so failing to persist it costs at most a redundant Stage-0
        # regeneration later.  Never let it kill the evaluation.
        tmp = out.with_suffix(f".json.tmp{os.getpid()}")
        try:
            tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            os.replace(tmp, out)
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink()
    block = payload.get("config_block", {})
    missing = [f for f in APPLIED_FIELDS if f not in block]
    if missing:
        raise RuntimeError(f"stage_0 config_block is missing {missing}; "
                           f"got {sorted(block)}")
    return payload


def design_weights(knobs: dict[str, float], *, baseline: Path,
                   design_scenario: str, workdir: Path) -> dict[str, float]:
    """The ``config_block`` scalars Stage 0 designs for these knobs."""
    block = design_payload(knobs, baseline=baseline,
                           design_scenario=design_scenario,
                           workdir=workdir)["config_block"]
    return {f: float(block[f]["designed"]) for f in APPLIED_FIELDS}


def zone_class_block(payload: dict[str, Any]) -> dict[int, dict[str, float]]:
    """``zone_g_w_class`` for the continuous classes, from Stage 0's per-area
    design.

    Only :data:`PER_ZONE_CLASSES` are carried over.  The per-area block also
    contains a per-zone ``g_w_tso_oltc``, and applying that too would change the
    tap price at the same time as the loop gain -- so a Phase-2 row would answer
    "does zone 1 do better with a different design?" rather than the question
    asked, "does zone 1 do better at a higher *gain*?".
    """
    out: dict[int, dict[str, float]] = {}
    for e in payload.get("per_area", []):
        if e.get("kind") != "tso" or not e.get("fields"):
            continue
        spec = {name[len("g_w_"):]: float(d["designed"])
                for name, d in e["fields"].items()
                if name[len("g_w_"):] in PER_ZONE_CLASSES}
        if spec:
            out[int(e["area"])] = spec
    return out


#: Per-DSO voltage-authority factors held FIXED across the search.
#:
#: Empty reproduces every campaign up to 2026-08-18.  Populate it (e.g.
#: ``{"DSO_2": 20.0, "DSO_4": 20.0}``) to tune around a plant whose
#: spread-limited areas already have the relief -- see
#: ``docs/daily_log/08_2026/2026-08-18_dso4_voltage_relief.md``.
#:
#: Applied by :func:`configs.config.apply_dso_v_relief` AFTER the overlay sets
#: ``dso_g_v``, so it derives from the *searched* value: ``dso_g_v_ratio`` is a
#: coordinate, so an absolute relief would let the OLTC loop gain
#: ``dso_g_v / g_w_dso_oltc`` drift as the search walked, and that ratio is what
#: keeps the integer tap out of a limit cycle.
DSO_V_RELIEF_FACTORS: dict[str, float] = {}


def build_config(knobs: dict[str, float], weights: dict[str, float],
                 baseline_cfg, *, zone_g_w_class=None,
                 dso_v_relief: dict[str, float] | None = None):
    """Baseline + designed weights + the objective-trade-off knob, headless."""
    overlay: dict[str, Any] = dict(weights)
    overlay["dso_g_v"] = float(baseline_cfg.dso_g_v) * float(
        knobs["dso_g_v_ratio"])
    # Static weights: the preconditioner must stay off, or it would overwrite
    # exactly the numbers Stage 0 just designed.
    overlay["precondition_g_w"] = False
    if zone_g_w_class:
        # Absolute per-zone weights for the continuous classes; the runner
        # writes these onto the class block of each zone's g_w vector after
        # construction and before the first step.  The global scalars stay in
        # the config and remain the fallback for every class not listed.
        overlay["zone_g_w_class"] = zone_g_w_class
    overlay.update({
        "verbose": 0, "live_plot_controller": False, "live_plot_cascade": False,
        "live_plot_system": False, "run_stability_analysis": False,
    })
    cfg = dataclasses.replace(baseline_cfg, **overlay)
    # LAST, so it reads the searched dso_g_v and the designed dso_oltc rather
    # than the baseline's -- see DSO_V_RELIEF_FACTORS.
    if "dso_v_authority" in knobs:
        # Searched: one shared factor across the spread-limited areas.
        relief = {a: float(knobs["dso_v_authority"])
                  for a in DSO_V_RELIEF_AREAS}
    elif dso_v_relief is not None:
        relief = dso_v_relief
    else:
        relief = DSO_V_RELIEF_FACTORS
    return apply_dso_v_relief(cfg, relief)


def per_transformer_wear(records, duration_s: float) -> dict[str, dict[str, float]]:
    """``{transformer: {ops_per_h, reversals_per_h}}`` for one scenario.

    ``tuning.metrics._tap_wear`` computes exactly this internally and then
    returns only the maximum, because the *constraint* is on the worst
    transformer.  The Tier-2 audit has to report the distribution as well: a
    fleet in which one transformer sits at the budget and eleven sit near zero
    is a different asset-management picture from twelve at a third of it, and
    the worst-case scalar cannot distinguish them.

    Computed here rather than by changing ``tuning/metrics.py``, which would
    alter ``TrajectoryMetrics`` for every existing study.  The arithmetic is the
    same: ``ops`` is the summed absolute tap delta, a reversal is a sign change
    in that delta, and both are divided by the window length in hours.
    """
    from tuning.metrics import _count_tap_reversals, _stack_dict_arrays

    if not records or duration_s <= 0.0:
        return {}
    hours = duration_s / 3600.0
    out: dict[str, dict[str, float]] = {}

    def _add(labels: list[str], seq: np.ndarray) -> None:
        if seq.size == 0 or seq.shape[0] < 2:
            return
        ops = np.nansum(np.abs(np.diff(seq, axis=0)), axis=0)
        for c, lbl in enumerate(labels):
            if c >= seq.shape[1]:
                break
            rev = _count_tap_reversals(seq[:, [c]])
            out[lbl] = {"ops_per_h": float(ops[c]) / hours,
                        "reversals_per_h": float(rev) / hours}

    # TSO: dict[zone -> array of tap positions]; one column per transformer.
    tso_seq = _stack_dict_arrays([r.zone_oltc_taps for r in records])
    zones = sorted({k for r in records for k in r.zone_oltc_taps})
    widths: dict[Any, int] = {}
    for k in zones:
        for r in records:
            v = r.zone_oltc_taps.get(k)
            if v is not None:
                widths[k] = int(np.atleast_1d(np.asarray(v)).size)
                break
    labels = [f"TSO-z{k}-t{j}" for k in zones for j in range(widths.get(k, 0))]
    _add(labels, tso_seq)

    # DSO: dict[name -> scalar tap position].
    dso_keys = sorted({k for r in records for k in r.dso_trafo_tap_pos})
    if dso_keys:
        seq = np.full((len(records), len(dso_keys)), np.nan)
        for i, r in enumerate(records):
            for j, k in enumerate(dso_keys):
                v = r.dso_trafo_tap_pos.get(k)
                if v is not None:
                    seq[i, j] = float(v)
        _add([str(k) for k in dso_keys], seq)
    return out


# ---------------------------------------------------------------------------
# Worker: design + evaluate one candidate
# ---------------------------------------------------------------------------

def evaluate_one(knobs: dict[str, float], args) -> dict[str, Any]:
    from tuning._io import load_config_yaml
    from tuning.metrics import MetricScales
    from tuning.objectives_v2 import (
        PERF_WEIGHT_PROFILES, _run_scenario, _worst_settling_s,
    )
    from tuning_mc.metrics import score_candidate
    from tuning_mc.scenarios_mc import holdout_set_mc, tune_set_mc, wear_day_set
    from tuning_mc.scenarios_mc_v2 import (
        WINDOW_META, tier1_confirm_set, tier1_design_set, tier2_audit_set,
    )

    baseline_cfg = load_config_yaml(Path(args.baseline))
    payload = design_payload(knobs, baseline=Path(args.baseline),
                             design_scenario=args.design_scenario,
                             workdir=Path(args.out) / "designs")
    block = payload["config_block"]
    weights = {f: float(block[f]["designed"]) for f in APPLIED_FIELDS}
    zblock = (zone_class_block(payload)
              if _zone_lambda_spec(knobs) is not None else None)
    cfg = build_config(knobs, weights, baseline_cfg, zone_g_w_class=zblock)

    sets = {
        # 0814 banks, kept so that campaign's results stay reproducible.
        "tune": tune_set_mc, "holdout": holdout_set_mc, "wear": wear_day_set,
        # 0815 banks.
        "tier1": tier1_design_set, "confirm": tier1_confirm_set,
        "audit": tier2_audit_set,
    }
    scenarios = sets[args.scenario_set]()
    scales = MetricScales()
    perf = PERF_WEIGHT_PROFILES[args.perf_weights]

    results, settling = [], []
    wear: dict[str, dict[str, dict[str, float]]] = {}
    t0 = time.perf_counter()
    for sc in scenarios:
        res, records = _run_scenario(sc, cfg, scales)
        results.append(res)
        settling.append(_worst_settling_s(records, sc.event_times_s))
        times = [float(r.time_s) for r in records
                 if math.isfinite(float(r.time_s))]
        wear[sc.name] = per_transformer_wear(
            records, (max(times) - min(times)) if len(times) >= 2 else 0.0)

    limits = load_limits(args.limits)
    score = score_candidate(
        results, cfg, settling_s_by_scenario=settling, limits=limits,
        weights=perf, scales=scales, cvar_pct=args.cvar_pct,
        ds_criterion=args.ds_criterion)

    # Per-stratum aggregates.  Additive only -- ``f_ts`` / ``f_q`` above are
    # untouched, so this cannot change any acceptance decision.  It exists
    # because ``tau``, ``lambda_dso`` and ``dso_g_v_ratio`` are *structurally*
    # inert where DER reactive capability is zero (VDE dead zone), so an f_q
    # averaged across strata mixes a signal with a constant and understates the
    # response of every reactive-allocation coordinate.
    by_stratum: dict[str, dict[str, float]] = {}
    for name, m in score.per_scenario.items():
        s = WINDOW_META.get(name, {}).get("stratum", "unknown")
        acc = by_stratum.setdefault(s, {"n": 0, "f_ts": 0.0, "f_q": 0.0})
        acc["n"] += 1
        acc["f_ts"] += float(m["f_ts"])
        acc["f_q"] += float(m["f_q"])
    for acc in by_stratum.values():
        acc["f_ts"] /= max(acc["n"], 1)
        acc["f_q"] /= max(acc["n"], 1)

    return {
        "knobs": knobs, "key": knob_key(knobs), "weights": weights,
        "dso_g_v": float(cfg.dso_g_v), "wall_s": time.perf_counter() - t0,
        "scenario_set": args.scenario_set,
        "_bank_fingerprint": bank_fingerprint(),
        "_scoring_fingerprint": scoring_fingerprint(args.ds_criterion, limits),
        "ds_criterion": args.ds_criterion,
        "zone_g_w_class": zblock,
        "by_stratum": by_stratum,
        "per_transformer_wear": wear,
        "window_meta": {n: WINDOW_META[n] for n in score.per_scenario
                        if n in WINDOW_META},
        # Stamped, never defaulted: a silently-defaulted limit set changes what
        # "feasible" means without changing anything visible in the result.
        "limits": dataclasses.asdict(limits),
        "limits_source": str(args.limits) if args.limits else "DEFAULTS",
        **score.as_dict(),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _clamp(name: str, value: float) -> float:
    lo, hi = BOUNDS[name]
    return float(min(max(value, lo), hi))


def apply_x0_override(spec: str) -> None:
    """Re-anchor the design point, e.g. ``lambda_tso=0.15,lambda_dso=1.2``.

    :data:`X0` ships with the *analytic* values, which is where the campaign
    starts.  Once the two contraction coordinates have been calibrated against
    a measurement, the probe and the pattern search must start from the
    calibrated point instead -- otherwise Phase A measures identifiability at a
    design point already known to be infeasible, and Phase B spends its first
    polls walking back to it.  The override is a CLI argument rather than an
    edit to :data:`X0` so that the analytic origin stays visible in the source
    and every result file records which point it was run from.
    """
    for kv in spec.split(","):
        name, _, val = kv.partition("=")
        name = name.strip()
        if name not in BOUNDS:
            raise SystemExit(f"[stage1] unknown knob {name!r} in --x0")
        X0[name] = _clamp(name, float(val))
    print(f"[stage1] design point re-anchored: "
          f"{ {k: round(v, 6) for k, v in X0.items()} }", flush=True)


def _design_is_current(knobs: dict[str, float], cached: dict[str, Any],
                       args) -> bool:
    """Do the weights this cached evaluation ran with still match the rule?

    Every result records the ``weights`` it was evaluated at, so the check is
    exact rather than a heuristic on timestamps: regenerate the design (a no-op
    when its fingerprint is current) and compare.  A 1e-9 relative tolerance
    absorbs the JSON round-trip and nothing else.
    """
    try:
        fresh = design_weights(knobs, baseline=Path(args.baseline),
                               design_scenario=args.design_scenario,
                               workdir=Path(args.out) / "designs")
    except RuntimeError:
        return True          # cannot re-derive; keep what we have and say so
    old = cached.get("weights") or {}
    return all(
        f in old and abs(float(old[f]) - v) <= 1e-9 * max(1.0, abs(v))
        for f, v in fresh.items())


def _launch(batch: list[dict[str, float]], args) -> list[dict[str, Any]]:
    """Evaluate a batch of candidates, ``--workers`` at a time."""
    out_dir = Path(args.out) / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                "PYTHONPATH": str(_REPO_ROOT)})

    # Deduplicate by knob key.  A probe or a poll can generate the same
    # candidate twice once the bounds bite -- with lambda_dso at 1.0 the x2 and
    # x4 probes both clamp to 1.90 -- and two identical candidates launched in
    # the same wave race on the same result file and cost a worker slot each.
    pending, seen = [], set()
    for knobs in batch:
        k = knob_key(knobs)
        if k not in seen:
            seen.add(k)
            pending.append(knobs)
    if len(pending) < len(batch):
        print(f"    ({len(batch) - len(pending)} duplicate candidate(s) "
              f"dropped: distinct after clamping)", flush=True)

    running, done = [], []
    while pending or running:
        while pending and len(running) < args.workers:
            knobs = pending.pop(0)
            key = knob_key(knobs)
            res_path = out_dir / f"{args.scenario_set}_{key}.json"
            if res_path.exists():                      # cache / resume
                cached = json.loads(res_path.read_text(encoding="utf-8"))
                bank_ok = (cached.get("_bank_fingerprint") in
                           (None, bank_fingerprint()))
                # A cached row written before scoring_fingerprint existed has
                # no stamp; it cannot be shown to match, so treat it as stale.
                scoring_ok = (cached.get("_scoring_fingerprint")
                              == scoring_fingerprint(args.ds_criterion,
                                                     load_limits(args.limits)))
                if not bank_ok:
                    stale = res_path.with_suffix(".json.bank_changed")
                    res_path.replace(stale)
                    print(f"    [{key}] cached result used a different scenario "
                          f"bank; re-evaluating (old kept at {stale.name})",
                          flush=True)
                elif not scoring_ok:
                    stale = res_path.with_suffix(".json.scoring_changed")
                    res_path.replace(stale)
                    print(f"    [{key}] cached result was scored under a "
                          f"different objective; re-evaluating "
                          f"(old kept at {stale.name})", flush=True)
                elif _design_is_current(knobs, cached, args):
                    done.append(cached)
                    continue
                else:
                    # The rule that produced these weights has since changed, so
                    # the row is no longer comparable with the rest of the sweep.
                    # Set it aside (never delete: it is the only record of what
                    # ran) and re-evaluate.
                    stale = res_path.with_suffix(".json.superseded")
                    res_path.replace(stale)
                    print(f"    [{key}] cached result used superseded weights; "
                          f"re-evaluating (old kept at {stale.name})",
                          flush=True)
            spec = out_dir / f"spec_{key}.json"
            spec.write_text(json.dumps(knobs), encoding="utf-8")
            log = out_dir / f"log_{args.scenario_set}_{key}.txt"
            cmd = [sys.executable, "-m", "tuning_mc.stage_1_search",
                   "--eval-one", str(spec), "--result", str(res_path),
                   "--baseline", str(args.baseline), "--out", str(args.out),
                   "--scenario-set", args.scenario_set,
                   "--design-scenario", args.design_scenario,
                   "--perf-weights", args.perf_weights,
                   "--cvar-pct", str(args.cvar_pct),
                   "--ds-criterion", args.ds_criterion]
            if args.limits:
                cmd += ["--limits", str(args.limits)]
            if args.limits:
                cmd += ["--limits", str(args.limits)]
            fh = log.open("w", encoding="utf-8")
            running.append((subprocess.Popen(cmd, cwd=_REPO_ROOT, env=env,
                                             stdout=fh, stderr=subprocess.STDOUT),
                            res_path, fh, key))
            time.sleep(2.0)                            # stagger the net build
        still = []
        for proc, res_path, fh, key in running:
            if proc.poll() is None:
                still.append((proc, res_path, fh, key))
                continue
            fh.close()
            if res_path.exists():
                done.append(json.loads(res_path.read_text(encoding="utf-8")))
                r = done[-1]
                print(f"    [{key}] f_ts={r['f_ts']:.5f} f_q={r['f_q']:.5f} "
                      f"feasible={r['feasible']} ({r['wall_s'] / 60:.1f} min)",
                      flush=True)
            else:
                print(f"    [{key}] FAILED (rc={proc.returncode}); see log",
                      flush=True)
        running = still
        if running:
            time.sleep(5.0)
    return done


def rho_calibration(pairs: list[tuple[float, float]], rho_target: float,
                    *, tag: str = "stage1-lam",
                    rho_margin: float = 0.0) -> dict[str, Any]:
    """Fit the realised contraction against the design target and pick lambda*.

    The design target and the realised contraction are related **affinely**, not
    proportionally: the integer columns contribute a constant that no continuous
    weight can scale away, and ``lambda`` adds on top.  Reporting a ratio
    ``rho/lambda`` would hide exactly the term that decides feasibility, so the
    floor is fitted explicitly and reported as a first-class result.

    ``lambda*`` is the largest *measured* point meeting ``rho <= rho_target``;
    the interpolated boundary is reported next to it but never substituted for
    it, because only measured points have been simulated.
    """
    # ``rho_worst`` is a MAXIMUM over the bank's windows, so its in-sample value
    # is a downward-biased estimate of the same quantity on a fresh draw from the
    # same distribution.  Measured 2026-08-15: a lambda calibrated to
    # rho = 1.4743 against a ceiling of 1.5 -- a 1.7 % margin -- measured 1.5201
    # on a confirmation bank, i.e. INFEASIBLE out of sample.  Resampling all
    # C(18,12) twelve-window banks from the pooled windows showed why: one window
    # sits 1.5 % above every other, and whether a bank contains it is a coin
    # flip, so the max is bimodal with a ~3 % jump rather than smoothly variable.
    #
    # ``rho_margin`` shrinks the target the SELECTION uses, leaving the declared
    # physical ceiling untouched: the criterion stays "no window may exceed
    # rho_target", and the margin is an allowance for the instability of the
    # statistic used to check it.  Report both numbers, always.
    eff_target = float(rho_target) / (1.0 + float(rho_margin))
    out: dict[str, Any] = {"rho_target": float(rho_target),
                           "rho_margin": float(rho_margin),
                           "effective_target": eff_target,
                           "points": [{"lambda": float(v), "rho": float(r)}
                                      for v, r in pairs]}
    if rho_margin:
        print(f"\n[{tag}] selection target {eff_target:.4f} = {rho_target:g} / "
              f"(1 + {rho_margin:g}): the declared ceiling is {rho_target:g}, "
              f"and {100 * rho_margin:g} % is the measured bank-to-bank "
              f"instability of the worst-window statistic.")
    rho_target = eff_target
    ok = [(v, r) for v, r in pairs if math.isfinite(r) and r <= rho_target]
    lam_star = max((v for v, _ in ok), default=None)
    out["lambda_star"] = lam_star
    if lam_star is None:
        print(f"\n[{tag}] NO lambda in the swept range meets rho <= "
              f"{rho_target}. Widen the sweep downward before proceeding; do "
              f"not relax the criterion silently.")
    else:
        rho_at = dict(pairs)[lam_star]
        out["rho_at_lambda_star"] = float(rho_at)
        print(f"\n[{tag}] lambda* = {lam_star:g} (largest MEASURED point "
              f"meeting the criterion): rho = {rho_at:.4f}")

    finite = [(v, r) for v, r in pairs if math.isfinite(r)]
    if len(finite) >= 3:
        lam_arr = np.array([v for v, _ in finite], dtype=float)
        rho_arr = np.array([r for _, r in finite], dtype=float)
        A = np.vstack([lam_arr, np.ones_like(lam_arr)]).T
        (slope, floor), *_ = np.linalg.lstsq(A, rho_arr, rcond=None)
        resid = float(np.abs(rho_arr - (slope * lam_arr + floor)).max())
        out.update({"slope": float(slope), "floor": float(floor),
                    "max_residual": resid})
        print(f"[{tag}] measured relation: rho = {floor:.4f} + "
              f"{slope:.4f} * lambda   (max residual {resid:.4f})")
        print(f"[{tag}] FLOOR = {floor:.4f}: the contraction the integer "
              f"columns impose on their own.  No continuous weight can go "
              f"below it; rho <= {floor:.2f} is reachable only by re-pricing "
              f"the OLTCs.")
        if rho_target > floor:
            boundary = (rho_target - floor) / slope
            out["lambda_at_boundary"] = float(boundary)
            print(f"[{tag}] lambda at the criterion boundary: {boundary:.4f}")
        # The margin below the hard bound of 2, not the plant, is what sets
        # lambda: report the sensitivity so the choice is visible as a choice.
        margins = {}
        for tgt in (1.5, 1.6, 1.7, 1.8):
            if tgt > floor:
                margins[tgt] = float((tgt - floor) / slope)
        out["lambda_by_ceiling"] = margins
        if margins:
            txt = "  ".join(f"rho<={t:g}: lam={v:.3f}"
                            for t, v in sorted(margins.items()))
            print(f"[{tag}] ceiling sensitivity   {txt}")
    return out


def phase_lam(args) -> int:
    """Calibrate lambda against a measurable contraction criterion.

    The design target ``lambda_max(M)`` is computed over the *preconditioned*
    columns of a *cached* ``H`` with the integer columns excluded; the realised
    loop has none of those exemptions.  The two therefore differ by a factor
    that has to be measured rather than assumed -- at lambda = 0.9 the measured
    ``rho_emp_p95`` was 2.29-2.63, i.e. ~2.5x the design intent and above the
    hard OFO bound of 2.

    This sweep measures that derating directly: one shared lambda per point,
    ``rho_emp_p95`` recorded per scenario, and ``lambda*`` taken as the largest
    value meeting ``--rho-target``.  The criterion is stated independently of
    any existing controller -- 1.5 is a 25 % margin below the theoretical bound
    of 2 -- so nothing about the hand-tuned weights enters the choice.
    """
    vals = sorted((float(v) for v in args.lam_values.split(",")), reverse=True)
    batch = []
    for v in vals:
        k = dict(X0)
        k["lambda_tso"] = _clamp("lambda_tso", v)
        k["lambda_dso"] = _clamp("lambda_dso", v)
        batch.append(k)
    print(f"[stage1-lam] sweeping lambda over {vals}; criterion "
          f"rho_emp_p95 <= {args.rho_target}", flush=True)
    results = _launch(batch, args)
    by_key = {r["key"]: r for r in results}

    def _rho(r: dict) -> float:
        """Worst rho, recovered from per-scenario data for older cached rows.

        Results cached before ``worst_rho_emp_p95`` became a first-class field
        do not carry it; they do carry ``rho_emp_p95`` per scenario, which is
        what it was derived from.  Re-deriving keeps a 20-minute evaluation
        usable instead of discarding it over a schema change.
        """
        v = r.get("worst_rho_emp_p95")
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
        vals = [m.get("rho_emp_p95") for m in r.get("per_scenario", {}).values()
                if isinstance(m.get("rho_emp_p95"), (int, float))]
        return max(vals) if vals else float("nan")

    for r in results:                      # normalise in place for downstream
        r["worst_rho_emp_p95"] = _rho(r)

    print(f"\n{'lambda':>8}{'g_w_der':>10}{'g_w_pcc':>10}{'g_w_dso_der':>13}"
          f"{'worst rho':>11}{'f_ts':>10}{'f_q':>9}{'taps/h':>9}  feasible")
    rows = []
    for v, knobs in zip(vals, batch):
        r = by_key.get(knob_key(knobs))
        if r is None:
            print(f"{v:>8.3f}   (failed)")
            continue
        w = r["weights"]
        print(f"{v:>8.3f}{w['g_w_der']:>10.4g}{w['g_w_pcc']:>10.4g}"
              f"{w['g_w_dso_der']:>13.5g}{r['worst_rho_emp_p95']:>11.4f}"
              f"{r['f_ts']:>10.4f}{r['f_q']:>9.4f}"
              f"{r['worst_tap_ops_per_h']:>9.3f}  {r['feasible']}")
        rows.append((v, r))

    calib = rho_calibration([(v, r["worst_rho_emp_p95"]) for v, r in rows],
                            args.rho_target, rho_margin=args.rho_margin)
    lam_star = calib["lambda_star"]
    payload = {"lam_values": vals, "rho_target": args.rho_target,
               "lambda_star": lam_star, "calibration": calib,
               "rows": [{"lambda": v, **{k: r[k] for k in
                                         ("f_ts", "f_q", "worst_rho_emp_p95",
                                          "worst_tap_ops_per_h", "feasible",
                                          "weights", "per_scenario")}}
                        for v, r in rows]}
    p = Path(args.out) / "phase_lambda.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[stage1-lam] wrote {p}")
    return 0


def phase_scan(args) -> int:
    """Scan ONE knob with the others held fixed.

    The joint lambda sweep tied ``lambda_tso`` and ``lambda_dso`` together,
    which confounded the two layers: ``rho_emp_p95`` is a **TSO-only**
    diagnostic (``zone_contraction_lhs`` is written per TSO zone by the
    coordinator; no DSO equivalent is recorded anywhere), so every feasibility
    verdict there was decided by the TSO layer while the DSO weights moved by a
    factor of 8 as a passenger.  One knob at a time is the fix.
    """
    base = dict(X0)
    if args.fix:
        for kv in args.fix.split(","):
            name, _, val = kv.partition("=")
            name = name.strip()
            if name not in BOUNDS:
                raise SystemExit(f"[stage1-scan] unknown knob {name!r}")
            base[name] = _clamp(name, float(val))
    if args.scan_knob not in BOUNDS:
        raise SystemExit(f"[stage1-scan] unknown knob {args.scan_knob!r}")
    vals = [float(v) for v in args.scan_values.split(",")]
    batch = [dict(base, **{args.scan_knob: _clamp(args.scan_knob, v)})
             for v in vals]

    print(f"[stage1-scan] scanning {args.scan_knob} over {vals}", flush=True)
    print(f"[stage1-scan] held fixed: "
          f"{ {k: v for k, v in base.items() if k != args.scan_knob} }",
          flush=True)
    results = _launch(batch, args)
    by_key = {r["key"]: r for r in results}

    print(f"\n{args.scan_knob:>14}{'g_w_der':>10}{'g_w_pcc':>10}"
          f"{'g_w_dso_der':>13}{'rho(TSO)':>10}{'f_ts':>10}{'f_q':>9}"
          f"{'taps/h':>9}{'rev/h':>8}  feasible")
    rows = []
    for v, knobs in zip(vals, batch):
        r = by_key.get(knob_key(knobs))
        if r is None:
            print(f"{v:>14.4g}   (failed)")
            continue
        w = r["weights"]
        print(f"{v:>14.4g}{w['g_w_der']:>10.4g}{w['g_w_pcc']:>10.4g}"
              f"{w['g_w_dso_der']:>13.5g}"
              f"{r.get('worst_rho_emp_p95', float('nan')):>10.4f}"
              f"{r['f_ts']:>10.4f}{r['f_q']:>9.4f}"
              f"{r['worst_tap_ops_per_h']:>9.3f}"
              f"{r['worst_reversals_per_h']:>8.3f}  {r['feasible']}")
        rows.append({"value": v, **{k: r[k] for k in
                                    ("f_ts", "f_q", "feasible", "weights",
                                     "worst_tap_ops_per_h",
                                     "worst_reversals_per_h", "per_scenario")},
                     "rho": r.get("worst_rho_emp_p95")})

    # A scan of lambda_tso IS the lambda calibration, done cleanly: rho_emp_p95
    # is a TSO-only diagnostic, so the joint sweep of --phase lam moved the DSO
    # weights by the same factor as a passenger and its f_ts / f_q columns are
    # not attributable to either layer.  Holding lambda_dso fixed removes that
    # confound while measuring the same rho, which doubles as a check on the
    # layer decoupling: rho at a given lambda_tso must reproduce the joint
    # sweep's value regardless of where lambda_dso sits.
    calib = None
    if args.scan_knob == "lambda_tso":
        calib = rho_calibration(
            [(r["value"], float(r["rho"])) for r in rows
             if isinstance(r.get("rho"), (int, float))],
            args.rho_target, tag="stage1-scan",
            rho_margin=args.rho_margin)

    # Diagnostic: the mean of the worst three windows, alongside the max the
    # criterion is stated on.  Resampling put its bank-to-bank spread at 0.60 %
    # against the max's ~3 %, so a future campaign can move the criterion onto it
    # -- but only after stating what ceiling means for a mean rather than a max.
    for r in rows:
        per = r.get("per_scenario", {})
        vals = sorted((m["rho_emp_p95"] for m in per.values()
                       if isinstance(m.get("rho_emp_p95"), (int, float))
                       and math.isfinite(m["rho_emp_p95"])), reverse=True)
        r["rho_worst3_mean"] = (sum(vals[:3]) / min(3, len(vals))
                                if vals else float("nan"))

    ok = [r for r in rows if r["feasible"]]
    if ok:
        b_ts = min(ok, key=lambda r: r["f_ts"])
        b_q = min(ok, key=lambda r: r["f_q"])
        print(f"\n[stage1-scan] best f_ts: {args.scan_knob}="
              f"{b_ts['value']:g} -> {b_ts['f_ts']:.4f} (f_q {b_ts['f_q']:.4f})")
        print(f"[stage1-scan] best f_q : {args.scan_knob}="
              f"{b_q['value']:g} -> {b_q['f_q']:.4f} (f_ts {b_q['f_ts']:.4f})")
        if b_ts["value"] != b_q["value"]:
            print("[stage1-scan] the two criteria disagree -- this knob trades "
                  "TS voltage against interface-Q; report both, do not average.")
    p = Path(args.out) / f"scan_{args.scan_knob}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"knob": args.scan_knob, "fixed": base,
                             "calibration": calib, "rows": rows}, indent=1),
                 encoding="utf-8")
    print(f"[stage1-scan] wrote {p}")
    return 0


def _candidates_from(path: Path) -> list[dict[str, float]]:
    """Knob dicts out of a phase artefact or a plain list.

    Accepts what the campaign actually produces -- ``phase_b.json`` (filter +
    incumbent), ``phase_a.json`` (the probe's design point), a ``scan_*.json``
    (its rows carry no knobs, so those are rejected explicitly rather than
    silently yielding nothing), or a hand-written list of knob dicts.
    """
    def _clean(d: dict) -> dict[str, float]:
        """Keep only real knobs.

        Hand-written candidate lists carry ``_comment`` keys saying which point
        a row is -- which is exactly the annotation that makes them auditable --
        and ``knob_key`` calls ``float()`` on every value, so anything not in
        :data:`BOUNDS` has to be dropped here rather than blowing up mid-launch.
        """
        knobs = {k: float(v) for k, v in d.items() if k in BOUNDS}
        missing = [k for k in X0 if k not in knobs]
        if missing:
            raise SystemExit(f"[stage1-eval] candidate is missing {missing}: "
                             f"{ {k: v for k, v in d.items()} }")
        return knobs

    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        if data and all(isinstance(d, dict) and "lambda_tso" in d
                        for d in data):
            return [_clean(d) for d in data]
        if data and all(isinstance(d, dict) and "knobs" in d for d in data):
            return [_clean(d["knobs"]) for d in data]
        raise SystemExit(f"[stage1-eval] {path.name}: list is not knob dicts")
    out: list[dict[str, float]] = []
    for entry in data.get("filter", []):
        if "knobs" in entry:
            out.append(_clean(entry["knobs"]))
    for key in ("incumbent", "x0"):
        v = data.get(key)
        if isinstance(v, dict) and "lambda_tso" in v:
            out.append(_clean(v))
    if not out:
        raise SystemExit(
            f"[stage1-eval] no candidates in {path.name}; a scan_*.json records "
            f"knob VALUES per row, not knob dicts -- pass phase_b.json, "
            f"phase_a.json, or an explicit list")
    return out


def phase_eval(args) -> int:
    """Evaluate an explicit candidate list on ``--scenario-set``.

    The tier-2 audit and the confirmation run are both "take these finished
    points and measure them somewhere else", which no existing phase expresses:
    ``scan`` sweeps one knob and ``a``/``b`` generate their own candidates.
    Keeping this as its own phase means the audit is a *measurement* of points
    the search already chose, and cannot feed back into the choice.
    """
    if not args.candidates:
        raise SystemExit("[stage1-eval] --candidates is required")
    batch = _candidates_from(Path(args.candidates))
    if args.include_x0:
        batch.append(dict(X0))
    # De-duplicate here as well as in _launch, so the printed table has one row
    # per distinct candidate rather than repeats that resolve to one file.
    seen, uniq = set(), []
    for k in batch:
        kk = knob_key(k)
        if kk not in seen:
            seen.add(kk)
            uniq.append(k)
    print(f"[stage1-eval] {len(uniq)} candidate(s) on scenario-set "
          f"'{args.scenario_set}', limits={args.limits or 'DEFAULTS'}",
          flush=True)
    results = _launch(uniq, args)
    by_key = {r["key"]: r for r in results}

    print(f"\n{'candidate':<54}{'f_ts':>10}{'f_q':>10}{'rho':>9}"
          f"{'taps/h':>9}{'rev/h':>8}  feasible")
    rows = []
    for k in uniq:
        r = by_key.get(knob_key(k))
        if r is None:
            print(f"{str(k)[:53]:<54}   (failed)")
            continue
        txt = " ".join(f"{n}={k[n]:g}" for n in sorted(k))
        print(f"{txt[:53]:<54}{r['f_ts']:>10.5f}{r['f_q']:>10.5f}"
              f"{r.get('worst_rho_emp_p95', float('nan')):>9.4f}"
              f"{r.get('worst_tap_ops_per_h', float('nan')):>9.3f}"
              f"{r.get('worst_reversals_per_h', float('nan')):>8.3f}"
              f"  {r['feasible']}")
        rows.append(r)
    # ``--tag`` is not cosmetic: two --phase eval runs on the same scenario set,
    # launched concurrently, otherwise write the same summary file and the second
    # silently overwrites the first.  Measured 2026-08-17.  The per-candidate
    # files under evals/ are unaffected, so nothing is lost, but the summary is.
    suffix = f"_{args.tag}" if args.tag else ""
    p = Path(args.out) / f"eval_{args.scenario_set}{suffix}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"scenario_set": args.scenario_set,
                             "limits_source": str(args.limits) if args.limits
                             else "DEFAULTS",
                             "rows": rows}, indent=1), encoding="utf-8")
    print(f"[stage1-eval] wrote {p}")
    return 0


def phase_a(args) -> int:
    """One-at-a-time probe: which directions carry signal at all.

    Liveness is decided on **both** filter criteria, not on ``f_ts`` alone.
    Deciding it on the supervisory objective would re-introduce, in the
    screening step, exactly the failure the two-criterion filter exists to
    avoid: measured 2026-08-14, ``lambda_dso`` moves ``f_ts`` by at most
    ±0.4 % while moving ``f_q`` by 80 %, and ``tau`` by 0.5 % against 60 %.
    Both would have been declared dead and excluded from the pattern search on
    the strength of a criterion neither of them is responsible for -- and
    ``lambda_dso`` is the coordinate the preceding sweep selected on ``f_q``.
    A direction is live if either criterion responds.
    """
    from tuning_mc.metrics import CandidateScore

    mults = [float(m) for m in args.probe_multipliers.split(",")]
    batch = [dict(X0)]
    tags = [("x0", 1.0)]
    for name in X0:
        for m in mults:
            k = dict(X0)
            k[name] = _clamp(name, X0[name] * m)
            if abs(k[name] - X0[name]) < 1e-12:
                continue
            batch.append(k)
            tags.append((name, m))

    print(f"[stage1-A] {len(batch)} evaluations "
          f"({len(X0)} knobs x {len(mults)} multipliers + the design point), "
          f"{args.workers} workers", flush=True)
    results = _launch(batch, args)
    by_key = {r["key"]: r for r in results}

    base = by_key.get(knob_key(X0))
    if base is None:
        raise SystemExit("[stage1-A] the design point failed to evaluate")
    print(f"\n[stage1-A] design point: f_ts={base['f_ts']:.6f} "
          f"f_q={base['f_q']:.6f} feasible={base['feasible']}")

    print(f"\n{'knob':<16}{'x':>7}{'f_ts':>11}{'d f_ts':>10}"
          f"{'f_q':>11}{'d f_q':>10}  feasible")
    sens_ts: dict[str, float] = {}
    sens_q: dict[str, float] = {}
    for (name, m), knobs in zip(tags, batch):
        if name == "x0":
            continue
        r = by_key.get(knob_key(knobs))
        if r is None:
            print(f"{name:<16}{m:>7.2f}   (failed)")
            continue
        d_ts = (r["f_ts"] - base["f_ts"]) / base["f_ts"]
        d_q = (r["f_q"] - base["f_q"]) / base["f_q"] if base["f_q"] else float("nan")
        sens_ts[name] = max(sens_ts.get(name, 0.0), abs(d_ts))
        if math.isfinite(d_q):
            sens_q[name] = max(sens_q.get(name, 0.0), abs(d_q))
        print(f"{name:<16}{m:>7.2f}{r['f_ts']:>11.5f}{100 * d_ts:>9.2f}%"
              f"{r['f_q']:>11.5f}{100 * d_q:>9.2f}%  {r['feasible']}")

    thr = args.dead_threshold
    live = sorted(k for k in X0
                  if max(sens_ts.get(k, 0.0), sens_q.get(k, 0.0)) >= thr)
    dead = sorted(k for k in X0 if k not in live)
    print(f"\n[stage1-A] per-knob response (max over the four probes):")
    print(f"{'knob':<16}{'|d f_ts|':>10}{'|d f_q|':>10}   verdict")
    for k in sorted(X0):
        ts, q = sens_ts.get(k, 0.0), sens_q.get(k, 0.0)
        why = ("live" if k in live else "dead")
        if k in live and ts < thr <= q:
            why = "live (via f_q only)"
        elif k in live and q < thr <= ts:
            why = "live (via f_ts only)"
        print(f"{k:<16}{100 * ts:>9.2f}%{100 * q:>9.2f}%   {why}")
    print(f"\n[stage1-A] live directions (max response >= "
          f"{100 * thr:g} % on EITHER criterion): {live}")
    print(f"[stage1-A] dead directions (no signal on this design bank): {dead}")
    state = {"x0": X0, "sensitivity": sens_ts, "sensitivity_f_q": sens_q,
             "dead_threshold": thr, "live": live, "dead": dead, "base": base}
    p = Path(args.out) / "phase_a.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(f"[stage1-A] wrote {p}")
    return 0


def phase_b(args) -> int:
    """Compass search on the live directions, with the two-criterion filter."""
    from tuning_mc.metrics import CandidateScore, dominates

    a_path = Path(args.out) / "phase_a.json"
    if not a_path.exists():
        raise SystemExit(f"[stage1-B] run phase a first ({a_path} missing)")
    state = json.loads(a_path.read_text(encoding="utf-8"))
    live = state["live"] or sorted(X0)
    incumbent = dict(state["x0"])
    best = state["base"]
    filt = [best]

    def _score(r) -> CandidateScore:
        return CandidateScore(f_ts=r["f_ts"], f_q=r["f_q"], hard=(),
                              feasible=r["feasible"],
                              f_ds=float(r.get("f_ds", float("nan"))))

    wds = bool(args.filter_ds)
    if wds:
        print("[stage1-B] filter criteria: f_ts, f_q, f_ds (three)", flush=True)

    delta = float(args.delta0)
    history = []
    print(f"[stage1-B] live directions: {live}; delta0={delta} decades",
          flush=True)
    while delta >= args.delta_min:
        batch = []
        for name in live:
            for sgn in (+1.0, -1.0):
                k = dict(incumbent)
                k[name] = _clamp(name, incumbent[name] * 10.0 ** (sgn * delta))
                if abs(k[name] - incumbent[name]) > 1e-12:
                    batch.append(k)
        # Coupled directions: cheap insurance against a diagonal valley.
        if "lambda_tso" in live and "lambda_dso" in live:
            for sgn in (+1.0, -1.0):
                k = dict(incumbent)
                for nm in ("lambda_tso", "lambda_dso"):
                    k[nm] = _clamp(nm, incumbent[nm] * 10.0 ** (sgn * delta))
                batch.append(k)
        print(f"\n[stage1-B] poll at delta={delta:.3f}: {len(batch)} points",
              flush=True)
        results = _launch(batch, args)

        improving = [
            r for r in results
            if r["feasible"] and not any(
                dominates(_score(f), _score(r), tol=args.filter_tol,
                          with_ds=wds)
                for f in filt)
        ]
        better = [r for r in improving if r["f_ts"] < best["f_ts"]]
        if better:
            best = min(better, key=lambda r: r["f_ts"])
            incumbent = dict(best["knobs"])
            filt = [f for f in filt
                    if not dominates(_score(best), _score(f),
                                     tol=args.filter_tol, with_ds=wds)] + [best]
            print(f"[stage1-B] ACCEPT f_ts={best['f_ts']:.6f} "
                  f"f_q={best['f_q']:.6f}  knobs="
                  f"{ {k: round(v, 5) for k, v in incumbent.items()} }",
                  flush=True)
        else:
            delta *= 0.5
            print(f"[stage1-B] poll failed; shrink delta -> {delta:.4f}",
                  flush=True)
        for r in improving:
            if r not in filt and not any(
                    dominates(_score(f), _score(r), tol=args.filter_tol,
                          with_ds=wds)
                    for f in filt):
                filt.append(r)
        history.append({"delta": delta, "best_f_ts": best["f_ts"],
                        "best_f_q": best["f_q"], "knobs": incumbent})
        out = {"incumbent": incumbent, "best": best, "history": history,
               "filter": [{"knobs": f["knobs"], "f_ts": f["f_ts"],
                           "f_q": f["f_q"]} for f in filt]}
        (Path(args.out) / "phase_b.json").write_text(
            json.dumps(out, indent=1), encoding="utf-8")

    print(f"\n[stage1-B] converged at delta={delta:.4f}")
    print(f"[stage1-B] incumbent: { {k: round(v, 5) for k, v in incumbent.items()} }")
    print(f"[stage1-B] f_ts {state['base']['f_ts']:.6f} -> {best['f_ts']:.6f} "
          f"({100 * (best['f_ts'] / state['base']['f_ts'] - 1):+.2f} %)")
    print(f"[stage1-B] filter has {len(filt)} non-dominated points")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuning_mc.stage_1_search")
    p.add_argument("--phase", choices=("lam", "scan", "a", "b", "eval"),
                   default="a")
    p.add_argument("--candidates", type=Path, default=None,
                   help="--phase eval: JSON holding the candidates to measure. "
                        "phase_b.json (its filter + incumbent), phase_a.json, "
                        "or a plain list of knob dicts.")
    p.add_argument("--tag", default=None,
                   help="Suffix for the --phase eval summary filename, so two "
                        "concurrent eval runs on one scenario set do not "
                        "overwrite each other's summary.")
    p.add_argument("--include-x0", action="store_true",
                   help="--phase eval: also measure the design point, which is "
                        "the baseline every reported delta is against.")
    p.add_argument("--scan-knob", default="lambda_dso")
    p.add_argument("--scan-values", default="0.15,0.3,0.6,0.9,1.2,1.6")
    p.add_argument("--fix", default=None,
                   help="Knobs held fixed, e.g. 'lambda_tso=0.15'.")
    p.add_argument("--limits", type=Path, default=None,
                   help="JSON of ConstraintLimits fields. Stamped on every "
                        "result; omitting it uses package defaults, which is "
                        "recorded as limits_source=DEFAULTS.")
    p.add_argument("--lam-values", default="0.9,0.6,0.4,0.25,0.15,0.10")
    p.add_argument("--rho-target", type=float, default=1.5,
                   help="Contraction criterion for lambda*: 25 %% margin below "
                        "the hard OFO bound of 2.")
    p.add_argument("--rho-margin", type=float, default=0.0,
                   help="Allowance for the bank-to-bank instability of the "
                        "worst-window statistic. lambda* is selected against "
                        "rho_target/(1+margin) while the declared ceiling stays "
                        "rho_target. Measured 0.031 on this plant; at 0 the "
                        "calibration does not transfer out of sample.")
    p.add_argument("--filter-ds", action="store_true",
                   help="Add f_ds (the subordinate layer's own voltage cost) as "
                        "a third filter criterion. It is a stated controlled "
                        "output; without this the search spends it -- measured "
                        "2026-08-15, -47 %% f_ds for +15 %% f_q.")
    p.add_argument("--search-dso-v-authority", nargs="?", type=float,
                   const=20.0, default=None, metavar="START",
                   help="Promote 'dso_v_authority' into X0 so Phase A/B search "
                        "it, starting at START (default 20, the measured "
                        "operating point). One shared coordinate across "
                        f"{DSO_V_RELIEF_AREAS}. Without this the factor is "
                        "fixed at DSO_V_RELIEF_FACTORS and the guard criterion "
                        "only audits it.")
    p.add_argument("--ds-criterion", choices=("v_rms", "guard"), default="v_rms",
                   help="What f_ds measures. 'v_rms' (default, reproduces every "
                        "study up to 2026-08-18) is |v_mean_ds - v_set|, the "
                        "DSO profile's CENTRE -- which the interface OLTC drives "
                        "onto v_set by construction, so it is smallest exactly "
                        "when the tap has spent its authority. 'guard' is the "
                        "headroom measure: excess beyond a corridor shrunk by "
                        "DS_GUARD_HEADROOM_PU at both ends. Use 'guard' with "
                        "--filter-ds for new work.")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers", type=int, default=6,
                   help="Measured throughput peaks at 6 on this machine and "
                        "regresses past 8 (memory-bandwidth bound).")
    p.add_argument("--scenario-set", default="tune",
                   choices=("tune", "holdout", "wear",      # 0814 banks
                            "tier1", "confirm", "audit"))   # 0815 banks
    p.add_argument("--design-scenario", default="none",
                   help="Operating point at which Stage 0 caches H. 'none' "
                        "keeps the baseline's own (2016-01-05 08:00, rural_700 "
                        "-- the N-1 window, where DER capability is measured at "
                        "2988 Mvar). Do NOT point this at a zero-capability "
                        "window.")
    p.add_argument("--perf-weights", default="ts_voltage_primary")
    p.add_argument("--cvar-pct", type=float, default=100.0)
    p.add_argument("--probe-multipliers", default="0.25,0.5,2,4")
    p.add_argument("--dead-threshold", type=float, default=0.01,
                   help="A knob whose best probe moves f_ts by less than this "
                        "relative amount is dropped from Phase B.")
    p.add_argument("--delta0", type=float, default=0.3,
                   help="Initial compass step [decades].")
    p.add_argument("--delta-min", type=float, default=0.075)
    p.add_argument("--filter-tol", type=float, default=1e-4)
    p.add_argument("--x0", default=None,
                   help="Re-anchor the design point, e.g. "
                        "'lambda_tso=0.15,lambda_dso=1.2'. Use after the "
                        "contraction coordinates have been calibrated, so the "
                        "probe and the pattern search start from the measured "
                        "baseline rather than the analytic one.")
    p.add_argument("--eval-one", type=Path, default=None,
                   help="Worker mode: evaluate the knobs in this JSON file.")
    p.add_argument("--result", type=Path, default=None)
    args = p.parse_args(argv)

    if args.eval_one is not None:
        # utf-8-sig, not utf-8: a spec handed over from a PowerShell
        # ``Out-File -Encoding utf8`` carries a BOM, and json.loads rejects it.
        knobs = json.loads(Path(args.eval_one).read_text(encoding="utf-8-sig"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            res = evaluate_one(knobs, args)
        Path(args.result).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps({k: res[k] for k in ("f_ts", "f_q", "feasible")}))
        return 0

    # Promote the gated voltage-authority coordinate BEFORE --x0, so an
    # explicit --x0 dso_v_authority=... still wins.
    if getattr(args, "search_dso_v_authority", None) is not None:
        X0["dso_v_authority"] = _clamp("dso_v_authority",
                                       float(args.search_dso_v_authority))
        print(f"[stage1] searching dso_v_authority from "
              f"{X0['dso_v_authority']:g} across {DSO_V_RELIEF_AREAS}; "
              f"phase B costs +2 evaluations per poll while it stays live",
              flush=True)

    if args.x0:
        apply_x0_override(args.x0)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    return {"lam": phase_lam, "scan": phase_scan, "a": phase_a,
            "b": phase_b, "eval": phase_eval}[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
