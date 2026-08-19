"""The §9.3 selected weight design, reproduced rather than retyped.

Both §9.1 closed-loop experiments (`ch_9_1_ts_period_sweep`, the `T_TS` sweep,
and `ch_9_1_ninner_isolated_sts`, the isolated-STS `N_inner` measurement) must
run at the weights the chapter selects. ``N_inner`` is a joint property of the
period ratio and of ``G_w``: run it at the file defaults of
``experiments/run_multi_system_ofo.py`` (``g_w_der=10``, ``g_w_dso_der=1000``,
...) and the answer describes the *old in-service* controller, not the one
Chapter 9 designs.

The design is therefore rebuilt through the campaign's own recipe --
``load_config_yaml`` -> ``design_payload`` -> ``build_config``
(``tuning_mc/stage_1_search.py::evaluate_one``) -- so that a later correction to
Stage 0 or to the config overlay propagates here instead of leaving a stale
literal behind. The designed weights are then asserted against the archived
evaluation JSON, and a mismatch raises: silently running at *almost* the
selected point is the failure this module exists to prevent.

Selected candidate: campaign ``campaign_0815``, key ``aa4f6d4a8654`` -- the row
whose six coordinates match the chapter and whose ``rho_emp_p95 = 1.4480351``
is the 1.448 the chapter prints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Campaign and candidate the chapter selects. Named once, here.
CAMPAIGN = "campaign_0815"
CANDIDATE_KEY = "aa4f6d4a8654"

#: The six search coordinates, verbatim from
#: ``results/tuning_mc/campaign_0815/evals/spec_aa4f6d4a8654.json``.
KNOBS: Dict[str, float] = {
    "engage_tso_pu": 0.018,
    "lambda_tso": 0.371,
    "lambda_dso": 1.6,
    "tau": 1.0,
    "engage_dso_pu": 0.025,
    "dso_g_v_ratio": 1.5,
}

#: Relative tolerance on the weight assertion. The recipe is deterministic, so
#: this is a float-repr guard, not a tolerance band: anything larger than this
#: means the design changed and the run must not proceed.
WEIGHT_RTOL = 1e-9

_EVAL_DIR = REPO_ROOT / "results" / "tuning_mc" / CAMPAIGN / "evals"
_DESIGN_DIR = REPO_ROOT / "results" / "tuning_mc" / CAMPAIGN / "designs"


def archived_evaluation() -> Dict[str, Any]:
    """The Tier-1 evaluation record of the selected candidate."""
    path = _EVAL_DIR / f"tier1_{CANDIDATE_KEY}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"archived evaluation missing: {path}. The selected design cannot "
            f"be verified, so the run is refused rather than executed against "
            f"an unchecked config.")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_selected_config() -> Tuple[Any, Dict[str, Any]]:
    """``(cfg, provenance)`` at the §9.3 selected weights.

    Raises if the rebuilt weights differ from the archived ones, or if the
    knobs in the archive differ from :data:`KNOBS` -- either means this module
    is describing a different candidate from the one it names.
    """
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from tuning._io import load_config_yaml
    from tuning_mc.stage_1_search import (
        APPLIED_FIELDS, DEFAULT_BASELINE, build_config, design_payload,
        knob_key, stage0_fingerprint, _zone_lambda_spec,
    )

    archive = archived_evaluation()

    # (1) The archive must be the candidate this module claims.
    if knob_key(KNOBS) != CANDIDATE_KEY:
        raise AssertionError(
            f"knob_key(KNOBS) = {knob_key(KNOBS)!r} but this module names "
            f"{CANDIDATE_KEY!r}; the coordinates and the key disagree.")
    for field, want in KNOBS.items():
        got = float(archive["knobs"][field])
        if abs(got - want) > 1e-12:
            raise AssertionError(
                f"knob {field}: archive has {got!r}, this module has {want!r}")

    # (2) Rebuild through the campaign's own recipe.
    baseline_cfg = load_config_yaml(Path(DEFAULT_BASELINE))
    payload = design_payload(KNOBS, baseline=Path(DEFAULT_BASELINE),
                             design_scenario="none", workdir=_DESIGN_DIR)
    block = payload["config_block"]
    weights = {f: float(block[f]["designed"]) for f in APPLIED_FIELDS}

    # zone_g_w_class is null for this candidate: global scalars, no per-area
    # block. Recomputed rather than assumed, so a change of knobs is caught.
    zblock = None
    if _zone_lambda_spec(KNOBS) is not None:
        raise AssertionError(
            "this candidate carries a per-zone lambda spec, but the archived "
            "zone_g_w_class is null; the two cannot both be right.")
    if archive.get("zone_g_w_class") is not None:
        raise AssertionError(
            f"archive zone_g_w_class is {archive['zone_g_w_class']!r}, "
            f"expected null for this candidate.")

    # (3) The assertion the handoff asks for: designed == archived.
    mismatches = []
    for field, got in sorted(weights.items()):
        want = float(archive["weights"][field])
        if want == 0.0:
            ok = got == 0.0
        else:
            ok = abs(got - want) <= WEIGHT_RTOL * abs(want)
        if not ok:
            mismatches.append(f"  {field}: rebuilt {got!r} != archived {want!r}")
    if mismatches:
        raise AssertionError(
            "the rebuilt design does not reproduce the archived one -- Stage 0 "
            "or the config overlay has changed since campaign_0815, so this "
            "run would NOT be at the weights the chapter selects:\n"
            + "\n".join(mismatches))

    cfg = build_config(KNOBS, weights, baseline_cfg, zone_g_w_class=zblock)

    # (4) dso_g_v is set by the overlay from the searched ratio, not designed
    # by Stage 0, so it is checked against the archive separately.
    want_gv = float(archive["dso_g_v"])
    got_gv = float(cfg.dso_g_v)
    if abs(got_gv - want_gv) > WEIGHT_RTOL * abs(want_gv):
        raise AssertionError(
            f"dso_g_v: built {got_gv!r} != archived {want_gv!r}")

    provenance = {
        "campaign": CAMPAIGN,
        "candidate_key": CANDIDATE_KEY,
        "knobs": dict(KNOBS),
        "weights": weights,
        "dso_g_v": got_gv,
        "zone_g_w_class": None,
        "baseline_yaml": str(DEFAULT_BASELINE),
        "stage0_fingerprint": stage0_fingerprint(),
        "design_payload": str(_DESIGN_DIR / f"stage0_{CANDIDATE_KEY}.json"),
        "archived_eval": str(_EVAL_DIR / f"tier1_{CANDIDATE_KEY}.json"),
        "archived_rho_emp_p95": archive.get("worst_rho_emp_p95"),
        "verified": "rebuilt weights == archived weights",
    }
    return cfg, provenance


if __name__ == "__main__":
    cfg, prov = build_selected_config()
    print(json.dumps(prov, indent=2, default=str))
    print(f"\ntso_period_s={cfg.tso_period_s} dso_period_s={cfg.dso_period_s} "
          f"dt_s={cfg.dt_s} scenario={cfg.scenario} "
          f"precondition_g_w={cfg.precondition_g_w}")
