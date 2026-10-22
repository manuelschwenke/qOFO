"""The Chapter-9 selected weight design, loaded rather than reconstructed.

What this replaces
------------------
Until 2026-09-10 this module rebuilt the design at run time: seven search
coordinates were pushed through ``tuning_mc.stage_0_preconditioning`` to
*generate* weights, a per-area voltage relief was applied on top, a
post-selection correction then stripped and re-derived that relief at a
different factor and overrode one area's DER weight by hand, a frozen network
multiplier was applied last, and the result was asserted against an archived
JSON under ``results/``.  Five layers, and the module's own docstring recorded
that the chapter text named a *superseded* candidate.

The campaign now produces a single parameter set.  This module loads it.  If
the weights need to change, the campaign is re-run and the parameter set is
replaced; nothing is regenerated, patched or asserted at run time.

The 2026-09-15 correction: ONE structural configuration
-------------------------------------------------------
Until 2026-09-15 this module built its config from a bare ``MultiTSOConfig()``
while the Stage-II search, the run comparison and the RMS co-simulation all
built theirs from :func:`experiments.run_multi_system_ofo.make_config`.  Those
two bases differ in fourteen fields, and six of them change the physics:

======================================  ====================  ==================
field                                   ``make_config()``     bare dataclass
======================================  ====================  ==================
``install_tso_tertiary_shunts``         True                  False
``shunt_dispatch``                      ``miqp``              ``off``
``tie_boundary_equivalent``             ``thevenin``          ``pq``
``local_sensitivities_tso/dso``         True                  False
``zone_v_setpoints_pu``                 1.03 in all 3 zones   None
``dso_der_scale``/``dso_load_p_scale``  {} (symmetric)        DSO_3 x2
======================================  ====================  ==================

So the Chapter-9.1 response experiments ran with the MSC/MSR banks absent -- and
``shunt_int_g_w`` therefore inert -- a PQ rather than Thevenin boundary
equivalent, no local sensitivity refresh and no zone voltage setpoints, while
the weights they ran at were selected on the other configuration.  ``N_inner``
measured there is not ``N_inner`` of the controller the chapter reports.

The last row is the one difference the author chose to KEEP: the Chapter-9 batch
is declared on the doubled-DSO_3 network, so :func:`build_selected_config`
re-applies those multipliers on top of the runner configuration.  Everything
else now comes from ``make_config()`` unchanged.

The base is now :func:`~experiments.run_multi_system_ofo.make_config`, which is
what the author asked for in writing::

    from configs.paramsets import apply_paramset
    from experiments.run_multi_system_ofo import make_config

    cfg = apply_paramset(make_config(), "ch9_selected")

and which is exactly what :func:`build_selected_config` now returns.

Interface is unchanged -- ``build_selected_config() -> (cfg, provenance)`` --
but the provenance dict regained the ``campaign``, ``candidate_key`` and
``verified`` keys that the Chapter-9.1 entry points print, which had been
dropped in the 2026-09-10 rewrite and raised ``KeyError`` on every run since.

Author: Manuel Schwenke / Claude Code (2026-09-15)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The parameter set this chapter runs at.
#:
#: ``ch9_selected`` is candidate ``399e30d3b60b``, chosen 2026-09-11 from the
#: Stage-II evented 12 h bank by ``ch_9_3_select.py``; see
#: ``results/THESIS_ch9_3_stage2/refine_evented/selection_report.json``.
#:
#: It replaced ``stage1_start`` here on 2026-09-15.  ``stage1_start`` is the
#: Stage-I *analytic* design point and remains the correct parameter set for
#: Experiment B and for the initial-weight leg of Experiment C -- pass it
#: explicitly (``--paramset stage1_start``) rather than by editing this
#: constant, so that a run record always names the controller it ran at.
#:
#: 2026-10-20 (author decision): ``ch9_selected_10_2026`` -- Experiment B re-run
#: on the final configuration under docs/tuning/2026-10-16_protocol_experiment_
#: B_rerun.md (candidate df17c0c32904, every weight within +-5.4 % of the
#: September selection 399e30d3b60b; shunt_int_g_w 0.1605).  Was "ch9_selected".
SELECTED_PARAMSET = "ch9_selected_10_2026"

#: The Stage-I analytic design point, for the initial-weight legs (B and C).
INITIAL_PARAMSET = "stage1_start"

#: Per-DSO scenario multipliers of the 2026-09-09 "frozen network".
#:
#: Applied by default: the Chapter-9 batch is declared on the doubled-DSO_3
#: model, the PowerFactory project stores it, and the whole Chapter-9.1 lineage
#: ran doubled.  Confirmed by the author 2026-09-15.
#:
#: **This is a stated mismatch, not an oversight.**  ``make_config()`` -- the
#: base every weight in ``ch9_selected`` was selected on -- does NOT set these,
#: so the Stage-II screen, refinement and selection all ran on the SYMMETRIC
#: model.  Running the response experiments here therefore evaluates the
#: selected weights on a network the search never saw.  The chapter has to say
#: so; ``resolved_config.json`` and the run banner both record which network
#: was used, and ``apply_frozen_network=False`` gives the Stage-II-comparable
#: variant when that comparison is what is wanted.
FROZEN_DSO_DER_SCALE: Dict[str, float] = {"DSO_3": 2.0}
FROZEN_DSO_LOAD_P_SCALE: Dict[str, float] = {"DSO_3": 2.0}


#: Live-plot fields forced off for a measurement campaign.
#:
#: ``make_config()`` is the INTERACTIVE runner configuration and switches the
#: controller and cascade windows on.  Building on it (2026-09-15) therefore
#: turned them on in every Chapter-9 worker, where the bare dataclass lineage
#: had them off -- and ``live_plot_controller`` constructs a matplotlib
#: ``TSOControllerLivePlotter`` per case, inside each of 16 pool workers, with
#: a redraw every step.  Pure overhead in a batch: the plotters only read state
#: and draw, so they cannot change a result, but they cost wall time and have
#: no one to show it to.
#:
#: Not simply "the defaults": ``live_plot_show_reserves`` and
#: ``live_plot_show_tie_flows`` default to True and are display options of a
#: window that will not exist, so they are left alone.
_LIVE_PLOT_FIELDS: Tuple[str, ...] = (
    "live_plot_controller", "live_plot_cascade", "live_plot_system",
    "live_plot_tracking", "live_plot_sbx", "live_plot_sbxv",
)


def build_selected_config(
    *,
    paramset: str = SELECTED_PARAMSET,
    apply_frozen_network: bool = True,
    live_plots: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """``(cfg, provenance)`` at the named weights, on the runner configuration.

    ``cfg`` is ``apply_paramset(make_config(), paramset)`` -- the same object
    the Stage-II search, ``ch_9_3_compare_runs`` and the RMS co-simulation
    build -- so a result measured here is measured on the controller the rest
    of the chapter reports.

    Live plotting is forced OFF unless ``live_plots=True``; see
    :data:`_LIVE_PLOT_FIELDS`.  This is a measurement harness, and
    ``make_config()`` is the interactive configuration.

    Raises with an actionable message if the parameter set does not exist.
    That is deliberately not softened into a fallback: a fallback here is how
    the previous design came to run at weights nobody had selected.
    """
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from configs.paramsets import apply_paramset, available, load_paramset
    from experiments.run_multi_system_ofo import make_config

    if paramset not in available():
        raise FileNotFoundError(
            f"parameter set {paramset!r} does not exist. Available: "
            f"{available() or '(none)'}. The Chapter-9 weights are produced by "
            f"the Stage-I/II campaign (see tuning/stage1_screen.py and the "
            f"2026-09-10 weight reset in docs/daily_log); until it has run "
            f"there is no selected design to load."
        )

    doc = load_paramset(paramset)
    cfg = apply_paramset(make_config(), paramset, verbose=False)

    if not live_plots:
        cfg = dataclasses.replace(
            cfg, **{name: False for name in _LIVE_PLOT_FIELDS})

    network: Optional[Dict[str, Any]] = None
    if apply_frozen_network:
        cfg = dataclasses.replace(
            cfg,
            dso_der_scale=dict(FROZEN_DSO_DER_SCALE),
            dso_load_p_scale=dict(FROZEN_DSO_LOAD_P_SCALE),
        )
        network = {"dso_der_scale": dict(FROZEN_DSO_DER_SCALE),
                   "dso_load_p_scale": dict(FROZEN_DSO_LOAD_P_SCALE)}

    # The one invariant still worth asserting: the config that comes out must
    # be weight-complete.  Everything else the old module checked was
    # reconciling layers that no longer exist.
    cfg.require_weights()

    prov = doc.get("provenance", {})
    provenance: Dict[str, Any] = {
        "paramset": paramset,
        "title": doc.get("title", paramset),
        "fields": dict(doc.get("fields", {})),
        "paramset_provenance": prov,
        "base_config": "experiments.run_multi_system_ofo.make_config",
        "frozen_network": network,
        # -- keys the Chapter-9.1 entry points print.  Derived from the
        #    parameter set rather than restated, so they cannot drift from it.
        "campaign": prov.get("producer", paramset),
        "candidate_key": (prov.get("weight_fingerprint")
                          or prov.get("selected_key")
                          or prov.get("candidate_key")
                          or "(none)"),
        "verified": doc.get("title", paramset),
    }
    return cfg, provenance


def resolved_manifest(cfg: Any, provenance: Dict[str, Any]) -> Dict[str, Any]:
    """The complete resolved configuration, for the run record.

    Section 3 of the 2026-09-15 experiment plan requires a run to persist the
    configuration it actually resolved rather than the name of the helper that
    built it.  Every ``MultiTSOConfig`` field that is JSON-representable is
    written out; the rest are stringified, so the manifest is always complete
    and never raises on an exotic field.
    """
    import json

    fields: Dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            value = repr(value)
        fields[f.name] = value
    return {"design": provenance, "resolved_config": fields}


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="_ch9_selected_design")
    ap.add_argument("--paramset", default=SELECTED_PARAMSET)
    ap.add_argument("--symmetric-network", action="store_true",
                    help="drop the doubled-DSO_3 multipliers and run the "
                         "symmetric model the Stage-II search used")
    ap.add_argument("--manifest", action="store_true",
                    help="print the full resolved configuration")
    a = ap.parse_args()

    cfg, prov = build_selected_config(
        paramset=a.paramset,
        apply_frozen_network=not a.symmetric_network)
    if a.manifest:
        print(json.dumps(resolved_manifest(cfg, prov), indent=2, default=str))
    else:
        print(json.dumps(prov, indent=2, default=str))
    print(f"\ntso_period_s={cfg.tso_period_s} dso_period_s={cfg.dso_period_s} "
          f"dt_s={cfg.dt_s} scenario={cfg.scenario}")
    print(f"shunts={cfg.install_tso_tertiary_shunts}/{cfg.shunt_dispatch} "
          f"g_w_tso_shunt={cfg.g_w_tso_shunt} "
          f"boundary={cfg.tie_boundary_equivalent} "
          f"local_sens={cfg.local_sensitivities_tso}/"
          f"{cfg.local_sensitivities_dso}")
    print(f"dso_der_scale={cfg.dso_der_scale} "
          f"dso_load_p_scale={cfg.dso_load_p_scale}")
