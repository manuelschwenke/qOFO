"""tuning/scripts/save_baseline.py — write a baseline YAML for tuning.

Persists the ``MultiTSOConfig`` that the tuning CLIs consume via ``--baseline``.

Source of the baseline (changed 2026-07-31)
-------------------------------------------
It now comes from ``experiments/run_multi_system_ofo.py::make_config()`` — the
hand-tuned configuration that demonstrably controls well — instead of
``make_base_config()`` in ``experiments/002_M_TSO_M_DSO_COMPARE.py``.  Two
reasons:

1. **The 002 module no longer exists at that path.**  It moved to
   ``experiments/archived/``, so ``importlib.import_module`` raised and this
   script could not run at all.

2. **The 002 baseline describes a different plant** from the one in use:

   =============================== ============ ==============
   field                           make_config  002 baseline
   =============================== ============ ==============
   ``install_tso_tertiary_shunts`` True         False
   ``shunt_dispatch``              integrator   (absent -> off)
   ``coordination_mode``           sbx_h        (absent -> none)
   ``dt_s`` / ``dso_period_s``     20 / 20      60 / 10
   =============================== ============ ==============

   Tuning against it optimised a shunt-less, uncoordinated plant at a
   different timescale, so none of the resulting weights transfer.

The scenario overlay (``ScenarioSpec.overlay_on``) replaces the horizon, start
time, contingencies and timing, and ``FIXED_OVERRIDES`` forces the run headless,
so the plant-structure and actuator settings are what this file contributes:
shunt installation and integrator gains, OLTC cooldowns, coordination mode.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from tuning._io import load_config_yaml, save_config_yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_OUT = _SCRIPT_DIR / "configs" / "baseline_ieee39.yaml"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuning.scripts.save_baseline")
    p.add_argument(
        "--output", type=Path, default=_DEFAULT_OUT,
        help="Where to write the YAML.",
    )
    p.add_argument(
        "--no-verify", action="store_true",
        help="Skip the save/load round-trip check.",
    )
    p.add_argument(
        "--shunts", choices=("as-configured", "off"), default="as-configured",
        help=(
            "'off' removes the TSO tertiary shunts from the tuned plant "
            "entirely (they are dispatched by the separate integrator, not by "
            "the MIQP, so they are not a tuning degree of freedom). NOTE it "
            "sets BOTH shunt_dispatch='off' AND "
            "install_tso_tertiary_shunts=False: the runner reinterprets "
            "dispatch='off' as the legacy 'miqp' mode whenever the shunts are "
            "still installed, which would put shunt integers back INTO the "
            "MIQP -- the opposite of what is intended here."
        ),
    )
    args = p.parse_args(argv)

    from experiments.run_multi_system_ofo import make_config

    cfg = make_config()
    if args.shunts == "off":
        cfg = dataclasses.replace(
            cfg, shunt_dispatch="off", install_tso_tertiary_shunts=False,
        )
    save_config_yaml(cfg, args.output)
    print(f"[save_baseline] wrote {args.output}")

    if not args.no_verify:
        # A silent round-trip failure here would be costly: nested dataclasses
        # (`sbx_config`, `measurement_noise`) used to come back as plain dicts,
        # so `cfg.sbx_config.k_sched` raised AttributeError mid-run.
        back = load_config_yaml(args.output)
        mismatches = [
            name for name in (
                "sbx_config", "measurement_noise",
                "precondition_exclude_classes",
            )
            if type(getattr(cfg, name)) is not type(getattr(back, name))
        ]
        if mismatches:
            raise SystemExit(
                f"[save_baseline] round-trip changed the type of "
                f"{mismatches}; the YAML would misbehave at run time."
            )
        # Mapping KEYS are the other silent round-trip failure: `jsonable`
        # stringifies them, and every consumer falls back without raising --
        # `tie_thevenin_k` to THEVENIN_K_DEFAULT, `zone_*` to the scalar
        # default.  A Thevenin baseline would then declare measured
        # per-corridor impedances and run with none of them.
        key_mismatches = [
            name for name in (
                "tie_thevenin_k", "zone_g_w_scale", "zone_v_setpoints_pu",
            )
            if getattr(cfg, name) != getattr(back, name)
        ]
        if key_mismatches:
            raise SystemExit(
                f"[save_baseline] round-trip changed the CONTENT of "
                f"{key_mismatches} (usually stringified mapping keys); the "
                f"YAML would silently run a different model."
            )
        print("[save_baseline] round-trip verified (types + mapping keys)")

    print(
        f"[save_baseline] plant: shunts="
        f"{cfg.install_tso_tertiary_shunts} dispatch={cfg.shunt_dispatch!r} "
        f"coordination={cfg.coordination_mode!r} "
        f"dt_s={cfg.dt_s} tso={cfg.tso_period_s} dso={cfg.dso_period_s}"
    )
    print(
        f"[save_baseline] boundary: {cfg.tie_boundary_equivalent!r} "
        f"({len(cfg.tie_thevenin_k)} per-corridor k)"
        if isinstance(cfg.tie_thevenin_k, dict)
        else f"[save_baseline] boundary: {cfg.tie_boundary_equivalent!r} "
             f"(k={cfg.tie_thevenin_k})"
    )
    print(
        f"[save_baseline] TSO g_w: der={cfg.g_w_der} pcc={cfg.g_w_pcc} "
        f"oltc={cfg.g_w_tso_oltc} gen={cfg.g_w_gen:g}  "
        f"zone_g_w_scale={cfg.zone_g_w_scale}  "
        f"precondition_g_w={cfg.precondition_g_w}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
