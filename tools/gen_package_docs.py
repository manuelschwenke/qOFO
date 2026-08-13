#!/usr/bin/env python3
"""Generate one markdown overview per Python package.

Reads ``tools/doc_coverage.json`` (produced by ``tools/doc_audit.py``) and emits
``docs/architecture/packages/<slug>.md`` for every package, plus an index.

The module inventory and coverage figures are generated; the per-package
*purpose* text is curated below in :data:`PURPOSE` and is the part worth
reading. A package without a curated entry falls back to its ``__init__``
docstring, and is flagged in the index so the gap is visible rather than
papered over.

Docs are written to ``docs/architecture/packages/`` rather than as
``README.md`` inside each folder, because ``experiments/``,
``network/ieee39/analysis/`` and the ``_archive/`` trees already carry hand-
written READMEs that must not be overwritten.

Usage::

    python tools/doc_audit.py && python tools/gen_package_docs.py

Author: Manuel Schwenke / Claude Code (2026-07-31)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "docs" / "architecture" / "packages"

#: Curated one-paragraph purpose per package. Everything else is generated.
PURPOSE: Dict[str, str] = {
    "configs": (
        "Configuration dataclasses for the whole study. `MultiTSOConfig` is the "
        "single object every runner takes; `CascadeConfig` is the per-controller "
        "slice handed to a DSO. These types are the provenance record: each run "
        "serialises its config to `config.json`, and the analysis admission "
        "filters read that block to decide whether two runs are comparable."
    ),
    "core": (
        "Plant abstraction, measurements and the message types crossing the "
        "TSO/DSO seam. `core.plant.Plant` is the interface that lets the same "
        "controller stack drive either the quasi-static pandapower plant or the "
        "PowerFactory RMS plant; `core.profiles` loads and applies the annual "
        "time series."
    ),
    "controller": (
        "The OFO controllers themselves. The TSO controller dispatches AVR "
        "setpoints, OLTCs and shunts and issues Q setpoints downward; the DSO "
        "controller tracks those setpoints with local actuators and reports "
        "capability and tracking error upward. Controllers never see the plant: "
        "they act only on cached sensitivities and their own measurements."
    ),
    "optimisation": (
        "The MIQP solver layer that each OFO step calls. Note that the feasible "
        "set carries no weight, so scaling every objective weight by a constant "
        "leaves the argmin unchanged -- the invariance that makes the tuning "
        "search space degenerate as parameterised."
    ),
    "sensitivity": (
        "Cached linear models of the plant: Jacobian-derived voltage and "
        "reactive-power sensitivities. This is the *only* representation of the "
        "plant a controller has, so its accuracy bounds achievable performance, "
        "and re-linearisation policy is a design decision rather than an "
        "implementation detail."
    ),
    "network": (
        "Test-system construction. Builds the modified IEEE 39-bus transmission "
        "system, attaches the HV distribution underlays, and applies scenario "
        "and per-DSO multipliers. Scenario definitions live in "
        "`network/ieee39/scenarios/`."
    ),
    "network/ieee39": (
        "The IEEE 39-bus build itself: buses, lines, the HV sub-networks, DER "
        "capability diagrams and the constants that fix scenario-independent "
        "physical quantities (e.g. `DSO_Q_PROFILE_BASE_MVAR`)."
    ),
    "experiments": (
        "Entry points. `run_comparison_rms_cosim_qss.py` runs the quasi-static "
        "and PowerFactory RMS legs and compares them; `run_rms_cosim.py` runs the "
        "RMS leg alone; `run_openloop_qss_to_rms.py` is the genuine open-loop "
        "u -> y replay; `run_multi_system_ofo.py` holds the authoritative "
        "hand-tuned configuration. See `docs/architecture/simulation_workflows.md`."
    ),
    "experiments/runners": (
        "`run_multi_tso_dso` -- the closed-loop driver shared by every "
        "experiment. One function, ~4,500 lines, stepping the whole cascade. Its "
        "structure is documented step by step in "
        "`docs/architecture/simulation_workflows.md`."
    ),
    "experiments/helpers": (
        "Shared machinery for the co-simulation entry points: the common CLI and "
        "config builder, the record dataclasses, trajectory extraction and "
        "settling statistics."
    ),
    "pf": (
        "PowerFactory integration: session handling, model synchronisation, the "
        "RMS plant adapter, event-pool management and result export. Untracked "
        "in git as of 2026-07-31. `pf/probes/` holds read-only one-off probes "
        "that answer a single question about PF API or RMS solver semantics."
    ),
    "analysis": (
        "Post-processing of stored runs. Each study has its own admission filter "
        "deciding which runs are comparable -- see `deadband_selection.py`, whose "
        "`ADMIT` block is the worked example."
    ),
    "tuning": (
        "Offline Bayesian optimisation of the controller weights, plus the "
        "stability certificate machinery. See "
        "`docs/daily_log/07_2026/2026-07-31_bo_tuning_audit.md` for the audit of "
        "why the search as parameterised was not identifiable."
    ),
    "sbx_h": (
        "Horizontal scheduled-boundary coordination between TSO zones. The "
        "supported comparison is none versus SBX-H."
    ),
    "sbx_v": (
        "Vertical band/request/grant coordination across the TSO-DSO seam."
    ),
    "visualisation": (
        "Live plotting during a run and publication figures afterwards."
    ),
    "export": ("Snapshot export of built networks for external tools."),
    "tools": ("Repository maintenance utilities, including this documentation audit."),
    "network/ieee39/scenarios": (
        "Scenario definitions selecting the installed DER capacity per DSO. "
        "`base_410` and `rural_700` (410 vs 700 MW per DSO) share the "
        "transmission-side wind build but are **not** comparable to each other; "
        "every analysis filters on the scenario each run recorded."
    ),
    "network/ieee39/analysis": (
        "Read-only probes characterising the built network -- boundary "
        "conditions and DSO P/Q envelopes -- without running a controller."
    ),
    "network/nordic32": (
        "Nordic-32 conversion utilities. Not used by the current studies."
    ),
    "analysis/observer": (
        "Stability-observer analysis: contraction estimates and the certificate "
        "machinery applied to recorded runs."
    ),
    "experiments/CIGRE_2026": (
        "The CIGRE Calgary 2026 paper experiments, including the Monte-Carlo "
        "campaign over contingency schedules."
    ),
    "experiments/archived": (
        "Superseded entry points, retained because daily logs and shell history "
        "still reference them. Includes the deprecation shim for the former "
        "`run_rms_phase6_replay.py`."
    ),
    "pf/probes": (
        "One-off, read-only probes answering a single question about the "
        "PowerFactory API or RMS solver semantics -- event firing, ElmFile "
        "playback, frame connections. They print findings and restore state; "
        "probe bodies stay behind a `__main__` guard and are not part of the "
        "co-simulation import graph."
    ),
    "tuning/scripts": (
        "Drivers for the BO campaigns and for regenerating the baseline YAML "
        "from the authoritative hand-tuned configuration."
    ),
    "tuning/stability_certificate": (
        "LMI/IQC machinery producing stability certificates used as ceilings "
        "and floors on the controller weights."
    ),
    "tuning/reports": ("Report rendering for completed tuning studies."),
    "docs/daily_log": (
        "Not a code package: holds `_build_index.py`, which regenerates "
        "`docs/daily_log/INDEX.md` from the log headings."
    ),
}

SKIP_PREFIXES = ("tests", "_archive", ".pytest_cache")


def slug(pkg: str) -> str:
    return pkg.replace("/", ".").replace(".", "_") if pkg != "." else "root"


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.0f}%" if d else "n/a"


def main() -> int:
    data = json.loads((PROJECT_ROOT / "tools" / "doc_coverage.json")
                      .read_text(encoding="utf-8"))
    pkgs = data["packages"]
    mods = data["modules"]
    by_pkg: Dict[str, List[dict]] = {}
    for m in mods:
        by_pkg.setdefault(m["pkg"], []).append(m)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written, uncurated = [], []

    for pkg in sorted(by_pkg):
        is_support = pkg.startswith(SKIP_PREFIXES)
        purpose = PURPOSE.get(pkg)
        if purpose is None and not is_support:
            uncurated.append(pkg)
        s = pkgs.get(pkg, {})
        lines = [
            f"# `{pkg}`",
            "",
            "> Inventory generated by `tools/gen_package_docs.py`; the purpose",
            "> text is hand-written. Regenerate after adding modules.",
            "",
        ]
        if purpose:
            lines += [purpose, ""]
        elif is_support:
            lines += ["Test or archived code; retained for reference.", ""]
        else:
            lines += ["*(No curated description yet -- see the module summaries "
                      "below.)*", ""]

        lines += [
            "## At a glance",
            "",
            f"- modules: **{s.get('modules', 0)}**, "
            f"{s.get('loc', 0):,} lines",
            f"- module docstrings: {s.get('modules_doc', 0)}/{s.get('modules', 0)}"
            f" ({pct(s.get('modules_doc', 0), s.get('modules', 0))})",
            f"- public functions documented: {s.get('pub_fn_doc', 0)}/"
            f"{s.get('pub_fn', 0)} ({pct(s.get('pub_fn_doc', 0), s.get('pub_fn', 0))})",
            f"- classes documented: {s.get('cls_doc', 0)}/{s.get('cls', 0)}"
            f" ({pct(s.get('cls_doc', 0), s.get('cls', 0))})",
            "",
            "## Modules",
            "",
            "| module | LOC | fn | cls | summary |",
            "|---|---|---|---|---|",
        ]
        for m in sorted(by_pkg[pkg], key=lambda x: -x["loc"]):
            name = Path(m["rel"]).name
            summary = (m["summary"] or "").replace("|", r"\|").strip()
            if not summary:
                summary = "_no module docstring_"
            if len(summary) > 110:
                summary = summary[:107] + "..."
            lines.append(f"| `{name}` | {m['loc']:,} | {m['n_pub_fn']} | "
                         f"{m['n_cls']} | {summary} |")
        lines.append("")

        path = OUT_DIR / f"{slug(pkg)}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append((pkg, path.name, s.get("modules", 0), s.get("loc", 0)))

    idx = [
        "# Package overviews",
        "",
        "One file per Python package. Generated by "
        "`tools/gen_package_docs.py` from `tools/doc_coverage.json`;",
        "run `python tools/doc_audit.py` first to refresh the data.",
        "",
        "Coverage figures for the whole codebase: "
        "[doc_coverage.md](../doc_coverage.md).",
        "",
        "| package | modules | LOC | curated |",
        "|---|---|---|---|",
    ]
    for pkg, fname, nmod, loc in sorted(written, key=lambda r: -r[3]):
        mark = "yes" if pkg in PURPOSE else ("n/a" if pkg.startswith(SKIP_PREFIXES)
                                             else "**no**")
        idx.append(f"| [`{pkg}`]({fname}) | {nmod} | {loc:,} | {mark} |")
    if uncurated:
        idx += ["", "## Packages still lacking a curated description", "",
                *(f"- `{p}`" for p in sorted(uncurated))]
    (OUT_DIR / "README.md").write_text("\n".join(idx) + "\n", encoding="utf-8")

    print(f"wrote {len(written)} package docs + index to "
          f"{OUT_DIR.relative_to(PROJECT_ROOT)}")
    if uncurated:
        print(f"uncurated ({len(uncurated)}): {', '.join(sorted(uncurated))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
