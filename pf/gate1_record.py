"""
pf/gate1_record.py
==================
Gate-1 evidence collector (read-only): connects to the IEEE39_qOFO project,
runs the template load flow, and writes ``docs/pf_gate1_record.md`` with

* PF / Python provenance,
* the study-case inventory,
* every synchronous machine with its type data (rated S, voltage, inertia)
  -- this settles the "G 05: 300 vs 600 MVA" template question,
* all 39 bus voltages and angles (for the manual Table-10 comparison),
* the ComLdf option flags relevant to the Phase-2 parity option set.

Read-only: no project object is created or modified (the ComLdf executed
is the one already present in the active study case).

Run on the PF machine::

    python pf\\gate1_record.py

Author: Manuel Schwenke / Claude Code (2026-07-17)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pf.session import (  # noqa: E402
    DEFAULT_PROJECT_PATH,
    connect,
    get_all,
    list_study_cases,
    run_ldf,
)

OUT_PATH = Path(__file__).resolve().parents[1] / "docs" / "pf_gate1_record.md"

#: ComLdf attributes worth recording for the parity option set.  Names are
#: probed defensively -- releases differ; absent ones are reported as such.
_COMLDF_FLAGS = (
    "iopt_net",    # AC balanced / unbalanced / DC
    "iopt_pq",     # consider voltage dependency of loads
    "iopt_at",     # automatic tap adjustment of transformers
    "iopt_asht",   # automatic shunt adjustment
    "iopt_lim",    # consider reactive power limits
    "iopt_plim",   # consider active power limits
    "i_power",     # load-flow method / power balancing (release-dependent)
    "iopt_sim",
    "errlf",       # max. iteration error
)

#: TypSym attributes probed per machine (release-dependent spellings).
_TYPSYM_ATTRS = ("sgn", "ugn", "h", "cosn")


def _attr_or_none(obj, attr: str):
    try:
        return obj.GetAttribute(attr)
    except Exception:
        return None


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    app = connect(DEFAULT_PROJECT_PATH)
    prj = app.GetActiveProject()

    lines = []
    lines.append("# Gate-1 record — IEEE39_qOFO template (read-only probe)")
    lines.append("")
    lines.append(f"- Recorded: {datetime.now():%Y-%m-%d %H:%M}")
    lines.append(f"- Project: `{prj.GetFullName()}`")
    lines.append(f"- Python: `{sys.version.split()[0]}` "
                 f"(`{sys.executable}`)")
    for cand in ("e:version", "version"):
        v = _attr_or_none(app, cand)
        if v is not None:
            lines.append(f"- PF application attribute `{cand}`: `{v}`")
    lines.append("")

    # ── Study cases ──────────────────────────────────────────────────────
    lines.append("## Study cases")
    lines.append("")
    for c in list_study_cases(app):
        lines.append(f"- `{c.loc_name}`")
    lines.append("")

    # ── Machines and type data ───────────────────────────────────────────
    lines.append("## Synchronous machines (ElmSym -> TypSym)")
    lines.append("")
    lines.append("| machine | outserv | type | " +
                 " | ".join(_TYPSYM_ATTRS) + " |")
    lines.append("|---|---|---|" + "---|" * len(_TYPSYM_ATTRS))
    for m in sorted(get_all(app, "ElmSym"), key=lambda o: o.loc_name):
        typ = _attr_or_none(m, "typ_id")
        typ_name = typ.loc_name if typ is not None else "<none>"
        cells = [f"`{m.loc_name}`", str(_attr_or_none(m, "outserv")),
                 f"`{typ_name}`"]
        for a in _TYPSYM_ATTRS:
            v = _attr_or_none(typ, a) if typ is not None else None
            cells.append("n/a" if v is None else f"{v}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("(H reconciliation check: the build plan flags G 05 — the "
                 "PDF prints Sr = 300 MVA with H = 4.333 s on machine base, "
                 "which matches H = 26 s on 100 MVA base only for "
                 "Sr = 600 MVA.)")
    lines.append("")

    # ── Load flow + voltages ─────────────────────────────────────────────
    ldf = run_ldf(app)
    lines.append("## Template load flow (ComLdf of the active study case)")
    lines.append("")
    lines.append("Converged. Bus voltages for the manual Table-10 check:")
    lines.append("")
    lines.append("| bus | u [pu] | phi [deg] |")
    lines.append("|---|---|---|")
    for t in sorted(get_all(app, "ElmTerm"), key=lambda o: o.loc_name):
        u = _attr_or_none(t, "m:u")
        phi = _attr_or_none(t, "m:phiu")
        u_s = "n/a" if u is None else f"{u:.5f}"
        phi_s = "n/a" if phi is None else f"{phi:.4f}"
        lines.append(f"| `{t.loc_name}` | {u_s} | {phi_s} |")
    lines.append("")

    # ── ComLdf flags ─────────────────────────────────────────────────────
    lines.append("## ComLdf option flags (parity-relevant, as found)")
    lines.append("")
    for flag in _COMLDF_FLAGS:
        v = _attr_or_none(ldf, flag)
        lines.append(f"- `ComLdf.{flag}` = "
                     + ("`<not present>`" if v is None else f"`{v!r}`"))
    lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
