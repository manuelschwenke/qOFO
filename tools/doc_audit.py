#!/usr/bin/env python3
"""Static documentation audit of the Python codebase.

Parses every tracked ``.py`` file with :mod:`ast` and reports, per package and
per module, whether the module, its classes and its functions carry docstrings.
Nothing is imported, so the audit is safe to run without PowerFactory, without a
solver licence, and while a co-simulation is in flight.

Outputs
-------
``docs/architecture/doc_coverage.md``
    Human-readable coverage report, worst packages first.
``tools/doc_coverage.json``
    Machine-readable, for regeneration and diffing.

Conventions
-----------
* Private helpers (leading underscore) are counted separately: they are held to
  a weaker standard than the public surface.
* Nested functions are not counted; only module-level functions and methods.
* ``__init__``/dunder methods are excluded from the "needs a docstring" set --
  a documented class covers them.

Usage::

    python tools/doc_audit.py
    python tools/doc_audit.py --json-only

Author: Manuel Schwenke / Claude Code (2026-07-31)
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Directories excluded from the audit: vendored, generated, or not source.
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", ".claude", "results", "data", ".idea", ".vscode",
}


def is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def analyse(path: Path) -> Optional[dict]:
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        return {"path": path, "syntax_error": f"line {exc.lineno}: {exc.msg}",
                "loc": src.count("\n") + 1}

    rec = {
        "path": path,
        "loc": src.count("\n") + 1,
        "module_doc": ast.get_docstring(tree) is not None,
        "module_doc_first": (ast.get_docstring(tree) or "").strip().splitlines()[:1],
        "functions": [], "classes": [], "syntax_error": None,
    }

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rec["functions"].append({
                "name": node.name,
                "line": node.lineno,
                "doc": ast.get_docstring(node) is not None,
                "private": node.name.startswith("_"),
            })
        elif isinstance(node, ast.ClassDef):
            methods = []
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if is_dunder(sub.name):
                        continue
                    methods.append({
                        "name": sub.name,
                        "line": sub.lineno,
                        "doc": ast.get_docstring(sub) is not None,
                        "private": sub.name.startswith("_"),
                    })
            rec["classes"].append({
                "name": node.name,
                "line": node.lineno,
                "doc": ast.get_docstring(node) is not None,
                "methods": methods,
            })
    return rec


def collect(root: Path) -> List[dict]:
    out = []
    for p in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        rec = analyse(p)
        if rec:
            rec["rel"] = p.relative_to(root).as_posix()
            rec["pkg"] = (p.relative_to(root).parent.as_posix() or ".")
            out.append(rec)
    return out


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 100.0


def summarise(records: List[dict]) -> dict:
    pkgs: Dict[str, dict] = defaultdict(lambda: {
        "modules": 0, "modules_doc": 0, "loc": 0,
        "pub_fn": 0, "pub_fn_doc": 0, "priv_fn": 0, "priv_fn_doc": 0,
        "cls": 0, "cls_doc": 0, "meth": 0, "meth_doc": 0,
        "files": [], "syntax_errors": [],
    })
    for r in records:
        s = pkgs[r["pkg"]]
        s["modules"] += 1
        s["loc"] += r["loc"]
        s["files"].append(r["rel"])
        if r["syntax_error"]:
            s["syntax_errors"].append(f"{r['rel']}: {r['syntax_error']}")
            continue
        s["modules_doc"] += int(r["module_doc"])
        for f in r["functions"]:
            key = "priv_fn" if f["private"] else "pub_fn"
            s[key] += 1
            s[key + "_doc"] += int(f["doc"])
        for c in r["classes"]:
            s["cls"] += 1
            s["cls_doc"] += int(c["doc"])
            for m in c["methods"]:
                s["meth"] += 1
                s["meth_doc"] += int(m["doc"])
    return pkgs


def undocumented(records: List[dict], public_only: bool = True) -> List[str]:
    out = []
    for r in records:
        if r["syntax_error"]:
            continue
        for f in r["functions"]:
            if not f["doc"] and (not public_only or not f["private"]):
                out.append(f"{r['rel']}:{f['line']}  {f['name']}()")
        for c in r["classes"]:
            if not c["doc"]:
                out.append(f"{r['rel']}:{c['line']}  class {c['name']}")
            for m in c["methods"]:
                if not m["doc"] and (not public_only or not m["private"]):
                    out.append(f"{r['rel']}:{m['line']}  {c['name']}.{m['name']}()")
    return out


def write_report(records: List[dict], pkgs: dict, out: Path) -> None:
    tot = {k: sum(p[k] for p in pkgs.values())
           for k in ("modules", "modules_doc", "loc", "pub_fn", "pub_fn_doc",
                     "priv_fn", "priv_fn_doc", "cls", "cls_doc", "meth",
                     "meth_doc")}
    lines = [
        "# Documentation coverage",
        "",
        "Generated by `tools/doc_audit.py` (static `ast` parse; nothing is",
        "imported). Regenerate with `python tools/doc_audit.py`.",
        "",
        "## Totals",
        "",
        f"- **{tot['modules']} modules**, {tot['loc']:,} lines",
        f"- module docstrings: **{tot['modules_doc']}/{tot['modules']}** "
        f"({pct(tot['modules_doc'], tot['modules']):.0f} %)",
        f"- public functions: **{tot['pub_fn_doc']}/{tot['pub_fn']}** "
        f"({pct(tot['pub_fn_doc'], tot['pub_fn']):.0f} %)",
        f"- private functions: {tot['priv_fn_doc']}/{tot['priv_fn']} "
        f"({pct(tot['priv_fn_doc'], tot['priv_fn']):.0f} %)",
        f"- classes: **{tot['cls_doc']}/{tot['cls']}** "
        f"({pct(tot['cls_doc'], tot['cls']):.0f} %)",
        f"- public methods: {tot['meth_doc']}/{tot['meth']} "
        f"({pct(tot['meth_doc'], tot['meth']):.0f} %)",
        "",
        "## Per package",
        "",
        "Sorted by public-API coverage, weakest first.",
        "",
        "| package | modules | mod-doc | pub fn | classes | pub meth | LOC |",
        "|---|---|---|---|---|---|---|",
    ]

    def score(p: dict) -> float:
        d = p["pub_fn"] + p["cls"] + p["meth"]
        n = p["pub_fn_doc"] + p["cls_doc"] + p["meth_doc"]
        return pct(n, d)

    for name, p in sorted(pkgs.items(), key=lambda kv: score(kv[1])):
        lines.append(
            f"| `{name}` | {p['modules']} | "
            f"{p['modules_doc']}/{p['modules']} | "
            f"{p['pub_fn_doc']}/{p['pub_fn']} | "
            f"{p['cls_doc']}/{p['cls']} | "
            f"{p['meth_doc']}/{p['meth']} | {p['loc']:,} |")

    errs = [e for p in pkgs.values() for e in p["syntax_errors"]]
    if errs:
        lines += ["", "## Files that do not parse", ""]
        lines += [f"- `{e}`" for e in errs]

    missing = undocumented(records, public_only=True)
    lines += ["", f"## Undocumented public API ({len(missing)})", ""]
    if missing:
        lines.append("```")
        lines += missing[:400]
        if len(missing) > 400:
            lines.append(f"... and {len(missing) - 400} more")
        lines.append("```")
    else:
        lines.append("None.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args(argv)

    records = collect(args.root)
    pkgs = summarise(records)

    payload = {
        "packages": {k: {kk: vv for kk, vv in v.items() if kk != "files"}
                     for k, v in pkgs.items()},
        "modules": [
            {"rel": r["rel"], "pkg": r["pkg"], "loc": r["loc"],
             "module_doc": r["module_doc"],
             "summary": (r["module_doc_first"] or [""])[0],
             "n_pub_fn": sum(1 for f in r["functions"] if not f["private"]),
             "n_cls": len(r["classes"]),
             "syntax_error": r["syntax_error"]}
            for r in records
        ],
    }
    jpath = args.root / "tools" / "doc_coverage.json"
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    if not args.json_only:
        rpath = args.root / "docs" / "architecture" / "doc_coverage.md"
        write_report(records, pkgs, rpath)
        print(f"wrote {rpath.relative_to(args.root)}")
    print(f"wrote {jpath.relative_to(args.root)}")
    print(f"{len(records)} modules across {len(pkgs)} packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
