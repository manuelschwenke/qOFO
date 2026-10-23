"""
experiments/thesis_runs.py
==========================

Result folders for the dissertation, one tree per thesis chapter, each run
documented by the run itself.

Layout
------
The results tree mirrors the experiments tree::

    experiments/<chapter>/[<sub>/]<script>.py
    results/<chapter>/[<sub>/]<NNNN>_<YYYY-MM-DD_HHMMSS>_<script>[_<tag>]/
        README.md           the record a reader opens
        run_manifest.json   the same, machine readable; README and header are
                            rendered from it
        latex_header.tex    the provenance block for the thesis heading
        config.json/.pkl    the resolved config, when one is passed
        code_state.diff     ``git diff HEAD`` when the code tree was dirty

``<chapter>`` is ``ch_NN_<description>`` (zero-padded, so ``ch_10`` sorts
after ``ch_09``) or ``app_<letter>_<description>``.  The results folder is
derived from the script's own path, so a script cannot write into another
chapter's tree by passing a wrong name.

Why the LaTeX label and not the chapter number is the key
---------------------------------------------------------
The thesis was renumbered on 2026-09-22, 2026-10-06 and 2026-10-23.  A folder
name carries the number of the day it was made; the label (``ch:case_svr``)
survived every renumbering.  The README therefore records the label as the
purpose and the number only "as of" the run date, read from the thesis
``.aux`` files.

Superseded runs
---------------
A re-run that replaces a result never deletes the old run.  ``archive_run``
moves it to ``_local_archive/results/<same relative path>`` -- the archive
mirrors original paths, as the rest of ``_local_archive/`` does -- marks it
``superseded`` and names the run that replaced it.

Usage
-----
::

    from experiments.thesis_runs import add_run_arguments, start_thesis_run

    add_run_arguments(parser)                     # --note, --tag
    ...
    with start_thesis_run(__file__, target="ch:case_svr",
                          feeds=["fig:case_svr:pareto"], config=cfg,
                          paramset="ch9_selected", note=args.note,
                          tag=args.tag) as run:
        ...                                       # write into run.root
        run.describe_output("metrics.csv", "per-variant metrics")

Leaving the ``with`` block normally sets the status ``complete``; an exception
sets ``failed`` and is re-raised.  ``resume=<run dir>`` appends an invocation
to an existing run instead of creating one (variants added later on the same
seeds).

Command line
------------
::

    python -m experiments.thesis_runs header  <run_dir>
    python -m experiments.thesis_runs archive <old_run_dir> --by <new_run_dir>
    python -m experiments.thesis_runs check   [--thesis <latex dir>]

``check`` reads every ``RESULT PROVENANCE`` block in the thesis chapters and
appendices and verifies that the run exists, that it is ``complete`` and not
superseded, and that every label it names is defined in the thesis.

Author: Manuel Schwenke / Claude Code (2026-10-23)
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import platform
import re
import shutil
import socket
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.helpers.run_params import config_fingerprint, sanitise
from experiments.paths import PROJECT_ROOT, RESULTS_ROOT

__all__ = [
    "TimeSeries", "ThesisRun", "add_run_arguments", "start_thesis_run",
    "experiment_folder", "latest_run", "annotate_run", "archive_run",
    "render_header",
    "check_thesis", "thesis_label_numbers",
]

EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

#: The dissertation repository.  ``QOFO_THESIS_DIR`` points elsewhere on a
#: machine where it lives under another path; without it, label numbers are
#: recorded as unresolved and ``check`` needs ``--thesis``.
THESIS_DIR = Path(os.environ.get(
    "QOFO_THESIS_DIR",
    r"C:\Users\Manuel Schwenke\Desktop\Daten\01_Forschung\12_Dissertation"
    r"\latex_diss_ms"))

CHAPTER_RE = re.compile(r"^(ch_\d{2}|app_[a-z])_[a-z0-9_]+$")
RUN_RE = re.compile(
    r"^(?P<counter>\d{4})_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{6})_(?P<producer>.+)$")

SCHEMA = 1
MANIFEST = "run_manifest.json"
README = "README.md"
HEADER = "latex_header.tex"
DIFF = "code_state.diff"

STATUSES = ("running", "complete", "failed", "superseded")

_DESC_PLACEHOLDER = ("_(free text: what this run is for, what to look at, "
                     "anything unusual)_")
_RULE = 79
_HEADER_TITLE = "RESULT PROVENANCE"
_MAX_LISTED_OUTPUTS = 200


# ---------------------------------------------------------------------------
#  Small pieces
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TimeSeries:
    """Profile window of a time-series run."""

    start: datetime
    end: datetime
    step: str = ""          # e.g. "TS 180 s, STS 20 s"

    def as_dict(self) -> dict:
        return {
            "kind": "window",
            "start": self.start.isoformat(timespec="minutes"),
            "end": self.end.isoformat(timespec="minutes"),
            "duration_min": round((self.end - self.start).total_seconds() / 60.0, 1),
            "step": self.step,
        }


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """``--note`` and ``--tag``, identical in every thesis entry point."""
    parser.add_argument(
        "--note", default="",
        help="free-text description, written to the run's README and header")
    parser.add_argument(
        "--tag", default=None,
        help="short suffix of the run folder name, e.g. K1 or smoke")


def _now() -> datetime:
    return datetime.now().astimezone()


def _iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds")


def _parse(t: str | None) -> datetime | None:
    return datetime.fromisoformat(t) if t else None


def _show(t: str | None) -> str:
    d = _parse(t)
    return d.strftime("%Y-%m-%d %H:%M:%S %z") if d else "--"


def _rel(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _git(*args: str) -> str | None:
    """``git`` in the project root; None when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=180)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _git_state(run_root: Path, stamp: str) -> dict:
    """Commit, branch and dirty flag; the diff is saved when dirty.

    Provenance must never stop a run, so every failure degrades to None.
    Untracked files are not part of the diff -- a new module that is not yet
    added to git is invisible here, which the README says.
    """
    commit = (_git("rev-parse", "HEAD") or "").strip() or None
    branch = (_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None
    status = _git("status", "--porcelain", "--untracked-files=no")
    dirty_files = None if status is None else [
        line[3:] for line in status.splitlines() if line.strip()]
    diff_file = None
    if dirty_files:
        diff = _git("diff", "HEAD")
        if diff:
            diff_file = DIFF if stamp == "" else f"code_state_{stamp}.diff"
            (run_root / diff_file).write_text(diff, encoding="utf-8")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if dirty_files is None else bool(dirty_files),
        "dirty_files": dirty_files,
        "diff_file": diff_file,
    }


def _command(script: Path) -> str:
    """The command line, as ``python -m <module>`` when that is what ran."""
    argv = list(sys.argv)
    args = subprocess.list2cmdline(argv[1:])
    try:
        if argv and Path(argv[0]).resolve() == script:
            module = ".".join(script.relative_to(PROJECT_ROOT.resolve())
                              .with_suffix("").parts)
            return f"python -m {module}" + (f" {args}" if args else "")
    except (ValueError, OSError):
        pass
    return "python " + subprocess.list2cmdline(argv)


def _paramset_info(name: str | None) -> dict | None:
    if not name:
        return None
    path = PROJECT_ROOT / "configs" / "paramsets" / f"{name}.json"
    info: dict[str, Any] = {"name": name, "file": _rel(path, PROJECT_ROOT)}
    try:
        blob = path.read_bytes()
    except OSError:
        info["sha256"] = None
        info["title"] = "file not found"
        return info
    info["sha256"] = hashlib.sha256(blob).hexdigest()[:12]
    try:
        info["title"] = json.loads(blob).get("title", "")
    except (ValueError, AttributeError):
        info["title"] = ""
    return info


def _time_series(spec: Any, config: Any) -> dict:
    """Normalise the time-series argument; derive it from the config if absent."""
    if isinstance(spec, TimeSeries):
        return spec.as_dict()
    if isinstance(spec, (tuple, list)) and len(spec) in (2, 3):
        return TimeSeries(*spec).as_dict()
    if isinstance(spec, str):
        return {"kind": "text", "text": spec}
    if spec is None and config is not None and getattr(config, "use_profiles", False):
        start = getattr(config, "start_time", None)
        total = getattr(config, "n_total_s", None)
        if isinstance(start, datetime) and total:
            steps = []
            if getattr(config, "tso_period_s", None):
                steps.append(f"TS {config.tso_period_s:g} s")
            if getattr(config, "dso_period_s", None):
                steps.append(f"STS {config.dso_period_s:g} s")
            return TimeSeries(start, start + timedelta(seconds=float(total)),
                              ", ".join(steps)).as_dict()
    return {"kind": "none",
            "text": "no profile time series (snapshot, or not applicable)"}


def _series_text(ts: Mapping) -> str:
    if ts.get("kind") == "window":
        start = ts["start"].replace("T", " ")
        end = ts["end"].replace("T", " ")
        text = f"{start} -- {end} ({ts['duration_min']:g} min)"
        return text + (f", {ts['step']}" if ts.get("step") else "")
    return str(ts.get("text", "--"))


_NEWLABEL_RE = re.compile(r"\\newlabel\{([^{}@]+)\}\{\{([^{}]*)\}")


def thesis_label_numbers(thesis_dir: Path = THESIS_DIR) -> dict[str, str]:
    """Label -> printed number, from the thesis ``.aux`` files (last build)."""
    numbers: dict[str, str] = {}
    thesis_dir = Path(thesis_dir)
    for pattern in ("*.aux", "Chapters/*.aux", "Appendices/*.aux"):
        for aux in sorted(thesis_dir.glob(pattern)):
            try:
                text = aux.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for label, number in _NEWLABEL_RE.findall(text):
                numbers.setdefault(label, number)
    return numbers


def experiment_folder(script: str | Path,
                      experiments_root: Path = EXPERIMENTS_ROOT) -> Path:
    """``<chapter>/[<sub>/]`` of a script, relative to ``experiments/``."""
    script = Path(script).resolve()
    try:
        rel = script.parent.relative_to(Path(experiments_root).resolve())
    except ValueError:
        raise ValueError(
            f"{script} is not inside {experiments_root}: thesis runs are "
            f"produced by scripts in a chapter folder") from None
    if not rel.parts or not CHAPTER_RE.match(rel.parts[0]):
        raise ValueError(
            f"{script.name}: its folder {rel.as_posix() or '.'!r} is not a "
            f"chapter folder (ch_NN_<description> or app_<x>_<description>)")
    return rel


def _archive_root(results_root: Path) -> Path:
    return Path(results_root).parent / "_local_archive" / "results"


def _new_run_folder(parent: Path, archive_parent: Path, producer: str,
                    tag: str | None, started: datetime) -> Path:
    """Next counter in ``parent``; archived runs keep their numbers taken."""
    counters = [
        int(m["counter"])
        for base in (parent, archive_parent) if base.is_dir()
        for d in base.iterdir()
        if d.is_dir() and (m := RUN_RE.match(d.name))
    ]
    counter = max(counters, default=0) + 1
    suffix = producer + (f"_{tag}" if tag else "")
    while True:
        root = parent / f"{counter:04d}_{started:%Y-%m-%d_%H%M%S}_{suffix}"
        try:
            root.mkdir(parents=True)
            return root
        except FileExistsError:
            counter += 1


# ---------------------------------------------------------------------------
#  The run
# ---------------------------------------------------------------------------


class ThesisRun:
    """One documented run folder.  Create it with :func:`start_thesis_run`."""

    def __init__(self, root: Path, manifest: dict) -> None:
        self.root = Path(root)
        self.manifest = manifest

    # -- access ----------------------------------------------------------
    @classmethod
    def load(cls, root: str | Path) -> "ThesisRun":
        root = Path(root)
        with (root / MANIFEST).open(encoding="utf-8") as fh:
            return cls(root, json.load(fh))

    @property
    def status(self) -> str:
        return self.manifest["status"]

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def describe_output(self, name: str, meaning: str) -> None:
        """One line of the README's output table; written on the next save."""
        self.manifest.setdefault("outputs", {})[name] = meaning

    def record_export(self, path: str | Path, by: str | Path) -> None:
        """Note a file written outside the run (a thesis figure, a .dat table).

        The thesis consumes these, so the run must say where its numbers went.
        """
        entry = {"path": str(path), "by": Path(by).name,
                 "on": _iso(_now())}
        exports = self.manifest.setdefault("exports", [])
        exports[:] = [e for e in exports if e["path"] != entry["path"]]
        exports.append(entry)

    # -- lifecycle -------------------------------------------------------
    def finish(self, status: str = "complete") -> None:
        if status not in STATUSES:
            raise ValueError(f"status {status!r}; valid: {STATUSES}")
        now = _now()
        inv = self.manifest["invocations"][-1]
        inv["finished"] = _iso(now)
        inv["status"] = status
        inv["wall_clock_s"] = round(
            (now - _parse(inv["started"])).total_seconds(), 1)
        self.manifest["finished"] = _iso(now)
        self.manifest["status"] = status
        self.write()

    def fail(self, reason: str) -> None:
        """Mark the run failed although no exception reached the ``with``.

        For drivers that catch a failing case to keep the batch alive: the
        folder must not say ``complete`` when one of its parts did not run.
        """
        self.manifest["invocations"][-1]["error"] = reason
        self._failed = True

    def __enter__(self) -> "ThesisRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.finish("failed" if getattr(self, "_failed", False) else "complete")
        else:
            self.manifest["invocations"][-1]["error"] = (
                f"{exc_type.__name__}: {exc}")
            self.finish("failed")
        return False

    # -- files -----------------------------------------------------------
    def _adopt_edited_description(self) -> None:
        """Keep a description the author typed into the README by hand."""
        path = self.root / README
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
        m = re.search(r"^## Description\n(.*?)(?=^## |\Z)", text,
                      flags=re.S | re.M)
        if not m:
            return
        edited = m.group(1).strip()
        if edited and edited != _DESC_PLACEHOLDER:
            self.manifest["description"] = edited

    def _list_outputs(self) -> list[tuple[str, str]]:
        described = self.manifest.get("outputs", {})
        own = {README, MANIFEST, HEADER}
        rows = []
        for entry in sorted(self.root.iterdir(), key=lambda p: p.name):
            if entry.name in own:
                continue
            if entry.is_dir():
                n = sum(1 for p in entry.rglob("*") if p.is_file())
                name = f"{entry.name}/"
                meaning = described.get(entry.name, described.get(name, ""))
                rows.append((name, meaning or f"{n} file(s)"))
            else:
                rows.append((entry.name, described.get(entry.name, "")))
            if len(rows) >= _MAX_LISTED_OUTPUTS:
                rows.append(("...", "listing truncated"))
                break
        return rows

    def write(self) -> None:
        """Rewrite manifest, README and LaTeX header from the manifest."""
        self._adopt_edited_description()
        self.manifest["outputs_listed"] = self._list_outputs()
        with (self.root / MANIFEST).open("w", encoding="utf-8") as fh:
            json.dump(self.manifest, fh, indent=2, ensure_ascii=False)
        (self.root / README).write_text(render_readme(self.manifest),
                                        encoding="utf-8")
        (self.root / HEADER).write_text(render_header(self.manifest) + "\n",
                                        encoding="utf-8")


def _invocation(script: Path, root: Path, started: datetime, first: bool) -> dict:
    return {
        "started": _iso(started),
        "finished": None,
        "status": "running",
        "command": _command(script),
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "git": _git_state(root, "" if first else f"{started:%Y-%m-%d_%H%M%S}"),
    }


def start_thesis_run(
    script: str | Path,
    *,
    target: str | Sequence[str],
    feeds: Iterable[str] = (),
    config: Any = None,
    paramset: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    time_series: Any = None,
    note: str = "",
    tag: str | None = None,
    resume: str | Path | None = None,
    results_root: str | Path | None = None,
    experiments_root: str | Path | None = None,
    thesis_dir: str | Path | None = None,
) -> ThesisRun:
    """Create (or resume) the documented run folder of a thesis entry point.

    Parameters
    ----------
    script      : ``__file__`` of the entry point; fixes the results folder.
    target      : the LaTeX label(s) the result is reported under -- the
                  chapter, section or subsection that carries its header.
    feeds       : figure, table and equation labels the run supplies numbers to.
    config      : the config passed to the runner; saved, fingerprinted, and
                  read for the time-series window when ``time_series`` is None.
    paramset    : name in ``configs/paramsets`` the config was built from.
    parameters  : the few values a reader needs to know (variants, N, seeds,
                  overrides); shown in the README table and the header.
    time_series : ``TimeSeries``, ``(start, end[, step])``, a free-text string
                  (e.g. "ensemble of 100 episodes, see episodes.csv"), or None.
    note        : free-text description (``--note``).
    tag         : suffix of the folder name (``--tag``).
    resume      : an existing run folder to append an invocation to.
    """
    results_root = Path(results_root or RESULTS_ROOT).resolve()
    experiments_root = Path(experiments_root or EXPERIMENTS_ROOT)
    project_root = results_root.parent
    script = Path(script).resolve()
    folder = experiment_folder(script, experiments_root)
    targets = [target] if isinstance(target, str) else list(target)
    if not targets or not all(targets):
        raise ValueError("target: name at least one LaTeX label")
    started = _now()

    if resume is not None:
        run = ThesisRun.load(resume)
        run.manifest["invocations"].append(
            _invocation(script, run.root, started, first=False))
        run.manifest["status"] = "running"
        run.manifest["finished"] = None
        if note:
            run.manifest["description"] = (
                run.manifest.get("description", "") + "\n\n" + note).strip()
        run.write()
        return run

    root = _new_run_folder(results_root / folder,
                           _archive_root(results_root) / folder,
                           script.stem, tag, started)
    fingerprint = None
    config_files: list[str] = []
    if config is not None:
        # ``run_params`` ignores ``results_dir`` but not ``result_dir``, the
        # field's actual name, so the run folder would make every fingerprint
        # unique.  Hash a copy with the field neutralised.
        neutral = config
        if dataclasses.is_dataclass(config) and hasattr(config, "result_dir"):
            neutral = dataclasses.replace(config, result_dir="")
        fingerprint = config_fingerprint(neutral)
        if hasattr(config, "result_dir"):
            try:
                config.result_dir = str(root)
            except (AttributeError, dataclasses.FrozenInstanceError):
                pass
        with (root / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(sanitise(config), fh, indent=2, ensure_ascii=False)
        config_files.append("config.json")
        try:
            with (root / "config.pkl").open("wb") as fh:
                pickle.dump(config, fh, protocol=pickle.HIGHEST_PROTOCOL)
            config_files.append("config.pkl")
        except Exception:  # an unpicklable config still has its JSON
            (root / "config.pkl").unlink(missing_ok=True)

    numbers = thesis_label_numbers(Path(thesis_dir or THESIS_DIR))
    run = ThesisRun(root, {
        "schema": SCHEMA,
        "project": project_root.name,
        "experiment": folder.as_posix(),
        "producer": script.stem,
        "tag": tag,
        "script": _rel(script, project_root),
        "run": _rel(root, project_root),
        "save_location": str(root.resolve()),
        "target": targets,
        "target_numbers": {
            label: numbers.get(label, "unresolved") for label in targets},
        "numbers_as_of": started.date().isoformat(),
        "feeds": list(feeds),
        "status": "running",
        "started": _iso(started),
        "finished": None,
        "invocations": [_invocation(script, root, started, first=True)],
        "paramset": _paramset_info(paramset),
        "config_fingerprint": fingerprint,
        "config_files": config_files,
        "parameters": sanitise(dict(parameters or {})),
        "time_series": _time_series(time_series, config),
        "description": note.strip(),
        "outputs": {},
        "superseded_by": None,
        "archived": None,
    })
    run.write()
    return run


# ---------------------------------------------------------------------------
#  Rendering
# ---------------------------------------------------------------------------


def _code_text(git: Mapping, long: bool) -> str:
    commit = git.get("commit") or "unavailable"
    text = commit if long else commit[:7]
    if git.get("branch"):
        text += f" on {git['branch']}"
    if git.get("dirty"):
        n = len(git.get("dirty_files") or [])
        text += f", dirty ({n} file(s), see {git.get('diff_file') or 'no diff'})"
    elif git.get("dirty") is False:
        text += ", clean"
    return text


def _params_text(m: Mapping) -> str:
    parts = []
    ps = m.get("paramset")
    if ps:
        parts.append(f"paramset {ps['name']}"
                     + (f" (sha256 {ps['sha256']})" if ps.get("sha256") else ""))
    if m.get("config_fingerprint"):
        parts.append(f"config fingerprint {m['config_fingerprint']}")
    for key, value in (m.get("parameters") or {}).items():
        parts.append(f"{key} = {json.dumps(value, ensure_ascii=False)}")
    return "; ".join(parts) or "--"


def _date_text(m: Mapping) -> str:
    start, end = _parse(m.get("started")), _parse(m.get("finished"))
    if start is None:
        return "--"
    if end is None:
        return f"{start:%Y-%m-%d %H:%M %z} (not finished)"
    fmt = "%H:%M" if end.date() == start.date() else "%Y-%m-%d %H:%M"
    return f"{start:%Y-%m-%d %H:%M} -- {end:{fmt}} {start:%z}"


def render_header(m: Mapping) -> str:
    """The thesis comment block; identical structure for every run."""
    project = m.get("project", PROJECT_ROOT.name)
    inv = m["invocations"][-1]
    rows = [
        ("Target", ", ".join(m["target"])),
        ("Feeds", ", ".join(m.get("feeds") or []) or "--"),
        ("Script", f"{project}/{m['script']}"),
        ("Command", inv["command"]),
        ("Run", f"{project}/{m['run']}/"),
        ("Date", _date_text(m)),
        ("Code", f"{project} " + _code_text(inv["git"], long=False)),
        ("Params", _params_text(m)),
        ("Series", _series_text(m["time_series"])),
        ("Comment", (m.get("description") or "").split("\n\n")[0]),
    ]
    width = max(len(k) for k, _ in rows)
    lead = f"% {'':<{width}}   "
    lines = [f"%=== {_HEADER_TITLE} ".ljust(_RULE, "=")]
    for key, value in rows:
        first = f"% {key:<{width}} : "
        wrapped = textwrap.wrap(" ".join(str(value).split()), _RULE - len(first),
                                break_long_words=False, break_on_hyphens=False)
        lines.append((first + wrapped[0]) if wrapped else first.rstrip())
        lines.extend(lead + w for w in wrapped[1:])
    lines.append("%" + "=" * (_RULE - 1))
    return "\n".join(lines)


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_readme(m: Mapping) -> str:
    inv = m["invocations"][-1]
    numbers = m.get("target_numbers") or {}
    purpose = ", ".join(
        f"`{label}` ({_number_text(label, numbers.get(label))} on "
        f"{m.get('numbers_as_of', '?')})" for label in m["target"])
    status = f"**{m['status']}**"
    if m.get("superseded_by"):
        status += f", superseded by `{m['superseded_by']}`"
    if m.get("archived"):
        a = m["archived"]
        status += f"; archived from `{a['from']}` on {a['on'][:10]}"
    commits = {i["git"].get("commit") for i in m["invocations"]}
    table = [
        ("Status", status),
        ("Purpose in dissertation", purpose),
        ("Feeds", ", ".join(f"`{f}`" for f in m.get("feeds") or []) or "--"),
        ("Date of simulation", f"{_show(m.get('started'))} -> "
                               f"{_show(m.get('finished'))}"),
        ("Save location", f"`{m['save_location']}`"),
        ("Script", f"`{m['script']}`"),
        ("Command", f"`{inv['command']}`"),
        ("Code", _code_text(inv["git"], long=True)
         + ("" if len(commits) <= 1 else
            " -- **invocations ran on different commits, see below**")),
        ("Parameter set", (f"`{m['paramset']['name']}` (`{m['paramset']['file']}`, "
                           f"sha256 `{m['paramset']['sha256']}`)")
         if m.get("paramset") else "--"),
        ("Config fingerprint", f"`{m['config_fingerprint']}` "
                               f"(`experiments/helpers/run_params.py`)"
         if m.get("config_fingerprint") else "--"),
        ("Time series", _series_text(m["time_series"])),
        ("Host / Python", f"{inv['host']}, Python {inv['python']}"
                          + (f" (`{inv['conda_env']}`)" if inv.get("conda_env") else "")),
    ]
    out = [f"# {m['experiment']} -- run {m['run'].rsplit('/', 1)[-1]}", ""]
    out.append("Written by `experiments/thesis_runs.py` from `run_manifest.json`. "
               "Edit only the Description section; the rest is rewritten.")
    out += ["", "| Field | Value |", "|---|---|"]
    out += [f"| {k} | {_md(v)} |" for k, v in table]
    if m.get("parameters"):
        out += ["", "## Parameters", "", "| Name | Value |", "|---|---|"]
        out += [f"| `{k}` | `{_md(json.dumps(v, ensure_ascii=False))}` |"
                for k, v in m["parameters"].items()]
    out += ["", "## Description", "",
            (m.get("description") or "").strip() or _DESC_PLACEHOLDER, ""]
    out += ["## Outputs", "", "| File | Content |", "|---|---|"]
    out += [f"| `{name}` | {_md(meaning)} |"
            for name, meaning in m.get("outputs_listed") or []]
    if not m.get("outputs_listed"):
        out.append("| -- | nothing written yet |")
    if m.get("exports"):
        out += ["", "## Exported outside the run", "",
                "| Path | Written by | On |", "|---|---|---|"]
        out += [f"| `{_md(e['path'])}` | `{e['by']}` | {e['on'][:16].replace('T', ' ')} |"
                for e in m["exports"]]
    if len(m["invocations"]) > 1:
        out += ["", "## Invocations", "",
                "| Started | Finished | Status | Commit | Command |",
                "|---|---|---|---|---|"]
        out += [f"| {_show(i['started'])} | {_show(i.get('finished'))} | "
                f"{i['status']} | `{(i['git'].get('commit') or '?')[:10]}` | "
                f"`{_md(i['command'])}` |" for i in m["invocations"]]
    errors = [i["error"] for i in m["invocations"] if i.get("error")]
    if errors:
        out += ["", "## Errors", ""] + [f"- `{_md(e)}`" for e in errors]
    out += ["", "## LaTeX provenance block", "",
            "Paste under the heading of the chapter, section or subsection "
            "named in *Purpose* (also in `latex_header.tex`):", "",
            "```latex", render_header(m), "```", ""]
    if any(i["git"].get("dirty") for i in m["invocations"]):
        out += ["Untracked files are not part of the saved diff.", ""]
    return "\n".join(out)


def _number_text(label: str, number: str | None) -> str:
    if not number or number == "unresolved":
        return "number unresolved"
    return f"{'Sec.' if '.' in number else 'Ch'} {number}"


# ---------------------------------------------------------------------------
#  Finding and archiving runs
# ---------------------------------------------------------------------------


_ANY = object()


def latest_run(script_or_folder: str | Path, *, producer: str | None = None,
               tag: Any = _ANY, require_complete: bool = True,
               results_root: str | Path | None = None,
               experiments_root: str | Path | None = None) -> Path:
    """Newest run of a results folder, optionally of one producing script.

    ``script_or_folder`` is a script path (its own folder is searched) or a
    folder relative to ``results/`` such as ``"ch_10_case1_svr"``.  ``tag``
    restricts to one folder tag (None = untagged); by default any tag matches.
    """
    results_root = Path(results_root or RESULTS_ROOT)
    p = Path(script_or_folder)
    if p.suffix == ".py":
        folder = experiment_folder(p, Path(experiments_root or EXPERIMENTS_ROOT))
    else:
        folder = p
    parent = results_root / folder
    best: tuple[int, Path] | None = None
    for d in parent.iterdir() if parent.is_dir() else ():
        m = RUN_RE.match(d.name)
        if not (d.is_dir() and m):
            continue
        try:
            manifest = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = None
        if producer and (manifest or {}).get("producer", "") != producer:
            continue
        if tag is not _ANY and (manifest or {}).get("tag") != tag:
            continue
        if require_complete and (manifest or {}).get("status") != "complete":
            continue
        if best is None or int(m["counter"]) > best[0]:
            best = (int(m["counter"]), d)
    if best is None:
        raise FileNotFoundError(
            f"no {'complete ' if require_complete else ''}run"
            f"{f' of {producer}' if producer else ''} in {parent}")
    return best[1]


def annotate_run(run_dir: str | Path, *, by: str | Path,
                 outputs: Mapping[str, str] | None = None,
                 exports: Iterable[str | Path] = ()) -> bool:
    """Record what a post-processing script added to, or exported from, a run.

    Report and figure scripts read a finished run and write next to it or into
    the thesis.  This keeps the run's README true afterwards.  A legacy run
    folder without a manifest is left alone; returns whether one was updated.
    """
    if not (Path(run_dir) / MANIFEST).exists():
        return False
    run = ThesisRun.load(run_dir)
    for name, meaning in (outputs or {}).items():
        run.describe_output(name, f"{meaning} (by {Path(by).name})")
    for path in exports:
        run.record_export(path, by)
    run.write()
    return True


def archive_run(run_dir: str | Path, *, by: str | Path | None = None,
                results_root: str | Path | None = None) -> Path:
    """Move a superseded run to ``_local_archive/results/<same path>``."""
    results_root = Path(results_root or RESULTS_ROOT).resolve()
    run_dir = Path(run_dir).resolve()
    try:
        rel = run_dir.relative_to(results_root)
    except ValueError:
        raise ValueError(f"{run_dir} is not inside {results_root}") from None
    dest = _archive_root(results_root) / rel
    if dest.exists():
        raise FileExistsError(f"{dest} exists already")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(dest))
    project_root = results_root.parent
    by_rel = _rel(Path(by), project_root) if by else None
    archived = {"on": _iso(_now()), "from": _rel(run_dir, project_root),
                "to": _rel(dest, project_root)}
    if (dest / MANIFEST).exists():
        run = ThesisRun.load(dest)
        run.manifest.update(status="superseded", superseded_by=by_rel,
                            archived=archived)
        run.write()
    else:
        (dest / "ARCHIVED.md").write_text(
            f"# Archived\n\nMoved from `{archived['from']}` on "
            f"{archived['on'][:10]}"
            + (f", superseded by `{by_rel}`" if by_rel else "") + ".\n",
            encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
#  Thesis check
# ---------------------------------------------------------------------------


_BLOCK_OPEN = re.compile(rf"^%=== {_HEADER_TITLE} =*\s*$")
_BLOCK_CLOSE = re.compile(r"^%={10,}\s*$")
_FIELD = re.compile(r"^% (\w+)\s*: ?(.*)$")
_CONT = re.compile(r"^%\s{4,}(\S.*)$")
_LABEL_DEF = re.compile(r"\\label\{([^}]+)\}")
#: A label-shaped token inside a Target/Feeds field; other words are prose.
_LABEL_REF = re.compile(r"\b[A-Za-z]+:[\w:.\-]*\w")


def _thesis_sources(thesis_dir: Path) -> list[Path]:
    files = [thesis_dir / "ClassicThesis.tex"]
    for sub in ("Chapters", "Appendices", "FrontBackmatter"):
        files += sorted((thesis_dir / sub).glob("*.tex"))
    return [f for f in files if f.is_file()]


def parse_blocks(path: Path) -> list[dict]:
    """Every provenance block of one ``.tex`` file, with its line number."""
    blocks, current, key = [], None, None
    for n, line in enumerate(path.read_text(encoding="utf-8",
                                            errors="replace").splitlines(), 1):
        if _BLOCK_OPEN.match(line):
            current, key = {"_file": path, "_line": n}, None
        elif current is not None and _BLOCK_CLOSE.match(line):
            blocks.append(current)
            current = None
        elif current is not None:
            if (m := _FIELD.match(line)):
                key = m.group(1)
                current[key] = m.group(2).strip()
            elif key and (m := _CONT.match(line)):
                current[key] += " " + m.group(1).strip()
    return blocks


def check_thesis(thesis_dir: str | Path = THESIS_DIR,
                 project_root: Path = PROJECT_ROOT) -> int:
    """Print one verdict per provenance block; return the number of failures."""
    thesis_dir = Path(thesis_dir)
    sources = _thesis_sources(thesis_dir)
    if not sources:
        print(f"no thesis sources under {thesis_dir}")
        return 1
    labels = set()
    for f in sources:
        labels.update(_LABEL_DEF.findall(f.read_text(encoding="utf-8",
                                                     errors="replace")))
    blocks = [b for f in sources for b in parse_blocks(f)]
    failures = warnings = 0
    for b in blocks:
        where = f"{_rel(b['_file'], thesis_dir)}:{b['_line']}"
        problems, notes = [], []
        for field in ("Target", "Feeds"):
            for label in _LABEL_REF.findall(b.get(field, "")):
                if label not in labels:
                    problems.append(f"{field} label {label!r} is not defined")
        run_text = b.get("Run", "").rstrip("/")
        if not run_text:
            problems.append("no Run field")
        else:
            run_path = project_root.parent / run_text
            own = run_text.split("/", 1)[0] == project_root.name
            if not run_path.exists():
                archived = None
                if own and run_text.split("/", 2)[1:2] == ["results"]:
                    archived = (project_root / "_local_archive"
                                / run_text.split("/", 1)[1])
                if archived is not None and archived.exists():
                    problems.append(f"run was superseded and archived to "
                                    f"{_rel(archived, project_root.parent)}")
                else:
                    problems.append(f"run folder {run_text} not found")
            elif (run_path / MANIFEST).exists():
                m = json.loads((run_path / MANIFEST).read_text(encoding="utf-8"))
                if m.get("status") != "complete":
                    problems.append(f"run status is {m.get('status')!r}")
                if m.get("superseded_by"):
                    problems.append(f"superseded by {m['superseded_by']}")
            else:
                notes.append("legacy run (no run_manifest.json)")
            if not own:
                notes.append(f"lives outside {project_root.name}")
        verdict = "FAIL" if problems else ("WARN" if notes else "OK")
        failures += bool(problems)
        warnings += bool(notes) and not problems
        print(f"[{verdict:4}] {where}  {b.get('Target', '?')}  ->  {run_text or '?'}")
        for text in problems + notes:
            print(f"         - {text}")
    print(f"\n{len(blocks)} block(s): {failures} failing, {warnings} with warnings")
    return failures


# ---------------------------------------------------------------------------
#  Command line
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m experiments.thesis_runs",
                                 description=__doc__.split("\n\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("header", help="print the LaTeX block of a run")
    h.add_argument("run_dir")
    a = sub.add_parser("archive", help="move a superseded run to the archive")
    a.add_argument("run_dir")
    a.add_argument("--by", default=None, help="the run that replaces it")
    c = sub.add_parser("check", help="verify the provenance blocks of the thesis")
    c.add_argument("--thesis", default=str(THESIS_DIR))
    args = ap.parse_args(argv)

    if args.cmd == "header":
        print(render_header(ThesisRun.load(args.run_dir).manifest))
        return 0
    if args.cmd == "archive":
        print(archive_run(args.run_dir, by=args.by))
        return 0
    return 1 if check_thesis(Path(args.thesis)) else 0


if __name__ == "__main__":
    sys.exit(main())
