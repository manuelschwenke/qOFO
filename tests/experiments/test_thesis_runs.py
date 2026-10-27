"""Tests for experiments/thesis_runs.py -- chapter-mirrored, self-documenting runs."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

import pytest

from experiments import thesis_runs as tr


@dataclasses.dataclass
class _Cfg:
    g_q: float = 200.0
    use_profiles: bool = True
    start_time: datetime = datetime(2016, 1, 5, 8, 0)
    n_total_s: float = 3600.0
    tso_period_s: float = 180.0
    dso_period_s: float = 20.0
    result_dir: str = ""


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A throw-away project: experiments/, results/, a thesis, and fake git."""
    root = tmp_path / "proj"
    script = root / "experiments" / "ch_10_demo" / "sub" / "run_demo.py"
    script.parent.mkdir(parents=True)
    script.write_text("# demo\n", encoding="utf-8")
    (root / "results").mkdir()
    thesis = tmp_path / "thesis"
    (thesis / "Chapters").mkdir(parents=True)
    (thesis / "Chapters" / "Chapter10.aux").write_text(
        "\\newlabel{ch:demo}{{10}{123}{Demo}{chapter.10}{}}\n"
        "\\newlabel{ch:demo@cref}{{[chapter][10][]10}{[1][123][]123}}\n",
        encoding="utf-8")

    def fake_git(*args):
        if args[:2] == ("rev-parse", "HEAD"):
            return "0123456789abcdef\n"
        if args[:2] == ("rev-parse", "--abbrev-ref"):
            return "main\n"
        if args[0] == "status":
            return " M core/x.py\n"
        if args[0] == "diff":
            return "diff --git a/core/x.py b/core/x.py\n"
        return None

    monkeypatch.setattr(tr, "_git", fake_git)
    return {"root": root, "script": script, "thesis": thesis}


def _start(p, **kw):
    kw.setdefault("target", "ch:demo")
    return tr.start_thesis_run(
        p["script"], results_root=p["root"] / "results",
        experiments_root=p["root"] / "experiments", thesis_dir=p["thesis"], **kw)


def test_run_folder_mirrors_script_and_documents_itself(project):
    with _start(project, feeds=["fig:demo"], config=_Cfg(),
                paramset="does_not_exist", parameters={"variants": ["A", "B"]},
                note="first look", tag="smoke") as run:
        (run.root / "metrics.csv").write_text("a,b\n", encoding="utf-8")
        run.describe_output("metrics.csv", "per-variant metrics")

    rel = run.root.relative_to(project["root"] / "results")
    assert rel.parts[:2] == ("ch_10_demo", "sub")
    assert rel.parts[2].startswith("0001_") and rel.parts[2].endswith("_run_demo_smoke")
    m = json.loads((run.root / tr.MANIFEST).read_text(encoding="utf-8"))
    assert m["status"] == "complete"
    assert m["script"] == "experiments/ch_10_demo/sub/run_demo.py"
    assert m["target_numbers"] == {"ch:demo": "10"}
    assert m["time_series"]["start"] == "2016-01-05T08:00"
    assert m["time_series"]["end"] == "2016-01-05T09:00"
    assert m["invocations"][0]["git"]["dirty"] is True
    assert (run.root / tr.DIFF).exists()
    readme = (run.root / tr.README).read_text(encoding="utf-8")
    for needle in ("Purpose in dissertation", "`ch:demo` (Ch 10", "first look",
                   "per-variant metrics", "Save location", "2016-01-05 08:00"):
        assert needle in readme
    header = (run.root / tr.HEADER).read_text(encoding="utf-8")
    assert header.startswith("%=== RESULT PROVENANCE ")
    assert "% Run     : proj/results/ch_10_demo/sub/" in header
    assert all(line.startswith("%") for line in header.splitlines())


def test_script_outside_a_chapter_folder_is_refused(project):
    stray = project["root"] / "experiments" / "misc" / "x.py"
    stray.parent.mkdir()
    stray.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="not a chapter folder"):
        tr.start_thesis_run(stray, target="ch:demo",
                            results_root=project["root"] / "results",
                            experiments_root=project["root"] / "experiments")


def test_exception_marks_the_run_failed(project):
    with pytest.raises(RuntimeError):
        with _start(project) as run:
            raise RuntimeError("solver died")
    m = json.loads((run.root / tr.MANIFEST).read_text(encoding="utf-8"))
    assert m["status"] == "failed"
    assert "solver died" in m["invocations"][-1]["error"]


def test_fail_without_exception_marks_the_run_failed(project):
    with _start(project) as run:
        run.fail("variant(s) failed: O2")
    m = json.loads((run.root / tr.MANIFEST).read_text(encoding="utf-8"))
    assert m["status"] == "failed"
    assert "O2" in m["invocations"][-1]["error"]


def test_fingerprint_ignores_the_run_folder(project):
    a = _start(project, config=_Cfg())
    b = _start(project, config=_Cfg())
    assert a.root != b.root
    assert a.manifest["config_fingerprint"] == b.manifest["config_fingerprint"]
    assert _start(project, config=_Cfg(g_q=250.0)).manifest["config_fingerprint"] \
        != a.manifest["config_fingerprint"]


def test_description_typed_into_readme_survives_a_rewrite(project):
    run = _start(project)
    path = run.root / tr.README
    text = path.read_text(encoding="utf-8").replace(
        tr._DESC_PLACEHOLDER, "Author's own words.")
    path.write_text(text, encoding="utf-8")
    run.finish()
    assert run.manifest["description"] == "Author's own words."
    assert "Author's own words." in path.read_text(encoding="utf-8")


def test_resume_appends_an_invocation(project):
    with _start(project) as run:
        pass
    with tr.start_thesis_run(project["script"], target="ch:demo",
                             resume=run.root,
                             experiments_root=project["root"] / "experiments",
                             results_root=project["root"] / "results",
                             note="added variant C") as again:
        pass
    assert again.root == run.root
    assert [i["status"] for i in again.manifest["invocations"]] == ["complete"] * 2
    assert "added variant C" in again.manifest["description"]


def test_archive_moves_marks_and_keeps_the_counter(project):
    with _start(project) as old:
        pass
    with _start(project) as new:
        pass
    dest = tr.archive_run(old.root, by=new.root,
                          results_root=project["root"] / "results")
    assert not old.root.exists()
    assert dest == (project["root"] / "_local_archive" / "results"
                    / old.root.relative_to(project["root"] / "results"))
    m = json.loads((dest / tr.MANIFEST).read_text(encoding="utf-8"))
    assert m["status"] == "superseded"
    assert m["superseded_by"].endswith(new.root.name)
    with _start(project) as third:
        pass
    assert third.root.name.startswith("0003_")


def test_latest_run_filters_status_and_producer(project):
    with _start(project) as done:
        pass
    _start(project)                                  # still "running"
    found = tr.latest_run("ch_10_demo/sub", producer="run_demo",
                          results_root=project["root"] / "results")
    assert found == done.root
    with pytest.raises(FileNotFoundError):
        tr.latest_run("ch_10_demo/sub", producer="other",
                      results_root=project["root"] / "results")


def test_check_accepts_a_pasted_header_and_flags_an_archived_run(project, capsys):
    with _start(project, feeds=["fig:demo"]) as run:
        pass
    header = (run.root / tr.HEADER).read_text(encoding="utf-8")
    chapter = project["thesis"] / "Chapters" / "Chapter10.tex"
    chapter.write_text(
        "\\chapter{Demo}\\label{ch:demo}\n" + header
        + "\\begin{figure}\\label{fig:demo}\\end{figure}\n", encoding="utf-8")
    assert tr.check_thesis(project["thesis"], project_root=project["root"]) == 0
    assert "[OK  ]" in capsys.readouterr().out

    tr.archive_run(run.root, results_root=project["root"] / "results")
    assert tr.check_thesis(project["thesis"], project_root=project["root"]) == 1
    assert "superseded and archived" in capsys.readouterr().out


def test_check_flags_an_undefined_label(project, capsys):
    with _start(project, target="ch:missing") as run:
        pass
    (project["thesis"] / "Chapters" / "Chapter10.tex").write_text(
        "\\chapter{Demo}\\label{ch:demo}\n"
        + (run.root / tr.HEADER).read_text(encoding="utf-8"), encoding="utf-8")
    assert tr.check_thesis(project["thesis"], project_root=project["root"]) == 1
    assert "'ch:missing' is not defined" in capsys.readouterr().out


def _campaign(project, name="THESIS_demo_campaign"):
    src = project["root"] / "results" / name
    (src / "code_state").mkdir(parents=True)
    (src / "driver.py").write_text("# driver\n", encoding="utf-8")
    (src / "code_state" / "tracked_changes.patch").write_text("diff\n", encoding="utf-8")
    return src


_GIT = {"commit": "9d03c34af10f", "branch": "main", "dirty": True,
        "dirty_files": ["core/x.py"], "diff_file": "code_state/tracked_changes.patch"}


def _adopt(project, src, **kw):
    kw.setdefault("invocations", [
        {"started": "2026-10-25T21:54:33+01:00", "finished": "2026-10-27T11:58:15+01:00",
         "status": "complete", "command": "python driver.py", "git": _GIT}])
    return tr.adopt_run(
        src, experiment="ch_10_demo", producer="demo_driver",
        script="results/ch_10_demo/x/driver.py", target="ch:demo",
        results_root=project["root"] / "results", thesis_dir=project["thesis"], **kw)


def test_adopt_moves_a_campaign_and_documents_it_from_its_own_records(project):
    src = _campaign(project)
    run = _adopt(project, src, feeds=["fig:demo"], tag="band",
                 outputs={"driver.py": "the campaign driver"})
    assert not src.exists()
    assert run.root.parent == project["root"] / "results" / "ch_10_demo"
    assert run.root.name == "0001_2026-10-25_215433_demo_driver_band"
    assert (run.root / "driver.py").exists()
    m = json.loads((run.root / tr.MANIFEST).read_text(encoding="utf-8"))
    assert m["status"] == "complete"
    assert m["adopted"]["from"] == "results/THESIS_demo_campaign"
    assert m["invocations"][0]["git"]["commit"] == "9d03c34af10f"   # not fake_git's
    assert m["target_numbers"] == {"ch:demo": "10"}
    assert "adopted from `results/THESIS_demo_campaign`" in (
        run.root / tr.README).read_text(encoding="utf-8")
    header = (run.root / tr.HEADER).read_text(encoding="utf-8")
    assert "Code    : proj 9d03c34 on main, dirty" in header


def test_adopt_continues_the_counter_and_refuses_bad_input(project):
    (project["root"] / "results" / "ch_10_demo" / "0004_2026-01-01_000000_x").mkdir(parents=True)
    run = _adopt(project, _campaign(project))
    assert run.root.name.startswith("0005_")
    with pytest.raises(ValueError, match="already a documented run"):
        _adopt(project, run.root)
    with pytest.raises(ValueError, match="not a chapter folder"):
        tr.adopt_run(_campaign(project, "THESIS_b"), experiment="ch_9_demo",
                     producer="p", script="s", target="ch:demo",
                     invocations=[{"started": "2026-10-25T21:54:33", "finished": None,
                                   "status": "complete", "command": "c", "git": _GIT}],
                     results_root=project["root"] / "results")
    with pytest.raises(ValueError, match="lacks"):
        _adopt(project, project["root"] / "results" / "THESIS_b",
               invocations=[{"started": "2026-10-25T21:54:33"}])


def test_adopt_with_a_failed_invocation_is_failed(project):
    inv = {"started": "2026-10-25T21:54:33", "finished": "2026-10-26T10:00:00",
           "command": "c", "git": _GIT}
    run = _adopt(project, _campaign(project), invocations=[
        dict(inv, status="complete"), dict(inv, status="failed")])
    assert run.status == "failed"


def test_adopt_header_names_the_primary_invocation(project):
    base = {"finished": "2026-10-27T11:58:15", "status": "complete"}
    run = _adopt(project, _campaign(project), invocations=[
        dict(base, started="2026-10-25T21:54:33", command="python driver.py", git=_GIT),
        dict(base, started="2026-10-26T17:18:04", command="python followup.py",
             git=dict(_GIT, commit="d9633fe12700"))])
    header = (run.root / tr.HEADER).read_text(encoding="utf-8")
    assert "Command : python driver.py" in header and "9d03c34" in header
    assert "different commits" in (run.root / tr.README).read_text(encoding="utf-8")
