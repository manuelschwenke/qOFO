"""merge_mc: runs of a campaign share simulated elsewhere join the campaign of
record without overwriting it; a seed whose prescreen differs is left out."""
import json

from experiments.ch_11_case2_multisystem.mc import merge_mc as G


def _w(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d), encoding="utf-8")


def _pre(seed, valid=True, off=(4,)):
    return {"seed": seed, "valid": valid, "start_time": "2016-03-01T10:00:00",
            "events": [], "uc_offline": list(off)}


def test_merge_copies_missing_runs_and_keeps_existing_ones(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    for s in (1, 2, 3):
        _w(src / "prescreen" / f"{s}.json", _pre(s))
        _w(dst / "prescreen" / f"{s}.json", _pre(s, off=(4, 7) if s == 3 else (4,)))
        _w(src / "runs" / f"{s}_S1.json", {"seed": s, "variant": "S1", "outcome": "ok"})
        (src / "runs" / f"{s}_S1.npz").write_bytes(b"x")
    _w(src / "runs" / "1_M.json", {"seed": 1, "variant": "M", "outcome": "task_crash"})
    _w(dst / "runs" / "2_S1.json", {"seed": 2, "variant": "S1", "outcome": "fail_pf"})
    res = G.merge(src, dst)
    assert res["prescreen_mismatch"] == [3]
    assert res["runs_copied"] == 1 and (dst / "runs" / "1_S1.npz").exists()
    assert json.loads((dst / "runs" / "2_S1.json").read_text())["outcome"] == "fail_pf"
    assert not (dst / "runs" / "3_S1.json").exists()          # prescreen differs
    assert not (dst / "runs" / "1_M.json").exists()           # task crash: rerun
    assert (dst / "merged_from_src.json").exists()


def test_dry_run_changes_nothing(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _w(src / "prescreen" / "1.json", _pre(1))
    _w(src / "runs" / "1_S1.json", {"seed": 1, "variant": "S1", "outcome": "ok"})
    res = G.merge(src, dst, dry_run=True)
    assert res["runs_copied"] == 1 and res["prescreen_copied"] == 1
    assert not dst.exists()
