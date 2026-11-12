"""merge_mc --variants: only the listed variants' runs are copied (2026-11-12)."""
import json
from pathlib import Path

from experiments.ch_11_case2_multisystem.mc.merge_mc import merge


def _campaign(root: Path, seeds, variants, outcome="ok"):
    (root / "prescreen").mkdir(parents=True)
    (root / "runs").mkdir()
    for s in seeds:
        (root / "prescreen" / f"{s}.json").write_text(json.dumps(
            {"seed": s, "valid": True, "start_time": "t", "events": [], "uc_offline": []}))
        for v in variants:
            (root / "runs" / f"{s}_{v}.json").write_text(json.dumps(
                {"seed": s, "variant": v, "outcome": outcome}))
            (root / "runs" / f"{s}_{v}.log.pkl").write_bytes(b"x")


def test_variant_filter_copies_only_listed(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _campaign(src, [1, 2, 3], ["L2", "S2", "O3", "M"])
    _campaign(dst, [1, 2], ["S2", "O3"])
    res = merge(src, dst, dry_run=True, variants=["L2", "M"])
    assert res["variants_copied"] == ["L2", "M"]
    assert res["runs_copied"] == 6
    assert res["variants_filter"] == ["L2", "M"]
    res = merge(src, dst, variants=["L2", "M"])
    assert not (dst / "runs" / "3_S2.json").exists()      # old S2 never fills a gap
    assert (dst / "runs" / "3_L2.log.pkl").exists()
    assert (dst / "prescreen" / "3.json").exists()


def test_no_filter_unchanged(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _campaign(src, [1], ["L2", "S2"])
    _campaign(dst, [1], ["S2"])
    res = merge(src, dst, dry_run=True)
    assert res["runs_copied"] == 1 and res["variants_filter"] is None
