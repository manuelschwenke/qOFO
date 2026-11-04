"""ch_9_16_ref_after_v4: the W5 gate and the record additions (2026-11-04)."""
import pytest

from experiments.ch_09_parameter_selection import ch_9_16_ref_after_v4 as Q


def _doc(**prov):
    p = {"v4": {"config_set": ["sts_coupler_hold_after_bank=1"], "lambda_max_B_v4": 1.5}}
    p.update(prov)
    return {"fields": {"g_v": 1.0}, "provenance": p}


def test_w5_gate():
    assert Q.w5_clean(_doc(w5_report={"n_hunting": 0, "n_shunt_hunting": 0}))[0]
    assert not Q.w5_clean(_doc(w5_report={"n_hunting": 31, "n_shunt_hunting": 0}))[0]
    assert not Q.w5_clean(_doc(w5_report={"n_hunting": 0, "n_shunt_hunting": 1}))[0]
    assert not Q.w5_clean(_doc())[0]                 # no W5 report: not clean


def test_record_additions():
    add = Q.record_additions(_doc())
    assert add == {"fields": {"sts_coupler_hold_after_bank": 1},
                   "provenance": {"lambda_max_B": 1.5}}
    assert Q.record_additions(_doc(record_additions={})) == {}
    assert "provenance" not in Q.record_additions(_doc(lambda_max_B=1.4))


def test_record_additions_conflict_and_unknown_field():
    d = _doc()
    d["fields"]["sts_coupler_hold_after_bank"] = 0
    with pytest.raises(RuntimeError):
        Q.record_additions(d)
    d = _doc()
    d["provenance"]["v4"]["config_set"] = ["no_such_field=1"]
    with pytest.raises(RuntimeError):
        Q.record_additions(d)


def test_scenarios_and_variants():
    assert tuple(Q.SCENARIO_JOBS) == ("A", "B", "D", "E")   # GA, GE excluded (author 2026-11-04)
    assert sum(Q.SCENARIO_JOBS.values()) == 10 == Q.D_WORKERS  # e5wimi: 10 workers
    assert Q.AFFINITY_MASK == 0x55555
    assert sorted(Q.VARIANTS.split(",")) == sorted("L2,S1,S2,O2,O3,M".split(","))


def test_width_gate(tmp_path, monkeypatch):
    (tmp_path / "d1").mkdir()
    sel = tmp_path / "d1" / "selection.json"
    monkeypatch.setattr(Q, "ref_width", lambda: {"tso_qv_deadband_pu": 0.01,
                                                 "dso_qv_deadband_pu": 0.01})
    sel.write_text('{"delta_design_pu": 0.01, "delta_rule_pu": 0.0075}', "utf-8")
    assert Q.width_gate(tmp_path)[0]
    sel.write_text('{"delta_design_pu": 0.015, "delta_rule_pu": 0.0125}', "utf-8")
    assert not Q.width_gate(tmp_path)[0]
