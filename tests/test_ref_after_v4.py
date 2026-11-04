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
    assert sum(Q.SCENARIO_JOBS.values()) == 10                 # e5wimi: 10 workers
    assert Q.AFFINITY_MASK == 0x55555
    assert Q.REMOTE_WORKERS == 30                              # D1 + D2 on e5power
    assert sorted(Q.VARIANTS.split(",")) == sorted("L2,S1,S2,O2,O3,M".split(","))


def _fake_ref(tmp_path, monkeypatch, n_fail):
    """P.run that fails ``n_fail`` times without a run folder, then completes."""
    from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P
    monkeypatch.setattr(P, "RUN", tmp_path)
    monkeypatch.setattr(Q, "OUT", tmp_path)
    monkeypatch.setattr(Q, "REF_DIR", tmp_path / "ref")
    monkeypatch.setattr(P, "rel", str)               # tmp_path may be on another drive
    calls = []

    def run(argv, logfile):
        calls.append(argv)
        if len(calls) <= n_fail:
            return 1
        d = tmp_path / "ref" / "0010_2026-11-05_080000_run_ref_v4_A"
        for n in Q.VARIANTS.split(","):
            (d / n).mkdir(parents=True, exist_ok=True)
            (d / n / "status.json").write_text('{"status": "ok"}', "utf-8")
        return 0
    monkeypatch.setattr(P, "run", run)
    return calls


def test_ref_start_failures_are_repeated(tmp_path, monkeypatch):
    calls = _fake_ref(tmp_path, monkeypatch, n_fail=5)
    res = Q.ref_one(0, "A", "k", dry_run=False)
    assert len(calls) == 6 and set(res["status"].values()) == {"ok"}


def test_ref_late_failures_stop_after_retries(tmp_path, monkeypatch):
    calls = _fake_ref(tmp_path, monkeypatch, n_fail=50)
    monkeypatch.setattr(Q, "START_FAIL_S", -1.0)     # every exit counts as late
    res = Q.ref_one(0, "A", "k", dry_run=False)
    assert len(calls) == 1 + Q.RETRIES and res["folder"] is None


LOCAL = ["0008_2026-11-04_133847_ch_9_14_procedure_v4", "0009_2026-11-04_195947_ch_9_4_deadband_smoke"]


def _fake_remote(tmp_path, monkeypatch, remote, outcomes):
    """e5power as a list of folder names; each start applies the next outcome:
    None (start failure: no folder), "broken" (folder, no selection) or "ok"."""
    from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P
    monkeypatch.setattr(P, "RUN", tmp_path)
    monkeypatch.setattr(Q, "OUT", tmp_path)
    monkeypatch.setattr(Q, "local_names", lambda: list(LOCAL))
    monkeypatch.setattr(Q, "remote_names", lambda: list(remote))
    made, selected, calls = [], set(), []

    def remote_cmd(script):
        if script.startswith("New-Item"):
            made.append(script)
            remote.append(script.split("\\")[-1].split("'")[0])
        return "True" if any(n in script for n in selected) else "False"
    monkeypatch.setattr(Q, "_remote", remote_cmd)

    def run_remote(tag, argv, where, pull):
        calls.append(argv)
        out = outcomes[len(calls) - 1]
        if out is not None and "--reduce" not in argv:
            remote.append(f"{max(map(Q._counter, remote)) + 1:04d}_2026-11-05_080000_ch_9_4_deadband")
        if out == "ok":
            selected.add(remote[-1])
        return 0 if out == "ok" else 1
    monkeypatch.setattr(P, "run_remote", run_remote)
    monkeypatch.setattr(Q, "_pull", lambda name, rc: tmp_path / name)
    return made, calls


def test_reserve_number(tmp_path, monkeypatch):
    remote = ["0006_2026-11-03_211954_ch_9_11_procedure_v3"]
    made, _ = _fake_remote(tmp_path, monkeypatch, remote, [])
    assert Q.reserve_number() == 10 and len(made) == 1 and remote[-1] == LOCAL[-1]
    remote = ["0010_2026-10-31_093854_ch_9_5_avr_revalidation"]   # e5power ahead
    made, _ = _fake_remote(tmp_path, monkeypatch, remote, [])
    assert Q.reserve_number() == 11 and not made


def test_deadband_remote_retries(tmp_path, monkeypatch):
    remote = ["0006_2026-11-03_211954_ch_9_11_procedure_v3"]
    _, calls = _fake_remote(tmp_path, monkeypatch, remote, [None, None, "broken", "ok"])
    folder = Q.deadband("k", dry_run=False)
    assert folder.name == "0010_2026-11-05_080000_ch_9_4_deadband" and len(calls) == 4
    assert "--note" in calls[2] and calls[3][-2:] == [
        "--reduce", str(Q.DEADBAND_REL / folder.name)]
    _, calls = _fake_remote(tmp_path, monkeypatch, list(remote), ["broken"] * 9)
    assert Q.deadband("k", dry_run=False) is None and len(calls) == 1 + Q.RETRIES


def test_width_gate(tmp_path, monkeypatch):
    (tmp_path / "d1").mkdir()
    sel = tmp_path / "d1" / "selection.json"
    monkeypatch.setattr(Q, "ref_width", lambda: {"tso_qv_deadband_pu": 0.01,
                                                 "dso_qv_deadband_pu": 0.01})
    sel.write_text('{"delta_design_pu": 0.01, "delta_rule_pu": 0.0075}', "utf-8")
    assert Q.width_gate(tmp_path)[0]
    sel.write_text('{"delta_design_pu": 0.015, "delta_rule_pu": 0.0125}', "utf-8")
    assert not Q.width_gate(tmp_path)[0]
