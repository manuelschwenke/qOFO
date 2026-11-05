"""run_ref_remote: one run_ref invocation on e5power, completed with --run-dir and pulled."""
from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P
from experiments.ch_09_parameter_selection import ch_9_11_procedure_v3 as V3
from experiments.ch_11_case2_multisystem.ref import run_ref_remote as D


def test_remote_driver_completes_and_pulls(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "ROOT", tmp_path)
    monkeypatch.setattr(P, "RUN", tmp_path)          # main() sets it; restored after
    monkeypatch.setattr(P, "rel", str)
    monkeypatch.setattr(P, "remote_head", lambda: "abc1234")
    monkeypatch.setattr(V3, "sync_remote_code", lambda: None)
    remote, calls = ["0019_2026-11-05_215935_run_ref_v4_E_fixmt"], []
    monkeypatch.setattr(D.R, "remote_names", lambda *a: list(remote))
    monkeypatch.setattr(D.R, "reserve_number", lambda *a: 20)
    names = D.R.VARIANTS.split(",")

    def run_remote(tag, argv, where, pull):
        calls.append(argv)
        if "--run-dir" not in argv:
            remote.append("0020_2026-11-05_230000_run_ref_v4_F")
        return 1 if len(calls) == 1 else 0

    def pull(name, rc, synced, rel, what, need):
        d = tmp_path / rel / name
        for n in (names[:2] if len(calls) == 1 else names):
            (d / n).mkdir(parents=True, exist_ok=True)
            (d / n / "status.json").write_text('{"status": "ok"}', "utf-8")
        return d
    monkeypatch.setattr(P, "run_remote", run_remote)
    monkeypatch.setattr(D.R, "_pull", pull)
    rc = D.main(["--scenario", "F", "--tag", "v4_F", "--note", "n"])
    assert rc == 0 and len(calls) == 2
    assert calls[0][calls[0].index("--scenario") + 1] == "F"
    assert calls[0][calls[0].index("--tag") + 1] == "v4_F"
    assert calls[1][-2:] == ["--run-dir", str(D.REL / "0020_2026-11-05_230000_run_ref_v4_F")]
    assert calls[1][calls[1].index("--variants") + 1] == ",".join(names[2:])
    assert (tmp_path / D.REL / "_remote" / "v4_F" / "driver.log").exists()
