"""Remote shards of the Ch 9 drivers (ch_9_6_full_procedure.run_remote): the shard logs
are a record, so a failed pull of them never ends a run, and shards that finish together
do not copy the shared log folder at the same time."""
import threading
import time

from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P


def _fake_remote(monkeypatch, tmp_path, log_pull):
    monkeypatch.setattr(P, "RUN", tmp_path)
    monkeypatch.setattr(P, "rel", lambda p: str(p))
    monkeypatch.setattr(P, "push", lambda local, remote: None)
    monkeypatch.setattr(P, "ps", lambda script, timeout=1800.0:
                        'DONE {"rc": 0, "finished": "2026-11-02 08:16:00"}')

    def pull_dir(remote_dir, local_dir, overwrite=False):
        return log_pull() if local_dir.name == "_remote" else 1

    monkeypatch.setattr(P, "pull_dir", pull_dir)


def test_a_failed_shard_log_pull_does_not_end_the_run(monkeypatch, tmp_path):
    def locked():
        raise PermissionError(32, "The process cannot access the file")

    _fake_remote(monkeypatch, tmp_path, locked)
    where = tmp_path / "expC"
    assert P.run_remote("cand_x", ["noop"], where, [(where / "cand_x", True)]) == 0
    assert "shard logs not pulled" in (tmp_path / "driver.log").read_text("utf-8")


def test_shards_finishing_together_pull_their_logs_one_at_a_time(monkeypatch, tmp_path):
    active, peak, guard = [0], [0], threading.Lock()

    def counted():
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with guard:
            active[0] -= 1
        return 1

    _fake_remote(monkeypatch, tmp_path, counted)
    where = tmp_path / "expC"
    rcs = []
    threads = [threading.Thread(target=lambda t=t: rcs.append(
        P.run_remote(t, ["noop"], where, [(where / t, True)]))) for t in ("a", "b", "c")]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert rcs == [0, 0, 0]
    assert peak[0] == 1
