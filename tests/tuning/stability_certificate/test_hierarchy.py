from __future__ import annotations

from tuning.stability_certificate.hierarchy import load_config_factory


def test_default_factory_reads_run_multi_system_ofo_parameters() -> None:
    config = load_config_factory()

    assert config.g_w_der == 50
    assert config.g_w_pcc == 200
    assert config.g_w_gen == 5e9
    assert config.g_w_dso_der == 1000
    assert config.g_w_dso_oltc == 150
    assert config.precondition_g_w is False
