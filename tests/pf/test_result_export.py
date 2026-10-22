from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pf.result_export import export_comres_csv, load_comres_trajectories


class _Object:
    def __init__(self, full_name: str):
        self._full_name = full_name

    def GetFullName(self) -> str:
        return self._full_name


class _ComRes:
    def __init__(self):
        self.attributes = {}
        self.execute_count = 0

    def SetAttribute(self, name, value):
        self.attributes[name] = value

    def Execute(self):
        self.execute_count += 1
        Path(self.attributes["f_name"]).write_text("exported", encoding="utf-8")
        return 0


class _App:
    def __init__(self, comres):
        self.comres = comres

    def GetFromStudyCase(self, name):
        assert name == "ComRes"
        return self.comres


def _write_fixture(path: Path) -> None:
    objects = [
        r"Study Cases\02_RMS_CoSim\All calculations.ElmRes",
        r"Grid\Power Plant 03\AVR 03.ElmDsl",
        r"DSO_1\DER_DSO_1_s10_b50.ElmGenstat",
        r"DSO_1\DSO_1_s10_b50.ElmTerm",
    ]
    variables = [
        '"b:tnow in s"',
        '"s:usetp"',
        '"m:Q:bus1 in Mvar"',
        '"m:u in p.u."',
    ]
    rows = [
        ";".join((f"{index / 100:.2f}", f"{1 + index / 100:.2f}",
                  f"{10 + index:.2f}", f"{0.90 + index / 100:.2f}"))
        .replace(".", ",")
        for index in range(8)
    ]
    path.write_text(
        ";".join(objects) + "\n"
        + ";".join(variables) + "\n"
        + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_export_comres_csv_uses_validated_complete_csv_settings(tmp_path):
    comres = _ComRes()
    result = object()
    path = tmp_path / "result.csv"

    returned = export_comres_csv(_App(comres), result, path)

    assert returned == path
    assert comres.execute_count == 1
    assert comres.attributes == {
        "pResult": result,
        "iopt_exp": 6,
        "f_name": str(path),
        "iopt_sep": 1,
        "iopt_honly": 0,
        "iopt_csel": 0,
    }


def test_load_comres_trajectories_matches_full_pf_paths_and_keeps_final_row(
    tmp_path,
):
    path = tmp_path / "result.csv"
    _write_fixture(path)
    q_obj = _Object(
        r"\user.IntUser\qOFO\project.IntPrj"
        r"\Network Model.IntPrjfolder\Network Data.IntPrjfolder"
        r"\DSO_1.ElmNet\DER_DSO_1_s10_b50.ElmGenstat"
    )
    u_obj = _Object(
        r"\user.IntUser\qOFO\project.IntPrj"
        r"\Network Model.IntPrjfolder\Network Data.IntPrjfolder"
        r"\DSO_1.ElmNet\DSO_1_s10_b50.ElmTerm"
    )
    avr_obj = _Object(
        r"\user.IntUser\qOFO\project.IntPrj"
        r"\Network Model.IntPrjfolder\Network Data.IntPrjfolder"
        r"\Grid.ElmNet\Power Plant 03.ElmComp\AVR 03.ElmDsl"
    )
    monitors = [
        (q_obj, "m:Q:bus1", "qDER_park"),
        (u_obj, "m:u", "uDER_park"),
        (avr_obj, "s:usetp", "vref_G03"),
    ]

    trajectories = load_comres_trajectories(
        path,
        monitors,
        since_s=0.03,
        stride=3,
        labels=lambda label: label.startswith(("qDER_", "uDER_")),
        chunksize=4,
    )

    assert set(trajectories) == {"qDER_park", "uDER_park"}
    np.testing.assert_allclose(
        trajectories["qDER_park"][0], [0.03, 0.06, 0.07]
    )
    np.testing.assert_allclose(
        trajectories["qDER_park"][1], [13.0, 16.0, 17.0]
    )
    np.testing.assert_allclose(
        trajectories["uDER_park"][1], [0.93, 0.96, 0.97]
    )


def test_load_comres_trajectories_fails_on_unregistered_monitor(tmp_path):
    path = tmp_path / "result.csv"
    _write_fixture(path)
    unknown = _Object(r"\Grid.ElmNet\missing.ElmTerm")

    with pytest.raises(ValueError, match="monitor not found"):
        load_comres_trajectories(
            path,
            [(unknown, "m:u", "u_missing")],
        )
