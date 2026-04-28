"""
Tests for core.pf_adapter (PowerFactory plant-interface adapter).

Two test groups:

* **Offline** — exercise the parts of the module that don't need a PF
  install (import, dataclass construction, attribute-dict completeness,
  registry methods against a hand-rolled fake-object graph).  These run
  everywhere.

* **Live PF** — full round-trip against a real PF session.  Guarded by
  ``pytest.mark.skipif`` so they are automatically skipped on machines
  without the ``powerfactory`` binding (CI, laptops, colleagues).  To run
  them, set environment variable ``QOFO_PF_TEST_PROJECT`` to the name of
  a PF project containing at least one ElmTerm / ElmLne / ElmSym /
  ElmGenstat / ElmTr2.  Example::

      PF_PYTHON_PATH="C:\\Program Files\\DIgSILENT\\PowerFactory 2024 SP2\\Python\\3.12"
      QOFO_PF_TEST_PROJECT="Nordic_SM"
      pytest tests/test_pf_adapter.py -v

Author: Manuel Schwenke
Date: 2026-04-24
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.measurement import Measurement
from core.pf_adapter import (
    LoadFlowDidNotConverge,
    PFRegistry,
    PFSession,
    _ATTR,
    _read_vector,
    _resolve_pf_python_path,
    apply_zone_tso_controls_pf,
    build_pf_registry_from_project,
    measure_zone_tso_pf,
    probe_attribute_names,
    run_ldf_pf,
)


# ---------------------------------------------------------------------------
#  Offline tests — fake PF objects, no powerfactory needed
# ---------------------------------------------------------------------------


class _FakeObj:
    """Tiny stand-in for a PF DataObject with ``GetAttribute`` /
    ``SetAttribute`` / ``loc_name``."""

    def __init__(self, name: str, attrs: Dict[str, Any]) -> None:
        self.loc_name = name
        self._attrs = dict(attrs)

    def GetAttribute(self, key: str) -> Any:
        return self._attrs[key]

    def SetAttribute(self, key: str, value: Any) -> None:
        self._attrs[key] = value

    def GetAttributeNames(self) -> List[str]:
        return list(self._attrs.keys())


class _FakeZoneDef:
    """Structural match for ``ZoneDefinition`` — only the index fields
    we need for the adapter."""

    def __init__(self, **kwargs) -> None:
        self.zone_id = kwargs.get("zone_id", 0)
        self.bus_indices = kwargs.get("bus_indices", [])
        self.gen_indices = kwargs.get("gen_indices", [])
        self.tso_der_indices = kwargs.get("tso_der_indices", [])
        self.line_indices = kwargs.get("line_indices", [])
        self.pcc_trafo_indices = kwargs.get("pcc_trafo_indices", [])
        self.shunt_bus_indices = kwargs.get("shunt_bus_indices", [])
        self.oltc_trafo_indices = kwargs.get("oltc_trafo_indices", [])


class _FakeControllerOutput:
    def __init__(self, u_new: np.ndarray) -> None:
        self.u_new = u_new


def _build_fake_registry() -> PFRegistry:
    """Build a 3-bus toy registry sufficient to exercise read + write paths."""
    bus0 = _FakeObj("BUS_0", {"m:u": 1.02})
    bus1 = _FakeObj("BUS_1", {"m:u": 1.00})
    bus2 = _FakeObj("BUS_2", {"m:u": 0.98})

    line0 = _FakeObj("LINE_01", {"m:I:bus1": 0.35})
    tr2_0 = _FakeObj(
        "TR2_MACHINE", {"nntap": 0, "m:Q:bushv": 42.0},
    )
    sym0 = _FakeObj(
        "GEN_A",
        {
            "usetp": 1.03,
            "m:Psum:bus1": 250.0,
            "m:Qsum:bus1": 80.0,
        },
    )
    sgen0 = _FakeObj(
        "DER_1",
        {
            "m:Psum:bus1": 50.0,
            "m:Qsum:bus1": 5.0,
            "qgini": 0.0,
            "pgini": 50.0,
        },
    )
    shunt_on_bus1 = _FakeObj("SHUNT_B1", {"ncapx": 0})

    reg = PFRegistry(project_name="FAKE")
    reg.bus_by_idx = {0: bus0, 1: bus1, 2: bus2}
    reg.line_by_idx = {0: line0}
    reg.trafo2w_by_idx = {0: tr2_0}
    reg.sym_by_idx = {0: sym0}
    reg.sgen_by_idx = {0: sgen0}
    reg.shunt_by_bus_idx = {1: shunt_on_bus1}

    reg.bus_name_by_idx = {0: "BUS_0", 1: "BUS_1", 2: "BUS_2"}
    reg.sym_name_by_idx = {0: "GEN_A"}
    reg.sgen_name_by_idx = {0: "DER_1"}

    reg.reverse = {
        id(bus0): ("ElmTerm", 0),
        id(bus1): ("ElmTerm", 1),
        id(bus2): ("ElmTerm", 2),
        id(line0): ("ElmLne", 0),
        id(tr2_0): ("ElmTr2", 0),
        id(sym0): ("ElmSym", 0),
        id(sgen0): ("ElmGenstat", 0),
        id(shunt_on_bus1): ("ElmShnt", 1),
    }
    return reg


class TestAttributeDict:
    """The ``_ATTR`` dict is the single source of truth for PF attribute
    strings — assert the keys the measure/apply code reads are all present."""

    def test_required_read_keys(self) -> None:
        for key in (
            "bus_vm_pu", "line_i_ka", "tr2_q_hv_mvar", "tr3_q_hv_mvar",
            "sgen_p_mw", "sgen_q_mvar", "sym_p_mw", "sym_q_mvar",
            "sym_vm_setpoint", "tr2_tap_pos", "shunt_step",
        ):
            assert key in _ATTR, f"Missing read-side key: {key}"

    def test_required_write_keys(self) -> None:
        for key in (
            "sgen_q_set", "sgen_p_set", "sym_vm_set",
            "tr2_tap_pos_set", "shunt_step_set",
        ):
            assert key in _ATTR, f"Missing write-side key: {key}"


class TestReadVector:
    def test_empty_returns_empty_array(self) -> None:
        out = _read_vector({}, [], attr="m:u", class_tag="ElmTerm", dtype=np.float64)
        assert out.shape == (0,)

    def test_reads_in_order(self) -> None:
        reg = _build_fake_registry()
        out = _read_vector(
            reg.bus_by_idx, [2, 0, 1],
            attr=_ATTR["bus_vm_pu"], class_tag="ElmTerm",
            dtype=np.float64,
        )
        np.testing.assert_allclose(out, [0.98, 1.02, 1.00])

    def test_missing_index_raises(self) -> None:
        reg = _build_fake_registry()
        with pytest.raises(KeyError, match="ElmTerm"):
            _read_vector(
                reg.bus_by_idx, [99],
                attr=_ATTR["bus_vm_pu"], class_tag="ElmTerm",
                dtype=np.float64,
            )


class TestMeasureZoneTsoPF:
    """Drive ``measure_zone_tso_pf`` with the fake registry."""

    def test_returns_measurement_with_correct_shapes(self) -> None:
        reg = _build_fake_registry()
        zd = _FakeZoneDef(
            zone_id=0,
            bus_indices=[0, 1, 2],
            gen_indices=[0],
            tso_der_indices=[0],
            line_indices=[0],
            pcc_trafo_indices=[],     # no PCC in the toy model
            shunt_bus_indices=[1],
            oltc_trafo_indices=[0],
        )

        # The helper does not actually need a live PFSession, so pass None.
        meas = measure_zone_tso_pf(session=None, registry=reg, zone_def=zd, it=7)

        assert isinstance(meas, Measurement)
        assert meas.iteration == 7
        # Voltages cover ALL buses, not just zone buses — matches pandapower side.
        np.testing.assert_array_equal(meas.bus_indices, np.array([0, 1, 2]))
        np.testing.assert_allclose(meas.voltage_magnitudes_pu, [1.02, 1.00, 0.98])
        np.testing.assert_allclose(meas.current_magnitudes_ka, [0.35])
        np.testing.assert_allclose(meas.gen_vm_pu, [1.03])
        np.testing.assert_allclose(meas.gen_p_mw, [250.0])
        np.testing.assert_allclose(meas.gen_q_mvar, [80.0])
        np.testing.assert_allclose(meas.der_q_mvar, [5.0])
        np.testing.assert_allclose(meas.der_p_mw, [50.0])
        np.testing.assert_array_equal(meas.oltc_tap_positions, np.array([0]))
        np.testing.assert_array_equal(meas.shunt_states, np.array([0]))

    def test_empty_zone_fields_produce_empty_arrays(self) -> None:
        reg = _build_fake_registry()
        zd = _FakeZoneDef(
            zone_id=0,
            bus_indices=[0, 1, 2],
            gen_indices=[],
            tso_der_indices=[],
            line_indices=[],
            pcc_trafo_indices=[],
            shunt_bus_indices=[],
            oltc_trafo_indices=[],
        )
        meas = measure_zone_tso_pf(session=None, registry=reg, zone_def=zd, it=0)
        assert meas.current_magnitudes_ka.shape == (0,)
        assert meas.der_q_mvar.shape == (0,)
        assert meas.gen_vm_pu.shape == (0,)
        assert meas.oltc_tap_positions.shape == (0,)
        assert meas.shunt_states.shape == (0,)


class TestApplyZoneTsoControlsPF:
    def test_writes_actuator_attributes(self) -> None:
        reg = _build_fake_registry()
        zd = _FakeZoneDef(
            zone_id=0,
            gen_indices=[0],
            tso_der_indices=[0],
            pcc_trafo_indices=[],
            oltc_trafo_indices=[0],
        )
        # u = [Q_DER (1) | Q_PCC (0) | V_gen (1) | s_OLTC (1)]
        u = np.array([-12.5, 1.015, 1.9], dtype=np.float64)
        out = _FakeControllerOutput(u_new=u)

        apply_zone_tso_controls_pf(session=None, registry=reg, zone_def=zd, tso_out=out)

        assert reg.sgen_by_idx[0].GetAttribute("qgini") == pytest.approx(-12.5)
        assert reg.sym_by_idx[0].GetAttribute("usetp") == pytest.approx(1.015)
        # OLTC rounded to nearest integer (mirrors pandapower ``int(round(...))``)
        assert reg.trafo2w_by_idx[0].GetAttribute("nntap") == 2

    def test_short_u_raises(self) -> None:
        reg = _build_fake_registry()
        zd = _FakeZoneDef(
            zone_id=0,
            gen_indices=[0],
            tso_der_indices=[0],
            oltc_trafo_indices=[0],
        )
        out = _FakeControllerOutput(u_new=np.array([1.0], dtype=np.float64))
        with pytest.raises(ValueError, match="expects at least"):
            apply_zone_tso_controls_pf(
                session=None, registry=reg, zone_def=zd, tso_out=out,
            )


class TestPFSessionPathResolution:
    def test_explicit_path_wins(self, tmp_path) -> None:
        d = tmp_path / "pf_py"
        d.mkdir()
        assert _resolve_pf_python_path(str(d)) == str(d)

    def test_missing_raises(self) -> None:
        # Override env to an impossible path, pass nothing explicit.
        old = os.environ.get("PF_PYTHON_PATH")
        os.environ["PF_PYTHON_PATH"] = r"Z:\definitely_does_not_exist\pf"
        try:
            with pytest.raises(FileNotFoundError, match="PowerFactory"):
                _resolve_pf_python_path(None)
        finally:
            if old is None:
                os.environ.pop("PF_PYTHON_PATH", None)
            else:
                os.environ["PF_PYTHON_PATH"] = old


class TestExceptionShape:
    def test_loadflow_exception_is_runtimeerror(self) -> None:
        exc = LoadFlowDidNotConverge("ierr=-1")
        assert isinstance(exc, RuntimeError)


# ---------------------------------------------------------------------------
#  Live PF tests — require the powerfactory binding and a named project
# ---------------------------------------------------------------------------


def _pf_live_available() -> bool:
    if os.environ.get("QOFO_PF_TEST_PROJECT") is None:
        return False
    try:
        _resolve_pf_python_path(None)
    except FileNotFoundError:
        return False
    return True


@pytest.mark.skipif(
    not _pf_live_available(),
    reason=(
        "Live PF tests require both the powerfactory Python binding and "
        "QOFO_PF_TEST_PROJECT env var pointing at a PF project."
    ),
)
class TestPFLiveSmoke:
    """End-to-end smoke test against a running PowerFactory instance.

    Only runs if the environment is set up (see module docstring).  Marked
    as *smoke* because it does not validate controller behaviour — it
    exists to catch attribute-name drift between PF versions and broken
    registry builds early.
    """

    @pytest.fixture(scope="class")
    def session(self):
        project = os.environ["QOFO_PF_TEST_PROJECT"]
        with PFSession(project) as s:
            yield s

    @pytest.fixture(scope="class")
    def registry(self, session):
        return build_pf_registry_from_project(session)

    def test_registry_non_empty(self, registry):
        # Bare minimum: at least one bus and one generator in the model.
        assert len(registry.bus_by_idx) > 0, registry.summary()
        assert len(registry.sym_by_idx) > 0, registry.summary()

    def test_ldf_converges(self, session):
        assert run_ldf_pf(session, balanced=True) is True

    def test_probe_attribute_names_reports_expected(self, session, registry):
        names = probe_attribute_names(session, registry, sample_per_class=1)
        # If this fails, the model probably has no ElmTerm — caught by
        # test_registry_non_empty above, but an explicit assert helps the
        # error message.
        assert names, "probe_attribute_names returned nothing"

    def test_measure_minimal_zone(self, session, registry):
        from controller.multi_tso_coordinator import ZoneDefinition

        any_gen = next(iter(registry.sym_by_idx))
        any_bus = next(iter(registry.bus_by_idx))

        zd = ZoneDefinition(
            zone_id=0,
            bus_indices=[any_bus],
            gen_indices=[any_gen],
            gen_bus_indices=[any_bus],
            tso_der_indices=[],
            tso_der_buses=[],
            v_bus_indices=[any_bus],
            line_indices=[],
            line_max_i_ka=[],
        )

        run_ldf_pf(session, balanced=True)
        meas = measure_zone_tso_pf(session, registry, zd, it=0)

        assert meas.iteration == 0
        assert meas.voltage_magnitudes_pu.shape == (len(registry.bus_by_idx),)
        # Sanity: in a sensible operating point every bus voltage is in
        # [0.5, 1.5] p.u. (loose bounds — tighter would be model-specific).
        assert np.all(meas.voltage_magnitudes_pu > 0.5)
        assert np.all(meas.voltage_magnitudes_pu < 1.5)
        assert meas.gen_vm_pu.shape == (1,)
        assert 0.5 < float(meas.gen_vm_pu[0]) < 1.5
