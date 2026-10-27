"""S2: the classical scheme with access to the child STS (eq:svr-alignment-sts).

Design note docs/architecture/2026-10-27_S2_design_note.md (approved 2026-10-27).
Load convention at the HV port: supporting the TS voltage lowers the flow.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from controller.svr_controller import SVRZoneConfig, SVRZoneController
from core.message import CapabilityMessage


def _ctrl(n_pcc=2):
    zd = SimpleNamespace(zone_id=2, gen_indices=[], tso_der_indices=[],
                         pcc_trafo_indices=list(range(10, 10 + n_pcc)),
                         oltc_trafo_indices=[], shunt_bus_indices=[],
                         v_bus_indices=[])
    return SVRZoneController(zd, None, SVRZoneConfig(pilot_bus=1, sts_access=True))


def _net(q_now):
    return SimpleNamespace(res_trafo3w=pd.DataFrame(
        {"q_hv_mvar": q_now}, index=[10, 11][:len(q_now)]))


def _msg(dmin, dmax, qn):
    return CapabilityMessage("DSO_1", "tso_2", 0, np.array([10, 11]),
                             np.array(dmin, float), np.array(dmax, float),
                             q_neutral_mvar=np.array(qn, float))


def test_level_moves_the_reference_from_the_neutral_point_to_the_edges():
    c = _ctrl()
    c.receive_capability(_msg([-60.0, -40.0], [30.0, 20.0], [-10.0, 5.0]))
    net = _net([0.0, 0.0])               # interval [-60, 30] and [-40, 20]
    np.testing.assert_allclose(c.coupler_references(net, 0.0), [-10.0, 5.0])
    # raise the voltage: toward Q_lo (more injection into the TS)
    np.testing.assert_allclose(c.coupler_references(net, 1.0), [-60.0, -40.0])
    np.testing.assert_allclose(c.coupler_references(net, 0.5), [-35.0, -17.5])
    # lower the voltage: toward Q_hi
    np.testing.assert_allclose(c.coupler_references(net, -1.0), [30.0, 20.0])
    np.testing.assert_allclose(c.coupler_references(net, -0.5), [10.0, 12.5])


def test_zero_width_interval_drops_the_coupler_out_of_the_alignment():
    c = _ctrl()
    c.receive_capability(_msg([0.0, 0.0], [0.0, 0.0], [7.0, -3.0]))
    net = _net([7.0, -3.0])
    for level in (-1.0, -0.3, 0.0, 0.4, 1.0):
        np.testing.assert_allclose(c.coupler_references(net, level), [7.0, -3.0])


def test_a_neutral_point_outside_its_interval_is_clipped_and_counted():
    c = _ctrl()
    c.receive_capability(_msg([-10.0, -10.0], [10.0, 10.0], [50.0, 0.0]))
    refs = c.coupler_references(_net([0.0, 0.0]), 0.0)
    np.testing.assert_allclose(refs, [10.0, 0.0])
    assert c.n_neutral_clipped == 1


def test_only_the_areas_own_couplers_are_read():
    c = _ctrl(n_pcc=1)
    c.receive_capability(_msg([-5.0, -9.0], [5.0, 9.0], [1.0, 2.0]))
    assert set(c._cap) == {10}


def test_missing_capability_fails_fast():
    c = _ctrl()
    with pytest.raises(RuntimeError, match="no capability message"):
        c.coupler_references(_net([0.0, 0.0]), 0.2)


def test_message_neutral_point_is_optional():
    m = CapabilityMessage("a", "b", 0, np.array([1]), np.array([0.0]), np.array([0.0]))
    assert m.q_neutral_mvar is None
