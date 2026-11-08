"""Scarcity redesign candidates (2026-11-08): reserved IBG capability, the
"north" transmission wind layout, and line 9-39 not built.

The base scenario ``rural_700`` must be unchanged by all of it.
"""
from __future__ import annotations

import contextlib
import io

import pytest

from controller.der_qv_local_loop import _qv_capability
from core.actuator_bounds import RESERVED_OP_DIAGRAM, ActuatorBounds
from network.ieee39 import constants
from network.ieee39.build import build_ieee39_net
from network.ieee39.hv_networks import add_hv_networks


def _bounds(op, p_ratio, sn=100.0):
    ab = ActuatorBounds.__new__(ActuatorBounds)
    return ab._compute_single_der_q_capability(p_ratio, sn, op)


@pytest.mark.parametrize("p_ratio", [0.0, 0.05, 0.1, 0.15, 0.2, 0.6, 1.0])
def test_reserved_capability_is_the_full_box_at_every_p(p_ratio):
    assert _bounds(RESERVED_OP_DIAGRAM, p_ratio) == pytest.approx((-33.0, 41.0))
    assert _qv_capability(100.0, RESERVED_OP_DIAGRAM, 100.0 * p_ratio) == \
        pytest.approx((-33.0, 41.0))


@pytest.mark.parametrize("p_ratio,expected", [
    (0.0, (0.0, 0.0)), (0.05, (0.0, 0.0)), (0.15, (-21.5, 25.5)),
    (0.2, (-33.0, 41.0)), (1.0, (-33.0, 41.0))])
def test_4120_v2_unchanged(p_ratio, expected):
    assert _bounds("VDE-AR-N-4120-v2", p_ratio) == pytest.approx(expected)
    assert _qv_capability(100.0, "VDE-AR-N-4120-v2", 100.0 * p_ratio) == \
        pytest.approx(expected)


def test_reserved_name_is_one_string():
    assert constants.RESERVED_OP_DIAGRAM == RESERVED_OP_DIAGRAM


def _build(scenario):
    with contextlib.redirect_stdout(io.StringIO()):
        net, meta = build_ieee39_net(ext_grid_vm_pu=1.03, scenario=scenario)
    return net, meta


def _ts_parks(net):
    ts = net.sgen[net.sgen["subnet"] == "TN"]
    return sorted((int(b), float(sn)) for b, sn in zip(ts["bus"], ts["sn_mva"]))


def test_base_layout_unchanged():
    net, meta = _build("rural_700")
    assert _ts_parks(net) == [(5, 500.0), (18, 508.0), (21, 650.0), (24, 540.0)]
    assert bool(net.line["in_service"].all())


def test_north_layout():
    net, meta = _build("rural_700_north")
    base, _ = _build("rural_700")
    assert _ts_parks(net) == [(18, 508.0), (25, 1500.0)]
    assert sorted(net.gen["name"]) == sorted(base.gen["name"])
    assert sorted(int(s) for s in meta.tso_der_indices) == sorted(
        int(i) for i in net.sgen.index[net.sgen["subnet"] == "TN"])
    park = net.sgen[net.sgen["bus"] == 25].iloc[0]
    assert park["p_mw"] == pytest.approx(1500.0)       # rated P = later base_p_mw
    assert park["profile"] == "WP10"


def test_line_939_not_built():
    net, _ = _build("rural_700_north_res_no939")
    off = [(int(net.line.at[i, "from_bus"]) + 1, int(net.line.at[i, "to_bus"]) + 1)
           for i in net.line.index if not bool(net.line.at[i, "in_service"])]
    assert off == [(9, 39)]


@pytest.mark.parametrize("scenario,expected", [
    ("rural_700", "VDE-AR-N-4120-v2"),
    ("rural_700_res", RESERVED_OP_DIAGRAM),
    ("rural_700_north_res", RESERVED_OP_DIAGRAM)])
def test_coupling_parks_diagram(scenario, expected):
    net, meta = _build(scenario)
    with contextlib.redirect_stdout(io.StringIO()):
        add_hv_networks(net, meta, install_tso_tertiary_shunts=False, verbose=False)
    dn = net.sgen[net.sgen["subnet"] == "DN"]
    coupling = dn["name"].astype(str).str.contains("WP_STATCOM_HV")
    assert int(coupling.sum()) == 12
    assert set(dn.loc[coupling, "op_diagram"]) == {expected}
    assert set(dn.loc[~coupling, "op_diagram"]) == {"VDE-AR-N-4120-v2"}
