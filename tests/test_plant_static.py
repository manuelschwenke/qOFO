"""Gate-E regression: PandapowerStaticPlant reproduces today's behaviour.

The Phase-6 plant abstraction (core/plant.py) must be a pure re-plumbing:
for an identical dispatch, writing through ``PandapowerStaticPlant.apply_u``
+ ``advance`` must leave the network and every result table bit-identical
to the legacy path (``experiments/helpers/plant_io`` write helpers followed
by the main-loop ``pp.runpp`` call of
``experiments/runners/multi_tso_dso.py``).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandapower as pp
import pandas as pd
import pytest

from core.plant import ActuatorWrites, PandapowerStaticPlant, Plant
from experiments.helpers.plant_io import (
    apply_dso_controls,
    apply_zone_tso_controls,
)
from export.make_snapshots import DEFAULT_T0, build_snapshot_state

#: The exact main-loop plant power-flow call (multi_tso_dso.py step loop).
RUNNER_PF_KWARGS = dict(
    run_control=True,
    calculate_voltage_angles=True,
    max_iteration=50,
    max_iter=300,
    distributed_slack=True,
    enforce_q_lims=True,
)

RES_TABLES = ("res_bus", "res_line", "res_trafo", "res_trafo3w",
              "res_sgen", "res_gen", "res_shunt", "res_load")


@pytest.fixture(scope="module")
def full_state():
    return build_snapshot_state("full", DEFAULT_T0, verbose=0)


def _assert_nets_identical(net_a: pp.pandapowerNet, net_b: pp.pandapowerNet):
    """Element inputs and every result table bit-identical (check_exact)."""
    for tbl in ("sgen", "gen", "trafo", "trafo3w", "shunt"):
        pd.testing.assert_frame_equal(
            net_a[tbl], net_b[tbl], check_exact=True, obj=tbl)
    for tbl in RES_TABLES:
        pd.testing.assert_frame_equal(
            net_a[tbl], net_b[tbl], check_exact=True, obj=tbl)


def _dispatch_fixture(net, meta):
    """A representative dispatch touching every actuator class."""
    hv0 = meta.hv_networks[0]
    der = [int(s) for s in hv0.sgen_indices[:2]]
    t3w = int(hv0.coupling_trafo_indices[0])
    mt = int(meta.machine_trafo_indices[0])
    gen = int(net.gen.index[0])
    sh = int(net.shunt.index[0])
    return SimpleNamespace(
        der=der,
        der_q=[7.5, -3.25],
        t3w=t3w,
        t3w_tap=int(net.trafo3w.at[t3w, "tap_pos"]) + 1,
        mt=mt,
        mt_tap=int(net.trafo.at[mt, "tap_pos"]) - 1,
        gen=gen,
        gen_v=float(net.gen.at[gen, "vm_pu"]) + 0.01,
        sh=sh,
        sh_bus=int(net.shunt.at[sh, "bus"]),
        sh_step=1 - int(net.shunt.at[sh, "step"]),
    )


def test_static_plant_is_a_plant(full_state):
    plant = PandapowerStaticPlant(copy.deepcopy(full_state.net))
    assert isinstance(plant, Plant)


def test_read_y_returns_the_measurement_image(full_state):
    net = copy.deepcopy(full_state.net)
    plant = PandapowerStaticPlant(net)
    assert plant.read_y() is net


def test_advance_matches_runner_power_flow(full_state):
    net_a = copy.deepcopy(full_state.net)
    net_b = copy.deepcopy(full_state.net)

    pp.runpp(net_a, **RUNNER_PF_KWARGS)

    plant = PandapowerStaticPlant(net_b)
    plant.advance(10.0)

    _assert_nets_identical(net_a, net_b)


def test_apply_u_matches_plant_io_write_path(full_state):
    """Dispatch via plant_io helpers vs ActuatorWrites: bit-identical."""
    net = full_state.net
    d = _dispatch_fixture(net, full_state.meta)

    # ── legacy path: plant_io helpers + runner power flow ────────────────
    net_a = copy.deepcopy(net)
    dso_cfg = SimpleNamespace(der_indices=d.der, oltc_trafo_indices=[d.t3w])
    dso_out = SimpleNamespace(u_new=np.array(d.der_q + [d.t3w_tap], float))
    apply_dso_controls(net_a, dso_cfg, dso_out)

    zone_def = SimpleNamespace(
        tso_der_indices=[], pcc_trafo_indices=[], gen_indices=[d.gen],
        oltc_trafo_indices=[d.mt], shunt_bus_indices=[d.sh_bus])
    tso_out = SimpleNamespace(
        u_new=np.array([d.gen_v, d.mt_tap, d.sh_step], float))
    apply_zone_tso_controls(net_a, zone_def, tso_out)

    pp.runpp(net_a, **RUNNER_PF_KWARGS)

    # ── plant path: ActuatorWrites through the Plant interface ───────────
    net_b = copy.deepcopy(net)
    plant = PandapowerStaticPlant(net_b)
    plant.apply_u(ActuatorWrites(
        der_q_set_mvar=dict(zip(d.der, d.der_q)),
        gen_v_pu={d.gen: d.gen_v},
        tap_2w={d.mt: d.mt_tap},
        tap_3w={d.t3w: d.t3w_tap},
        shunt_step={d.sh: d.sh_step},
    ))
    plant.advance(10.0)

    _assert_nets_identical(net_a, plant.read_y())


def test_apply_u_rejects_unknown_shunt_index(full_state):
    plant = PandapowerStaticPlant(copy.deepcopy(full_state.net))
    with pytest.raises(ValueError, match="not in\\s+net.shunt"):
        plant.apply_u(ActuatorWrites(shunt_step={987654: 1}))
