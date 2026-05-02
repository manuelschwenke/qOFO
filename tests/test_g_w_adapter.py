"""Unit tests for ``controller/g_w_adapter.py``.

Tests cover the diagonal sign-rule (paper Eq. 16) implementation in
isolation from the rest of the OFO stack.  The synthetic ``grad_f`` and
``w`` arrays are crafted so the sign of ``s_i = -grad_f_i · w_i`` is
deterministic at every iteration, making the adapter trajectory exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from controller.g_w_adapter import (
    GwAdaptMeta,
    GwAdapter,
    make_adapter_from_class_indices,
)


# ---------------------------------------------------------------------------
# GwAdaptMeta validation
# ---------------------------------------------------------------------------

class TestGwAdaptMeta:
    def test_default_meta_is_valid(self) -> None:
        m = GwAdaptMeta()
        assert 0.0 <= m.beta1 < 1.0
        assert m.beta2 >= 0.0
        assert m.t_min > 0.0 and m.t_max >= m.t_min

    @pytest.mark.parametrize("beta1", [-0.1, 1.0, 1.5])
    def test_invalid_beta1(self, beta1: float) -> None:
        with pytest.raises(ValueError, match="beta1"):
            GwAdaptMeta(beta1=beta1)

    def test_invalid_beta2_negative(self) -> None:
        with pytest.raises(ValueError, match="beta2"):
            GwAdaptMeta(beta2=-0.01)

    def test_invalid_t_min_zero(self) -> None:
        with pytest.raises(ValueError, match="t_min"):
            GwAdaptMeta(t_min=0.0)

    def test_invalid_t_max_below_t_min(self) -> None:
        with pytest.raises(ValueError, match="t_max"):
            GwAdaptMeta(t_min=10.0, t_max=1.0)


# ---------------------------------------------------------------------------
# Constructor argument validation
# ---------------------------------------------------------------------------

def _ones_meta(n: int, **overrides: float) -> dict[str, np.ndarray]:
    """Helper: build per-variable meta arrays of length n."""
    defaults = {
        "beta1": 0.1,
        "beta2": 0.2,
        "t_min": 1e-3,
        "t_max": 1e3,
    }
    defaults.update(overrides)
    return {k: np.full(n, v, dtype=np.float64) for k, v in defaults.items()}


class TestGwAdapterConstructor:
    def test_shape_mismatch_raises(self) -> None:
        g0 = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="shape"):
            GwAdapter(
                g_w_init=g0,
                adapt_mask=np.ones(2, dtype=bool),  # wrong length
                beta1=np.ones(3) * 0.1,
                beta2=np.ones(3) * 0.2,
                t_min=np.ones(3) * 1e-3,
                t_max=np.ones(3) * 1e3,
            )

    def test_init_outside_clip_raises_when_adapted(self) -> None:
        g0 = np.array([100.0])  # above t_max
        meta = _ones_meta(1, t_max=10.0)
        with pytest.raises(ValueError, match="outside"):
            GwAdapter(
                g_w_init=g0,
                adapt_mask=np.ones(1, dtype=bool),
                **meta,
            )

    def test_init_outside_clip_ok_when_not_adapted(self) -> None:
        # Non-adapted entries can carry any positive g_w; the clip box
        # is irrelevant for them.
        g0 = np.array([100.0])
        meta = _ones_meta(1, t_max=10.0)
        adapter = GwAdapter(
            g_w_init=g0,
            adapt_mask=np.zeros(1, dtype=bool),
            **meta,
        )
        np.testing.assert_array_equal(adapter.g_w_live, [100.0])


# ---------------------------------------------------------------------------
# Sign-rule kernel: shrink / grow / deadband
# ---------------------------------------------------------------------------

def _make_adapter(g0: float, **kw) -> GwAdapter:
    """Single-variable adapter with reasonable defaults."""
    n = 1
    meta = _ones_meta(n, **kw)
    return GwAdapter(
        g_w_init=np.array([g0]),
        adapt_mask=np.ones(n, dtype=bool),
        **meta,
    )


class TestSignRule:
    def test_descent_regime_shrinks_g_w(self) -> None:
        # s = -grad * w > 0  ⇒ grad and w have opposite signs
        adapter = _make_adapter(g0=1.0, beta1=0.1, beta2=0.2, t_min=1e-6, t_max=1e6)
        grad = np.array([1.0])
        w = np.array([-1.0])  # s = -1 * -1 = +1 > 0
        adapter.update(grad, w)
        assert adapter.g_w_live[0] == pytest.approx(0.9)
        adapter.update(grad, w)
        assert adapter.g_w_live[0] == pytest.approx(0.81)

    def test_anti_descent_grows_g_w(self) -> None:
        # s = -grad * w < 0 ⇒ grad and w same sign
        adapter = _make_adapter(g0=1.0, beta1=0.1, beta2=0.2, t_min=1e-6, t_max=1e6)
        grad = np.array([1.0])
        w = np.array([1.0])  # s = -1 < 0
        adapter.update(grad, w)
        assert adapter.g_w_live[0] == pytest.approx(1.2)
        adapter.update(grad, w)
        assert adapter.g_w_live[0] == pytest.approx(1.44)

    def test_zero_inputs_holds_state(self) -> None:
        adapter = _make_adapter(g0=5.0)
        adapter.update(np.array([0.0]), np.array([0.0]))
        assert adapter.g_w_live[0] == pytest.approx(5.0)

    def test_deadband_holds_state(self) -> None:
        # |s| below deadband_rel * max(||g||·||w||, 1) ⇒ no change.
        # With small grad·w product, ||g||·||w|| < 1 so the floor is 1.0
        # and tol = deadband_rel = 1e-6.  Choose s = 1e-12 « tol.
        adapter = _make_adapter(g0=1.0)
        grad = np.array([1e-6])
        w = np.array([-1e-6])  # s = +1e-12, ||g||·||w|| = 1e-12
        adapter.update(grad, w)
        assert adapter.g_w_live[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Clipping behaviour
# ---------------------------------------------------------------------------

class TestClipping:
    def test_shrink_clips_at_t_min(self) -> None:
        adapter = _make_adapter(g0=1.0, beta1=0.5, beta2=0.5, t_min=0.5, t_max=10.0)
        grad = np.array([1.0])
        w = np.array([-1.0])  # always shrinks
        # 1.0 → 0.5 (clip) and stays
        for _ in range(20):
            adapter.update(grad, w)
        assert adapter.g_w_live[0] == pytest.approx(0.5)

    def test_grow_clips_at_t_max(self) -> None:
        adapter = _make_adapter(g0=1.0, beta1=0.5, beta2=0.5, t_min=0.1, t_max=2.0)
        grad = np.array([1.0])
        w = np.array([1.0])  # always grows
        for _ in range(20):
            adapter.update(grad, w)
        assert adapter.g_w_live[0] == pytest.approx(2.0)

    def test_geometric_decay_until_clip(self) -> None:
        # Predict the geometric trajectory exactly until the floor is hit.
        beta1 = 0.1
        adapter = _make_adapter(
            g0=1.0, beta1=beta1, beta2=0.0, t_min=0.5, t_max=1e6
        )
        grad = np.array([1.0])
        w = np.array([-1.0])
        expected = 1.0
        for _ in range(20):
            adapter.update(grad, w)
            expected = max(expected * (1.0 - beta1), 0.5)
            assert adapter.g_w_live[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Per-variable independence and adapt-mask
# ---------------------------------------------------------------------------

class TestPerVariable:
    def test_adapt_mask_protects_unmasked_entries(self) -> None:
        g0 = np.array([1.0, 1.0, 1.0])
        meta = _ones_meta(3, beta1=0.5, beta2=0.5, t_min=1e-6, t_max=1e6)
        adapter = GwAdapter(
            g_w_init=g0,
            adapt_mask=np.array([True, False, True]),
            **meta,
        )
        # All three would have s > 0 in a uniform descent regime.
        adapter.update(np.ones(3), -np.ones(3))
        # Adapted entries shrink; unmasked stays put.
        np.testing.assert_allclose(adapter.g_w_live, [0.5, 1.0, 0.5])

    def test_independent_directions(self) -> None:
        # Two adapted vars, opposite signs ⇒ one shrinks, one grows.
        g0 = np.array([1.0, 1.0])
        meta = _ones_meta(2, beta1=0.1, beta2=0.2, t_min=1e-6, t_max=1e6)
        adapter = GwAdapter(
            g_w_init=g0,
            adapt_mask=np.ones(2, dtype=bool),
            **meta,
        )
        grad = np.array([1.0, 1.0])
        w = np.array([-1.0, 1.0])  # s = [+1, -1]
        adapter.update(grad, w)
        np.testing.assert_allclose(adapter.g_w_live, [0.9, 1.2])

    def test_per_class_meta_via_helper(self) -> None:
        # Two classes with different beta values; both adapted.
        g0 = np.array([1.0, 1.0, 1.0, 1.0])
        class_idx = {
            "der": np.array([0, 1], dtype=np.int64),
            "oltc": np.array([2, 3], dtype=np.int64),
        }
        flags = {"der": True, "oltc": True}
        metas = {
            "der": GwAdaptMeta(beta1=0.1, beta2=0.0, t_min=1e-6, t_max=1e6),
            "oltc": GwAdaptMeta(beta1=0.5, beta2=0.0, t_min=1e-6, t_max=1e6),
        }
        adapter = make_adapter_from_class_indices(
            g0, class_idx, flags, metas,
        )
        grad = np.ones(4)
        w = -np.ones(4)  # s > 0 everywhere ⇒ all shrink at their own rate
        adapter.update(grad, w)
        np.testing.assert_allclose(
            adapter.g_w_live, [0.9, 0.9, 0.5, 0.5],
        )

    def test_helper_disables_unadapted_class(self) -> None:
        g0 = np.array([1.0, 1.0])
        class_idx = {
            "der": np.array([0], dtype=np.int64),
            "oltc": np.array([1], dtype=np.int64),
        }
        flags = {"der": True, "oltc": False}
        meta = GwAdaptMeta(beta1=0.5, beta2=0.5, t_min=1e-6, t_max=1e6)
        adapter = make_adapter_from_class_indices(g0, class_idx, flags, meta)
        adapter.update(np.array([1.0, 1.0]), np.array([-1.0, -1.0]))
        # der shrinks, oltc untouched
        np.testing.assert_allclose(adapter.g_w_live, [0.5, 1.0])

    def test_helper_overlap_raises(self) -> None:
        g0 = np.ones(3)
        class_idx = {
            "a": np.array([0, 1], dtype=np.int64),
            "b": np.array([1, 2], dtype=np.int64),
        }
        flags = {"a": True, "b": True}
        meta = GwAdaptMeta()
        with pytest.raises(ValueError, match="overlap"):
            make_adapter_from_class_indices(g0, class_idx, flags, meta)


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

class TestMultiTSOConfigPerClassHelper:
    """``MultiTSOConfig.make_g_w_adapt_meta()`` should return a single
    shared :class:`GwAdaptMeta` when no per-class override is set, and
    a per-class :class:`Mapping` once any override appears."""

    def _cfg(self, **overrides):
        from configs.multi_tso_config import MultiTSOConfig
        return __import__('dataclasses').replace(
            MultiTSOConfig(),
            adapt_g_w_der=True,
            adapt_g_w_pcc=True,
            adapt_g_w_dso_der=True,
            g_w_adapt_beta1=0.05,
            g_w_adapt_beta2=0.10,
            g_w_adapt_t_min=400.0,
            g_w_adapt_t_max=1e8,
            g_w_adapt_deadband_rel=1e-5,
            **overrides,
        )

    def test_no_per_class_returns_shared_meta(self) -> None:
        cfg = self._cfg()
        meta = cfg.make_g_w_adapt_meta()
        assert isinstance(meta, GwAdaptMeta)
        assert meta.t_min == pytest.approx(400.0)
        assert meta.beta1 == pytest.approx(0.05)

    def test_any_per_class_override_returns_dict(self) -> None:
        cfg = self._cfg(
            g_w_adapt_t_min_per_class={"der": 200.0, "pcc": 130.0},
        )
        meta = cfg.make_g_w_adapt_meta()
        assert isinstance(meta, dict)
        # Dict spans all adapted classes; missing ones fall back to shared.
        assert set(meta.keys()) == {"der", "pcc", "dso_der"}
        assert meta["der"].t_min == pytest.approx(200.0)
        assert meta["pcc"].t_min == pytest.approx(130.0)
        assert meta["dso_der"].t_min == pytest.approx(400.0)  # fallback
        # β / t_max / deadband_rel: all from shared scalar
        for cls in meta:
            assert meta[cls].beta1 == pytest.approx(0.05)
            assert meta[cls].beta2 == pytest.approx(0.10)
            assert meta[cls].t_max == pytest.approx(1e8)
            assert meta[cls].deadband_rel == pytest.approx(1e-5)

    def test_per_class_dict_includes_typo_entries(self) -> None:
        # An entry for a class that is not in the adapted set still
        # lands in the dict — caller can detect typos by inspecting
        # the resulting keys.  Adapter is not created for unmatched
        # classes (the ``_init_g_w_adapter`` filters by
        # ``adapt_g_w_classes``), so this is a debugging aid only.
        cfg = self._cfg(
            g_w_adapt_t_min_per_class={"der": 200.0, "typo_class": 1.0},
        )
        meta = cfg.make_g_w_adapt_meta()
        assert isinstance(meta, dict)
        assert "typo_class" in meta

    def test_per_class_overrides_only_one_knob(self) -> None:
        # Override t_min for one class; β stays shared.
        cfg = self._cfg(
            g_w_adapt_t_min_per_class={"der": 200.0},
        )
        meta = cfg.make_g_w_adapt_meta()
        assert isinstance(meta, dict)
        assert meta["der"].t_min == pytest.approx(200.0)
        assert meta["der"].beta1 == pytest.approx(0.05)


class TestBookkeeping:
    def test_n_updates_increments(self) -> None:
        adapter = _make_adapter(g0=1.0)
        assert adapter.n_updates == 0
        adapter.update(np.array([1.0]), np.array([-1.0]))
        adapter.update(np.array([1.0]), np.array([1.0]))
        assert adapter.n_updates == 2

    def test_g_w_live_view_is_readonly(self) -> None:
        adapter = _make_adapter(g0=1.0)
        view = adapter.g_w_live
        with pytest.raises(ValueError):
            view[0] = 99.0


# ---------------------------------------------------------------------------
# Integration: BaseOFOController wiring via DSOController
# ---------------------------------------------------------------------------

class TestDSOControllerIntegration:
    """Verify that ``adapt_g_w_classes`` activates the adapter and that
    one ``step()`` call produces a feasible MIQP solution while leaving
    or moving ``g_w_live`` exactly where the sign rule predicts."""

    @pytest.fixture
    def dso_with_adapter(self):
        # Reuse the existing test_controller helpers — they already
        # construct mock network state, sensitivities, and measurements
        # consistent with the DSO control vector layout.
        from tests.test_controller import (
            _make_network_state,
            _make_actuator_bounds,
            _make_mock_sensitivities,
            _make_dso_measurement,
        )
        from controller.dso_controller import DSOController, DSOControllerConfig
        from controller.base_controller import OFOParameters

        der_buses = [2, 3]
        oltc_trafos = [0]
        shunt_buses = [4]
        interface_trafos = [0]
        voltage_buses = [1, 2, 3]
        current_lines = [0, 1]
        n_der = len(der_buses)
        n_oltc = len(oltc_trafos)
        n_shunt = len(shunt_buses)
        n_interface = len(interface_trafos)
        n_v = len(voltage_buses)
        n_i = len(current_lines)
        n_outputs = n_interface + n_v + n_i

        cfg = DSOControllerConfig(
            der_indices=der_buses,
            oltc_trafo_indices=oltc_trafos,
            shunt_bus_indices=shunt_buses,
            shunt_q_steps_mvar=[50.0],
            interface_trafo_indices=interface_trafos,
            voltage_bus_indices=voltage_buses,
            current_line_indices=current_lines,
        )
        # Enable adaptation on the DER class only; OLTC/shunt stay
        # static.  Wide clip box keeps the rule active for many steps.
        params = OFOParameters(
            g_w=10.0,
            g_z=1000.0,
            g_u=0.01,
            adapt_g_w_classes=("dso_der",),
            g_w_adapt_meta=GwAdaptMeta(
                beta1=0.1, beta2=0.2,
                t_min=1e-3, t_max=1e6,
                deadband_rel=0.0,  # disable deadband for deterministic test
            ),
        )
        controller = DSOController(
            controller_id="dso_adapt_test",
            params=params,
            config=cfg,
            network_state=_make_network_state(),
            actuator_bounds=_make_actuator_bounds(n_der, n_oltc, n_shunt),
            sensitivities=_make_mock_sensitivities(
                n_outputs=n_outputs,
                n_der=n_der,
                n_oltc=n_oltc,
                n_shunt=n_shunt,
                der_indices=der_buses,
                oltc_trafo_indices=oltc_trafos,
                interface_trafo_indices=interface_trafos,
            ),
        )
        meas = _make_dso_measurement(
            der_indices=der_buses,
            oltc_trafo_indices=oltc_trafos,
            shunt_bus_indices=shunt_buses,
            interface_trafo_indices=interface_trafos,
            voltage_bus_indices=voltage_buses,
            current_line_indices=current_lines,
        )
        return controller, meas

    def test_adapter_active_after_initialise(self, dso_with_adapter) -> None:
        controller, meas = dso_with_adapter
        controller.initialise(meas)
        assert controller._g_w_adapter is not None
        # DER class adapted, OLTC + shunt left untouched (mask False there).
        mask = controller._g_w_adapter.adapt_mask
        # Layout: [DER..., OLTC, SHUNT] — n_der=2, n_oltc=1, n_shunt=1
        np.testing.assert_array_equal(mask, [True, True, False, False])

    def test_step_runs_and_increments_adapter(self, dso_with_adapter) -> None:
        controller, meas = dso_with_adapter
        controller.initialise(meas)
        n_before = controller._g_w_adapter.n_updates
        result = controller.step(meas)
        assert result.is_feasible
        # Exactly one update happened.
        assert controller._g_w_adapter.n_updates == n_before + 1

    def test_disabled_adaptation_leaves_adapter_none(self, dso_with_adapter) -> None:
        # Reuse the fixture but strip adapt classes.
        controller, meas = dso_with_adapter
        from controller.base_controller import OFOParameters
        controller.params = OFOParameters(
            g_w=10.0, g_z=1000.0, g_u=0.01,
            adapt_g_w_classes=(),  # disabled
        )
        controller.initialise(meas)
        assert controller._g_w_adapter is None

    def test_per_class_meta_via_dict_at_dso_controller(
        self, dso_with_adapter,
    ) -> None:
        """Adapter accepts a per-class meta dict from OFOParameters
        and applies the correct per-class clip / β to the live state."""
        from controller.dso_controller import (
            DSOController, DSOControllerConfig,
        )
        from controller.base_controller import OFOParameters
        from controller.g_w_adapter import GwAdaptMeta
        from tests.test_controller import (
            _make_network_state,
            _make_actuator_bounds,
            _make_mock_sensitivities,
            _make_dso_measurement,
        )

        cfg = DSOControllerConfig(
            der_indices=[2, 3],
            oltc_trafo_indices=[0],
            shunt_bus_indices=[4],
            shunt_q_steps_mvar=[50.0],
            interface_trafo_indices=[0],
            voltage_bus_indices=[1, 2, 3],
            current_line_indices=[0, 1],
        )
        # Two adapted classes with very different t_min values: dso_der
        # (continuous DER) is clipped at 4.0, dso_oltc (discrete) at
        # 50.0.  Per-class metas come in as a dict on g_w_adapt_meta.
        params = OFOParameters(
            g_w=10.0,           # uniform initial across all variables
            g_z=1000.0,
            g_u=0.01,
            adapt_g_w_classes=("dso_der", "dso_oltc"),
            g_w_adapt_meta={
                "dso_der":  GwAdaptMeta(
                    beta1=0.5, beta2=0.5,
                    t_min=4.0, t_max=1e3, deadband_rel=0.0,
                ),
                "dso_oltc": GwAdaptMeta(
                    beta1=0.5, beta2=0.5,
                    t_min=8.0, t_max=1e3, deadband_rel=0.0,
                ),
            },
        )
        controller = DSOController(
            controller_id="dso_per_class",
            params=params,
            config=cfg,
            network_state=_make_network_state(),
            actuator_bounds=_make_actuator_bounds(2, 1, 1),
            sensitivities=_make_mock_sensitivities(
                n_outputs=6, n_der=2, n_oltc=1, n_shunt=1,
                der_indices=[2, 3],
                oltc_trafo_indices=[0],
                interface_trafo_indices=[0],
            ),
        )
        meas = _make_dso_measurement(
            der_indices=[2, 3],
            oltc_trafo_indices=[0],
            shunt_bus_indices=[4],
            interface_trafo_indices=[0],
            voltage_bus_indices=[1, 2, 3],
            current_line_indices=[0, 1],
        )
        controller.initialise(meas)
        adapter = controller._g_w_adapter
        assert adapter is not None

        # adapt_mask: dso_der + dso_oltc → True; shunt → False
        np.testing.assert_array_equal(
            adapter.adapt_mask, [True, True, True, False],
        )

        # Drive 100 descent updates so each adapted entry hits its
        # per-class floor.  10 → t_min via (1-0.5)^N: 4 steps → 0.625,
        # so we definitely clip after a handful.
        for _ in range(50):
            adapter.update(np.ones(4), -np.ones(4))

        live = adapter.g_w_live
        # DER entries clipped at 4.0; OLTC entry clipped at 8.0;
        # shunt entry untouched.
        np.testing.assert_allclose(live[:2], [4.0, 4.0])
        assert live[2] == pytest.approx(8.0)
        assert live[3] == pytest.approx(10.0)  # not adapted, original

    def test_disabled_adaptation_first_step_matches_baseline(
        self, dso_with_adapter,
    ) -> None:
        # The adapter's first step must use g_w_init = static g_w, so
        # the MIQP solution should be identical (within solver tolerance)
        # to the same controller without adaptation.
        controller_with, meas = dso_with_adapter
        controller_with.initialise(meas)
        result_with = controller_with.step(meas)

        # Build a parallel controller with adaptation off but the same
        # static g_w.
        from tests.test_controller import (
            _make_network_state,
            _make_actuator_bounds,
            _make_mock_sensitivities,
            _make_dso_measurement,
        )
        from controller.dso_controller import (
            DSOController, DSOControllerConfig,
        )
        from controller.base_controller import OFOParameters
        cfg = controller_with.config
        params_static = OFOParameters(
            g_w=10.0, g_z=1000.0, g_u=0.01,
        )
        controller_off = DSOController(
            controller_id="dso_static_ref",
            params=params_static,
            config=cfg,
            network_state=_make_network_state(),
            actuator_bounds=_make_actuator_bounds(2, 1, 1),
            sensitivities=_make_mock_sensitivities(
                n_outputs=6, n_der=2, n_oltc=1, n_shunt=1,
                der_indices=[2, 3],
                oltc_trafo_indices=[0],
                interface_trafo_indices=[0],
            ),
        )
        controller_off.initialise(meas)
        result_off = controller_off.step(meas)
        np.testing.assert_allclose(
            result_with.sigma, result_off.sigma, atol=1e-6,
        )
