"""Component-wise metering errors for controller-facing measurements.

Pandapower result tables remain plant truth. This module perturbs only freshly
constructed Measurement packets before they reach a controller or plotter.
"""

from __future__ import annotations

import hashlib
from typing import Any, Hashable, Optional

import numpy as np

from configs.config import MeasurementNoiseConfig
from core.measurement import Measurement


class MeasurementNoiseModel:
    """Apply persistent and per-sample measurement-chain errors.

    Accuracy-class bounds are component half-widths. For every physical
    component/channel, sample_noise_fraction of the bound is redrawn for each
    sample_id and the remainder is a persistent bias. Each component therefore
    remains inside its class bound while retaining a stable calibration error.

    Voltage magnitude passes through a VT/CVT and voltage meter. Current passes
    through a CT and current meter. P and Q are transformed together as complex
    power so gain and phase errors are physically coupled.
    """

    def __init__(self, config: MeasurementNoiseConfig) -> None:
        config.validate()
        self.config = config
        self.components = config.profile_components()
        self._sample_id: Optional[Hashable] = None
        self._bias_cache: dict[tuple[Any, ...], float] = {}
        self._sample_cache: dict[tuple[Any, ...], float] = {}

    def apply(
        self,
        measurement: Measurement,
        net,
        *,
        sample_id: Hashable,
        initialisation: bool = False,
    ) -> Measurement:
        """Perturb a fresh measurement packet in place and return it."""
        if not self.config.enabled:
            return measurement
        if initialisation and not self.config.apply_during_initialisation:
            return measurement
        if sample_id != self._sample_id:
            self._sample_id = sample_id
            self._sample_cache.clear()

        self._apply_voltage_magnitudes(measurement, net)
        self._apply_line_currents(measurement, net)
        self._apply_interface_power(measurement, net)
        self._apply_der_power(measurement, net)
        self._apply_generator_power(measurement, net)
        self._apply_tie_line_power(measurement, net)
        return measurement

    def _stable_draw(self, stream: str, key: tuple[Any, ...]) -> float:
        """Return a deterministic U(-1, 1) draw for seed/stream/key."""
        payload = repr((self.config.seed, stream, key)).encode("utf-8")
        integer = int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(), "big"
        )
        unit = (integer + 0.5) / float(1 << 64)
        return 2.0 * unit - 1.0

    def _component_error(self, half_width: float, *channel_key: Any) -> float:
        """Draw a bounded component error with persistent/sample parts."""
        key = tuple(channel_key)
        if key not in self._bias_cache:
            self._bias_cache[key] = self._stable_draw("bias", key)
        if key not in self._sample_cache:
            sample_key = (self._sample_id,) + key
            self._sample_cache[key] = self._stable_draw("sample", sample_key)
        fraction = float(self.config.sample_noise_fraction)
        return float(
            (1.0 - fraction) * half_width * self._bias_cache[key]
            + fraction * half_width * self._sample_cache[key]
        )

    @staticmethod
    def _positive_table_value(
        table, idx: int, columns: tuple[str, ...],
    ) -> float:
        if table is None or idx not in table.index:
            return float("nan")
        for column in columns:
            if column not in table.columns:
                continue
            try:
                value = abs(float(table.at[idx, column]))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0.0:
                return value
        return float("nan")

    @staticmethod
    def _integer_table_value(table, idx: int, column: str) -> Optional[int]:
        if table is None or idx not in table.index or column not in table.columns:
            return None
        try:
            return int(table.at[idx, column])
        except (TypeError, ValueError):
            return None

    def _bus_nominal_kv(self, net, bus_idx: Optional[int]) -> float:
        if bus_idx is None:
            return float("nan")
        return self._positive_table_value(
            getattr(net, "bus", None), int(bus_idx), ("vn_kv",)
        )

    def _voltage_transformer_bound(
        self, net, bus_idx: Optional[int],
    ) -> float:
        nominal_kv = self._bus_nominal_kv(net, bus_idx)
        if (
            np.isfinite(nominal_kv)
            and nominal_kv >= self.config.ehv_voltage_threshold_kv
        ):
            return self.components["ehv_voltage_transformer"]
        return self.components["hv_voltage_transformer"]

    def _voltage_transformer_error(
        self, net, bus_idx: Optional[int],
    ) -> float:
        key_bus = -1 if bus_idx is None else int(bus_idx)
        return self._component_error(
            self._voltage_transformer_bound(net, bus_idx),
            "vt_ratio", key_bus,
        )

    def _voltage_meter_error(self, bus_idx: int) -> float:
        return self._component_error(
            self.components["voltage_meter"],
            "voltage_meter", int(bus_idx),
        )

    def _ct_error(self, *channel_key: Any) -> float:
        return self._component_error(
            self.components["current_transformer"],
            "ct_ratio", *channel_key,
        )

    def _current_meter_error(self, *channel_key: Any) -> float:
        return self._component_error(
            self.components["current_meter"],
            "current_meter", *channel_key,
        )

    def _power_meter_gain_error(self, *channel_key: Any) -> float:
        return self._component_error(
            self.components["power_meter_gain"],
            "power_meter_gain", *channel_key,
        )

    def _power_phase_error_rad(self, *channel_key: Any) -> float:
        half_width_rad = float(np.deg2rad(
            self.components["power_phase_angle_deg"]
        ))
        return self._component_error(
            half_width_rad, "power_phase", *channel_key,
        )

    def _apply_voltage_magnitudes(self, measurement: Measurement, net) -> None:
        for pos, bus_idx_raw in enumerate(measurement.bus_indices):
            bus_idx = int(bus_idx_raw)
            value = float(measurement.voltage_magnitudes_pu[pos])
            if not np.isfinite(value):
                continue
            factor = (
                1.0 + self._voltage_transformer_error(net, bus_idx)
            ) * (1.0 + self._voltage_meter_error(bus_idx))
            noisy = value * factor
            if self.config.clip_nonnegative_magnitudes:
                noisy = max(0.0, noisy)
            measurement.voltage_magnitudes_pu[pos] = noisy

    def _ct_rating_ka(self, net, line_idx: int) -> float:
        line = getattr(net, "line", None)
        rating = self._positive_table_value(
            line, line_idx, tuple(self.config.ct_rating_columns)
        )
        if np.isfinite(rating):
            return rating
        if self.config.allow_line_rating_as_ct_fallback:
            return self._positive_table_value(line, line_idx, ("max_i_ka",))
        return float("nan")

    def _apply_line_currents(self, measurement: Measurement, net) -> None:
        line = getattr(net, "line", None)
        for pos, line_idx_raw in enumerate(measurement.branch_indices):
            line_idx = int(line_idx_raw)
            value = float(measurement.current_magnitudes_ka[pos])
            if not np.isfinite(value):
                continue
            endpoint = self._integer_table_value(line, line_idx, "from_bus")
            endpoint_key = -1 if endpoint is None else endpoint
            channel_key = ("line", line_idx, endpoint_key)
            relative_error = (
                (1.0 + self._ct_error(*channel_key))
                * (1.0 + self._current_meter_error(*channel_key))
                - 1.0
            )
            scale = abs(value)
            rating = self._ct_rating_ka(net, line_idx)
            if np.isfinite(rating):
                scale = max(
                    scale,
                    float(self.config.current_rating_floor) * rating,
                )
            noisy = value + relative_error * scale
            if self.config.clip_nonnegative_magnitudes:
                noisy = max(0.0, noisy)
            measurement.current_magnitudes_ka[pos] = noisy

    def _perturb_complex_power(
        self,
        p_mw: float,
        q_mvar: float,
        net,
        *,
        voltage_bus: Optional[int],
        ct_channel_key: tuple[Any, ...],
        meter_channel_key: tuple[Any, ...],
    ) -> tuple[float, float]:
        """Apply shared gain and phase displacement to one P/Q channel."""
        if not np.isfinite(p_mw) or not np.isfinite(q_mvar):
            return p_mw, q_mvar
        gain_factor = (
            (1.0 + self._voltage_transformer_error(net, voltage_bus))
            * (1.0 + self._ct_error(*ct_channel_key))
            * (1.0 + self._power_meter_gain_error(*meter_channel_key))
        )
        phase = self._power_phase_error_rad(*meter_channel_key)
        noisy = (
            complex(float(p_mw), float(q_mvar))
            * gain_factor
            * np.exp(1j * phase)
        )
        return float(noisy.real), float(noisy.imag)

    def _interface_active_power(self, net, idx: int) -> float:
        for table_name in ("res_trafo3w", "res_trafo"):
            table = getattr(net, table_name, None)
            if (
                table is not None
                and idx in table.index
                and "p_hv_mw" in table.columns
            ):
                return float(table.at[idx, "p_hv_mw"])
        return 0.0

    def _interface_hv_bus(self, net, idx: int) -> Optional[int]:
        bus = self._integer_table_value(
            getattr(net, "trafo3w", None), idx, "hv_bus"
        )
        if bus is not None:
            return bus
        return self._integer_table_value(
            getattr(net, "trafo", None), idx, "hv_bus"
        )

    def _apply_interface_power(self, measurement: Measurement, net) -> None:
        for pos, trafo_idx_raw in enumerate(
            measurement.interface_transformer_indices
        ):
            idx = int(trafo_idx_raw)
            _, noisy_q = self._perturb_complex_power(
                self._interface_active_power(net, idx),
                float(measurement.interface_q_hv_side_mvar[pos]),
                net,
                voltage_bus=self._interface_hv_bus(net, idx),
                ct_channel_key=("trafo", idx, "hv"),
                meter_channel_key=("interface_power", idx, "hv"),
            )
            measurement.interface_q_hv_side_mvar[pos] = noisy_q

    def _element_bus(self, table, idx: int) -> Optional[int]:
        return self._integer_table_value(table, idx, "bus")

    def _apply_der_power(self, measurement: Measurement, net) -> None:
        sgen = getattr(net, "sgen", None)
        for pos, der_idx_raw in enumerate(measurement.der_indices):
            idx = int(der_idx_raw)
            has_p = (
                measurement.der_p_mw is not None
                and pos < len(measurement.der_p_mw)
            )
            has_q = pos < len(measurement.der_q_mvar)
            if not has_p and not has_q:
                continue
            noisy_p, noisy_q = self._perturb_complex_power(
                float(measurement.der_p_mw[pos]) if has_p else 0.0,
                float(measurement.der_q_mvar[pos]) if has_q else 0.0,
                net,
                voltage_bus=self._element_bus(sgen, idx),
                ct_channel_key=("sgen", idx),
                meter_channel_key=("der_power", idx),
            )
            if has_p:
                measurement.der_p_mw[pos] = noisy_p
            if has_q:
                measurement.der_q_mvar[pos] = noisy_q

    def _apply_generator_power(self, measurement: Measurement, net) -> None:
        gen = getattr(net, "gen", None)
        for pos, gen_idx_raw in enumerate(measurement.gen_indices):
            idx = int(gen_idx_raw)
            has_p = pos < len(measurement.gen_p_mw)
            has_q = pos < len(measurement.gen_q_mvar)
            if not has_p and not has_q:
                continue
            noisy_p, noisy_q = self._perturb_complex_power(
                float(measurement.gen_p_mw[pos]) if has_p else 0.0,
                float(measurement.gen_q_mvar[pos]) if has_q else 0.0,
                net,
                voltage_bus=self._element_bus(gen, idx),
                ct_channel_key=("gen", idx),
                meter_channel_key=("generator_power", idx),
            )
            if has_p:
                measurement.gen_p_mw[pos] = noisy_p
            if has_q:
                measurement.gen_q_mvar[pos] = noisy_q

    def _tie_active_power(
        self,
        measurement: Measurement,
        net,
        pos: int,
        line_idx: int,
        endpoint: int,
    ) -> float:
        if pos < len(measurement.tie_line_p_mw):
            return float(measurement.tie_line_p_mw[pos])
        line = getattr(net, "line", None)
        res_line = getattr(net, "res_line", None)
        if line is None or res_line is None or line_idx not in res_line.index:
            return 0.0
        from_bus = self._integer_table_value(line, line_idx, "from_bus")
        to_bus = self._integer_table_value(line, line_idx, "to_bus")
        if endpoint == from_bus and "p_from_mw" in res_line.columns:
            return float(res_line.at[line_idx, "p_from_mw"])
        if endpoint == to_bus and "p_to_mw" in res_line.columns:
            return float(res_line.at[line_idx, "p_to_mw"])
        return 0.0

    def _apply_tie_line_power(self, measurement: Measurement, net) -> None:
        for pos, line_idx_raw in enumerate(measurement.tie_line_indices):
            idx = int(line_idx_raw)
            endpoint = (
                int(measurement.tie_line_endpoint_buses[pos])
                if pos < len(measurement.tie_line_endpoint_buses)
                else -1
            )
            has_p = pos < len(measurement.tie_line_p_mw)
            has_q = pos < len(measurement.tie_line_q_mvar)
            if not has_p and not has_q:
                continue
            noisy_p, noisy_q = self._perturb_complex_power(
                self._tie_active_power(measurement, net, pos, idx, endpoint),
                float(measurement.tie_line_q_mvar[pos]) if has_q else 0.0,
                net,
                voltage_bus=None if endpoint < 0 else endpoint,
                ct_channel_key=("line", idx, endpoint),
                meter_channel_key=("tie_power", idx, endpoint),
            )
            if has_p:
                measurement.tie_line_p_mw[pos] = noisy_p
            if has_q:
                measurement.tie_line_q_mvar[pos] = noisy_q

    # -- coverage beyond the OFO packets ------------------------------------

    def _set_sample(self, sample_id: Hashable) -> None:
        if sample_id != self._sample_id:
            self._sample_id = sample_id
            self._sample_cache.clear()

    def result_view(self, net, *, sample_id: Hashable):
        """Noisy view of the result tables for a controller that reads ``net``.

        ``res_bus.vm_pu``, ``res_gen`` P/Q, ``res_sgen`` P/Q and the coupler
        flows ``res_trafo3w.q_hv_mvar`` are perturbed with exactly the channel
        keys of :meth:`apply` (VT/meter per bus; CT/meter/phase per generator,
        per DER and per interface), so at the same
        ``sample_id`` a channel carries the same error whether an OFO packet or
        the classical SVR reads it.  Everything else is the true ``net``.
        Returns ``net`` itself when the model is disabled.
        """
        if not self.config.enabled:
            return net
        self._set_sample(sample_id)
        res_bus = net.res_bus.copy()
        vm = res_bus["vm_pu"].to_numpy(dtype=float, copy=True)
        for pos, bus_idx in enumerate(res_bus.index):
            if not np.isfinite(vm[pos]):
                continue
            noisy = vm[pos] * (
                1.0 + self._voltage_transformer_error(net, int(bus_idx))
            ) * (1.0 + self._voltage_meter_error(int(bus_idx)))
            if self.config.clip_nonnegative_magnitudes:
                noisy = max(0.0, noisy)
            vm[pos] = noisy
        res_bus["vm_pu"] = vm

        def _power_table(res, element, ct_name, meter_name):
            out = res.copy()
            if out.empty:
                return out
            p = out["p_mw"].to_numpy(dtype=float, copy=True)
            q = out["q_mvar"].to_numpy(dtype=float, copy=True)
            for pos, idx in enumerate(out.index):
                p[pos], q[pos] = self._perturb_complex_power(
                    p[pos], q[pos], net,
                    voltage_bus=self._element_bus(element, int(idx)),
                    ct_channel_key=(ct_name, int(idx)),
                    meter_channel_key=(meter_name, int(idx)),
                )
            out["p_mw"] = p
            out["q_mvar"] = q
            return out

        res_t3 = getattr(net, "res_trafo3w", None)
        if (res_t3 is not None and not res_t3.empty
                and "q_hv_mvar" in res_t3.columns):
            res_t3 = res_t3.copy()
            q_hv = res_t3["q_hv_mvar"].to_numpy(dtype=float, copy=True)
            for pos, idx in enumerate(res_t3.index):
                _, q_hv[pos] = self._perturb_complex_power(
                    self._interface_active_power(net, int(idx)), q_hv[pos], net,
                    voltage_bus=self._interface_hv_bus(net, int(idx)),
                    ct_channel_key=("trafo", int(idx), "hv"),
                    meter_channel_key=("interface_power", int(idx), "hv"),
                )
            res_t3["q_hv_mvar"] = q_hv
        return NoisyResultView(
            net, res_bus,
            _power_table(net.res_gen, net.gen, "gen", "generator_power"),
            _power_table(net.res_sgen, net.sgen, "sgen", "der_power"),
            res_t3,
        )


class NoisyResultView:
    """A read-only stand-in for ``net`` whose result tables carry metering errors.

    Only ``res_bus``, ``res_gen`` and ``res_sgen`` are replaced; every other
    attribute is the true network's, so set-points and topology are read
    exactly, as a real controller reads its own commands.
    """

    def __init__(self, net, res_bus, res_gen, res_sgen, res_trafo3w=None) -> None:
        self._net = net
        self.res_bus = res_bus
        self.res_gen = res_gen
        self.res_sgen = res_sgen
        if res_trafo3w is not None:
            self.res_trafo3w = res_trafo3w

    def __getattr__(self, name: str):
        return getattr(self._net, name)


class DerVoltageMeter:
    """The terminal-voltage reading of a converter's own droop (DAD input).

    One channel per converter (``der_vt``/``der_vmeter`` keyed by the sgen
    index), the VT class of its bus level and the persistent/per-sample split of
    the configured profile.  The sample id is the PLANT step, set by the runner
    before each step's first power flow, so a variant without any supervisory
    controller (L2) draws the same sequence as the others.
    """

    def __init__(self, config: MeasurementNoiseConfig) -> None:
        self._model = MeasurementNoiseModel(config)

    def set_sample(self, sample_id: Hashable) -> None:
        self._model._set_sample(sample_id)

    def measure(self, net, sgen_idx: int, bus_idx: int) -> float:
        v = float(net.res_bus.at[bus_idx, "vm_pu"])
        if not np.isfinite(v):
            return v
        m = self._model
        vt = m._component_error(
            m._voltage_transformer_bound(net, bus_idx), "der_vt", int(sgen_idx))
        mt = m._component_error(
            m.components["voltage_meter"], "der_vmeter", int(sgen_idx))
        return v * (1.0 + vt) * (1.0 + mt)


#: Key under which the runner stores a :class:`DerVoltageMeter` on the plant
#: net.  Absent (the default) means the droop reads the true voltage.
DER_VOLTAGE_METER_KEY = "_der_voltage_meter"


def measured_der_voltage(net, sgen_idx: int, bus_idx: int) -> float:
    """The voltage a converter's droop reads at ``bus_idx``.

    The single point through which both the droop law and the anchor written
    at a dispatch read the terminal voltage, so the two always come from the
    same channel.  True ``res_bus.vm_pu`` unless a meter is installed.
    """
    getter = getattr(net, "get", None)
    meter = getter(DER_VOLTAGE_METER_KEY) if callable(getter) else None
    if meter is None:
        return float(net.res_bus.at[bus_idx, "vm_pu"])
    return meter.measure(net, int(sgen_idx), int(bus_idx))
