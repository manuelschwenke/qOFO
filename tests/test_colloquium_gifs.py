"""Focused checks for scientific presentation exports; no simulation reruns."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from visualisation.colloquium_gifs import (
    scalar, matrix, band_polygons, envelope_limits, AnimationFigure, trafo_groups,
)


class ColloquiumGifTests(unittest.TestCase):
    def test_missing_measurement_stays_nan(self):
        records = [SimpleNamespace(m={1: 1.02}), SimpleNamespace(m={})]
        values = scalar(records, "m", 1)
        self.assertEqual(values[0], 1.02)
        self.assertTrue(np.isnan(values[1]))

    def test_device_channel_gaps_and_changed_shape(self):
        records = [SimpleNamespace(m={1: [1, 2]}), SimpleNamespace(m={})]
        values = matrix(records, "m", 1)
        np.testing.assert_equal(values[1], [np.nan, np.nan])
        records.append(SimpleNamespace(m={1: [1]}))
        with self.assertRaisesRegex(ValueError, "Changing channel"):
            matrix(records, "m", 1)

    def test_no_measurements_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "No recorded"):
            matrix([SimpleNamespace(m={})], "m", 1)

    def test_band_does_not_bridge_measurement_gap(self):
        x = np.arange(5)
        lo = np.array([0, 0, np.nan, 1, 1])
        polys = band_polygons(x, lo, lo + 1)
        self.assertEqual(len(polys), 2)
        self.assertEqual(polys[0][:, 0].max(), 1)
        self.assertEqual(polys[1][:, 0].min(), 3)

    def test_fixed_scale_contains_extrema(self):
        lo, hi = envelope_limits([1.03, np.nan, 1.07], [1.01])
        self.assertLess(lo, 1.01)
        self.assertGreater(hi, 1.07)

    def test_group_mapping_must_not_change(self):
        log = [SimpleNamespace(dso_trafo_group={"x": "DSO_1"}),
               SimpleNamespace(dso_trafo_group={"x": "DSO_2"})]
        with self.assertRaisesRegex(ValueError, "group changed"):
            trafo_groups(log, ["DSO_1", "DSO_2"])

    def test_constant_discrete_position_has_integer_axis_ticks(self):
        import matplotlib.pyplot as plt
        f = AnimationFigure([SimpleNamespace(time_s=20)], "Test", "", 1, 1, "")
        try:
            p = f.panel(0, 0, "Tap", "Step", (-.4, .4), integer=True)
            f.fig.canvas.draw()
            ticks = p.ax.get_yticks()
            np.testing.assert_allclose(ticks, np.round(ticks))
        finally:
            plt.close(f.fig)

    def test_pi_plots_do_not_require_unrecorded_ofo_measurements(self):
        from visualisation.colloquium_gifs import voltage_figure, actuator_figure
        import matplotlib.pyplot as plt
        cfg = SimpleNamespace(
            v_min_pu=.9, v_max_pu=1.1, zone_v_setpoints_pu={1:1.03,2:1.03,3:1.03},
            svr_pilot_buses={1:25,2:7,3:20},
        )
        records, zones = [], {}
        for step in (1, 2, 3):
            r = SimpleNamespace(step=step, time_s=step*20,
                zone_v_min={}, zone_v_mean={}, zone_v_max={},
                zone_q_gen={}, zone_q_der={}, zone_oltc_taps={},
                zone_svr_pilot_error_pu={}, zone_v_meas_mean={})
            for z in (1, 2, 3):
                r.zone_v_min[z], r.zone_v_mean[z], r.zone_v_max[z] = 1.01, 1.03, 1.05
                r.zone_q_gen[z] = np.array([10.0])
                r.zone_q_der[z] = np.array([5.0])
                r.zone_oltc_taps[z] = np.array([0])
                r.zone_svr_pilot_error_pu[z] = .002
                zones[str(z)] = {"generators":[z], "der":[z],
                    "details":{"oltc_trafo_indices":[z],"shunt_bus_indices":[]}}
            records.append(r)
        self.assertFalse(records[0].zone_v_meas_mean)
        figures = [
            voltage_figure(records, cfg, "pi", (1., 1.06)),
            actuator_figure(records, cfg, "pi", {"zones":zones}),
        ]
        try:
            for f in figures:
                f.update(1)
                f.fig.canvas.draw()
                self.assertTrue(all(len(line.get_ydata()) == 3
                                    for p in f.panels for line, _ in p.lines))
        finally:
            for f in figures:
                plt.close(f.fig)

    def test_progressive_reveal_events_and_gif_roundtrip(self):
        log = [SimpleNamespace(time_s=t) for t in [20, 1800, 14400]]
        f = AnimationFigure(log, "TEST DATA", "Exporter verification", 1, 1, "TEST")
        p = f.panel(0, 0, "Test", "Value", (0, 4))
        artist = p.line(np.array([1, 2, 3]))
        f.update(0)
        self.assertEqual(len(artist.get_xdata()), 0)
        self.assertFalse(any(a.get_visible() for a, _ in f.events))
        f.update(30)
        np.testing.assert_equal(artist.get_ydata(), [1, 2])
        self.assertTrue(next(a for a, t in f.events if t == 30).get_visible())
        self.assertFalse(next(a for a, t in f.events if t == 120).get_visible())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "roundtrip.gif"
            result = f.save(path, frame_seconds=7200)
            self.assertEqual(result["frames"], 3)
            self.assertEqual(result["duration_ms"], 2300)
            with Image.open(path) as image:
                self.assertEqual(image.info["loop"], 0)
                image.seek(2)
                image.load()


if __name__ == "__main__":
    unittest.main()
