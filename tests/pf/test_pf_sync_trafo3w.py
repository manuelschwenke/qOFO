"""Unit regressions for the pandapower -> PowerFactory 3W mapping."""

from pf.pf_sync import _push_trafo3w_type


class _RecordingContext:
    def __init__(self):
        self.writes = []

    def set_attr(self, obj, attr, value, *, label):
        self.writes.append((obj, attr, value, label))


def _record():
    return {
        "vk_hv_percent": 12.0,
        "vk_mv_percent": 8.0,
        "vk_lv_percent": 10.0,
        "vkr_hv_percent": 0.3,
        "vkr_mv_percent": 0.2,
        "vkr_lv_percent": 0.25,
        "sn_hv_mva": 300.0,
        "sn_mv_mva": 300.0,
        "sn_lv_mva": 75.0,
        "vn_hv_kv": 345.0,
        "vn_mv_kv": 110.0,
        "vn_lv_kv": 20.0,
        "pfe_kw": 80.0,
        "i0_percent": 0.04,
        "shift_mv_degree": 0.0,
        "shift_lv_degree": 150.0,
        "tap_side": "hv",
        "tap_changer_type": "Ratio",
        "tap_at_star_point": False,
        "tap_min": -13,
        "tap_max": 13,
        "tap_neutral": 0,
        "tap_step_percent": 1.25,
    }


def test_terminal_side_tap_maps_to_powerfactory_enum_one():
    ctx = _RecordingContext()
    typ = object()

    _push_trafo3w_type(ctx, typ, _record(), "test-type")

    writes = {attr: value for obj, attr, value, _ in ctx.writes if obj is typ}
    assert writes["itapos"] == 1
