"""Regressions for connection re-pointing and orphan detection in pf_sync.

Both behaviours were added on 2026-07-29 after the DSO coupling-geometry
change (``SUBNET_DEFS[*].hv_buses`` reordered to ``(0, 3, 8)``) moved each
DSO's couplers 0 and 1 onto each other's 110 kV bus.  Before the fix the sync
matched those ``ElmTr3`` by ``loc_name``, found every attribute equal and
reported "unchanged" while the model kept the old ``busmv``.
"""

import pytest

from pf.pf_sync import _psym_has_live_member, _reconnect


class _Cubicle:
    def __init__(self, term, name):
        self.cterm = term
        self.loc_name = name
        self.obj_id = None          # vacated unless an element claims it

    def GetAttribute(self, attr):
        return getattr(self, attr)


class _Terminal:
    def __init__(self, name):
        self.loc_name = name


class _Element:
    def __init__(self, **cubicles):
        self._attrs = dict(cubicles)

    def GetAttribute(self, attr):
        return self._attrs[attr]

    def SetAttribute(self, attr, value):
        self._attrs[attr] = value


class _Report:
    def __init__(self):
        self.updated = []
        self.deleted = []


class _Ctx:
    """The slice of SyncContext that _reconnect touches."""

    def __init__(self, terminals, *, dry_run=False):
        self.dry_run = dry_run
        self.report = _Report()
        # bus index -> loc_name, mirroring pf.naming.build_name_map
        self.names = {("bus", idx): name for idx, name in terminals.items()}
        self.term_alias = {name: _Terminal(name) for name in terminals.values()}
        self.created = []

    def cubicle(self, term, tag):
        cub = _Cubicle(term, f"Cub_qofo_{tag}")
        self.created.append(cub.loc_name)
        return cub

    def delete(self, obj, *, label):
        self.report.deleted.append(label)


def _ctx(**kwargs):
    return _Ctx({43: "DSO_1_bus43", 46: "DSO_1_bus46"}, **kwargs)


def test_reconnect_moves_a_winding_to_the_snapshot_bus():
    ctx = _ctx()
    old = _Cubicle(_Terminal("DSO_1_bus46"), "Cub_qofo_NC3W_DSO_1_t0_m")
    tr3 = _Element(busmv=old)

    moved = _reconnect(ctx, tr3, "busmv", 43, "NC3W_DSO_1_t0_m",
                       "NC3W_DSO_1_t0")

    assert moved is True
    assert tr3.GetAttribute("busmv").cterm.loc_name == "DSO_1_bus43"
    assert ctx.created == ["Cub_qofo_NC3W_DSO_1_t0_m"]
    assert "NC3W_DSO_1_t0.busmv" in ctx.report.updated[0]
    # the vacated cubicle must not be left behind
    assert ctx.report.deleted == [
        "StaCubic 'Cub_qofo_NC3W_DSO_1_t0_m' (vacated by NC3W_DSO_1_t0)"
    ]


def test_reconnect_is_a_no_op_when_already_correct():
    ctx = _ctx()
    tr3 = _Element(busmv=_Cubicle(_Terminal("DSO_1_bus43"), "Cub_x"))

    assert _reconnect(ctx, tr3, "busmv", 43, "tag", "NC3W_DSO_1_t0") is False
    assert ctx.report.updated == []
    assert ctx.created == []


def test_reconnect_keeps_a_cubicle_that_is_still_occupied():
    ctx = _ctx()
    old = _Cubicle(_Terminal("DSO_1_bus46"), "Cub_shared")
    old.obj_id = object()                       # another element still on it
    tr3 = _Element(busmv=old)

    _reconnect(ctx, tr3, "busmv", 43, "tag", "NC3W_DSO_1_t0")

    assert ctx.report.deleted == []


def test_reconnect_reports_but_does_not_touch_the_model_in_a_dry_run():
    ctx = _ctx(dry_run=True)
    tr3 = _Element(busmv=_Cubicle(_Terminal("DSO_1_bus46"), "Cub_x"))

    assert _reconnect(ctx, tr3, "busmv", 43, "tag", "NC3W_DSO_1_t0") is True
    assert len(ctx.report.updated) == 1
    assert ctx.created == []
    assert tr3.GetAttribute("busmv").cterm.loc_name == "DSO_1_bus46"


def test_reconnect_handles_an_unconnected_attribute():
    ctx = _ctx()
    tr3 = _Element(busmv=None)

    assert _reconnect(ctx, tr3, "busmv", 46, "tag", "NC3W_DSO_1_t1") is True
    assert tr3.GetAttribute("busmv").cterm.loc_name == "DSO_1_bus46"
    assert ctx.report.deleted == []


# --------------------------------------------------------------------------
#  Orphan station controllers
# --------------------------------------------------------------------------

class _DeadHandle:
    """A PF handle whose object was deleted: attribute access raises."""

    def __getattr__(self, name):
        raise RuntimeError("object has been deleted")


class _LiveMachine:
    def __init__(self, name="DER_DSO_1_s7_b46"):
        self.loc_name = name

    def IsDeleted(self):
        return False


class _Controller:
    def __init__(self, psym):
        self._psym = psym

    def GetAttribute(self, attr):
        if attr != "psym":
            raise KeyError(attr)
        return self._psym


def test_controller_with_only_dead_machines_is_an_orphan():
    assert _psym_has_live_member(_Controller([_DeadHandle()])) is False


def test_controller_with_no_machines_is_an_orphan():
    assert _psym_has_live_member(_Controller([])) is False
    assert _psym_has_live_member(_Controller(None)) is False


def test_controller_keeping_one_live_machine_is_not_an_orphan():
    ctrl = _Controller([_DeadHandle(), _LiveMachine()])
    assert _psym_has_live_member(ctrl) is True


def test_unreadable_psym_fails_safe():
    class _Unreadable:
        def GetAttribute(self, attr):
            raise RuntimeError("no such attribute")

    # The predicate only ever authorises a deletion, so anything it cannot
    # read must count as live.
    assert _psym_has_live_member(_Unreadable()) is True


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
