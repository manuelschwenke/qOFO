from __future__ import annotations

import pytest

from pf.screening import (
    EVENT_INERT_TIME_S,
    PERSISTENT_EVENT_PREFIXES,
    ScreeningContext,
)
from pf.session import PFSessionError


class _Target:
    def __init__(self, name: str):
        self.loc_name = name

    def GetFullName(self):
        return f"grid\\{self.loc_name}"


class _Event:
    def __init__(self, folder, class_name: str, name: str):
        self.folder = folder
        self.class_name = class_name
        self.loc_name = name
        self.attributes = {}
        self.set_calls = []

    def SetAttribute(self, name, value):
        self.attributes[name] = value
        self.set_calls.append((name, value))
    def GetAttribute(self, name):
        return self.attributes[name]

    def GetClassName(self):
        return self.class_name


    def GetFullName(self):
        return f"events\\{self.loc_name}"

    def Delete(self):
        self.folder.events.remove(self)


class _EventFolder:
    def __init__(self):
        self.events = []

    def CreateObject(self, class_name: str, name: str):
        event = _Event(self, class_name, name)
        self.events.append(event)
        return event

    def GetContents(self, *_args):
        return list(self.events)

class _Sim:
    def __init__(self):
        self.attributes = {}
        self.execute_count = 0

    def SetAttribute(self, name, value):
        self.attributes[name] = value

    def Execute(self):
        self.execute_count += 1
        return 0


class _App:
    def __init__(self):
        self.reset_count = 0

    def ResetCalculation(self):
        self.reset_count += 1


def _context(*, strict: bool = True,
             persistent: bool = False) -> ScreeningContext:
    ctx = object.__new__(ScreeningContext)
    ctx.app = _App()
    ctx.sim = _Sim()
    ctx.evt_folder = _EventFolder()
    ctx.require_preallocated_events = strict
    ctx.persistent_event_pool = persistent
    ctx._param_event_slots = {}
    ctx._param_event_next = {}
    ctx._load_event_slots = {}
    ctx._load_event_next = {}
    ctx._tap_event_slots = {}
    ctx._tap_event_next = {}
    ctx._pool_serial = {
        class_name: 0 for class_name in PERSISTENT_EVENT_PREFIXES
    }
    ctx._calculation_active = False
    ctx._new_events_pending_admission = 0
    ctx._persistent_events_created = 0
    ctx._persistent_events_discovered = 0
    ctx._admission_executes = 0
    ctx._retired_events = 0
    ctx._armed_persistent_events = []
    return ctx


def test_param_slots_are_consumed_once_and_exhaust_fail_fast():
    ctx = _context()
    target = _Target("qvpre")
    ctx.preallocate_param_events(
        target, "qset", 2, initial_value=-0.1
    )

    assert len(ctx.evt_folder.events) == 2
    assert all(
        event.attributes["time"] == EVENT_INERT_TIME_S
        for event in ctx.evt_folder.events
    )

    ctx.add_param_event(target, "qset", 0.1, 1.01)
    ctx.add_param_event(target, "qset", 0.2, 21.01)

    first, second = ctx.evt_folder.events
    assert first.attributes["value"] == repr(0.1)
    assert first.attributes["time"] == 1.01
    assert second.attributes["value"] == repr(0.2)
    assert second.attributes["time"] == 21.01
    assert first.set_calls[-2:][0][0] == "value"
    assert first.set_calls[-2:][1][0] == "time"
    assert ctx.event_pool_stats()["param_used"] == 2

    with pytest.raises(PFSessionError, match="slots exhausted"):
        ctx.add_param_event(target, "qset", 0.3, 41.01)
    assert len(ctx.evt_folder.events) == 2


def test_load_slot_arms_payload_before_time():
    ctx = _context()
    target = _Target("load")
    ctx.preallocate_load_events(target, 1)

    ctx.add_load_event(target, 1.5, -2.5, 0.01)

    event = ctx.evt_folder.events[0]
    assert event.set_calls[-3:] == [
        ("dP", 1.5),
        ("dQ", -2.5),
        ("time", 0.01),
    ]
    assert ctx.event_pool_stats() == {
        "param_total": 0,
        "param_used": 0,
        "load_total": 1,
        "load_used": 1,
        "tap_total": 0,
        "tap_used": 0,
        "discovered": 0,
        "created": 1,
        "pending_admission": 0,
        "admission_executes": 0,
        "retired": 0,
    }


def test_strict_context_refuses_unregistered_mid_run_creation():
    ctx = _context(strict=True)
    target = _Target("unregistered")

    with pytest.raises(PFSessionError, match="refusing mid-run CreateObject"):
        ctx.add_param_event(target, "qset", 0.1, 0.01)
    with pytest.raises(PFSessionError, match="refusing mid-run CreateObject"):
        ctx.add_load_event(target, 1.0, 1.0, 0.01)
    assert ctx.evt_folder.events == []


def test_legacy_non_strict_context_keeps_single_event_probe_support():
    ctx = _context(strict=False)
    target = _Target("probe")

    ctx.add_param_event(target, "qset", 0.1, 1.0)
    ctx.add_load_event(target, 5.0, 0.0, 1.0)

    assert [event.class_name for event in ctx.evt_folder.events] == [
        "EvtParam",
        "EvtLod",
    ]


def test_purge_deletes_objects_and_clears_pool_references():
    ctx = _context()
    target = _Target("load")
    ctx.preallocate_load_events(target, 2)

    ctx.purge_events()

    assert ctx.app.reset_count == 1
    assert ctx.evt_folder.events == []
    assert ctx.event_pool_stats() == {
        "param_total": 0,
        "param_used": 0,
        "load_total": 0,
        "load_used": 0,
        "tap_total": 0,
        "tap_used": 0,
        "discovered": 0,
        "created": 0,
        "pending_admission": 0,
        "admission_executes": 0,
        "retired": 0,
    }

def test_persistent_pool_grows_admits_and_retires_events():
    ctx = _context(strict=False, persistent=True)
    ctx._calculation_active = True
    param_target = _Target("qvpre")
    load_target = _Target("load")
    tap_target = _Target("trafo")

    ctx.add_param_event(param_target, "qset", 0.1, 1.0)
    ctx.add_load_event(load_target, 2.0, -1.0, 1.0)
    ctx.add_tap_event(tap_target, 1, 1.0)

    assert len(ctx.evt_folder.events) == 3
    assert ctx.event_pool_stats()["pending_admission"] == 3
    assert ctx.event_pool_stats()["created"] == 3
    assert all(
        event.loc_name.startswith("qofo_pool_")
        for event in ctx.evt_folder.events
    )

    calls = ctx.admit_new_events(0.0, batch_size=1)
    assert calls == 3
    assert ctx.sim.execute_count == 3
    assert ctx.sim.attributes["tstop"] == 0.0
    assert ctx.event_pool_stats()["pending_admission"] == 0

    ctx.simulate(2.0)
    assert ctx.sim.execute_count == 4
    assert all(
        event.attributes["time"] == EVENT_INERT_TIME_S
        for event in ctx.evt_folder.events
    )
    assert ctx.event_pool_stats()["retired"] == 3


def test_prepare_discovers_existing_pool_and_removes_unmanaged_events():
    ctx = _context(strict=False, persistent=True)
    param_target = _Target("qvpre")
    load_target = _Target("load")
    tap_target = _Target("trafo")
    ctx.ensure_param_event_capacity(param_target, "qset", 1)
    ctx.ensure_load_event_capacity(load_target, 1)
    ctx._append_tap_slot(tap_target)
    retained = tuple(ctx.evt_folder.events)
    for event in retained:
        event.SetAttribute("time", 12.0)
    ctx.evt_folder.CreateObject("EvtParam", "foreign_event")

    stats = ctx.prepare_persistent_event_pool()

    assert ctx.app.reset_count == 1
    assert tuple(ctx.evt_folder.events) == retained
    assert stats["unmanaged_removed"] == 1
    assert stats["discovered"] == 3
    assert stats["created"] == 0
    assert stats["param_total"] == 1
    assert stats["load_total"] == 1
    assert stats["tap_total"] == 1
    assert all(
        event.attributes["time"] == EVENT_INERT_TIME_S
        for event in retained
    )
