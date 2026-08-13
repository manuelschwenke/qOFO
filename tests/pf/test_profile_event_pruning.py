from pf.screening import PERSISTENT_EVENT_PREFIXES, ScreeningContext


class _Target:
    def __init__(self, name):
        self.loc_name = name

    def GetFullName(self):
        return f"grid\\{self.loc_name}"


class _Event:
    def __init__(self, folder, class_name, name):
        self.folder = folder
        self.class_name = class_name
        self.loc_name = name
        self.attributes = {}

    def SetAttribute(self, name, value):
        self.attributes[name] = value

    def GetAttribute(self, name):
        return self.attributes[name]

    def GetClassName(self):
        return self.class_name

    def Delete(self):
        self.folder.events.remove(self)


class _Folder:
    def __init__(self):
        self.events = []

    def CreateObject(self, class_name, name):
        event = _Event(self, class_name, name)
        self.events.append(event)
        return event

    def GetContents(self):
        return list(self.events)


class _App:
    def ResetCalculation(self):
        return None


def _context():
    ctx = object.__new__(ScreeningContext)
    ctx.app = _App()
    ctx.evt_folder = _Folder()
    ctx.require_preallocated_events = False
    ctx.persistent_event_pool = True
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


def test_prepare_discards_only_load_and_pref_in_profile_slots():
    ctx = _context()
    load = _Target("profile_load")
    wgo = _Target("wgo")
    qvpre = _Target("qvpre")
    trafo = _Target("trafo")
    ctx.ensure_load_event_capacity(load, 2)
    ctx.ensure_param_event_capacity(wgo, "Pref_in", 2)
    ctx.ensure_param_event_capacity(qvpre, "qset", 2)
    ctx._append_tap_slot(trafo)

    stats = ctx.prepare_persistent_event_pool(
        discard_param_targets=((wgo, "Pref_in"),),
        discard_load_targets=(load,),
    )

    assert stats["owned_discarded"] == 4
    assert stats["load_total"] == 0
    assert stats["param_total"] == 2
    assert stats["tap_total"] == 1
    assert {
        (event.GetClassName(), event.GetAttribute("variable"))
        for event in ctx.evt_folder.events
        if event.GetClassName() == "EvtParam"
    } == {("EvtParam", "qset")}
