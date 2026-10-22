r"""Dump the slot/connection namespace of a WECC composite model frame.

Read-only introspection of ``WECC_WP_TSO_s0_b18``'s ``BlkDef`` frame: which
child-query variant returns the slots, and which slot/frame attributes are
non-empty.  Written against the same PF machine as ``pf/wecc_introspect.py``.

Usage (PowerFactory machine)::

    python pf\probes\probe_wecc_frame_connections.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pf.session import (  # noqa: E402
    connect,
    deactivate_variations_except,
    get_all,
    set_variation_active,
)


def safe(obj, attr):
    try:
        value = obj.GetAttribute(attr)
    except Exception as exc:
        return f"<ERR {type(exc).__name__}: {exc}>"
    if isinstance(value, list):
        return [getattr(v, "loc_name", v) for v in value]
    return getattr(value, "loc_name", value)


def main() -> None:
    app = connect(study_case="02_RMS_CoSim")
    deactivate_variations_except(app, keep=None)
    set_variation_active(app, "wind_replace", True)
    set_variation_active(app, "full", True)
    for grid in app.GetProjectFolder("netdat").GetContents("*.ElmNet"):
        if grid.loc_name.startswith("DSO_") and not grid.IsCalcRelevant():
            grid.Activate()

    comp = next(
        c for c in get_all(app, "ElmComp") if c.loc_name == "WECC_WP_TSO_s0_b18"
    )
    frame = comp.GetAttribute("typ_id")
    print("FRAME", frame.loc_name, frame.GetFullName())

    queries = (
        ("GetContents()", lambda: frame.GetContents()),
        ("GetContents('*', 1)", lambda: frame.GetContents("*", 1)),
        ("GetChildren(1, 1, '*')", lambda: frame.GetChildren(1, 1, "*")),
    )
    for label, query in queries:
        print("\n===", label, "===")
        try:
            objs = query()
        except Exception as exc:
            print("ERROR", type(exc).__name__, exc)
            continue
        print("count", len(objs))
        for obj in objs:
            print(obj.GetClassName(), repr(obj.loc_name), obj.GetFullName())

    print("\n=== SLOT COMPLETE INPUT NAMESPACE ===")
    attrs = list(app.GetAvailableAttributes("BlkSlot", "", 1, "e") or [])
    for slot in [o for o in frame.GetContents() if o.GetClassName() == "BlkSlot"]:
        print("\nSLOT", repr(slot.loc_name))
        for attr in attrs:
            value = safe(slot, attr)
            if value not in (None, "", [], 0, 0.0):
                print(f"  {attr} = {value!r}")

    print("\n=== FRAME NONEMPTY ATTRIBUTES ===")
    for namespace in ("e", "s"):
        try:
            attrs = list(
                app.GetAvailableAttributes("BlkDef", "", 1, namespace) or []
            )
        except Exception as exc:
            print("ATTR QUERY ERROR", namespace, exc)
            continue
        print("namespace", namespace, "count", len(attrs))
        for attr in attrs:
            value = safe(frame, attr)
            if value not in (None, "", [], 0, 0.0):
                print(f"  {attr} = {value!r}")


if __name__ == "__main__":
    main()
