# Chapter 9: experiment order and rerun plan

Timestamp: 2026-09-15T09:15:01+02:00
Reason: author requested that analytical weights precede closed-loop period selection, final tuning precede deadband selection, and experiment names follow this order.

## Changes and method
- Reordered Chapter09 into A (plant), B (analytical weights), C (initial subordinate response), D (final weights), E (period confirmation), F1 (routine inactivity), F2 (disturbance demonstration).
- Preserved archived measurements with explicit configuration/old-protocol qualifications. No new simulation result is claimed.
- Revised F1 to actual held anchors and filtered droop inputs, retained ordinary discrete actions, and added selected-width confirmation. Revised F2 to a six-window response demonstration with two detailed examples.
- Updated related references in Appendices B/C/D and Chapter10. Kept semantic labels and original archive paths.
- Wrote a separate executable-work plan, including conditional reuse of E and the initial/final weight mismatch.
- Source backups and edit manifest: tmp/ch9_order_20260915/before/.
- No simulation runner or controller code was modified; no simulation was launched.

## Scope and assumptions
Controlled quantities remain nodal voltages and TS-STS interface Q. Actuators remain AVR references, DER Q, taps and shunts, with physical voltage/current/capability/device constraints.
A is retained. E is reusable only for the same resolved controller; current ch9_selected differs from the dissertation archive.
Configuration unification and provenance-interface repair are planned implementation tasks, not completed changes.

## Validation
Source checks found balanced equation/table/figure/summary environments and no duplicate Chapter09 labels. All pre-existing semantic labels are retained.
An isolated full LuaLaTeX build and page inspection are pending; results will be appended here.

## Suggested Obsidian note
Parameter-selection dependency chain and rerun matrix: A -> B -> C -> D -> E -> F1 -> F2.
