# 2026-08-17 — Ch 6: figures 6.2 and 6.3 merged into 6.1

**Reason:** Author request. Chapter 6 carried three figures of the same object —
the reduced network a controller linearises on — one composite (6.1) and two
zoom-ins (6.2 vertical interface, 6.3 horizontal interface). The zoom-ins
existed mainly to state one contrast: a virtual actuator is superimposed on the
vertical boundary equivalent and *not* on the horizontal one. That contrast is
stated more directly inside a single panel than across two figures.

Files are the Desktop working copies
(`\\E5UserData\folder_redirection\mschwenke\Desktop\`), not this repo — the
thesis `graphics/Ch6/` tree is not checked in here.

## 1. What changed

| file | change |
| --- | --- |
| `local_models.tex` | rewritten: panels stacked vertically, virtual actuator added to panel (a), horizontal "no virtual actuator" annotation added, shared-state band extended |
| `Chapter06.tex` | fig. 6.1 caption absorbs both deleted captions; fig. 6.2 and 6.3 float blocks deleted; 2 `\cref`s retargeted; 2 retired labels re-declared on 6.1 |
| `ds_pq_equivalent.tex`, `tso_tso_equivalent.tex` | untouched on disk, no longer `\input` by any chapter |

## 2. Structure of the merged figure

Stacked, not side by side. With the merged content the side-by-side form is
~19 × 10 units, i.e. a 0.64 shrink under `\resizebox{\textwidth}{!}`, which puts
the `\scriptsize` annotations near 4.5 pt. Stacked it is ~12.6 × 15.9 units and
renders at roughly 1:1. The float is now `[!tbp]` — at that height `htb` alone
would defer it indefinitely.

Visual grammar, applied in both panels:

* green = an actuator of the controller that owns this model (virtual actuator,
  TSO-owned tertiary bank in panel (a));
* grey = a frozen boundary equivalent, not controllable;
* white + dashed = something deliberately absent from the model.

The virtual actuator is a green `PQ` node on the $b_j$ busbar of panel (a), i.e.
an injection at the *primary* bus, superimposed on — not replacing — the
retained transformer and its constant-PQ equivalent. That is the separation of
roles the deleted fig. 6.2(b) made: the transformer contributes nothing to the
interface response and is kept for the measurement.

## 2b. Revision the same day (author feedback)

* **Network boxes carry a name only** — "transmission network of area $i$",
  "sub-transmission network of area $m$". The inventories that were in them
  (buses, lines, SGs with AVR, OLTCs, angle reference, "fully modelled") are
  either in the caption or already stated by an annotation, so the drawing was
  saying them twice.
* **The tertiary bank is now MSC *and* MSR in parallel**, each with its own
  switch arrow — a plate pair for the capacitor, a three-arc coil for the
  reactor. The old drawing showed a switched capacitor only, which
  under-specifies what `shunt_integrator.py` dispatches and what the text calls
  MSC/MSR. Drawn identically in both panels, green in (a) and grey in (b), so
  "same bank, other role" is visually true.
* **Annotations and caption cut to one claim each.** Annotation column now has
  min. gap 0.31 units (was 0.20); figure caption is down from ~2400 to ~1650
  characters, i.e. shorter than the merged captions it replaced and shorter than
  Fig. 6.1's own pre-merge caption.

## 3. Label handling

`fig:multisystem:ds_equivalent` and `fig:multisystem:tso_tso_equivalent` are
re-declared as extra `\label`s on the merged figure, after its `\caption`, so
they resolve to the merged figure number. Chapters outside Ch 6 were not
available to check for incoming references; this makes that check unnecessary.

## 4. Open inconsistency, NOT resolved here

The horizontal boundary type is described two ways in the same chapter:

* Thévenin — §6.1 (line 67) and this figure (switched 2026-08-14);
* constant PQ, explicitly *retained* — §6.3.1 and all of §6.3.2
  (`ch:architectures:multitso:boundary:hypothesis` / `:retained` / `:conclusion`).

Code state does not settle it either: `configs/config.py:579` still defaults
`tie_boundary_equivalent = "pq"`, but the live campaign baseline is
`tuning/scripts/configs/baseline_ieee39_thevenin.yaml`.

The merged figure draws the Thévenin form, following the newest revision of
fig. 6.1 and §6.1. §6.3.2's argument was left untouched — it is a substantive
claim (fidelity is not worth its information cost), not an editorial detail, and
rewriting it is an author decision. Note that constant PQ is the
$\cmplx{Z}_{\mathrm{th},c} \to \infty$ limit of what is drawn (§6.3.2, line 549),
so the figure is not wrong under either reading, only more specific than one of
them.

## 5. Not verified

No LaTeX toolchain on this server (`pdflatex`, `lualatex`, `xelatex`, `latexmk`
all absent), so the figure has **not been compiled**. Checked mechanically
instead: brace/bracket/paren balance, statement termination, every node anchor
defined, every applied style declared, and no overlap in the stacked annotation
column (min. gap 0.20 units on a conservative character-count height estimate).
Element-level collisions inside the drawing were checked by hand against the
pre-merge coordinates, which were unchanged wherever possible.
