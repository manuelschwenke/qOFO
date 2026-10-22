"""
pf/session.py
=============
PowerFactory Python API session helpers (engine mode or embedded).

Design constraints
------------------
* **Standalone**: imports nothing outside the standard library, so the file
  can be copied to / run on the PF machine without the qOFO_GH environment.
* **Fail-Fast**: every PF call that can fail raises :class:`PFSessionError`
  with the PF error text and a remediation hint; nothing is defaulted
  silently.
* **Logged**: every PF interaction is logged at DEBUG level on the
  ``qofo.pf`` logger (enable via ``logging.basicConfig(level=logging.DEBUG)``
  in the calling script).
* **Single engine session**: the ``powerfactory`` module supports only one
  ``GetApplication*`` call per process in engine mode; the application
  handle is therefore cached in a module-level singleton.

Execution modes
---------------
1. *Embedded* (script run inside PowerFactory via a ComPython object): the
   ``powerfactory`` module is importable directly; ``GetApplication()``
   returns the running instance.
2. *External engine mode* (plan default): a normal Python process imports
   PowerFactory's ``powerfactory`` module.  The interpreter version must
   match one of the ``Python\\3.x`` folders shipped with the PF release;
   that folder is taken from :data:`DEFAULT_PF_PYTHON_PATH` (PowerFactory
   2025 SP4, ``Python\\3.12``) unless the environment variable
   ``QOFO_PF_PYTHON_PATH`` overrides it.

See docs/pf_api_notes.md for the manual smoke test.

Author: Manuel Schwenke / Claude Code (2026-07-17)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger("qofo.pf")

#: PF database path of the project, relative to the active user account.
#: Confirmed by the user 2026-07-17: folder ``qOFO``, project ``IEEE39_qOFO``.
DEFAULT_PROJECT_PATH = r"qOFO\IEEE39_qOFO"

#: Environment variable overriding the PF Python API directory (engine mode).
ENV_PF_PYTHON_PATH = "QOFO_PF_PYTHON_PATH"

#: PF Python API directory used when the override is unset.  Machine
#: reality 2026-07-17: PowerFactory 2025 SP4, Python 3.12 (matches the
#: qOFO_clean interpreter 3.12.x).
DEFAULT_PF_PYTHON_PATH = r"C:\Program Files\DIgSILENT\PowerFactory 2025 SP4\Python\3.12"

#: Module-level application singleton (engine mode allows only one).
_APP: Any = None


class PFSessionError(RuntimeError):
    """A PowerFactory API call failed or returned an error code."""


# =====================================================================
#  Module import and application handle
# =====================================================================

def _import_powerfactory():
    """Import the ``powerfactory`` module (embedded or via env-var path)."""
    try:
        import powerfactory  # type: ignore
        logger.debug("powerfactory module already importable: %s",
                     getattr(powerfactory, "__file__", "<builtin>"))
        return powerfactory
    except ImportError:
        pass

    pf_path = os.environ.get(ENV_PF_PYTHON_PATH) or DEFAULT_PF_PYTHON_PATH
    if not os.path.isdir(pf_path):
        raise PFSessionError(
            f"PF Python API directory {pf_path!r} does not exist "
            f"(from the {ENV_PF_PYTHON_PATH} environment variable if set, "
            f"else DEFAULT_PF_PYTHON_PATH in pf/session.py). Point it at "
            f"the PF Python folder matching this interpreter, e.g. "
            r"C:\Program Files\DIgSILENT\PowerFactory 2025 SP4\Python\3.12"
        )
    logger.debug("Appending PF Python path: %s", pf_path)
    sys.path.append(pf_path)
    try:
        import powerfactory  # type: ignore
    except ImportError as exc:
        raise PFSessionError(
            f"'powerfactory' not importable from {pf_path!r}: {exc}. "
            f"Check that the folder matches this interpreter "
            f"(Python {sys.version_info.major}.{sys.version_info.minor}) "
            f"and the PF architecture (64-bit)."
        ) from exc
    return powerfactory


def get_application():
    """Return the (cached) PowerFactory application handle.

    Tries ``GetApplicationExt()`` first (raises with a descriptive message
    on failure), falls back to ``GetApplication()`` for older releases.
    """
    global _APP
    if _APP is not None:
        return _APP

    pf = _import_powerfactory()
    app = None
    err_ext: Optional[str] = None
    if hasattr(pf, "GetApplicationExt"):
        try:
            logger.debug("Calling GetApplicationExt() ...")
            app = pf.GetApplicationExt()
        except Exception as exc:  # pf.ExitError carries code + message
            err_ext = f"{type(exc).__name__}: {exc}"
            logger.debug("GetApplicationExt failed: %s", err_ext)
    if app is None:
        logger.debug("Calling GetApplication() ...")
        app = pf.GetApplication()
    if app is None:
        raise PFSessionError(
            "PowerFactory returned no application handle"
            + (f" (GetApplicationExt: {err_ext})" if err_ext else "")
            + ". Typical causes: no free licence seat (close the PF GUI or "
              "free a seat), wrong user profile, or a mismatched Python "
              "version."
        )
    _APP = app
    logger.debug("Application handle acquired: %r", app)
    return app


# =====================================================================
#  Project / study case activation
# =====================================================================

def connect(project_path: str = DEFAULT_PROJECT_PATH,
            *, study_case: Optional[str] = None):
    """Acquire the application and activate ``project_path``.

    Parameters
    ----------
    project_path : str
        PF database path relative to the active user account
        (default ``qOFO\\IEEE39_qOFO``).  A fully qualified path
        (``\\username\\qOFO\\IEEE39_qOFO``) also works.
    study_case : str, optional
        Study case ``loc_name`` to activate after project activation.

    Returns
    -------
    The PowerFactory application handle with the project activated.
    """
    app = get_application()

    logger.debug("ActivateProject(%r)", project_path)
    ierr = app.ActivateProject(project_path)
    if ierr:
        raise PFSessionError(
            f"ActivateProject({project_path!r}) returned {ierr}. Check the "
            f"path in the Data Manager (folder\\project, relative to your "
            f"user account) and that no other session holds the project."
        )
    prj = app.GetActiveProject()
    if prj is None:
        raise PFSessionError(
            f"ActivateProject({project_path!r}) reported success but "
            f"GetActiveProject() is None"
        )
    logger.debug("Active project: %s", prj.GetFullName())

    if study_case is not None:
        activate_study_case(app, study_case)
    return app


def list_study_cases(app) -> List[Any]:
    """All IntCase objects of the active project (recursive)."""
    folder = app.GetProjectFolder("study")
    if folder is None:
        raise PFSessionError(
            "GetProjectFolder('study') returned None -- is a project active?"
        )
    cases = folder.GetContents("*.IntCase", True)
    logger.debug("Found %d study case(s): %s",
                 len(cases), [c.loc_name for c in cases])
    return cases


def activate_study_case(app, name: str):
    """Activate the study case with exact ``loc_name`` ``name``.

    Idempotent: PF returns error code 1 when the case is already active
    (project activation restores the last active case), so that state is
    detected up front and treated as success.
    """
    active = app.GetActiveStudyCase()
    if active is not None and active.loc_name == name:
        logger.debug("Study case %r already active", name)
        return active
    cases = list_study_cases(app)
    matches = [c for c in cases if c.loc_name == name]
    if len(matches) != 1:
        raise PFSessionError(
            f"Study case {name!r}: found {len(matches)} matches; available: "
            f"{sorted(c.loc_name for c in cases)}"
        )
    logger.debug("Activating study case %r", name)
    ierr = matches[0].Activate()
    if ierr:
        raise PFSessionError(
            f"Activate() on study case {name!r} returned {ierr}"
        )
    return matches[0]


# =====================================================================
#  Variations (IntScheme / IntSstage)
# =====================================================================
#
# A Variation records model changes as a delta on the base grid.  The
# RMS build keeps ``base`` (variation off) and ``wind_replace`` /
# ``full`` (variation on) in one project so parity oracles never
# silently inherit each other's topology.  A single expansion stage per
# variation, activated well in the past, is active whenever the variation
# itself is active in the current study case.

#: Activation time for the single expansion stage (2000-01-01, safely
#: before any study-case time so the stage is always the recording stage).
_STAGE_ACTIVATION_EPOCH = 946684800

# When several variations are active at once (the layered
# base -> wind_replace -> full build), PowerFactory records new objects in
# the *latest* expansion stage whose activation time is <= the study-case
# time.  Equal stage times make that resolution ambiguous -- empirically PF
# then records into ``wind_replace``, which silently leaks the DSO underlay
# out of ``full`` and breaks the wind_replace-alone parity (Gate B).  Giving
# each layer a strictly increasing stage epoch makes the recording target
# deterministic: the topmost active layer always wins.  All epochs stay well
# before the 2014-05-28 study-case time.
STAGE_EPOCHS: dict[str, int] = {
    "wind_replace": 946684800,   # 2000-01-01
    "full":         1104537600,  # 2005-01-01 (> wind_replace, < study time)
}


def get_variation_folder(app):
    folder = app.GetProjectFolder("scheme")
    if folder is None:
        raise PFSessionError(
            "GetProjectFolder('scheme') returned None -- no active project?"
        )
    return folder


def find_variation(app, name: str):
    """Return the IntScheme named ``name`` or None."""
    folder = get_variation_folder(app)
    matches = [s for s in folder.GetContents("*.IntScheme", False)
               if s.loc_name == name]
    if len(matches) > 1:
        raise PFSessionError(
            f"{len(matches)} variations named {name!r}; expected at most one"
        )
    return matches[0] if matches else None


def stage_epoch_for(name: str) -> int:
    """Expansion-stage activation epoch for a named layer.

    Layers listed in :data:`STAGE_EPOCHS` get a deterministic, strictly
    ordered time so the topmost active layer owns the recording stage;
    anything else falls back to the shared default.
    """
    return STAGE_EPOCHS.get(name, _STAGE_ACTIVATION_EPOCH)


def ensure_variation(app, name: str):
    """Find-or-create an IntScheme ``name`` with one expansion stage.

    The stage activation time is taken from :func:`stage_epoch_for` so that,
    when multiple layers are active together, PowerFactory's recording stage
    resolves deterministically to the topmost layer (see :data:`STAGE_EPOCHS`).
    A pre-existing stage whose time disagrees is corrected in place -- this
    also repairs projects built before the ordering was introduced.

    Returns the IntScheme (not activated).
    """
    epoch = stage_epoch_for(name)
    scheme = find_variation(app, name)
    if scheme is None:
        folder = get_variation_folder(app)
        logger.debug("Creating variation %r", name)
        scheme = folder.CreateObject("IntScheme", name)
        if scheme is None:
            raise PFSessionError(f"CreateObject IntScheme {name!r} failed")
        if scheme.loc_name != name:
            scheme.loc_name = name
    stages = scheme.GetContents("*.IntSstage", False)
    if not stages:
        stage = scheme.CreateObject("IntSstage", "stage1")
        if stage is None:
            raise PFSessionError(f"CreateObject IntSstage in {name!r} failed")
        stage.SetAttribute("tAcTime", epoch)
    else:
        for stage in stages:
            if int(stage.GetAttribute("tAcTime")) != epoch:
                logger.debug("Correcting stage epoch of %r to %d", name, epoch)
                stage.SetAttribute("tAcTime", epoch)
    return scheme


def active_variation_names(app) -> set:
    """Names of the currently active IntSchemes."""
    return {s.loc_name for s in app.GetActiveNetworkVariations()}


def set_variation_active(app, name: str, active: bool) -> bool:
    """Activate or deactivate variation ``name`` (creating it if activating).

    Idempotent: ``Activate()``/``Deactivate()`` return a non-zero code when
    the variation is already in the requested state, so the active set is
    checked first.  Returns True when the state actually changed.
    """
    scheme = ensure_variation(app, name) if active else find_variation(app, name)
    if scheme is None:
        return False
    already = name in active_variation_names(app)
    if active and not already:
        logger.debug("Activating variation %r", name)
        ierr = scheme.Activate()
        if ierr:
            raise PFSessionError(
                f"Activate() on variation {name!r} returned {ierr}"
            )
        return True
    if not active and already:
        logger.debug("Deactivating variation %r", name)
        scheme.Deactivate()
        return True
    return False


def deactivate_variations_except(app, keep: Optional[str] = None) -> None:
    """Deactivate every active variation whose name differs from ``keep``.

    Variations are deactivated in reverse folder order so the upper ``full``
    layer is removed before its lower ``wind_replace`` layer.
    """
    active = active_variation_names(app)
    schemes = list(
        get_variation_folder(app).GetContents("*.IntScheme", False) or []
    )
    for scheme in reversed(schemes):
        if scheme.loc_name != keep and scheme.loc_name in active:
            logger.debug("Deactivating variation %r", scheme.loc_name)
            ierr = scheme.Deactivate()
            if ierr:
                raise PFSessionError(
                    f"Deactivate() on variation {scheme.loc_name!r} "
                    f"returned {ierr}"
                )


# =====================================================================
#  Object lookup and attribute access
# =====================================================================

def get_by_name(app, loc_name: str, class_name: str):
    """Exactly one calculation-relevant object ``loc_name.class_name``.

    Raises when zero or more than one object matches -- ambiguity always
    indicates a naming-convention violation (see docs/pf_naming.md) or an
    inactive grid/variation.
    """
    pattern = f"{loc_name}.{class_name}"
    objs = app.GetCalcRelevantObjects(pattern)
    exact = [o for o in objs if o.loc_name == loc_name]
    logger.debug("get_by_name(%r): %d raw / %d exact match(es)",
                 pattern, len(objs), len(exact))
    if len(exact) == 0:
        raise PFSessionError(
            f"No calculation-relevant object {pattern!r}. Check that the "
            f"grid / variation containing it is active in the current "
            f"study case."
        )
    if len(exact) > 1:
        raise PFSessionError(
            f"{len(exact)} objects match {pattern!r}: "
            f"{[o.GetFullName() for o in exact]} -- loc_names must be "
            f"unique per class (docs/pf_naming.md)."
        )
    return exact[0]


def get_all(app, class_name: str) -> List[Any]:
    """All calculation-relevant objects of one class (e.g. ``ElmTerm``)."""
    objs = app.GetCalcRelevantObjects(f"*.{class_name}")
    logger.debug("get_all(%s): %d object(s)", class_name, len(objs))
    return objs


def get_attr(obj, attribute: str) -> Any:
    """Read one attribute with logging; raises on failure."""
    try:
        value = obj.GetAttribute(attribute)
    except Exception as exc:
        raise PFSessionError(
            f"GetAttribute({attribute!r}) failed on "
            f"{getattr(obj, 'loc_name', obj)!r}: {exc}"
        ) from exc
    logger.debug("get_attr %s.%s = %r",
                 getattr(obj, "loc_name", "?"), attribute, value)
    return value


def set_attrs(obj, attributes: Mapping[str, Any]) -> None:
    """Set several attributes with old->new DEBUG logging; Fail-Fast."""
    for attr, new in attributes.items():
        try:
            old = obj.GetAttribute(attr)
        except Exception:
            old = "<unreadable>"
        try:
            obj.SetAttribute(attr, new)
        except Exception as exc:
            raise PFSessionError(
                f"SetAttribute({attr!r}, {new!r}) failed on "
                f"{getattr(obj, 'loc_name', obj)!r}: {exc}"
            ) from exc
        logger.debug("set_attr %s.%s: %r -> %r",
                     getattr(obj, "loc_name", "?"), attr, old, new)


# =====================================================================
#  Calculations
# =====================================================================

def run_ldf(app, settings: Optional[Mapping[str, Any]] = None):
    """Execute the study case's load flow (ComLdf); raise on failure.

    Parameters
    ----------
    settings : Mapping, optional
        ComLdf attributes applied (and logged) before execution, e.g.
        ``{"iopt_net": 0}``.  The parity tooling (Phase 2) owns the
        authoritative option set; this helper applies whatever it is given
        and nothing else.
    """
    ldf = app.GetFromStudyCase("ComLdf")
    if ldf is None:
        raise PFSessionError(
            "GetFromStudyCase('ComLdf') returned None -- no active study case?"
        )
    if settings:
        set_attrs(ldf, settings)
    logger.debug("Executing ComLdf ...")
    ierr = ldf.Execute()
    if ierr:
        raise PFSessionError(
            f"ComLdf.Execute() returned {ierr} (non-zero = load flow "
            f"failed; check the PF output window for the reason)."
        )
    logger.debug("ComLdf converged.")
    return ldf
