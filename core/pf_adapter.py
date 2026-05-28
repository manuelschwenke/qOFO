"""
core/pf_adapter.py
==================
PowerFactory plant-interface adapter for the multi-zone reactive-power
control stack.

This module replaces pandapower as the *plant simulator* while leaving the
controller layer (``Measurement``, ``ZoneDefinition``, ``TSOController``)
untouched.  It mirrors the three pandapower touch-points used by
``experiments/000_M_TSO_M_DSO.py``:

    pandapower                                -> PowerFactory
    ---------------------------------------      ---------------------------
    pp.runpp(net, ...)                           PFSession.run_ldf()
    core.measurement.measure_zone_tso(net,...)   measure_zone_tso_pf(...)
    helpers.plant_io.apply_zone_tso_controls(..) apply_zone_tso_controls_pf(..)

Scope (first cut)
-----------------
* TSO-only.  DSO-side helpers (``measure_zone_dso_pf`` /
  ``apply_dso_controls_pf``) are a planned follow-up once HV sub-networks
  are modelled in PF.
* Synchronous machines (``ElmSym``), static generators / DER
  (``ElmGenstat``), switchable shunts (``ElmShnt``), 2-winding transformers
  with OLTC (``ElmTr2``), lines (``ElmLne``), terminals (``ElmTerm``).
* Balanced positive-sequence load flow only.

Design notes
------------
* ``powerfactory`` is imported lazily inside :class:`PFSession` so this
  module remains importable on machines without a PF installation (for
  type-checking, linting, CI on other OSes).
* Integer indices used by the controllers come from a stable catalogue
  (:class:`PFRegistry`) built once from the active project.  Objects are
  sorted by name, which is deterministic as long as the PF model's names
  are stable.
* No modifications to ``core.measurement.Measurement``,
  ``controller.multi_tso_coordinator.ZoneDefinition``, or any existing
  helper.  The adapter is additive.

Reference: PowerFactory Python API manual (PF 2024 SP2).  Attribute
strings (``m:u``, ``m:Q:bushv``, ``n:nntap``, ``usetp``, ``qgini``,
``ncapx``) are validated against PF 2019+ but should be probed on first
use; see :func:`probe_attribute_names` helper at the bottom of this file.

Author: Manuel Schwenke
Date: 2026-04-24
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from core.measurement import Measurement

if TYPE_CHECKING:
    from controller.base_controller import ControllerOutput
    from controller.multi_tso_coordinator import ZoneDefinition


# ---------------------------------------------------------------------------
#  Exceptions
# ---------------------------------------------------------------------------


class LoadFlowDidNotConverge(RuntimeError):
    """Raised when ``ComLdf.Execute()`` returns a non-zero error code.

    Mirrors pandapower's ``LoadflowNotConverged`` so upstream code can
    handle both plant types with the same try/except pattern.
    """


# ---------------------------------------------------------------------------
#  Default path resolution for the PF Python binding
# ---------------------------------------------------------------------------


_DEFAULT_PF_PATHS: tuple[str, ...] = (
    r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP2\Python\3.12",
    r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP1\Python\3.12",
    r"C:\Program Files\DIgSILENT\PowerFactory 2024\Python\3.12",
)


def _resolve_pf_python_path(explicit: Optional[str]) -> str:
    """Pick the first existing PF Python-binding directory.

    Preference order:
        1. ``explicit`` (argument to :class:`PFSession`), if given.
        2. Environment variable ``PF_PYTHON_PATH``.
        3. Hard-coded fallbacks in :data:`_DEFAULT_PF_PATHS`.

    Raises :class:`FileNotFoundError` if none of the candidates exist.
    """
    candidates: list[str] = []
    if explicit is not None:
        candidates.append(explicit)
    env = os.environ.get("PF_PYTHON_PATH")
    if env:
        candidates.append(env)
    candidates.extend(_DEFAULT_PF_PATHS)

    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "No PowerFactory Python binding directory found. Tried: "
        + ", ".join(candidates)
        + ". Pass pf_python_path=... to PFSession or set PF_PYTHON_PATH."
    )


# ---------------------------------------------------------------------------
#  PFSession — lifecycle wrapper around the powerfactory application
# ---------------------------------------------------------------------------


class PFSession:
    """Context-managed wrapper around ``powerfactory.GetApplicationExt()``.

    Usage
    -----
    >>> with PFSession("Nordic_SM", study_case="Base Case") as session:
    ...     registry = build_pf_registry_from_project(session)
    ...     session.run_ldf()
    ...     meas = measure_zone_tso_pf(session, registry, zone_def, it=0)

    The adapter assumes engine mode (external Python process).  If you run
    scripts inside PF itself, swap ``GetApplicationExt`` for
    ``GetApplication`` -- everything else works the same way.

    Parameters
    ----------
    project_name
        Name (or path) of the PF project to activate.
    study_case
        Study-case name.  If ``None``, the project's currently active study
        case is used.
    pf_python_path
        Override for the PowerFactory Python-binding directory.  If
        ``None``, resolved via :func:`_resolve_pf_python_path`.
    username, password
        Optional credentials for PF user-login in engine mode.
    """

    def __init__(
        self,
        project_name: str,
        *,
        study_case: Optional[str] = None,
        pf_python_path: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.project_name = project_name
        self.study_case = study_case
        self.pf_python_path = pf_python_path
        self.username = username
        self.password = password

        self._pf = None  # powerfactory module (lazy import)
        self._app: Any = None  # IntApplication
        self._project: Any = None  # IntPrj
        self._ldf: Any = None  # ComLdf cached handle

    # -- context-manager protocol -----------------------------------------
    def __enter__(self) -> "PFSession":
        path = _resolve_pf_python_path(self.pf_python_path)
        if path not in sys.path:
            sys.path.append(path)

        import powerfactory  # type: ignore[import-not-found]

        self._pf = powerfactory
        if self.username is not None:
            self._app = powerfactory.GetApplicationExt(self.username, self.password)
        else:
            self._app = powerfactory.GetApplicationExt()
        if self._app is None:
            raise RuntimeError("GetApplicationExt() returned None.")

        ierr = self._app.ActivateProject(self.project_name)
        if ierr != 0:
            raise RuntimeError(
                f"ActivateProject('{self.project_name}') returned ierr={ierr}"
            )
        self._project = self._app.GetActiveProject()
        if self._project is None:
            raise RuntimeError(f"Failed to activate project '{self.project_name}'.")

        if self.study_case is not None:
            sc = self._find_study_case(self.study_case)
            if sc is None:
                raise RuntimeError(
                    f"Study case '{self.study_case}' not found in project "
                    f"'{self.project_name}'."
                )
            sc.Activate()

        self._ldf = self._app.GetFromStudyCase("ComLdf")
        if self._ldf is None:
            raise RuntimeError("Could not obtain ComLdf from active study case.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # ``Exit`` only exists in engine mode and can fail silently; the
        # Python process holding the COM reference ending is sufficient.
        try:
            if self._app is not None and hasattr(self._app, "Exit"):
                self._app.Exit()
        except Exception:
            pass

    # -- load-flow --------------------------------------------------------
    def run_ldf(
        self,
        *,
        balanced: bool = True,
        reset_calc: bool = False,
    ) -> bool:
        """Execute ``ComLdf`` once and return ``True`` on convergence.

        Parameters
        ----------
        balanced
            ``True`` (default) -> balanced positive-sequence LDF
            (``iopt_net = 0``).  ``False`` -> unbalanced ABC
            (``iopt_net = 1``).
        reset_calc
            If ``True``, call ``app.ResetCalculation()`` before executing
            the LDF (clears cached results from previous runs).  Rarely
            needed; included for parity with pandapower ``init='auto'``.

        Raises
        ------
        LoadFlowDidNotConverge
            If ``ComLdf.Execute()`` returns a non-zero error code.
        """
        if self._app is None or self._ldf is None:
            raise RuntimeError("PFSession not entered (use 'with PFSession(...) as s').")

        if reset_calc and hasattr(self._app, "ResetCalculation"):
            self._app.ResetCalculation()

        try:
            self._ldf.SetAttribute("iopt_net", 0 if balanced else 1)
        except Exception:
            # Attribute name identical from PF 2019 onward; swallow for
            # forward/back-compat — wrong mode just means the LDF runs in
            # the study-case default.
            pass

        ierr = self._ldf.Execute()
        if ierr != 0:
            raise LoadFlowDidNotConverge(
                f"ComLdf.Execute() returned ierr={ierr}."
            )
        return True

    # -- accessors --------------------------------------------------------
    @property
    def app(self) -> Any:
        return self._app

    @property
    def project(self) -> Any:
        return self._project

    @property
    def ldf(self) -> Any:
        return self._ldf

    @property
    def pf(self) -> Any:
        return self._pf

    # -- internals --------------------------------------------------------
    def _find_study_case(self, name: str) -> Any:
        sc_folder = self._app.GetProjectFolder("study")
        if sc_folder is None:
            return None
        for sc in sc_folder.GetContents("*.IntCase", 1):
            if str(sc.loc_name) == name:
                return sc
        return None


# ---------------------------------------------------------------------------
#  PFRegistry — stable integer-index catalogue of PF objects
# ---------------------------------------------------------------------------


@dataclass
class PFRegistry:
    """Immutable catalogue mapping integer indices to PowerFactory objects.

    The integer keys are the same labels that populate
    :class:`ZoneDefinition` (``bus_indices``, ``gen_indices``,
    ``tso_der_indices`` …) and :class:`core.measurement.Measurement`.
    They are assigned at build-time by sorting the PF objects of each
    class by ``loc_name``.  As long as the PF model's object names are
    stable, the registry reproduces the same mapping across sessions.

    Attributes
    ----------
    bus_by_idx, line_by_idx, ...
        ``Dict[int, pf.DataObject]`` per object class.
    bus_name_by_idx, ...
        Parallel ``Dict[int, str]`` for diagnostics and human-readable
        logging.
    reverse
        ``Dict[pf.DataObject, tuple[str, int]]`` — the object-back-to-
        ``(class_tag, idx)`` reverse lookup, used by the apply helpers.
    project_name
        The PF project this registry was built against (diagnostic).
    """

    bus_by_idx: Dict[int, Any] = field(default_factory=dict)
    line_by_idx: Dict[int, Any] = field(default_factory=dict)
    trafo2w_by_idx: Dict[int, Any] = field(default_factory=dict)
    trafo3w_by_idx: Dict[int, Any] = field(default_factory=dict)
    sym_by_idx: Dict[int, Any] = field(default_factory=dict)
    sgen_by_idx: Dict[int, Any] = field(default_factory=dict)
    shunt_by_bus_idx: Dict[int, Any] = field(default_factory=dict)
    slack_obj: Optional[Any] = None

    bus_name_by_idx: Dict[int, str] = field(default_factory=dict)
    line_name_by_idx: Dict[int, str] = field(default_factory=dict)
    trafo2w_name_by_idx: Dict[int, str] = field(default_factory=dict)
    trafo3w_name_by_idx: Dict[int, str] = field(default_factory=dict)
    sym_name_by_idx: Dict[int, str] = field(default_factory=dict)
    sgen_name_by_idx: Dict[int, str] = field(default_factory=dict)

    reverse: Dict[int, tuple[str, int]] = field(default_factory=dict)
    """Maps ``id(pf_obj)`` -> ``(class_tag, idx)``.  Python dicts do not
    accept PF DataObject as key reliably (no stable ``__hash__``), hence
    ``id(...)`` is used."""

    project_name: str = ""

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"PFRegistry[{self.project_name}]: "
            f"{len(self.bus_by_idx)} buses, "
            f"{len(self.line_by_idx)} lines, "
            f"{len(self.trafo2w_by_idx)} 2W trafos, "
            f"{len(self.trafo3w_by_idx)} 3W trafos, "
            f"{len(self.sym_by_idx)} sync gens, "
            f"{len(self.sgen_by_idx)} static gens, "
            f"{len(self.shunt_by_bus_idx)} shunts"
        )


def _sorted_by_name(objs: Iterable[Any]) -> list[Any]:
    """Sort a list of PF DataObjects by ``loc_name`` (stable, case-insensitive)."""
    return sorted(objs, key=lambda o: str(o.loc_name).lower())


def _build_name_lookup(session: PFSession, class_filter: str) -> list[Any]:
    """Return the list of calc-relevant objects of the given class, sorted."""
    objs = session.app.GetCalcRelevantObjects(class_filter)
    if objs is None:
        return []
    return _sorted_by_name(list(objs))


def build_pf_registry_from_project(
    session: PFSession,
    *,
    shunt_bus_attr: str = "bus1",
    name_overrides: Optional[Dict[str, List[str]]] = None,
) -> PFRegistry:
    """Discover calc-relevant objects in the active PF project and index them.

    Parameters
    ----------
    session
        An *entered* :class:`PFSession`.
    shunt_bus_attr
        Attribute name on ``ElmShnt`` holding the host terminal reference.
        PF 2019+ exposes it as ``bus1`` (a ``StaCubic``).  The registry
        resolves ``cubicle.GetAttribute('cterm')`` to the ``ElmTerm`` and
        uses that terminal's bus index as the shunt key -- matching
        ``DSOControllerConfig.shunt_bus_indices`` semantics.
    name_overrides
        Optional ``{class_tag: [name_1, name_2, ...]}`` to pin the
        integer order for a class (e.g. ``{"ElmSym": ["G1", "G2", ...]}``
        fixes generator indices to match a reference numbering).  Objects
        not in the override list are appended in sorted order after those
        that are.

    Returns
    -------
    registry
        Populated :class:`PFRegistry`.

    Notes
    -----
    No load flow is run by this function; only the *topology* is
    inspected.  Call :meth:`PFSession.run_ldf` before any
    ``measure_zone_tso_pf(...)``.
    """
    registry = PFRegistry(project_name=str(session.project.loc_name))

    def _apply_override(objs: list[Any], class_tag: str) -> list[Any]:
        if not name_overrides or class_tag not in name_overrides:
            return objs
        want = list(name_overrides[class_tag])
        by_name = {str(o.loc_name): o for o in objs}
        ordered: list[Any] = []
        for n in want:
            if n in by_name:
                ordered.append(by_name.pop(n))
        # Append any remaining (not mentioned in override) in sorted order
        ordered.extend(_sorted_by_name(by_name.values()))
        return ordered

    # -- Terminals (buses) ------------------------------------------------
    buses = _apply_override(_build_name_lookup(session, "*.ElmTerm"), "ElmTerm")
    for idx, obj in enumerate(buses):
        registry.bus_by_idx[idx] = obj
        registry.bus_name_by_idx[idx] = str(obj.loc_name)
        registry.reverse[id(obj)] = ("ElmTerm", idx)

    # -- Lines ------------------------------------------------------------
    lines = _apply_override(_build_name_lookup(session, "*.ElmLne"), "ElmLne")
    for idx, obj in enumerate(lines):
        registry.line_by_idx[idx] = obj
        registry.line_name_by_idx[idx] = str(obj.loc_name)
        registry.reverse[id(obj)] = ("ElmLne", idx)

    # -- 2W transformers --------------------------------------------------
    tr2 = _apply_override(_build_name_lookup(session, "*.ElmTr2"), "ElmTr2")
    for idx, obj in enumerate(tr2):
        registry.trafo2w_by_idx[idx] = obj
        registry.trafo2w_name_by_idx[idx] = str(obj.loc_name)
        registry.reverse[id(obj)] = ("ElmTr2", idx)

    # -- 3W transformers --------------------------------------------------
    tr3 = _apply_override(_build_name_lookup(session, "*.ElmTr3"), "ElmTr3")
    for idx, obj in enumerate(tr3):
        registry.trafo3w_by_idx[idx] = obj
        registry.trafo3w_name_by_idx[idx] = str(obj.loc_name)
        registry.reverse[id(obj)] = ("ElmTr3", idx)

    # -- Synchronous machines --------------------------------------------
    syms = _apply_override(_build_name_lookup(session, "*.ElmSym"), "ElmSym")
    for idx, obj in enumerate(syms):
        registry.sym_by_idx[idx] = obj
        registry.sym_name_by_idx[idx] = str(obj.loc_name)
        registry.reverse[id(obj)] = ("ElmSym", idx)
        # Slack: PF flags the reference machine via ``ip_ctrl`` (int, 1 if ref).
        try:
            if int(obj.GetAttribute("ip_ctrl")) == 1:
                registry.slack_obj = obj
        except Exception:
            pass

    # -- Static generators / DER -----------------------------------------
    sgens = _apply_override(_build_name_lookup(session, "*.ElmGenstat"), "ElmGenstat")
    for idx, obj in enumerate(sgens):
        registry.sgen_by_idx[idx] = obj
        registry.sgen_name_by_idx[idx] = str(obj.loc_name)
        registry.reverse[id(obj)] = ("ElmGenstat", idx)

    # -- Switchable shunts, keyed by host-bus index ----------------------
    # ElmShnt exposes its terminal via ``bus1`` -> ``StaCubic`` ->
    # ``cterm`` (``ElmTerm``).  We map each shunt to the *bus idx* of its
    # terminal, to match DSOControllerConfig.shunt_bus_indices semantics.
    bus_obj_to_idx = {id(o): i for i, o in registry.bus_by_idx.items()}
    for shunt in _build_name_lookup(session, "*.ElmShnt"):
        try:
            cubicle = shunt.GetAttribute(shunt_bus_attr)
            terminal = cubicle.GetAttribute("cterm") if cubicle is not None else None
        except Exception:
            terminal = None
        if terminal is None:
            continue
        bus_idx = bus_obj_to_idx.get(id(terminal))
        if bus_idx is None:
            # Shunt on a bus that is not in the calc-relevant set -- skip.
            continue
        registry.shunt_by_bus_idx[bus_idx] = shunt
        registry.reverse[id(shunt)] = ("ElmShnt", bus_idx)

    return registry


# ---------------------------------------------------------------------------
#  PF attribute strings
# ---------------------------------------------------------------------------
#
# Centralised so they can be swapped in one place if a PF version exposes
# them differently.  Cross-reference with :func:`probe_attribute_names`.

_ATTR = {
    # -- result-side (``m:`` prefix = post-LDF measurement) -----------------
    "bus_vm_pu":       "m:u",          # ElmTerm voltage magnitude [p.u.]
    "line_i_ka":       "m:I:bus1",     # ElmLne current at bus1 side [kA]
    "tr2_q_hv_mvar":   "m:Q:bushv",    # ElmTr2 reactive power at HV side [Mvar]
    "tr3_q_hv_mvar":   "m:Q:bushv",    # ElmTr3 reactive power at HV side [Mvar]
    "sgen_p_mw":       "m:Psum:bus1",  # ElmGenstat active power [MW]
    "sgen_q_mvar":     "m:Qsum:bus1",  # ElmGenstat reactive power [Mvar]
    "sym_p_mw":        "m:Psum:bus1",  # ElmSym active power [MW]
    "sym_q_mvar":      "m:Qsum:bus1",  # ElmSym reactive power [Mvar]
    # -- parameter-side (no prefix = editable parameter) -------------------
    "sym_vm_setpoint": "usetp",        # ElmSym AVR voltage setpoint [p.u.]
    "tr2_tap_pos":     "nntap",        # ElmTr2 tap position (integer)
    "shunt_step":      "ncapx",        # ElmShnt current number of switched steps
    # -- writes (same parameter names, listed separately for clarity) ------
    "sgen_q_set":      "qgini",        # ElmGenstat Q dispatch in constq mode [Mvar]
    "sgen_p_set":      "pgini",        # ElmGenstat P dispatch [MW]
    "sym_vm_set":      "usetp",        # ElmSym AVR setpoint [p.u.]
    "tr2_tap_pos_set": "nntap",        # ElmTr2 tap position (integer)
    "shunt_step_set":  "ncapx",        # ElmShnt step state (integer)
}


# ---------------------------------------------------------------------------
#  Measurement helper (TSO zone)
# ---------------------------------------------------------------------------


def run_ldf_pf(session: PFSession, *, balanced: bool = True) -> bool:
    """Thin functional wrapper around :meth:`PFSession.run_ldf`.

    Provided for symmetry with the module-level ``measure_*`` / ``apply_*``
    helpers (so callers can keep a functional style).
    """
    return session.run_ldf(balanced=balanced)


def measure_zone_tso_pf(
    session: PFSession,
    registry: PFRegistry,
    zone_def: "ZoneDefinition",
    it: int,
) -> Measurement:
    """Build a TSO-level :class:`Measurement` from the current PF state.

    Mirrors :func:`core.measurement.measure_zone_tso` field-for-field --
    same ordering, same units.  Assumes :meth:`PFSession.run_ldf` has been
    called since the last actuator change.

    Parameters
    ----------
    session
        An *entered* :class:`PFSession` with a converged load flow.
    registry
        Catalogue built by :func:`build_pf_registry_from_project`.
    zone_def
        ``ZoneDefinition`` whose integer indices key into ``registry``.
    it
        OFO iteration number to stamp on the resulting ``Measurement``.

    Returns
    -------
    measurement
        Populated :class:`core.measurement.Measurement`.

    Raises
    ------
    KeyError
        If an index in ``zone_def`` is missing from the registry (model
        mismatch).  Includes the offending index in the message.
    """
    # -- All bus voltages (global, same as pandapower version) -----------
    all_bus = np.array(sorted(registry.bus_by_idx.keys()), dtype=np.int64)
    vm_all = np.empty(all_bus.size, dtype=np.float64)
    for k, idx in enumerate(all_bus):
        vm_all[k] = float(registry.bus_by_idx[int(idx)].GetAttribute(_ATTR["bus_vm_pu"]))

    # -- Line currents within the zone -----------------------------------
    i_ka = _read_vector(
        registry.line_by_idx, zone_def.line_indices,
        attr=_ATTR["line_i_ka"], class_tag="ElmLne",
        dtype=np.float64,
    )

    # -- Interface Q at PCC (3W trafos for IEEE-39 pattern) --------------
    # ZoneDefinition does not distinguish 2W vs 3W PCCs; try 3W first,
    # fall back to 2W on KeyError.  Both share the same attribute name.
    q_iface = np.zeros(len(zone_def.pcc_trafo_indices), dtype=np.float64)
    for k, t in enumerate(zone_def.pcc_trafo_indices):
        obj = registry.trafo3w_by_idx.get(int(t)) or registry.trafo2w_by_idx.get(int(t))
        if obj is None:
            raise KeyError(
                f"pcc_trafo_indices[{k}]={t} not found in registry.trafo3w_by_idx "
                f"or registry.trafo2w_by_idx."
            )
        q_iface[k] = float(obj.GetAttribute(_ATTR["tr3_q_hv_mvar"]))

    # -- TSO-DER Q and P --------------------------------------------------
    der_q = _read_vector(
        registry.sgen_by_idx, zone_def.tso_der_indices,
        attr=_ATTR["sgen_q_mvar"], class_tag="ElmGenstat",
        dtype=np.float64,
    )
    der_p = _read_vector(
        registry.sgen_by_idx, zone_def.tso_der_indices,
        attr=_ATTR["sgen_p_mw"], class_tag="ElmGenstat",
        dtype=np.float64,
    )

    # -- Shunt states -----------------------------------------------------
    shunt_states = np.zeros(len(zone_def.shunt_bus_indices), dtype=np.int64)
    for k, sb in enumerate(zone_def.shunt_bus_indices):
        obj = registry.shunt_by_bus_idx.get(int(sb))
        if obj is not None:
            shunt_states[k] = int(round(float(obj.GetAttribute(_ATTR["shunt_step"]))))

    # -- Generator setpoint (AVR), P, Q ----------------------------------
    gen_vm = _read_vector(
        registry.sym_by_idx, zone_def.gen_indices,
        attr=_ATTR["sym_vm_setpoint"], class_tag="ElmSym",
        dtype=np.float64,
    )
    gen_p = _read_vector(
        registry.sym_by_idx, zone_def.gen_indices,
        attr=_ATTR["sym_p_mw"], class_tag="ElmSym",
        dtype=np.float64,
    )
    gen_q = _read_vector(
        registry.sym_by_idx, zone_def.gen_indices,
        attr=_ATTR["sym_q_mvar"], class_tag="ElmSym",
        dtype=np.float64,
    )

    # -- OLTC tap positions (machine trafos, 2W in IEEE-39 convention) ----
    oltc_taps = np.zeros(len(zone_def.oltc_trafo_indices), dtype=np.int64)
    for k, t in enumerate(zone_def.oltc_trafo_indices):
        obj = registry.trafo2w_by_idx.get(int(t))
        if obj is None:
            raise KeyError(
                f"oltc_trafo_indices[{k}]={t} not found in registry.trafo2w_by_idx."
            )
        oltc_taps[k] = int(round(float(obj.GetAttribute(_ATTR["tr2_tap_pos"]))))

    return Measurement(
        iteration=it,
        bus_indices=all_bus,
        voltage_magnitudes_pu=vm_all,
        branch_indices=np.array(zone_def.line_indices, dtype=np.int64),
        current_magnitudes_ka=i_ka,
        interface_transformer_indices=np.array(zone_def.pcc_trafo_indices, dtype=np.int64),
        interface_q_hv_side_mvar=q_iface,
        der_indices=np.array(zone_def.tso_der_indices, dtype=np.int64),
        der_q_mvar=der_q,
        der_p_mw=der_p,
        oltc_indices=np.array(zone_def.oltc_trafo_indices, dtype=np.int64),
        oltc_tap_positions=oltc_taps,
        shunt_indices=np.array(zone_def.shunt_bus_indices, dtype=np.int64),
        shunt_states=shunt_states,
        gen_indices=np.array(zone_def.gen_indices, dtype=np.int64),
        gen_vm_pu=gen_vm,
        gen_p_mw=gen_p,
        gen_q_mvar=gen_q,
    )


# ---------------------------------------------------------------------------
#  Apply-controls helper (TSO zone)
# ---------------------------------------------------------------------------


def apply_zone_tso_controls_pf(
    session: PFSession,
    registry: PFRegistry,
    zone_def: "ZoneDefinition",
    tso_out: "ControllerOutput",
) -> None:
    """Write a TSO ``ControllerOutput`` to the active PF model.

    Mirrors :func:`experiments.helpers.plant_io.apply_zone_tso_controls`.
    Control-vector layout
        ``u = [Q_DER | Q_PCC_set | V_gen | s_OLTC]``
    PCC Q setpoints are *not* written to the plant (they are forwarded
    to DSO controllers -- same semantics as the pandapower version).

    Notes
    -----
    * No load flow is triggered here -- the caller is expected to invoke
      :meth:`PFSession.run_ldf` after applying all controls for the
      current step (to mirror the pandapower ``pp.runpp(...)`` cadence).
    * Static-generator Q is written as ``qgini`` (Mvar, absolute).  The
      adapter does *not* change ``av_mode`` (PF voltage-control mode);
      it assumes the Nordic SM model has the DER configured in a mode
      where ``qgini`` is the dispatched reactive power (typically
      ``constq``).  If a DER is in ``constv`` the write is ignored by
      PF -- explicitly document in the test harness.
    """
    u = np.asarray(tso_out.u_new, dtype=np.float64)
    n_der = len(zone_def.tso_der_indices)
    n_pcc = len(zone_def.pcc_trafo_indices)
    n_gen = len(zone_def.gen_indices)
    n_oltc = len(zone_def.oltc_trafo_indices)

    expected = n_der + n_pcc + n_gen + n_oltc
    if u.size < expected:
        raise ValueError(
            f"tso_out.u_new has {u.size} elements but zone '{zone_def.zone_id}' "
            f"expects at least {expected} (DER={n_der}, PCC={n_pcc}, "
            f"GEN={n_gen}, OLTC={n_oltc})."
        )

    off = 0
    # TS-DER Q
    for k, s_idx in enumerate(zone_def.tso_der_indices):
        obj = registry.sgen_by_idx.get(int(s_idx))
        if obj is None:
            raise KeyError(f"tso_der_indices[{k}]={s_idx} not in registry.sgen_by_idx.")
        obj.SetAttribute(_ATTR["sgen_q_set"], float(u[off + k]))
    off += n_der

    # PCC Q setpoints: forwarded to DSO controllers, NOT written to plant.
    off += n_pcc

    # AVR voltage setpoints
    for k, g_idx in enumerate(zone_def.gen_indices):
        obj = registry.sym_by_idx.get(int(g_idx))
        if obj is None:
            raise KeyError(f"gen_indices[{k}]={g_idx} not in registry.sym_by_idx.")
        obj.SetAttribute(_ATTR["sym_vm_set"], float(u[off + k]))
    off += n_gen

    # Machine-trafo OLTC taps (integer rounding to mirror pandapower)
    for k, t_idx in enumerate(zone_def.oltc_trafo_indices):
        obj = registry.trafo2w_by_idx.get(int(t_idx))
        if obj is None:
            raise KeyError(f"oltc_trafo_indices[{k}]={t_idx} not in registry.trafo2w_by_idx.")
        obj.SetAttribute(_ATTR["tr2_tap_pos_set"], int(round(float(u[off + k]))))
    off += n_oltc

    # Shunt switching omitted for IEEE-39 / Nordic-SM TSO setup (mirrors
    # ``apply_zone_tso_controls`` in plant_io.py).


# ---------------------------------------------------------------------------
#  Diagnostics
# ---------------------------------------------------------------------------


def _read_vector(
    obj_by_idx: Dict[int, Any],
    idx_list: Iterable[int],
    *,
    attr: str,
    class_tag: str,
    dtype,
) -> NDArray:
    """Read a single attribute from a list of registered objects.

    Raises :class:`KeyError` on missing indices, with a message naming
    the class tag -- caught by tests as a registry-vs-zone_def mismatch.
    """
    idx_seq = list(idx_list)
    if not idx_seq:
        return np.array([], dtype=dtype)
    out = np.empty(len(idx_seq), dtype=dtype)
    for k, i in enumerate(idx_seq):
        obj = obj_by_idx.get(int(i))
        if obj is None:
            raise KeyError(f"Index {i} not found in registry ({class_tag}).")
        out[k] = dtype(obj.GetAttribute(attr))
    return out


def probe_attribute_names(
    session: PFSession,
    registry: PFRegistry,
    *,
    sample_per_class: int = 1,
) -> Dict[str, List[str]]:
    """Return available attribute names for a sample of each class.

    Useful one-off helper when porting the adapter to a new PF version;
    call it after :func:`build_pf_registry_from_project` and compare the
    output to :data:`_ATTR`.  No modification of the model occurs.
    """
    def _names(obj: Any) -> List[str]:
        try:
            return list(obj.GetAttributeNames())  # PF-provided
        except Exception:
            try:
                return list(obj.AttributeType().keys())
            except Exception:
                return []

    result: Dict[str, List[str]] = {}
    for tag, d in (
        ("ElmTerm", registry.bus_by_idx),
        ("ElmLne", registry.line_by_idx),
        ("ElmTr2", registry.trafo2w_by_idx),
        ("ElmTr3", registry.trafo3w_by_idx),
        ("ElmSym", registry.sym_by_idx),
        ("ElmGenstat", registry.sgen_by_idx),
        ("ElmShnt", registry.shunt_by_bus_idx),
    ):
        for k, obj in list(d.items())[:sample_per_class]:
            result[f"{tag}[{k}]"] = _names(obj)
    return result
