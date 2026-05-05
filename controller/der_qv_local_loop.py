"""
DER plant-side reactive-power controllers (refactor_v2).
========================================================

Pandapower controllers that simulate each DER's converter response in
steady state — one controller per ``net.sgen`` row.  Two flavours are
provided, dispatched by ``net.sgen.q_mode``:

* :class:`QVLocalLoop` (``q_mode == "qv"``) — piecewise-linear V_Q
  characteristic with optional symmetric deadband, shifted horizontally
  by ``V_cor = q_cor_mvar / R``.  Matches Soleimani & Van Cutsem,
  *Combined Local and Centralized Voltage Control*, eq. (1)–(2) for the
  shifted curve and eq. (15)/(17)/(18) for the linearised closed-loop
  response.

* :class:`CosPhiConstLoop` (``q_mode == "cosphi"``) — fixed power factor
  ``Q = sign · |P| · tan(acos(cosφ))`` with ``sign ∈ {+1, −1}``
  (over- vs. under-excited).  Q is independent of voltage; the
  controller does not see Q_cor (cos-phi DERs are excluded from the
  OFO action vector).

Both loops read their parameters from columns on ``net.sgen`` populated
by :func:`network.ieee39.build.tag_der_q_modes`:

  q_mode, qv_slope_pu, qv_vref_pu, qv_deadband_pu, cosphi, cosphi_sign,
  q_cor_mvar.

The convergence loop uses damped fixed-point iteration to keep the
inner ``run_control=True`` loop stable for STATCOM-class units.

Backward compatibility
----------------------
For networks that have only the legacy ``vm_pu_ref`` column (the
pre-refactor Stage-2 system), :class:`QVLocalLoop` falls back to
reading ``vm_pu_ref`` and treats ``q_cor_mvar`` and ``qv_deadband_pu``
as zero.  This preserves the original Q = clip(−k·(V−V_ref), Q_min, Q_max)
behaviour so existing call sites that have not yet migrated to
``tag_der_q_modes`` keep working.

The legacy module path ``controller.dso_qv_local_loop`` is preserved
as a thin shim that re-exports from this module.

Author: Manuel Schwenke
Date: 2026-05-05 (refactor_v2 commit 3)
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.control.basic_controller import Controller


# ---------------------------------------------------------------------------
#  Capability helper (shared)
# ---------------------------------------------------------------------------

def _qv_capability(sn: float, op_diagram: str, p_mw: float) -> tuple[float, float]:
    """Per-iteration Q-capability bounds (Mvar) from op_diagram + current P.

    Mirrors :meth:`core.actuator_bounds.ActuatorBounds._compute_single_der_q_capability`
    so the local loop and the OFO's bound computation agree exactly.
    """
    if sn <= 0.0:
        return 0.0, 0.0
    if op_diagram == "STATCOM":
        p_pu_sq = min((p_mw / sn) ** 2, 1.0)
        q_pu = math.sqrt(max(1.0 - p_pu_sq, 0.0))
        return -q_pu * sn, q_pu * sn

    # VDE-AR-N-4120-v2 (default for non-STATCOM): piecewise linear with
    # deadband below P/S_n = 0.1.
    p_ratio = abs(p_mw) / sn
    if p_ratio < 0.1:
        return 0.0, 0.0
    if p_ratio < 0.2:
        t = (p_ratio - 0.1) / 0.1
        q_min = (-0.10 + t * (-0.33 - (-0.10))) * sn
        q_max = (0.10 + t * (0.41 - 0.10)) * sn
        return q_min, q_max
    return -0.33 * sn, 0.41 * sn


def compute_qcor_h_transform(
    K_diag: np.ndarray,
    S_VQ: np.ndarray,
) -> Optional[np.ndarray]:
    """Soleimani §IV-B eq. (18): closed-loop sensitivity from Q_cor to
    realised Q under local Q(V) droop.

    ::

        T' = (I + diag(K) · S_VQ)^{-1}

    where:

    * ``K_diag`` (length n_b) — per-bus droop gain ``K_b = sum_i (S_n,i / slope_i)``
      summed across DERs hosted at that bus (Mvar / pu_v).  Saturated DERs
      contribute zero (active-set: at the rail Q does not respond).
    * ``S_VQ`` (n_b × n_b) — bus-to-bus voltage-Q sensitivity at the DER
      buses (pu_v / Mvar).
    * ``T'`` (n_b × n_b) — multiplied into the DER columns of the OFO's
      H matrix to map ``∂y/∂Q`` ⇒ ``∂y/∂Q_cor``.

    Returns ``None`` on singular ``M = I + diag(K) · S_VQ`` so the caller
    can fall back to identity.
    """
    n = len(K_diag)
    if n == 0:
        return np.zeros((0, 0))
    M = np.eye(n) + np.diag(K_diag) @ S_VQ
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return None


def _read_or(net: pp.pandapowerNet, sgen_idx: int, col: str,
             fallback: float) -> float:
    """Read net.sgen[col] for a row, returning *fallback* if column is
    missing or value is NaN.  Used to keep legacy networks (without the
    refactor_v2 columns) working with the new controllers."""
    if col not in net.sgen.columns:
        return fallback
    v = net.sgen.at[sgen_idx, col]
    if pd.isna(v):
        return fallback
    return float(v)


# ---------------------------------------------------------------------------
#  QVLocalLoop — piecewise-linear V_Q with deadband and Q_cor offset
# ---------------------------------------------------------------------------

class QVLocalLoop(Controller):
    """One pandapower controller per ``q_mode == "qv"`` DER sgen.

    Reads parameters from columns on ``net.sgen`` each PF iteration so
    the OFO can update ``q_cor_mvar`` (and, in principle, the V_ref /
    slope / deadband settings) between simulation steps without
    re-creating the controller object.

    Steady-state target Q (Soleimani §III-A, eq. (2), with V_cor shift):

    ::

        V_eff = V − V_ref − V_cor    where V_cor = q_cor_mvar / R
        Q_d(V_eff) =
            −R · (V_eff − db)        if V_eff >  db
             0                       if |V_eff| ≤ db
            −R · (V_eff + db)        if V_eff < −db
        Q = clip(Q_d, Q_min(P), Q_max(P))

    where ``R = S_n / qv_slope_pu`` (Mvar/pu_v) and ``db = qv_deadband_pu``.

    Setting ``db = 0`` recovers the smooth linear droop through V_ref.
    """

    def __init__(
        self,
        net: pp.pandapowerNet,
        sgen_idx: int,
        slope_pu: float = 0.07,
        damping: float = 0.5,
        max_step_frac: Optional[float] = 1.0,
        tol_mvar: float = 0.01,
        in_service: bool = True,
        order: int = 0,
        level: int = 0,
        index: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            net,
            in_service=in_service,
            order=order,
            level=level,
            index=index,
            **kwargs,
        )
        self.sgen_idx = int(sgen_idx)
        # slope_pu is now usually read from net.sgen.qv_slope_pu each
        # iteration; the constructor arg becomes a fallback used only
        # when the column is missing (legacy networks).
        self.slope_fallback = float(slope_pu)
        self.damping = float(damping)
        self.tol_mvar = float(tol_mvar)
        # Snapshot static parameters that can't change during a PF.
        self.bus_idx = int(net.sgen.at[self.sgen_idx, "bus"])
        self.sn_mva = float(net.sgen.at[self.sgen_idx, "sn_mva"])
        if "op_diagram" in net.sgen.columns:
            od = net.sgen.at[self.sgen_idx, "op_diagram"]
            self.op_diagram = (
                str(od) if od is not None and str(od) != "nan"
                else "VDE-AR-N-4120-v2"
            )
        else:
            self.op_diagram = "VDE-AR-N-4120-v2"
        self.max_step_mvar = (
            float(max_step_frac) * self.sn_mva
            if max_step_frac is not None else None
        )
        # pandapower convention: ``applied`` flips True after the first
        # ``control_step`` so ``is_converged`` knows the iteration has
        # actually run at least once.
        self.applied = False

    # ------------------------------------------------------------------
    #  Internal: target Q at current operating point
    # ------------------------------------------------------------------

    def _compute_target(self, net: pp.pandapowerNet) -> float:
        """Return the Q target (Mvar) for the current PF state."""
        s = self.sgen_idx
        if not net.sgen.at[s, "in_service"]:
            return 0.0

        # Read V from the bus, P from the DER (active is exogenous,
        # set by the time-series profile or the runner).
        v_pu = float(net.res_bus.at[self.bus_idx, "vm_pu"])
        p_mw = float(net.res_sgen.at[s, "p_mw"])

        # Read the q_mode parameters.  Prefer the new columns; fall
        # back to the legacy vm_pu_ref / constructor slope when
        # missing (transition guarantee for unmigrated networks).
        slope_pu = _read_or(net, s, "qv_slope_pu", self.slope_fallback)
        v_ref    = _read_or(net, s, "qv_vref_pu",
                            _read_or(net, s, "vm_pu_ref", 1.0))
        q_cor    = _read_or(net, s, "q_cor_mvar", 0.0)
        db       = _read_or(net, s, "qv_deadband_pu", 0.0)

        R = self.sn_mva / slope_pu if slope_pu > 0.0 else 0.0
        v_cor = (q_cor / R) if R > 0.0 else 0.0
        v_eff = v_pu - v_ref - v_cor

        q_min, q_max = _qv_capability(self.sn_mva, self.op_diagram, p_mw)

        if v_eff > db:
            q_target = -R * (v_eff - db)
        elif v_eff < -db:
            q_target = -R * (v_eff + db)
        else:
            q_target = 0.0

        return float(np.clip(q_target, q_min, q_max))

    # ------------------------------------------------------------------
    #  pandapower Controller interface
    # ------------------------------------------------------------------

    def initialize_control(self, net: pp.pandapowerNet) -> None:
        """Reset ``applied`` so a fresh PF call starts from convergence
        check, not from the cached state of the previous step."""
        self.applied = False

    def control_step(self, net: pp.pandapowerNet) -> None:
        """Damped Q write toward the current Q(V) target.

        ``Q_{k+1} = Q_k + alpha * clip(target - Q_k, +-max_step)``.
        """
        s = self.sgen_idx
        if not net.sgen.at[s, "in_service"]:
            self.applied = True
            return
        target = self._compute_target(net)
        q_current = float(net.sgen.at[s, "q_mvar"])
        delta = target - q_current
        if self.max_step_mvar is not None:
            delta = float(np.clip(delta, -self.max_step_mvar, self.max_step_mvar))
        q_new = q_current + self.damping * delta
        net.sgen.at[s, "q_mvar"] = q_new
        self.applied = True

    def is_converged(self, net: pp.pandapowerNet) -> bool:
        """True when the un-damped error is within tolerance."""
        s = self.sgen_idx
        if not self.applied:
            return False
        if not net.sgen.at[s, "in_service"]:
            return True
        target = self._compute_target(net)
        q_current = float(net.sgen.at[s, "q_mvar"])
        return abs(target - q_current) < self.tol_mvar


# ---------------------------------------------------------------------------
#  CosPhiConstLoop — fixed power factor
# ---------------------------------------------------------------------------

class CosPhiConstLoop(Controller):
    """One pandapower controller per ``q_mode == "cosphi"`` DER sgen.

    Steady-state target Q:

    ::

        Q = cosphi_sign · |P| · tan(acos(cosphi))

    clipped to the per-DER capability envelope.

    cos-phi DERs are **excluded from the OFO action vector** (they are
    not actuators); their Q is determined purely by P and the configured
    cosphi/sign on ``net.sgen``.  Q does not depend on voltage, so the
    loop converges in a single damped step under nominal conditions.
    """

    def __init__(
        self,
        net: pp.pandapowerNet,
        sgen_idx: int,
        damping: float = 1.0,
        max_step_frac: Optional[float] = None,
        tol_mvar: float = 0.01,
        in_service: bool = True,
        order: int = 0,
        level: int = 0,
        index: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            net,
            in_service=in_service,
            order=order,
            level=level,
            index=index,
            **kwargs,
        )
        self.sgen_idx = int(sgen_idx)
        self.damping = float(damping)
        self.tol_mvar = float(tol_mvar)
        self.bus_idx = int(net.sgen.at[self.sgen_idx, "bus"])
        self.sn_mva = float(net.sgen.at[self.sgen_idx, "sn_mva"])
        if "op_diagram" in net.sgen.columns:
            od = net.sgen.at[self.sgen_idx, "op_diagram"]
            self.op_diagram = (
                str(od) if od is not None and str(od) != "nan"
                else "VDE-AR-N-4120-v2"
            )
        else:
            self.op_diagram = "VDE-AR-N-4120-v2"
        self.max_step_mvar = (
            float(max_step_frac) * self.sn_mva
            if max_step_frac is not None else None
        )
        self.applied = False

    # ------------------------------------------------------------------

    def _compute_target(self, net: pp.pandapowerNet) -> float:
        """Return the Q target (Mvar) for the current PF state."""
        s = self.sgen_idx
        if not net.sgen.at[s, "in_service"]:
            return 0.0

        cosphi = _read_or(net, s, "cosphi", 1.0)
        sign = int(round(_read_or(net, s, "cosphi_sign", -1.0)))
        p_mw = float(net.res_sgen.at[s, "p_mw"])

        # Clamp cosphi to (1e-3, 1.0] to avoid div-by-zero at cos→0.
        cosphi = max(min(cosphi, 1.0), 1e-3)
        if cosphi >= 1.0 - 1e-12:
            q_target = 0.0
        else:
            tan_phi = math.sqrt(1.0 / (cosphi * cosphi) - 1.0)
            q_target = float(sign) * abs(p_mw) * tan_phi

        q_min, q_max = _qv_capability(self.sn_mva, self.op_diagram, p_mw)
        return float(np.clip(q_target, q_min, q_max))

    def initialize_control(self, net: pp.pandapowerNet) -> None:
        self.applied = False

    def control_step(self, net: pp.pandapowerNet) -> None:
        s = self.sgen_idx
        if not net.sgen.at[s, "in_service"]:
            self.applied = True
            return
        target = self._compute_target(net)
        q_current = float(net.sgen.at[s, "q_mvar"])
        delta = target - q_current
        if self.max_step_mvar is not None:
            delta = float(np.clip(delta, -self.max_step_mvar, self.max_step_mvar))
        q_new = q_current + self.damping * delta
        net.sgen.at[s, "q_mvar"] = q_new
        self.applied = True

    def is_converged(self, net: pp.pandapowerNet) -> bool:
        s = self.sgen_idx
        if not self.applied:
            return False
        if not net.sgen.at[s, "in_service"]:
            return True
        target = self._compute_target(net)
        q_current = float(net.sgen.at[s, "q_mvar"])
        return abs(target - q_current) < self.tol_mvar


# ---------------------------------------------------------------------------
#  S_VQ self-bus cache (legacy Q-shim helper, kept for back-compat)
# ---------------------------------------------------------------------------

def cache_per_sgen_svq(
    net: pp.pandapowerNet,
    sgen_indices: Sequence[int],
    sensitivities,
) -> None:
    """Compute the self-bus voltage-Q sensitivity for each sgen and
    cache it as ``net.sgen.loc[s, 'svq_self_pu_per_mvar']``.

    Used by the legacy Q+shim apply step (Stage 2 V_ref-via-Q-shim
    path).  The refactor_v2 plant-side controllers do not need it; it
    is kept here so the legacy diagnostic scripts under ``tests/diag_*``
    continue to import successfully.

    Units note: ``JacobianSensitivities.compute_dV_dQ_der`` returns
    sensitivities in pu (``dV_pu / dQ_pu`` on the system base
    ``net.sn_mva``).  This helper divides by ``sn_mva`` once to yield
    ``dV_pu / dQ_Mvar`` which is what the Q+shim apply step needs.
    """
    if "svq_self_pu_per_mvar" not in net.sgen.columns:
        net.sgen["svq_self_pu_per_mvar"] = float("nan")

    if not sgen_indices:
        return

    der_buses_unique: list[int] = []
    for s in sgen_indices:
        b = int(net.sgen.at[int(s), "bus"])
        if b not in der_buses_unique:
            der_buses_unique.append(b)

    try:
        S_VQ_full, obs_map, der_map = sensitivities.compute_dV_dQ_der(
            der_bus_indices=der_buses_unique,
            observation_bus_indices=der_buses_unique,
        )
    except Exception:
        # Fall back to NaN — apply step will treat NaN as "use legacy
        # V_ref = V_bus + Q/k (no S_VQ correction)".
        return

    obs_perm = [obs_map.index(b) for b in der_buses_unique]
    der_perm = [der_map.index(b) for b in der_buses_unique]
    S_VQ_pu_per_pu = S_VQ_full[np.ix_(obs_perm, der_perm)]
    s_base = float(getattr(net, "sn_mva", 1.0))
    if s_base <= 0:
        s_base = 1.0
    diag = np.diag(S_VQ_pu_per_pu) / s_base
    bus_to_svq = {b: float(diag[i]) for i, b in enumerate(der_buses_unique)}

    for s in sgen_indices:
        b = int(net.sgen.at[int(s), "bus"])
        net.sgen.at[int(s), "svq_self_pu_per_mvar"] = bus_to_svq.get(b, float("nan"))


# ---------------------------------------------------------------------------
#  Installers
# ---------------------------------------------------------------------------

def install_qv_local_loops(
    net: pp.pandapowerNet,
    sgen_indices: Sequence[int],
    *,
    slope_pu: float = 0.07,
    damping: float = 0.1,
    max_step_frac: Optional[float] = 1.0,
    tol_mvar: float = 0.01,
) -> List[int]:
    """Install one :class:`QVLocalLoop` per sgen index, regardless of
    ``q_mode``.  Kept for back-compat with pre-refactor_v2 callers; new
    code should call :func:`install_der_q_loops`, which dispatches per
    sgen on ``net.sgen.q_mode``.

    Idempotent: if a controller is already attached to a given sgen,
    this function skips it (so re-calling at startup is safe).

    Returns the list of registered controller indices (newly created
    only — pre-existing ones are not re-reported).
    """
    # Legacy column maintenance: keep these for any caller that still
    # writes vm_pu_ref directly (Stage-2 Q-shim apply path).  The new
    # tag_der_q_modes does NOT touch these columns.
    if "vm_pu_ref" not in net.sgen.columns:
        net.sgen["vm_pu_ref"] = 1.03
    if "qv_local_loop" not in net.sgen.columns:
        net.sgen["qv_local_loop"] = False

    existing_sgens: set[int] = set()
    if hasattr(net, "controller") and len(net.controller) > 0:
        for _, row in net.controller.iterrows():
            obj = row["object"]
            if isinstance(obj, QVLocalLoop):
                existing_sgens.add(obj.sgen_idx)

    new_indices: List[int] = []
    for s in sgen_indices:
        s_int = int(s)
        if s_int in existing_sgens:
            continue
        if s_int not in net.sgen.index:
            continue
        cc = QVLocalLoop(
            net,
            sgen_idx=s_int,
            slope_pu=slope_pu,
            damping=damping,
            max_step_frac=max_step_frac,
            tol_mvar=tol_mvar,
        )
        net.sgen.at[s_int, "qv_local_loop"] = True
        new_indices.append(int(cc.index))
    return new_indices


def install_der_q_loops(
    net: pp.pandapowerNet,
    sgen_indices: Sequence[int],
    *,
    qv_damping: float = 0.5,
    qv_max_step_frac: Optional[float] = 1.0,
    qv_tol_mvar: float = 0.01,
    cosphi_damping: float = 1.0,
    cosphi_tol_mvar: float = 0.01,
) -> List[int]:
    """Install per-sgen plant-side controllers based on ``net.sgen.q_mode``.

    For each sgen index in *sgen_indices*:

    * ``q_mode == "qv"``     ⇒ :class:`QVLocalLoop`.
    * ``q_mode == "cosphi"`` ⇒ :class:`CosPhiConstLoop`.

    Idempotent at the (sgen, controller-class) level: an sgen that
    already has the right controller is skipped.  An sgen that has the
    *wrong* controller class (e.g. CosPhiConstLoop where QVLocalLoop is
    expected) raises :class:`ValueError` — re-classification at runtime
    is not supported in this commit; tear down and re-install.

    Returns the list of newly-created controller indices.
    """
    if "q_mode" not in net.sgen.columns:
        raise ValueError(
            "net.sgen.q_mode column missing — call "
            "network.ieee39.build.tag_der_q_modes(net, meta, ...) first."
        )

    # Inventory existing controllers so we can check per-sgen class.
    existing: dict[int, type] = {}
    if hasattr(net, "controller") and len(net.controller) > 0:
        for _, row in net.controller.iterrows():
            obj = row["object"]
            if isinstance(obj, (QVLocalLoop, CosPhiConstLoop)):
                existing[obj.sgen_idx] = type(obj)

    new_indices: List[int] = []
    for s in sgen_indices:
        s_int = int(s)
        if s_int not in net.sgen.index:
            continue
        mode = str(net.sgen.at[s_int, "q_mode"])
        if mode == "qv":
            target_cls = QVLocalLoop
            kw = dict(damping=qv_damping, max_step_frac=qv_max_step_frac,
                      tol_mvar=qv_tol_mvar)
        elif mode == "cosphi":
            target_cls = CosPhiConstLoop
            kw = dict(damping=cosphi_damping, max_step_frac=None,
                      tol_mvar=cosphi_tol_mvar)
        else:
            raise ValueError(
                f"Unknown q_mode {mode!r} on sgen {s_int}; expected "
                f"'qv' or 'cosphi'."
            )

        prev_cls = existing.get(s_int)
        if prev_cls is target_cls:
            continue
        if prev_cls is not None:
            raise ValueError(
                f"Sgen {s_int} already has a {prev_cls.__name__} but its "
                f"q_mode is {mode!r}; remove the existing controller "
                f"(remove_der_q_loops) before re-installing."
            )

        cc = target_cls(net, sgen_idx=s_int, **kw)
        new_indices.append(int(cc.index))
    return new_indices


def remove_qv_local_loops(net: pp.pandapowerNet) -> int:
    """Remove every :class:`QVLocalLoop` from ``net.controller``.

    Returns the number of controllers removed.  Used by tests that
    rebuild the controller stack between scenarios.  Does NOT remove
    :class:`CosPhiConstLoop` — use :func:`remove_der_q_loops` to remove
    every refactor_v2 plant-side loop.
    """
    if not hasattr(net, "controller") or len(net.controller) == 0:
        return 0
    drop_idx = []
    for idx, row in net.controller.iterrows():
        if isinstance(row["object"], QVLocalLoop):
            drop_idx.append(idx)
    if drop_idx:
        net.controller.drop(index=drop_idx, inplace=True)
    return len(drop_idx)


def remove_der_q_loops(net: pp.pandapowerNet) -> int:
    """Remove every :class:`QVLocalLoop` and :class:`CosPhiConstLoop`
    from ``net.controller``.  Returns the number of controllers removed."""
    if not hasattr(net, "controller") or len(net.controller) == 0:
        return 0
    drop_idx = []
    for idx, row in net.controller.iterrows():
        if isinstance(row["object"], (QVLocalLoop, CosPhiConstLoop)):
            drop_idx.append(idx)
    if drop_idx:
        net.controller.drop(index=drop_idx, inplace=True)
    return len(drop_idx)
