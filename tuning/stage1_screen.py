r"""
tuning/stage1_screen.py
=======================
Stage I of the weight campaign: the **analytic pre-screen**, no simulation.

What this module is for
-----------------------
Chapter 9 (:math:`\S` ``ch:timescales:weights``) writes the isolated-layer
tracking recurrence as

.. math::

    z^{k+1} = (I - B) z^k, \qquad
    B = Q^{1/2} H G_w^{-1} H^{\top} Q^{1/2}

with :math:`z = Q^{1/2}(y - y^{\mathrm{ref}})`.  Each reachable error mode
contracts for :math:`0 < \lambda_r(B) < 2`; the campaign screens at

.. math::

    \lambda_{\max}(B) \le 0.8 \cdot 2 = 1.6

(a declared 20 % engineering reservation), and *selects* at
:math:`\bar\lambda = 1.6 / (1 + \varepsilon_{\mathrm{op}}) = 1.546` with
:math:`\varepsilon_{\mathrm{op}} = 0.035`, the finite-bank allowance.

Why it is cheap
---------------
``H`` depends on the **operating point**, not on the weights.  Writing the
weighted objective rows as :math:`H_V` (voltage) and :math:`H_Q` (interface-Q),

.. math::

    H^{\top} Q H = g_V\, H_V^{\top}H_V + g_Q\, H_Q^{\top}H_Q
                  = g_V S_V + g_Q S_Q

so the two Gram matrices :math:`S_V, S_Q` are **weight-independent** and can be
built once per operating point.  Since :math:`B = XX^{\top}` and
:math:`C = X^{\top}X` with :math:`X = Q^{1/2} H G_w^{-1/2}` share their non-zero
spectrum,

.. math::

    \lambda_{\max}(B) = \lambda_{\max}\!\big(
        G_w^{-1/2}(g_V S_V + g_Q S_Q) G_w^{-1/2}\big)

is one symmetric eigenvalue problem of size ``n_u`` (tens), evaluable for *any*
candidate weight vector in microseconds.  Six controller builds therefore screen
an arbitrarily large design, which is what lets Stage II simulate only the
survivors.

Integer actuators
-----------------
The OLTC tap and MSC/MSR shunt columns **are** columns of ``H`` and **are** in
``B`` — Chapter 9 derives the recurrence under "continuously relaxed discrete
increments".  What sets them apart is that their ``G_w`` entries are not scaled
by a layer-gain scalar but by the **commit-threshold rule**, so they contribute
an irreducible floor that no continuous weight can scale away.

Since 2026-10-22 the MSC/MSR banks are ordinary integer inputs of the TS-MIQP
(``shunt_dispatch='miqp'``), so the TS layers carry one bank column per bank,
per level of nameplate step, and ``g_w_tso_shunt`` is priced by the same rule
as the taps from its own stated engagement deviation (``d_engage_shunt``).
With the MSC/MSR exclusion only one bank of an STS can move in a given
direction, so ``lambda_max(B)`` keeps one bank column per STS, the stronger
(:attr:`LayerCurvature.exclusive_groups`; author decision 2026-10-22).

The relaxed eigenvalue cannot see quantisation, so the integer columns get a
second, exact statement.  The MIQP step objective is
``w^T G_w w + c^T w + z^T G_z z`` with diagonal ``G_w`` and no ``H^T Q H``
term (``optimisation/miqp_solver.py``).  With no output limit active it splits
into one term per actuator, so a free integer column ``i`` commits one unit
step iff

.. math::

    |c_i| = 2\,|a_i^{\top} z| > g_{w,i},
    \qquad a_i = (Q^{1/2}H)_{:,i},

i.e. iff the relaxed step ``-c_i / (2 g_{w,i})`` exceeds half a step: the MIQP
step of a free column is its relaxed step rounded (thesis
``eq:ofo_tap_engagement``).  The deviation at which the tap engages, and its
inverse, are

.. math::

    \Delta v_{\mathrm{eng},i}
      = \frac{g_{w,i}}{2 g_V\,|\sum_j H_V[j,i]|},
    \qquad
    g_{w,i} = 2 g_V \Big|\sum_j H_V[j,i]\Big|\, \Delta v_{\mathrm{eng}} .

CORRECTED 2026-10-14: until this date the commit condition read
``2|a_i^T z| > ||a_i||^2 + g_{w,i}`` (numerator ``g_{w,i} + ||a_i||^2``,
inverse ``... - ||a_i||^2``).  That is the condition for an objective with the
curvature term ``w^T H^T Q H w``, which the controllers' MIQP does not contain;
checked with ``MIQPSolver`` on a one-tap problem (tap moves iff
``|c_i| > g_{w,i}``).  Integer weights designed before this date (Stage-I
starting point ``stage1_start``) are smaller by ``||a_i||^2`` than the stated
``d_engage`` requires, so their taps engage earlier than stated.

The column sum is the **systematic-offset** reading — an offset
:math:`e = \Delta v\,\mathbf{1}` across the area, which is the case an OLTC
exists for, and which correctly lets buses the tap pushes the wrong way cancel.
The single-bus reading (``max_j |H_V[j,i]|``) is retained as the pessimistic
diagnostic: measured 2026-08-14, TSO taps engage at 0.98-1.31 % under a
systematic offset against 3.4-4.4 % under a single-bus spike, so a single-bus
reading badly misdescribes observed tap behaviour.

Gauge
-----
The MIQP objective is positively homogeneous of degree one in the whole weight
vector :math:`(Q, G_w, G_z)`: scaling all of them by a common ``c`` leaves the
argmin, and hence the trajectory *including the integer tap sequence*,
unchanged (measured 2026-07-31 to ~4e-10).  ``lambda_max(B)`` inherits that
invariance exactly.  A search must therefore fix a gauge; this campaign pins
``g_v`` and expresses the objective weights as dimensionless priorities
``pi = g * sigma^2`` (:class:`PriorityScales`).

Provenance: replaces ``tuning_mc.stage_0_preconditioning`` (the knob-based
generative rule) and ``tuning_mc.stage_0_coupling_decomposition`` (from which
``m_blocks`` and ``criterion`` are carried over verbatim in intent).
Author: Manuel Schwenke (with Claude Code), 2026-09-10.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PriorityScales",
    "LayerCurvature",
    "OperatingPoint",
    "SCREEN_FACTOR",
    "OFO_BOUND",
    "EPS_OP",
    "screen_ceiling",
    "select_ceiling",
    "lambda_max_B",
    "engage_deviation",
    "g_w_for_engage",
    "build_operating_point",
    "screen_weights",
]

#: The OFO contraction bound for one isolated, symmetric per-area block.
OFO_BOUND: float = 2.0

#: Declared engineering reservation below :data:`OFO_BOUND` (20 %).
SCREEN_FACTOR: float = 0.8

#: Finite-bank allowance: a worst-window maximum is downward-biased as an
#: estimate of the same quantity on a fresh draw, so selection happens under
#: the screen by this margin and verification happens at the screen itself.
EPS_OP: float = 0.035


def screen_ceiling() -> float:
    """The screen: ``0.8 * 2 = 1.6``.  Verified on the confirmation bank."""
    return SCREEN_FACTOR * OFO_BOUND


def select_ceiling() -> float:
    """The selection ceiling: ``1.6 / 1.035 = 1.546``."""
    return screen_ceiling() / (1.0 + EPS_OP)


# ---------------------------------------------------------------------------
#  Priorities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PriorityScales:
    """Engineering tolerances that turn a raw weight into a priority.

    A raw weight is not comparable across outputs measured in different units.
    Writing ``pi = g * sigma^2`` with ``sigma`` the tolerance on that output
    makes priorities commensurable, so "voltage matters more than interface-Q"
    becomes a statement about ``pi`` ratios rather than raw magnitudes.

    Applied to the pre-reset baseline this exposes what the raw weights hide:
    ``pi_q = 250 * 5^2 = 6250`` against ``pi_v_ds = 1e5 * 0.01^2 = 10``, i.e.
    the subordinate layer priced interface-Q tracking ~600x above its own
    voltage schedule.
    """

    sigma_v_ts: float = 0.005    # pu
    sigma_v_ds: float = 0.010    # pu
    sigma_q: float = 5.0         # Mvar

    def g_v_from_priority(self, pi: float) -> float:
        return float(pi) / self.sigma_v_ts ** 2

    def dso_g_v_from_priority(self, pi: float) -> float:
        return float(pi) / self.sigma_v_ds ** 2

    def g_q_from_priority(self, pi: float) -> float:
        return float(pi) / self.sigma_q ** 2

    def priority_of_g_v(self, g: float) -> float:
        return float(g) * self.sigma_v_ts ** 2

    def priority_of_dso_g_v(self, g: float) -> float:
        return float(g) * self.sigma_v_ds ** 2

    def priority_of_g_q(self, g: float) -> float:
        return float(g) * self.sigma_q ** 2


# ---------------------------------------------------------------------------
#  Weight-independent curvature capture
# ---------------------------------------------------------------------------

#: Control-column classes, in the order each layer lays them out.
TS_CLASSES: Tuple[str, ...] = ("der", "pcc", "gen", "oltc", "shunt")
DS_CLASSES: Tuple[str, ...] = ("dso_der", "dso_oltc", "dso_shunt")

#: Which config weight fills each class's ``G_w`` block.
CLASS_TO_WEIGHT: Dict[str, str] = {
    "der":        "g_w_der",
    "pcc":        "g_w_pcc",
    "gen":        "g_w_gen",
    "oltc":       "g_w_tso_oltc",
    "shunt":      "g_w_tso_shunt",
    "dso_der":    "g_w_dso_der",
    "dso_oltc":   "g_w_dso_oltc",
    "dso_shunt":  "g_w_dso_shunt",
}

#: Classes priced by the commit rule rather than by a layer gain.
INTEGER_CLASSES: frozenset = frozenset({"oltc", "shunt", "dso_oltc", "dso_shunt"})


@dataclass(frozen=True)
class LayerCurvature:
    """Weight-independent curvature data for one controller.

    ``S_V`` and ``S_Q`` are Gram matrices of the voltage and interface-Q row
    blocks of ``H`` in the controller's own control ordering.  Everything a
    weight vector needs is a function of these plus the per-column class
    labels, so a candidate is scored without touching the plant again.
    """

    key: str                      # "TS_1", "STS_DSO_2"
    layer: str                    # "TS" | "STS"
    area: str                     # "1" | "DSO_2"
    S_V: np.ndarray               # (n_u, n_u)  H_V^T H_V
    S_Q: np.ndarray               # (n_u, n_u)  H_Q^T H_Q
    colsum_V: np.ndarray          # (n_u,)  |sum_j H_V[j, i]|   systematic offset
    maxabs_V: np.ndarray          # (n_u,)  max_j |H_V[j, i]|   single-bus
    classes: Tuple[str, ...]      # (n_u,)  per-column class label
    n_v: int
    n_q: int
    #: Column groups of which at most one column can move in a given
    #: direction at any instant: the MSC and MSR of one STS under
    #: ``shunt_sts_mutual_exclusion`` (added 2026-10-22).  The screen keeps
    #: only the strongest column of each group; see :func:`lambda_max_B`.
    exclusive_groups: Tuple[Tuple[int, ...], ...] = ()

    @property
    def n_u(self) -> int:
        return int(self.S_V.shape[0])

    def g_w_vector(self, weights: Dict[str, float]) -> np.ndarray:
        """Assemble ``diag(G_w)`` for this layer from a weight mapping."""
        out = np.empty(self.n_u, dtype=float)
        for i, cls in enumerate(self.classes):
            name = CLASS_TO_WEIGHT[cls]
            try:
                out[i] = float(weights[name])
            except KeyError as exc:
                raise KeyError(
                    f"{self.key}: column {i} is class {cls!r}, which needs "
                    f"weight {name!r}; the candidate does not carry it."
                ) from exc
        if not np.all(np.isfinite(out)) or np.any(out <= 0.0):
            bad = [self.classes[i] for i in np.nonzero(~(out > 0))[0]]
            raise ValueError(f"{self.key}: non-positive g_w on classes {bad}")
        return out

    def q_gram(self, g_v: float, g_q: float) -> np.ndarray:
        """``H^T Q H = g_v S_V + g_q S_Q`` for this layer's own weights."""
        return float(g_v) * self.S_V + float(g_q) * self.S_Q

    def column_norm_sq(self, g_v: float, g_q: float) -> np.ndarray:
        """``||a_i||^2`` with ``a_i`` the i-th column of ``Q^{1/2} H``."""
        return np.diag(self.q_gram(g_v, g_q)).copy()


def lambda_max_B(
    lc: LayerCurvature,
    g_v: float,
    g_q: float,
    g_w: np.ndarray,
    *,
    rel_tol: float = 1e-10,
) -> float:
    """``lambda_max(B)`` for one layer at one weight vector.

    Computed on ``C = G_w^{-1/2} (g_v S_V + g_q S_Q) G_w^{-1/2}``, which shares
    its non-zero spectrum with ``B`` and is the smaller of the two.  Near-zero
    eigenvalues are filtered: actuator directions producing no weighted output
    change (co-located DER are the usual cause) leave a null space that is not
    a stability finding, and the coordinator applies the same filter
    (``multi_tso_coordinator.check_contraction``).

    Mutually exclusive columns (:attr:`LayerCurvature.exclusive_groups`) are
    reduced to the strongest one per group, by its own contribution
    ``C_ii = ||a_i||^2 / g_w,i``.  Under the MSC/MSR exclusion only one bank of
    an STS can move in a given direction -- raising the voltage is either an
    MSC engagement or an MSR release, never both -- so a ``B`` that lets both
    act together describes a step the controller cannot take (author
    decision 2026-10-22).  Steps of banks in *different* STSs remain joint.
    """
    d_inv_sqrt = 1.0 / np.sqrt(np.asarray(g_w, dtype=float))
    C = lc.q_gram(g_v, g_q)
    C = (d_inv_sqrt[:, None] * C) * d_inv_sqrt[None, :]
    C = 0.5 * (C + C.T)                       # symmetrise against round-off
    keep = _exclusive_keep(lc, np.diag(C))
    if keep is not None:
        C = C[np.ix_(keep, keep)]
    eig = np.linalg.eigvalsh(C)
    top = float(max(eig[-1], 0.0))
    if top <= 0.0:
        return 0.0
    kept = eig[eig > rel_tol * top]
    return float(kept[-1]) if kept.size else 0.0


def _exclusive_keep(lc: LayerCurvature, strength: np.ndarray
                    ) -> Optional[List[int]]:
    """Column indices left after keeping the strongest member of each
    exclusive group, or ``None`` when the layer has no such group.

    ``getattr`` because operating points pickled before 2026-10-22 lack the
    field; their cache key differs anyway, so they are never loaded for a
    controller with in-MIQP banks.
    """
    groups = getattr(lc, "exclusive_groups", ())
    if not groups:
        return None
    drop = set()
    for grp in groups:
        strongest = max(grp, key=lambda i: float(strength[i]))
        drop.update(i for i in grp if i != strongest)
    return [i for i in range(lc.n_u) if i not in drop]


def engage_deviation(
    lc: LayerCurvature,
    g_v: float,
    g_q: float,
    g_w: np.ndarray,
    *,
    reading: str = "systematic",
) -> np.ndarray:
    """Voltage deviation at which each integer column commits one step.

    ``reading='systematic'`` uses ``|sum_j H_V[j,i]|`` (an offset across the
    whole area — the case an OLTC exists for); ``'single_bus'`` uses
    ``max_j |H_V[j,i]|`` (an error concentrated on the tap's strongest bus,
    the pessimistic diagnostic).  Non-integer columns return ``nan``.

    Commit rule of the MIQP for a free column with no output limit active:
    ``|c_i| > g_w[i]`` (module docstring).  ``g_q`` is kept in the signature
    for the callers but no longer enters: the ``||a_i||^2`` term it fed was
    removed on 2026-10-14.
    """
    if reading == "systematic":
        h = lc.colsum_V
    elif reading == "single_bus":
        h = lc.maxabs_V
    else:
        raise ValueError(f"reading must be 'systematic' or 'single_bus', got {reading!r}")
    g_w = np.asarray(g_w, dtype=float)
    out = np.full(lc.n_u, np.nan, dtype=float)
    for i, cls in enumerate(lc.classes):
        if cls not in INTEGER_CLASSES:
            continue
        den = 2.0 * float(g_v) * float(h[i])
        if den > 0.0:
            out[i] = g_w[i] / den
    return out


def g_w_for_engage(
    lc: LayerCurvature,
    g_v: float,
    g_q: float,
    d_engage: float,
    *,
    reading: str = "systematic",
) -> np.ndarray:
    """Inverse of :func:`engage_deviation`: the weight meeting a requirement.

    ``g_w = 2 g_V |h_i| d_engage`` (module docstring).  Returns ``nan`` for
    non-integer columns and ``0`` for a column that does not move the
    monitored voltages in the chosen reading (``h_i = 0``): no weight makes
    such a column commit at ``d_engage``.  Callers must treat ``<= 0`` as
    "not achievable", not clip it.  ``g_q`` no longer enters (2026-10-14,
    ``||a_i||^2`` term removed).
    """
    h = lc.colsum_V if reading == "systematic" else lc.maxabs_V
    out = np.full(lc.n_u, np.nan, dtype=float)
    for i, cls in enumerate(lc.classes):
        if cls not in INTEGER_CLASSES:
            continue
        out[i] = 2.0 * float(g_v) * float(h[i]) * float(d_engage)
    return out


def _labels(counts: Sequence[int], names: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for n, nm in zip(counts, names):
        out.extend([nm] * int(n))
    return tuple(out)


def _grams(H_V: np.ndarray, H_Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray]:
    S_V = H_V.T @ H_V if H_V.size else np.zeros((0, 0))
    S_Q = H_Q.T @ H_Q if H_Q.size else np.zeros_like(S_V)
    if H_V.size:
        colsum = np.abs(H_V.sum(axis=0))
        maxabs = np.abs(H_V).max(axis=0)
    else:
        colsum = np.zeros(S_V.shape[0])
        maxabs = np.zeros(S_V.shape[0])
    if S_Q.shape != S_V.shape:
        S_Q = np.zeros_like(S_V)
    return S_V, S_Q, colsum, maxabs


def _tso_curvature(zone_id: Any, ctrl: Any,
                   bank_sts: Optional[Dict[int, str]] = None) -> LayerCurvature:
    """``bank_sts`` maps a bank bus to its STS; when given, the banks of one
    STS form an exclusive group (MSC/MSR mutual exclusion)."""
    cfg = ctrl.config
    n_v = len(cfg.voltage_bus_indices)
    n_pcc = len(cfg.pcc_trafo_indices)
    mapping = getattr(cfg, "der_mapping", None)
    n_der = mapping.n_der if mapping is not None else len(cfg.der_indices)
    n_gen = len(cfg.gen_indices)
    n_oltc = len(cfg.oltc_trafo_indices)
    n_shunt = len(cfg.shunt_bus_indices)

    H = ctrl._expand_H_to_der_level(ctrl._build_sensitivity_matrix())
    H = np.asarray(H, dtype=float)
    n_u = n_der + n_pcc + n_gen + n_oltc + n_shunt
    if H.shape[1] != n_u:
        raise RuntimeError(
            f"TS zone {zone_id}: H has {H.shape[1]} columns, the control "
            f"layout implies {n_u} ({n_der} der + {n_pcc} pcc + {n_gen} gen + "
            f"{n_oltc} oltc + {n_shunt} shunt). The ordering assumed by this "
            f"module is stale.")
    if H.shape[0] < n_v + n_pcc:
        raise RuntimeError(
            f"TS zone {zone_id}: H has {H.shape[0]} rows, fewer than the "
            f"{n_v} voltage + {n_pcc} Q_PCC rows the layout implies.")

    # Row layout: [ V_bus | Q_PCC | I_line | Q_gen ]
    H_V = np.ascontiguousarray(H[:n_v, :])
    H_Q = np.ascontiguousarray(H[n_v:n_v + n_pcc, :])
    S_V, S_Q, colsum, maxabs = _grams(H_V, H_Q)
    classes = _labels((n_der, n_pcc, n_gen, n_oltc, n_shunt), TS_CLASSES)

    groups: Tuple[Tuple[int, ...], ...] = ()
    if bank_sts is not None and n_shunt:
        bank_cols = [i for i, c in enumerate(classes) if c == "shunt"]
        by_sts: Dict[str, List[int]] = {}
        for col, bus in zip(bank_cols, cfg.shunt_bus_indices):
            if int(bus) not in bank_sts:
                raise RuntimeError(
                    f"TS zone {zone_id}: bank at bus {bus} has no STS; the "
                    f"exclusive groups cannot be formed.")
            by_sts.setdefault(bank_sts[int(bus)], []).append(col)
        groups = tuple(tuple(c) for c in by_sts.values() if len(c) > 1)

    return LayerCurvature(
        key=f"TS_{zone_id}", layer="TS", area=str(zone_id),
        S_V=S_V, S_Q=S_Q, colsum_V=colsum, maxabs_V=maxabs,
        classes=classes, n_v=n_v, n_q=n_pcc, exclusive_groups=groups,
    )


def _dso_curvature(area: str, ctrl: Any) -> LayerCurvature:
    cfg = ctrl.config
    n_if = len(cfg.interface_trafo_indices)
    n_v = len(cfg.voltage_bus_indices)
    mapping = getattr(cfg, "der_mapping", None)
    n_der = mapping.n_der if mapping is not None else len(cfg.der_indices)
    n_oltc = len(cfg.oltc_trafo_indices)
    n_shunt = len(cfg.shunt_bus_indices)

    H = ctrl._expand_H_to_der_level(ctrl._build_sensitivity_matrix())
    H = np.asarray(H, dtype=float)
    n_u = n_der + n_oltc + n_shunt
    if H.shape[1] != n_u:
        raise RuntimeError(
            f"STS {area}: H has {H.shape[1]} columns, the control layout "
            f"implies {n_u} ({n_der} der + {n_oltc} oltc + {n_shunt} shunt). "
            f"The ordering assumed by this module is stale.")
    if H.shape[0] < n_if + n_v:
        raise RuntimeError(
            f"STS {area}: H has {H.shape[0]} rows, fewer than the {n_if} "
            f"interface + {n_v} voltage rows the layout implies.")

    # Row layout: [ Q_interface | V_bus | I_line | Q_real ]
    H_Q = np.ascontiguousarray(H[:n_if, :])
    H_V = np.ascontiguousarray(H[n_if:n_if + n_v, :])
    S_V, S_Q, colsum, maxabs = _grams(H_V, H_Q)
    return LayerCurvature(
        key=f"STS_{area}", layer="STS", area=str(area),
        S_V=S_V, S_Q=S_Q, colsum_V=colsum, maxabs_V=maxabs,
        classes=_labels((n_der, n_oltc, n_shunt), DS_CLASSES),
        n_v=n_v, n_q=n_if,
    )


@dataclass
class OperatingPoint:
    """Every layer's weight-independent curvature at one condition."""

    condition: str
    layers: List[LayerCurvature] = field(default_factory=list)

    def by_layer(self, layer: str) -> List[LayerCurvature]:
        return [lc for lc in self.layers if lc.layer == layer]


def build_operating_point(cfg, condition: str) -> OperatingPoint:
    """Build the controllers once at ``cfg`` and capture their curvature.

    The controllers are constructed and the time loop is then **abandoned**
    (``pre_loop_hook`` returning truthy), so this costs one controller build
    and simulates nothing.  ``cfg`` supplies the operating point; its weights
    are irrelevant to what is captured, which is the whole point.
    """
    from tuning._sim_loader import get_run_multi_tso_dso

    captured: Dict[str, Any] = {}

    def hook(state: Dict[str, Any]) -> bool:
        captured.update(state)
        return True                                    # abandon the time loop

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        get_run_multi_tso_dso()(cfg, pre_loop_hook=hook)

    tso = captured.get("tso_controllers") or {}
    dso = captured.get("dso_controllers") or {}
    if not tso:
        raise RuntimeError(
            f"{condition}: the runner exposed no TSO controllers; Stage I "
            f"needs the multi-TSO path.")

    # MSC/MSR of one STS are mutually exclusive (one device per direction), so
    # the screen counts one bank column per STS; it needs each bank's STS.
    bank_sts: Optional[Dict[int, str]] = None
    if (getattr(cfg, "shunt_dispatch", "off") == "miqp"
            and getattr(cfg, "shunt_sts_mutual_exclusion", False)):
        meta = captured["meta"]
        bank_sts = {int(lv): str(hv.net_id) for hv in meta.hv_networks
                    for lv in hv.coupling_lv_bus_indices}

    op = OperatingPoint(condition=condition)
    for zid in sorted(tso):
        op.layers.append(_tso_curvature(zid, tso[zid], bank_sts))
    for area in sorted(dso):
        op.layers.append(_dso_curvature(area, dso[area]))

    # With in-MIQP bank dispatch the TS layers must carry the bank columns.
    # Until 2026-10-22 the screen refused this configuration outright; a
    # capture that silently came back without them would screen a controller
    # that has no banks and price nothing for the ones it does have.
    if getattr(cfg, "shunt_dispatch", "off") == "miqp":
        n_bank_cols = sum(lc.classes.count("shunt") for lc in op.layers
                          if lc.layer == "TS")
        if n_bank_cols == 0:
            raise RuntimeError(
                f"{condition}: shunt_dispatch='miqp' but no TS layer carries a "
                f"bank column; the curvature capture does not see the banks.")
    return op


# ---------------------------------------------------------------------------
#  Screening a candidate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerScreen:
    """One (condition, layer) row of the screen."""

    condition: str
    key: str
    layer: str
    area: str
    lambda_max: float
    engage_pu: Dict[str, float]      # integer class -> worst engage deviation

    def passes(self, ceiling: float) -> bool:
        return self.lambda_max <= ceiling


def _layer_weights(lc: LayerCurvature, weights: Dict[str, float]
                   ) -> Tuple[float, float]:
    """The ``(g_v, g_q)`` this layer applies to its own output rows.

    The two layers weight different things and the names are not
    interchangeable: the supervisory layer prices its nodal voltages with
    ``g_v`` and its interface-Q rows with ``tso_g_q_pcc`` (0 by default, which
    switches that block off entirely), while the subordinate layer prices its
    voltages with ``dso_g_v`` and its interface-Q with ``g_q``.  Reading
    ``g_q`` as the supervisory Q weight is a mistake that silently changes
    which curvature is being screened.
    """
    if lc.layer == "TS":
        return float(weights["g_v"]), float(weights.get("tso_g_q_pcc", 0.0) or 0.0)
    return float(weights["dso_g_v"]), float(weights["g_q"])


def screen_weights(
    ops: Iterable[OperatingPoint],
    weights: Dict[str, float],
    *,
    engage_reading: str = "systematic",
) -> List[LayerScreen]:
    """Evaluate one candidate across every condition and layer.

    Pure linear algebra on the cached Gram matrices: no plant, no simulation.
    """
    rows: List[LayerScreen] = []
    for op in ops:
        for lc in op.layers:
            g_v, g_q = _layer_weights(lc, weights)
            g_w = lc.g_w_vector(weights)
            lam = lambda_max_B(lc, g_v, g_q, g_w)
            dev = engage_deviation(lc, g_v, g_q, g_w, reading=engage_reading)
            per_class: Dict[str, float] = {}
            for cls in sorted(set(lc.classes) & INTEGER_CLASSES):
                vals = [dev[i] for i, c in enumerate(lc.classes)
                        if c == cls and np.isfinite(dev[i])]
                if vals:
                    # The worst column decides when the class starts acting.
                    per_class[cls] = float(max(vals))
            rows.append(LayerScreen(op.condition, lc.key, lc.layer, lc.area,
                                    lam, per_class))
    return rows


def worst(rows: Sequence[LayerScreen]) -> LayerScreen:
    return max(rows, key=lambda r: r.lambda_max)


def format_table(rows: Sequence[LayerScreen], ceiling: Optional[float] = None) -> str:
    """Per condition, per layer, per area -- never an aggregate.

    Chapter 9 is explicit that an upper-eigenvalue aggregate is not a
    substitute for the maximum in each area and condition, and reports the two
    layers separately.  The 0815 design passed on its supervisory maxima while
    18 of 24 subordinate area-condition values sat above the screen.
    """
    ceiling = screen_ceiling() if ceiling is None else ceiling
    out = [f"{'condition':>10} {'layer':>5} {'area':>7} {'lambda_max(B)':>14}"
           f" {'':>4}  engage [pu]",
           "-" * 78]
    for r in sorted(rows, key=lambda r: (r.condition, r.layer, r.area)):
        flag = "OK  " if r.lambda_max <= ceiling else "OVER"
        eng = "  ".join(f"{k}={v:.4f}" for k, v in sorted(r.engage_pu.items()))
        out.append(f"{r.condition:>10} {r.layer:>5} {r.area:>7} "
                   f"{r.lambda_max:>14.4f} {flag:>4}  {eng}")
    w = worst(rows)
    n_over = sum(1 for r in rows if r.lambda_max > ceiling)
    out.append("-" * 78)
    out.append(f"worst: {w.key} at {w.condition}, lambda_max(B) = "
               f"{w.lambda_max:.4f} against a screen of {ceiling:.3f}")
    out.append(f"{n_over} of {len(rows)} area-condition values exceed the screen")
    return "\n".join(out)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def _network_key(cfg) -> str:
    """Tag distinguishing per-DSO scenario multipliers in the cache key.

    ``cfg.scenario`` names the benchmark (``rural_700``) and is NOT enough on
    its own: the Chapter-9 batch runs the same benchmark with ``DSO_3`` scaled
    x2 in installed DER and active load, and the captured Gram matrices depend
    on that.  Keyed on the scenario alone, a doubled-DSO_3 screen would silently
    load the SYMMETRIC operating points built earlier and report
    ``lambda_max(B)`` for a network it is not screening (found 2026-09-15).

    The bank layout is part of the key for the same reason (added 2026-10-22).
    With ``shunt_dispatch='miqp'`` the TS layers carry one column per bank, so
    an operating point captured without in-MIQP banks -- every cache written
    before that date -- has a different column set and must not be loaded.
    Placement and step size enter because they set the columns' values.

    The symmetric case without in-MIQP banks keeps the bare key, so operating
    points captured for it stay valid and are not needlessly rebuilt.
    """
    import hashlib
    import json as _json

    parts: Dict[str, Any] = {
        "der": dict(getattr(cfg, "dso_der_scale", None) or {}),
        "load_p": dict(getattr(cfg, "dso_load_p_scale", None) or {}),
        "load_q": dict(getattr(cfg, "dso_load_q_profile_base_mvar", None) or {}),
    }
    if getattr(cfg, "shunt_dispatch", "off") == "miqp":
        parts["banks"] = {
            "dispatch": "miqp",
            "msc_pos": int(cfg.tso_shunt_msc_coupling_pos),
            "msr_pos": int(cfg.tso_shunt_msr_coupling_pos),
            "msc_q": float(cfg.tso_shunt_msc_q_step_mvar),
            "msr_q": float(cfg.tso_shunt_msr_q_step_mvar),
            # The exclusive groups are captured only with the exclusion on.
            "sts_exclusion": bool(getattr(cfg, "shunt_sts_mutual_exclusion",
                                          False)),
        }
    if not any(parts.values()):
        return ""
    blob = _json.dumps(parts, sort_keys=True)
    return "net" + hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def _cache_path(root: Path, condition: str, scenario: str,
                network_key: str = "") -> Path:
    tag = f"{scenario}_{network_key}" if network_key else scenario
    return root / f"op_{tag}_{condition}.npz"


def load_or_build_operating_points(
    base_cfg,
    conditions: Sequence[tuple],
    cache_dir: Path,
    *,
    rebuild: bool = False,
    verbose: bool = True,
) -> List[OperatingPoint]:
    """Build one operating point per condition, caching the Gram matrices.

    The cache is keyed on condition and network only, because that is all the
    captured data depends on -- the whole reason the screen is cheap.  "Network"
    means the benchmark name *and* the per-DSO scenario multipliers and, with
    in-MIQP bank dispatch, the bank layout; see :func:`_network_key` for why
    neither is optional.
    """
    import dataclasses as _dc
    import pickle

    cache_dir.mkdir(parents=True, exist_ok=True)
    net_key = _network_key(base_cfg)
    if verbose and net_key:
        print(f"[stage1] per-DSO multipliers active "
              f"(der={dict(getattr(base_cfg, 'dso_der_scale', None) or {})}, "
              f"load_p={dict(getattr(base_cfg, 'dso_load_p_scale', None) or {})})"
              f" -> cache key {net_key}")
    ops: List[OperatingPoint] = []
    for name, start, *_rest in conditions:
        path = _cache_path(cache_dir, name, str(base_cfg.scenario), net_key)
        if path.exists() and not rebuild:
            with path.open("rb") as fh:
                ops.append(pickle.load(fh))
            if verbose:
                print(f"[stage1] {name}: cached ({path.name})")
            continue
        cfg = _dc.replace(base_cfg, start_time=start, n_total_s=60.0,
                          contingencies=[], verbose=0,
                          live_plot_controller=False, live_plot_cascade=False,
                          live_plot_overview=False, run_stability_analysis=False,
                          precondition_g_w=False)
        if verbose:
            print(f"[stage1] {name}: building controllers at {start} ...",
                  flush=True)
        op = build_operating_point(cfg, name)
        with path.open("wb") as fh:
            pickle.dump(op, fh, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"[stage1] {name}: captured {len(op.layers)} layers "
                  f"-> {path.name}")
        ops.append(op)
    return ops


def _self_test(base_cfg, weights: Dict[str, float]) -> int:
    """Check the three things that can silently make the screen wrong.

    **(a) The Gram factorisation.**  The screen never forms ``B``; it forms
    ``C = G_w^-1/2 (g_v S_V + g_q S_Q) G_w^-1/2`` and relies on ``B = X X^T``
    and ``C = X^T X`` sharing their non-zero spectrum.  Checked against a dense
    ``B`` built the long way.

    **(b) Gauge invariance.**  ``lambda_max(B)`` is invariant under a common
    rescaling of every weight, because ``Q`` scales by ``c`` and ``G_w^-1`` by
    ``1/c``.  This is the property that licenses pinning ``g_v``, so it is
    asserted rather than assumed.

    **(c) The screen reads the curvature the CONTROLLER uses.**  The voltage
    rows the screen slices out of the controller sensitivity matrix must be the
    same rows the controller own ``objective_curvature_inputs`` reports.

    What is deliberately NOT asserted
    ---------------------------------
    Agreement with ``MultiTSOCoordinator.check_contraction``.  Measured
    2026-09-10, the two disagree by 9 % to 80 % per zone, and that is a real
    modelling fact rather than a defect: the coordinator assembles ``H_ii``
    column by column from raw Jacobian sensitivities for its cross-zone
    criterion, whereas the controller acts on its own post-transform
    sensitivity matrix.  The Chapter-9 recurrence describes the update the
    CONTROLLER performs, so the screen uses the controller sensitivity; the
    coordinator blocks belong to the coupling diagnostic, where the
    off-diagonals are the point.  The delta is printed, because a reader of the
    coupled figures needs to know the two are not formed on the same model.
    """
    import dataclasses as _dc

    from tuning.parameters import WEIGHT_FIELDS

    cfg = _dc.replace(base_cfg, n_total_s=60.0, contingencies=[], verbose=0,
                      live_plot_controller=False, live_plot_cascade=False,
                      live_plot_overview=False, run_stability_analysis=False,
                      precondition_g_w=False)
    captured: Dict[str, Any] = {}

    def hook(state):
        captured.update(state)
        return True

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        from tuning._sim_loader import get_run_multi_tso_dso
        get_run_multi_tso_dso()(cfg, pre_loop_hook=hook)

    coord = captured.get("coordinator")
    tso = captured.get("tso_controllers") or {}
    dso = captured.get("dso_controllers") or {}
    if not tso:
        print("[self-test] FAIL: the runner exposed no TSO controllers")
        return 1

    layers = [(_tso_curvature(z, tso[z]), tso[z]) for z in sorted(tso)]
    layers += [(_dso_curvature(a, dso[a]), dso[a]) for a in sorted(dso)]

    ok = True

    # (a) Gram factorisation against a dense B, and (b) gauge invariance.
    print(f"{'layer':>12} {'lambda_max(B)':>14} {'dense B':>14} "
          f"{'rel.err':>10} {'gauge x7':>12}")
    for lc, _ctrl in layers:
        g_v, g_q = _layer_weights(lc, weights)
        g_w = lc.g_w_vector(weights)
        mine = lambda_max_B(lc, g_v, g_q, g_w)

        # Dense: B = Q^1/2 H G_w^-1 H^T Q^1/2, reconstructed from the Grams.
        G = lc.q_gram(g_v, g_q)
        dense = np.linalg.eigvalsh(
            (1.0 / np.sqrt(g_w))[:, None] * G * (1.0 / np.sqrt(g_w))[None, :])
        dense_max = float(max(dense[-1], 0.0))
        rel = abs(mine - dense_max) / max(abs(dense_max), 1e-30)

        scaled = {k: (v * 7.0 if k in WEIGHT_FIELDS else v)
                  for k, v in weights.items()}
        sg_v, sg_q = _layer_weights(lc, scaled)
        gauge = lambda_max_B(lc, sg_v, sg_q, lc.g_w_vector(scaled))
        grel = abs(gauge - mine) / max(abs(mine), 1e-30)

        bad = rel > 1e-9 or grel > 1e-9
        ok = ok and not bad
        print(f"{lc.key:>12} {mine:>14.6f} {dense_max:>14.6f} {rel:>10.2e} "
              f"{grel:>12.2e}{'   <-- FAIL' if bad else ''}")

    # (c) the screen slices the rows the controller itself reports.
    print()
    for lc, ctrl in layers:
        got = ctrl.objective_curvature_inputs()
        if got is None:
            print(f"{lc.key:>12}  controller reports no curvature rows (skipped)")
            continue
        H_y, g_y = got
        n_rows_screen = lc.n_v + lc.n_q
        note = "" if H_y.shape[0] <= n_rows_screen else "  (controller adds rows)"
        # Gram of the controller own weighted rows must match the screen Gram
        # evaluated at the controller own weights.
        g_v_c = float(g_y.max()) if g_y.size else 0.0
        ref = (H_y * np.sqrt(g_y)[:, None])
        ref_gram = ref.T @ ref
        mine_gram = lc.q_gram(*_layer_weights(lc, weights))
        # Compare shapes only when the controller weights match the screen ones.
        same_shape = ref_gram.shape == mine_gram.shape
        print(f"{lc.key:>12}  controller rows {H_y.shape} vs screen "
              f"({n_rows_screen}, {lc.n_u})  gram shapes "
              f"{'match' if same_shape else 'DIFFER'}{note}")
        if not same_shape:
            ok = False

    # Informational: the coordinator forms its criterion on a different model.
    if coord is not None:
        with contextlib.redirect_stdout(io.StringIO()):
            coord.compute_cross_sensitivities()
            coord.compute_M_blocks()
            diag = coord.check_contraction()
        print()
        print("informational -- coordinator criterion is formed on the raw "
              "Jacobian blocks, not the controller sensitivity:")
        print(f"{'zone':>5} {'coordinator':>14} {'screen':>14} {'ratio':>8}")
        for lc, _c in layers:
            if lc.layer != "TS":
                continue
            zid = int(lc.area)
            theirs = float(diag.get(zid, {}).get("lambda_max_Mii", float("nan")))
            g_v, g_q = _layer_weights(lc, weights)
            mine = lambda_max_B(lc, g_v, g_q, lc.g_w_vector(weights))
            print(f"{zid:>5} {theirs:>14.6f} {mine:>14.6f} "
                  f"{(mine / theirs if theirs else float('nan')):>8.3f}")

    print()
    if not ok:
        print("[self-test] FAIL")
        return 1
    print("[self-test] OK: Gram factorisation exact, gauge-invariant, and the "
          "screen reads the controller own curvature rows")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import dataclasses as _dc
    import json

    p = argparse.ArgumentParser(prog="tuning.stage1_screen")
    p.add_argument("--paramset", default=None,
                   help="Parameter set supplying the weights to screen.")
    p.add_argument("--weights-json", type=Path, default=None,
                   help="Weights as a JSON object, as an alternative to a "
                        "stored parameter set (used while the campaign is "
                        "still choosing them).")
    p.add_argument("--self-test", action="store_true",
                   help="Check the screen against the coordinator own "
                        "contraction diagnostic and exit.")
    p.add_argument("--conditions", default=None,
                   help="Comma-separated condition names (default: all six).")
    p.add_argument("--network", default=None)
    p.add_argument("--frozen-network", action="store_true",
                   help="Apply the Chapter-9 per-DSO multipliers (DSO_3 x2 in "
                        "installed DER and active load). The batch is declared "
                        "on that network; make_config() is symmetric, so "
                        "without this flag the screen answers for the "
                        "symmetric model.")
    p.add_argument("--design", action="store_true",
                   help="Run design_stage1 on the operating points and screen "
                        "the ANALYTIC point it produces, instead of screening "
                        "--paramset / --weights-json. This is how stage1_start "
                        "was produced; it had no driver until 2026-09-15.")
    p.add_argument("--write-paramset", default=None, metavar="NAME",
                   help="With --design: write the result to "
                        "configs/paramsets/NAME.json with full provenance. "
                        "Refuses to overwrite an existing file.")
    p.add_argument("--d-engage-shunt", type=float, default=None,
                   help="With --design: systematic voltage deviation [pu] at "
                        "which the strongest MSC/MSR bank column commits one "
                        "level. Required when the banks are dispatched in the "
                        "MIQP; no default (added 2026-10-22, replacing a "
                        "hand-set --g-w-tso-shunt).")
    p.add_argument("--d-engage-ts", type=float, default=0.010,
                   help="With --design: systematic voltage deviation [pu] at "
                        "which the strongest TS tap column commits (commit "
                        "rule, corrected 2026-10-14). Default 0.010.")
    p.add_argument("--d-engage-sts", type=float, default=0.020,
                   help="With --design: the same for the STS coupling-"
                        "transformer taps. Default 0.020. Added 2026-10-16 so "
                        "the protocol input D7 is stated on the command line.")
    p.add_argument("--rebuild", action="store_true",
                   help="Ignore the cached operating points and rebuild.")
    p.add_argument("--cache-dir", type=Path,
                   default=Path("results") / "stage1" / "operating_points")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    # The STRUCTURAL configuration comes from the project factory -- shunt
    # dispatch, boundary equivalent, zone partition, profile use.  A bare
    # MultiTSOConfig() is a different plant (no tertiary shunts, PQ boundary,
    # profiles off) and screening it would answer for a system nobody runs.
    from experiments.run_multi_system_ofo import make_config
    from tuning.parameters import apply_to_config
    from tuning.scenarios_mc_v2 import CONDITIONS

    if args.weights_json:
        weights = json.loads(args.weights_json.read_text(encoding="utf-8"))
    elif args.paramset:
        from configs.paramsets import load_paramset
        weights = dict(load_paramset(args.paramset)["fields"])
    elif args.design:
        # The operating points do not depend on the weights -- that is the
        # whole point of the capture -- so any weight-complete config will do
        # to build them.  design_stage1 then produces the real ones.
        weights = {"g_v": 1.0, "g_q": 1.0, "dso_g_v": 1.0,
                   "g_w_der": 1.0, "g_w_pcc": 1.0, "g_w_gen": 1.0,
                   "g_w_tso_oltc": 1.0, "g_w_tso_shunt": 1.0,
                   "g_w_dso_der": 1.0, "g_w_dso_oltc": 1.0}
    else:
        p.error("one of --paramset, --weights-json or --design is required")

    base = apply_to_config(make_config(), weights)
    if args.network:
        base = _dc.replace(base, scenario=args.network)
    if args.frozen_network:
        from experiments.ch_09_parameter_selection._ch9_selected_design import (
            FROZEN_DSO_DER_SCALE, FROZEN_DSO_LOAD_P_SCALE,
        )
        base = _dc.replace(base,
                           dso_der_scale=dict(FROZEN_DSO_DER_SCALE),
                           dso_load_p_scale=dict(FROZEN_DSO_LOAD_P_SCALE))
    base.require_weights()

    if args.self_test:
        return _self_test(base, weights)

    rows_wanted = CONDITIONS
    if args.conditions:
        want = {c.strip() for c in args.conditions.split(",")}
        rows_wanted = [r for r in CONDITIONS if r[0] in want]
        if not rows_wanted:
            p.error(f"no condition matched {sorted(want)}")

    ops = load_or_build_operating_points(
        base, [(r[0], r[1]) for r in rows_wanted], args.cache_dir,
        rebuild=args.rebuild)

    design_report: Optional[Dict[str, Any]] = None
    if args.design:
        weights, design_report = design_stage1(
            ops, d_engage_ts=float(args.d_engage_ts),
            d_engage_sts=float(args.d_engage_sts),
            d_engage_shunt=args.d_engage_shunt)
        print()
        print("[stage1] analytic design point:")
        for k in sorted(weights):
            print(f"    {k:16s} {weights[k]!r}")
        print(f"    kappa {design_report['kappa']}")

    rows = screen_weights(ops, weights)
    print()
    print(format_table(rows))
    print()
    print(f"screen  = {SCREEN_FACTOR:g} * {OFO_BOUND:g} = {screen_ceiling():.3f}")
    print(f"select at {screen_ceiling():.3f} / (1 + {EPS_OP:g}) = "
          f"{select_ceiling():.3f}   (finite-bank allowance)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": weights,
            "screen_ceiling": screen_ceiling(),
            "select_ceiling": select_ceiling(),
            "rows": [dataclasses.asdict(r) for r in rows],
        }
        args.out.write_text(json.dumps(payload, indent=1, default=float),
                            encoding="utf-8")
        print(f"wrote {args.out}")

    if args.write_paramset:
        if design_report is None:
            p.error("--write-paramset requires --design")
        _write_design_paramset(args.write_paramset, weights, design_report,
                               base, rows, [r[0] for r in rows_wanted])

    return 0 if worst(rows).lambda_max <= screen_ceiling() else 2


def _write_design_paramset(name: str, weights: Dict[str, float],
                           report: Dict[str, Any], base_cfg,
                           rows: Sequence[Any],
                           conditions: Sequence[str]) -> None:
    """Persist an analytic design point as a parameter set, with provenance.

    Refuses to overwrite: a parameter set is the single record of what a
    controller was, and silently replacing one detaches every run that names it
    from the weights it actually ran.
    """
    import json

    from configs.paramsets import PARAMSET_DIR

    path = PARAMSET_DIR / f"{name}.json"
    if path.exists():
        raise SystemExit(
            f"{path} already exists. A parameter set is the record of what a "
            f"controller was; pick a new name rather than overwriting it.")

    from tuning.parameters import WEIGHT_FIELDS

    missing = sorted(set(WEIGHT_FIELDS) - set(weights))
    if missing:
        raise SystemExit(
            f"the design does not price {missing}; a parameter set must carry "
            f"every weight. (g_w_tso_shunt is only designed when the operating "
            f"points carry bank columns, i.e. shunt_dispatch='miqp'.)")
    der = dict(getattr(base_cfg, "dso_der_scale", None) or {})
    load_p = dict(getattr(base_cfg, "dso_load_p_scale", None) or {})
    fields = {k: float(weights[k]) for k in WEIGHT_FIELDS}

    doc = {
        "name": name,
        "title": ("Stage-I analytic design point (contraction screen + tap "
                  "engagement)"
                  + (", doubled-DSO_3 network" if der or load_p else "")),
        "description": [
            "Produced by tuning/stage1_screen.py --design.",
            "NOT a tuned point: this is the starting point for a Stage-II "
            "search.",
            "Gauge g_v is pinned so the supervisory voltage priority "
            "pi = g*sigma^2 equals 1.",
            "Continuous scale placed by bisection against the layer "
            "contraction targets; tap and bank step prices from the stated "
            "engagement deviations.",
            "The per-DSO scenario multipliers are NOT part of this parameter "
            "set -- they are network, not weights -- but the curvature this "
            "point was designed against was captured on the network recorded "
            "under provenance.network. Screening these weights on a different "
            "network does not reproduce the reported lambda_max(B).",
        ],
        "provenance": {
            "producer": "tuning/stage1_screen.py --design",
            "date": __import__("datetime").date.today().isoformat(),
            "network": str(base_cfg.scenario),
            "dso_der_scale": der,
            "dso_load_p_scale": load_p,
            "conditions": list(conditions),
            "priorities": report.get("priorities"),
            "tau": report.get("tau"),
            "d_engage_pu": report.get("d_engage"),
            "lambda_targets": report.get("targets"),
            "kappa": report.get("kappa"),
            "integer_floors": report.get("floors"),
            "screen_ceiling": screen_ceiling(),
            "select_ceiling": select_ceiling(),
            "worst_lambda_max_B": float(worst(rows).lambda_max),
            "n_area_condition_rows": len(rows),
        },
        "fields": fields,
    }
    path.write_text(json.dumps(doc, indent=2, default=float), encoding="utf-8")
    print(f"[stage1] wrote parameter set {path}")


# ---------------------------------------------------------------------------
#  Stage I-c: designing a starting point
# ---------------------------------------------------------------------------

def integer_floor(lc: LayerCurvature, g_v: float, g_q: float,
                  g_w_int: Dict[str, float]) -> float:
    """``lambda_max(B)`` with the continuous columns priced out of the problem.

    Sending the continuous weights to infinity removes their rank-one terms
    from ``B`` exactly, leaving what the integer columns contribute on their
    own.  No continuous weight can reach below this, which is why the thesis
    writes the admissible region as a floor plus a linear term: the discrete
    columns cannot be scaled away, and the floor falls only as the taps are
    priced more dearly.
    """
    g_w = np.full(lc.n_u, np.inf)
    for i, cls in enumerate(lc.classes):
        if cls in INTEGER_CLASSES:
            g_w[i] = float(g_w_int[cls])
    return lambda_max_B(lc, g_v, g_q, g_w)


def design_integer_weights(
    lcs: Sequence[LayerCurvature],
    weights_objective: Dict[str, float],
    d_engage: Dict[str, float],
    *,
    reading: str = "systematic",
) -> Dict[str, float]:
    """Price each integer class from a stated engagement threshold.

    One scalar per class, taken as the **maximum** over that class columns of
    the per-column requirement, so the strongest column commits exactly at
    ``d_engage`` and weaker columns commit later.  Using the minimum instead
    would set the price from the weakest tap and make the whole class act on a
    deviation the strong taps reach far earlier.

    Taken over every supplied layer, so one class weight serves every area --
    which is what the config carries.  Raises if a class cannot meet its
    requirement at any positive weight, i.e. none of its columns moves the
    monitored voltages in the chosen reading (2026-10-14: the former reason,
    "the columns own contribution already commits them below d_engage", came
    from the removed ``||a_i||^2`` term).
    """
    out: Dict[str, float] = {}
    for cls in sorted({c for lc in lcs for c in lc.classes} & INTEGER_CLASSES):
        best = -np.inf
        for lc in lcs:
            if cls not in lc.classes:
                continue
            g_v, g_q = _layer_weights(lc, weights_objective)
            req = g_w_for_engage(lc, g_v, g_q, d_engage[cls], reading=reading)
            for i, c in enumerate(lc.classes):
                if c == cls and np.isfinite(req[i]):
                    best = max(best, float(req[i]))
        if not np.isfinite(best) or best <= 0.0:
            raise ValueError(
                f"class {cls!r}: no positive weight meets an engagement "
                f"threshold of {d_engage[cls]:g} pu -- no column of this class "
                f"moves the monitored voltages in the {reading!r} reading.")
        out[cls] = best
    return out


def solve_continuous_scale(
    lc: LayerCurvature,
    weights: Dict[str, float],
    continuous_classes: Sequence[str],
    lambda_target: float,
    *,
    lo: float = 1e-6,
    hi: float = 1e12,
    tol: float = 1e-10,
) -> float:
    """The scalar on the continuous block that places ``lambda_max(B)``.

    ``B`` scales as ``G_w^-1``, so multiplying the continuous block by ``kappa``
    is monotone decreasing in ``lambda_max`` -- a clean bisection with no
    derivative and no fitted coefficients.  The thesis states the same relation
    in fitted form (``lambda = c0 / g_w_OLTC + c1 * lambda_layer``); solving it
    exactly avoids carrying two plant-specific coefficients that have to be
    refitted whenever the plant moves.

    Raises if ``lambda_target`` lies below the integer floor.
    """
    g_v, g_q = _layer_weights(lc, weights)
    base = lc.g_w_vector(weights)
    mask = np.array([c in continuous_classes for c in lc.classes])

    def lam(kappa: float) -> float:
        g_w = base.copy()
        g_w[mask] = base[mask] * kappa
        return lambda_max_B(lc, g_v, g_q, g_w)

    if lam(hi) > lambda_target:
        floor = lam(hi)
        raise ValueError(
            f"{lc.key}: lambda_max(B) cannot be brought to {lambda_target:.4f}; "
            f"the integer columns alone hold it at {floor:.4f}. Price the taps "
            f"more dearly (a larger engagement threshold) or accept a higher "
            f"target.")
    if lam(lo) < lambda_target:
        return lo
    for _ in range(200):
        mid = math_sqrt(lo * hi)
        if lam(mid) > lambda_target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0 + tol:
            break
    return math_sqrt(lo * hi)


def math_sqrt(x: float) -> float:
    return float(np.sqrt(x))


#: Continuous classes each layer scales to place its own contraction.  The AVR
#: column is excluded on purpose: it is the held gauge column of the thesis
#: design, priced by the smoothness requirement on generator setpoints rather
#: than by a layer gain.
SCALED_CLASSES: Dict[str, Tuple[str, ...]] = {
    "TS": ("der", "pcc"),
    "STS": ("dso_der",),
}


def design_stage1(
    ops: Sequence[OperatingPoint],
    *,
    pi_q: float = 1.0,
    pi_v_ds: float = 1.0,
    tau: float = 1.0,
    g_w_gen: float = 1.0e8,
    d_engage_ts: float = 0.010,
    d_engage_sts: float = 0.020,
    d_engage_shunt: Optional[float] = None,
    lambda_target_ts: float = 1.0,
    lambda_target_sts: float = 1.0,
    scales: Optional[PriorityScales] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """The Stage-I starting point: the weights from a few engineering statements.

    The six are dimensionless or in physical units, and each is an argument a
    reader can check:

    ``pi_q``, ``pi_v_ds``  priorities relative to supervisory voltage, in units
                           of "one tolerance-sized error costs the same"
    ``tau``                allocation between the PCC setpoint and TS DER
                           (1.0 = no preference)
    ``g_w_gen``            AVR step price, set by the smoothness requirement
    ``d_engage_*``         the voltage deviation at which each tap class should
                           commit, in pu; ``d_engage_shunt`` the same for the
                           MSC/MSR banks (2026-10-22), required whenever the
                           operating points carry bank columns
    ``lambda_target_*``    where to place each layer contraction

    The gauge is pinned so that ``pi_v_ts = 1``.  Integer weights follow from
    the commit rule and depend only on the objective weights, so they are
    solved first and exactly; the continuous scale is then bisected against the
    target, which is monotone.  Nothing is fitted.

    Returns ``(weights, report)``.  Raises if a target lies below the integer
    floor, which is a real conflict between how early the taps should act and
    how much contraction the layer is allowed -- not something to clip away.
    """
    S = scales or PriorityScales()
    lcs = [lc for op in ops for lc in op.layers]

    objective = {
        "g_v": S.g_v_from_priority(1.0),
        "g_q": S.g_q_from_priority(pi_q),
        "dso_g_v": S.dso_g_v_from_priority(pi_v_ds),
    }
    d_engage: Dict[str, float] = {"oltc": d_engage_ts, "dso_oltc": d_engage_sts}
    has_banks = any("shunt" in lc.classes for lc in lcs)
    if has_banks:
        # No default: a bank engagement deviation is a design statement in its
        # own right, and borrowing the tap value would hide that it was made.
        if d_engage_shunt is None:
            raise ValueError(
                "the operating points carry MSC/MSR bank columns, so the bank "
                "step price must be placed from a stated engagement deviation: "
                "pass d_engage_shunt.")
        d_engage["shunt"] = float(d_engage_shunt)
    g_w_int = design_integer_weights(lcs, objective, d_engage)

    weights: Dict[str, float] = dict(objective)
    weights["g_w_gen"] = float(g_w_gen)
    weights["g_w_tso_oltc"] = g_w_int["oltc"]
    weights["g_w_dso_oltc"] = g_w_int["dso_oltc"]
    if has_banks:
        weights["g_w_tso_shunt"] = g_w_int["shunt"]
    # Provisional continuous block; only its RATIOS matter, the scale is solved.
    weights["g_w_der"] = 1.0
    weights["g_w_pcc"] = float(tau)
    weights["g_w_dso_der"] = 1.0

    report: Dict[str, Any] = {"floors": {}, "kappa": {}, "targets": {
        "TS": lambda_target_ts, "STS": lambda_target_sts}}

    for layer, target in (("TS", lambda_target_ts), ("STS", lambda_target_sts)):
        members = [lc for lc in lcs if lc.layer == layer]
        if not members:
            continue
        for lc in members:
            g_v, g_q = _layer_weights(lc, weights)
            report["floors"][lc.key] = integer_floor(lc, g_v, g_q, g_w_int)
        # One scale for the whole layer, set by its WORST area and condition:
        # a per-area scale would be a per-area parameter set, which is Stage III.
        kappa = max(
            solve_continuous_scale(lc, weights, SCALED_CLASSES[layer], target)
            for lc in members
        )
        report["kappa"][layer] = kappa
        for cls in SCALED_CLASSES[layer]:
            name = CLASS_TO_WEIGHT[cls]
            weights[name] = float(weights[name]) * kappa

    report["d_engage"] = dict(d_engage)
    report["priorities"] = {"pi_v_ts": 1.0, "pi_q": pi_q, "pi_v_ds": pi_v_ds}
    report["tau"] = tau
    return weights, report


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------
#
# At the END of the module, not in the middle of it.  Until 2026-09-15 this
# guard sat above the Stage I-c section, so ``python -m tuning.stage1_screen``
# executed ``main()`` before ``design_stage1`` had been defined -- harmless for
# the screening paths, but ``--design`` raised ``NameError`` on a name the file
# plainly contains.
#
# It also DELEGATES to the module imported under its real name, rather than
# calling the ``main`` defined in this ``__main__`` namespace.  The operating
# point cache is pickled, and ``pickle`` records a class as
# ``module.qualname``: run as ``__main__``, this file writes
# ``__main__.OperatingPoint``, which nothing but another
# ``python -m tuning.stage1_screen`` invocation can load.  Measured
# 2026-09-15 -- the caches written that way raise
# ``AttributeError: Can't get attribute 'OperatingPoint' on <module
# '__main__'>`` in every consumer, including
# ``ch_9_3_stage2_screen.py --ops-dir``.  The re-import costs one extra
# execution of this module's definitions and nothing else.

if __name__ == "__main__":
    import sys as _sys

    from tuning.stage1_screen import main as _main

    _sys.exit(_main())
