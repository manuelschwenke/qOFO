"""
sensitivity/network_reduction.py
================================
Build reduced pandapower networks for **per-controller local sensitivity**
computation.

The default operating mode of ``run_multi_tso_dso`` is to give every
TSO and DSO controller the *same* :class:`sensitivity.jacobian.JacobianSensitivities`
instance, built from the full IEEE 39-bus + HV sub-networks plant net.
Each controller's H matrix is then a sub-block of one global Jacobian.

The functions in this module produce an alternative: a Ward-style
*reduced* network per controller that contains only the buses/elements
the controller can actually see and act on, with the rest of the system
condensed into equivalent boundary representations from the cached
operating point.

Two boundary conventions are used (chosen to match the user prompt of
2026-05-27):

* **TSO zone (``build_tso_local_net``):** every boundary is a *PQ load* at
  the boundary bus; the slack lives on a synchronous generator inside the
  zone (the original IEEE 39 slack-gen if it is in the zone, otherwise
  the largest gen is promoted).  Boundaries are:

  - Tie-line far-end bus  → PQ load representing the rest-of-system draw.
  - 3W-trafo primary (HV/TS) bus of every DSO whose sub-network attaches
    in this zone → PQ load = cached ``(p_hv_mw, q_hv_mvar)`` of that
    coupling 3W trafo; the trafo itself and the HV sub-network are
    dropped.

  TSO-owned bipolar shunts originally sit on the LV (20 kV tertiary) side
  of the 3W coupler.  Under reduction the tertiary is gone, so the TSO
  controller's sensitivity Jacobian instead sees a *synthetic shunt at
  the 3W primary bus* with the same ``q_mvar`` per step and the same
  cached step value.  The mapping ``synthetic_shunt_map`` (returned
  alongside the net) tells the runner how to translate between the
  plant tertiary shunt bus and the local synthetic primary bus.

* **DSO sub-network (``build_dso_local_net``):** the boundary is a
  *virtual slack-gen* at the 3W primary bus pinned to ``V_cached``.  No
  explicit PQ load is added there — the slack auto-dispatches the cached
  HV flow at the cached operating point (a separate PQ load at the slack
  bus would only double-count the same injection).  Inside the kept
  region (HV sub-network + 3W trafos + tertiary buses + TSO-owned
  tertiary shunts) every element is preserved unchanged.

The returned reduced nets are converged by ``pp.runpp`` so the caller can
hand them straight to :class:`sensitivity.jacobian.JacobianSensitivities`.

Notes
-----
* All bus indices in the reduced net match those in the original plant
  net (pandapower preserves explicit row labels through deepcopy +
  selective drop), so the controllers' existing index-based lookups
  (``self.config.der_indices`` → ``self.sensitivities.net.sgen.at[i, ...]``)
  keep working without any controller-side change.
* The reduced nets do **not** keep ``distributed_slack=True`` — the
  reduced TSO zone has too few gens (3-4) for the dispatch to make sense
  numerically, and the reduced DSO has none.  The Jacobian we extract
  later runs ``run_control=False, distributed_slack=False`` internally
  inside :class:`JacobianSensitivities.__init__`.
* Synthetic shunts are placed at the 3W primary bus with the same
  ``q_mvar`` per step as the original tertiary shunt.  The 3W coupler's
  series impedance is low enough that the susceptance effect on TN
  voltages is approximately the same magnitude as if the shunt were
  placed at the tertiary, so this approximation preserves the *sign*
  and *order of magnitude* of the TSO MIQP's shunt actuator column.

Author: Manuel Schwenke / Claude Code
Date: 2026-05-27
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandapower as pp

from network.ieee39.meta import HVNetworkInfo, IEEE39NetworkMeta
from sensitivity.jacobian import runpp_with_stored_jacobian


# ---------------------------------------------------------------------------
#  Result containers
# ---------------------------------------------------------------------------

@dataclass
class TSOLocalNetResult:
    """Return container for :func:`build_tso_local_net`.

    Attributes
    ----------
    net : pp.pandapowerNet
        Reduced pandapower net, converged at the cached operating point.
    synthetic_shunt_map : Dict[int, int]
        Maps each TSO-owned tertiary shunt bus (plant index) to the
        synthetic shunt bus in the reduced net (always the matching 3W
        primary bus).  Empty when the zone has no TSO-owned shunts.
    slack_gen_idx : Optional[int]
        ``net.gen`` index of the slack-gen used in the reduced net.
        ``None`` if no gen was needed as slack (degenerate zone).
    promoted_slack_oltc_indices : Tuple[int, ...]
        ``net.trafo`` indices of machine OLTCs that the controller must
        mark out-of-service in this zone because one of their endpoints
        (typically the LV gen-terminal bus) became the slack-reference
        bus in the reduced net — :meth:`JacobianSensitivities.compute_dV_ds_2w`
        cannot produce a sensitivity column for a trafo touching the
        slack bus.  Empty when the original plant slack-gen is in the
        zone (no promotion needed).
    """
    net: pp.pandapowerNet
    synthetic_shunt_map: Dict[int, int] = field(default_factory=dict)
    slack_gen_idx: Optional[int] = None
    promoted_slack_oltc_indices: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class DSOLocalNetResult:
    """Return container for :func:`build_dso_local_net`."""
    net: pp.pandapowerNet
    virtual_slack_gen_indices: Tuple[int, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
#  Thevenin boundary helper (shared by the TSO and DSO reductions)
# ---------------------------------------------------------------------------

#: Per-unit base used when converting a boundary impedance to per-unit for
#: the EMF back-solve.  Any consistent value works -- it cancels -- but the
#: system base keeps the intermediate numbers interpretable.
_S_BASE_MVA: float = 100.0


def add_thevenin_boundary(
    net: pp.pandapowerNet,
    bus: int,
    p_inj_mw: float,
    q_inj_mvar: float,
    z_ohm: complex,
    *,
    vm_pu: Optional[float] = None,
    va_degree: Optional[float] = None,
    slack: bool = False,
    name: str = "WARD_THEVENIN",
) -> Tuple[int, int, int]:
    """Attach a Thevenin source behind *z_ohm* to *bus*.

    Creates an auxiliary bus, a series branch of impedance *z_ohm*, and a
    voltage source at the auxiliary bus.  The EMF and its active in-feed are
    back-solved so that the ORIGINAL operating point at *bus* -- its cached
    voltage phasor and the cached injection ``(p_inj_mw, q_inj_mvar)`` --
    is reproduced exactly, for any *z_ohm*.

    This makes the whole family a one-parameter sweep that always matches at
    the linearisation point and differs only in the derivative, which is what
    the sensitivity model is made of.  The two conventions already in use are
    its endpoints: ``z -> inf`` is the constant PQ equivalent, ``z -> 0`` the
    stiff voltage source.

    The decisive structural difference from ``z = 0``: the source sits on the
    auxiliary bus, so *bus* itself stays an ordinary PQ bus of the reduced
    network.  It keeps a row in the reduced Jacobian and therefore HAS a
    voltage sensitivity -- which a slack or PV bus structurally cannot.

    Parameters
    ----------
    net : pp.pandapowerNet
        Reduced network, already carrying converged ``res_bus`` values for
        *bus* (read before the surrounding tables are edited).
    bus : int
        Boundary bus the equivalent attaches to.
    p_inj_mw, q_inj_mvar : float
        Cached injection INTO *bus* from the system being condensed.
    z_ohm : complex
        Series impedance of the equivalent, in ohms at the nominal voltage
        of *bus*.  Must have non-zero magnitude; use the ``"pv"`` boundary
        for the exact zero-impedance case.
    vm_pu, va_degree : float, optional
        Cached voltage phasor at *bus*.  Pass these when the caller has
        already captured them: the reduction drops buses, and pandapower's
        ``drop_buses`` prunes the matching ``res_bus`` rows with them, so by
        the time the boundary is attached the result table may no longer
        hold the value.  Falls back to ``net.res_bus`` and then to a flat
        1.0 / 0 deg.
    slack : bool
        If True the source carries the angle reference.  Used on the DSO
        side, where the overlaying system supplies it.
    name : str
        Element name prefix, kept for later identification.

    Returns
    -------
    (aux_bus, branch_idx, gen_idx)
    """
    if abs(z_ohm) <= 0.0:
        raise ValueError("z_ohm must be non-zero; use tie_boundary='pv' for Z=0")

    vn_kv = float(net.bus.at[bus, "vn_kv"])
    if vm_pu is None:
        vm_pu = (float(net.res_bus.at[bus, "vm_pu"])
                 if bus in net.res_bus.index else 1.0)
    if va_degree is None:
        va_degree = (float(net.res_bus.at[bus, "va_degree"])
                     if bus in net.res_bus.index else 0.0)
    vm, va = float(vm_pu), float(va_degree)
    if not np.isfinite(vm) or vm <= 0.0 or not np.isfinite(va):
        vm, va = 1.0, 0.0

    # Back-solve the EMF in per unit on (_S_BASE_MVA, vn_kv).
    z_base = (vn_kv ** 2) / _S_BASE_MVA
    z_pu = complex(z_ohm) / z_base
    v_pu = vm * np.exp(1j * np.deg2rad(va))
    s_pu = complex(p_inj_mw, q_inj_mvar) / _S_BASE_MVA
    i_pu = np.conj(s_pu / v_pu)
    e_pu = v_pu + z_pu * i_pu
    p_src_mw = float(np.real(e_pu * np.conj(i_pu)) * _S_BASE_MVA)

    aux = int(pp.create_bus(
        net, vn_kv=vn_kv, name=f"{name}_AUX_{int(bus)}",
        type="b", in_service=True,
    ))
    # A line rather than an impedance element: r/x are then plain ohms at the
    # bus nominal voltage, with no per-unit base to get wrong.  Zero shunt
    # capacitance keeps it a pure series Thevenin.
    br = int(pp.create_line_from_parameters(
        net, from_bus=int(bus), to_bus=aux, length_km=1.0,
        r_ohm_per_km=float(np.real(z_ohm)),
        x_ohm_per_km=float(np.imag(z_ohm)),
        c_nf_per_km=0.0, max_i_ka=1.0e3,
        name=f"{name}_Z_{int(bus)}",
    ))
    gi = int(pp.create_gen(
        net, bus=aux, p_mw=p_src_mw, vm_pu=float(np.abs(e_pu)),
        slack=bool(slack),
        min_p_mw=-1e6, max_p_mw=1e6,
        min_q_mvar=-1e6, max_q_mvar=1e6,
        name=f"{name}_SRC_{int(bus)}",
    ))

    # Seed the auxiliary bus in ``res_bus`` with its EXACT solution -- ``e_pu``
    # IS the aux-bus voltage phasor by construction.  Mandatory, not cosmetic:
    # the reduced net inherits the plant's cached ``res_bus``, and the caller
    # converges it with an ``init="results"`` warm start.  A bus with no result
    # row makes that start incomplete; pandapower neither raises nor warns, and
    # Newton can then land on the SPURIOUS LOW-VOLTAGE ROOT -- V=0 satisfies the
    # mismatch equation identically at a zero-injection bus, and the reduced net
    # is full of those (tertiary stubs stripped of their load in step 6).
    #
    # Measured 2026-08-13 before this seed, zone 2 at 2016-01-05 08:00: the net
    # reported ``converged=True`` with TN buses at 0.44-0.66 pu and the tertiary
    # stubs at ~1e-62 pu, so every H entry was linearised at a fictitious
    # operating point and ``compute_dV_dQ_shunt`` (which scales by V_pu^2)
    # returned ~1e-127 -- a structurally dead shunt column.  See
    # 00_daily_log/2026-08-13_thevenin_spurious_root.md.
    if net.res_bus is not None and not net.res_bus.empty:
        seed = {c: 0.0 for c in net.res_bus.columns}
        seed["vm_pu"] = float(np.abs(e_pu))
        seed["va_degree"] = float(np.rad2deg(np.angle(e_pu)))
        net.res_bus.loc[aux] = [seed[c] for c in net.res_bus.columns]
        net.res_bus.sort_index(inplace=True)

    return aux, br, gi


#: Default tolerance of the reduced-net operating-point guard [pu].  A
#: gross-failure detector, NOT an accuracy check: the reduction reproduces the
#: cached state to ~1e-10 pu where the zone holds the system slack, and to
#: ~1.7e-2 pu where the slack had to be promoted (the reduced net solves with
#: distributed_slack=False, so the promoted machine absorbs a mismatch the
#: plant spread over all machines).  0.1 pu sits ~6x above the worst healthy
#: case and ~4x below the spurious low-voltage root it exists to catch.
_OP_POINT_TOL_PU: float = 0.1


def _assert_reproduces_cached_state(
    sub: pp.pandapowerNet,
    net: pp.pandapowerNet,
    *,
    exclude_buses: Iterable[int] = (),
    tol_pu: float = _OP_POINT_TOL_PU,
    label: str = "",
) -> None:
    """Fail fast if the reduced net did not converge to the cached state.

    The entire premise of the reduction is that every boundary equivalent
    reproduces the cached operating point by construction and differs only in
    the *derivative*.  If the solved voltage profile does not match the plant's
    cached one, the extracted H is a linearisation about a fictitious state and
    every downstream number -- sensitivities, MIQP steps, the shunt
    integrator's projected gradient -- is silently wrong.

    This is not hypothetical.  ``pp.runpp`` reports ``converged=True`` on the
    spurious low-voltage root, which a reduced net is unusually prone to: the
    stripped tertiary and far-end stubs carry zero injection, and V=0 satisfies
    their mismatch equation identically.  Measured 2026-08-13 (see
    ``00_daily_log/2026-08-13_thevenin_spurious_root.md``): TSO zones 2 and 3
    under ``tie_boundary="thevenin"`` solved at 0.44-0.66 pu on the TN with
    tertiaries at ~1e-62 pu, and nothing anywhere raised.

    Parameters
    ----------
    sub, net
        Reduced net (already converged) and the plant net holding the cached
        ``res_bus``.
    exclude_buses
        Buses to skip -- the auxiliary buses created by boundary equivalents,
        which have no counterpart in the plant.  NOTE: ``pp.create_bus``
        assigns them ``max(index) + 1`` of the *reduced* net, so their indices
        can coincide with plant buses that this zone dropped; comparing them
        against ``net.res_bus`` would be meaningless.
    tol_pu
        Maximum tolerated ``|vm_reduced - vm_cached|`` [pu].
    label
        Prefix for the error message (e.g. the zone).
    """
    skip = {int(b) for b in exclude_buses}
    bad: List[Tuple[int, float, float]] = []
    for b in sub.bus.index:
        b = int(b)
        if b in skip or b not in net.res_bus.index:
            continue
        v_cached = float(net.res_bus.at[b, "vm_pu"])
        if not np.isfinite(v_cached):
            continue
        v_red = (
            float(sub.res_bus.at[b, "vm_pu"])
            if b in sub.res_bus.index else float("nan")
        )
        if not np.isfinite(v_red) or abs(v_red - v_cached) > tol_pu:
            bad.append((b, v_red, v_cached))
    if bad:
        shown = ", ".join(
            f"bus {b}: {v:.4g} vs cached {vc:.4f}" for b, v, vc in bad[:8]
        )
        raise ValueError(
            f"{label}reduced net did not reproduce the cached operating point: "
            f"{len(bad)}/{len(sub.bus)} bus(es) deviate by more than "
            f"{tol_pu} pu ({shown}"
            f"{', ...' if len(bad) > 8 else ''}).  The power flow reports "
            f"convergence but has landed on a different root, so any H "
            f"extracted from this net is linearised about a fictitious state."
        )


#: Measured physical Thevenin stiffness per corridor, k = |Z_th| / |Z_line|,
#: from ``experiments/CIGRE_2026/007f_ZTH_PER_CORRIDOR.py`` on the IEEE 39-bus
#: ``base_410`` case at 2016-01-05 08:00.
#:
#: Keyed by ``(line_idx, far_end_bus)`` and NOT by line alone: a tie line is
#: shared by two zones, and each zone looks into a different external system
#: through a different far-end terminal, so the same line carries two different
#: values.  Line 2 is 1.82 seen from zone 1 (far bus 2) but 0.83 seen from zone 2
#: (far bus 1); line 5 is 2.14 from zone 2 and 1.40 from zone 3.
#:
#: The spread is physical: corridors terminating at an AVR-regulated machine are
#: stiff (line 14 / bus 38 at 0.05 ends essentially ON a machine), corridors
#: ending deep in the neighbour are soft (line 14 / bus 8 at 2.73).
#:
#: Operating-point specific.  Re-measure with 007f for a different scenario.
THEVENIN_K_PER_CORRIDOR: Dict[Tuple[int, int], float] = {
    # zone 1
    (2, 2): 1.82, (14, 8): 2.73, (25, 16): 1.24,
    # zone 2
    (2, 1): 0.83, (5, 17): 2.14, (14, 38): 0.05, (18, 14): 1.19,
    # zone 3
    (5, 2): 1.40, (18, 13): 0.89, (25, 26): 2.52,
}

#: Population mean of the above; the fallback for corridors not listed, and the
#: best single default from the 007d H-error sweep (the two agree).
THEVENIN_K_DEFAULT: float = 1.5


def line_series_impedance_ohm(net: pp.pandapowerNet, line_idx: int) -> complex:
    """Series impedance of one line in ohms (parallel circuits accounted for)."""
    r = float(net.line.at[line_idx, "r_ohm_per_km"])
    x = float(net.line.at[line_idx, "x_ohm_per_km"])
    ln = float(net.line.at[line_idx, "length_km"])
    par = float(net.line.at[line_idx, "parallel"]) if "parallel" in net.line.columns else 1.0
    par = max(par, 1.0)
    return complex(r * ln / par, x * ln / par)


# ---------------------------------------------------------------------------
#  TSO reduction
# ---------------------------------------------------------------------------

def build_tso_local_net(
    net: pp.pandapowerNet,
    zone_bus_indices: Iterable[int],
    gen_indices_in_zone: Iterable[int],
    machine_trafo_indices_in_zone: Iterable[int],
    tie_line_indices: Iterable[int],
    tie_line_endpoint_buses: Iterable[int],
    hv_networks_in_zone: Iterable[HVNetworkInfo],
    tso_shunt_buses_in_zone: Iterable[int],
    tso_shunt_q_steps_mvar_in_zone: Iterable[float],
    *,
    tie_boundary: str = "pq",
    tie_thevenin_k: float = 1.0,
    op_point_tol_pu: float = _OP_POINT_TOL_PU,
    verbose: int = 0,
) -> TSOLocalNetResult:
    """Build the reduced TSO network for one zone.

    Parameters mirror the index sets already gathered by the runner.  The
    returned net contains:

    * every TN bus in the zone (``zone_bus_indices``),
    * every generator + its machine 2W trafo + LV terminal bus,
    * every tie line (the in-zone endpoint stays, the far-end bus is
      kept as a "stub" PQ-load bus),
    * every 3W primary bus (HV/TS side) of every DSO in the zone, as a
      PQ-load stub (the 3W trafo and HV sub-network are dropped),
    * one synthetic shunt per TSO-owned tertiary shunt, placed on the
      corresponding 3W primary bus.

    Everything else is deleted.  The slack is the existing IEEE 39 slack-
    gen if it lives in the zone, otherwise the largest gen in the zone is
    promoted to slack.

    Parameters
    ----------
    net : pp.pandapowerNet
        Plant network at the cached operating point.  Must be converged
        (``net.res_*`` tables populated).
    zone_bus_indices : Iterable[int]
        TN bus indices that belong to this zone (TN-only — gen terminal
        buses are added below from ``machine_trafo_indices_in_zone``).
    gen_indices_in_zone : Iterable[int]
        ``net.gen`` indices in the zone.
    machine_trafo_indices_in_zone : Iterable[int]
        ``net.trafo`` indices for the machine 2W trafos of the zone's
        gens.  Their LV (terminal) buses are added to the keep set.
    tie_line_indices : Iterable[int]
        ``net.line`` indices of tie lines monitored by this zone.
    tie_line_endpoint_buses : Iterable[int]
        IN-ZONE endpoint of each tie line (parallel to
        ``tie_line_indices``).
    hv_networks_in_zone : Iterable[HVNetworkInfo]
        HV sub-network metadata objects whose ``zone`` matches this zone.
        Their coupling 3W trafos are *dropped*; their primary buses are
        kept as PQ-load stubs.
    tso_shunt_buses_in_zone : Iterable[int]
        Plant tertiary bus index of each TSO-owned shunt in this zone.
    tso_shunt_q_steps_mvar_in_zone : Iterable[float]
        Per-shunt rated Mvar per step (same order).
    tie_boundary : {"pq", "pv", "z"}
        Boundary condition placed at each tie-line far-end stub bus,
        i.e. how the neighbouring TSO area is condensed.  All three
        reproduce the cached operating point at the far-end bus by
        construction and differ only in the *derivative* -- which is
        the whole content of the H matrix extracted afterwards:

        * ``"pq"`` (default, historical behaviour): constant PQ load
          at the cached corridor flow.  Infinite Thevenin impedance
          behind the boundary: the far-end voltage moves freely, so
          the neighbour offers no voltage support at all.
        * ``"pv"``: a PV generator holding the cached far-end voltage
          magnitude and the cached active in-flow, reactive power
          free.  Zero Thevenin impedance: the neighbour holds the
          boundary voltage perfectly.
        * ``"z"``: a constant shunt admittance matched to the cached
          flow, ``y = S*/|V|^2``.  Finite stiffness between the two
          extremes, at no extra information cost.  Only well posed
          when the equivalent *absorbs* -- a constant-Z negative load
          injects more as the voltage rises, so an importing corridor
          falls back to ``"pq"`` for that stub (a warning is printed
          at ``verbose >= 1``).

        * ``"thevenin"``: voltage source behind a series impedance of
          ``tie_thevenin_k`` times the tie line's own series impedance,
          on an auxiliary bus.  The finite-impedance case the other
          three approximate, and the only one that leaves the far-end
          bus an ordinary PQ bus with a voltage sensitivity of its own.

        ``"pq"`` and ``"pv"`` bracket the true finite-impedance
        equivalent; see the horizontal-interface discussion in the
        thesis (Ch 6, multi-area local model).
    tie_thevenin_k : float or dict
        Only read when ``tie_boundary="thevenin"``.  Boundary impedance
        as a multiple of the tie line's own series impedance -- a
        dimensionless, sweepable stiffness knob.  ``k -> 0`` approaches
        the ``"pv"`` limit, ``k -> inf`` the ``"pq"`` limit.

        A float applies one value to every corridor.  A dict keyed by
        ``(line_idx, far_end_bus)`` sets it per corridor; corridors not
        listed fall back to :data:`THEVENIN_K_DEFAULT`.  The key is the
        pair and not the line alone because a tie line is shared by two
        zones which look into different external systems through
        different terminals, so one line carries two values.
        :data:`THEVENIN_K_PER_CORRIDOR` holds the measured set.
    op_point_tol_pu : float
        Tolerance [pu] of the post-solve guard that checks the reduced net
        actually landed on the cached operating point rather than on another
        root of the power flow equations.  See
        :func:`_assert_reproduces_cached_state`.

    Returns
    -------
    TSOLocalNetResult
    """
    if tie_boundary not in ("pq", "pv", "z", "thevenin"):
        raise ValueError(
            f"tie_boundary must be one of 'pq', 'pv', 'z', 'thevenin'; "
            f"got {tie_boundary!r}"
        )
    if tie_boundary == "thevenin":
        if isinstance(tie_thevenin_k, dict):
            bad = {k: v for k, v in tie_thevenin_k.items() if not float(v) > 0.0}
            if bad:
                raise ValueError(
                    f"tie_thevenin_k entries must be > 0; offending: {bad}"
                )
        elif not (float(tie_thevenin_k) > 0.0):
            raise ValueError(
                f"tie_thevenin_k must be > 0 for tie_boundary='thevenin'; "
                f"got {tie_thevenin_k}"
            )

    def _k_for(line_idx: int, far_bus: int) -> float:
        """Boundary stiffness for one corridor stub.

        A dict is keyed by ``(line_idx, far_bus)`` rather than by line alone
        because a tie line is shared between two zones, and each looks into a
        different external system through a different terminal -- so the same
        line legitimately carries two different values.  Corridors absent from
        the mapping fall back to the population mean.
        """
        if not isinstance(tie_thevenin_k, dict):
            return float(tie_thevenin_k)
        key = (int(line_idx), int(far_bus))
        if key in tie_thevenin_k:
            return float(tie_thevenin_k[key])
        if verbose >= 1:
            print(f"  [build_tso_local_net] no k for corridor {key}; "
                  f"using default {THEVENIN_K_DEFAULT}")
        return float(THEVENIN_K_DEFAULT)
    sub = copy.deepcopy(net)

    zone_bus_set: set = set(int(b) for b in zone_bus_indices)
    gen_set: set = set(int(g) for g in gen_indices_in_zone)
    machine_trafos_in_zone: List[int] = [int(t) for t in machine_trafo_indices_in_zone]
    tie_lines: List[int] = [int(li) for li in tie_line_indices]
    tie_in_endpoints: List[int] = [int(b) for b in tie_line_endpoint_buses]
    hv_list: List[HVNetworkInfo] = list(hv_networks_in_zone)
    shunt_buses: List[int] = [int(b) for b in tso_shunt_buses_in_zone]
    shunt_q_steps: List[float] = [float(q) for q in tso_shunt_q_steps_mvar_in_zone]

    # ── 1. Compute keep-bus set ───────────────────────────────────────────
    keep_buses: set = set(zone_bus_set)

    # Add LV terminal buses of in-zone machine trafos (gen terminals)
    for t in machine_trafos_in_zone:
        keep_buses.add(int(sub.trafo.at[t, "lv_bus"]))
        keep_buses.add(int(sub.trafo.at[t, "hv_bus"]))

    # Add tie-line far-end buses
    far_end_buses: List[Tuple[int, int]] = []  # (line_idx, far_bus)
    for li, in_bus in zip(tie_lines, tie_in_endpoints):
        if li not in sub.line.index:
            continue
        from_bus = int(sub.line.at[li, "from_bus"])
        to_bus = int(sub.line.at[li, "to_bus"])
        if in_bus == from_bus:
            far = to_bus
        elif in_bus == to_bus:
            far = from_bus
        else:
            # in_bus doesn't match either endpoint — skip
            continue
        keep_buses.add(far)
        far_end_buses.append((li, far))

    # Add 3W primary buses for DSOs in this zone
    primary_bus_for_3w: Dict[int, int] = {}   # 3w_idx → primary bus
    for hv in hv_list:
        for t3w in hv.coupling_trafo_indices:
            if t3w not in sub.trafo3w.index:
                continue
            primary = int(sub.trafo3w.at[t3w, "hv_bus"])
            keep_buses.add(primary)
            primary_bus_for_3w[int(t3w)] = primary

    # ── 2. Capture cached boundary flows BEFORE editing tables ────────────
    # Tie-line far-end PQ-load values: net injection from "rest of system"
    # into b_far at cached state = +p_xxx_mw_at_far (pandapower's
    # res_line.p_xxx is power INTO the line at side xxx, so power into
    # bus from rest-of-system = +p_xxx_mw_at_far).  Load draws this much
    # ⇒ load.p_mw = -p_xxx_mw_at_far.
    # Each entry is (bus, p_inj_mw, q_inj_mvar, v_pu, va_deg, line_idx),
    # where the injection is what the rest-of-system pushes INTO the far-end
    # bus at the cached state.  The voltage phasor is captured HERE, before
    # ``pp.drop_buses`` prunes the matching ``res_bus`` rows, and the line
    # index lets the "thevenin" variant scale its impedance to the tie
    # line's own.  The "pq" variant uses only the injection.
    tie_load_specs: List[Tuple[int, float, float, float, float, int]] = []
    for li, far in far_end_buses:
        if li not in sub.res_line.index:
            continue
        from_bus = int(sub.line.at[li, "from_bus"])
        if far == from_bus:
            p_far = float(sub.res_line.at[li, "p_from_mw"])
            q_far = float(sub.res_line.at[li, "q_from_mvar"])
        else:
            p_far = float(sub.res_line.at[li, "p_to_mw"])
            q_far = float(sub.res_line.at[li, "q_to_mvar"])
        if far in sub.res_bus.index:
            v_far = float(sub.res_bus.at[far, "vm_pu"])
            va_far = float(sub.res_bus.at[far, "va_degree"])
        else:
            v_far, va_far = 1.0, 0.0
        if not np.isfinite(v_far) or v_far <= 0.0 or not np.isfinite(va_far):
            v_far, va_far = 1.0, 0.0
        tie_load_specs.append((far, p_far, q_far, v_far, va_far, int(li)))

    # 3W primary PQ-load values: trafo was drawing (p_hv_mw, q_hv_mvar)
    # from the TN at cached state; after we delete the trafo, replace by a
    # load that draws the same.
    primary_load_specs: List[Tuple[int, float, float, int]] = []
    # (primary_bus, p_mw, q_mvar, trafo3w_idx)
    for t3w, primary in primary_bus_for_3w.items():
        if t3w not in sub.res_trafo3w.index:
            continue
        p_hv = float(sub.res_trafo3w.at[t3w, "p_hv_mw"])
        q_hv = float(sub.res_trafo3w.at[t3w, "q_hv_mvar"])
        primary_load_specs.append((primary, p_hv, q_hv, t3w))

    # Cached voltage at every primary bus (for synthetic-shunt q_mvar
    # scaling, if we choose to scale; currently we use a 1:1 mapping).
    primary_v_cached: Dict[int, float] = {}
    for _, primary in primary_bus_for_3w.items():
        primary_v_cached[primary] = float(sub.res_bus.at[primary, "vm_pu"])

    # ── 3. Keep the in-zone PCC 3W trafos (with their primary/MV/LV) ──
    # The user prompt asked for "primary-side PQ injection" but the TSO
    # controller's :meth:`_build_sensitivity_matrix` requires a *live*
    # 3W coupler row in ``net.trafo3w`` so that
    # ``compute_dQtrafo3w_hv_*`` can populate the Q_PCC output rows AND
    # ``compute_dV_dQ_der`` at the primary bus can populate the
    # Q_PCC,set actuator columns.  Without a live trafo, those blocks
    # come out as zeros or NaN and the OFO sees no V-tracking leverage
    # from PCC dispatch (observed symptom: TSO commands a constant
    # Q_PCC,set forever).
    #
    # We therefore keep every PCC 3W trafo + its MV bus + LV (tertiary)
    # bus alive, plus the primary bus.  The Ward equivalent moves *one
    # bus deeper* than the user's literal spec: the PQ load lands on
    # the MV-side bus (the boundary between the trafo and the dropped
    # HV sub-network).  Semantically the user's intent is preserved —
    # the entire HV sub-network behind the trafo is replaced by a
    # constant PQ injection — only the injection bus moves from the
    # primary (HV side, TN) to the MV side.  TSO-owned tertiary shunts
    # (on the LV bus) stay where they are.
    pcc_trafo3w_in_zone = set(int(t) for t in primary_bus_for_3w.keys())
    pcc_t3w_mv_buses: List[Tuple[int, int]] = []   # (mv_bus, trafo_idx)
    pcc_t3w_lv_buses: List[int] = []
    for hv in hv_list:
        for t, mv_bus, lv_bus in zip(
            hv.coupling_trafo_indices,
            hv.coupling_hv_bus_indices,
            hv.coupling_lv_bus_indices,
        ):
            if int(t) in pcc_trafo3w_in_zone:
                pcc_t3w_mv_buses.append((int(mv_bus), int(t)))
                pcc_t3w_lv_buses.append(int(lv_bus))
                keep_buses.add(int(mv_bus))
                keep_buses.add(int(lv_bus))

    # Cached MV-side flow (Ward injection value) — read BEFORE we touch
    # the trafo3w table or the surrounding net.  Sign: pandapower's
    # ``q_mv_mvar`` is the Q flowing INTO the trafo at the MV bus
    # (load convention from the bus's perspective).  After we drop the
    # HV sub-network, the MV bus loses its downstream load that used
    # to draw this Q; the new ``pp.create_load`` substitutes for it.
    mv_load_specs: List[Tuple[int, float, float]] = []  # (mv_bus, p_mw, q_mvar)
    for mv_bus, t in pcc_t3w_mv_buses:
        if t not in sub.res_trafo3w.index:
            continue
        # Power flowing INTO the trafo at the MV bus = power leaving the
        # bus through the trafo. Power that the bus consumed from the
        # rest of the HV sub-network = -q_mv_mvar (= what flows into the
        # bus from the rest of the sub-net to be sent through the trafo).
        # When we delete the HV sub-network, we must replace that draw
        # by a new load at the MV bus with the SAME consumed power.
        # Cached ``q_mv_mvar`` IS positive when the bus loses Q to the
        # trafo, so the load that USED to supply it is +q_mv_mvar in
        # magnitude, modelled here as a (negative-q) generator or a
        # load with q_mvar = -q_mv_mvar.  Equivalently we add a load
        # equal to (-p_mv_mw, -q_mv_mvar) so the bus net injection
        # stays at the cached operating point.
        p_mv = float(sub.res_trafo3w.at[t, "p_mv_mw"])
        q_mv = float(sub.res_trafo3w.at[t, "q_mv_mvar"])
        mv_load_specs.append((mv_bus, -p_mv, -q_mv))

    # ── 4. Drop HV sub-network elements (buses, lines, sgens, loads, …) ─
    hv_buses_to_drop: set = set()
    for hv in hv_list:
        for b in hv.bus_indices:
            hv_buses_to_drop.add(int(b))
        # Tertiary buses are explicit in coupling_lv_bus_indices
        for b in hv.coupling_lv_bus_indices:
            hv_buses_to_drop.add(int(b))
        # MV-side (110 kV) coupling bus indices
        for b in hv.coupling_hv_bus_indices:
            hv_buses_to_drop.add(int(b))
    if hv_buses_to_drop:
        # Use pp.drop_buses to cascade-delete attached elements cleanly.
        # Exclude any bus we still want to keep (primary + MV + LV of
        # the PCC trafos in this zone, plus the original TN buses).
        hv_buses_to_drop -= keep_buses
        if hv_buses_to_drop:
            pp.drop_buses(sub, list(hv_buses_to_drop))

    # Strip every element attached to the surviving MV/LV stubs except
    # the trafo itself (and any TSO-owned tertiary shunt on the LV bus).
    # Specifically: drop the original load that used to absorb the MV
    # flow downstream — the new ``mv_load_specs`` adds a fresh load
    # with the cached Ward injection.
    for b in {bus for (bus, _t) in pcc_t3w_mv_buses}:
        # Drop loads/sgens/gens at b (downstream HV sub-net loads).
        for tbl in ("load", "sgen", "gen"):
            df = getattr(sub, tbl)
            if not df.empty:
                mask = df["bus"] == b
                if mask.any():
                    df.drop(index=df.index[mask], inplace=True)
        # Drop lines attached to b (HV sub-net lines going downstream).
        mask_line = (sub.line["from_bus"] == b) | (sub.line["to_bus"] == b)
        for li in sub.line.index[mask_line]:
            sub.line.drop(index=li, inplace=True)
    for b in pcc_t3w_lv_buses:
        # Drop loads/sgens/gens at the tertiary (but keep TSO shunts).
        for tbl in ("load", "sgen", "gen"):
            df = getattr(sub, tbl)
            if not df.empty:
                mask = df["bus"] == b
                if mask.any():
                    df.drop(index=df.index[mask], inplace=True)
        mask_line = (sub.line["from_bus"] == b) | (sub.line["to_bus"] == b)
        for li in sub.line.index[mask_line]:
            sub.line.drop(index=li, inplace=True)

    # ── 5. Drop every bus not in keep_buses ───────────────────────────────
    remaining_buses = set(int(b) for b in sub.bus.index)
    extra_drop = remaining_buses - keep_buses
    if extra_drop:
        pp.drop_buses(sub, list(extra_drop))

    # ── 6. Strip every element attached to far-end "stub" buses ──────────
    # After pp.drop_buses, the keep-buses are still alive but the tie-line
    # far-end buses sit outside the zone — they exist in the reduced net
    # only as anchors for the tie line.  We strip every attached element
    # (loads, sgens, gens, shunts, other lines) so the far-end becomes a
    # pure PQ stub, then add a fresh Ward-equivalent load in step 7.
    #
    # 3W-primary buses are *also* regular zone-TN buses with legitimate
    # in-zone line / load / gen attachments; we already removed the
    # offending 3W trafo (and its HV sub-network) in step 3-4.  Their
    # remaining TN attachments must stay — otherwise the zone's TN
    # backbone is severed.  We only add the Ward-equivalent load (step 7)
    # to substitute for the dropped 3W's HV-side draw.
    far_end_set = set(b for _, b in far_end_buses)

    for b in far_end_set:
        # Drop loads at b
        mask = sub.load["bus"] == b
        if mask.any():
            sub.load.drop(index=sub.load.index[mask], inplace=True)
        # Drop sgens at b
        mask = sub.sgen["bus"] == b
        if mask.any():
            sub.sgen.drop(index=sub.sgen.index[mask], inplace=True)
        # Drop gens at b
        mask = sub.gen["bus"] == b
        if mask.any():
            sub.gen.drop(index=sub.gen.index[mask], inplace=True)
        # Drop shunts at b
        if not sub.shunt.empty:
            mask = sub.shunt["bus"] == b
            if mask.any():
                sub.shunt.drop(index=sub.shunt.index[mask], inplace=True)
        # Drop every line attached EXCEPT the tie line itself
        keep_lines = {li for li, fb in far_end_buses if fb == b}
        mask_line = (sub.line["from_bus"] == b) | (sub.line["to_bus"] == b)
        for li in sub.line.index[mask_line]:
            if int(li) not in keep_lines:
                sub.line.drop(index=li, inplace=True)
        # Drop every 2W trafo with a leg at b (the far-end has no machine
        # gen by construction, but defensive).
        mask_tr = (sub.trafo["hv_bus"] == b) | (sub.trafo["lv_bus"] == b)
        for t in sub.trafo.index[mask_tr]:
            sub.trafo.drop(index=t, inplace=True)

    # ── 7. Add equivalent PQ loads at boundary stubs ──────────────────────
    # Tie-line far-end stubs get a Ward load representing the rest-of-
    # system net injection.  The 3W coupler's HV sub-network (downstream
    # of the MV bus) is represented by a load at the MV bus — the
    # trafo itself stays alive and the primary bus remains a normal
    # zone-TN bus with its existing TN connectivity intact.
    #
    # ``tie_boundary`` selects WHICH boundary condition the neighbouring
    # area is condensed to.  All three match the cached injection at the
    # cached voltage, so the reduced power flow below reproduces the same
    # far-end operating point in every variant; they differ only in how
    # that bus responds when the zone's own actuators move, which is what
    # the extracted H matrix is made of.
    n_tie_z_fallback = 0
    boundary_aux_buses: List[int] = []
    for far, p_inj, q_inj, v_far, va_far, li_tie in tie_load_specs:
        if far not in sub.bus.index:
            continue

        if tie_boundary == "thevenin":
            # Voltage source behind a finite impedance on an auxiliary bus.
            # The far-end bus stays an ordinary PQ bus and therefore keeps a
            # row in the reduced Jacobian -- unlike the "pv" case, where the
            # boundary voltage is structurally fixed.
            z_line = (
                line_series_impedance_ohm(sub, li_tie)
                if li_tie in sub.line.index else complex(0.0, 0.0)
            )
            if abs(z_line) <= 0.0:
                # Degenerate line data: fall back to a 1 % reactance on the
                # bus base rather than divide by zero.
                vn = float(sub.bus.at[far, "vn_kv"])
                z_line = complex(0.0, 0.01 * vn ** 2 / _S_BASE_MVA)
            aux_b, _, _ = add_thevenin_boundary(
                sub, int(far), p_inj, q_inj,
                complex(_k_for(li_tie, far)) * z_line,
                vm_pu=v_far, va_degree=va_far,
                slack=False, name="WARD_TIE_TH",
            )
            boundary_aux_buses.append(int(aux_b))
            continue

        if tie_boundary == "pv":
            # Neighbour holds the boundary voltage: PV bus at the cached
            # magnitude and cached active in-feed, reactive power free.
            # No angle reference here -- the in-zone slack of step 9 still
            # supplies it, so this stays a PV and not a second slack.
            pp.create_gen(
                sub, bus=int(far), p_mw=p_inj, vm_pu=float(v_far),
                slack=False,
                min_p_mw=-1e6, max_p_mw=1e6,
                min_q_mvar=-1e6, max_q_mvar=1e6,
                name="WARD_TIE_PV",
            )
            continue

        if tie_boundary == "z":
            # Constant admittance matched at the cached voltage.  A
            # pandapower shunt CONSUMES (p_mw, q_mvar) at 1 pu and scales
            # with V^2, so match consumption -(p_inj, q_inj) at v_far.
            p_sh = -p_inj / (v_far ** 2)
            q_sh = -q_inj / (v_far ** 2)
            if p_sh >= 0.0 and q_sh >= 0.0:
                # Genuinely passive absorber: stiffening with rising V,
                # which is the direction the true equivalent moves in.
                pp.create_shunt(
                    sub, bus=int(far), p_mw=p_sh, q_mvar=q_sh, step=1,
                    name="WARD_TIE_Z",
                )
                continue
            # Otherwise the constant-Z form would be an active source whose
            # output GROWS with voltage -- softer than constant power, i.e.
            # further from the truth, not closer.  Fall back to PQ.
            n_tie_z_fallback += 1

        pp.create_load(sub, bus=int(far), p_mw=-p_inj, q_mvar=-q_inj,
                       name="WARD_TIE")
    if verbose >= 1 and n_tie_z_fallback:
        print(f"  [build_tso_local_net] tie_boundary='z': "
              f"{n_tie_z_fallback}/{len(tie_load_specs)} stub(s) are net "
              f"sources -> constant-Z ill-posed, fell back to PQ there")
    for mv_bus, p_load, q_load in mv_load_specs:
        if mv_bus in sub.bus.index:
            pp.create_load(sub, bus=int(mv_bus), p_mw=p_load, q_mvar=q_load,
                           name="WARD_3W_MV")

    # ── 8. Synthetic shunts at 3W primary buses ───────────────────────────
    synthetic_shunt_map: Dict[int, int] = {}
    if shunt_buses:
        # For each TSO-owned tertiary shunt, look up which 3W trafo it
        # belongs to (via the plant net's res_trafo3w), then map to the
        # corresponding primary bus we've kept.  hv_list already restricts
        # to this zone's DSOs.
        plant_3w_lv: Dict[int, int] = {}
        for hv in hv_list:
            for t3w, lv_bus, hv_bus in zip(
                hv.coupling_trafo_indices,
                hv.coupling_lv_bus_indices,
                hv.coupling_ieee_buses,
            ):
                plant_3w_lv[int(lv_bus)] = int(hv_bus)
        for tert_bus, q_step in zip(shunt_buses, shunt_q_steps):
            primary = plant_3w_lv.get(int(tert_bus))
            if primary is None or primary not in sub.bus.index:
                continue
            # Read the cached step from the plant net so the synthetic
            # susceptance matches the plant susceptance at the cached
            # operating point.
            plant_shunt_mask = net.shunt["bus"] == int(tert_bus)
            cached_step = 0
            if plant_shunt_mask.any():
                cached_step = int(
                    net.shunt.at[net.shunt.index[plant_shunt_mask][0], "step"]
                )
            pp.create_shunt(
                sub, bus=int(primary), q_mvar=q_step, p_mw=0.0,
                step=cached_step, max_step=10,
                name="SYNTH_TSO_TERTIARY_SHUNT",
            )
            synthetic_shunt_map[int(tert_bus)] = int(primary)

    # ── 9. Slack handling ─────────────────────────────────────────────────
    # Pick a slack gen inside the zone (preserve existing slack-gen if
    # it's in the zone; otherwise promote the largest gen).  Track
    # whether the slack-gen was newly promoted — in that case its
    # machine OLTC sits on the slack-reference bus and the Jacobian's
    # ``compute_dV_ds_2w`` cannot produce a sensitivity column for it.
    # The caller (runner) flags that trafo as out-of-service on the
    # controller via ``promoted_slack_oltc_indices``.
    slack_gen_idx: Optional[int] = None
    slack_promoted: bool = False
    if not sub.gen.empty:
        zone_gen_mask = sub.gen.index.isin([int(g) for g in gen_set])
        zone_gens_all = sub.gen.index[zone_gen_mask].tolist()
        # Filter to in-service gens — a tripped gen can't anchor a slack
        # reference (pp.runpp emits 'No reference bus is available').
        in_service_mask = (
            sub.gen.loc[zone_gens_all, "in_service"].astype(bool)
            if "in_service" in sub.gen.columns
            else None
        )
        if in_service_mask is not None:
            zone_gens = [
                int(g) for g, ok in zip(zone_gens_all, in_service_mask) if bool(ok)
            ]
        else:
            zone_gens = [int(g) for g in zone_gens_all]
        if zone_gens:
            # Identify existing in-service slack-gens
            if "slack" in sub.gen.columns:
                existing_slacks = [
                    int(g) for g in zone_gens
                    if bool(sub.gen.at[g, "slack"])
                ]
            else:
                existing_slacks = []
            if existing_slacks:
                # Keep the first existing slack; clear any others
                slack_gen_idx = existing_slacks[0]
                for g in sub.gen.index:
                    if "slack" in sub.gen.columns:
                        sub.gen.at[g, "slack"] = (int(g) == slack_gen_idx)
            else:
                # Promote the largest in-service gen in the zone to slack
                sn_series = (
                    sub.gen.loc[zone_gens, "sn_mva"]
                    if "sn_mva" in sub.gen.columns else None
                )
                if sn_series is not None and not sn_series.empty:
                    slack_gen_idx = int(sn_series.idxmax())
                else:
                    slack_gen_idx = int(zone_gens[0])
                if "slack" not in sub.gen.columns:
                    sub.gen["slack"] = False
                sub.gen["slack"] = False
                sub.gen.at[slack_gen_idx, "slack"] = True
                slack_promoted = True

    # Machine OLTC(s) attached to the promoted slack gen's terminal bus
    # — these need to be flagged OOS on the controller's mask.  Look up
    # via the trafo's LV bus (machine trafos have lv_bus == gen terminal).
    promoted_slack_oltc_indices: List[int] = []
    if slack_promoted and slack_gen_idx is not None:
        slack_bus = int(sub.gen.at[slack_gen_idx, "bus"])
        for t in machine_trafos_in_zone:
            if t not in sub.trafo.index:
                continue
            if int(sub.trafo.at[t, "lv_bus"]) == slack_bus:
                promoted_slack_oltc_indices.append(int(t))
            elif int(sub.trafo.at[t, "hv_bus"]) == slack_bus:
                promoted_slack_oltc_indices.append(int(t))

    # Clear any external grid (the original network has none on IEEE 39
    # since the slack-gen replaced the ext_grid in build_ieee39_net, but
    # we guard defensively).
    if not sub.ext_grid.empty:
        # Drop external grids that are inside the zone (they conflict with
        # the slack-gen).  Keep none — we use the gen-side slack.
        sub.ext_grid.drop(index=sub.ext_grid.index, inplace=True)

    if slack_gen_idx is None and sub.ext_grid.empty:
        # No slack — return un-converged net; caller will fail at
        # JacobianSensitivities.
        if verbose >= 1:
            print("  [build_tso_local_net] no slack candidate; returning empty result")
        return TSOLocalNetResult(net=sub, synthetic_shunt_map=synthetic_shunt_map,
                                  slack_gen_idx=None)

    # ── 9c. Generators: use CACHED OUTPUT, not the scheduled setpoint ─────
    # The plant solves with ``distributed_slack=True``, so a generator's actual
    # output differs from its ``p_mw`` setpoint -- the machines share the slack
    # burden through their participation factors, and ``res_gen.p_mw`` is what
    # they really produced. The reduced net solves with
    # ``distributed_slack=False``, which holds every PV generator exactly at
    # ``p_mw``; carrying the setpoint over therefore injects power the plant
    # never had. Measured 2026-08-01, zone 0 at 2016-01-05 08:00: setpoints
    # 250.0 / 830.0 MW against cached outputs of 160.1 / 740.1 MW, i.e. 180 MW
    # of phantom injection, which the promoted slack then had to absorb
    # (-214.0 MW solved against +100.6 MW cached).
    #
    # ``res_gen.p_mw`` is the TSO's own machine telemetry, so this uses no
    # information from beyond the zone boundary. The slack generator's p_mw is
    # ignored by the solver and is overwritten here only for consistency.
    #
    # Boundary gens added by ``tie_boundary="pv"`` are skipped by name: they
    # are new rows whose freshly assigned index can collide with a DIFFERENT
    # machine's index in the plant's ``res_gen`` (the reduced net dropped
    # rows, so ``create_gen`` reuses a gap), which would overwrite the
    # boundary in-feed with an unrelated machine's cached output.
    if not sub.gen.empty and net.res_gen is not None and not net.res_gen.empty:
        for g in sub.gen.index:
            if str(sub.gen.at[g, "name"]).startswith("WARD_"):
                continue
            if g in net.res_gen.index:
                p_cached = float(net.res_gen.at[g, "p_mw"])
                if np.isfinite(p_cached):
                    sub.gen.at[g, "p_mw"] = p_cached

    # ── 10. Converge the reduced net ──────────────────────────────────────
    # Try a results warm start from the cached plant state; the shared guard
    # perturbs a genuine PQ/PV Newton state rather than the commonly fixed
    # first/slack bus, falls back to a flat start, and verifies that J exists.
    runpp_with_stored_jacobian(
        sub,
        run_control=False,
        distributed_slack=False,
        calculate_voltage_angles=True,
        enforce_q_lims=False,
    )

    # ── 10b. Operating-point guard ────────────────────────────────────────
    # Every boundary variant matches the cached state by construction, so a
    # converged reduced net that does NOT is on the wrong root and its H is
    # linearised about a fictitious state.  Fail here rather than let the
    # controllers cache it (see the helper's docstring for the 2026-08-13 case).
    _assert_reproduces_cached_state(
        sub, net,
        exclude_buses=boundary_aux_buses,
        tol_pu=op_point_tol_pu,
        label=f"[build_tso_local_net] tie_boundary={tie_boundary!r}: ",
    )

    if verbose >= 2:
        print(f"  [build_tso_local_net] reduced net: "
              f"{len(sub.bus)} buses, {len(sub.line)} lines, "
              f"{len(sub.gen)} gens, {len(sub.load)} loads, "
              f"{len(sub.shunt)} shunts, slack_gen={slack_gen_idx}")

    return TSOLocalNetResult(
        net=sub,
        synthetic_shunt_map=synthetic_shunt_map,
        slack_gen_idx=slack_gen_idx,
        promoted_slack_oltc_indices=tuple(promoted_slack_oltc_indices),
    )


# ---------------------------------------------------------------------------
#  DSO reduction
# ---------------------------------------------------------------------------

def build_dso_local_net(
    net: pp.pandapowerNet,
    hv_info: HVNetworkInfo,
    *,
    boundary: str = "slack",
    thevenin_k: float = 1.0,
    verbose: int = 0,
) -> DSOLocalNetResult:
    """Build the reduced DSO network for one HV sub-network.

    Kept elements:

    * Every 110 kV bus in ``hv_info.bus_indices``.
    * Every 20 kV tertiary bus in ``hv_info.coupling_lv_bus_indices``.
    * Every MV-side coupling bus in ``hv_info.coupling_hv_bus_indices``
      (these are typically the same as the HV sub-network's 110 kV
      buses, but listed separately).
    * Every coupling 3W trafo (controlled by the DSO via OLTC).
    * Every 3W primary bus (HV/TS side) — kept as a *virtual slack-gen*
      pinned to V_cached.
    * Every line, load, sgen, shunt internal to the sub-network or
      attached to the kept buses (including the TSO-owned tertiary shunt
      so the DSO can see its disturbance effect).

    Dropped: every bus, line, trafo, etc., that is not in the keep set
    (i.e., the entire TN backbone, all other zones, all other DSOs).

    Parameters
    ----------
    net : pp.pandapowerNet
        Plant network at the cached operating point.
    hv_info : HVNetworkInfo
        Metadata for the HV sub-network whose local Jacobian we build.
    boundary : {"slack", "thevenin"}
        How the overlaying transmission system is condensed at each
        coupling-transformer primary bus.

        * ``"slack"`` (default, historical): a voltage source placed
          DIRECTLY on the primary bus -- the first coupler carries the
          angle reference, the rest are PV at the cached active in-feed.
          Consequence: the primary buses are a slack and PV buses of the
          reduced network, so they have no ``dV/dQ`` at all.  The slack
          bus is eliminated from the reduced Jacobian outright and PV
          buses have ``d|V|/dQ = 0`` by construction, so the \\gls{DSO}
          cannot monitor, constrain, or even evaluate a sensitivity at
          its own transmission-side terminals.
        * ``"thevenin"``: the same source moved one bus back, behind a
          series impedance of ``thevenin_k`` times the coupling
          transformer's own HV-MV short-circuit impedance, on an
          auxiliary bus.  The primary bus then stays an ordinary PQ bus
          of the reduced network, keeps its row in the reduced Jacobian,
          and therefore HAS a voltage sensitivity.  ``k -> 0`` recovers
          the ``"slack"`` behaviour in the limit.

        Both reproduce the cached operating point at the primary bus, so
        they differ only in the derivative.
    thevenin_k : float
        Boundary impedance for ``boundary="thevenin"``, as a multiple of
        the coupling transformer's HV-MV short-circuit impedance.

    Returns
    -------
    DSOLocalNetResult
    """
    if boundary not in ("slack", "thevenin"):
        raise ValueError(
            f"boundary must be 'slack' or 'thevenin'; got {boundary!r}"
        )
    if boundary == "thevenin" and not (float(thevenin_k) > 0.0):
        raise ValueError(
            f"thevenin_k must be > 0 for boundary='thevenin'; got {thevenin_k}"
        )

    sub = copy.deepcopy(net)

    # ── 1. Build the keep-bus set ─────────────────────────────────────────
    keep_buses: set = set(int(b) for b in hv_info.bus_indices)
    keep_buses.update(int(b) for b in hv_info.coupling_lv_bus_indices)
    keep_buses.update(int(b) for b in hv_info.coupling_hv_bus_indices)
    # The sub-network's own auxiliary buses. Omitting these dropped part of the
    # DSO ITSELF -- `pp.drop_buses` takes their loads and sgens with them, so
    # the reduced net was not a reduction of the DSO but a different network.
    # Measured 2026-08-01 at DSO 4, 2016-01-05 08:00: internal surplus
    # (sgen 265.6 - load 49.7 = 215.9 MW) against a coupler outflow of only
    # 83.8 MW, i.e. 132 MW of the DSO's own injection missing, which the
    # boundary slack then absorbed (-137.0 MW solved against -26.5 MW cached).
    keep_buses.update(int(b) for b in
                      getattr(hv_info, "internal_aux_bus_indices", ()) or ())
    primary_buses: List[int] = [int(b) for b in hv_info.coupling_ieee_buses]
    keep_buses.update(primary_buses)

    # ── 2. Cached V at primary buses ─────────────────────────────────────
    primary_v_cached: Dict[int, float] = {}
    primary_va_cached: Dict[int, float] = {}
    for b in primary_buses:
        if b in sub.res_bus.index:
            primary_v_cached[b] = float(sub.res_bus.at[b, "vm_pu"])
            primary_va_cached[b] = float(sub.res_bus.at[b, "va_degree"])
        else:
            primary_v_cached[b] = 1.0  # fallback
            primary_va_cached[b] = 0.0

    # Cached ACTIVE power the TN supplies into each coupler, keyed by primary
    # bus. Only the first primary bus becomes the slack; the others are PV
    # gens, whose P is FIXED at whatever they are created with. Leaving them at
    # zero forces a multi-coupler DSO to push its entire real-power exchange
    # through the single slack coupler, which is not the combined solution.
    # Measured 2026-08-01 at DSO 4, 2016-01-05 08:00: cached coupler flows were
    # -26.5 / -64.9 / +7.9 MW, so ~73 MW was misrouted through one transformer
    # and the buses behind it sat 0.08-0.10 pu low while those behind the other
    # two were within 0.015 pu.
    #
    # ``p_hv_mw`` is power flowing INTO the trafo at the primary bus; after the
    # TN is deleted the boundary gen is the only other element there, so it must
    # inject exactly that.
    primary_p_cached: Dict[int, float] = {}
    # Reactive counterpart and the coupler's own short-circuit impedance are
    # needed only by the "thevenin" boundary, which must reproduce the FULL
    # cached injection (not just P) to land on the same operating point, and
    # needs a physical impedance scale for its stiffness knob.
    primary_q_cached: Dict[int, float] = {}
    primary_z_ref_ohm: Dict[int, complex] = {}
    for _t3w, _pb in zip(hv_info.coupling_trafo_indices, primary_buses):
        _t3w, _pb = int(_t3w), int(_pb)
        if _t3w in net.res_trafo3w.index:
            primary_p_cached[_pb] = float(net.res_trafo3w.at[_t3w, "p_hv_mw"])
            primary_q_cached[_pb] = float(net.res_trafo3w.at[_t3w, "q_hv_mvar"])
        if _t3w in net.trafo3w.index:
            # Z_hv-mv referred to the HV side, in ohms.
            vk = float(net.trafo3w.at[_t3w, "vk_hv_percent"]) / 100.0
            vkr = float(net.trafo3w.at[_t3w, "vkr_hv_percent"]) / 100.0
            sn = float(net.trafo3w.at[_t3w, "sn_hv_mva"])
            vn = float(net.trafo3w.at[_t3w, "vn_hv_kv"])
            if sn > 0.0 and vn > 0.0 and vk > 0.0:
                z_mag = vk * vn ** 2 / sn
                r = vkr * vn ** 2 / sn
                x = float(np.sqrt(max(z_mag ** 2 - r ** 2, 0.0)))
                primary_z_ref_ohm[_pb] = complex(r, x if x > 0 else z_mag)

    # ── 3. Drop everything not in keep_buses ──────────────────────────────
    # First drop other trafo3w (other DSOs' couplers).  Then pp.drop_buses
    # on the remaining out-of-set TN/other-DSO buses cascades to lines,
    # loads, sgens, gens, shunts.
    other_t3w = [
        int(t) for t in sub.trafo3w.index
        if int(t) not in [int(x) for x in hv_info.coupling_trafo_indices]
    ]
    if other_t3w:
        sub.trafo3w.drop(index=other_t3w, inplace=True)

    remaining_buses = set(int(b) for b in sub.bus.index)
    extra_drop = remaining_buses - keep_buses
    if extra_drop:
        pp.drop_buses(sub, list(extra_drop))

    # ── 4. Strip every element at the primary buses EXCEPT 3W trafos ─────
    # The primary buses become slack-gen anchors; everything else there
    # (loads from upstream zone, gens, other 2W trafos, lines, sgens) goes
    # away.
    for b in primary_buses:
        if b not in sub.bus.index:
            continue
        # Loads
        mask = sub.load["bus"] == b
        if mask.any():
            sub.load.drop(index=sub.load.index[mask], inplace=True)
        # Sgens
        mask = sub.sgen["bus"] == b
        if mask.any():
            sub.sgen.drop(index=sub.sgen.index[mask], inplace=True)
        # Gens (including the original IEEE 39 slack-gen if one was here)
        mask = sub.gen["bus"] == b
        if mask.any():
            sub.gen.drop(index=sub.gen.index[mask], inplace=True)
        # Shunts attached to the primary bus
        if not sub.shunt.empty:
            mask = sub.shunt["bus"] == b
            if mask.any():
                sub.shunt.drop(index=sub.shunt.index[mask], inplace=True)
        # Lines attached to the primary bus (any remaining TN line stub)
        mask_line = (sub.line["from_bus"] == b) | (sub.line["to_bus"] == b)
        for li in sub.line.index[mask_line]:
            sub.line.drop(index=li, inplace=True)
        # 2W trafos with a leg at the primary bus
        mask_tr = (sub.trafo["hv_bus"] == b) | (sub.trafo["lv_bus"] == b)
        for t in sub.trafo.index[mask_tr]:
            sub.trafo.drop(index=t, inplace=True)

    # ── 5. Add virtual slack-gens at each primary bus ─────────────────────
    virtual_slacks: List[int] = []
    if "slack" not in sub.gen.columns:
        sub.gen["slack"] = False
    else:
        # Clear any stray slack flag inherited from the original net.
        sub.gen["slack"] = False
    if not sub.ext_grid.empty:
        sub.ext_grid.drop(index=sub.ext_grid.index, inplace=True)
    for k, b in enumerate(primary_buses):
        if b not in sub.bus.index:
            continue
        v_cached = primary_v_cached.get(b, 1.0)
        # Only the first primary bus becomes the true slack; any
        # additional primary buses (multi-trafo DSOs) get pinned via PV
        # gens at the same V_cached.  pandapower allows only one slack.
        is_slack = (k == 0)

        if boundary == "thevenin":
            # Move the source one bus back, behind a finite impedance, so
            # the primary bus itself stays an ordinary PQ bus of the
            # reduced network and keeps its Jacobian row.
            z_ref = primary_z_ref_ohm.get(int(b))
            if z_ref is None or abs(z_ref) <= 0.0:
                vn_b = float(sub.bus.at[b, "vn_kv"])
                z_ref = complex(0.0, 0.10 * vn_b ** 2 / _S_BASE_MVA)
            _aux, _br, gi = add_thevenin_boundary(
                sub, int(b),
                float(primary_p_cached.get(int(b), 0.0)),
                float(primary_q_cached.get(int(b), 0.0)),
                complex(thevenin_k) * z_ref,
                vm_pu=v_cached,
                va_degree=primary_va_cached.get(int(b), 0.0),
                slack=is_slack,
                name="WARD_DSO_BOUNDARY_TH",
            )
            virtual_slacks.append(int(gi))
            continue

        # The slack gen's p_mw is ignored by the solver; for the PV boundary
        # gens it IS the coupler's real-power exchange and must not be 0.
        gi = pp.create_gen(
            sub, bus=int(b), p_mw=float(primary_p_cached.get(int(b), 0.0)),
            vm_pu=float(v_cached),
            slack=is_slack,
            min_p_mw=-1e6, max_p_mw=1e6,
            min_q_mvar=-1e6, max_q_mvar=1e6,
            name=f"WARD_DSO_BOUNDARY_{k}",
        )
        virtual_slacks.append(int(gi))

    # ── 6. Converge ───────────────────────────────────────────────────────
    # Start flat so NR actually runs and the Jacobian gets stored (see comment
    # in :func:`build_tso_local_net`), then fall back to a DC start.
    #
    # The flat point is a poor guess for a reduced net carrying heavy DER
    # injection behind a single Ward boundary slack, and it is not merely slow
    # but divergent: measured 2026-07-30 on scenario ``rural_700`` with the DER
    # Q(V) dead zone at delta = 0, where every DER answers any deviation so the
    # reduced net's |Q| is at its largest, NR failed after 100 flat iterations
    # (delta = 0.005 on the same window converges).  The DC start supplies a
    # consistent angle profile and gets through.
    #
    # Routed through the shared helper, as :func:`build_tso_local_net` already
    # is, so the "J was actually stored" guarantee is enforced here too rather
    # than being inferred from the flat start having iterated -- the caller
    # feeds this net straight into JacobianSensitivities, which requires J.
    runpp_with_stored_jacobian(
        sub,
        run_control=False,
        distributed_slack=False,
        calculate_voltage_angles=True,
        enforce_q_lims=False,
        # DC FIRST, flat only as a fallback. The ladder is tried in order and
        # stops at the first convergence, so leading with a flat start meant
        # the DC start was never reached whenever flat converged -- and on this
        # network flat converges to the LOW-VOLTAGE root. Measured 2026-08-01
        # at 2016-01-05 08:00: the flat solution put the whole HV sub-network
        # near 0.74 pu against 1.03 pu in the combined solution (max error
        # 0.29 pu, tertiary buses collapsing to 0.0), while the DC start lands
        # within 0.023 pu. Every DSO sensitivity was therefore linearised about
        # a point the plant is nowhere near. A flat start is a poor guess for a
        # sub-network carrying heavy DER injection behind one boundary slack;
        # the DC start supplies a consistent angle profile and finds the
        # operational branch.
        init_sequence=(("dc", 200), ("flat", 100)),
    )

    if verbose >= 2:
        print(f"  [build_dso_local_net {hv_info.net_id}] reduced net: "
              f"{len(sub.bus)} buses, {len(sub.line)} lines, "
              f"{len(sub.gen)} gens, {len(sub.load)} loads, "
              f"{len(sub.shunt)} shunts, "
              f"virtual_slacks={virtual_slacks}")

    return DSOLocalNetResult(
        net=sub,
        virtual_slack_gen_indices=tuple(virtual_slacks),
    )
