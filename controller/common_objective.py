"""
controller/common_objective.py
==============================
Common coordination objective Φ = Σ_i Φ_i for the Boundary Marginal
Exchange (BME) scheme — spec §3.3, §3.4 and §3.5 Convention A (see
``docs/BME_STATUS.md``).

Symbol map (code ↔ spec)
------------------------
* ``phi_zone(net, i)``    ↔ Φ_i(v) = w_loss · P_i^loss(v)
                            + Σ_{n ∈ N_i^own} φ_band(v_n)  (§3.3), with
                            P_i^loss the ACTUAL active losses over the
                            branches owned by area i (res tables; tie-line
                            losses split per D1, default 50/50) and
                            N_i^own = ``topology.zone_buses(i)``.
* ``phi_global(net)``     ↔ Φ(v), computed INDEPENDENTLY of the ownership
                            map (all in-service branches, all in-service
                            buses) — the oracle for the partition
                            invariant Σ_i Φ_i == Φ_global (§3.3).
* ``phi_band(v)``         ↔ φ_band(v) = w_band · ([max(0, v − v^soft,max)]²
                            + [max(0, v^soft,min − v)]²) — a C¹ quadratic
                            hinge; its second derivative is discontinuous
                            at the band edges (note for Q_Φ, §3.3/§3.10).
* ``ZoneGradients.mu()``  ↔ μ_i = dΦ_i/dv_b (§3.4), assembled area-locally:
                            (∂x_int/∂v_b)ᵀ·∇_{x_int}Φ_i (via
                            ``MarginalComputer.mu_x``) plus the direct
                            terms ∂Φ_i/∂v_b at adjacent boundary buses.
* ``ZoneGradients.d_*``   ↔ components of g_i^own = ∂Φ_i/∂u_i |_{v_b fixed}
                            (§3.5 Convention A): port-frozen internal
                            response chained with ∇_{x_int}Φ_i, plus the
                            explicit ∂Φ_i/∂u_i terms (V_gen, taps).

Loss-gradient formulation
-------------------------
dP^loss/d(θ, V) is analytic from the ppc branch flow equations:
S_f = diag(C_f V)·conj(Y_f V), S_t = diag(C_t V)·conj(Y_t V),
P^loss_ℓ = Re(S_f,ℓ + S_t,ℓ). With w the ownership weight per ppc branch
(1 owned, tie share on ties, 0 otherwise) and
r1 = C_fᵀ(w ∘ conj(I_f)) + C_tᵀ(w ∘ conj(I_t)),
r2 = conj(Y_fᵀ)(w ∘ V_f) + conj(Y_tᵀ)(w ∘ V_t):

    dP/dθ = Re( j·(V ∘ r1 − conj(V) ∘ r2) )        [pu per rad]
    dP/dV = Re( E ∘ r1 + conj(E) ∘ r2 ),  E = V/|V| [pu per pu]

(the standard MATPOWER dSbr_dV identities, row-summed with weights).
Values are converted to MW with the system base, matching the res tables
(reconciliation verified numerically at build time of this module).

Direct tap term: for an owned 2W transformer with hv-side tap ratio τ,
∂S_f/∂τ = −(S_f + |V_f|²·conj(Y_ff))/τ and
∂S_t/∂τ = −(S_t − |V_t|²·conj(Y_tt))/τ, so
∂P^loss_ℓ/∂τ = Re(∂S_f/∂τ + ∂S_t/∂τ).

Locality note (§3.9)
--------------------
Every Φ_i gradient piece uses only zone-owned branches, zone-owned buses
and the zone's own port-frozen operators — area-local by construction.
The single supra-local object of the scheme (H_{b,i}) does NOT appear in
this module; it enters the price term in Phase 4 via the
``RestrictedSensitivityProvider``.

Fail-fast: non-converged results (NaN), unknown zones, foreign actuators,
cross-zone branches that are not registered ties, and missing tie lines
raise with precise messages; there are no silent defaults (w_band has no
default — DECISION D2 leaves its magnitude to the Phase 6 calibration).

Author: Manuel Schwenke / Claude Code
Date: 2026-07-02 (BME Phase 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandapower as pp
from numpy.typing import NDArray

from network.boundary_topology import BoundaryTopology
from sensitivity.marginal_computer import MarginalComputer
from sensitivity.index_helper import (
    get_ppc_line_index,
    get_ppc_trafo_index,
    get_ppc_trafo3w_branch_indices,
    pp_bus_to_ppc_bus,
)


@dataclass(frozen=True)
class PhiBreakdown:
    """Value of Φ_i split into its two terms (§3.3)."""

    zone_id: int
    loss_mw: float          # P_i^loss (owned branches, tie shares applied)
    band_penalty: float     # Σ φ_band over N_i^own (already × w_band)
    w_loss: float

    @property
    def total(self) -> float:
        return self.w_loss * self.loss_mw + self.band_penalty


class CommonObjective:
    """Per-area Φ_i, the global Φ oracle, and area-local gradients.

    Parameters
    ----------
    topology :
        Boundary topology carrying the ownership map (D1).
    w_band :
        Band-penalty weight (must be given explicitly — its magnitude is
        a Phase 6 calibration item, D2; ``0.0`` is the losses-only
        ablation rung).
    w_loss :
        Loss weight (D2: 1.0).
    v_soft_min, v_soft_max :
        Soft band edges in pu (D2 starting point: ±3 % around nominal).
    """

    def __init__(
        self,
        topology: BoundaryTopology,
        *,
        w_band: float,
        w_loss: float = 1.0,
        v_soft_min: float = 0.97,
        v_soft_max: float = 1.03,
    ) -> None:
        if w_loss < 0.0 or w_band < 0.0:
            raise ValueError(
                f"w_loss ({w_loss}) and w_band ({w_band}) must be ≥ 0"
            )
        if not (v_soft_min < v_soft_max):
            raise ValueError(
                f"v_soft_min ({v_soft_min}) must be < v_soft_max "
                f"({v_soft_max})"
            )
        self.topology = topology
        self.w_loss = float(w_loss)
        self.w_band = float(w_band)
        self.v_soft_min = float(v_soft_min)
        self.v_soft_max = float(v_soft_max)

    # ------------------------------------------------------------------
    #  φ_band hinge (§3.3)
    # ------------------------------------------------------------------

    def phi_band(self, v: float) -> float:
        """φ_band(v): zero inside [v^soft,min, v^soft,max], quadratic
        outside; C¹ at the edges."""
        over = max(0.0, float(v) - self.v_soft_max)
        under = max(0.0, self.v_soft_min - float(v))
        return self.w_band * (over * over + under * under)

    def phi_band_grad(self, v: float) -> float:
        """dφ_band/dv — continuous (the hinge is C¹); the second
        derivative jumps from 0 to 2·w_band at the edges."""
        over = max(0.0, float(v) - self.v_soft_max)
        under = max(0.0, self.v_soft_min - float(v))
        return 2.0 * self.w_band * (over - under)

    # ------------------------------------------------------------------
    #  Values (from converged power-flow results)
    # ------------------------------------------------------------------

    @staticmethod
    def _require_results(net: pp.pandapowerNet) -> None:
        if not len(net.res_bus):
            raise ValueError(
                "the net carries no power-flow results — run a converged "
                "power flow before evaluating Φ."
            )

    def phi_zone(self, net: pp.pandapowerNet, zone: int) -> PhiBreakdown:
        """Φ_i on a converged net (the plant, or a test sub-network that
        contains all of the zone's owned branches and buses).

        The net's res tables must be populated; NaN results raise.
        """
        topo = self.topology
        if zone not in topo.zone_ids:
            raise ValueError(
                f"unknown zone {zone}; known: {topo.zone_ids}"
            )
        self._require_results(net)
        shares = topo.tie_loss_shares()

        # Every tie the zone owns a share of must be present — a subnet
        # missing its ties would silently lose the D1 share.
        for tie in topo.zone_ties(zone):
            if tie.line_idx not in net.line.index:
                raise ValueError(
                    f"zone {zone}: tie line {tie.line_idx} is missing from "
                    "the given net — Φ_i requires all owned branches."
                )

        loss = 0.0
        for li in net.line.index:
            if not bool(net.line.at[li, "in_service"]):
                continue
            zf = topo.bus_owner(int(net.line.at[li, "from_bus"]))
            zt = topo.bus_owner(int(net.line.at[li, "to_bus"]))
            if zf == zt:
                w = 1.0 if zf == zone else 0.0
            else:
                if int(li) not in shares:
                    raise ValueError(
                        f"line {li} spans zones {zf}–{zt} but is not a "
                        "registered tie — ownership map (D1) violated."
                    )
                w = shares[int(li)].get(zone, 0.0)
            if w == 0.0:
                continue
            pl = float(net.res_line.at[li, "pl_mw"])
            if not np.isfinite(pl):
                raise ValueError(
                    f"res_line.pl_mw at line {li} is not finite — run a "
                    "converged power flow before evaluating Φ."
                )
            loss += w * pl

        loss += self._owned_trafo_losses(net, zone)

        band = 0.0
        for b in topo.zone_buses(zone):
            if b not in net.bus.index:
                raise ValueError(
                    f"zone {zone}: owned bus {b} is missing from the "
                    "given net — Φ_i requires all owned buses."
                )
            if not bool(net.bus.at[b, "in_service"]):
                continue
            vm = float(net.res_bus.at[b, "vm_pu"])
            if not np.isfinite(vm):
                raise ValueError(
                    f"res_bus.vm_pu at bus {b} is not finite — run a "
                    "converged power flow before evaluating Φ."
                )
            band += self.phi_band(vm)

        return PhiBreakdown(
            zone_id=int(zone), loss_mw=loss, band_penalty=band,
            w_loss=self.w_loss,
        )

    def _owned_trafo_losses(self, net: pp.pandapowerNet, zone: int) -> float:
        """2W/3W transformer losses of the zone (transformers never cross
        zones — the topology build asserts this on the plant net; it is
        re-checked here because Φ may be evaluated on other nets)."""
        topo = self.topology
        loss = 0.0
        for t in net.trafo.index:
            if not bool(net.trafo.at[t, "in_service"]):
                continue
            owners = {
                topo.bus_owner(int(net.trafo.at[t, c]))
                for c in ("hv_bus", "lv_bus")
            }
            if len(owners) > 1:
                raise ValueError(
                    f"trafo {t} spans zones {sorted(owners)} — separator "
                    "assumption violated (spec §3.2)."
                )
            if owners != {zone}:
                continue
            pl = float(net.res_trafo.at[t, "pl_mw"])
            if not np.isfinite(pl):
                raise ValueError(
                    f"res_trafo.pl_mw at trafo {t} is not finite — run a "
                    "converged power flow before evaluating Φ."
                )
            loss += pl
        if hasattr(net, "trafo3w") and len(net.trafo3w):
            for t in net.trafo3w.index:
                if not bool(net.trafo3w.at[t, "in_service"]):
                    continue
                owners = {
                    topo.bus_owner(int(net.trafo3w.at[t, c]))
                    for c in ("hv_bus", "mv_bus", "lv_bus")
                }
                if len(owners) > 1:
                    raise ValueError(
                        f"trafo3w {t} spans zones {sorted(owners)} — "
                        "separator assumption violated (spec §3.2)."
                    )
                if owners != {zone}:
                    continue
                pl = float(net.res_trafo3w.at[t, "pl_mw"])
                if not np.isfinite(pl):
                    raise ValueError(
                        f"res_trafo3w.pl_mw at trafo3w {t} is not finite."
                    )
                loss += pl
        return loss

    def phi_global(self, net: pp.pandapowerNet) -> float:
        """Global Φ, computed WITHOUT the ownership map (§3.3 oracle):
        all in-service branch losses plus φ_band over all in-service
        buses. The partition invariant Σ_i Φ_i == Φ_global is a unit
        test, not an assumption."""
        self._require_results(net)
        loss = 0.0
        for tab, res in (
            ("line", "res_line"),
            ("trafo", "res_trafo"),
            ("trafo3w", "res_trafo3w"),
        ):
            if not hasattr(net, tab) or not len(getattr(net, tab)):
                continue
            table = getattr(net, tab)
            res_table = getattr(net, res)
            for i in table.index:
                if not bool(table.at[i, "in_service"]):
                    continue
                pl = float(res_table.at[i, "pl_mw"])
                if not np.isfinite(pl):
                    raise ValueError(
                        f"{res}.pl_mw at {tab} {i} is not finite — run a "
                        "converged power flow before evaluating Φ."
                    )
                loss += pl

        band = 0.0
        for b in net.bus.index:
            if not bool(net.bus.at[b, "in_service"]):
                continue
            vm = float(net.res_bus.at[b, "vm_pu"])
            if not np.isfinite(vm):
                raise ValueError(
                    f"res_bus.vm_pu at bus {b} is not finite — run a "
                    "converged power flow before evaluating Φ."
                )
            band += self.phi_band(vm)

        return self.w_loss * loss + band

    # ------------------------------------------------------------------
    #  Gradients (area-local, at a cached Jacobian operating point)
    # ------------------------------------------------------------------

    def gradients(self, comp: MarginalComputer) -> "ZoneGradients":
        """Gradient bundle of Φ_zone at the operating point cached in
        ``comp.sens`` — everything needed for μ_i (§3.4) and for the
        Convention-A own gradient g_i^own (§3.5)."""
        if comp.topology is not self.topology:
            raise ValueError(
                "MarginalComputer and CommonObjective must share the same "
                "BoundaryTopology instance."
            )
        return ZoneGradients(self, comp)


class ZoneGradients:
    """Area-local gradient bundle of Φ_i for one zone at one operating
    point (the state cached in the zone's :class:`MarginalComputer`).

    Not built directly — use :meth:`CommonObjective.gradients`.
    """

    def __init__(
        self, objective: CommonObjective, comp: MarginalComputer
    ) -> None:
        self._obj = objective
        self._comp = comp
        self._topo = objective.topology
        self._zone = comp.zone_id
        self._sens = comp.sens
        self._net = self._sens.net
        self._sn = float(self._net.sn_mva)

        self._build_loss_state_gradient()
        self._build_grad_x_and_direct()

    # -- ppc-space loss gradient over owned branches -------------------

    def _build_loss_state_gradient(self) -> None:
        net = self._net
        internal = net._ppc["internal"]
        Yf = internal["Yf"].tocsr()
        Yt = internal["Yt"].tocsr()
        n_bus = Yf.shape[1]

        bus = net._ppc["bus"]
        V = bus[:n_bus, 7] * np.exp(1j * np.deg2rad(bus[:n_bus, 8]))
        br = net._ppc["branch"]
        f = np.real(br[:, 0]).astype(int)
        t = np.real(br[:, 1]).astype(int)
        status = np.real(br[:, 10]).astype(float)

        w = self._branch_weights(int(br.shape[0])) * status

        If = Yf @ V
        It = Yt @ V
        r1 = np.zeros(n_bus, dtype=np.complex128)
        np.add.at(r1, f, w * np.conj(If))
        np.add.at(r1, t, w * np.conj(It))
        r2 = Yf.conj().T @ (w * V[f]) + Yt.conj().T @ (w * V[t])
        r2 = np.asarray(r2).ravel()

        E = V / np.abs(V)
        # MW per rad / MW per pu (matches the res-table loss values)
        self._gVa = np.real(1j * (V * r1 - np.conj(V) * r2)) * self._sn
        self._gVm = np.real(E * r1 + np.conj(E) * r2) * self._sn
        self._V_ppc = V

    def _branch_weights(self, n_br: int) -> NDArray[np.float64]:
        """Ownership weight per ppc branch (D1): 1 for owned branches,
        the tie share on ties, 0 otherwise."""
        net = self._net
        topo = self._topo
        zone = self._zone
        shares = topo.tie_loss_shares()
        w = np.zeros(n_br, dtype=np.float64)

        for li in net.line.index:
            if not bool(net.line.at[li, "in_service"]):
                continue
            ppc = get_ppc_line_index(net, li)
            if ppc is None:
                raise ValueError(
                    f"line {li}: no ppc branch index — internal lookup "
                    "inconsistent with the cached power flow."
                )
            zf = topo.bus_owner(int(net.line.at[li, "from_bus"]))
            zt = topo.bus_owner(int(net.line.at[li, "to_bus"]))
            if zf == zt:
                w[ppc] = 1.0 if zf == zone else 0.0
            else:
                if int(li) not in shares:
                    raise ValueError(
                        f"line {li} spans zones {zf}–{zt} but is not a "
                        "registered tie — ownership map (D1) violated."
                    )
                w[ppc] = shares[int(li)].get(zone, 0.0)

        for t in net.trafo.index:
            if not bool(net.trafo.at[t, "in_service"]):
                continue
            owners = {
                topo.bus_owner(int(net.trafo.at[t, c]))
                for c in ("hv_bus", "lv_bus")
            }
            if len(owners) > 1:
                raise ValueError(
                    f"trafo {t} spans zones {sorted(owners)} — separator "
                    "assumption violated (spec §3.2)."
                )
            if owners == {zone}:
                ppc = get_ppc_trafo_index(net, t)
                if ppc is None:
                    raise ValueError(
                        f"trafo {t}: no ppc branch index — internal "
                        "lookup inconsistent with the cached power flow."
                    )
                w[ppc] = 1.0

        if hasattr(net, "trafo3w") and len(net.trafo3w):
            for t in net.trafo3w.index:
                if not bool(net.trafo3w.at[t, "in_service"]):
                    continue
                owners = {
                    topo.bus_owner(int(net.trafo3w.at[t, c]))
                    for c in ("hv_bus", "mv_bus", "lv_bus")
                }
                if len(owners) > 1:
                    raise ValueError(
                        f"trafo3w {t} spans zones {sorted(owners)} — "
                        "separator assumption violated (spec §3.2)."
                    )
                if owners == {zone}:
                    hv_br, mv_br, lv_br, _ = (
                        get_ppc_trafo3w_branch_indices(net, int(t))
                    )
                    w[hv_br] = 1.0
                    w[mv_br] = 1.0
                    w[lv_br] = 1.0
        return w

    # -- state gradient aligned with the MarginalComputer --------------

    def _build_grad_x_and_direct(self) -> None:
        obj = self._obj
        net = self._net
        _, labels = self._comp.response_full()

        grad = np.zeros(len(labels), dtype=np.float64)
        for i, (kind, ident) in enumerate(labels):
            if kind == "theta":
                grad[i] = obj.w_loss * self._gVa[
                    pp_bus_to_ppc_bus(net, int(ident))
                ]
            elif kind == "v":
                ppc_b = pp_bus_to_ppc_bus(net, int(ident))
                vm = float(net.res_bus.at[int(ident), "vm_pu"])
                grad[i] = (
                    obj.w_loss * self._gVm[ppc_b]
                    + obj.phi_band_grad(vm)
                )
            elif kind == "theta_aux":
                # 3W star states carry a ppc index directly; no band
                # penalty on auxiliary (non-physical) buses.
                grad[i] = obj.w_loss * self._gVa[int(ident)]
            elif kind == "v_aux":
                grad[i] = obj.w_loss * self._gVm[int(ident)]
            else:  # pragma: no cover - future label kinds must fail loud
                raise ValueError(f"unknown state label kind '{kind}'")
        self._grad_x = grad

        # Direct terms ∂Φ_i/∂v_b (§3.4): explicit dependence of owned
        # branch losses on adjacent boundary magnitudes; band penalty
        # only at the zone's OWN boundary buses (D1 — the far endpoint's
        # band belongs to its owner).
        own_ports = set(self._comp.ports)
        direct: Dict[int, float] = {}
        for b in self._comp.adjacent:
            val = obj.w_loss * self._gVm[pp_bus_to_ppc_bus(net, int(b))]
            if b in own_ports:
                vm = float(net.res_bus.at[int(b), "vm_pu"])
                val += obj.phi_band_grad(vm)
            direct[int(b)] = float(val)
        self._mu_direct = direct

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @property
    def grad_x_int(self) -> NDArray[np.float64]:
        """∇_{x_int} Φ_zone aligned with the MarginalComputer's state
        labels (θ in MW/rad, V in MW/pu plus the band term)."""
        return self._grad_x.copy()

    @property
    def mu_direct(self) -> Dict[int, float]:
        """Direct terms ∂Φ_zone/∂v_b per adjacent boundary bus."""
        return dict(self._mu_direct)

    def mu(self) -> NDArray[np.float64]:
        """μ_zone = dΦ_zone/dv_b ∈ R^{|B|} (§3.4), registry order,
        exactly zero outside the zone's adjacent boundary set."""
        return self._comp.mu_x(self._grad_x, self._mu_direct)

    def d_q_injection(self, bus: int) -> float:
        """∂Φ_zone/∂Q |_{v_b fixed} for a reactive injection at a
        zone-owned bus, per Mvar (generator convention). No explicit
        term: Φ depends on injections only through the state."""
        resp = self._comp.response_to_q_injection(int(bus))
        return float(self._grad_x @ resp)

    def d_pcc_set(self, hv_bus: int) -> float:
        """∂Φ_zone/∂Q_PCC^set |_{v_b fixed} per Mvar — load convention
        on the HV port, i.e. the negated injection column (mirrors the
        controller and the RestrictedSensitivityProvider)."""
        return -self.d_q_injection(int(hv_bus))

    def d_vgen(self, gen_idx: int) -> float:
        """∂Φ_zone/∂V_gen |_{v_b fixed} per pu for a zone-owned PV
        generator: port-frozen internal response plus the explicit
        terms at the pinned terminal bus (owned branch losses and its
        band penalty — the terminal magnitude IS the input)."""
        net = self._net
        if gen_idx not in net.gen.index:
            raise ValueError(f"gen {gen_idx} not in net.gen")
        term_bus = int(net.gen.at[gen_idx, "bus"])
        resp = self._comp.response_to_vgen(term_bus)
        vm = float(net.res_bus.at[term_bus, "vm_pu"])
        direct = (
            self._obj.w_loss * self._gVm[pp_bus_to_ppc_bus(net, term_bus)]
            + self._obj.phi_band_grad(vm)
        )
        return float(self._grad_x @ resp) + float(direct)

    def d_tap_2w(self, trafo_idx: int) -> float:
        """∂Φ_zone/∂s |_{v_b fixed} per whole tap step for a zone-owned
        2W transformer: port-frozen internal response plus the explicit
        ∂P^loss_ℓ/∂τ term of the transformer's own branch."""
        resp = self._comp.response_to_tap_2w(int(trafo_idx))
        indirect = float(self._grad_x @ resp)
        direct = (
            self._obj.w_loss * self._dploss_dtau_step(int(trafo_idx))
        )
        return indirect + direct

    def d_shunt(self, bus: int, q_step_mvar: float) -> float:
        """∂Φ_zone/∂s |_{v_b fixed} per shunt step at a zone-owned bus.
        Mirrors ``compute_dV_dQ_shunt``: load-convention sign flip and
        the constant-susceptance V² scaling of the rated step."""
        resp = self._comp.response_to_q_injection(int(bus))
        vm = float(self._net.res_bus.at[int(bus), "vm_pu"])
        return float(
            -float(q_step_mvar) * vm * vm * (self._grad_x @ resp)
        )

    # -- explicit tap-loss derivative -----------------------------------

    def _dploss_dtau_step(self, trafo_idx: int) -> float:
        """Explicit ∂P^loss_ℓ/∂s [MW per tap step] of the transformer's
        own ppc branch (module header formulas), zero-weighted if the
        transformer is not owned (unreachable after the ownership check
        in ``response_to_tap_2w``)."""
        net = self._net
        ppc_br = get_ppc_trafo_index(net, trafo_idx)
        if ppc_br is None:
            raise ValueError(
                f"trafo {trafo_idx}: no ppc branch index."
            )
        row = net._ppc["branch"][ppc_br]
        if float(np.real(row[10])) == 0.0:
            raise ValueError(f"trafo {trafo_idx} is out of service.")

        tau = float(np.real(row[8]))
        if tau == 0.0:
            tau = 1.0
        shift = np.deg2rad(float(np.real(row[9])))
        ys = 1.0 / complex(float(np.real(row[2])), float(np.real(row[3])))
        bc = float(np.real(row[4]))
        Yff = (ys + 1j * bc / 2.0) / (tau * tau)
        Ytt = ys + 1j * bc / 2.0
        Yft = -ys / (tau * np.exp(-1j * shift))
        Ytf = -ys / (tau * np.exp(1j * shift))

        fb = int(np.real(row[0]))
        tb = int(np.real(row[1]))
        Vf = self._V_ppc[fb]
        Vt = self._V_ppc[tb]
        Sf = Vf * np.conj(Yff * Vf + Yft * Vt)
        St = Vt * np.conj(Ytf * Vf + Ytt * Vt)
        dPf = np.real(-(Sf + abs(Vf) ** 2 * np.conj(Yff)) / tau)
        dPt = np.real(-(St - abs(Vt) ** 2 * np.conj(Ytt)) / tau)

        s0 = float(net.trafo.at[trafo_idx, "tap_pos"])
        delta_tau = float(net.trafo.at[trafo_idx, "tap_step_percent"]) / 100.0
        side = str(net.trafo.at[trafo_idx, "tap_side"])
        denom = 1.0 + s0 * delta_tau
        if denom == 0.0:
            raise ValueError(
                f"trafo {trafo_idx}: degenerate tap ratio (1 + s·Δτ = 0)."
            )
        # τ enters multiplicatively: τ = c·(1 + s·Δτ) on the hv side,
        # τ = c/(1 + s·Δτ) on the lv side.
        dtau_ds = tau * delta_tau / denom
        if side == "lv":
            dtau_ds = -dtau_ds
        elif side != "hv":
            raise ValueError(
                f"trafo {trafo_idx}: unsupported tap_side '{side}'."
            )
        return float((dPf + dPt) * dtau_ds * self._sn)
