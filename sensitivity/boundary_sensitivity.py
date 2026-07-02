"""
sensitivity/boundary_sensitivity.py
===================================
Boundary sensitivity H_{b,i} behind an access-restriction wrapper — BME
spec §3.5 and §3.9 (see ``docs/BME_STATUS.md``).

Symbol map (code ↔ spec)
------------------------
* ``h_b(zone)``  ↔ H_{b,i} = ∂v_b/∂u_i ∈ R^{|B| × n_i} — sensitivity of ALL
                   boundary voltage magnitudes (registry order of
                   :class:`network.boundary_topology.BoundaryTopology`) to
                   zone i's inputs, continuous AND discrete columns.
* ``ZoneInputSpec`` ↔ the column structure of u_i, mirroring the TSO
                   controller's ordering
                   ``[Q_DER (bus-level) | Q_PCC_set | V_gen | s_OLTC | s_shunt]``
                   (see ``TSOController._build_sensitivity_matrix``,
                   controller/tso_controller.py). DER columns are BUS-level;
                   per-DER expansion happens controller-side via the existing
                   ``_expand_H_to_der_level`` E-matrix (Phase 4).
* Access restriction ↔ §3.9's informational concession, made enforceable:
                   a zone-bound :class:`ZoneBoundaryView` exposes ONLY
                   rows(B) × cols(u_i); any other access raises.

Conventions (mirroring the controller exactly)
----------------------------------------------
* Q_DER / shunt / PCC columns are per-Mvar; OLTC columns are per tap step.
* Q_PCC_set uses LOAD convention on the HV port → the generator-convention
  Jacobian column is negated (see controller/tso_controller.py, PCC column
  assembly).
* Shunt columns delegate to ``compute_dV_dQ_shunt`` (load-convention flip
  and V² scaling live there), with the same ``shunt_q_steps_mvar`` values
  the controller config carries.
* Boundary buses without a voltage state in the Jacobian (slack / PV buses,
  e.g. IEEE 39 bus 38) get an exactly-zero row: their magnitude is pinned by
  a voltage source, so ∂v_b/∂u = 0 structurally. They are listed in
  ``pinned_boundary_buses`` rather than silently absorbed.

Fail-fast: every requested actuator must resolve to a Jacobian column;
anything missing raises with the offending identifier.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-02 (BME Phase 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from network.boundary_topology import BoundaryTopology
from sensitivity.jacobian import JacobianSensitivities
from sensitivity.index_helper import get_jacobian_indices


@dataclass(frozen=True)
class ZoneInputSpec:
    """Column specification of one zone's input vector u_i (§3.1).

    Column order (fixed, mirrors the TSO controller):
    ``[Q_DER (bus-level) | Q_PCC_set | V_gen | s_OLTC | s_shunt]``.
    """

    zone_id: int
    der_bus_indices: Tuple[int, ...] = field(default_factory=tuple)
    pcc_trafo_indices: Tuple[int, ...] = field(default_factory=tuple)
    gen_indices: Tuple[int, ...] = field(default_factory=tuple)
    oltc_trafo_indices: Tuple[int, ...] = field(default_factory=tuple)
    shunt_bus_indices: Tuple[int, ...] = field(default_factory=tuple)
    shunt_q_steps_mvar: Tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.shunt_bus_indices) != len(self.shunt_q_steps_mvar):
            raise ValueError(
                f"zone {self.zone_id}: shunt_bus_indices "
                f"({len(self.shunt_bus_indices)}) and shunt_q_steps_mvar "
                f"({len(self.shunt_q_steps_mvar)}) must have equal length"
            )

    @property
    def n_columns(self) -> int:
        return (
            len(self.der_bus_indices)
            + len(self.pcc_trafo_indices)
            + len(self.gen_indices)
            + len(self.oltc_trafo_indices)
            + len(self.shunt_bus_indices)
        )

    def column_labels(self) -> List[Tuple[str, int]]:
        """``(kind, identifier)`` per column, for tests and diagnostics."""
        labels: List[Tuple[str, int]] = []
        labels += [("der", int(b)) for b in self.der_bus_indices]
        labels += [("pcc", int(t)) for t in self.pcc_trafo_indices]
        labels += [("vgen", int(g)) for g in self.gen_indices]
        labels += [("oltc", int(t)) for t in self.oltc_trafo_indices]
        labels += [("shunt", int(b)) for b in self.shunt_bus_indices]
        return labels


class ZoneBoundaryView:
    """Zone-bound access handle: exposes ONLY this zone's H_{b,i}."""

    def __init__(
        self, provider: "RestrictedSensitivityProvider", zone_id: int
    ) -> None:
        self._provider = provider
        self._zone_id = int(zone_id)

    @property
    def zone_id(self) -> int:
        return self._zone_id

    def h_b(self) -> NDArray[np.float64]:
        """H_{b,zone} — rows(B) × cols(u_zone). The only permitted read."""
        return self._provider.h_b(self._zone_id)


class RestrictedSensitivityProvider:
    """Wraps the global Jacobian; serves H_{b,i} per zone, nothing else.

    The provider is constructed with the full-network
    :class:`JacobianSensitivities` (the shared instance of the runner's
    default mode — audit A1, Convention B) but each zone may only obtain
    the rows(B) × cols(u_i) slice through :meth:`view`. Requesting an
    unregistered zone raises; the view object exposes no other reads.
    """

    def __init__(
        self,
        sensitivities: JacobianSensitivities,
        topology: BoundaryTopology,
        zone_specs: Dict[int, ZoneInputSpec],
    ) -> None:
        if not zone_specs:
            raise ValueError("zone_specs must not be empty")
        for z, spec in zone_specs.items():
            if z not in topology.zone_ids:
                raise ValueError(
                    f"zone_specs contains unknown zone {z}; topology "
                    f"knows {topology.zone_ids}"
                )
            if spec.zone_id != z:
                raise ValueError(
                    f"zone_specs[{z}].zone_id == {spec.zone_id}; must match"
                )
        self._sens = sensitivities
        self._topo = topology
        self._specs = dict(zone_specs)
        self._cache: Dict[int, NDArray[np.float64]] = {}

        # Split the registry into buses with a V state (PQ) and pinned
        # buses (slack / PV): the latter get exactly-zero rows.
        net = self._sens.net
        self._b_pq: List[int] = []
        self.pinned_boundary_buses: List[int] = []
        for b in topology.registry:
            _, v_idx = get_jacobian_indices(net, b)
            if v_idx is None:
                self.pinned_boundary_buses.append(b)
            else:
                self._b_pq.append(b)
        if not self._b_pq:
            raise ValueError(
                "No boundary bus has a voltage state in the Jacobian — "
                "H_{b,i} would be identically zero."
            )

    # ------------------------------------------------------------------
    #  Access-restricted API
    # ------------------------------------------------------------------

    def view(self, zone_id: int) -> ZoneBoundaryView:
        """Zone-bound handle (the only object handed to a zone controller)."""
        self._require_zone(zone_id)
        return ZoneBoundaryView(self, zone_id)

    def h_b(self, zone_id: int) -> NDArray[np.float64]:
        """H_{b,i} for a registered zone, rows in registry order.

        Raises
        ------
        PermissionError
            If the zone is not registered (out-of-scope access).
        """
        self._require_zone(zone_id)
        if zone_id not in self._cache:
            self._cache[zone_id] = self._assemble(zone_id)
        return self._cache[zone_id].copy()

    def invalidate_cache(self) -> None:
        """Drop cached H_{b,i} (call after a Jacobian refresh)."""
        self._cache.clear()

    def _require_zone(self, zone_id: int) -> None:
        if zone_id not in self._specs:
            raise PermissionError(
                f"zone {zone_id} is not registered with this provider "
                f"(registered: {sorted(self._specs)}); out-of-scope "
                "sensitivity access is forbidden (spec §3.9)."
            )

    # ------------------------------------------------------------------
    #  Assembly (mirrors TSOController._build_sensitivity_matrix's
    #  V-row conventions, with observation buses = the boundary registry)
    # ------------------------------------------------------------------

    def _assemble(self, zone_id: int) -> NDArray[np.float64]:
        spec = self._specs[zone_id]
        n_b = len(self._topo.registry)
        H_b = np.zeros((n_b, spec.n_columns), dtype=np.float64)
        if spec.n_columns == 0:
            raise ValueError(
                f"zone {zone_id}: ZoneInputSpec has no columns — a zone "
                "without inputs cannot participate in BME."
            )

        col = 0
        col = self._fill_q_injection_columns(
            H_b, col, list(spec.der_bus_indices), negate=False,
            kind="DER", zone_id=zone_id,
        )
        pcc_hv_buses = self._pcc_hv_buses(spec)
        col = self._fill_q_injection_columns(
            H_b, col, pcc_hv_buses, negate=True,
            kind="Q_PCC_set (load convention)", zone_id=zone_id,
        )
        col = self._fill_vgen_columns(H_b, col, spec)
        col = self._fill_oltc_columns(H_b, col, spec)
        col = self._fill_shunt_columns(H_b, col, spec)
        assert col == spec.n_columns  # internal invariant
        return H_b

    def _row_of(self, bus: int) -> int:
        return self._topo.registry_pos[bus]

    def _fill_q_injection_columns(
        self,
        H_b: NDArray[np.float64],
        col0: int,
        inj_buses: List[int],
        *,
        negate: bool,
        kind: str,
        zone_id: int,
    ) -> int:
        """∂V_b/∂Q columns for Q-injection actuators (DER / PCC-set)."""
        if not inj_buses:
            return col0
        mat, obs_map, inj_map = self._sens.compute_dV_dQ_der(
            der_bus_indices=inj_buses,
            observation_bus_indices=self._b_pq,
        )
        if list(inj_map) != [int(b) for b in inj_buses]:
            missing = [b for b in inj_buses if b not in inj_map]
            raise ValueError(
                f"zone {zone_id}: {kind} buses {missing} have no voltage "
                "state in the Jacobian (PV/slack bus?) — cannot build "
                "H_{b,i} columns."
            )
        if list(obs_map) != self._b_pq:
            raise ValueError(
                f"zone {zone_id}: boundary observation mapping mismatch "
                f"({obs_map} != {self._b_pq})"
            )
        sign = -1.0 if negate else 1.0
        for j in range(len(inj_buses)):
            for i, b in enumerate(self._b_pq):
                H_b[self._row_of(b), col0 + j] = sign * mat[i, j]
        return col0 + len(inj_buses)

    def _pcc_hv_buses(self, spec: ZoneInputSpec) -> List[int]:
        """HV port bus per PCC coupler, preferring the 3W table (mirrors
        the controller's pcc_in_trafo3w / pcc_in_trafo logic)."""
        if not spec.pcc_trafo_indices:
            return []
        net = self._sens.net
        in_3w = (
            hasattr(net, "trafo3w")
            and len(net.trafo3w)
            and all(t in net.trafo3w.index for t in spec.pcc_trafo_indices)
        )
        in_2w = (
            not in_3w
            and len(net.trafo)
            and all(t in net.trafo.index for t in spec.pcc_trafo_indices)
        )
        if in_3w:
            return [
                int(net.trafo3w.at[t, "hv_bus"])
                for t in spec.pcc_trafo_indices
            ]
        if in_2w:
            return [
                int(net.trafo.at[t, "hv_bus"])
                for t in spec.pcc_trafo_indices
            ]
        raise ValueError(
            f"zone {spec.zone_id}: PCC trafo indices "
            f"{list(spec.pcc_trafo_indices)} not found consistently in "
            "net.trafo3w or net.trafo."
        )

    def _fill_vgen_columns(
        self, H_b: NDArray[np.float64], col0: int, spec: ZoneInputSpec
    ) -> int:
        if not spec.gen_indices:
            return col0
        net = self._sens.net
        term_buses = []
        for g in spec.gen_indices:
            if g not in net.gen.index:
                raise ValueError(
                    f"zone {spec.zone_id}: gen index {g} not in net.gen"
                )
            term_buses.append(int(net.gen.at[g, "bus"]))
        mat, obs_map, gen_map = self._sens.compute_dV_dVgen_matrix(
            gen_bus_indices_pp=term_buses,
            observation_bus_indices=self._b_pq,
        )
        if list(gen_map) != term_buses:
            missing = [b for b in term_buses if b not in gen_map]
            raise ValueError(
                f"zone {spec.zone_id}: V_gen terminal buses {missing} "
                "could not be resolved in the Jacobian."
            )
        if list(obs_map) != self._b_pq:
            raise ValueError(
                f"zone {spec.zone_id}: boundary observation mapping "
                f"mismatch in V_gen columns ({obs_map} != {self._b_pq})"
            )
        for j in range(len(term_buses)):
            for i, b in enumerate(self._b_pq):
                H_b[self._row_of(b), col0 + j] = mat[i, j]
        return col0 + len(term_buses)

    def _fill_oltc_columns(
        self, H_b: NDArray[np.float64], col0: int, spec: ZoneInputSpec
    ) -> int:
        if not spec.oltc_trafo_indices:
            return col0
        mat, obs_map, trafo_map = self._sens.compute_dV_ds_2w_matrix(
            trafo_indices=list(spec.oltc_trafo_indices),
            observation_bus_indices=self._b_pq,
        )
        # compute_dV_ds_2w_matrix silently skips failing trafos —
        # restore fail-fast by requiring the full mapping back.
        if list(trafo_map) != [int(t) for t in spec.oltc_trafo_indices]:
            missing = [
                t for t in spec.oltc_trafo_indices if t not in trafo_map
            ]
            raise ValueError(
                f"zone {spec.zone_id}: OLTC trafo(s) {missing} yielded no "
                "tap sensitivity column (out of service / slack-adjacent?)."
            )
        if list(obs_map) != self._b_pq:
            raise ValueError(
                f"zone {spec.zone_id}: boundary observation mapping "
                f"mismatch in OLTC columns ({obs_map} != {self._b_pq})"
            )
        for j in range(len(trafo_map)):
            for i, b in enumerate(self._b_pq):
                H_b[self._row_of(b), col0 + j] = mat[i, j]
        return col0 + len(trafo_map)

    def _fill_shunt_columns(
        self, H_b: NDArray[np.float64], col0: int, spec: ZoneInputSpec
    ) -> int:
        for j, (bus, q_step) in enumerate(
            zip(spec.shunt_bus_indices, spec.shunt_q_steps_mvar)
        ):
            colv, obs_map = self._sens.compute_dV_dQ_shunt(
                shunt_bus_idx=int(bus),
                observation_bus_indices=self._b_pq,
                q_step_mvar=float(q_step),
            )
            if list(obs_map) != self._b_pq:
                raise ValueError(
                    f"zone {spec.zone_id}: boundary observation mapping "
                    f"mismatch in shunt column at bus {bus}"
                )
            for i, b in enumerate(self._b_pq):
                H_b[self._row_of(b), col0 + j] = colv[i]
        return col0 + len(spec.shunt_bus_indices)
