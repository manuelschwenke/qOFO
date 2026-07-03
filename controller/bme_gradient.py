"""
controller/bme_gradient.py
==========================
Convention-A BME gradient assembly for one TSO zone — spec §3.5 (see
``docs/BME_STATUS.md`` §0.2 revision note).

Symbol map (code ↔ spec)
------------------------
* ``g_own()``      ↔ g_i^own = ∂Φ_i/∂u_i |_{v_b fixed} (§3.5 Convention A),
                     assembled over the zone's input columns
                     ``[Q_DER (bus-level) | Q_PCC_set | V_gen | s_OLTC | s_shunt]``
                     (:class:`sensitivity.boundary_sensitivity.ZoneInputSpec`,
                     mirroring the TSO controller's u ordering; per-DER
                     expansion stays controller-side via the existing
                     ``_expand_H_to_der_level`` E matrix).
* ``mu()``         ↔ μ_i (§3.4, D7 REVISED 2026-07-02) — the zone's OWN
                     marginal over the COMPLEX boundary coordinates,
                     stacked ``[dΦ_i/dVm_b | dΦ_i/dθ_b]`` ∈ R^{2|B|};
                     published on the bus AND entering the price term
                     locally, undelayed and unfiltered (Convention A:
                     J = ALL zones including i itself). The Phase 4
                     identity test demonstrated that the magnitude-only
                     channel misses the boundary-angle term of the loss
                     objective — Manuel's pre-authorised fallback
                     (complex boundary quantities) applies.
* ``g_bme(...)``   ↔ g_i^bme = g_i^own + H_{b,i}ᵀ · μ_total, with
                     μ_total = μ_i + Σ_{j ≠ i} μ_j^filt(k − d) supplied by
                     the caller (`MarginalReceiver.mu_neighbour_sum` plus
                     the local self term), all in stacked coordinates.
* ``h_b_stacked``  ↔ H_{b,i} = ∂(Vm_b, θ_b)/∂u_i, served EXCLUSIVELY
                     through the access-restricted
                     :class:`ZoneBoundaryView` (§3.9 — the single
                     supra-local object of the scheme; complex boundary
                     coordinates do not widen the informational scope:
                     still the zone's own columns at jointly observable
                     boundary buses).

Identity pinned by the Phase 4 hard-gate test (stacked coordinates):
``dΦ/du_i = ∂Φ_i/∂u_i|_{v_b fixed} + H_{b,i}ᵀ · Σ_{all j} μ_j``.

Fail-fast: zone mismatches between spec / view / gradients raise; a
μ_total of the wrong length raises; every u column must resolve through
the ownership-enforced Phase 2 primitives (foreign actuators raise
there).

Author: Manuel Schwenke / Claude Code
Date: 2026-07-02 (BME Phase 4)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from controller.common_objective import ZoneGradients
from sensitivity.boundary_sensitivity import (
    ZoneBoundaryView,
    ZoneInputSpec,
    actuator_active,
)


def pcc_hv_buses(net, spec: ZoneInputSpec) -> List[int]:
    """HV port bus per PCC coupler, preferring the 3W table (mirrors
    ``RestrictedSensitivityProvider._pcc_hv_buses`` and the controller's
    pcc_in_trafo3w / pcc_in_trafo logic — keep the three in lockstep)."""
    if not spec.pcc_trafo_indices:
        return []
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
            int(net.trafo3w.at[t, "hv_bus"]) for t in spec.pcc_trafo_indices
        ]
    if in_2w:
        return [
            int(net.trafo.at[t, "hv_bus"]) for t in spec.pcc_trafo_indices
        ]
    raise ValueError(
        f"zone {spec.zone_id}: PCC trafo indices "
        f"{list(spec.pcc_trafo_indices)} not found consistently in "
        "net.trafo3w or net.trafo."
    )


class BMEGradientAssembler:
    """Assembles g_i^own and g_i^bme for one zone (§3.5 Convention A).

    Rebuild together with :class:`ZoneGradients` whenever the cached
    Jacobian operating point is refreshed — all three collaborators
    (spec, boundary view, gradients) must describe the same zone.
    """

    def __init__(
        self,
        spec: ZoneInputSpec,
        gradients: ZoneGradients,
        view: ZoneBoundaryView,
    ) -> None:
        zone = gradients.zone_id
        if spec.zone_id != zone or view.zone_id != zone:
            raise ValueError(
                f"zone mismatch: spec={spec.zone_id}, "
                f"gradients={zone}, view={view.zone_id}"
            )
        if spec.n_columns == 0:
            raise ValueError(
                f"zone {zone}: ZoneInputSpec has no columns — a zone "
                "without inputs cannot assemble a gradient."
            )
        self.zone_id = zone
        self._spec = spec
        self._grads = gradients
        self._view = view

    # ------------------------------------------------------------------

    def mu(self) -> NDArray[np.float64]:
        """μ_zone in stacked complex-boundary coordinates
        ``[dΦ/dVm_b | dΦ/dθ_b]`` (published AND used locally)."""
        return self._grads.mu_stacked()

    def g_own(self) -> NDArray[np.float64]:
        """g_i^own over the ZoneInputSpec columns (bus-level DER).

        Out-of-service / isolated actuators (e.g. a tripped machine and
        its trafo) keep their column but contribute exactly zero —
        mirroring the controller's OOS column masking and the zero
        columns of ``h_b_stacked`` (see
        :func:`sensitivity.boundary_sensitivity.actuator_active`)."""
        spec = self._spec
        net = self._grads.net
        cols: List[float] = []
        for bus in spec.der_bus_indices:
            cols.append(
                self._grads.d_q_injection(int(bus))
                if actuator_active(net, "der", int(bus)) else 0.0
            )
        for hv_bus in pcc_hv_buses(net, spec):
            cols.append(
                self._grads.d_pcc_set(int(hv_bus))
                if actuator_active(net, "pcc", int(hv_bus)) else 0.0
            )
        for gen_idx in spec.gen_indices:
            cols.append(
                self._grads.d_vgen(int(gen_idx))
                if actuator_active(net, "vgen", int(gen_idx)) else 0.0
            )
        for trafo_idx in spec.oltc_trafo_indices:
            cols.append(
                self._grads.d_tap_2w(int(trafo_idx))
                if actuator_active(net, "oltc", int(trafo_idx)) else 0.0
            )
        for bus, q_step in zip(
            spec.shunt_bus_indices, spec.shunt_q_steps_mvar
        ):
            cols.append(
                self._grads.d_shunt(int(bus), float(q_step))
                if actuator_active(net, "shunt", int(bus)) else 0.0
            )
        out = np.asarray(cols, dtype=np.float64)
        assert out.shape == (spec.n_columns,)  # internal invariant
        return out

    def g_bme(
        self, mu_total: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """g_i^bme = g_i^own + H_{b,i}ᵀ · μ_total (§3.5), in STACKED
        complex-boundary coordinates (length 2|B|).

        ``mu_total`` must already contain the LOCAL self-marginal μ_i
        (undelayed, unfiltered) plus the filtered, delayed neighbour sum
        from the :class:`core.coordination_bus.MarginalReceiver` —
        Convention A sums ALL zones including i itself.
        """
        h_b = self._view.h_b_stacked()
        mu_total = np.asarray(mu_total, dtype=np.float64)
        if mu_total.shape != (h_b.shape[0],):
            raise ValueError(
                f"zone {self.zone_id}: mu_total has shape "
                f"{mu_total.shape}; the stacked boundary coordinate "
                f"length is {h_b.shape[0]}."
            )
        return self.g_own() + h_b.T @ mu_total
