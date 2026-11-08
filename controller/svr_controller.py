"""
controller/svr_controller.py
============================

Classical pilot-node secondary voltage regulation (SVR) as a reference
controller -- thesis variant ``S1``, ``\\cref{ch:setup:references:svr}``.

Structure follows Corsi's regional hierarchy (Regional Voltage Regulator on the
pilot node, plant-level Reactive Power Regulators beneath it) in the compact
central-controller form given by Milano: no coupling compensation, no discrete
actuators, static participation, and no information exchange between areas.

Both feedback layers may use PI action. By default, ``k_p_rvr`` acts on the
pilot-voltage error at the regional 180 s sample and ``k_p_rpr`` acts on a
machine's reactive-power error at the 20 s sample. ``rvr_p_every_step`` can
instead recompute only the outer algebraic P path on the 20 s grid while its
integrator remains on the regional grid. The integral states are stored
separately from their algebraic proportional outputs, so a previous P term is
never integrated again. Setting either gain to zero reproduces the corresponding
I-only update.

The optional ``rpr_voltage_priority`` condition blocks a machine's RPR move if
its cached pilot sensitivity predicts that the move would increase the current
pilot-voltage error. This prevents Q alignment from withdrawing voltage support
while the regional reactive level is held between dispatches. It is an explicit
extension of the unconditional classical RPR, not a new outer control law.
Leave it disabled to reproduce that classical update. Service-status handling
is always active: unavailable generators keep their last AVR reference.

Why this file emits an OFO-shaped ``u``
---------------------------------------
The controller returns its dispatch in the *same* per-zone actuator layout the
TSO-OFO uses,

    u = [ Q_DER | Q_PCC_set | V_gen_set | s_OLTC | s_shunt ]

(see :func:`core.plant.writes_from_zone_tso`), so the runner applies, records
and scores it through the identical path.  That is what makes the ``S1 -> O1``
comparison a comparison of control laws rather than of harnesses.

Consequences of that layout, each of them deliberate:

* ``Q_PCC_set`` is held at zero -- the classical scheme dispatches no interface
  setpoint, and has no port through which a subordinate capability interval
  could enter.  Interface metrics are therefore not reported for ``S1``.
* ``s_OLTC`` / ``s_shunt`` are echoed back unchanged from the current plant
  state.  The discrete devices belong to the local rule-based logic in this
  variant, and writing the measured position back is a no-op that keeps the
  block-offset contract with ``writes_from_zone_tso`` intact.
* Synchronous machines are driven through ``V_gen_set`` by the RPR, which is the
  classical actuation path.  Converter-interfaced TS-DER are commanded in Q
  directly (the RPR is assumed converged for a converter), which is the
  documented extension of the alignment law to non-synchronous devices.

Gain normalisation
------------------
``k_i`` is not a free parameter.  It follows from ``T_RVR`` and the pilot
sensitivity ``s_p,a`` so that the nominal per-dispatch loop gain is
``T_TS / T_RVR`` independently of the area -- the same nominal isolated-loop
screen that fixes the OFO weights.  ``s_p,a`` is estimated ONCE at
initialisation by perturbation on a *copy* of the plant, then frozen: classical
SVR computes its participation structure offline and does not refresh it.

OFO form of the two loops (variant O0)
--------------------------------------
Added 2026-10-06 for Case 1 of the thesis (``ch:case_svr``).  With
``rvr_law='ofo'`` the regional update is the OFO step (``eq:ofo_qp``) of a
controller with ONE input, the reactive level q_a, and ONE tracked output, the
pilot voltage::

    sigma = argmin_w  g_w w^2 - 2 w_p s_p e_p w + g_z ||z||^2
            s.t.  q_min <= q_a + w <= q_max
                  v_min - z <= v_j + (dv_j/dq_a) w <= v_max + z   (monitored buses)

with ``g_w = w_p s_p^2 T_RVR / T_TS`` (the match condition of the thesis).
Without a violated output constraint its solution is the integral RVR
increment ``T_TS e_p / (T_RVR s_p)``, saturated exactly as the RVR saturates
its level.  ``rpr_law='ofo'`` does the same per machine: input v_set, output
Q_i, ``g_w = w_Q h_i^2 T_RPR / dt``, the machine capability as soft output
constraint.  Both require the integral form (``k_p = 0``) and no
voltage-priority guard: the OFO step has no proportional path, and the guard is
not part of it.  The voltage limits and slack weights are the TS-OFO's own, so
the OFO form differs from the TS-OFO only in objective and structure.

Author: added 2026-09-01 for the S1 reference variant.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandapower as pp
from numpy.typing import NDArray

__all__ = ["SVRZoneConfig", "SVRZoneController", "SVRCoordinator", "SVROutput"]


# ---------------------------------------------------------------------------
#  Output (duck-types the TSO-OFO's per-zone output)
# ---------------------------------------------------------------------------


@dataclass
class SVROutput:
    """Per-zone dispatch, shaped like ``TSOController``'s output.

    ``objective_value`` carries the squared pilot error so the runner's existing
    per-zone objective column stays meaningful; it is NOT an optimisation
    objective, because this controller does not optimise.
    """

    u_new: NDArray[np.float64]
    objective_value: float
    solver_status: str
    solve_time_s: float
    # -- SVR-specific diagnostics ------------------------------------------
    q_level: float = 0.0
    pilot_v_pu: float = float("nan")
    pilot_v_ref_pu: float = float("nan")
    pilot_error_pu: float = 0.0
    saturated: bool = False
    q_ref_mvar: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    gen_available: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    rpr_blocked: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    rvr_p_term_level: float = 0.0
    rpr_p_term_pu: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    # -- OFO form (variant O0): active sets of the one-input steps -----------
    ofo_rvr_n_active: int = 0
    ofo_rvr_box_active: bool = False
    #: Per machine: its OUTPUT constraint (capability) binds.  Only this can
    #: make the OFO form depart from the integral law.
    ofo_rpr_active: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    #: Per machine: its AVR-reference bound binds -- the same saturation the
    #: integral law applies, so it never causes a departure.
    ofo_rpr_box_active: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))


# ---------------------------------------------------------------------------
#  One-input OFO step (variant O0)
# ---------------------------------------------------------------------------


def _ofo_scalar_step(
    w0: float,
    lo: float,
    hi: float,
    g: float,
    rows: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    g_z: float,
) -> tuple:
    """Exact OFO step of a controller with one input.

    Minimises ``g (w - w0)^2 + g_z * sum(max(0, a - r w)^2 + max(0, r w - b)^2)``
    over ``lo <= w <= hi``.  This is ``eq:ofo_qp`` with the slack variables
    eliminated -- for a quadratic slack penalty the optimal slack is the
    constraint violation itself -- and with ``w0`` the unconstrained step.

    The objective is convex and piecewise quadratic, so its derivative is
    monotone and bisection on it is exact to machine precision.  When the
    box-clipped unconstrained step violates no output constraint it is returned
    unchanged; this keeps the equivalence with the integral law exact rather
    than approximate.

    Returns ``(w, n_out_active, box_active)``.
    """
    rows = np.asarray(rows, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    def violations(w: float):
        y = rows * w
        return np.maximum(0.0, a - y), np.maximum(0.0, y - b)

    w_c = float(min(max(w0, lo), hi))
    under, over = violations(w_c)
    if rows.size == 0 or not (np.any(under > 0.0) or np.any(over > 0.0)):
        return w_c, 0, bool(w_c != w0)

    def slope(w: float) -> float:
        u_, o_ = violations(w)
        return 2.0 * g * (w - w0) + 2.0 * g_z * float(np.sum(rows * (o_ - u_)))

    if slope(lo) >= 0.0:
        w = float(lo)
    elif slope(hi) <= 0.0:
        w = float(hi)
    else:
        left, right = float(lo), float(hi)
        for _ in range(200):
            mid = 0.5 * (left + right)
            if slope(mid) > 0.0:
                right = mid
            else:
                left = mid
            if right - left <= 1e-15 * max(1.0, abs(left), abs(right)):
                break
        w = 0.5 * (left + right)
    under, over = violations(w)
    n_active = int(np.sum((under > 1e-12) | (over > 1e-12)))
    box_active = bool(w <= lo + 1e-15 or w >= hi - 1e-15)
    return w, n_active, box_active


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


@dataclass
class SVRZoneConfig:
    """Tuning of one regional regulator.  See the module docstring on gains."""

    pilot_bus: int
    v_ref_pu: float = 1.03
    t_rvr_s: float = 300.0
    t_rpr_s: float = 60.0
    k_p_rvr: float = 0.0
    k_p_rpr: float = 0.0
    rvr_p_every_step: bool = False
    deadband_pu: float = 0.01
    rpr_voltage_priority: bool = False
    q_level_min: float = -1.0
    q_level_max: float = 1.0
    v_set_min_pu: float = 0.95
    v_set_max_pu: float = 1.05
    n_rpr_substeps: int = 1
    calib_dv_pu: float = 5e-3
    calib_dq_mvar: float = 5.0
    # -- OFO form of the two loops (variant O0, 2026-10-06) ------------------
    #: "pi" = classical regional regulator, "ofo" = one-input OFO step.
    rvr_law: str = "pi"
    #: "pi" = classical plant regulator, "ofo" = one-input OFO step per machine.
    rpr_law: str = "pi"
    #: Pilot tracking weight; the runner passes the TS-OFO's voltage weight g_v.
    ofo_w_p: float = 1.0
    #: Machine-Q tracking weight of the OFO plant step.
    ofo_w_q: float = 1.0
    #: Voltage limits of the monitored buses; the runner passes the TS-OFO's.
    ofo_v_min_pu: float = 0.90
    ofo_v_max_pu: float = 1.10
    #: Slack weights; the runner passes the TS-OFO's g_z_voltage / g_z_q_gen.
    ofo_g_z_v: float = 1e9
    ofo_g_z_q: float = 1e2
    # -- access to the child STS (variant S2, 2026-10-27) --------------------
    #: The couplers of the area's child STS join the alignment law
    #: (``eq:svr-alignment-sts``); the STS-OFOs track the PCC block of ``u``.
    sts_access: bool = False


# ---------------------------------------------------------------------------
#  Zone controller
# ---------------------------------------------------------------------------


@dataclass
class _ZoneState:
    q_level: float = 0.0
    integrator: float = 0.0
    rvr_p_term: float = 0.0
    error: float = 0.0
    saturated: bool = False


class SVRZoneController:
    """One RVR + its RPRs, for one TSO control area."""

    def __init__(
        self,
        zone_def,
        actuator_bounds,
        cfg: SVRZoneConfig,
        *,
        non_dispatchable_gens: Optional[Sequence[int]] = None,
        verbose: int = 0,
    ) -> None:
        self.zone_def = zone_def
        self.bounds = actuator_bounds
        self.cfg = cfg
        self.verbose = verbose
        if not np.isfinite(cfg.k_p_rvr) or cfg.k_p_rvr < 0.0:
            raise ValueError("k_p_rvr must be finite and non-negative")
        if not np.isfinite(cfg.k_p_rpr) or cfg.k_p_rpr < 0.0:
            raise ValueError("k_p_rpr must be finite and non-negative")
        for _law in ("rvr_law", "rpr_law"):
            if getattr(cfg, _law) not in ("pi", "ofo"):
                raise ValueError(
                    f"{_law} must be 'pi' or 'ofo', got {getattr(cfg, _law)!r}")
        if cfg.rvr_law == "ofo" and cfg.k_p_rvr > 0.0:
            raise ValueError("rvr_law='ofo' requires k_p_rvr == 0: the OFO step "
                             "has no proportional path")
        if cfg.rpr_law == "ofo" and cfg.k_p_rpr > 0.0:
            raise ValueError("rpr_law='ofo' requires k_p_rpr == 0: the OFO step "
                             "has no proportional path")
        if cfg.rpr_law == "ofo" and cfg.rpr_voltage_priority:
            raise ValueError("rpr_law='ofo' excludes rpr_voltage_priority: the "
                             "guard is not part of the OFO step")
        self.state = _ZoneState()
        self._initialised = False

        self.gen_indices: List[int] = [int(g) for g in zone_def.gen_indices]

        # The slack / network-equivalent machine stays in ``zone_def.gen_indices``
        # (it is observed) but its AVR reference is NOT an actuator -- the runner
        # withholds it from the OFO through ``non_dispatchable_gen_indices``, and
        # this reference scheme must withhold it too or the comparison is not
        # controlled.  Driving a slack machine's voltage reference toward a
        # reactive target is meaningless in any case: the slack absorbs whatever
        # the rest of the network leaves over, so the RPR would be integrating
        # against a bus that never shows the error it is trying to correct.
        _nd = {int(g) for g in (non_dispatchable_gens or ())}
        self.gen_mask: NDArray[np.float64] = np.array(
            [0.0 if g in _nd else 1.0 for g in self.gen_indices], dtype=float
        )
        self.n_gen_withheld = int(np.sum(self.gen_mask == 0.0))
        # The RPR integrator is not the previous AVR output when k_p_rpr > 0.
        # NaN marks a machine that needs bumpless initialisation/restoration.
        self._rpr_integrator = np.full(len(self.gen_indices), np.nan, dtype=float)
        self._gen_was_available = np.zeros(len(self.gen_indices), dtype=bool)
        self.der_indices: List[int] = [int(s) for s in zone_def.tso_der_indices]
        self.n_pcc = len(zone_def.pcc_trafo_indices)
        self.n_oltc = len(zone_def.oltc_trafo_indices)
        self.n_shunt = len(zone_def.shunt_bus_indices)

        #: dV_pilot / dQ_i for every reactive device of the area [pu / Mvar].
        self.dv_pilot_dq: Optional[NDArray[np.float64]] = None
        #: dQ_i / dV_set,i for the synchronous machines [Mvar / pu].
        self.dq_dvset: Optional[NDArray[np.float64]] = None
        #: Aggregate pilot sensitivity s_p,a [pu per unit reactive level].
        self.pilot_sensitivity: float = float("nan")
        #: No reactive resource in service at calibration (set there).
        self.no_resources: bool = False
        #: Monitored buses of the zone (the TS-OFO's ``v_bus_indices``).  Only
        #: the OFO form of the RVR uses them, as soft voltage constraints.
        self.v_bus_indices: List[int] = [
            int(b) for b in (getattr(zone_def, "v_bus_indices", None) or [])
        ]
        #: dV_bus / dQ_i for the monitored buses [pu / Mvar], shape
        #: (n_vbus, n_gen + n_der).  Calibrated with s_p,a and frozen with it.
        self.dv_bus_dq: Optional[NDArray[np.float64]] = None

        # -- S2: the couplers of the area's child STS -------------------------
        #: Coupler (trafo3w) indices in the order of the PCC block of ``u``.
        self.pcc_trafo_indices: List[int] = [
            int(t) for t in zone_def.pcc_trafo_indices]
        #: dV_pilot / dQ_inj at each coupler's TS-side bus [pu / Mvar].
        self.dv_pilot_dq_coupler: Optional[NDArray[np.float64]] = None
        #: Latest CAIR message per coupler: (dq_min, dq_max, q_neutral) [Mvar].
        self._cap: Dict[int, Tuple[float, float, float]] = {}
        #: The coupler term of s_p,a is added once, with the headroom of the
        #: first CAIR message, and then held (participation is fixed).
        self.pilot_sensitivity_couplers: float = 0.0
        self._coupler_term_added = False
        #: Interface references [Mvar, load convention], held between updates.
        self.q_ref_pcc: NDArray[np.float64] = np.full(self.n_pcc, np.nan)
        #: How often a neutral point outside its interval had to be clipped.
        self.n_neutral_clipped = 0

    # -- calibration -------------------------------------------------------

    def calibrate(self, net: pp.pandapowerNet, runpp_kwargs: Dict) -> None:
        """Estimate ``s_p,a`` and ``dQ/dV_set`` by perturbation on a net copy.

        Runs once, at initialisation, on ``copy.deepcopy(net)`` so the live
        plant is never disturbed.  The result is then held fixed for the whole
        campaign, which is what classical SVR does with participation.
        """
        probe = copy.deepcopy(net)
        p = int(self.cfg.pilot_bus)
        n_gen, n_der = len(self.gen_indices), len(self.der_indices)

        def _solve() -> None:
            pp.runpp(probe, **runpp_kwargs)

        _solve()

        dv_pilot_dq = np.zeros(n_gen + n_der, dtype=float)
        dq_dvset = np.zeros(n_gen, dtype=float)
        # The same perturbations also give the monitored-bus columns used by the
        # OFO form of the RVR; reading more result rows costs no extra solve.
        vb = list(self.v_bus_indices)
        dv_bus_dq = np.zeros((len(vb), n_gen + n_der), dtype=float)

        def _vbus() -> NDArray[np.float64]:
            if not vb:
                return np.zeros(0)
            return probe.res_bus.loc[vb, "vm_pu"].to_numpy(dtype=float)

        # -- synchronous machines: perturb the AVR setpoint ----------------
        # Withheld machines keep dq_dvset = 0, which makes step() pass their
        # current setpoint straight through untouched.
        active_gen = self._active_generators(probe)
        for k, g in enumerate(self.gen_indices):
            if not active_gen[k]:
                continue
            v0 = float(probe.gen.at[g, "vm_pu"])
            readings = {}
            for sign in (+1.0, -1.0):
                probe.gen.at[g, "vm_pu"] = v0 + sign * self.cfg.calib_dv_pu
                _solve()
                readings[sign] = (
                    float(probe.res_gen.at[g, "q_mvar"]),
                    float(probe.res_bus.at[p, "vm_pu"]),
                    _vbus(),
                )
            probe.gen.at[g, "vm_pu"] = v0
            _solve()

            dq = (readings[+1.0][0] - readings[-1.0][0]) / (2 * self.cfg.calib_dv_pu)
            dvp = (readings[+1.0][1] - readings[-1.0][1]) / (2 * self.cfg.calib_dv_pu)
            dvb = (readings[+1.0][2] - readings[-1.0][2]) / (2 * self.cfg.calib_dv_pu)
            dq_dvset[k] = dq
            # dV_p/dQ_i = (dV_p/dV_set,i) / (dQ_i/dV_set,i)
            dv_pilot_dq[k] = dvp / dq if abs(dq) > 1e-9 else 0.0
            if vb:
                dv_bus_dq[:, k] = dvb / dq if abs(dq) > 1e-9 else 0.0

        # -- converter DER: perturb the reactive command ------------------
        # FIX 2026-11-07: perturb the DER through the plant's command path.
        # A DER with a plant-side Q(V) loop (q_mode "qv", every TS-DER) is
        # commanded through ``q_set_mvar`` at a V_ref anchor reset to the
        # measured voltage (core.plant.write_der_q_set).  Its loop rewrites
        # ``q_mvar`` inside the run_control solve, so perturbing ``q_mvar``
        # alone was undone and gave DER columns ~60x too small (step campaign
        # 2026-11-06: S1 integral gain ~38 instead of 0.6 per update in area 2
        # at OP-R, where no machine is committed).  Here the command and the
        # output move together, with the anchor at the base-point voltage:
        # the base operating point is unchanged, and a DER without a loop
        # sees the same ``q_mvar`` perturbation as before.
        from core.measurement_noise import measured_der_voltage
        for col, fill in (("q_set_mvar", 0.0), ("qv_vref_anchor_pu", float("nan"))):
            if col not in probe.sgen.columns:
                probe.sgen[col] = fill
        for k, s in enumerate(self.der_indices):
            q0 = float(probe.sgen.at[s, "q_mvar"])
            qset0 = probe.sgen.at[s, "q_set_mvar"]
            anchor0 = probe.sgen.at[s, "qv_vref_anchor_pu"]
            anchor = measured_der_voltage(probe, s, int(probe.sgen.at[s, "bus"]))
            readings = {}
            for sign in (+1.0, -1.0):
                q_cmd = q0 + sign * self.cfg.calib_dq_mvar
                probe.sgen.at[s, "q_mvar"] = q_cmd
                probe.sgen.at[s, "q_set_mvar"] = q_cmd
                probe.sgen.at[s, "qv_vref_anchor_pu"] = anchor
                _solve()
                readings[sign] = (float(probe.res_bus.at[p, "vm_pu"]), _vbus())
            probe.sgen.at[s, "q_mvar"] = q0
            probe.sgen.at[s, "q_set_mvar"] = qset0
            probe.sgen.at[s, "qv_vref_anchor_pu"] = anchor0
            _solve()
            dv_pilot_dq[n_gen + k] = (
                readings[+1.0][0] - readings[-1.0][0]
            ) / (2 * self.cfg.calib_dq_mvar)
            if vb:
                dv_bus_dq[:, n_gen + k] = (
                    readings[+1.0][1] - readings[-1.0][1]
                ) / (2 * self.cfg.calib_dq_mvar)

        # -- couplers of the child STS (S2): a reactive injection at the
        # coupler's TS-side bus, through a temporary generator on the copy.
        if self.cfg.sts_access and self.n_pcc:
            dv_c = np.zeros(self.n_pcc, dtype=float)
            for j, t in enumerate(self.pcc_trafo_indices):
                hv_bus = int(probe.trafo3w.at[t, "hv_bus"])
                sid = pp.create_sgen(probe, bus=hv_bus, p_mw=0.0, q_mvar=0.0,
                                     name="_svr_coupler_calibration")
                readings = {}
                for sign in (+1.0, -1.0):
                    probe.sgen.at[sid, "q_mvar"] = sign * self.cfg.calib_dq_mvar
                    _solve()
                    readings[sign] = float(probe.res_bus.at[p, "vm_pu"])
                probe.sgen.at[sid, "q_mvar"] = 0.0
                probe.sgen.at[sid, "in_service"] = False
                _solve()
                dv_c[j] = (readings[+1.0] - readings[-1.0]) / (
                    2 * self.cfg.calib_dq_mvar)
            self.dv_pilot_dq_coupler = dv_c

        self.dv_pilot_dq = dv_pilot_dq
        self.dq_dvset = dq_dvset
        self.dv_bus_dq = dv_bus_dq

        q_max = self._headroom(net, upper=True)
        self.pilot_sensitivity = float(np.sum(dv_pilot_dq * q_max))

        if self.verbose >= 1:
            _held = (f", {self.n_gen_withheld} withheld"
                     if self.n_gen_withheld else "")
            print(f"  [SVR z{self.zone_def.zone_id}] pilot bus {p}: "
                  f"s_p={self.pilot_sensitivity:+.5f} pu/level, "
                  f"{n_gen - self.n_gen_withheld} gen + {n_der} DER{_held}")
        # A zone with no reactive resource in service at all (no committed
        # machine, no TS IBG with capability; 2026-11-08: zone 2 of
        # ``rural_700_north_res`` while unit commitment keeps G3 off) has
        # nothing to dispatch: its SVR holds (``step`` skips the RVR while
        # s_p = 0).  With STS access the couplers join s_p at the first
        # capability message (``_add_coupler_term``).  An insensitive pilot
        # in a zone that HAS resources is still a configuration error.
        self.no_resources = float(np.sum(q_max)) < 1e-9
        if self.no_resources:
            print(f"  [SVR z{self.zone_def.zone_id}] no reactive resource in "
                  f"service at calibration; the SVR holds"
                  f"{' until the STS couplers join' if self.cfg.sts_access and self.n_pcc else ''}")
        elif abs(self.pilot_sensitivity) < 1e-9:
            raise ValueError(
                f"zone {self.zone_def.zone_id}: pilot bus {p} is insensitive to "
                "the area's reactive resources -- the pilot bus is a poor choice."
            )

    # -- S2: capability of the child STS ----------------------------------

    def receive_capability(self, message) -> None:
        """Store the CAIR interval and neutral point of this area's couplers.

        The same message the TS-OFO receives; the SVR reads only the couplers
        it parents.  Intervals are changes from the current flow, in load
        convention at the HV port (``core.message.CapabilityMessage``).
        """
        qn = getattr(message, "q_neutral_mvar", None)
        for i, t in enumerate(message.interface_transformer_indices):
            t = int(t)
            if t in self.pcc_trafo_indices:
                self._cap[t] = (
                    float(message.q_min_mvar[i]), float(message.q_max_mvar[i]),
                    float(qn[i]) if qn is not None else float("nan"),
                )

    def _coupler_interval(self, net, t: int) -> Tuple[float, float, float]:
        """Absolute (Q_lo, Q_0, Q_hi) of coupler ``t`` [Mvar, load convention]."""
        if t not in self._cap:
            raise RuntimeError(
                f"zone {self.zone_def.zone_id}: S2 has no capability message "
                f"for coupler {t}")
        dmin, dmax, qn = self._cap[t]
        q_now = float(net.res_trafo3w.at[t, "q_hv_mvar"])
        lo, hi = q_now + dmin, q_now + dmax
        q0 = qn if np.isfinite(qn) else q_now
        if q0 < lo or q0 > hi:
            self.n_neutral_clipped += 1
            q0 = min(max(q0, lo), hi)
        return lo, q0, hi

    def _add_coupler_term(self, net) -> None:
        """Add the couplers to s_p,a once, with the first CAIR headroom.

        Supporting the voltage means lowering the load-convention flow, so the
        headroom in the raising direction is ``Q_0 - Q_lo``.
        """
        term = 0.0
        for j, t in enumerate(self.pcc_trafo_indices):
            lo, q0, _hi = self._coupler_interval(net, t)
            term += float(self.dv_pilot_dq_coupler[j]) * (q0 - lo)
        self.pilot_sensitivity_couplers = term
        self.pilot_sensitivity += term
        self._coupler_term_added = True
        if self.verbose >= 1:
            print(f"  [SVR z{self.zone_def.zone_id}] S2: {self.n_pcc} couplers "
                  f"join the alignment, s_p += {term:+.5f} -> "
                  f"{self.pilot_sensitivity:+.5f} pu/level")

    def coupler_references(self, net, level: float) -> NDArray[np.float64]:
        """``eq:svr-alignment-sts`` in load convention.

        ``level >= 0`` (raise the voltage) moves the reference from the
        neutral point toward ``Q_lo`` (more injection into the TS);
        ``level < 0`` toward ``Q_hi``.
        """
        out = np.empty(self.n_pcc, dtype=float)
        for j, t in enumerate(self.pcc_trafo_indices):
            lo, q0, hi = self._coupler_interval(net, t)
            out[j] = (q0 - level * (q0 - lo) if level >= 0.0
                      else q0 + (-level) * (hi - q0))
        return out

    # -- bounds ------------------------------------------------------------

    def _active_generators(self, net: pp.pandapowerNet) -> NDArray[np.bool_]:
        """Dispatch permission AND measured service status, in generator order.

        Service status is an actuator-availability measurement, not a model
        refresh. A tripped machine retains its last AVR reference and contributes
        no reactive capability until it returns to service. Never mutate the
        static withholding mask: a restored machine must be dispatchable again.
        """
        return np.asarray([
            bool(self.gen_mask[k]) and bool(net.gen.at[g, "in_service"])
            for k, g in enumerate(self.gen_indices)
        ], dtype=bool)

    def _headroom(self, net: pp.pandapowerNet, *, upper: bool) -> NDArray[np.float64]:
        """Per-device reactive headroom [Mvar] at the current operating point.

        Upper bounds are positive, lower bounds are returned as magnitudes, so
        the alignment law can multiply either by ``|q_level|``.
        """
        active_gen = self._active_generators(net)
        # Out-of-service result rows can contain NaN. Supply finite dummy
        # inputs to the capability calculation, then discard those bounds.
        # Multiplication by a zero mask is insufficient: 0 * NaN is still NaN.
        gen_p = np.array(
            [float(net.res_gen.at[g, "p_mw"]) if active_gen[k] else 0.0
             for k, g in enumerate(self.gen_indices)], dtype=float
        ) if self.gen_indices else np.zeros(0)
        gen_v = np.array(
            [float(net.gen.at[g, "vm_pu"]) if active_gen[k] else 1.0
             for k, g in enumerate(self.gen_indices)], dtype=float
        ) if self.gen_indices else np.zeros(0)
        der_p = np.array(
            [float(net.sgen.at[s, "p_mw"]) for s in self.der_indices], dtype=float
        ) if self.der_indices else np.zeros(0)

        g_lo, g_hi = self.bounds.compute_gen_q_bounds(gen_p, gen_v)
        d_lo, d_hi = self.bounds.compute_der_q_bounds(der_p)

        # Withheld and tripped machines contribute no dispatchable capability.
        g_hi = np.where(active_gen, np.asarray(g_hi, float), 0.0)
        g_lo = np.where(active_gen, np.asarray(g_lo, float), 0.0)

        if upper:
            return np.concatenate([g_hi, np.asarray(d_hi, float)])
        return np.abs(np.concatenate([g_lo, np.asarray(d_lo, float)]))

    def _gen_q_limits(self, net: pp.pandapowerNet) -> tuple:
        """Signed capability band ``(Q_min, Q_max)`` [Mvar] of the machines.

        Used only by the OFO form of the plant loop, as its soft output
        constraint.  Unavailable machines get a zero band, as in ``_headroom``.
        """
        active_gen = self._active_generators(net)
        if not self.gen_indices:
            return np.zeros(0), np.zeros(0)
        gen_p = np.array(
            [float(net.res_gen.at[g, "p_mw"]) if active_gen[k] else 0.0
             for k, g in enumerate(self.gen_indices)], dtype=float)
        gen_v = np.array(
            [float(net.gen.at[g, "vm_pu"]) if active_gen[k] else 1.0
             for k, g in enumerate(self.gen_indices)], dtype=float)
        g_lo, g_hi = self.bounds.compute_gen_q_bounds(gen_p, gen_v)
        g_lo = np.where(active_gen, np.asarray(g_lo, float), 0.0)
        g_hi = np.where(active_gen, np.asarray(g_hi, float), 0.0)
        return g_lo, g_hi

    def _ofo_rvr_step(
        self, net: pp.pandapowerNet, err: float, dt_rvr_s: float,
    ) -> tuple:
        """New reactive level from the one-input OFO step (variant O0).

        Weights follow the match condition ``g_w / w_p = T_RVR s_p^2 / T_TS``,
        so the unconstrained step equals the integral RVR increment.  The
        monitored-bus voltages are soft output constraints, anchored at their
        MEASURED values and predicted through the frozen calibration columns
        along the headroom branch the level is heading for.

        Returns ``(new_level, n_out_active, box_active)``.
        """
        cfg, st = self.cfg, self.state
        s = float(self.pilot_sensitivity)
        g = cfg.ofo_w_p * s * s * cfg.t_rvr_s / dt_rvr_s     # match condition
        w0 = cfg.ofo_w_p * s * err / g                        # unconstrained step
        q = float(st.integrator)
        lo, hi = cfg.q_level_min - q, cfg.q_level_max - q
        rows = a = b = np.zeros(0)
        if self.v_bus_indices and self.dv_bus_dq is not None:
            head = self._headroom(net, upper=(q + w0 >= 0.0))
            rows = self.dv_bus_dq @ head
            v = net.res_bus.loc[self.v_bus_indices, "vm_pu"].to_numpy(dtype=float)
            a = cfg.ofo_v_min_pu - v
            b = cfg.ofo_v_max_pu - v
        w, n_act, box = _ofo_scalar_step(w0, lo, hi, g, rows, a, b, cfg.ofo_g_z_v)
        return q + w, n_act, box

    # -- initialisation ----------------------------------------------------

    def _measured_q(self, net: pp.pandapowerNet) -> NDArray[np.float64]:
        """Reactive output of commanded devices [Mvar], in GEN-then-DER order.

        Withheld machines read zero, matching their zero headroom, so the
        bumpless start seeds the level from the dispatch this controller is
        actually responsible for.
        """
        active_gen = self._active_generators(net)
        q_gen = np.asarray(
            [float(net.res_gen.at[g, "q_mvar"]) if active_gen[k] else 0.0
             for k, g in enumerate(self.gen_indices)],
            dtype=float,
        )
        q_der = np.asarray(
            [float(net.res_sgen.at[s, "q_mvar"]) for s in self.der_indices],
            dtype=float,
        )
        return np.concatenate([q_gen, q_der])

    def _initialise_level(self, net: pp.pandapowerNet) -> None:
        """Seed the integrator with the reactive utilisation already in service.

        Uses the AGGREGATE ratio (total dispatched Q over total headroom in that
        direction) rather than a per-device mean, so the area's total reactive
        output is preserved across the handover instead of merely its average
        utilisation.
        """
        q_now = self._measured_q(net)
        q_tot = float(np.sum(q_now))
        # Both branches take a POSITIVE denominator: ``_headroom(upper=False)``
        # already returns magnitudes, so the sign of q0 comes from q_tot alone.
        # Negating the lower branch as well would hand an absorbing area a
        # positive level -- the exact bump this initialisation exists to avoid.
        denom = float(np.sum(self._headroom(net, upper=(q_tot >= 0.0))))
        q0 = 0.0 if abs(denom) < 1e-9 else q_tot / denom
        q0 = float(np.clip(q0, self.cfg.q_level_min, self.cfg.q_level_max))
        # Parallel PI needs a tracking initial condition: x = q0 - Kp*e/s_p.
        # Otherwise merely enabling Kp changes the handover command by the full
        # algebraic term before any disturbance has occurred. The current S1
        # operating point keeps this state inside the output limits; clipping is
        # retained as a defensive bound for pathological initial conditions.
        p0 = 0.0
        if self.cfg.k_p_rvr > 0.0 and not self._holding():
            sensitivity = float(self.pilot_sensitivity)
            if sensitivity <= 1e-9:
                raise ValueError(
                    f"zone {self.zone_def.zone_id}: PI RVR requires positive "
                    f"pilot sensitivity, got {sensitivity:+.6g}"
                )
            v_pilot = float(net.res_bus.at[int(self.cfg.pilot_bus), "vm_pu"])
            err0 = self._deadband(self.cfg.v_ref_pu - v_pilot)
            p0 = self.cfg.k_p_rvr * err0 / sensitivity
        self.state.integrator = float(np.clip(
            q0 - p0, self.cfg.q_level_min, self.cfg.q_level_max,
        ))
        self.state.rvr_p_term = p0
        self.state.q_level = q0
        if self.verbose >= 1:
            print(f"  [SVR z{self.zone_def.zone_id}] bumpless start: "
                  f"q_level={q0:+.3f} (Q_area={q_tot:+.1f} Mvar)")

    # -- control law -------------------------------------------------------

    def _holding(self) -> bool:
        """True while the zone has nothing to dispatch (``calibrate``): no
        resource in service and, with STS access, no coupler term yet."""
        return (self.no_resources
                and abs(float(self.pilot_sensitivity)) < 1e-9)

    def _deadband(self, err: float) -> float:
        db = self.cfg.deadband_pu
        if db <= 0.0:
            return err
        if abs(err) <= db:
            return 0.0
        return err - np.sign(err) * db

    def step(
        self,
        net: pp.pandapowerNet,
        dt_rpr_s: float,
        *,
        rvr_due: bool,
        dt_rvr_s: float,
    ) -> SVROutput:
        """Advance the two loops and emit an OFO-shaped ``u``.

        The two regulators run on different grids, which is the whole point of
        the classical structure and is preserved here rather than collapsed:

        * the **RVR** integrates only when ``rvr_due`` (the slow dispatch grid,
          ``tso_period_s``), with per-dispatch gain ``dt_rvr_s / T_RVR``;
        * the **RPR** runs on every call (the inner grid, ``dt_s``), with
          per-step gain ``dt_rpr_s / T_RPR``.

        Collapsing both onto the dispatch grid would put the RPR at a per-step
        gain above one and make it overshoot -- an artefact of the harness, not
        a property of the scheme.
        """
        t0 = perf_counter()
        cfg, st = self.cfg, self.state

        if self.dv_pilot_dq is None:
            raise RuntimeError("calibrate() must run before step()")

        # -- bumpless start -------------------------------------------------
        # The reactive level is an area-wide UTILISATION fraction, so an
        # integrator starting at zero commands zero reactive output -- on a
        # loaded system that dumps the machines' reactive support in one step
        # and collapses the voltage long before the RVR can integrate back.
        # Classical SVR does not start that way: the integrator is initialised
        # from the reactive dispatch already in service.  This is ordinary
        # anti-bump initialisation of an integral controller, NOT a repair of
        # the scheme's absences -- those (no inequality constraints, no
        # discrete actuators, no inter-area exchange) stay exactly as they are.
        if not self._initialised:
            self._initialise_level(net)
            self._initialised = True
        if (cfg.sts_access and self.n_pcc and not self._coupler_term_added
                and self.dv_pilot_dq_coupler is not None):
            self._add_coupler_term(net)

        v_pilot = float(net.res_bus.at[int(cfg.pilot_bus), "vm_pu"])
        err = self._deadband(cfg.v_ref_pu - v_pilot)

        rvr_p_term = st.rvr_p_term
        ofo_rvr_n_active, ofo_rvr_box_active = 0, False
        p_update = (rvr_due or (cfg.rvr_p_every_step and cfg.k_p_rvr > 0.0)) \
            and not self._holding()
        if p_update:
            # q_level -> V_pilot must have positive local gain for both terms
            # to provide negative feedback. The selected pilots satisfy this.
            sensitivity = float(self.pilot_sensitivity)
            if cfg.k_p_rvr > 0.0 and sensitivity <= 1e-9:
                raise ValueError(
                    f"zone {self.zone_def.zone_id}: PI RVR requires positive "
                    f"pilot sensitivity, got {sensitivity:+.6g}"
                )
            rvr_p_term = cfg.k_p_rvr * err / sensitivity
            st.rvr_p_term = rvr_p_term
            if rvr_due and cfg.rvr_law == "ofo":
                # One-input OFO step (variant O0).  k_p_rvr == 0 is enforced at
                # construction, so there is no P path and level == state.
                candidate_x, ofo_rvr_n_active, ofo_rvr_box_active = (
                    self._ofo_rvr_step(net, err, dt_rvr_s))
                raw_level = candidate_x
                st.integrator = candidate_x
            elif rvr_due:
                # The P term is algebraic; only the separate state receives dt.
                previous_x = st.integrator
                candidate_x = float(np.clip(
                    previous_x + dt_rvr_s * err /
                    (cfg.t_rvr_s * abs(sensitivity)),
                    cfg.q_level_min, cfg.q_level_max,
                ))
                raw_level = candidate_x + rvr_p_term
                # Conditional integration: retain the previous state only when
                # the error would push a saturated PI output farther out.
                if ((raw_level > cfg.q_level_max and err > 0.0) or
                        (raw_level < cfg.q_level_min and err < 0.0)):
                    candidate_x = previous_x
                    raw_level = candidate_x + rvr_p_term
                st.integrator = candidate_x
            else:
                # Optional multirate PI: update the algebraic path from the
                # current measurement while holding the integral state exactly.
                raw_level = st.integrator + rvr_p_term
            st.q_level = float(np.clip(
                raw_level, cfg.q_level_min, cfg.q_level_max,
            ))
            st.error = err
            st.saturated = (
                st.q_level <= cfg.q_level_min + 1e-9 or
                st.q_level >= cfg.q_level_max - 1e-9
            )
        trial = st.q_level

        # -- RPR: distribute the level over the area's own headroom --------
        q_max = self._headroom(net, upper=True)
        q_min_abs = self._headroom(net, upper=False)
        q_ref = trial * (q_max if trial >= 0.0 else q_min_abs)

        n_gen = len(self.gen_indices)
        u = np.zeros(
            len(self.der_indices) + self.n_pcc + n_gen + self.n_oltc + self.n_shunt,
            dtype=float,
        )

        # DER block: direct reactive command (RPR assumed converged)
        for k, _s in enumerate(self.der_indices):
            u[k] = float(q_ref[n_gen + k])

        # PCC block: zero for the classical scheme, which dispatches no
        # interface set-point (module docstring).  With access (S2) it holds
        # the coupler references of the alignment law, recomputed at each RVR
        # update and held in between, like the level they follow.
        if cfg.sts_access and self.n_pcc:
            if rvr_due or not np.all(np.isfinite(self.q_ref_pcc)):
                self.q_ref_pcc = self.coupler_references(net, trial)
            u[len(self.der_indices):len(self.der_indices) + self.n_pcc] = (
                self.q_ref_pcc)
        off = len(self.der_indices) + self.n_pcc

        # Machine block: PI RPR moves the AVR setpoint toward the Q target.
        # Its integral state is separate from the algebraic output.
        gain = dt_rpr_s / max(cfg.t_rpr_s, 1e-9)
        active_gen = self._active_generators(net)
        rpr_blocked = np.zeros(n_gen, dtype=bool)
        rpr_p_term = np.zeros(n_gen, dtype=float)
        rpr_ofo_active = np.zeros(n_gen, dtype=bool)
        rpr_ofo_box = np.zeros(n_gen, dtype=bool)
        if cfg.rpr_law == "ofo":
            g_lo_all, g_hi_all = self._gen_q_limits(net)
        for k, g in enumerate(self.gen_indices):
            v_now = float(net.gen.at[g, "vm_pu"])
            if not active_gen[k]:
                # An unavailable actuator cannot track Q. Freeze its actual
                # reference, including on restoration's preceding sample.
                u[off + k] = v_now
                self._gen_was_available[k] = False
                continue
            q_now = float(net.res_gen.at[g, "q_mvar"])
            slope = float(self.dq_dvset[k]) if self.dq_dvset is not None else 0.0
            if abs(slope) < 1e-9:
                u[off + k] = v_now
                self._gen_was_available[k] = False
                continue
            q_error = float(q_ref[k]) - q_now
            rpr_p_term[k] = cfg.k_p_rpr * q_error / slope
            if (not self._gen_was_available[k] or
                    not np.isfinite(self._rpr_integrator[k])):
                # Cancel the initial algebraic P contribution so enabling PI
                # creates no extra bump at start-up or restoration.
                self._rpr_integrator[k] = v_now - rpr_p_term[k]
            previous_y = float(self._rpr_integrator[k])
            if cfg.rpr_law == "ofo":
                # One-input OFO step per machine (variant O0): input v_set,
                # output Q_i, g_w / w_Q = T_RPR h^2 / dt, capability as a soft
                # output constraint anchored at the measured Q.  k_p_rpr == 0
                # is enforced at construction, so there is no P path.
                g_k = cfg.ofo_w_q * slope * slope * cfg.t_rpr_s / dt_rpr_s
                d0 = cfg.ofo_w_q * slope * q_error / g_k
                d, n_k, box_k = _ofo_scalar_step(
                    d0,
                    cfg.v_set_min_pu - previous_y,
                    cfg.v_set_max_pu - previous_y,
                    g_k,
                    np.array([slope]),
                    np.array([float(g_lo_all[k]) - q_now]),
                    np.array([float(g_hi_all[k]) - q_now]),
                    cfg.ofo_g_z_q,
                )
                candidate_y = previous_y + d
                raw_v = candidate_y
                rpr_ofo_active[k] = bool(n_k > 0)
                rpr_ofo_box[k] = bool(box_k)
            else:
                delta_y = gain * q_error / slope
                candidate_y = float(np.clip(
                    previous_y + delta_y, cfg.v_set_min_pu, cfg.v_set_max_pu,
                ))
                raw_v = candidate_y + rpr_p_term[k]
                if ((raw_v > cfg.v_set_max_pu and delta_y > 0.0) or
                        (raw_v < cfg.v_set_min_pu and delta_y < 0.0)):
                    candidate_y = previous_y
                    raw_v = candidate_y + rpr_p_term[k]
            v_new = float(np.clip(raw_v, cfg.v_set_min_pu, cfg.v_set_max_pu))
            if cfg.rpr_voltage_priority:
                # The two cached derivatives recover the pilot response to
                # this AVR reference, including its sign. No plant model is
                # refreshed here. Use the same dead-banded pilot error as RVR.
                dv_pilot = float(self.dv_pilot_dq[k]) * slope * (v_new - v_now)
                if err * dv_pilot < -1e-12:
                    v_new = v_now
                    rpr_blocked[k] = True
                    candidate_y = previous_y
            self._rpr_integrator[k] = candidate_y
            self._gen_was_available[k] = True
            u[off + k] = v_new
        off += n_gen

        # Discrete blocks: echo the measured position back (local logic owns them)
        for k, t in enumerate(self.zone_def.oltc_trafo_indices):
            u[off + k] = float(net.trafo.at[int(t), "tap_pos"])
        off += self.n_oltc
        for k, sb in enumerate(self.zone_def.shunt_bus_indices):
            sel = net.shunt.index[net.shunt["bus"] == int(sb)]
            u[off + k] = float(net.shunt.at[sel[0], "step"]) if len(sel) else 0.0

        return SVROutput(
            u_new=u,
            objective_value=float(err ** 2),
            solver_status="svr",
            solve_time_s=perf_counter() - t0,
            q_level=trial,
            pilot_v_pu=v_pilot,
            pilot_v_ref_pu=cfg.v_ref_pu,
            pilot_error_pu=err,
            saturated=st.saturated,
            # Participation order is GEN then DER, unlike the OFO-shaped u.
            q_ref_mvar=q_ref.copy(),
            gen_available=active_gen.copy(),
            rpr_blocked=rpr_blocked,
            rvr_p_term_level=rvr_p_term,
            rpr_p_term_pu=rpr_p_term,
            ofo_rvr_n_active=ofo_rvr_n_active,
            ofo_rvr_box_active=ofo_rvr_box_active,
            ofo_rpr_active=rpr_ofo_active,
            ofo_rpr_box_active=rpr_ofo_box,
        )


# ---------------------------------------------------------------------------
#  Coordinator (duck-types MultiTSOCoordinator.step for the runner)
# ---------------------------------------------------------------------------


class SVRCoordinator:
    """Runs one independent RVR per area.

    No signal is exchanged between areas: they interact only through the
    physics.  Inter-area counter-actuation is a property of the scheme, not a
    bug -- it is one of the things the comparison is meant to expose.
    """

    def __init__(self, controllers: Dict[int, SVRZoneController]) -> None:
        self.controllers = controllers
        #: Present so the runner's diagnostics lookup does not need a branch.
        self.last_coupling_diagnostics: Dict[int, Dict] = {}

    def calibrate(self, net: pp.pandapowerNet, runpp_kwargs: Dict) -> None:
        for ctrl in self.controllers.values():
            ctrl.calibrate(net, runpp_kwargs)

    def step(
        self,
        net: pp.pandapowerNet,
        dt_rpr_s: float,
        *,
        rvr_due: bool,
        dt_rvr_s: float,
    ) -> Dict[int, SVROutput]:
        out = {
            z: c.step(net, dt_rpr_s, rvr_due=rvr_due, dt_rvr_s=dt_rvr_s)
            for z, c in self.controllers.items()
        }
        self.last_coupling_diagnostics = {
            z: {"contraction_lhs": float("nan"), "q_level": o.q_level}
            for z, o in out.items()
        }
        return out
