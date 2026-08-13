"""
network/ieee39/load_model.py
============================
Voltage-dependent (ZIP) plant load model, exact exponent image
(kpu, kqu) = (1, 2) anchored at a chosen voltage.

Decision 2026-07-17 (RMS co-simulation, PowerFactory build plan): both the
pandapower oracle and the PowerFactory RMS model use voltage-dependent
loads

    P_served(V) = P_prof · (V / V_anchor)^1
    Q_served(V) = Q_prof · (V / V_anchor)^2

where ``P_prof`` / ``Q_prof`` are the SimBench-profile-resolved powers and
``V_anchor`` (default 1.03 pu, the network voltage setpoint) is the voltage
at which the profile values are served exactly.  The SimBench series are
voltage-agnostic power scalings, so the anchor is a modelling convention;
1.03 pu preserves the pre-ZIP power balance at the operating setpoint.

Exact pandapower realisation
----------------------------
pandapower evaluates loads as ``P = p_mw · (cz·V² + ci·V + cp)`` with V
relative to 1.0 pu, shares in percent and ``cp = 1 − cz − ci``.  A pure
exponent anchored away from 1.0 pu therefore folds the anchor into the
base value:

    P: ``const_i_p_percent = 100`` and ``p_mw → p_mw / V_anchor``
       ⇒ P_served = (P_prof/V_anchor)·V            ≡ P_prof·(V/V_anchor)
    Q: ``const_z_q_percent = 100`` and ``q_mvar → q_mvar / V_anchor²``
       ⇒ Q_served = (Q_prof/V_anchor²)·V²          ≡ Q_prof·(V/V_anchor)²

The mapping is exact (no approximation).  ``base_p_mw`` / ``base_q_mvar``
are rescaled identically so that ``apply_profiles`` keeps producing the
correctly anchored nominals at every timestep.

The identical convention on the PowerFactory side: per-load LDF/RMS voltage
exponents kpu = 1, kqu = 2 with P0/Q0 taken directly from the snapshot
(which stores the *rescaled* bases, i.e. the 1.0-pu-equivalent values), and
the ComLdf option "consider voltage dependence of loads" enabled.

Conventions
-----------
* Applies to **every** load (TN and DN) present at call time.
* Contingency stress loads (``prepare_load_contingencies``) are created
  later and deliberately stay constant-PQ: they are specified disturbance
  magnitudes, not physical demand.
* Calling twice raises (the rescale must never compound); the applied
  anchor is recorded per load in the ``zip_anchor_vm_pu`` column, which the
  dynamic snapshot serialises for the PF sync.

Author: Manuel Schwenke / Claude Code (2026-07-17)
"""

from __future__ import annotations

import pandapower as pp


def apply_zip_load_model(
    net: pp.pandapowerNet,
    *,
    anchor_vm_pu: float = 1.03,
    verbose: bool = False,
) -> None:
    """Convert every load to the anchored (kpu, kqu) = (1, 2) ZIP image.

    Parameters
    ----------
    net : pp.pandapowerNet
        Fully built network (after ``build_ieee39_net`` and, when used,
        ``add_hv_networks``) whose loads carry ``base_p_mw`` /
        ``base_q_mvar`` columns.  Modified in place.
    anchor_vm_pu : float
        Voltage [pu] at which the profile powers are served exactly.
    verbose : bool
        Print a one-line summary.

    Raises
    ------
    ValueError
        If the anchor is non-positive, loads lack base columns, or the
        model was already applied (``zip_anchor_vm_pu`` column present).
    """
    if anchor_vm_pu <= 0.0:
        raise ValueError(f"anchor_vm_pu must be positive, got {anchor_vm_pu}")
    if len(net.load) == 0:
        raise ValueError("apply_zip_load_model: net has no loads")
    if "zip_anchor_vm_pu" in net.load.columns:
        raise ValueError(
            "apply_zip_load_model was already applied to this network "
            "(zip_anchor_vm_pu column exists); the base rescale must not "
            "compound"
        )
    for col in ("base_p_mw", "base_q_mvar"):
        if col not in net.load.columns:
            raise ValueError(
                f"net.load lacks {col!r}; apply_zip_load_model must run "
                f"after the builder set the base columns"
            )
        if net.load[col].isna().any():
            bad = net.load.index[net.load[col].isna()].tolist()
            raise ValueError(f"net.load.{col} is NaN for loads {bad}")

    # Exact exponent image: 100 % constant-current P, 100 % constant-
    # impedance Q; anchor folded into the bases (see module docstring).
    net.load["const_z_p_percent"] = 0.0
    net.load["const_i_p_percent"] = 100.0
    net.load["const_z_q_percent"] = 100.0
    net.load["const_i_q_percent"] = 0.0

    net.load["p_mw"] = net.load["p_mw"].astype(float) / anchor_vm_pu
    net.load["base_p_mw"] = net.load["base_p_mw"].astype(float) / anchor_vm_pu
    net.load["q_mvar"] = net.load["q_mvar"].astype(float) / anchor_vm_pu ** 2
    net.load["base_q_mvar"] = (
        net.load["base_q_mvar"].astype(float) / anchor_vm_pu ** 2
    )

    net.load["zip_anchor_vm_pu"] = float(anchor_vm_pu)

    if verbose:
        print(
            f"[apply_zip_load_model] {len(net.load)} loads -> "
            f"P ~ V^1, Q ~ V^2 anchored at {anchor_vm_pu:.3f} pu "
            f"(const-I P / const-Z Q, bases rescaled by "
            f"1/{anchor_vm_pu:.3f} and 1/{anchor_vm_pu ** 2:.4f})"
        )
