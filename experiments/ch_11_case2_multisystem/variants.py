"""
experiments/ch_11_case2_multisystem/variants.py
===============================================

The compared variants of Case Study 2 (thesis ``ch:case3``, table
``tab:case3:variants``), resolved on ONE base configuration, and the guard that
every controlled step changes only its own mechanism.

Base
----
``run_case1.base_config``: ``make_config()`` + DSO_3 x2 (the Chapter-8 batch
network) + the selected Chapter-9 paramset + horizon, no live plots.  Two
things are then fixed here rather than inherited:

* **Events.**  ``make_config()`` carries the interactive runner's own event
  list, which has drifted from the thesis reference period (a 7th load event
  since c23e67b, 2026-10-21) and is edited for other campaigns.  A Case-2 run
  therefore starts with NO events and receives its schedule explicitly from
  the producer (``ref/scenarios.py``, the MC draw, the step battery).
* **Noise.**  The measurement-noise model (accuracy class "minimum") covers
  every controller -- the classical SVR (``apply_to_svr``) and the droop's own
  terminal-voltage reading (``apply_to_der_voltage``) included -- so no variant
  reads plant truth.  ``noise_seed`` selects the realisation, so a campaign can
  give every episode its own realisation while all variants of that episode
  share it (author decision 2026-10-27).

Variants
--------
==== ============================================== =====================
L2   local q(v) at both layers, taps held, no banks  floor
S1   integral RVR + RPR + safeguard (Case 1's S1)    classical, no access
S2   S1 with the couplers in the alignment law;     classical, with access
     the STS-OFOs track the resulting references
O2   TS-OFO, STS under local q(v)                    OFO, no access
O3   TS-OFO + interface set-points + STS-OFO         proposed cascade
M    one OFO on the coupled model, every step        centralised reference
==== ============================================== =====================

``M`` runs at the selected weights, unscaled, with the paramset's ``dso_g_v``
on the STS buses, and dispatches the banks through the same holding-time /
mutual-exclusion / rank-1 machinery as the zones.  Its transmission taps and
banks keep the cascade's timescale: after a switch they are locked as long as
in the cascade (:func:`ts_equivalent_cooldown`, 15 min), the banks' own
30-min holding time on top.  The loop gain is checked over the continuous
actuators (0.98 < 2), the discrete ones being rate-limited.  Adopted
2026-10-28 (author) after the comparison of four tunings in
``results/ch_11_case2_multisystem/ref/0007_*_m_tuning`` -- the earlier uniform
scaling of all weights to the design's loop gain (kappa 2.42) slowed every
continuous actuator to tame a mode the taps and banks of all areas create when
they may move every 20 s, and left M worse than O3.

Author: Manuel Schwenke / Claude Code (2026-10-27)
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Tuple

from configs.config import MultiTSOConfig
from configs.paramsets import load_paramset
from experiments.ch_10_case1_svr import run_case1 as _rc1

PARAMSET = _rc1.PARAMSET

#: Thesis labels the Case-2 producers report under.
TARGET = "ch:case3"

_O_STS_LABELS = dict(local_der_mode="qv", tso_q_mode="qv", dso_q_mode="qv")

VARIANTS: Dict[str, Dict[str, Any]] = {
    "L2": dict(_rc1.VARIANTS["L2"]),
    "S1": dict(_rc1.VARIANTS["S1"]),
    # S1 with access: the couplers of each area's child STS join the
    # alignment law between their neutral point and the CAIR interval edges,
    # and the STS-OFOs track the references (design note 2026-10-27, approved).
    "S2": dict(_rc1.VARIANTS["S1"], dso_mode="ofo", svr_sts_access=True),
    "O2": dict(_rc1.VARIANTS["O2"]),
    # The runner default is the cascade.  ``local_der_mode`` is only a
    # provenance label in dso_mode="ofo"; it is set as in O2 so that O2 -> O3
    # differs in ``dso_mode`` alone.
    "O3": dict(tso_mode="ofo", dso_mode="ofo", **_O_STS_LABELS),
    # One MIQP over every actuator and measurement, fired every step: it
    # replaces the 20 s STS layer as well as the 180 s TS layer, so pinning it
    # to 180 s would make the "bound" nine times slower than the scheme it
    # bounds (measured 2026-09-02, superseded ladder).  The selected weights,
    # unscaled; the transmission taps and banks keep the cascade's timescale
    # (cooldown set in ``resolved``) and the loop gain is checked over the
    # continuous actuators (author decision 2026-10-28, candidate "M_rate").
    "M": dict(control_scope="central", tso_mode="ofo", dso_mode="local",
              local_sensitivities_tso=False, local_sensitivities_dso=False,
              central_period_s=None, central_lambda_rule="continuous",
              **_O_STS_LABELS),
}

#: Registered variants that must not enter a run of record yet.
NOT_READY: Dict[str, str] = {}

#: First wave of the reference run (S2 and M join later on the same run).
WAVE_1: List[str] = ["L2", "S1", "O2", "O3"]

_SVR_FIELDS = set(_rc1._SVR_FIELDS)

#: What each controlled step may change.  Enforced on RESOLVED configs.
STEP_ALLOWED: Dict[Tuple[str, str], set] = {
    ("L2", "S1"): _SVR_FIELDS,
    # The access step under the classical law.
    ("S1", "S2"): {"dso_mode", "svr_sts_access"},
    # The law step with the access held (objective and discrete devices too).
    ("S2", "O3"): _SVR_FIELDS | {"tso_oltc_mode", "install_tso_tertiary_shunts",
                                 "shunt_dispatch", "svr_sts_access"},
    # The law step: SVR -> OFO, which brings the MT taps and the banks.
    ("S1", "O2"): _SVR_FIELDS | {"tso_oltc_mode",
                                 "install_tso_tertiary_shunts",
                                 "shunt_dispatch"},
    # The access step under the OFO law.
    ("O2", "O3"): {"dso_mode"},
    # Information, not a rung: one controller, full model, every step --
    # a centralised reference at equal weights and equal device rate limits,
    # not a bound (author 2026-10-28).
    ("O3", "M"): {"control_scope", "dso_mode", "local_sensitivities_tso",
                  "local_sensitivities_dso", "central_period_s",
                  "central_dso_g_v", "central_lambda_target",
                  "central_lambda_rule", "central_ts_discrete_cooldown"},
}


def base_config(minutes: int, events: Iterable[Any] = (),
                noise_seed: Optional[int] = None) -> MultiTSOConfig:
    """Case-1 base + the given events (none inherited) + the noise seed."""
    cfg = _rc1.base_config(minutes)
    noise = dataclasses.replace(cfg.measurement_noise, apply_to_svr=True,
                                apply_to_der_voltage=True)
    if noise_seed is not None:
        noise = dataclasses.replace(noise, seed=int(noise_seed))
    return dataclasses.replace(cfg, contingencies=list(events),
                               measurement_noise=noise)


def resolved(name: str, minutes: int, events: Iterable[Any] = (),
             noise_seed: Optional[int] = None) -> MultiTSOConfig:
    """The config the runner receives for variant ``name``."""
    if name not in VARIANTS:
        raise KeyError(f"unknown Case-2 variant {name!r}; "
                       f"valid: {sorted(VARIANTS)}")
    cfg = base_config(minutes, events, noise_seed)
    cfg = dataclasses.replace(cfg, **VARIANTS[name])
    if name == "M":
        # M weights the STS buses as the STS-OFOs do (the paramset's dso_g_v),
        # not with the dataclass default 2e4, which was ~9x the selected value.
        # Its loop gain over the continuous actuators is capped at
        # M_LAMBDA_TARGET: the runner scales all of M's g_w by
        # kappa = lambda_max / target when it lies above (v4 weights: 1.781).
        cfg = dataclasses.replace(
            cfg, central_dso_g_v=float(cfg.dso_g_v),
            central_lambda_target=M_LAMBDA_TARGET,
            central_ts_discrete_cooldown=ts_equivalent_cooldown(cfg))
    return cfg


def ts_equivalent_cooldown(cfg: MultiTSOConfig) -> int:
    """Lock of a transmission tap or bank in M, in central steps, equal in
    time to the cascade's.

    A device that switches at controller iteration k is locked until
    iteration ``k + 1 + c`` (``BaseOFOController``), with c the TS layer's
    lock ``tso_int_cooldown`` (``int_cooldown`` if unset).  The TS-OFO
    iterates every ``tso_period_s``, so in the cascade it can move again
    ``(1 + c) * tso_period_s`` after the switch (15 min since 2026-10-30, 18
    before).  M iterates every ``dt_s``; with its lock ``k + 1 + n`` the same
    time needs ``n = (1 + c) * tso_period_s / dt_s - 1`` (44).  The banks'
    own holding time (``shunt_t_dwell_s``) applies in both on top.
    """
    ratio = float(cfg.tso_period_s) / float(cfg.dt_s)
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("tso_period_s must be a multiple of dt_s")
    c = cfg.int_cooldown if cfg.tso_int_cooldown is None else cfg.tso_int_cooldown
    return int((1 + int(c)) * round(ratio) - 1)


#: M's loop-gain ceiling over the continuous actuators (author 2026-11-06, "Use 0.90
#: instead").  Until then M was capped at the selected design's lambda_max_B (1.405), and
#: in REF E (run 0019) it ran a period-2 cycle of its TS-IBG and AVR set-points between the
#: opening of 9-39 and the load step: with the plant's post-disturbance sensitivities its
#: mismatch loop gain is 3.12 at that cap (shared notes cen_mismatch.py), and 0.90 brings it
#: to 2.0.  Taps and banks are rate-limited instead (:func:`ts_equivalent_cooldown`).
M_LAMBDA_TARGET = 0.90


def selected_lambda_max() -> float:
    """``lambda_max_B`` of the selected design, from the paramset provenance.

    M's loop-gain ceiling until 2026-11-06 (now :data:`M_LAMBDA_TARGET`, below it).
    """
    prov = load_paramset(PARAMSET).get("provenance") or {}
    lam = prov.get("lambda_max_B")
    if lam is None:
        raise KeyError(f"paramset {PARAMSET!r} records no lambda_max_B; "
                       f"M has no loop gain to match")
    return float(lam)


def assert_case2_controlled(minutes: int = 10,
                            events: Iterable[Any] = (),
                            noise_seed: Optional[int] = None) -> None:
    """Raise if a step changes more than the mechanism it isolates.

    Every pair is resolved on the same base, events and noise seed, so a
    difference in any field outside the step's allowed set -- including a
    stray event or seed -- aborts the run before anything is simulated.
    """
    events = list(events)
    problems: List[str] = []
    for (a, b), allowed in STEP_ALLOWED.items():
        ca = resolved(a, minutes, events, noise_seed)
        cb = resolved(b, minutes, events, noise_seed)
        for f in dataclasses.fields(MultiTSOConfig):
            k = f.name
            if k in allowed or k in ("result_dir", "verbose"):
                continue
            va, vb = getattr(ca, k, None), getattr(cb, k, None)
            try:
                same = bool(va == vb)
            except Exception:
                same = repr(va) == repr(vb)
            if not same:
                problems.append(f"  {a}->{b}: {k}  {a}={va!r}  {b}={vb!r}")
    if problems:
        raise RuntimeError("Case 2 is NOT controlled:\n" + "\n".join(problems))
