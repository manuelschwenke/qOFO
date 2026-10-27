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
S2   S1 with the couplers in the alignment law       NOT YET (plan 1.5)
O2   TS-OFO, STS under local q(v)                    OFO, no access
O3   TS-OFO + interface set-points + STS-OFO         proposed cascade
M    one OFO on the coupled model, every step        bound (not a rung)
==== ============================================== =====================

``M`` weights the STS buses with the paramset's ``dso_g_v`` (the STS-OFOs'
own weight), dispatches the banks through the same holding-time / mutual-
exclusion / rank-1 machinery as the zones, and refuses to run if its loop gain
``lambda_max`` reaches 2 (rebuilt 2026-10-27).  At the selected weights the
single loop has lambda_max ~5.1, so M's whole ``g_w`` block is scaled by one
factor to the TS-OFO's loop gain (author decision 2026-10-27); the factor is
printed and recorded (``central_gw_kappa``).  It stays in :data:`NOT_READY`
until a smoke run has confirmed it.

Author: Manuel Schwenke / Claude Code (2026-10-27)
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Tuple

from configs.config import MultiTSOConfig
from experiments.ch_10_case1_svr import run_case1 as _rc1

PARAMSET = _rc1.PARAMSET

#: Thesis labels the Case-2 producers report under.
TARGET = "ch:case3"

_O_STS_LABELS = dict(local_der_mode="qv", tso_q_mode="qv", dso_q_mode="qv")

VARIANTS: Dict[str, Dict[str, Any]] = {
    "L2": dict(_rc1.VARIANTS["L2"]),
    "S1": dict(_rc1.VARIANTS["S1"]),
    "O2": dict(_rc1.VARIANTS["O2"]),
    # The runner default is the cascade.  ``local_der_mode`` is only a
    # provenance label in dso_mode="ofo"; it is set as in O2 so that O2 -> O3
    # differs in ``dso_mode`` alone.
    "O3": dict(tso_mode="ofo", dso_mode="ofo", **_O_STS_LABELS),
    # One MIQP over every actuator and measurement, fired every step: it
    # replaces the 20 s STS layer as well as the 180 s TS layer, so pinning it
    # to 180 s would make the "bound" nine times slower than the scheme it
    # bounds (measured 2026-09-02, superseded ladder).
    "M": dict(control_scope="central", tso_mode="ofo", dso_mode="local",
              local_sensitivities_tso=False, local_sensitivities_dso=False,
              central_period_s=None, central_match_tso_loop_gain=True,
              **_O_STS_LABELS),
}

#: Registered variants that must not enter a run of record yet.
NOT_READY: Dict[str, str] = {
    "M": "central controller not repaired for the MIQP shunt banks and its STS "
         "voltage weight is a default (plan Phase 1.6)",
}

#: First wave of the reference run (S2 and M join later on the same run).
WAVE_1: List[str] = ["L2", "S1", "O2", "O3"]

_SVR_FIELDS = set(_rc1._SVR_FIELDS)

#: What each controlled step may change.  Enforced on RESOLVED configs.
STEP_ALLOWED: Dict[Tuple[str, str], set] = {
    ("L2", "S1"): _SVR_FIELDS,
    # The law step: SVR -> OFO, which brings the MT taps and the banks.
    ("S1", "O2"): _SVR_FIELDS | {"tso_oltc_mode",
                                 "install_tso_tertiary_shunts",
                                 "shunt_dispatch"},
    # The access step under the OFO law.
    ("O2", "O3"): {"dso_mode"},
    # Information, not a rung: one controller, full model, every step.
    ("O3", "M"): {"control_scope", "dso_mode", "local_sensitivities_tso",
                  "local_sensitivities_dso", "central_period_s",
                  "central_dso_g_v", "central_match_tso_loop_gain"},
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
        # "At the same weights": M weights the STS buses as the STS-OFOs do
        # (the paramset's dso_g_v), not with the dataclass default 2e4, which
        # was ~9x the selected value.
        cfg = dataclasses.replace(cfg, central_dso_g_v=float(cfg.dso_g_v))
    return cfg


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
