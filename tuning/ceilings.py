"""
tuning/ceilings.py
==================
One-shot extraction of per-actuator-class LMI ceilings from a baseline
:class:`MultiTSOConfig`, by running
:func:`analyse_multi_zone_stability` once at the nominal operating
point and reading back the spectral diagnostics.

Conceptually
------------
The LMI condition gives a *minimum* ``g_w`` for which the contraction
certificate holds.  Any larger ``g_w`` is also stable (more proximal
regularization → smaller steps → more conservative).  BO uses this
threshold as the *upper* bound on its search: there is no need to
search above it (sluggish-but-stable region), so we focus the budget
below (the practically interesting region where empirical stability
holds at much smaller ``g_w``).

For ``g_v`` (objective weight on voltage tracking) the direction
inverts: ``g_v`` scales the curvature ``C``, so the critical threshold
is the *largest* ``g_v`` for which ``rho(M_full^c) < 1``.

Approximation (Task 1)
----------------------
A precise per-block solve requires knowing the block partition of
``M_full^c`` and re-assembling it for candidate ``g_w`` values.  For
Task 1 we use a coarse isotropic-scaling approximation:

* For continuous blocks (``g_w_der``, ``g_w_pcc``, ``g_w_dso_der``,
  ``g_v``): exploit the linear scaling ``rho ~ 1/g_w`` and
  ``rho ~ g_v`` to set the ceiling at the value where
  ``rho(M_full^c) ≈ 1``.
* For TSO OLTC (``g_w_tso_oltc``): use ``c3_discrete.g_min_required``
  directly (it already gives the analytical threshold per OLTC; we
  take the max across zones).
* For DSO OLTC (``g_w_dso_oltc``): no analytical bound (DSO OLTCs settle
  finitely by Proposition 1.3); ceiling is ``np.inf`` and BO falls
  back to the parameter's ``fallback_high``.

The approximation is documented in the returned ``Ceilings.notes``.

Caching
-------
Ceilings depend only on (network topology, baseline operating point,
``g_v``, ``g_q``, ``dso_g_v``, slack penalties, profile config).  They
are cached to JSON keyed by a SHA256 hash of these inputs.  ``g_w_*``
fields are explicitly excluded from the hash because they are exactly
the values being tuned.

Implementation note on capturing the result
-------------------------------------------
The experiment runner module is named ``experiments/000_M_TSO_M_DSO.py``.
The leading digit prevents normal ``import``, so we load it via
``importlib.util``.  The runner executes the stability analysis
internally (via ``_run_delayed_stability_analysis``) but does not attach
the resulting :class:`MultiZoneStabilityResult` to any iteration record,
so we monkey-patch that helper on the loaded module to capture its
return value.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from analysis.stability_analysis import MultiZoneStabilityResult
from configs.multi_tso_config import MultiTSOConfig
from tuning._sim_loader import get_runner_module
from tuning._types import Ceilings


CACHE_DIR = Path("results/tuning/ceilings_cache")


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _cache_key(cfg: MultiTSOConfig) -> str:
    """Deterministic 16-hex-character hash of the inputs that affect
    ceilings.

    Includes: scenario, zone partition method, voltage setpoint, the
    objective weights ``g_v`` / ``g_q`` / ``dso_g_v``, slack-penalty
    weights ``g_z_*``, and profile configuration (``use_profiles``,
    ``start_time``, control periods).

    Excludes (intentionally): all ``g_w_*`` values (those are the BO
    decision variables we are sizing), contingencies (ceilings are
    nominal), and outputs/result_dir (irrelevant to the LMI).
    """
    payload = {
        "scenario":        cfg.scenario,
        "use_fixed_zones": bool(cfg.use_fixed_zones),
        "v_setpoint_pu":   float(cfg.v_setpoint_pu),
        "g_v":             float(cfg.g_v),
        "g_q":             float(cfg.g_q),
        "dso_g_v":         float(cfg.dso_g_v),
        "g_z_voltage":     float(cfg.g_z_voltage),
        "g_z_current":     float(cfg.g_z_current),
        "g_z_q_gen":       float(cfg.g_z_q_gen),
        "use_profiles":    bool(cfg.use_profiles),
        "start_time":      cfg.start_time.isoformat(),
        "tso_period_s":    float(cfg.tso_period_s),
        "dso_period_s":    float(cfg.dso_period_s),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_ceilings(
    baseline_cfg: MultiTSOConfig,
    *,
    use_cache: bool = True,
    cache_dir: Path = CACHE_DIR,
) -> Ceilings:
    """Compute per-block LMI ceilings.

    Strategy
    --------
    1. If cached and ``use_cache``, return the cached result.
    2. Otherwise: run a one-TSO-step closed-loop simulation with
       ``run_stability_analysis=True`` and ``stability_analysis_at_s=0.0``.
       Capture the :class:`MultiZoneStabilityResult` produced internally.
    3. Derive per-block ceilings from the spectral diagnostics
       (see :func:`_ceilings_from_result`).
    4. Cache and return.

    Parameters
    ----------
    baseline_cfg
        Baseline config defining the operating point.  Live plots,
        verbose, observers, and contingencies are forced off internally.
    use_cache
        If True, read from / write to JSON cache.
    cache_dir
        Cache directory.  Created if missing.

    Returns
    -------
    Ceilings
        Frozen dataclass of per-block ceilings.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(baseline_cfg)
    cache_file = cache_dir / f"ceilings_{key}.json"

    if use_cache and cache_file.exists():
        with cache_file.open("r") as f:
            data = json.load(f)
        return Ceilings(**data)

    result = _run_one_step_stability(baseline_cfg)
    ceilings = _ceilings_from_result(result, baseline_cfg)

    with cache_file.open("w") as f:
        json.dump(asdict(ceilings), f, indent=2, default=str)

    return ceilings


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _run_one_step_stability(cfg: MultiTSOConfig) -> MultiZoneStabilityResult:
    """Run one TSO step with stability analysis on, return the captured
    :class:`MultiZoneStabilityResult`.

    The runner does not attach the result to any iteration record, so we
    monkey-patch ``_run_delayed_stability_analysis`` on the loaded
    module to capture its return value.
    """
    mod = get_runner_module()

    captured: list[MultiZoneStabilityResult] = []
    original = mod._run_delayed_stability_analysis

    def _capturing_wrapper(*args: Any, **kwargs: Any) -> MultiZoneStabilityResult:
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    mod._run_delayed_stability_analysis = _capturing_wrapper

    short_cfg = dataclasses.replace(
        cfg,
        # Keep the simulation just long enough for the t=0 stability
        # analysis hook to fire.  ``stability_analysis_at_s=0.0`` triggers
        # on the first record; one TSO step plus a one-second margin
        # ensures the loop reaches that record.
        n_total_s=float(cfg.tso_period_s) + 1.0,
        stability_analysis_at_s=0.0,
        run_stability_analysis=True,
        contingencies=[],
        verbose=0,
        live_plot_controller=False,
        live_plot_cascade=False,
        live_plot_system=False,
        load_tuned_params_path=None,
    )

    try:
        mod.run_multi_tso_dso(short_cfg)
    finally:
        mod._run_delayed_stability_analysis = original

    if not captured:
        raise RuntimeError(
            "Stability analysis did not run during the one-step simulation; "
            "captured list is empty."
        )
    return captured[0]


def _ceilings_from_result(
    result: MultiZoneStabilityResult,
    cfg: MultiTSOConfig,
) -> Ceilings:
    """Derive per-block ``g_w`` ceilings from a stability result.

    Approximation (Task 1)
    ----------------------
    For continuous blocks (``g_w_der``, ``g_w_pcc``, ``g_w_dso_der``,
    ``g_v``) we exploit the leading-order scaling of the iteration
    matrix ``M = G_w^{-1/2} (R + K^T Q K) G_w^{-1/2}``:

    * Increasing every ``g_w`` by factor ``f`` scales every entry of
      ``M`` by ``1/f``, so the spectral radius scales as ``1/f``.
    * Increasing ``g_v`` (and therefore the diagonal of ``Q``) by
      factor ``f`` scales every entry of ``M`` by ``f``, so the
      spectral radius scales as ``f``.

    Hence the threshold (``rho ≈ 1``) sits at:

        ``g_w_critical   = g_w_baseline   * rho_baseline``
        ``g_v_critical   = g_v_baseline   / rho_baseline``

    This is *isotropic*: it pretends a single block can be scaled in
    isolation without affecting the others.  In reality the partition
    matters and the per-block bound is tighter.  Task 1.5 will refine
    this if the BO results show the bound is too loose.

    For ``g_w_tso_oltc`` we use ``c3_discrete.g_min_required`` directly:
    it already provides the analytical threshold per OLTC actuator
    per zone.  We take the maximum across zones (worst case).

    For ``g_w_dso_oltc`` no analytical bound is available (Proposition
    1.3: DSO OLTCs settle finitely without affecting cascade stability),
    so the ceiling is ``inf`` and BO falls back to ``fallback_high``.
    """
    notes_parts: list[str] = []

    # ── Continuous ceilings (isotropic-scaling approximation) ──────────
    rho_c = float(result.c2_continuous.spectral_radius)
    g_w_der_ceil:     float = math.inf
    g_w_pcc_ceil:     float = math.inf
    g_w_dso_der_ceil: float = math.inf
    g_v_ceil:         float = math.inf
    if rho_c > 0.0 and math.isfinite(rho_c):
        g_w_der_ceil     = float(cfg.g_w_der)     * rho_c
        g_w_pcc_ceil     = float(cfg.g_w_pcc)     * rho_c
        g_w_dso_der_ceil = float(cfg.g_w_dso_der) * rho_c
        g_v_ceil         = float(cfg.g_v) / rho_c
        notes_parts.append(
            f"continuous via isotropic 1/g_w scaling, "
            f"rho(M_full^c)={rho_c:.4g} at baseline"
        )
    else:
        notes_parts.append(
            f"continuous ceilings unavailable: rho(M_full^c)={rho_c!r}"
        )

    # ── TSO OLTC and shunt: use C3 g_min_required (analytical) ────────
    # The C3 analysis names discrete actuators "OLTC_k" / "Shunt_k" per
    # _discrete_actuator_names() in analysis/stability_analysis.py.
    g_w_tso_oltc_ceil:  float = math.inf
    g_w_tso_shunt_ceil: float = math.inf
    oltc_g_mins:  list[float] = []
    shunt_g_mins: list[float] = []
    for zone_dict in result.c3_discrete.g_min_required.values():
        for name, g_min in zone_dict.items():
            if not math.isfinite(float(g_min)):
                continue
            if name.startswith("OLTC"):
                oltc_g_mins.append(float(g_min))
            elif name.startswith("Shunt"):
                shunt_g_mins.append(float(g_min))
    if oltc_g_mins:
        g_w_tso_oltc_ceil = max(oltc_g_mins)
        notes_parts.append(
            f"g_w_tso_oltc from C3.g_min_required (max over {len(oltc_g_mins)} OLTCs)"
        )
    else:
        notes_parts.append("g_w_tso_oltc: no OLTC actuators in stability result")
    if shunt_g_mins:
        g_w_tso_shunt_ceil = max(shunt_g_mins)
        notes_parts.append(
            f"g_w_tso_shunt from C3.g_min_required (max over {len(shunt_g_mins)} shunts)"
        )
    else:
        notes_parts.append("g_w_tso_shunt: no shunt actuators in stability result")

    # ── DSO OLTC: no analytical bound ─────────────────────────────────
    g_w_dso_oltc_ceil: float = math.inf
    notes_parts.append(
        "g_w_dso_oltc: no analytical bound (DSO OLTCs settle finitely "
        "by Prop. 1.3); ceiling=inf, BO uses fallback_high"
    )

    return Ceilings(
        g_w_der=g_w_der_ceil,
        g_w_pcc=g_w_pcc_ceil,
        g_w_tso_oltc=g_w_tso_oltc_ceil,
        g_w_tso_shunt=g_w_tso_shunt_ceil,
        g_w_dso_der=g_w_dso_der_ceil,
        g_w_dso_oltc=g_w_dso_oltc_ceil,
        g_v=g_v_ceil,
        notes="; ".join(notes_parts),
    )
