"""Unit tests for ``tuning/ceilings.py``."""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path

import pytest

from configs.multi_tso_config import MultiTSOConfig
from tuning._types import Ceilings
from tuning.ceilings import _cache_key, compute_ceilings


# ---------------------------------------------------------------------------
# 1. Cache-key determinism and selectivity
# ---------------------------------------------------------------------------

def test_cache_key_deterministic(baseline_cfg: MultiTSOConfig) -> None:
    """Same config → same key.  Changing ``g_v`` changes the key.
    Changing ``g_w_der`` does not (g_w is excluded from the hash)."""
    k_a = _cache_key(baseline_cfg)
    k_b = _cache_key(baseline_cfg)
    assert k_a == k_b, "same config should yield the same key"

    # g_v IS hashed → key changes
    diff_g_v = dataclasses.replace(baseline_cfg, g_v=baseline_cfg.g_v * 2.0)
    assert _cache_key(diff_g_v) != k_a, "g_v change must change the key"

    # g_w_der is NOT hashed → key stable
    diff_g_w_der = dataclasses.replace(baseline_cfg, g_w_der=baseline_cfg.g_w_der * 10.0)
    assert _cache_key(diff_g_w_der) == k_a, "g_w_der must not affect the key"

    # Other g_w_* are also excluded
    diff_g_w_pcc = dataclasses.replace(baseline_cfg, g_w_pcc=baseline_cfg.g_w_pcc + 1.0)
    diff_g_w_tso_oltc = dataclasses.replace(
        baseline_cfg, g_w_tso_oltc=baseline_cfg.g_w_tso_oltc + 1.0
    )
    diff_g_w_dso_der = dataclasses.replace(
        baseline_cfg, g_w_dso_der=baseline_cfg.g_w_dso_der + 1.0
    )
    diff_g_w_dso_oltc = dataclasses.replace(
        baseline_cfg, g_w_dso_oltc=baseline_cfg.g_w_dso_oltc + 1.0
    )
    for c in (diff_g_w_pcc, diff_g_w_tso_oltc, diff_g_w_dso_der, diff_g_w_dso_oltc):
        assert _cache_key(c) == k_a, "g_w_* fields must not affect the key"


# ---------------------------------------------------------------------------
# 2. Ceilings dataclass: frozen + as_dict
# ---------------------------------------------------------------------------

def test_ceilings_dataclass_immutable() -> None:
    c = Ceilings(
        g_w_der=1.0, g_w_pcc=2.0, g_w_tso_oltc=3.0, g_w_tso_shunt=3.5,
        g_w_dso_der=4.0, g_w_dso_oltc=5.0, g_v=6.0,
        notes="test",
    )
    # frozen: assignment raises FrozenInstanceError
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.g_w_der = 99.0  # type: ignore[misc]

    d = c.as_dict()
    assert set(d.keys()) == {
        "g_w_der", "g_w_pcc", "g_w_tso_oltc", "g_w_tso_shunt",
        "g_w_dso_der", "g_w_dso_oltc", "g_v",
    }
    for v in d.values():
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# 3. Smoke test (slow): real one-step simulation
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_compute_ceilings_smoke(
    baseline_cfg: MultiTSOConfig,
    tmp_path: Path,
) -> None:
    """Run :func:`compute_ceilings` end-to-end with caching disabled and
    verify the contract:

    * returns a :class:`Ceilings` instance
    * every numeric field is positive (or ``inf``)
    * the cache file is created on disk for the next call
    """
    cache_dir = tmp_path / "ceilings_cache"
    ceilings = compute_ceilings(
        baseline_cfg, use_cache=False, cache_dir=cache_dir,
    )

    assert isinstance(ceilings, Ceilings)
    for name, value in ceilings.as_dict().items():
        assert isinstance(value, float)
        assert math.isnan(value) is False, f"{name} must not be NaN"
        assert value > 0.0, f"{name}={value} must be positive (or inf)"

    cached_files = list(cache_dir.glob("ceilings_*.json"))
    assert len(cached_files) == 1, (
        f"expected exactly one cache file, found {len(cached_files)}"
    )


# ---------------------------------------------------------------------------
# 4. Cache hit on second call
# ---------------------------------------------------------------------------

def test_compute_ceilings_uses_cache(
    baseline_cfg: MultiTSOConfig,
    tmp_path: Path,
) -> None:
    """Pre-populate the cache, then verify that a second
    ``compute_ceilings`` call returns the same object quickly without
    invoking the simulator."""
    cache_dir = tmp_path / "ceilings_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    seed = Ceilings(
        g_w_der=1.5, g_w_pcc=2.5, g_w_tso_oltc=3.5, g_w_tso_shunt=3.75,
        g_w_dso_der=4.5, g_w_dso_oltc=math.inf, g_v=7.5,
        notes="seeded for cache test",
    )
    cache_file = cache_dir / f"ceilings_{_cache_key(baseline_cfg)}.json"
    import json
    with cache_file.open("w") as f:
        json.dump(dataclasses.asdict(seed), f, default=str)

    t0 = time.monotonic()
    out = compute_ceilings(baseline_cfg, use_cache=True, cache_dir=cache_dir)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.5, (
        f"cached compute_ceilings took {elapsed:.3f}s; should be << 0.1s "
        "but giving headroom for filesystem latency"
    )
    assert out == seed
