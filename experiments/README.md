# Experiments — index and classification

Audit of 2026-07-03 (BME Phase 6). Every numbered script is a distinct,
still-cited study; none is deletable. Naming convention since 2026-07-03:
files belonging to a TSO–TSO horizontal-coordination scheme carry the
scheme name (`VREF` = two-loop ΔV_ref / gradient-exchange coordinator,
`BME` = Boundary Marginal Exchange). Files without a scheme tag belong to
the vertical TSO–DSO cascade line (CIGRE 2026 case study) or to
infrastructure studies.

| Script | Line of work | Role / status |
|---|---|---|
| `000_M_TSO_M_DSO.py` | vertical cascade | Multi-TSO/multi-DSO entry point (IEEE 39, 3 zones); referenced by tuning, observer analysis, diagnostics. **Keep.** |
| `001_S_TSO_S_DSO.py` | vertical cascade | Single-TSO/single-DSO cascade entry point (`runners/cascade.py`). **Keep.** |
| `002_M_TSO_M_DSO_COMPARE.py` | vertical cascade | L0/L1/T0/T1/C control-mode comparison; `make_base_config()` is imported by 003/004/009. **Keep.** |
| `003_S_DSO_CIGRE_2026.py` | CIGRE 2026 paper | Single-DSO study (for Johannes). Paper artefact. **Keep.** |
| `004_LOCAL_VS_FULL_SENS.py` | sensitivity provenance | Full-network vs Ward-reduced local Jacobians; the "experiment-004 trade-off" cited in the BME design (Convention A discussion). **Keep.** |
| `004b_REFRESH_PROOF.py` | sensitivity provenance | Jacobian-staleness proof (frozen vs refreshed `shared_jac`); motivates `refresh_shared_jac_on_tso`, which the BME rung requires. **Keep.** |
| `005_CIGRE_MULTI.py` | CIGRE 2026 paper | V1–V5 ladder, 360-min case study; `make_cigre_config()` is THE shared scenario imported by 006/007/011 and several diagnostics. **Keep — do not rename** (imported as `experiments.005_CIGRE_MULTI` in multiple places). |
| `006_CIGRE_MONTECARLO.py` | CIGRE 2026 paper | Monte-Carlo robustness extension of 005; template for the BME Phase 6 MC campaign. **Keep.** |
| `007_VREF_TIE_COORDINATION.py` | **VREF** (TSO–TSO) | Divergent-schedule case study of the ΔV_ref coordinator (was `007_TIE_COORDINATION.py`). Documents the vref lineage the BME chapter builds on. **Keep.** |
| `008_VREF_MUTUAL_GRADIENT_DEMO.ipynb` | **VREF** (TSO–TSO) | Mutual-gradient (γ-exchange) coordinator demo (was `008_TIE_MUTUAL_GRADIENT_DEMO.ipynb`). Historical evidence for the incommensurability flaw → BME motivation. **Keep.** |
| `009_TSO_LOSS_VREF_SWEEP.py` | TSO loss objective × **VREF** | `tso_g_loss` weight sweep crossed with the vref coordinator (was `009_TSO_LOSS_TIE_SWEEP.py`; results remain in `results/009_loss_tie_sweep/`). **Keep.** |
| `010_VREF_HETEROGENEOUS_STRATEGIES_DEMO.ipynb` | **VREF** (TSO–TSO) | Heterogeneous per-zone strategies under the vref coordinator (was `010_TSO_HETEROGENEOUS_STRATEGIES_DEMO.ipynb`). Documents the long-run degradation / sticky-OLTC evidence → BME discrete hygiene. **Keep.** |
| `011_BME_LADDER.py` | **BME** (TSO–TSO) | Phase 6 evaluation ladder (none / vref / bme / bme_loss / oracle) on the shared 005 scenario, uniform Φ metric, ledger export, evaluation figures (`--plot`). **Active.** |

Support packages: `helpers/` (records, metrics, plant I/O, contingencies),
`runners/` (`cascade.py`, `multi_tso_dso.py`), `paths.py`.

Renames are `git mv` (history preserved). Old names may still appear in
daily-log entries and notebook markdown — those are historical records and
are intentionally left unchanged.
