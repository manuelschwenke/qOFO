# Gate-1 record — IEEE39_qOFO template (read-only probe)

**Verdict (2026-07-17, PDF comparison): PASS — template pristine.**
All 39 bus voltages/angles below match Table 10 of
`docs/39_Bus_New_England_System.pdf` (Rev. 2) to the full printed
precision: max |Δu| = 0.5·10⁻⁴ pu, max |Δφ| = 0.5·10⁻² deg — pure rounding
of the printed 4/2 decimals. Gate 1 is complete.

**G 05 — corrected analysis (2026-07-17, after the ngnum discovery).**
The Phase-2 API probe revealed `G 05.ngnum = 2`: the template models G 05
as **two parallel 300 MVA units** (2 × 254 MW = 508 MW). That overturns the
first reading of Tables 7/8: plant kinetic energy is 2·4.333·300 =
2600 MVA·s = exactly the literature 26 s · 100 MVA — **H is correct as
shipped**. The actual inconsistency is in the per-unit reactances: each
half-plant unit should carry literature·(600/100) = e.g. xd 4.02, but
ships with literature·(300/100) = 2.01 — **every reactance is half its
correct value** (pattern verified on xd, xq, x′d, x′q, x″, xl), i.e. the
G 05 plant is twice as stiff as the IEEE task-force data. Fix, if
base-scenario RMS fidelity is ever needed: double all `Type Gen 05`
reactances (do **not** touch H, do not change Sr). Deferred — G 05 is out
of service in wind_replace. Load-flow parity is unaffected either way
(machine impedances do not enter the balanced LDF).

- Recorded: 2026-07-17 23:04
- Project: `\mschwenke.IntUser\qOFO\IEEE39_qOFO.IntPrj`
- Python: `3.12.13` (`F:\python_environments\qOFO_clean\python.exe`)

## Study cases

- `1. Power Flow`
- `2.1 Simulation Fault Bus 16 Stable`
- `2.2 Simulation Fault Bus 16 Unstable`
- `2.3 Simulation Fault Bus 31 Stable`
- `2.4 Simulation Fault Bus 31 Unstable`
- `2.5 Simulation Fault Line 2-3 Stable`
- `2.6 Simulation Fault Line 2-3 Unstable`
- `3. Small Signal Analysis (Eigenvalues)`
- `4. EMT Simulation Fault Bus 03`

## Synchronous machines (ElmSym -> TypSym)

| machine | outserv | type | sgn | ugn | h | cosn |
|---|---|---|---|---|---|---|
| `G 01` | 0 | `Type Gen 01` | 10000.0 | 345.0 | 5.0 | 0.8500000238418579 |
| `G 02` | 0 | `Type Gen 02` | 700.0 | 16.5 | 4.328999996185303 | 0.8500000238418579 |
| `G 03` | 0 | `Type Gen 03` | 800.0 | 16.5 | 4.474999904632568 | 0.8500000238418579 |
| `G 04` | 0 | `Type Gen 04` | 800.0 | 16.5 | 3.5749998092651367 | 0.8500000238418579 |
| `G 05` | 0 | `Type Gen 05` | 300.0 | 16.5 | 4.333000183105469 | 0.8500000238418579 |
| `G 06` | 0 | `Type Gen 06` | 800.0 | 16.5 | 4.349999904632568 | 0.8500000238418579 |
| `G 07` | 0 | `Type Gen 07` | 700.0 | 16.5 | 3.7710001468658447 | 0.8500000238418579 |
| `G 08` | 0 | `Type Gen 08` | 700.0 | 16.5 | 3.4710001945495605 | 0.8500000238418579 |
| `G 09` | 0 | `Type Gen 09` | 1000.0 | 16.5 | 3.450000047683716 | 0.8500000238418579 |
| `G 10` | 0 | `Type Gen 10` | 1000.0 | 16.5 | 4.199999809265137 | 0.8500000238418579 |

(H reconciliation check: the build plan flags G 05 — the PDF prints Sr = 300 MVA with H = 4.333 s on machine base, which matches H = 26 s on 100 MVA base only for Sr = 600 MVA.)

## Template load flow (ComLdf of the active study case)

Converged. Bus voltages for the manual Table-10 check:

| bus | u [pu] | phi [deg] |
|---|---|---|
| `Bus 01` | 1.04736 | -8.4387 |
| `Bus 02` | 1.04874 | -5.7538 |
| `Bus 03` | 1.03017 | -8.5985 |
| `Bus 04` | 1.00386 | -9.6067 |
| `Bus 05` | 1.00531 | -8.6119 |
| `Bus 06` | 1.00767 | -7.9497 |
| `Bus 07` | 0.99700 | -10.1238 |
| `Bus 08` | 0.99602 | -10.6154 |
| `Bus 09` | 1.02823 | -10.3220 |
| `Bus 10` | 1.01715 | -5.4271 |
| `Bus 11` | 1.01269 | -6.2843 |
| `Bus 12` | 1.00015 | -6.2436 |
| `Bus 13` | 1.01431 | -6.0977 |
| `Bus 14` | 1.01173 | -7.6564 |
| `Bus 15` | 1.01538 | -7.7361 |
| `Bus 16` | 1.03177 | -6.1875 |
| `Bus 17` | 1.03356 | -7.3013 |
| `Bus 18` | 1.03093 | -8.2239 |
| `Bus 19` | 1.04986 | -1.0228 |
| `Bus 20` | 0.99118 | -2.0147 |
| `Bus 21` | 1.03176 | -3.7805 |
| `Bus 22` | 1.04979 | 0.6683 |
| `Bus 23` | 1.04479 | 0.4700 |
| `Bus 24` | 1.03731 | -6.0679 |
| `Bus 25` | 1.05757 | -4.3634 |
| `Bus 26` | 1.05208 | -5.5267 |
| `Bus 27` | 1.03774 | -7.4954 |
| `Bus 28` | 1.05012 | -2.0149 |
| `Bus 29` | 1.04994 | 0.7444 |
| `Bus 30` | 1.04750 | -3.3340 |
| `Bus 31` | 0.98200 | 0.0000 |
| `Bus 32` | 0.98310 | 2.5690 |
| `Bus 33` | 0.99720 | 4.1947 |
| `Bus 34` | 1.01230 | 3.1750 |
| `Bus 35` | 1.04930 | 5.6301 |
| `Bus 36` | 1.06350 | 8.3229 |
| `Bus 37` | 1.02780 | 2.4211 |
| `Bus 38` | 1.02650 | 7.8077 |
| `Bus 39` | 1.03000 | -10.0530 |

## ComLdf option flags (parity-relevant, as found)

- `ComLdf.iopt_net` = `0`
- `ComLdf.iopt_pq` = `0`
- `ComLdf.iopt_at` = `0`
- `ComLdf.iopt_asht` = `0`
- `ComLdf.iopt_lim` = `0`
- `ComLdf.iopt_plim` = `0`
- `ComLdf.i_power` = `1`
- `ComLdf.iopt_sim` = `0`
- `ComLdf.errlf` = `1.0`
