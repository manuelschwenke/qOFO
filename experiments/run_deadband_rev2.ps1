# Dead-band selection study -- FULL RE-RUN on corrected sensitivities (rev 2).
#
# Supersedes everything produced before 2026-08-01. The reduced-network
# sensitivity construction was wrong in three independent ways
# (docs/daily_log/08_2026/2026-08-01_reduced_network_fidelity_defect.md):
#
#   * the DSO reductions solved on the WRONG POWER-FLOW BRANCH -- 0.10-0.36 pu
#     from the combined solution, tertiary buses collapsed to 0.0 pu. Near the
#     nose of the PV curve dV/dQ can be wrong in magnitude and even in sign, so
#     these were not slightly-off derivatives but derivatives of a different
#     regime;
#   * the reductions dropped the sub-network's own `internal_aux` buses, so the
#     reduced net was not a reduction of the DSO at all (132 MW of its own
#     injection missing);
#   * every non-slack boundary coupler was pinned at zero active power, forcing
#     a multi-coupler DSO to push its whole real exchange through one
#     transformer; and TSO zones carried generator SETPOINTS rather than actual
#     dispatch under distributed slack (180 MW of phantom injection in zone 0).
#
# After the fixes all 16 DSO reductions reproduce the combined solution to
# 0.000000 pu and TSO zone 0 improved up to 5x. Measured effect on results: at
# 2016-01-05 08:00, delta = 0.005, interface Q moved 0.4410 -> 0.4929 Mvar
# (~12%), which is why the whole study is re-run rather than patched.
#
# Runs carry `sensitivity_reduction_rev = 2`; analysis/deadband_selection.py
# requires it, so rev-1 runs cannot enter the same curves.
#
# ---------------------------------------------------------------------------
# WINDOWS
#
# Five live windows spanning net infeed -117 .. +2200 MW:
#
#   2016-02-22 13:00   -117 MW   the only import-regime hour with capable DER
#   2016-01-05 08:00   +409 MW   mild
#   2016-01-15 03:00   +805 MW   mid
#   2016-12-18 14:00  +1367 MW   stressed
#   2016-05-01 16:00  +2200 MW   NEWLY FEASIBLE -- this window aborted with
#                                LoadflowNotConverged under rev 1 and converges
#                                under rev 2, because removing ~180 MW of
#                                phantom generation from zone 0 was enough to
#                                give the reduced net a solution.
#
# 2016-07-15 03:00 is DEGENERATE (all 44 parks at P = 0, hence zero VDE
# reactive capability, hence delta cannot bind). That is a property of the
# operating point and unaffected by the sensitivity fix, so it gets two
# confirmation cells rather than nine.
#
# ---------------------------------------------------------------------------
# Usage:  powershell -File experiments\run_deadband_rev2.ps1
#         powershell -File experiments\run_deadband_rev2.ps1 -Only 1,2

param(
    [string]   $Scenario = 'rural_700',
    [string[]] $LiveWindows = @('2016-02-22 13:00', '2016-01-05 08:00',
                                '2016-01-15 03:00', '2016-12-18 14:00',
                                '2016-05-01 16:00'),
    [string]   $DegenerateWindow = '2016-07-15 03:00',
    [string[]] $StepWindows = @('2016-01-05 08:00', '2016-12-18 14:00'),
    [string[]] $Deltas = @('0', '0.001', '0.0025', '0.005', '0.0075',
                           '0.01', '0.015', '0.02', '0.03'),
    [double]   $StepTime = 100,
    [double[]] $StepFactors = @(1.10, 1.25),
    [int[]]    $Only = @(1, 2, 3)
)

$ErrorActionPreference = 'Continue'
$PRJ   = Split-Path -Parent $PSScriptRoot
$SWEEP = Join-Path $PSScriptRoot 'run_deadband_sweep.ps1'
if (-not (Test-Path $SWEEP)) { Write-Host "!!! not found: $SWEEP"; exit 2 }
Set-Location $PRJ

function Stamp { Get-Date -Format 'yyyy-MM-dd HH:mm:ss' }

$nLive = $LiveWindows.Count * $Deltas.Count
$nStep = $StepWindows.Count * $Deltas.Count * $StepFactors.Count
Write-Host "=== dead-band REV-2 RE-RUN started $(Stamp) ==="
Write-Host "    phases        : $($Only -join ', ')"
Write-Host "    live windows  : $($LiveWindows.Count) x $($Deltas.Count) deltas = $nLive runs"
Write-Host "    degenerate    : 2 confirmation runs"
Write-Host "    disturbance   : $nStep runs"
Write-Host "    TOTAL         : $($nLive + 2 + $nStep) runs, ~13 min each"

# -- Phase 1: the undisturbed matrix --------------------------------------
# -ContinueOnError: 2016-05-01 16:00 only became feasible today and some of its
# cells may still diverge; one bad cell must not discard the rest.
if ($Only -contains 1) {
    Write-Host "`n>>> PHASE 1: undisturbed matrix ($nLive runs)  $(Stamp)"
    & $SWEEP -Scenario $Scenario -Deltas $Deltas -Windows $LiveWindows `
        -ContinueOnError
    Write-Host ">>> PHASE 1 done (exit=$LASTEXITCODE)  $(Stamp)"
}

# -- Phase 2: confirm the degenerate window is still inert -----------------
if ($Only -contains 2) {
    Write-Host "`n>>> PHASE 2: degenerate-window confirmation (2 runs)  $(Stamp)"
    & $SWEEP -Scenario $Scenario -Deltas @('0.005', '0.015') `
        -Windows @($DegenerateWindow) -ContinueOnError
    Write-Host ">>> PHASE 2 done (exit=$LASTEXITCODE)  $(Stamp)"
}

# -- Phase 3: disturbance rejection ---------------------------------------
if ($Only -contains 3) {
    foreach ($f in $StepFactors) {
        Write-Host "`n>>> PHASE 3: load step x$f " `
                   "($($StepWindows.Count * $Deltas.Count) runs)  $(Stamp)"
        & $SWEEP -Scenario $Scenario -Deltas $Deltas -Windows $StepWindows `
            -LoadStepTime $StepTime -LoadStepFactor $f -ContinueOnError
        Write-Host ">>> PHASE 3 x$f done (exit=$LASTEXITCODE)  $(Stamp)"
    }
}

Write-Host "`n=== REV-2 RE-RUN DONE $(Stamp) ==="
Write-Host "analyse:"
Write-Host "  python -m analysis.deadband_selection"
Write-Host "  python -m analysis.deadband_pareto"
Write-Host "  python -m analysis.deadband_disturbance"
exit 0
