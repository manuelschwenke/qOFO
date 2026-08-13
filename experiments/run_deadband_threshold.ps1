# Design E -- dead-zone threshold characterisation by excitation amplitude.
#
# WHY THIS EXISTS
#
# The profiled-operation sweep cannot define delta*. Its metric (mean interface-Q
# error over a 300 s window) is not a well-defined function of delta at stressed
# operating points: the cascade settles into different equilibria, so the curves
# are non-monotone with ~25% scatter and the argmin lands anywhere between 0.0025
# and 0.03 with no relation to loading (CV 0.715 across five windows).
#
# Design A -- counting actuator motion instead of reading the end state -- is much
# better behaved (monotone in every window, knee CV 0.418), but still gives a
# 3x range rather than a number. Design B, normalising delta by the local voltage
# variability, was tested and REFUTED: it makes the spread slightly worse
# (CV 0.427 vs 0.418), so delta is not simply a threshold in units of sigma_V.
#
# WHAT THIS MEASURES
#
# A dead zone is an amplitude-selective filter: the droop should engage only when
# the voltage deviation exceeds delta. That is a property of the CHARACTERISTIC,
# not of an operating point, so it should be measured by controlled excitation
# rather than inferred from annual windows.
#
# One fixed operating point. For each delta, sweep the amplitude of an exogenous
# load step and measure how much the local droop responds. The expected signature
# is a threshold: negligible DER motion while the induced |dV| < delta, rising
# response above it. Fitting that knee per delta gives the transfer
# characteristic directly, and -- unlike an argmin over a bistable metric -- it is
# a quantity the dead band actually determines.
#
# Read out with:  python -m analysis.deadband_activity   (post-step traverse and
# reversals per run), cross-tabulated by step amplitude instead of by window.
#
# DESIGN
#   window        2016-01-05 08:00 -- the only window whose profiled response is
#                 unambiguous (clean U-curve, single equilibrium, lowest
#                 sigma_V = 0.00128 pu, so the step dominates the background)
#   deltas        0.0025, 0.005, 0.01, 0.02   (4)
#   step factors  1.1 .. 3.0                  (5)  -> 20 runs, ~4.3 h
#
# AMPLITUDE RANGE -- corrected 2026-08-02 after the first batch.
#
# The original set (1.01 .. 1.25) was far too small. Measured induced |dV| at
# the DER terminals: x1.01 -> 0.00026 pu, x1.10 -> 0.00106, x1.25 -> 0.00487.
# Four of those eight amplitudes sat BELOW even the smallest dead band (0.0025)
# and probed nothing, and none could ever exceed 0.01 or 0.02.
#
# A static-plant scan gives the amplitude needed to reach a given deviation:
#
#     x1.10  load 5129 MW   mean |dV| 0.00091   max 0.00202
#     x1.50       5896            0.00272           0.00628
#     x2.00       6854            0.00518           0.01401
#     x2.50       7812            0.01766           0.02822
#     x3.00       8770            0.03969           0.04871
#
# So the set below spans |dV| from ~0.001 to ~0.04 pu and brackets every delta.
#
# Note what that costs physically: exceeding a 0.02 pu dead band needs the
# system load roughly TRIPLED. That is not a plausible disturbance, and it is a
# finding in its own right -- at this operating point delta >= 0.01 means the
# local droop is never activated by any realistic event, i.e. those dead bands
# effectively disable it. The largest amplitudes are included to locate the
# threshold, not because they represent credible operation.
#
# x1.10 and x1.25 at all nine dead bands already exist from the disturbance
# study and are picked up automatically by analysis/deadband_threshold.py, so
# they need not be repeated here.
#
# Usage:  powershell -File experiments\run_deadband_threshold.ps1

param(
    [string]   $Scenario = 'rural_700',
    [string]   $Window   = '2016-01-05 08:00',
    [string[]] $Deltas   = @('0.0025', '0.005', '0.01', '0.02'),
    [double[]] $Factors  = @(1.5, 2.0, 2.5, 3.0),
    [double]   $StepTime = 100
)

$ErrorActionPreference = 'Continue'
$PRJ   = Split-Path -Parent $PSScriptRoot
$SWEEP = Join-Path $PSScriptRoot 'run_deadband_sweep.ps1'
if (-not (Test-Path $SWEEP)) { Write-Host "!!! not found: $SWEEP"; exit 2 }
Set-Location $PRJ

$total = $Deltas.Count * $Factors.Count
Write-Host "=== dead-zone THRESHOLD sweep started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "    window : $Window"
Write-Host "    deltas : $($Deltas -join ', ')"
Write-Host "    factors: $($Factors -join ', ')"
Write-Host "    total  : $total runs (~13 min each, ~$([math]::Round($total * 13 / 60, 1)) h)"

$n = 0
foreach ($f in $Factors) {
    $n++
    Write-Host "`n>>> amplitude $n/$($Factors.Count): x$f   $(Get-Date -Format 'HH:mm:ss')"
    & $SWEEP -Scenario $Scenario -Deltas $Deltas -Windows @($Window) `
        -LoadStepTime $StepTime -LoadStepFactor $f -ContinueOnError
    Write-Host ">>> x$f done (exit=$LASTEXITCODE)  $(Get-Date -Format 'HH:mm:ss')"
}

Write-Host "`n=== THRESHOLD SWEEP DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
exit 0
