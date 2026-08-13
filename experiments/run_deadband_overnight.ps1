# Dead-band study -- overnight extension matrix (2026-07-31 -> 2026-08-01).
#
# Four phases, ~58 runs, ~12.5 h at the measured ~13 min/run.
#
#   1  delta = 0.001 on the three existing live windows            3 runs
#   2  load-step SMOKE: one run exercising the new code in PF      1 run
#   3  two additional windows, full 9-point delta grid            18 runs
#   4  disturbance rejection: 9 deltas x 2 step sizes x 2 windows 36 runs
#
# ---------------------------------------------------------------------------
# THE TWO ADDITIONAL WINDOWS
#
# The three live windows are all NET EXPORT (+409, +805, +1367 MW) -- the
# voltage-rise regime. Screening all 35,136 quarter-hours for the two gaps:
#
#   2016-02-22 13:00   net  -117.4 MW   DER 2262 MW   44/44 parks live
#     The most import-leaning hour that still has capable DER. A deep-import
#     window with live DER DOES NOT EXIST in this data: DER output and load are
#     correlated, so every hour with DER >= 2200 MW has net infeed > -300 MW.
#     This is the only available point below zero, and it extends the loading
#     axis into the voltage-drop regime that no current window tests.
#
#   2016-05-01 16:00   net +2200.0 MW   DER 3459 MW
#     Extends the export axis above the current maximum while staying below
#     +3259 MW, which is empirically known to diverge in the static leg
#     (2016-07-25 13:00, LoadflowNotConverged after 200 iterations).
#     Still a risk: offline screening does not reproduce runner behaviour.
#     Phase 3 therefore runs -ContinueOnError.
#
# ---------------------------------------------------------------------------
# THE LOAD STEP IS NOT A CONTINGENCY
#
# `contingencies` raises NotImplementedError on the RMS plant, because it
# mutates `net` directly and `net` is only a measurement mirror there. A load
# step needs none of that machinery: it perturbs the interpolated profile
# frame, which BOTH plants already consume through supported paths -- the
# static plant via apply_profiles, the RMS plant via Plant.apply_exogenous
# (EvtLod). No element is switched.
#
# The step is applied AFTER interpolation, at dt_s = 20 s resolution. Stepping
# the source CSV would not work: load_profiles interpolates the 15-minute
# source linearly, which would smear any step into a 15-minute RAMP. Validated
# 2026-07-31: pre-step ratios exactly 1.0, post-step exactly the factor, zero
# intermediate samples.
#
# Step sizes: x1.10 (mild) and x1.25 (severe) on every active-load profile
# (mv_rural_pload, HS4_pload, HS5_pload) -- a system-wide load step. Reactive
# columns are excluded because mv_rural_qload is signed, so scaling it would
# deepen a capacitive injection rather than add load.
#
# Fired at t = 100 s of a 300 s run: 5 dispatch intervals before, 10 after,
# and the 200 s of recovery spans a full 180 s TSO period.
#
# EXPECTATION to test: a tighter dead band should reject the step better,
# because the local droop begins responding at a smaller voltage deviation.
# Note this is the OPPOSITE direction to the profiled-operation result, where
# wider bands tracked the interface better under stress.
#
# ---------------------------------------------------------------------------
# Load-step runs land in the SAME results root with an otherwise identical
# configuration. analysis/deadband_selection.py now carries
# `load_step_time_s: None` in ADMIT so they cannot contaminate the
# profiled-operation curves.
#
# Usage:  powershell -File experiments\run_deadband_overnight.ps1
#         powershell -File experiments\run_deadband_overnight.ps1 -Only 1,3

param(
    [string]   $Scenario = 'rural_700',
    [string[]] $ExistingWindows = @('2016-01-05 08:00', '2016-01-15 03:00',
                                    '2016-12-18 14:00'),
    [string[]] $NewWindows      = @('2016-02-22 13:00', '2016-05-01 16:00'),
    [string[]] $StepWindows     = @('2016-01-05 08:00', '2016-12-18 14:00'),
    [string[]] $FullDeltas      = @('0', '0.001', '0.0025', '0.005', '0.0075',
                                    '0.01', '0.015', '0.02', '0.03'),
    [double]   $StepTime        = 100,
    [double[]] $StepFactors     = @(1.10, 1.25),
    [int[]]    $Only            = @(1, 2, 3, 4)
)

$ErrorActionPreference = 'Continue'
$PRJ   = Split-Path -Parent $PSScriptRoot
$SWEEP = Join-Path $PSScriptRoot 'run_deadband_sweep.ps1'
if (-not (Test-Path $SWEEP)) { Write-Host "!!! not found: $SWEEP"; exit 2 }
Set-Location $PRJ

function Stamp { Get-Date -Format 'yyyy-MM-dd HH:mm:ss' }

Write-Host "=== dead-band OVERNIGHT started $(Stamp) ==="
Write-Host "    phases requested: $($Only -join ', ')"
Write-Host "    new windows     : $($NewWindows -join ' | ')"
Write-Host "    step windows    : $($StepWindows -join ' | ')"
Write-Host "    step            : t=$StepTime s, factors $($StepFactors -join ', ')"

$smokeOk = $true

# -- Phase 1: delta = 0.001 on the proven windows -------------------------
if ($Only -contains 1) {
    Write-Host "`n>>> PHASE 1: delta=0.001 on existing windows (3 runs)  $(Stamp)"
    & $SWEEP -Scenario $Scenario -Deltas @('0.001') -Windows $ExistingWindows
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!!! phase 1 failed on PROVEN windows -- something is wrong; stopping."
        exit 1
    }
    Write-Host ">>> PHASE 1 done  $(Stamp)"
}

# -- Phase 2: load-step smoke test ----------------------------------------
# One run exercising the new --load-step-* path end to end in PowerFactory,
# before 36 runs are committed to it.
if ($Only -contains 2) {
    Write-Host "`n>>> PHASE 2: load-step SMOKE (1 run)  $(Stamp)"
    & $SWEEP -Scenario $Scenario -Deltas @('0.005') -Windows @($StepWindows[0]) `
        -LoadStepTime $StepTime -LoadStepFactor $StepFactors[0]
    if ($LASTEXITCODE -ne 0) {
        $smokeOk = $false
        Write-Host "!!! load-step SMOKE FAILED -- phase 4 will be SKIPPED."
        Write-Host "!!! phases 1 and 3 are unaffected and still worth having."
    } else {
        Write-Host ">>> PHASE 2 smoke passed  $(Stamp)"
    }
}

# -- Phase 3: two additional windows, full grid ---------------------------
# -ContinueOnError: these windows are unproven, and one divergent cell must
# not discard the rest of the night.
if ($Only -contains 3) {
    Write-Host "`n>>> PHASE 3: new windows x $($FullDeltas.Count) deltas " `
               "($($NewWindows.Count * $FullDeltas.Count) runs)  $(Stamp)"
    & $SWEEP -Scenario $Scenario -Deltas $FullDeltas -Windows $NewWindows `
        -ContinueOnError
    Write-Host ">>> PHASE 3 done (exit=$LASTEXITCODE)  $(Stamp)"
}

# -- Phase 4: disturbance rejection ---------------------------------------
if ($Only -contains 4) {
    if (-not $smokeOk) {
        Write-Host "`n>>> PHASE 4 SKIPPED: load-step smoke failed."
    } else {
        foreach ($f in $StepFactors) {
            Write-Host "`n>>> PHASE 4: load step x$f " `
                       "($($StepWindows.Count * $FullDeltas.Count) runs)  $(Stamp)"
            & $SWEEP -Scenario $Scenario -Deltas $FullDeltas -Windows $StepWindows `
                -LoadStepTime $StepTime -LoadStepFactor $f -ContinueOnError
            Write-Host ">>> PHASE 4 x$f done (exit=$LASTEXITCODE)  $(Stamp)"
        }
    }
}

Write-Host "`n=== OVERNIGHT DONE $(Stamp) ==="
Write-Host "analyse:  F:\python_environments\qOFO_clean\python.exe -X utf8 -m analysis.deadband_selection"
Write-Host "          F:\python_environments\qOFO_clean\python.exe -X utf8 -m analysis.deadband_pareto"
exit 0
