# 2D dead-band sweep: delta_TS x delta_DS under a localised TS load step.
#
# Question this answers
# ---------------------
# The 1D study varied ONE dead band applied to every park at both voltage
# levels, so it could not say whether the TS-connected and DS-connected DER
# populations want the same value. This sweep varies them independently and
# reads off the Pareto front over (interface-Q tracking, maximum voltage
# deviation) -- the latter being the design parameter the width is chosen
# against.
#
# Disturbance
# -----------
# A LOCALISED additive load step on the TRANSMISSION side, at bus 41
# (AUX_LOAD|grid_bus18), NOT the uniform multiplicative step used earlier.
# Bus 41 is the only EHV load bus that excites both populations comparably
# (measured open-loop at 2016-01-05 08:00, |dV| at the worst park):
#
#     dP at bus 41     TS parks     DS parks
#        700 MW        0.0082 pu    0.0059 pu     <- mild
#       1100 MW        0.0202 pu    0.0195 pu     <- severe
#
# Bus 42 (AUX_LOAD|grid_bus24) gives a far stronger TS excitation per MW but
# reaches only 0.0028 pu at the DS parks, which would leave delta_DS with no
# leverage and make the second Pareto axis meaningless. Bus 41 is preferred
# for balance, not for severity.
#
# LIMITATION, to be stated with the results: 700/1100 MW is 14%/22% of the
# 4938 MW system load. These amplitudes are NOT credible single contingencies;
# they are chosen to place the open-loop |dV| inside the 0.005-0.02 pu design
# region. The EHV level of this test system is stiff (meshed, high short-circuit
# power), so nothing smaller moves the TS parks appreciably -- the earlier
# UNIFORM step needed ~3900 MW for 0.02 pu, so the localised step is already
# 3.5x more efficient.
#
# Matrix
# ------
# 4 delta_TS x 4 delta_DS x {no step, mild, severe} = 48 runs. The undisturbed
# twins are NOT optional: every rejection metric is referenced to the
# same-delta twin, so an unstepped run is needed per cell.
#
# Cost, measured 2026-08-02 rather than assumed: a quiet cell takes ~13 min
# (run 0273) but a CHATTERING cell takes ~17 min (0275 at delta=0.0025, 0279 at
# delta_DS=0), because a chattering plant costs the solver more. The delta = 0
# row and column are 7 of the 16 dead-band pairs, which puts the matrix at
# ~15 min/run on average, i.e. ~12 h -- not the ~10.5 h a uniform 13 min would
# suggest. Fixed overhead dominates the horizon: a 160 s run still took 12.5 min
# (0278), so shortening -Duration buys much less than it appears to.
#
# Requires the per-level dead-band plumbing added 2026-08-02: before it, the
# RMS plant anchored its Q(V) pre-controllers from the exported snapshot's
# DEFAULT dead band, and the only per-run channel was a single blanket scalar
# -- delta_TS != delta_DS was not representable. --tso-deadband/--dso-deadband
# clear that blanket automatically.
#
# Usage:  powershell -File experiments\run_deadband_2d.ps1
#         powershell -File experiments\run_deadband_2d.ps1 -Amplitudes 1100

param(
    [string]   $Scenario   = 'rural_700',
    [string[]] $DeltasTS   = @('0', '0.005', '0.01', '0.02'),
    [string[]] $DeltasDS   = @('0', '0.005', '0.01', '0.02'),
    [string]   $Window     = '2016-01-05 08:00',
    [int]      $Duration   = 300,
    [int]      $StepBus    = 41,
    [double]   $StepTime   = 100,
    # 0 = the undisturbed twin. Keep it first: if the night is cut short, the
    # twins are what every other cell is measured against.
    [double[]] $Amplitudes = @(0, 700, 1100),
    [string]   $Python     = 'F:\python_environments\qOFO_clean\python.exe',
    [string]   $LogDir     = '',
    [switch]   $FailFast
)

$ErrorActionPreference = 'Continue'

$PRJ = Split-Path -Parent $PSScriptRoot
if (-not $LogDir) { $LogDir = Join-Path $PRJ 'results\deadband_2d\logs' }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force $LogDir | Out-Null }
if (-not (Test-Path $Python)) { Write-Host "!!! python not found: $Python"; exit 2 }
Set-Location $PRJ

# Format INVARIANTLY: this shell runs a de-DE culture and .ToString() would
# emit '1100,0', which float() rejects. Interpolation "$x" is invariant, but
# the difference is invisible at the call site, so pin it explicitly.
$inv  = [System.Globalization.CultureInfo]::InvariantCulture
$tStr = $StepTime.ToString($inv)

$wtag  = ($Window -replace '[ :\-]', '')
$total = $DeltasTS.Count * $DeltasDS.Count * $Amplitudes.Count
$n = 0
$failed = @()

Write-Host "=== 2D dead-band sweep started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "    scenario=$Scenario  window=$Window"
Write-Host "    delta_TS = $($DeltasTS -join ' ')"
Write-Host "    delta_DS = $($DeltasDS -join ' ')"
Write-Host "    step: bus $StepBus at t=$tStr s, dP in $($Amplitudes -join ' ') MW  (0 = twin)"
Write-Host "    $total runs -> $LogDir"

foreach ($amp in $Amplitudes) {
    $aStr = $amp.ToString($inv)
    if ($amp -le 0) {
        $stepTag  = 'nostep'
        $stepArgs = @()
    } else {
        $stepTag  = "b{0}p{1}" -f $StepBus, ([int][math]::Round($amp))
        $stepArgs = @('--load-step-time', $tStr,
                      '--load-step-bus', "$StepBus",
                      '--load-step-delta-mw', $aStr)
    }
    foreach ($ts in $DeltasTS) {
        foreach ($ds in $DeltasDS) {
            $n++
            $tag = "{0}_{1}_ts{2}_ds{3}_{4}" -f $Scenario, $wtag,
                   ($ts -replace '\.', ''), ($ds -replace '\.', ''), $stepTag
            $logFile = Join-Path $LogDir "$tag.log"
            Write-Host "--- [$n/$total] dTS=$ts dDS=$ds $stepTag   $(Get-Date -Format 'HH:mm:ss') ---"

            $pyArgs = @(
                '-X', 'utf8', '-m', 'experiments.run_comparison_rms_cosim_qss',
                '--duration', "$Duration",
                '--profiles', '--profile-delivery', 'elmfile',
                '--dso-oltc-switch-cost', '200',
                '--physical-capability',
                '--tso-deadband', $ts,
                '--dso-deadband', $ds,
                '--start-time', $Window,
                '--scenario', $Scenario,
                '--no-pdf', '--verbose', '1'
            ) + $stepArgs

            # *> captures every stream; read $LASTEXITCODE, not $?, because a
            # native command's stderr makes $? false in PS 5.1 even on success.
            & $Python @pyArgs *> $logFile
            $rc = $LASTEXITCODE

            Write-Host "    exit=$rc  $(Get-Date -Format 'HH:mm:ss')"
            if ($rc -ne 0) {
                $failed += $tag
                if ($FailFast) {
                    Write-Host "!!! ABORTING at run $n ($tag): exit=$rc"
                    Write-Host "!!! see $logFile"
                    exit 1
                }
                Write-Host "!!! FAILED run $n ($tag): exit=$rc -- continuing"
                Write-Host "!!! see $logFile"
            }
        }
    }
}

if ($failed.Count) {
    Write-Host ""
    Write-Host "!!! $($failed.Count) of $total run(s) FAILED:"
    $failed | ForEach-Object { Write-Host "      $_" }
}

Write-Host "=== 2D sweep DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "analyse:  & '$Python' -X utf8 -m analysis.deadband_2d"
