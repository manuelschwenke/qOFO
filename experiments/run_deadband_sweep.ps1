# Dead-band selection sweep -- thesis Ch. 8 sec. 2.
#
# Sweeps the DER Q(V) dead-zone half-width delta across several profile windows, so
# the interior optimum can be shown to be (or not to be) a property of the system
# rather than of one operating point. Analyse with:
#
#     python -m analysis.deadband_selection
#
# WINDOWS are stratified on the QSS per-interval voltage excursion from the Tier-1
# season screening (measured on an older topology, so used ONLY to choose a spread
# of operating points, never quoted as a result):
#     2016-01-05 08:00   0.00828 pu   1.0x   low (original reference window)
#     2016-01-15 03:00   0.01573 pu   1.9x   mid
#     2016-07-15 03:00   0.02051 pu   2.5x   high / annual maximum
#
# SCENARIO is passed EXPLICITLY on every run. The config default is base_410, and
# relying on it is what silently produced one base_410 run in the middle of a
# rural_700 series on 2026-07-29. base_410 and rural_700 differ in installed DSO DER
# (410 vs 700 MW per DSO) and their results are NOT comparable. analysis/
# deadband_selection.py additionally filters on the scenario each run recorded.
#
# delta = 0 is NOT in the default list, and that is now known to be a mistake.
# The exclusion rested on a recorded cost of ~4.9 h per delta = 0 run. VOID:
# measured 2026-07-31, the three delta = 0 runs took 12.3 / 12.8 / 13.2 min, i.e.
# the same as every other cell and ~23x below the estimate. delta = 0 is also the
# most informative point in the study -- interface Q is 3x its optimum there in
# every live window, which is what makes the two-sided argument evidential.
# ADD '0' to -Deltas. It is an ordinary ~13 min cell.
#
# 3 windows x 5 dead bands = 15 runs, ~28 min each ~= 7 h.
# FAIL-FAST: aborts on the first run with a non-zero exit code, so a broken
# configuration does not burn the whole night.
#
# Usage:  powershell -File experiments\run_deadband_sweep.ps1
#         powershell -File experiments\run_deadband_sweep.ps1 -Scenario base_410

param(
    [string]   $Scenario = 'rural_700',
    [string[]] $Deltas   = @('0.0025', '0.005', '0.0075', '0.01', '0.015'),
    [string[]] $Windows  = @('2016-01-05 08:00', '2016-01-15 03:00', '2016-07-15 03:00'),
    [int]      $Duration = 300,
    [string]   $Python   = 'F:\python_environments\qOFO_clean\python.exe',
    [string]   $LogDir   = '',
    # -- exogenous load step (disturbance-rejection studies) ---------------
    # Perturbs the interpolated profile frame, so it reaches BOTH plants
    # through supported paths (static: apply_profiles, RMS: EvtLod). This is
    # NOT a contingency; no element is switched.
    [double]   $LoadStepTime   = -1,      # <0 = no step
    [double]   $LoadStepFactor = 1.0,
    # -- unattended operation ----------------------------------------------
    # Default is fail-fast: a broken configuration must not burn the night.
    # For an overnight matrix spanning UNPROVEN windows, one divergent cell
    # should not discard every later cell -- pass -ContinueOnError there.
    [switch]   $ContinueOnError
)

$ErrorActionPreference = 'Continue'

$PRJ = Split-Path -Parent $PSScriptRoot
if (-not $LogDir) { $LogDir = Join-Path $PRJ 'results\deadband_selection\logs' }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force $LogDir | Out-Null }
if (-not (Test-Path $Python)) { Write-Host "!!! python not found: $Python"; exit 2 }
Set-Location $PRJ

$n = 0
$total = $Windows.Count * $Deltas.Count
Write-Host "=== dead-band sweep started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "    scenario=$Scenario  $($Windows.Count) windows x $($Deltas.Count) dead bands = $total runs"
Write-Host "    logs -> $LogDir"

$failed = @()
$stepTag = ''
$stepArgs = @()
if ($LoadStepTime -ge 0) {
    # Tag by percent so db0005_ls110 and db0005_ls125 cannot collide with the
    # undisturbed db0005.
    # Format INVARIANTLY and explicitly. This shell runs a de-DE culture, and
    # PowerShell is inconsistent about which culture it uses:
    #     "$x"          -> "1.25"  (invariant)   <- what interpolation gives
    #     $x.ToString() -> "1,25"  (current)     <- float("1,25") raises
    #     $a -join ', ' -> "1,25"  (current, via ToString)
    # Interpolation happens to be safe, so this is belt-and-braces rather than
    # a fix for an observed failure -- but the difference is invisible at the
    # call site, and a later edit to .ToString() would break python silently.
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    $fStr = $LoadStepFactor.ToString($inv)
    $tStr = $LoadStepTime.ToString($inv)
    $stepTag  = "_ls{0}" -f ([int][math]::Round($LoadStepFactor * 100))
    $stepArgs = @('--load-step-time', $tStr, '--load-step-factor', $fStr)
    Write-Host "    load step: x$fStr at t=$tStr s  (tag $stepTag)"
}

foreach ($w in $Windows) {
    $wtag = ($w -replace '[ :\-]', '')
    foreach ($db in $Deltas) {
        $n++
        $tag = "{0}_{1}_db{2}{3}" -f $Scenario, $wtag, ($db -replace '\.', ''), $stepTag
        $logFile = Join-Path $LogDir "$tag.log"
        Write-Host "--- [$n/$total] $w  delta=$db   $(Get-Date -Format 'HH:mm:ss') ---"

        # run_comparison_rms_cosim_qss, NOT run_rms_cosim. The RMS-only entry
        # point writes to results\rms_cosim\ and stores runner_static=None, and
        # analysis\deadband_selection.py reads results\rms_phase6_replay\ and
        # rejects any run whose config.json has no runner_static block -- such a
        # run could never enter the study. This module is also what the
        # deprecated run_rms_phase6_replay shim delegates to, so it is the same
        # code path that produced run 0080. (Renamed 2026-07-31; the old name
        # was a misnomer, nothing was ever replayed.)
        $pyArgs = @(
            '-X', 'utf8', '-m', 'experiments.run_comparison_rms_cosim_qss',
            '--duration', "$Duration",
            '--profiles', '--profile-delivery', 'elmfile',
            '--dso-oltc-switch-cost', '200',
            '--physical-capability',
            '--der-deadband', $db,
            '--start-time', $w,
            '--scenario', $Scenario,
            '--no-pdf', '--verbose', '1'
        ) + $stepArgs
        # *> captures every stream into the log; exit status is read from
        # $LASTEXITCODE, not $?, because a native command's stderr makes $? false
        # in PowerShell 5.1 even on success.
        & $Python @pyArgs *> $logFile
        $rc = $LASTEXITCODE

        Write-Host "    exit=$rc  $(Get-Date -Format 'HH:mm:ss')"
        if ($rc -ne 0) {
            $failed += $tag
            if ($ContinueOnError) {
                Write-Host "!!! FAILED run $n ($tag): exit=$rc -- continuing"
                Write-Host "!!! see $logFile"
            } else {
                Write-Host "!!! ABORTING at run $n ($tag): exit=$rc"
                Write-Host "!!! see $logFile"
                exit 1
            }
        }
    }
}

if ($failed.Count) {
    Write-Host ""
    Write-Host "!!! $($failed.Count) of $total run(s) FAILED:"
    $failed | ForEach-Object { Write-Host "      $_" }
}

Write-Host "=== sweep DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "analyse:  & '$Python' -X utf8 -m analysis.deadband_selection --scenario $Scenario"
