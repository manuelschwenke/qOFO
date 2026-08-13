# =====================================================================
#  FALSE ACTIVATION on profile drift -- UNDISTURBED runs  (Ch. 9, panel a)
# =====================================================================
#
#  Companion to run_qstep_sweep.ps1.  That script measures how well the local
#  Q(V) layer rejects a disturbance; this one measures the price of a narrow
#  dead band when there is NO disturbance -- how often ordinary profile drift
#  alone pushes a park out of its dead zone and makes the layer act.
#
#  WHY THESE RUNS ARE DIFFERENT FROM THE SWEEP:
#
#  * NO CONTINGENCY.  The whole run is the undisturbed operating point.
#  * THE DEAD BAND IS SET NORMALLY, through --tso-deadband/--dso-deadband,
#    NOT through --qv-deadband-at-contingency.  The sweep keeps the configured
#    band at 0.5 and installs the real one at the step, precisely so the legs
#    share a run-up; here there is no step, so the band must be live from t=0
#    or the run contains nothing to measure.
#  * 600 s, and NO adaptive stepping.  There is no switching transient to
#    resolve, so the 10 ms fixed step is enough, and 600 s gives 30 DS
#    dispatch intervals against 240 s's 12 -- the statistical basis the
#    published panel (a) used.  Cost per run is about the same either way.
#
#  DROOP INDEPENDENCE is ASSUMED, deliberately (user decision 2026-08-06), so
#  this battery runs at ONE droop and its result is quoted for both.  The
#  grounds: while the voltage is INSIDE the dead zone the droop does nothing,
#  so the crossing that triggers activation is open-loop and slope-
#  independent; the slope only changes how hard the layer pulls back
#  afterwards.  The published panel (a) reached the same conclusion
#  empirically -- its title is "both droops".
#
#  ``-SpotCheck`` runs the narrowest bands at a second droop if the assumption
#  ever needs evidence rather than argument.  It is OFF by default.
#
#  Usage:
#     powershell -NoProfile -ExecutionPolicy Bypass -File run_qstep_falseactivation.ps1 `
#         -Windows '2016-02-22 13:00' -Droop 0.05 -SpotCheck
# =====================================================================

param(
    [string]   $Prj      = 'Z:\Python_Projekte\qOFO_GH',
    [string]   $Python   = 'F:\python_environments\qOFO_clean\python.exe',
    [string]   $Droop    = '0.05',
    [string[]] $Windows  = @('2016-02-22 13:00'),
    [string[]] $Deadbands = @('0', '0.0025', '0.005', '0.0075', '0.01',
                              '0.02', '0.03', '0.05', '0.1', '0.2'),
    [string]   $Duration = '600',
    [switch]   $SpotCheck,
    [string]   $SpotDroop = '0.10',
    [string[]] $SpotDeadbands = @('0.0025', '0.01'),
    [string]   $LogDir   = ''
)

$ErrorActionPreference = 'Continue'

if (-not (Test-Path -LiteralPath $Prj))    { Write-Host "!! project not reachable: $Prj"; exit 1 }
if (-not (Test-Path -LiteralPath $Python)) { Write-Host "!! python not found: $Python"; exit 1 }
if (-not $LogDir) { $LogDir = Join-Path $Prj 'results\qstep_sweep\logs' }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force $LogDir | Out-Null }
Set-Location $Prj

# (droop, window, deadband) triples, main set then the spot check
$jobs = @()
foreach ($w in $Windows) {
    foreach ($db in $Deadbands) { $jobs += ,@($Droop, $w, $db) }
}
if ($SpotCheck) {
    foreach ($w in $Windows) {
        foreach ($db in $SpotDeadbands) { $jobs += ,@($SpotDroop, $w, $db) }
    }
}

$total = $jobs.Count
Write-Host "=== false-activation battery (undisturbed) ==="
Write-Host "   windows    $($Windows -join ' | ')"
Write-Host "   deadbands  $($Deadbands -join ' ')"
Write-Host "   droop      $Droop$(if ($SpotCheck) { "  + spot check at $SpotDroop on $($SpotDeadbands -join ',')" })"
Write-Host "   duration   $Duration s, FIXED 10 ms step (no adaptive)"
Write-Host "   $total runs, ~$([math]::Round($total * 15.0 / 60.0, 1)) h"
Write-Host "=== started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

$n = 0
$failed = @()
foreach ($j in $jobs) {
    $m = $j[0]; $w = $j[1]; $db = $j[2]
    $n++
    $wtag  = ($w -replace '[ :\-]', '')
    $tag   = "fa_${wtag}_db$($db -replace '\.', '')_dr$($m -replace '\.', '')"
    $log   = Join-Path $LogDir "$tag.log"

    $pyArgs = @(
        '-X', 'utf8', '-m', 'experiments.run_comparison_rms_cosim_qss',
        '--duration', $Duration,
        '--profiles', '--profile-delivery', 'elmfile',
        '--dso-oltc-switch-cost', '200',
        '--physical-capability',
        '--start-time', $w,
        '--scenario', 'rural_700',
        '--der-slope', $m,
        # live from t=0 -- see the header
        '--tso-deadband', $db, '--dso-deadband', $db,
        '--no-pdf', '--verbose', '1'
    )

    Write-Host "--- [$n/$total] db=$db droop=$m window=$w   $(Get-Date -Format 'HH:mm:ss') ---"
    & $Python @pyArgs *> $log
    $rc = $LASTEXITCODE
    if ($rc -eq 139) {
        Write-Host "    exit=139 (PF exit-segfault, results written)  $(Get-Date -Format 'HH:mm:ss')"
    } elseif ($rc -ne 0) {
        $failed += $tag
        Write-Host "!!! FAILED run $n ($tag): exit=$rc -- continuing; see $log"
    } else {
        Write-Host "    exit=0  $(Get-Date -Format 'HH:mm:ss')"
    }
}

Write-Host ""
Write-Host "=== DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "  results: $Prj\results\rms_phase6_replay"
if ($failed.Count -gt 0) {
    Write-Host "  $($failed.Count) FAILED of ${total}:"; $failed | ForEach-Object { Write-Host "    $_" }
} else { Write-Host "  all $total runs completed" }
