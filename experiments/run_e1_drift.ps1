# =====================================================================
#  E1 -- DEAD-BAND LOWER BOUND / PROFILE-DRIFT INSTRUMENT   (Ch. 9)
# =====================================================================
#
#  Two UNDISTURBED RMS runs that measure how far the park terminal
#  voltages drift between two consecutive OFO dispatches while the local
#  Q(V) droop layer is PROVABLY out of the loop.  That drift is the lower
#  bound on the droop's dead-band half-width: the band must be wide enough
#  that ordinary profile movement is left to the OFO's gradient descent.
#
#  DIFFERENCE FROM run_qstep_falseactivation.ps1
#  ---------------------------------------------
#  That battery installs the CANDIDATE band live from t = 0 and measures
#  activation directly, one run per candidate.  Here the band is pinned at
#  0.5 pu -- far beyond any voltage the plant can reach -- so the local law
#  is inert for the whole run and the recorded trajectory is the OPEN-LOOP
#  profile drift.  Every candidate half-width is then obtained OFFLINE by
#  thresholding the same drift samples, so all ten candidates share one
#  trajectory and cannot differ through the closed loop.
#
#  Consequences of the pinned band:
#    * `--qv-deadband-at-contingency` is NOT passed.  There is no
#      contingency, and passing it would make the band change mid-run.
#    * ONE droop slope only (0.05).  The slope only scales the response
#      once the band is left; with the band at 0.5 it is never left, so a
#      second slope would reproduce the first bit for bit.
#
#  HORIZON.  3600 s, against the false-activation battery's 600 s: 19
#  usable TSO windows per park instead of 3.  `data/profiles.csv` is
#  resampled to 20 s over the whole of 2016, so both start instants have
#  the required 181 rows; nothing is shortened.
#
#  SOLVER.  Fixed 10 ms step (NO --adaptive-step), stride 10 -> 0.1 s
#  samples, exactly as runs 0498-0536.  Adaptive stepping exists to resolve
#  a switching transient; these runs contain none.
#
#  Everything else -- scenario, DSO_3 multipliers, physical VDE-AR-N-4120
#  capability, OLTC switch cost, rev-2 sensitivities, profile delivery --
#  is the dead-band campaign baseline.
#
#  Runs land in results\rms_phase6_replay as 0537, 0538 (the counter is
#  held at 0536 by the anchor directory there) and are MOVED afterwards to
#  results\deadband_droop_e1_drift\data.  results\deadband_droop\data is
#  the q-step campaign and is not touched.
#
#  Usage:
#     powershell -NoProfile -ExecutionPolicy Bypass -File run_e1_drift.ps1
# =====================================================================

param(
    [string]   $Prj      = 'Z:\Python_Projekte\qOFO_GH',
    [string]   $Python   = 'F:\python_environments\qOFO_clean\python.exe',
    [string]   $Droop    = '0.05',
    [string]   $Deadband = '0.5',
    [string[]] $Windows  = @('2016-02-22 13:00', '2016-01-05 08:00'),
    [string]   $Duration = '3600',
    [string]   $LogDir   = ''
)

$ErrorActionPreference = 'Continue'

if (-not (Test-Path -LiteralPath $Prj))    { Write-Host "!! project not reachable: $Prj"; exit 1 }
if (-not (Test-Path -LiteralPath $Python)) { Write-Host "!! python not found: $Python"; exit 1 }
if (-not $LogDir) { $LogDir = Join-Path $Prj 'results\deadband_droop_e1_drift\logs' }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force $LogDir | Out-Null }
Set-Location $Prj

$total = $Windows.Count
Write-Host "=== E1 profile-drift instrument (undisturbed, band pinned open) ==="
Write-Host "   windows    $($Windows -join ' | ')"
Write-Host "   deadband   $Deadband pu on BOTH levels, live from t=0"
Write-Host "   droop      $Droop  (single slope: the band is never left)"
Write-Host "   duration   $Duration s, FIXED 10 ms step (no adaptive)"
Write-Host "   $total runs"
Write-Host "=== started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

$n = 0
$failed = @()
foreach ($w in $Windows) {
    $n++
    $wtag = ($w -replace '[ :\-]', '')
    $tag  = "e1_${wtag}_db$($Deadband -replace '\.', '')_dr$($Droop -replace '\.', '')"
    $log  = Join-Path $LogDir "$tag.log"

    # -u: the runner's per-step progress line carries no flush=True, so a
    # block-buffered stdout hides ~150 step lines at a time -- on a 180-step
    # run that is the whole run.  Unbuffered output costs nothing here and
    # makes the log readable while the run is in flight.
    $pyArgs = @(
        '-u', '-X', 'utf8', '-m', 'experiments.run_comparison_rms_cosim_qss',
        '--duration', $Duration,
        '--profiles', '--profile-delivery', 'elmfile',
        '--dso-oltc-switch-cost', '200',
        '--physical-capability',
        '--start-time', $w,
        '--scenario', 'rural_700',
        '--der-slope', $Droop,
        # live from t=0 and pinned open for the whole run
        '--tso-deadband', $Deadband, '--dso-deadband', $Deadband,
        '--no-pdf', '--verbose', '1'
    )

    Write-Host "--- [$n/$total] window=$w db=$Deadband droop=$Droop   $(Get-Date -Format 'HH:mm:ss') ---"
    & $Python @pyArgs *> $log
    $rc = $LASTEXITCODE
    if ($rc -eq 139) {
        Write-Host "    exit=139 (PF exit-segfault, results written)  $(Get-Date -Format 'HH:mm:ss')"
    } elseif ($rc -eq 1) {
        # main() returns 1 whenever Gate E does not certify the run.  Over a
        # 3600 s profiled horizon the 2 % settling gate is expected to fail on
        # some intervals; that is a statement about tracking, not about the
        # drift measurement, which reads the raw RMS trajectory.
        Write-Host "    exit=1 (Gate E not certified -- expected on a 3600 s profiled run)  $(Get-Date -Format 'HH:mm:ss')"
    } elseif ($rc -ne 0) {
        $failed += $tag
        Write-Host "!!! FAILED run $n ($tag): exit=$rc -- continuing; see $log"
    } else {
        Write-Host "    exit=0  $(Get-Date -Format 'HH:mm:ss')"
    }
}

Write-Host ""
Write-Host "=== DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "  results: $Prj\results\rms_phase6_replay  (move to deadband_droop_e1_drift\data)"
if ($failed.Count -gt 0) {
    Write-Host "  $($failed.Count) FAILED of ${total}:"; $failed | ForEach-Object { Write-Host "    $_" }
} else { Write-Host "  all $total runs completed" }
