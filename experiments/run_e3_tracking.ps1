# =====================================================================
#  E3 -- TRACKING ACCURACY vs DEAD-BAND HALF-WIDTH  (Ch. 9, dead-band bound)
# =====================================================================
#
#  WHAT THIS MEASURES
#
#  How much the local Q(V) droop layer degrades the OFO's tracking accuracy
#  as its dead band narrows.  A narrow band lets the droop answer voltage
#  movement that the OFO is itself producing, so the two layers compete.
#  Expected shape: tracking accuracy flat above some half-width, degrading
#  below it.  That knee is the LOWER BOUND on the dead band, measured on the
#  harm itself rather than on the false-activation proxy of E1.
#
#  32 cells = 8 half-widths x 2 droop slopes x 2 operating windows.
#
#  THE DEAD BAND IS LIVE FROM t = 0.  This is the opposite choice from
#  run_qstep_sweep.ps1 and the same choice as run_qstep_falseactivation.ps1:
#  the band is set through --tso-deadband/--dso-deadband and
#  --qv-deadband-at-contingency is NOT used.  There is no disturbance here,
#  so a band installed at a contingency would never be installed at all and
#  the run would contain nothing to measure.
#
#  CONSEQUENCE, and it is deliberate: --tso-deadband/--dso-deadband also feed
#  the CONTROLLERS and the STATIC plant, so the legs do NOT share a common
#  run-up -- each leg is its own closed-loop system from t = 0.  That is
#  correct for this experiment, but it means legs may be compared ONLY through
#  aggregate metrics.  Pointwise trajectory differencing against a shared
#  baseline is invalid here and must not be done.
#
#  NO CONTINGENCY.  Undisturbed runs throughout.
#
#  900 s and NO adaptive stepping.  There is no switching transient to
#  resolve, so the fixed 10 ms step is enough (same reasoning as E1).  900 s
#  gives 45 DS dispatch instants and 5 TSO instants; discarding the first TSO
#  period as initialisation transient leaves 36 DS and 4 TSO instants.
#
#  0.2 pu is the droop-out-of-loop reference leg: the band is so wide that
#  ordinary profile drift never crosses it, so the local layer never acts and
#  the run measures the OFO's tracking accuracy on its own.
#
#  ORDER is by complete cell (window, droop): all eight half-widths of one
#  cell before moving on, so an interrupted campaign leaves whole, analysable
#  cells rather than 32 scattered fragments.
#
#  SHARDING.  The 32 cells are fully independent and shard on any axis via
#  -Windows / -Droops / -Deadbands.  Whatever shards, the profile day,
#  -StartTimes, -Duration, the scenario and every other baseline flag below
#  MUST be identical across shards or the results will not pool.
#
#  RESULT DIRECTORIES land in results/rms_phase6_replay (that is what
#  new_run_dir does) and are moved to results/deadband_droop_e3_tracking/data
#  afterwards by finalize_e3_runs.py.  Cell identity is recoverable from each
#  run's own config.json -- never from these logs.
#
#  Usage:
#     powershell -NoProfile -ExecutionPolicy Bypass -File run_e3_tracking.ps1
#     powershell ... -File run_e3_tracking.ps1 -Droops 0.05      # one shard
# =====================================================================

param(
    # NOTE: the project share is mounted on a DIFFERENT DRIVE LETTER per
    # account -- V: on ms_admin, Z: on mschwenke.  A wrong letter makes the
    # campaign exit in about a second on the Test-Path below.
    [string]   $Prj      = 'V:\',
    [string]   $Python   = 'F:\python_environments\qOFO_clean\python.exe',
    [string[]] $Droops   = @('0.05', '0.10'),
    [string[]] $Windows  = @('2016-02-22 13:00', '2016-01-05 08:00'),
    [string[]] $Deadbands = @('0', '0.0025', '0.005', '0.0075', '0.01',
                              '0.02', '0.05', '0.2'),
    [string]   $Duration = '900',
    [string]   $Scenario = 'rural_700',
    [string]   $LogDir   = ''
)

$ErrorActionPreference = 'Continue'

if (-not (Test-Path -LiteralPath $Prj))    { Write-Host "!! project not reachable: $Prj"; exit 1 }
if (-not (Test-Path -LiteralPath $Python)) { Write-Host "!! python not found: $Python"; exit 1 }
if (-not $LogDir) { $LogDir = Join-Path $Prj 'results\deadband_droop_e3_tracking\logs' }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force $LogDir | Out-Null }
Set-Location $Prj

# cells ordered (window, droop); half-widths innermost
$jobs = @()
foreach ($w in $Windows) {
    foreach ($m in $Droops) {
        foreach ($db in $Deadbands) { $jobs += ,@($m, $w, $db) }
    }
}

$total = $jobs.Count
Write-Host "=== E3 tracking-accuracy vs dead-band half-width (undisturbed) ==="
Write-Host "   windows    $($Windows -join ' | ')"
Write-Host "   droops     $($Droops -join ' ')"
Write-Host "   deadbands  $($Deadbands -join ' ')   (LIVE FROM t=0)"
Write-Host "   duration   $Duration s, FIXED 10 ms step (no adaptive), NO contingency"
Write-Host "   scenario   $Scenario, physical capability, rev-2 sensitivities"
Write-Host "   $total runs, ~$([math]::Round($total * 25.0 / 60.0, 1)) h at ~25 min/run"
Write-Host "   logs -> $LogDir"
Write-Host "=== started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

$n = 0
$failed = @()
$lastCell = ''
foreach ($j in $jobs) {
    $m = $j[0]; $w = $j[1]; $db = $j[2]
    $cell = "$w|$m"
    if ($cell -ne $lastCell) {
        Write-Host ""
        Write-Host "### cell: window $w, droop $m  --  $(Get-Date -Format 'HH:mm:ss')"
        $lastCell = $cell
    }
    $n++
    $wtag = ($w -replace '[ :\-]', '')
    $tag  = "e3_${wtag}_db$($db -replace '\.', '')_dr$($m -replace '\.', '')"
    $log  = Join-Path $LogDir "$tag.log"

    $pyArgs = @(
        '-X', 'utf8', '-m', 'experiments.run_comparison_rms_cosim_qss',
        '--duration', $Duration,
        '--profiles', '--profile-delivery', 'elmfile',
        '--dso-oltc-switch-cost', '200',
        '--physical-capability',
        '--start-time', $w,
        '--scenario', $Scenario,
        '--der-slope', $m,
        # LIVE FROM t=0 -- see the header.  No --qv-deadband-at-contingency.
        '--tso-deadband', $db, '--dso-deadband', $db,
        '--no-pdf', '--verbose', '1'
    )

    Write-Host "--- [$n/$total] db=$db droop=$m window=$w   $(Get-Date -Format 'HH:mm:ss') ---"
    & $Python @pyArgs *> $log
    $rc = $LASTEXITCODE
    # The PowerFactory exit-segfault is HARMLESS: every artefact is written
    # before it, and Gate E has already printed its verdict.  It has TWO codes
    # and the older sweep scripts only ever checked one of them:
    #
    #   139          POSIX 128 + SIGSEGV -- never produced on this platform
    #   -1073741819  0xC0000005 STATUS_ACCESS_VIOLATION -- what Windows returns
    #
    # Measured on E3 run 0540 (2026-08-07): exit -1073741819 with Gate E PASS,
    # all 45 dispatch steps, rms_records.pkl and a complete 193 MB
    # rms_comres_full.csv on disk.  Treating it as a failure would have
    # condemned all 32 runs of a healthy campaign.
    #
    # exit 1 is main()'s Gate-E verdict, NOT a crash: the run completed and
    # wrote every artefact, but some dispatch interval did not settle inside
    # its window.  At narrow dead bands that is the very effect this
    # experiment exists to measure, so it must not abort or be counted as a
    # failure.  Genuine crashes surface as a traceback in the log and a
    # different code.
    if ($rc -eq 139 -or $rc -eq -1073741819) {
        Write-Host "    exit=$rc (PF exit-segfault, results written)  $(Get-Date -Format 'HH:mm:ss')"
    } elseif ($rc -eq 1) {
        Write-Host "    exit=1  (Gate-E verdict not PASS -- expected at narrow bands; run kept)  $(Get-Date -Format 'HH:mm:ss')"
    } elseif ($rc -ne 0) {
        $failed += $tag
        Write-Host "!!! FAILED run $n ($tag): exit=$rc -- continuing; see $log"
    } else {
        Write-Host "    exit=0  $(Get-Date -Format 'HH:mm:ss')"
    }
}

Write-Host ""
Write-Host "=== DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "  raw results: $Prj\results\rms_phase6_replay  (move with finalize_e3_runs.py)"
if ($failed.Count -gt 0) {
    Write-Host "  $($failed.Count) FAILED of ${total}:"; $failed | ForEach-Object { Write-Host "    $_" }
} else { Write-Host "  all $total runs completed" }
Write-Host ""
Write-Host "  An EXIT CODE is not a health test on its own.  Gate E reports FAIL"
Write-Host "  (exit 1) whenever any dispatch interval fails to settle, which is"
Write-Host "  an expected outcome at narrow dead bands and is precisely what this"
Write-Host "  experiment measures -- do NOT treat exit=1 as a broken run.  Check"
Write-Host "  instead that each run dir has csv\rms_comres_full.csv reaching"
Write-Host "  t=$Duration s and a non-empty rms_records.pkl."
