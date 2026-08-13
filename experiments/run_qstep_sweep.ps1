# =====================================================================
#  Dead-band sweep under a REACTIVE LOAD STEP  (Ch. 9)
# =====================================================================
#
#  One droop slope per invocation, so the campaign can be split across two
#  PowerFactory sessions (mschwenke / ms_admin) and finish in one night:
#
#     -Droop 0.05  -> 40 runs, ~8.9 h
#     -Droop 0.10  -> 40 runs, ~8.9 h
#
#  Run-dir allocation in results/rms_phase6_replay is atomic, so the two
#  accounts can write to the same tree with no merge step.
#
#  THE DISTURBANCE is +200 Mvar on the reactive power of the single
#  in-service load at a TS bus, at t = 200 s (delivered to the RMS plant at
#  t = 180.5 s, inside the dispatch interval whose measurements report it).
#  It leaves the topology unchanged, so unlike a machine trip it does not by
#  itself invalidate Gate E.  Sized 2026-08-06: at either location no bus
#  reaches the converter ride-through threshold Vdip = 0.90 pu (lowest 0.9612
#  at bus 7, 0.9108 at bus 11); 300 Mvar at bus 11 breached it at four buses.
#
#  LOCATIONS are both remote from AVR-regulated machines, which is what gives
#  the local layer any authority at all: measured on the gen-trip runs, buses
#  next to machines score a droop/no-droop ratio of ~1.0 (the AVRs do the
#  work), while these two score 1.71 and 1.57.
#     bus  7 -- DSO_1 coupler, a real load centre (202.7 MW + 66.6 Mvar)
#     bus 11 -- DSO_2 coupler, behind two 21.8 % transformers
#
#  THE DEAD BAND IS NOT SET THROUGH --tso-deadband.  That value also feeds the
#  CONTROLLERS and the STATIC plant, so using it to separate the legs makes
#  the closed loops diverge from t = 0 (measured 1.33e-2 pu of run-up
#  divergence).  Every run therefore carries --tso-deadband/--dso-deadband 0.5
#  and installs its actual dead band on the RMS parks AT THE STEP, via
#  --qv-deadband-at-contingency.  Run-ups are then bit-identical across legs
#  (verified 0.0e0 pu pairwise), so every post-step difference is disturbance
#  rejection alone.
#
#  ORDER is by complete cell (window, location): all ten dead bands of one
#  cell before moving on.  A night cut short by a licence grab or a crash then
#  leaves whole, analysable cells rather than 40 scattered fragments.
#
#  METRIC (analysis, not this script): time-weighted mean |V - V_pre| over
#  [step, 200 s), the remainder of the DS dispatch interval -- the span in
#  which the local layer is the only fast actor.  Do NOT use peak |dV|: the
#  peak is the switching transient 0.5 ms after the step, before any control
#  can act, and it is identical across legs to four decimals.
#
#  Usage:
#     powershell -NoProfile -ExecutionPolicy Bypass -File run_qstep_sweep.ps1 -Droop 0.05
# =====================================================================

param(
    [string]   $Prj      = 'Z:\Python_Projekte\qOFO_GH',
    [string]   $Python   = 'F:\python_environments\qOFO_clean\python.exe',
    [string]   $Droop    = '0.05',
    [string[]] $Windows  = @('2016-02-22 13:00', '2016-01-05 08:00'),
    [int[]]    $Buses    = @(11, 7),
    [string]   $StepMvar = '200',
    [string[]] $Deadbands = @('0', '0.0025', '0.005', '0.0075', '0.01',
                              '0.02', '0.03', '0.05', '0.1', '0.2'),
    [string]   $Duration = '240',
    [string]   $LogDir   = ''
)

$ErrorActionPreference = 'Continue'

if (-not (Test-Path -LiteralPath $Prj))    { Write-Host "!! project not reachable: $Prj"; exit 1 }
if (-not (Test-Path -LiteralPath $Python)) { Write-Host "!! python not found: $Python"; exit 1 }
if (-not $LogDir) { $LogDir = Join-Path $Prj 'results\qstep_sweep\logs' }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force $LogDir | Out-Null }
Set-Location $Prj

$total = $Windows.Count * $Buses.Count * $Deadbands.Count
Write-Host "=== reactive-step dead-band sweep ==="
Write-Host "   droop      $Droop pu"
Write-Host "   windows    $($Windows -join ' | ')"
Write-Host "   buses      $($Buses -join ' ')   step $StepMvar Mvar"
Write-Host "   deadbands  $($Deadbands -join ' ')"
Write-Host "   $total runs, ~$([math]::Round($total * 13.4 / 60.0, 1)) h"
Write-Host "   logs -> $LogDir"
Write-Host "=== started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

$n = 0
$failed = @()
foreach ($w in $Windows) {
    $wtag = ($w -replace '[ :\-]', '')
    foreach ($b in $Buses) {
        Write-Host ""
        Write-Host "### cell: window $w, bus $b, droop $Droop  --  $(Get-Date -Format 'HH:mm:ss')"
        foreach ($db in $Deadbands) {
            $n++
            $dtag  = ($Droop -replace '\.', '')
            $dbtag = ($db -replace '\.', '')
            $tag   = "qstep_${wtag}_b${b}_db${dbtag}_dr${dtag}"
            $log   = Join-Path $LogDir "$tag.log"

            $pyArgs = @(
                '-X', 'utf8', '-m', 'experiments.run_comparison_rms_cosim_qss',
                '--duration', $Duration,
                '--profiles', '--profile-delivery', 'elmfile',
                '--dso-oltc-switch-cost', '200',
                '--physical-capability',
                '--start-time', $w,
                '--scenario', 'rural_700',
                '--der-slope', $Droop,
                '--adaptive-step', '--rms-step-ms', '1', '--rms-step-max-ms', '10',
                # config dead band stays wide on purpose -- see the header
                '--tso-deadband', '0.5', '--dso-deadband', '0.5',
                '--qv-deadband-at-contingency', $db,
                '--q-step-bus', "$b", '--q-step-mvar', $StepMvar,
                '--trip-time', '200',
                '--no-pdf', '--verbose', '1'
            )

            Write-Host "--- [$n/$total] db=$db bus=$b   $(Get-Date -Format 'HH:mm:ss') ---"
            & $Python @pyArgs *> $log
            $rc = $LASTEXITCODE
            # exit 139 is the known-harmless PowerFactory segfault at process
            # exit once the desktop has been hidden: results are written
            # first.  Treat it as success, but say so.
            if ($rc -eq 139) {
                Write-Host "    exit=139 (PF exit-segfault, results written)  $(Get-Date -Format 'HH:mm:ss')"
            } elseif ($rc -ne 0) {
                $failed += $tag
                Write-Host "!!! FAILED run $n ($tag): exit=$rc -- continuing"
                Write-Host "!!! see $log"
            } else {
                Write-Host "    exit=0  $(Get-Date -Format 'HH:mm:ss')"
            }
        }
    }
}

Write-Host ""
Write-Host "=== DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "  results: $Prj\results\rms_phase6_replay"
if ($failed.Count -gt 0) {
    Write-Host "  $($failed.Count) FAILED of ${total}:"
    $failed | ForEach-Object { Write-Host "    $_" }
} else {
    Write-Host "  all $total runs completed"
}
Write-Host ""
Write-Host "  A run's EXIT CODE is not a health test on its own -- check that"
Write-Host "  each run dir has csv\rms_comres_full.csv reaching t=$Duration s, and"
Write-Host "  that the log carries a '[qvpre] Q(V) dead band re-installed' line"
Write-Host "  for every deadband except where none was requested."
