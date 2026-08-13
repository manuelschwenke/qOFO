# ===================================================================
#  Dead band x droop -- droop 0.05, TWO WINDOWS   (mschwenke account)
# ===================================================================
#
#  Tonight, split ONE DROOP PER MACHINE so each produces a complete,
#  self-contained level that can be analysed without waiting on the other:
#
#     this account (mschwenke) : droop 0.05 -> 48 runs, ~17.6 h
#     ms_admin                 : droop 0.10 -> 48 runs, ~17.6 h
#
#  2 windows x 8 dead bands x {twin, gen 1 trip, gen 5 trip} = 48 runs.
#
#  WINDOWS -- the pair that stresses the conclusion hardest:
#     2016-01-05 08:00  (+409 MW net infeed)  reference; already has droop 0.06
#                                             data at 11 dead bands
#     2016-02-22 13:00  (-117 MW, net import) the window where TS drift is
#                                             2.4x higher and where delta=0.005
#                                             FAILED (13.3 % false activation)
#  2016-12-18 (+1367 MW) is omitted: it behaved almost identically to +409, so
#  it adds the least per hour.
#
#  After tonight this window carries THREE droop levels (0.05 / 0.06 / 0.10),
#  which is a real droop comparison rather than two points.
#
#  NAMING: the CLI flag is --der-slope but the quantity IS the droop. It
#  divides the voltage error (static R = S_n/slope, RMS Kdroop = 1/slope), so
#  0.05 pu of deviation commands full rated Q -- a 5 % droop. The grid code
#  permits 5-15 %; 0.05 and 0.10 are its lower edge and its middle.
#
#  VERIFIED 2026-08-04 before launch -- the droop reaches the RMS plant, not
#  only the static one:
#     [qvpre] anchored 44 Q(V) pre-controllers; ... droops applied: [0.1] pu
#  Until that plumbing was fixed the RMS side silently kept 0.06 whatever the
#  flag said, which would have made every run here void.
#
#  EVENTS: gen 1 (650 MW, zone 2, open-loop peak 0.083 pu) and gen 5 (560 MW,
#  zone 3, 0.051 pu) -- chosen for spread and for lying in different zones.
#  gen 7 (830 MW) is excluded on purpose: at 0.22 pu it saturates, so every
#  dead band below 0.05 behaves alike and it cannot discriminate.
#
#  Usage:
#     powershell -NoProfile -ExecutionPolicy Bypass -File RUN_TONIGHT_droop005.ps1
# ===================================================================

param(
    [string]   $Prj      = 'Z:\Python_Projekte\qOFO_GH',
    [string]   $Python   = 'F:\python_environments\qOFO_clean\python.exe',
    [string]   $Droop    = '0.05',
    [string[]] $Windows  = @('2016-01-05 08:00', '2016-02-22 13:00'),
    [string[]] $Deltas   = @('0.0025', '0.005', '0.0075', '0.01',
                             '0.025', '0.05', '0.1', '0.5'),
    [int[]]    $TripGens = @(-1, 1, 5)
)

$ErrorActionPreference = 'Continue'

Write-Host "=== pre-flight ==="
# Validate the root BEFORE any Join-Path: Join-Path on an unmapped drive throws
# a DriveNotFoundException and the failure cascades into a wall of binding
# errors instead of one legible message.
if (-not (Test-Path -LiteralPath $Prj)) {
    Write-Host "!! project not reachable: $Prj"
    Write-Host "   pass -Prj with the correct drive letter or UNC path"
    exit 1
}
$sweep = Join-Path $Prj 'experiments\run_deadband_n1.ps1'
$logs  = Join-Path $Prj 'results\deadband_n1\logs'
if (-not (Test-Path $sweep))  { Write-Host "!! sweep script not found under $Prj"; exit 1 }
if (-not (Test-Path $Python)) { Write-Host "!! python not found: $Python"; exit 1 }
New-Item -ItemType Directory -Force $logs | Out-Null
Set-Location $Prj

$n = $Windows.Count * $Deltas.Count * $TripGens.Count
Write-Host "   project $Prj"
Write-Host "   droop   $Droop pu"
Write-Host "   windows $($Windows -join ' | ')"
Write-Host "   deltas  $($Deltas -join ' ')"
Write-Host "   gens    $($TripGens -join ' ')   (-1 = undisturbed twin)"
Write-Host "   $n runs, ~$([math]::Round($n * 22 / 60.0, 1)) h"

Write-Host ""
Write-Host "=== started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
$dtag = $Droop -replace '\.', ''
for ($i = 0; $i -lt $Windows.Count; $i++) {
    Write-Host ""
    Write-Host "### window $($i+1)/$($Windows.Count) -- $($Windows[$i]) -- $(Get-Date -Format 'HH:mm:ss')"
    $wtag = ($Windows[$i] -replace '[ :\-]', '')
    & $sweep -Window $Windows[$i] -Deltas $Deltas -TripGens $TripGens `
             -Droop $Droop -Python $Python `
        *> (Join-Path $logs ("_night_dr" + $dtag + "_" + $wtag + ".log"))
}

Write-Host ""
Write-Host "=== DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "  results: $Prj\results\rms_phase6_replay  (shared with the other"
Write-Host "  account; run-dir allocation is atomic, so no merge is needed)"
Write-Host ""
Write-Host "  A run's EXIT CODE is NOT a health test: every N-1 run exits 1"
Write-Host "  because Gate E validates QSS/RMS equivalence, which a topology"
Write-Host "  change legitimately breaks. Check for csv\rms_der_raw.csv."
