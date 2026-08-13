# ===================================================================
#  Dead band x droop -- droop 0.10, TWO WINDOWS   (ms_admin account)
# ===================================================================
#
#  Tonight, split ONE DROOP PER MACHINE so each produces a complete,
#  self-contained level that can be analysed without waiting on the other:
#
#     this account (ms_admin)  : droop 0.10 -> 48 runs, ~17.6 h
#     mschwenke                : droop 0.05 -> 48 runs, ~17.6 h
#
#  2 windows x 8 dead bands x {twin, gen 1 trip, gen 5 trip} = 48 runs.
#
#  WINDOWS -- the pair that stresses the conclusion hardest:
#     2016-01-05 08:00  (+409 MW net infeed)  reference; already has droop 0.06
#                                             data at 11 dead bands
#     2016-02-22 13:00  (-117 MW, net import) the window where TS drift is
#                                             2.4x higher and where delta=0.005
#                                             FAILED (13.3 % false activation)
#  2016-12-18 (+1367 MW) is omitted: it behaved almost identically to +409.
#
#  NAMING: the CLI flag is --der-slope but the quantity IS the droop. It
#  divides the voltage error (static R = S_n/slope, RMS Kdroop = 1/slope), so
#  0.10 pu of deviation commands full rated Q -- a 10 % droop, the middle of
#  the 5-15 % the grid code permits.
#
#  VERIFIED 2026-08-04 before launch -- the droop reaches the RMS plant, not
#  only the static one:
#     [qvpre] anchored 44 Q(V) pre-controllers; ... droops applied: [0.1] pu
#  Until that plumbing was fixed the RMS side silently kept 0.06 whatever the
#  flag said, which would have made every run here void.
#
#  EVENTS: gen 1 (650 MW, zone 2, open-loop peak 0.083 pu) and gen 5 (560 MW,
#  zone 3, 0.051 pu) -- chosen for spread and for lying in different zones.
#  gen 7 (830 MW) is excluded on purpose: at 0.22 pu it saturates.
#
#  SHARED RESULTS FOLDER IS SAFE, no merge step needed. Run-directory
#  allocation in experiments/results_io.py calls mkdir() WITHOUT exist_ok --
#  an atomic create-if-not-exists -- and retries with the next counter on
#  FileExistsError, so two concurrent writers cannot claim one directory.
#
#  Usage on this account (V:\ is mapped directly to the project folder):
#     powershell -NoProfile -ExecutionPolicy Bypass -File V:\experiments\RUN_TONIGHT_droop010.ps1 -Prj 'V:\'
# ===================================================================

param(
    # ms_admin maps V:\ directly to the project folder, so the project ROOT is
    # V:\ itself. The trailing backslash matters: bare 'V:' means "the current
    # directory on drive V:", which Join-Path would turn into a relative path.
    [string]   $Prj      = 'V:\',
    [string]   $Python   = 'F:\python_environments\qOFO_clean\python.exe',
    [string]   $Droop    = '0.10',
    [string[]] $Windows  = @('2016-01-05 08:00', '2016-02-22 13:00'),
    [string[]] $Deltas   = @('0.0025', '0.005', '0.0075', '0.01',
                             '0.025', '0.05', '0.1', '0.5'),
    [int[]]    $TripGens = @(-1, 1, 5),
    [switch]   $SkipSmoke
)

$ErrorActionPreference = 'Continue'

Write-Host "=== pre-flight ==="
if (-not (Test-Path -LiteralPath $Prj)) {
    Write-Host "!! project not reachable: $Prj"
    Write-Host "   pass -Prj with the correct drive letter or UNC path, e.g."
    Write-Host "   -Prj '\\e5server\homefolders`$\mschwenke\Python_Projekte\qOFO_GH'"
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
Write-Host "   $n runs, ~$([math]::Round($n * 22 / 60.0, 1)) h"

function Test-NewestRunHasTraces {
    $root = Join-Path $Prj 'results\rms_phase6_replay'
    if (-not (Test-Path $root)) { return $false }
    $d = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
         Sort-Object CreationTime -Descending | Select-Object -First 1
    if (-not $d) { return $false }
    return (Test-Path (Join-Path $d.FullName 'csv\rms_der_raw.csv'))
}

if (-not $SkipSmoke) {
    Write-Host ""
    Write-Host "=== smoke test (1 run, ~20 min): can THIS account drive PowerFactory? ==="
    Write-Host "    failing here beats 48 broken runs overnight"
    & $Python -X utf8 -m experiments.run_comparison_rms_cosim_qss `
        --duration 600 --profiles --profile-delivery elmfile `
        --dso-oltc-switch-cost 200 --physical-capability `
        --tso-deadband 0.0075 --dso-deadband 0.0075 --der-slope $Droop `
        --start-time '2016-01-05 08:00' --scenario rural_700 `
        --no-pdf --verbose 1 *> (Join-Path $logs '_smoke_ms_admin.log')

    if (-not (Test-NewestRunHasTraces)) {
        Write-Host ""
        Write-Host "!! SMOKE TEST FAILED -- no result traces were written."
        Write-Host "   Most likely: PowerFactory cannot be driven from this account"
        Write-Host "   while the other account holds a session (single-session"
        Write-Host "   licence), or the project name differs."
        Write-Host "   See $logs\_smoke_ms_admin.log"
        exit 1
    }
    Write-Host "   smoke OK -- PowerFactory usable from this account"
    $sm = Get-Content (Join-Path $logs '_smoke_ms_admin.log') -Raw -ErrorAction SilentlyContinue
    if ($sm -and $sm -match 'droops applied: \[([^\]]*)\]') {
        Write-Host "   RMS droop applied: [$($Matches[1])] pu   (MUST contain $Droop)"
    } else {
        Write-Host "   !! could not confirm the RMS droop -- STOP and report"
    }
}

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
Write-Host "  results: $Prj\results\rms_phase6_replay  (shared; no merge needed)"
Write-Host "  Exit code 1 per run is EXPECTED (Gate E). Check csv\rms_der_raw.csv."
