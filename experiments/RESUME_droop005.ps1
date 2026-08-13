# ===================================================================
#  RESUME the droop-0.05 sweep -- only the cells still missing
# ===================================================================
#
#  The first attempt (15:37 -> 21:43) died because the 200 GB share filled up.
#  Symptoms that looked like other faults: the driver exited with the sweep
#  script's "python not found" code (2) although 15 runs had just used that
#  interpreter, and the in-flight run wrote two log lines and stopped. Both
#  were ENOSPC downstream. Verified afterwards: NO trace file was truncated --
#  every run either wrote a complete 25.7-27.6 MB CSV or none at all -- so
#  nothing already recorded is corrupt.
#
#  DISK BUDGET: a completed run directory is ~180 MB. This script runs 34 runs
#  and therefore needs about 6.5 GB free, and it CHECKS before starting. If the
#  other account is also running, budget for both (~12 GB total).
#
#  Missing when written (34 runs, ~12.5 h):
#     window +409 MW : gen 1 at delta 0.1 and 0.5   (2 runs)
#                      gen 5 at all 8 deltas        (8 runs)
#     window -117 MW : everything                   (24 runs)
#  The 14 completed +409 cells are NOT re-run. A duplicate would be harmless
#  anyway: the analysis keys cells by (window, delta, gen) and announces the
#  supersession.
#
#  Usage:
#     powershell -NoProfile -ExecutionPolicy Bypass -File RESUME_droop005.ps1
# ===================================================================

param(
    [string] $Prj    = 'Z:\Python_Projekte\qOFO_GH',
    [string] $Python = 'F:\python_environments\qOFO_clean\python.exe',
    [string] $Droop  = '0.05',
    [double] $MinFreeGB = 6.5
)

$ErrorActionPreference = 'Continue'

if (-not (Test-Path -LiteralPath $Prj)) { Write-Host "!! project not reachable: $Prj"; exit 1 }
$sweep = Join-Path $Prj 'experiments\run_deadband_n1.ps1'
$logs  = Join-Path $Prj 'results\deadband_n1\logs'
if (-not (Test-Path $sweep))  { Write-Host "!! sweep script not found under $Prj"; exit 1 }
if (-not (Test-Path $Python)) { Write-Host "!! python not found: $Python"; exit 1 }

# Disk check FIRST -- running out of space mid-sweep is what killed the last
# attempt, and it fails in a way that looks like several other faults.
$drive = (Split-Path -Qualifier $Prj).TrimEnd(':')
$free  = (Get-PSDrive -Name $drive).Free / 1GB
Write-Host ("=== free space on {0}: {1:N1} GB (need >= {2:N1}) ===" -f $drive, $free, $MinFreeGB)
if ($free -lt $MinFreeGB) {
    Write-Host "!! not enough free space -- refusing to start."
    Write-Host "   a run directory is ~180 MB; 34 runs need ~6.5 GB."
    exit 1
}

New-Item -ItemType Directory -Force $logs | Out-Null
Set-Location $Prj

$ALL = @('0.0025', '0.005', '0.0075', '0.01', '0.025', '0.05', '0.1', '0.5')
$stages = @(
    @{ w = '2016-01-05 08:00'; d = @('0.1', '0.5'); g = @(1);        tag = 'resume_w1_gen1' },
    @{ w = '2016-01-05 08:00'; d = $ALL;            g = @(5);        tag = 'resume_w1_gen5' },
    @{ w = '2016-02-22 13:00'; d = $ALL;            g = @(-1, 1, 5); tag = 'resume_w2_all'  }
)

$total = 0
foreach ($s in $stages) { $total += $s.d.Count * $s.g.Count }
Write-Host "=== RESUME droop $Droop -- $total runs, ~$([math]::Round($total * 22 / 60.0, 1)) h ==="
Write-Host "    started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

foreach ($s in $stages) {
    Write-Host ""
    Write-Host "### $($s.tag) -- $($s.w) -- gens $($s.g -join ',') -- $(Get-Date -Format 'HH:mm:ss')"
    & $sweep -Window $s.w -Deltas $s.d -TripGens $s.g -Droop $Droop -Python $Python `
        *> (Join-Path $logs ("_night_dr005_" + $s.tag + ".log"))
    $free = (Get-PSDrive -Name $drive).Free / 1GB
    Write-Host ("### stage done $(Get-Date -Format 'HH:mm:ss')  --  free space now {0:N1} GB" -f $free)
    if ($free -lt 1.0) {
        Write-Host "!! free space below 1 GB -- stopping before runs start failing."
        exit 1
    }
}

Write-Host ""
Write-Host "=== RESUME DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
