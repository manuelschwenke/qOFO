# ===================================================================
#  Generic RESUME for a dead-band x droop sweep -- keeps completed runs
# ===================================================================
#
#  Works for either account and any droop. It asks tools/missing_cells.py which
#  (delta, gen) cells are absent for each window and runs ONLY those, so an
#  interrupted sweep can be restarted without repeating finished work
#  (~22 min and ~180 MB per run).
#
#  A cell counts as done only if its trace file exists AND is >= --MinTraceMB.
#  Size is the completeness test: every good trace in this study is 25.7-27.6 MB,
#  so a smaller file is a truncated write and the cell is re-run. This matters
#  because a partial CSV still PARSES -- it would silently shorten a run's time
#  series instead of failing loudly.
#
#  DISK: checked before starting and after every window. The 2026-08-04 failure
#  was the 200 GB share filling up, which surfaced as a bogus "python not found"
#  exit and a run that wrote two log lines and stopped -- not as a disk error.
#
#  Examples
#    # this account, droop 0.05
#    powershell -File RESUME_SWEEP.ps1 -Droop 0.05
#
#    # ms_admin, droop 0.10 (V:\ is mapped directly to the project folder)
#    powershell -File V:\experiments\RESUME_SWEEP.ps1 -Droop 0.10 -Prj 'V:\'
#
#    # see what is missing without running anything
#    powershell -File RESUME_SWEEP.ps1 -Droop 0.10 -WhatIfOnly
# ===================================================================

param(
    [string]   $Prj    = 'Z:\Python_Projekte\qOFO_GH',
    [string]   $Python = 'F:\python_environments\qOFO_clean\python.exe',
    [Parameter(Mandatory = $true)]
    [string]   $Droop,
    [string[]] $Windows = @('2016-01-05 08:00', '2016-02-22 13:00'),
    [string[]] $Deltas  = @('0.0025', '0.005', '0.0075', '0.01',
                            '0.025', '0.05', '0.1', '0.5'),
    [int[]]    $TripGens = @(-1, 1, 5),
    [double]   $MinTraceMB = 20.0,
    [double]   $MinFreeGB  = 2.0,
    [double]   $GBPerRun   = 0.19,
    [switch]   $WhatIfOnly
)

$ErrorActionPreference = 'Continue'

if (-not (Test-Path -LiteralPath $Prj)) { Write-Host "!! project not reachable: $Prj"; exit 1 }
$sweep = Join-Path $Prj 'experiments\run_deadband_n1.ps1'
$logs  = Join-Path $Prj 'results\deadband_n1\logs'
if (-not (Test-Path $sweep))  { Write-Host "!! sweep script not found under $Prj"; exit 1 }
if (-not (Test-Path $Python)) { Write-Host "!! python not found: $Python"; exit 1 }
New-Item -ItemType Directory -Force $logs | Out-Null
Set-Location $Prj

$drive = (Split-Path -Qualifier $Prj).TrimEnd(':')
function Get-FreeGB { (Get-PSDrive -Name $drive).Free / 1GB }

# ---- work out what is missing, per window -------------------------------
$plan = @()
foreach ($w in $Windows) {
    $args = @('-X', 'utf8', '-m', 'tools.missing_cells',
              '--droop', $Droop, '--window', $w, '--min-mb', "$MinTraceMB",
              '--deltas') + $Deltas + @('--gens') + ($TripGens | ForEach-Object { "$_" })
    $out = & $Python @args
    foreach ($line in $out) {
        if ($line -match '^\s*([0-9.]+)\s+(-?\d+)\s*$') {
            $plan += [PSCustomObject]@{ Window = $w; Delta = $Matches[1]; Gen = [int]$Matches[2] }
        }
    }
}

$n = $plan.Count
Write-Host "=== RESUME droop $Droop ==="
foreach ($w in $Windows) {
    $c = @($plan | Where-Object { $_.Window -eq $w }).Count
    $tot = $Deltas.Count * $TripGens.Count
    Write-Host ("    {0}: {1}/{2} still to run" -f $w, $c, $tot)
}
Write-Host ("    total {0} runs, ~{1:N1} h, ~{2:N1} GB" -f $n, ($n * 22 / 60.0), ($n * $GBPerRun))

if ($n -eq 0) { Write-Host "    nothing to do -- sweep already complete"; exit 0 }
if ($WhatIfOnly) {
    $plan | ForEach-Object { Write-Host ("      {0}  delta={1} gen={2}" -f $_.Window, $_.Delta, $_.Gen) }
    exit 0
}

$need = $n * $GBPerRun + $MinFreeGB
$free = Get-FreeGB
Write-Host ("=== free space {0:N1} GB (need >= {1:N1}) ===" -f $free, $need)
if ($free -lt $need) {
    Write-Host "!! not enough free space -- refusing to start."
    Write-Host "   a completed run directory is ~$GBPerRun GB."
    exit 1
}

Write-Host "=== started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
$dtag = $Droop -replace '\.', ''
$i = 0
foreach ($cell in $plan) {
    $i++
    Write-Host ("--- [{0}/{1}] {2}  delta={3} gen={4}  {5}" -f `
                $i, $n, $cell.Window, $cell.Delta, $cell.Gen, (Get-Date -Format 'HH:mm:ss'))
    $wtag = ($cell.Window -replace '[ :\-]', '')
    & $sweep -Window $cell.Window -Deltas @($cell.Delta) -TripGens @($cell.Gen) `
             -Droop $Droop -Python $Python `
        *> (Join-Path $logs ("_resume_dr" + $dtag + "_" + $wtag + "_d" + ($cell.Delta -replace '\.', '') + "_g" + $cell.Gen + ".log"))

    $free = Get-FreeGB
    if ($free -lt $MinFreeGB) {
        Write-Host ("!! free space {0:N1} GB below {1:N1} -- stopping before runs start failing." -f $free, $MinFreeGB)
        Write-Host "   re-run this script after freeing space; completed cells will be skipped."
        exit 1
    }
}

Write-Host ""
Write-Host "=== RESUME DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  free {0:N1} GB ==="
