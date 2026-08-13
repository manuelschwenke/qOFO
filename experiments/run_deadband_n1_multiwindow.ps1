# N-1 dead-band study across operating windows -- stage driver.
#
# Runs run_deadband_n1.ps1 three times so the study spans three operating
# points on ONE common dead-band ladder.
#
# Common ladder (8 values):
#     0, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5
# 0.5 is the NO-DROOP reference and is labelled as such. It replaces the
# earlier 0.15, which was NOT a valid no-droop reference: measured 2026-08-03,
# the gen-7 trip drove |V - V_anchor| to 0.1988 pu at the TS parks and 0.2234 pu
# at the DS parks, so the dead zone was crossed in 2/16 TS and 40/1200 DS
# windows and the droop engaged in the very run used as the "droop disabled"
# denominator. Every gen-7 compensation figure computed against delta = 0.15 is
# therefore a LOWER bound. 0.5 is unreachable by a factor of two.
#
# ORDER MATTERS. The stages are sequenced so that an interrupted run loses the
# least:
#   1. backfill of the window already measured -- cheapest, and it repairs the
#      gen-7 no-droop reference there;
#   2. 2016-12-18 14:00 -- the fattest DS drift tail in the screening, hence the
#      window most likely to move the LOWER bound, which is what delta = 0.005
#      is most exposed on;
#   3. 2016-02-22 13:00 -- the only net-import window, spanning the other end.
# After stage 2 there are two complete windows on the common ladder, which is
# already enough for a reproducibility statement.
#
# Window 1 (2016-01-05 08:00) already holds 0, 0.005, 0.01, 0.025, 0.05 from the
# first sweep, plus 0.001 / 0.075 / 0.15 which lie outside the common ladder and
# remain usable for that window alone. Only the three missing values are re-run.
#
# Cost at the measured 20.8 min/run: 9 + 24 + 24 = 57 runs ~= 19.8 h.
#
# Usage:  powershell -File experiments\run_deadband_n1_multiwindow.ps1

param(
    [string] $Python = 'F:\python_environments\qOFO_clean\python.exe'
)

$ErrorActionPreference = 'Continue'
$PRJ = Split-Path -Parent $PSScriptRoot
$SWEEP = Join-Path $PSScriptRoot 'run_deadband_n1.ps1'
Set-Location $PRJ

$LADDER   = @('0', '0.0025', '0.005', '0.01', '0.025', '0.05', '0.1', '0.5')
$BACKFILL = @('0.0025', '0.1', '0.5')

Write-Host "################################################################"
Write-Host "### N-1 dead-band study, 3 windows, started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "### common ladder: $($LADDER -join ' ')   (0.5 = no droop)"
Write-Host "### 57 runs total, ~19.8 h at the measured 20.8 min/run"
Write-Host "################################################################"

# The sweep script is invoked with & rather than -File: PowerShell's -File
# parameter binding collapses an array argument to its first element (observed
# 2026-07-31, when -Deltas silently became a single value and the next argument
# was bound to -Python).
$stages = @(
    @{ n = 1; win = '2016-01-05 08:00'; d = $BACKFILL; note = 'backfill (+409 MW)' },
    @{ n = 2; win = '2016-12-18 14:00'; d = $LADDER;   note = 'new window (+1367 MW)' },
    @{ n = 3; win = '2016-02-22 13:00'; d = $LADDER;   note = 'new window (-117 MW)' }
)

foreach ($s in $stages) {
    Write-Host ""
    Write-Host "################################################################"
    Write-Host "### STAGE $($s.n)/3 -- $($s.note) -- $($s.win)"
    Write-Host "### deltas: $($s.d -join ' ')   ($($s.d.Count * 3) runs)"
    Write-Host "### $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "################################################################"
    & $SWEEP -Window $s.win -Deltas $s.d -Python $Python
    Write-Host "### STAGE $($s.n)/3 done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

Write-Host ""
Write-Host "################################################################"
Write-Host "### ALL STAGES DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "### analyse: & '$Python' -X utf8 -m analysis.deadband_n1"
Write-Host "###          & '$Python' -X utf8 -m analysis.deadband_n1_figures"
Write-Host "################################################################"
