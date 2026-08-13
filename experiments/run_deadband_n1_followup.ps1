# N-1 dead-band follow-up: delta = 0.2, then a medium-event probe.
#
# STAGE A -- delta = 0.2 at all three windows: twin + gen 1 only (6 runs).
# Extends the ladder above 0.1 without going to the 0.5 no-droop reference, so
# the decay of compensation between 0.1 and 0.5 is resolved rather than
# interpolated.
#
# gen 7 is deliberately NOT extended to delta = 0.2. It is under review as the
# severe case precisely because it does not discriminate (compensation flat at
# 0.55/0.55/0.54/0.54/0.58 across delta = 0...0.025), and stage B may replace it.
# Adding ladder points to a case that may be dropped spends ~1.1 h for nothing.
# The gen 7 data already collected across the full ladder at three windows is
# retained -- this only declines to EXTEND it.
#
# STAGE B -- medium-event probe (3 runs, reference window, delta = 0.5).
# The 830 MW machine (gen 7) is too severe to discriminate: its compensation is
# flat across delta = 0...0.025 (0.55/0.55/0.54/0.54/0.58) because the excursion
# overwhelms every dead band in the interesting range. gen 1 (650 MW) already
# discriminates far better (0.78 -> 0.08). A SMALLER machine should discriminate
# better still, so the open-loop excursion of three untried units is measured at
# delta = 0.5 (droop provably silent) before committing to a full sweep:
#
#     gen 0   250 MW  bus 29   <- smallest; the natural medium candidate
#     gen 5   560 MW  bus 39
#     gen 2   632 MW  bus 32
#
# No twin is needed: the delta = 0.5 twin for this window already exists (0311).
# gen 9 (1000 MW) is excluded -- it diverges in the static screening.
#
# The useful event is the one whose open-loop peak lands near 0.02-0.05 pu:
# large enough to exceed every candidate dead band, small enough that delta
# still changes the outcome.
#
# Cost: 9 runs at the measured ~22 min = ~3.3 h.
#
# Usage:  powershell -File experiments\run_deadband_n1_followup.ps1

param(
    [string] $Python = 'F:\python_environments\qOFO_clean\python.exe'
)

$ErrorActionPreference = 'Continue'
$PRJ = Split-Path -Parent $PSScriptRoot
$SWEEP = Join-Path $PSScriptRoot 'run_deadband_n1.ps1'
Set-Location $PRJ

Write-Host "################################################################"
Write-Host "### N-1 follow-up started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "### stage A: delta=0.2, 3 windows x {twin, gen1} = 6 runs (gen7 excluded)"
Write-Host "### stage B: medium-event probe, gen 0/5/2 at delta=0.5 = 3 runs"
Write-Host "################################################################"

$windows = @('2016-01-05 08:00', '2016-12-18 14:00', '2016-02-22 13:00')
foreach ($w in $windows) {
    Write-Host ""
    Write-Host "### STAGE A -- delta=0.2 -- $w -- $(Get-Date -Format 'HH:mm:ss')"
    # gen 7 excluded: it may be replaced by the stage-B candidate.
    & $SWEEP -Window $w -Deltas @('0.2') -TripGens @(-1, 1) -Python $Python
}

Write-Host ""
Write-Host "### STAGE B -- medium-event probe -- $(Get-Date -Format 'HH:mm:ss')"
# delta = 0.5 so the droop is provably silent: this measures the event itself,
# not the controlled response to it.
& $SWEEP -Window '2016-01-05 08:00' -Deltas @('0.5') -TripGens @(0, 5, 2) -Python $Python

Write-Host ""
Write-Host "################################################################"
Write-Host "### FOLLOW-UP DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "################################################################"
