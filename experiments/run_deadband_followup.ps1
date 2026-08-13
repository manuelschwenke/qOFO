# Dead-band selection -- follow-up matrix (thesis Ch. 8 sec. 2).
#
# Runs AFTER the main 15-run sweep (experiments\run_deadband_sweep.ps1) and
# completes it in the three places the main matrix leaves open. Every phase is
# driven through the same sweep script, so scenario, --physical-capability and
# the rest of the argument vector cannot drift between them.
#
# ---------------------------------------------------------------------------
# WHY THE THIRD WINDOW CHANGED
#
# The main matrix used 2016-07-15 03:00 as its stressed window. It is
# DEGENERATE: all five dead bands returned bit-identical results
# (ifQ 2.167 Mvar, TS V 0.00592, DS V 0.01570 -- and byte-identical 12 MB DER
# records). Cause: July, 03:00, no PV and negligible wind, so aggregate DER
# infeed is 29.2 MW against 2605.7 MW in window 2. Under --physical-capability
# the VDE-AR-N-4120-v2 diagram makes Q capability contingent on P, so ALL 44
# parks report "zero Q capability -- the park cannot act as a Q actuator" and
# both zone PCC capabilities are [0.0, 0.0] Mvar. With no DER able to move, the
# Q(V) characteristic never binds and its dead zone cannot matter.
#
# That is a property of the operating point, not a bug, and it is worth
# reporting: dead-band selection presupposes DER reactive headroom.
#
# REPLACEMENT: 2016-12-18 14:00 -- net infeed +1367.1 MW, all 44 parks live.
#
# 2016-07-25 13:00 was tried FIRST and rejected empirically: at +3259.4 MW net
# infeed (top of the annual distribution) the static leg's Jacobian probe power
# flow does not converge -- LoadflowNotConverged after 200 iterations, run
# aborted in 32 s. Do not re-pick from the extreme tail.
#
# Offline screening could not predict this. A pandapower probe with
# init="flat" marked many feasible hours DIV, and with runner-like settings
# (distributed_slack) even 2016-07-25 13:00 converges -- so neither setting
# reproduces what the runner actually does. The reliable evidence is empirical:
# +805 MW (window 2) runs, +3259 MW does not.
#
# +1367.1 MW is therefore chosen for margin, not for maximum stress: 1.7x
# window 2's net infeed, comparable to the 1.9x excursion ratio between windows
# 1 and 2, and well below the known failure point.
#
# NO CAPABILITY CONFOUND (corrected 2026-07-31 from the run's own log).
# An earlier note here claimed this window had 44/44 parks live against 28/44
# for windows 1 and 2, and warned that it therefore differed in both stress and
# actuator authority. That was an artefact of the screening proxy, which counted
# a park as "live" at P > 0.5 MW -- not the VDE capability threshold. Run 0114
# reports 16/44 parks with zero Q capability and PCC [-42.0, 46.1] Mvar, i.e.
# the SAME count as windows 1 and 2 (16/44, ~ +-45 Mvar). The three live windows
# therefore vary loading and infeed at essentially constant DER reactive
# capability, which is what makes a delta* shift attributable to the operating
# point rather than to a change in the actuator set.
#
# NOTE: net infeed is a SCREENING proxy, not a voltage excursion. It is not
# added to EXCURSION in analysis/deadband_selection.py, because those figures
# were measured on the older topology and mixing a new-topology number into the
# same column would produce an incomparable quantity.
#
# 2016-07-15 03:00 is deliberately absent from every phase below: a delta = 0
# run there would cost ~4.9 h to reproduce numbers already known to be
# identical.
# ---------------------------------------------------------------------------
#
# PHASE A -- replacement window, base sweep. 5 runs, ~65 min.
# PHASE B -- wide extension, 4 runs, ~52 min.
#   delta = 0.02, 0.03 on window 2 and the replacement. In window 2 the
#   interface-Q metric is still falling at the top of the swept range, so its
#   argmin sits AT the edge -- a bound, not a measured optimum (the analysis
#   flags this via `at_range_edge`). Without this phase the chapter can say
#   delta* moves with the operating point, but not where it lands under stress.
# PHASE C -- zero anchor, 3 runs, ~15 h.
#   delta = 0 on windows 1, 2 and the replacement. The true zero-dead-band
#   anchor: window 1's interior minimum at 0.005 rests on a single narrow-side
#   point, and both voltage metrics bottom out at the narrow edge (0.0025), so
#   they are unbracketed from below. Cost: a zero dead band's static-leg
#   initialisation runs ~4.9 h against ~13 min for every other delta, which is
#   why the main sweep excludes it.
#
# Each phase runs only if the previous one succeeded -- same fail-fast reasoning
# as the main sweep. A and B are short, so they are a cheap check that the
# configuration is sound before committing 15 h to C.
#
# Usage:  powershell -File experiments\run_deadband_followup.ps1
#         powershell -File experiments\run_deadband_followup.ps1 -Only A,B
#         powershell -File experiments\run_deadband_followup.ps1 -Replacement '2016-04-10 16:00'

param(
    [string]   $Scenario    = 'rural_700',
    [string]   $Replacement = '2016-12-18 14:00',
    [string]   $Window1     = '2016-01-05 08:00',
    [string]   $Window2     = '2016-01-15 03:00',
    [string[]] $BaseDeltas  = @('0.0025', '0.005', '0.0075', '0.01', '0.015'),
    [string[]] $WideDeltas  = @('0.02', '0.03'),
    [string[]] $Only        = @('A', 'B', 'C')
)

$ErrorActionPreference = 'Continue'
$PRJ   = Split-Path -Parent $PSScriptRoot
$SWEEP = Join-Path $PSScriptRoot 'run_deadband_sweep.ps1'
# powershell.exe is NOT on PATH in the Claude Code shell -- resolve it from
# $PSHOME. Verified 2026-07-31; bare 'powershell.exe' raises
# CommandNotFoundException and would have failed at launch, not at authoring.
$PS    = Join-Path $PSHOME 'powershell.exe'
if (-not (Test-Path $SWEEP)) { Write-Host "!!! not found: $SWEEP"; exit 2 }
if (-not (Test-Path $PS))    { Write-Host "!!! not found: $PS";    exit 2 }
Set-Location $PRJ

function Invoke-Phase {
    param([string]$Tag, [string]$What, [string[]]$Deltas, [string[]]$Windows)
    if ($Only -notcontains $Tag) {
        Write-Host ">>> PHASE $Tag skipped (not in -Only)"
        return 0
    }
    $n = $Deltas.Count * $Windows.Count
    Write-Host ""
    Write-Host ">>> PHASE ${Tag}: $What  ($n runs)  $(Get-Date -Format 'HH:mm:ss')"
    Write-Host "    deltas : $($Deltas -join ', ')"
    Write-Host "    windows: $($Windows -join ' | ')"
    # Call the script directly, NOT via `powershell -File`. With -File every
    # argument is a separate literal token, so an array parameter collapses to
    # its first element and the leftovers bind positionally to whatever comes
    # next -- observed 2026-07-31: -Deltas kept only 0.0025 and $Python was
    # bound to '0.0075', so the sweep hit its "python not found" branch and
    # exited 2 in one second. Invoking with `&` passes arrays natively; `exit`
    # inside the called script returns control here and sets $LASTEXITCODE
    # (verified, parent survives).
    & $SWEEP -Scenario $Scenario -Deltas $Deltas -Windows $Windows
    $rc = $LASTEXITCODE
    Write-Host ">>> PHASE $Tag exit=$rc  $(Get-Date -Format 'HH:mm:ss')"
    return $rc
}

Write-Host "=== dead-band FOLLOW-UP started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "    replacement window: $Replacement"
Write-Host "    phases requested  : $($Only -join ', ')"

$rc = Invoke-Phase 'A' 'replacement window, base sweep' $BaseDeltas @($Replacement)
if ($rc -ne 0) { Write-Host "!!! phase A failed -- stopping."; exit 1 }

# Window1 is included for a SYMMETRIC grid (Manuel, 2026-07-31), not because it
# needs bracketing: window 1's interface-Q minimum at 0.005 is already interior
# (0.463, 0.441, 0.466, 0.510, 0.533), so 0.02/0.03 only extend an ascending
# arm. Its unbracketed side is the narrow one, which phase C covers.
# NOTE: the 2026-07-31 execution launched before this edit, so window 1's two
# wide runs were appended separately after phase C rather than running here.
$rc = Invoke-Phase 'B' 'wide extension' $WideDeltas @($Window1, $Window2, $Replacement)
if ($rc -ne 0) { Write-Host "!!! phase B failed -- NOT starting the 15 h phase C."; exit 1 }

$rc = Invoke-Phase 'C' 'zero anchor' @('0') @($Window1, $Window2, $Replacement)
if ($rc -ne 0) { Write-Host "!!! phase C failed"; exit 1 }

Write-Host ""
Write-Host "=== FOLLOW-UP DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "analyse:  F:\python_environments\qOFO_clean\python.exe -X utf8 -m analysis.deadband_selection --scenario $Scenario"
# Explicit: $LASTEXITCODE still holds the last phase's code, and a non-zero
# leftover would make a successful follow-up look like a failure.
exit 0
