# =====================================================================
#  Wait for the mschwenke false-activation battery, then run the
#  timescale study.  Detached: survives the Claude session ending.
# =====================================================================
#
#  Why a watcher and not a chained command: the false-activation batch was
#  launched from a shell the harness has since lost track of.  The WORK is
#  unaffected (it ran the whole 40-run sweep to completion after the harness
#  declared it stopped), but no completion signal will arrive, so the next
#  stage cannot be triggered by a notification.  This polls the evidence on
#  disk instead.
#
#  DONE is defined as: all ten fa_*_dr005.log files exist AND the newest has
#  not been written for -StaleMinutes.  A 600 s run writes its log
#  continuously, so five minutes of silence means the batch has moved on or
#  finished; ten files plus silence means finished.
#
#  The 07:00 guard is the user's condition: the timescale study is a
#  nice-to-have that must not still be starting when they wake up.
#
#  ms_admin is running its own PowerFactory session throughout.  That is the
#  arrangement the previous campaign used (one session per account) and is
#  fine; what kills a run is a SECOND session under the SAME account, which
#  is why this waits for our own batch rather than for PowerFactory to be
#  idle machine-wide.
# =====================================================================

param(
    [string] $Prj          = 'Z:\Python_Projekte\qOFO_GH',
    [string] $Python       = 'F:\python_environments\qOFO_clean\python.exe',
    [string] $LogDir       = 'Z:\Python_Projekte\qOFO_GH\results\qstep_sweep\logs',
    [int]    $ExpectedFa   = 10,
    [int]    $StaleMinutes = 5,
    [int]    $MaxWaitHours = 9,
    [string] $SkipAfter    = '07:00'
)

$ErrorActionPreference = 'Continue'
$status = Join-Path $LogDir '_watcher_status.log'
function Say($m) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m"
    Write-Host $line
    Add-Content -Path $status -Value $line -Encoding utf8
}

Say "watcher started; waiting for $ExpectedFa fa_*_dr005.log, stale >= $StaleMinutes min"

$deadline = (Get-Date).AddHours($MaxWaitHours)
$ready = $false
while ((Get-Date) -lt $deadline) {
    $fa = @(Get-ChildItem -Path (Join-Path $LogDir 'fa_*_dr005.log') -ErrorAction SilentlyContinue)
    if ($fa.Count -ge $ExpectedFa) {
        $newest = ($fa | Sort-Object LastWriteTime -Descending)[0]
        $quiet  = ((Get-Date) - $newest.LastWriteTime).TotalMinutes
        if ($quiet -ge $StaleMinutes) {
            Say "batch complete: $($fa.Count) logs, newest quiet for $([math]::Round($quiet,1)) min"
            $ready = $true
            break
        }
        Say "all $($fa.Count) logs present, newest still active ($([math]::Round($quiet,1)) min quiet)"
    } else {
        Say "waiting: $($fa.Count)/$ExpectedFa false-activation logs"
    }
    Start-Sleep -Seconds 120
}

if (-not $ready) {
    Say "TIMED OUT after $MaxWaitHours h -- timescale study NOT started"
    exit 1
}

$now = Get-Date
$cut = [datetime]::ParseExact($SkipAfter, 'HH:mm', $null)
# the guard is about the wall clock in the morning, so only skip when we are
# past the cut-off AND it is genuinely morning (not the previous evening)
if ($now.Hour -ge $cut.Hour -and $now.Hour -lt 12) {
    Say "it is $($now.ToString('HH:mm')), past the $SkipAfter cut-off -- timescale study SKIPPED as instructed"
    exit 0
}

$tlog = Join-Path $LogDir '_timescale_study.log'
Say "starting timescale study -> $tlog"
Set-Location $Prj
& $Python -X utf8 (Join-Path $Prj 'pf\timescale_study.py') *> $tlog
$rc = $LASTEXITCODE
if ($rc -eq 139) {
    Say "timescale study finished, exit=139 (PF exit-segfault, results written)"
} else {
    Say "timescale study finished, exit=$rc"
}
