# Dead-band selection under an N-1 generator outage -- RMS co-simulation.
#
# Question
# --------
# Choose the DER Q(V) dead-zone half-width delta so the local droop acts as an
# INTER-OFO-STEP STABILISER and a DISTURBANCE REJECTOR. The droop compares
# |V - V_anchor| against delta, and V_anchor is re-anchored every time the OFO
# writes that park's setpoint (core/plant.py). So delta does not discriminate
# voltage levels -- it discriminates DRIFT SINCE THE LAST DISPATCH, i.e. it is
# a DETECTOR THRESHOLD separating ordinary drift from a real event.
#
# Measured 2026-08-02 (anchor-referenced, undisturbed, wide delta):
#     normal drift   TS median 0.00087, max 0.0031 (32 samples, thin)
#                    DS median 0.00051, p90 0.0013, max 0.0111
#     N-1 excursion  gen 7  0.0104   gen 0  0.0539   gen 5  0.0618
#                    gen 1  0.0854   gen 2  0.1025
# The two distributions OVERLAP (DS drift max 0.0111 > gen 7 at 0.0104), so no
# delta is both always-quiet and always-responsive: the choice is a quantile
# trade-off between false activation and missed detection, not an optimum.
#
# Why RMS and not QSS
# -------------------
# 1. The pp-vs-RMS gap is a DEAD-BAND-EDGE phenomenon (2026-07-24: "a genuine
#    solver-vs-solver" divergence, the droop being multi-valued at the edge).
#    Selecting a dead band on QSS would measure the parameter with an
#    instrument known to be unreliable at that parameter's critical point.
# 2. The design parameter is a PEAK voltage deviation. QSS's first post-event
#    sample is 20 s later, by which time the electromechanical transient is
#    over -- it cannot see the peak at all.
# 3. QSS redistributes the lost machine instantly through distributed slack;
#    RMS does it through governor droop and AVR response.
#
# Disturbance
# -----------
# A generator TRIP, not a load step. Credible N-1, and far stronger per unit of
# realism: the largest load step that settles (+400 MW at bus 41) moves the
# worst park 0.0035 pu against 0.0104-0.1025 pu for an outage. Impact tracks
# the LOST AVR VOLTAGE SUPPORT, not MW -- gen 7 is the largest unit (830 MW)
# and the weakest disturbance. gen 9 is excluded: it diverges in the static
# outage scan.
#
#   gen 7  mild    0.0104 pu -- sits ON the detection boundary, where drift and
#                              event distributions overlap; the informative case
#   gen 1  severe  0.0854 pu -- far above every delta; tests rejection depth
#
# Timing
# ------
# Trip at t = 200 s, i.e. 20 s AFTER a TSO dispatch (TSO fires at 0/180/360/540,
# DSO every 20 s). That maximises the time until the slow layer can help -- 160 s
# of droop-only operation -- which is precisely the inter-OFO-step stress this
# study is about. Horizon 600 s gives two post-trip TSO dispatches (360, 540).
#
# Matrix
# ------
# 8 dead bands x {undisturbed twin, gen 7, gen 1} = 24 runs.
# The twins are NOT optional: at gen 7 the event excursion (0.0104) is the same
# order as ordinary profile drift, so the response must be referenced to the
# same-delta undisturbed run or the drift is counted as rejection.
#
# delta ladder: placed on the MEASURED distributions above, covering both
# transitions rather than round numbers.
#     0, 0.001          droop-dominant
#     0.005, 0.01       drift tail and the gen-7 boundary: detection is decided
#     0.025,0.05,0.075  severe-event range (gen 0, gen 5, gen 1)
#     0.15              droop fully disabled reference
#
# delta_TS = delta_DS here (stage 1). Stage 2 opens the second axis only around
# the region this stage identifies, rather than spending 36 cells discovering
# that most of the plane is flat. Both stages use --tso-deadband/--dso-deadband
# so they share ONE code path (the per-sgen map) and stay comparable.
#
# Usage:  powershell -File experiments\run_deadband_n1.ps1

param(
    [string]   $Scenario = 'rural_700',
    [string[]] $Deltas   = @('0', '0.001', '0.005', '0.01',
                             '0.025', '0.05', '0.075', '0.15'),
    [string]   $Window   = '2016-01-05 08:00',
    [int]      $Duration = 600,
    [double]   $TripTime = 200,
    # -1 = the undisturbed twin. FIRST on purpose: if the night is cut short,
    # the twins are what every other cell is measured against.
    #
    # The sentinel is -1, NOT 0: pandapower gen index 0 is a real machine (the
    # 250 MW unit at bus 29), and using 0 as "no trip" would silently run a
    # twin whenever that outage was requested.
    [int[]]    $TripGens = @(-1, 7, 1),
    # DER Q(V) droop [pu]. Despite the CLI name --der-slope this is the DROOP:
    # it divides the voltage error (static R = S_n/slope, RMS Kdroop = 1/slope),
    # so 0.10 means 0.10 pu of deviation commands full rated Q -- a 10 % droop,
    # the middle of the 5-15 % the grid code permits. Empty = leave the config
    # default (0.06), which is what every run before 2026-08-04 used.
    [string]   $Droop    = '',
    [string]   $Python   = 'F:\python_environments\qOFO_clean\python.exe',
    [string]   $LogDir   = '',
    [switch]   $FailFast
)

$ErrorActionPreference = 'Continue'

$PRJ = Split-Path -Parent $PSScriptRoot
if (-not $LogDir) { $LogDir = Join-Path $PRJ 'results\deadband_n1\logs' }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force $LogDir | Out-Null }
if (-not (Test-Path $Python)) { Write-Host "!!! python not found: $Python"; exit 2 }
Set-Location $PRJ

# Format INVARIANTLY: this shell runs a de-DE culture and .ToString() would emit
# '200,0', which float() rejects. Interpolation "$x" is invariant, but the
# difference is invisible at the call site, so pin it explicitly.
$inv  = [System.Globalization.CultureInfo]::InvariantCulture
$tStr = $TripTime.ToString($inv)
$wtag = ($Window -replace '[ :\-]', '')
$droopArgs = @()
$droopTag  = ''
if ($Droop) {
    $droopArgs = @('--der-slope', $Droop)
    # tag so a droop-0.1 cell cannot overwrite the droop-0.06 log of the
    # same (window, delta, gen)
    $droopTag  = '_dr' + ($Droop -replace '\.', '')
}

$total = $Deltas.Count * $TripGens.Count
$n = 0
$failed = @()

Write-Host "=== N-1 dead-band sweep started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "    scenario=$Scenario  window=$Window  duration=${Duration}s  trip at t=${tStr}s"
Write-Host "    deltas: $($Deltas -join ' ')"
Write-Host "    gens  : $($TripGens -join ' ')   (-1 = undisturbed twin)"
Write-Host "    droop : $(if ($Droop) { "$Droop pu" } else { 'config default (0.06)' })"
Write-Host "    $total runs -> $LogDir"

foreach ($g in $TripGens) {
    if ($g -lt 0) {
        $gTag = 'twin'
        $gArgs = @()
    } else {
        $gTag = "gen$g"
        $gArgs = @('--trip-gen', "$g", '--trip-time', $tStr)
    }
    foreach ($db in $Deltas) {
        $n++
        $tag = "{0}_{1}_db{2}_{3}{4}" -f $Scenario, $wtag, ($db -replace '\.', ''), $gTag, $droopTag
        $logFile = Join-Path $LogDir "$tag.log"
        Write-Host "--- [$n/$total] delta=$db $gTag   $(Get-Date -Format 'HH:mm:ss') ---"

        $pyArgs = @(
            '-X', 'utf8', '-m', 'experiments.run_comparison_rms_cosim_qss',
            '--duration', "$Duration",
            '--profiles', '--profile-delivery', 'elmfile',
            '--dso-oltc-switch-cost', '200',
            '--physical-capability',
            '--tso-deadband', $db,
            '--dso-deadband', $db,
            '--start-time', $Window,
            '--scenario', $Scenario,
            '--no-pdf', '--verbose', '1'
        ) + $gArgs + $droopArgs

        # *> captures every stream; read $LASTEXITCODE, not $?, because a native
        # command's stderr makes $? false in PS 5.1 even on success.
        & $Python @pyArgs *> $logFile
        $rc = $LASTEXITCODE

        Write-Host "    exit=$rc  $(Get-Date -Format 'HH:mm:ss')"
        if ($rc -ne 0) {
            $failed += $tag
            if ($FailFast) {
                Write-Host "!!! ABORTING at run $n ($tag): exit=$rc"
                Write-Host "!!! see $logFile"
                exit 1
            }
            Write-Host "!!! FAILED run $n ($tag): exit=$rc -- continuing"
            Write-Host "!!! see $logFile"
        }
    }
}

if ($failed.Count) {
    Write-Host ""
    Write-Host "!!! $($failed.Count) of $total run(s) FAILED:"
    $failed | ForEach-Object { Write-Host "      $_" }
}

Write-Host "=== N-1 sweep DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "analyse:  & '$Python' -X utf8 -m analysis.deadband_n1"
