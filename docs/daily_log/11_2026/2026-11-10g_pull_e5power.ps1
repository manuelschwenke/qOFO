# Wait for the 3 CEN screen runs on e5power (status.json written at the end of each), then pull them into the
# local screen folder next to S1.  Polls every 120 s, gives up after 4 h.
$remote = 'F:\qOFO_runs\qOFO_cen_screen'
$local = 'F:\qOFO_screens\2026-11-10_cen_rule_gamma\E2'
$labels = 'M145_g100', 'M145_g050', 'M145_g025'
$deadline = (Get-Date).AddHours(4)
while ((Get-Date) -lt $deadline) {
    try {
        $n = Invoke-Command -ComputerName e5power -ScriptBlock {
            param($r, $l) ($l | Where-Object { Test-Path (Join-Path $r "E2\$_\status.json") }).Count
        } -ArgumentList $remote, $labels -ErrorAction Stop
    } catch { $n = -1 }
    "{0:HH:mm} done {1}/3" -f (Get-Date), $n
    if ($n -eq 3) { break }
    Start-Sleep -Seconds 120
}
if ($n -ne 3) { "timeout"; exit 1 }
$s = New-PSSession -ComputerName e5power
foreach ($l in $labels) {
    Copy-Item -FromSession $s -Path (Join-Path $remote "E2\$l") -Destination $local -Recurse -Force
    "pulled $l"
}
Copy-Item -FromSession $s -Path (Join-Path $remote 'driver.log') -Destination 'F:\qOFO_screens\2026-11-10_cen_rule_gamma\driver_e5power.log' -Force
Remove-PSSession $s
Get-ChildItem $local | ForEach-Object { $_.Name + ' ' + (Get-Content (Join-Path $_.FullName 'status.json') -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json).status }
