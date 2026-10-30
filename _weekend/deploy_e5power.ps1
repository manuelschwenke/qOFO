# Deploy the weekend branch to e5power (git bundle over a PowerShell session) and
# run a short smoke there.  Usage: powershell -File _weekend\deploy_e5power.ps1
$ErrorActionPreference = 'Stop'
$branch = 'weekend/harmonised-limits-2026-10-30'
$bundle = 'F:\qOFO_weekend_2026-10-30.bundle'
git -c safe.directory=* -C F:\qOFO_weekend bundle create $bundle "400a037..$branch"
$s = New-PSSession -ComputerName e5power
Copy-Item -ToSession $s -Path $bundle -Destination 'F:\qOFO_runs\weekend_2026-10-30.bundle' -Force
Invoke-Command -Session $s -ArgumentList $branch -ScriptBlock {
    param($branch)
    Set-Location F:\qOFO_runs\qOFO_weekend
    git -c safe.directory=* fetch F:\qOFO_runs\weekend_2026-10-30.bundle "${branch}:${branch}" 2>&1 | Out-String
    git -c safe.directory=* checkout $branch 2>&1 | Out-String
    git -c safe.directory=* log --oneline -1
}
Remove-PSSession $s
