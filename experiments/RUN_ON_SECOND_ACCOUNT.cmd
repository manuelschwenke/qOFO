@echo off
REM Thin launcher for RUN_ON_SECOND_ACCOUNT.ps1 -- double-clickable.
REM All logic lives in the .ps1; batch is avoided because its delayed-expansion
REM rules make the "did the run write traces?" check unreliable.
setlocal
set "PS=C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS%" (
  echo !! powershell not found at %PS%
  exit /b 1
)
"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_ON_SECOND_ACCOUNT.ps1" %*
echo(
echo === exit code %ERRORLEVEL% ===
pause
