@echo off
rem Health check: verifies delivered data, not configuration. See healthcheck.py.
rem Scheduled every 30 minutes and INTERACTIVE on purpose, unlike the logger:
rem a toast needs a desktop to appear on, and a service account has none. The
rem trade is that alerts wait until you are signed in, which is also the only
rem time you could act on one.
setlocal
set "REPO=%~dp0"
set "PY=C:\Users\david\anaconda3\python.exe"
if not exist "%REPO%logs" mkdir "%REPO%logs"
cd /d "%REPO%"
"%PY%" healthcheck.py --notify --quiet --json "%REPO%logs\health.json" >> "%REPO%logs\health.log" 2>&1
exit /b %ERRORLEVEL%
