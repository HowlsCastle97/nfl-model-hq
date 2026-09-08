@echo off
rem One Kalshi snapshot per run, then republish the fast price feed.
rem Task Scheduler fires this every 10 minutes; see install_logger_task.ps1.
rem
rem Two jobs, deliberately in this order. The snapshot comes first because the
rem price history in kalshi_prices.db cannot be backfilled, so it must not be
rem starved by a slow or failing push. Publishing second keeps the public feed
rem current: GitHub's cron is best effort and has been delivering the "every 5
rem minutes" job roughly every 30, so this machine is the more reliable clock.
rem The Actions workflow stays on as a backstop for when this machine is off.
rem
rem kalshi_prices.db, logs\ and logs\prices.json are all gitignored. Nothing
rem this script writes should ever be committed.
setlocal
set "REPO=%~dp0"
set "PY=C:\Users\david\anaconda3\python.exe"
set "LOG=%REPO%logs\kalshi_logger.log"
if not exist "%REPO%logs" mkdir "%REPO%logs"
cd /d "%REPO%"

"%PY%" kalshi_logger.py --once >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

"%PY%" publish_prices.py --out "%REPO%logs\prices.json" --push >> "%LOG%" 2>&1
if not "%ERRORLEVEL%"=="0" set "RC=%ERRORLEVEL%"

exit /b %RC%
