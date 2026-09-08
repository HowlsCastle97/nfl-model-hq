@echo off
rem One Kalshi snapshot per run. Task Scheduler fires this every 10 minutes; see
rem install_logger_task.ps1 to register or re-register that task.
rem
rem The price history in kalshi_prices.db is irreplaceable and gitignored, and so
rem is logs\. Nothing this script writes should ever be committed.
setlocal
set "REPO=%~dp0"
if not exist "%REPO%logs" mkdir "%REPO%logs"
cd /d "%REPO%"
"C:\Users\david\anaconda3\python.exe" kalshi_logger.py --once >> "%REPO%logs\kalshi_logger.log" 2>&1
exit /b %ERRORLEVEL%
