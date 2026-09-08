<#
.SYNOPSIS
  Registers the scheduled task that snapshots Kalshi NFL prices every 10 minutes.

.DESCRIPTION
  Run this from an ELEVATED PowerShell to get an S4U task, which collects whether
  or not you are signed in. Without elevation it falls back to Interactive, which
  only collects while you are signed in and may blink a console window.

  Safe to re-run: it unregisters the existing task first, so this is also how you
  change the interval or repoint the action after moving the repo.

.PARAMETER WakeToRun
  Wake the machine from sleep for each snapshot. Off by default: it keeps a
  laptop from ever settling and costs battery. Worth it on a desktop that sleeps.
#>
param(
    [int]$IntervalMinutes = 10,
    [switch]$WakeToRun
)

$name = "Kalshi NFL price logger"
$wrapper = Join-Path $PSScriptRoot "run_kalshi_logger.cmd"
if (-not (Test-Path $wrapper)) { throw "wrapper not found: $wrapper" }

$action = New-ScheduledTaskAction -Execute $wrapper
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)
# An empty duration is how Task Scheduler spells "repeat indefinitely".
# [TimeSpan]::MaxValue is accepted by the cmdlet and then rejected at registration.
$trigger.Repetition.Duration = ""

$settingArgs = @{
    AllowStartIfOnBatteries    = $true
    DontStopIfGoingOnBatteries = $true
    StartWhenAvailable         = $true
    MultipleInstances          = "IgnoreNew"
    ExecutionTimeLimit         = (New-TimeSpan -Minutes 5)
    Hidden                     = $true
}
if ($WakeToRun) { $settingArgs["WakeToRun"] = $true }
$settings = New-ScheduledTaskSettingsSet @settingArgs

$desc = "Snapshots Kalshi NFL markets (KXNFLGAME, KXNFLSPREAD, KXNFLTOTAL) into " +
        "kalshi_prices.db every $IntervalMinutes minutes. The price history is " +
        "irreplaceable and cannot be backfilled, so this only ever appends."

try { Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction Stop } catch {}

$user = "$env:USERDOMAIN\$env:USERNAME"
$done = $false
try {
    $p = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
        -Settings $settings -Principal $p -Description $desc -ErrorAction Stop | Out-Null
    Write-Host "S4U: collects whether or not you are signed in." -ForegroundColor Green
    $done = $true
} catch {
    Write-Host "S4U unavailable: $($_.Exception.Message.Trim())" -ForegroundColor Yellow
    Write-Host "Re-run this script from an elevated PowerShell to fix that."
}
if (-not $done) {
    $p = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
        -Settings $settings -Principal $p -Description $desc -ErrorAction Stop | Out-Null
    Write-Host "Interactive: collects only while you are signed in." -ForegroundColor Yellow
}

Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 10
$t = Get-ScheduledTask -TaskName $name
$i = Get-ScheduledTaskInfo -TaskName $name
Write-Host ""
Write-Host ("LogonType      : " + $t.Principal.LogonType)
Write-Host ("Interval       : every " + $t.Triggers[0].Repetition.Interval)
Write-Host ("LastTaskResult : " + $i.LastTaskResult + "  (0 = success)")
Write-Host ("NextRunTime    : " + $i.NextRunTime)
