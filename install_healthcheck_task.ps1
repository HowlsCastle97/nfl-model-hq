<#
.SYNOPSIS
  Registers the scheduled task that runs healthcheck.py every 30 minutes.

.DESCRIPTION
  Deliberately Interactive, and deliberately NOT elevated, unlike
  install_logger_task.ps1. The alert is a message box, which needs a desktop to
  render on; an S4U service account has none, so an alert raised there would be
  silently swallowed, which is the exact failure mode this whole thing exists to
  prevent.

  The cost is that alerts wait until you are signed in. StartWhenAvailable means
  a check missed while the machine was off runs shortly after you log back in,
  so a failure that began overnight still reaches you.

  Safe to re-run.
#>
param([int]$IntervalMinutes = 30)

$name = "NFL Model HQ health check"
$wrapper = Join-Path $PSScriptRoot "run_healthcheck.cmd"
if (-not (Test-Path $wrapper)) { throw "wrapper not found: $wrapper" }

$action = New-ScheduledTaskAction -Execute $wrapper
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)
$trigger.Repetition.Duration = ""   # empty means indefinitely; see the logger installer
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -Hidden
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive -RunLevel Limited
$desc = "Checks delivered data every $IntervalMinutes minutes: price history age " +
        "per series, published feed age, site build age, GitHub Actions results, " +
        "and the logger task's own exit code. Raises a message box on failure."

try { Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction Stop } catch {}
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Description $desc -ErrorAction Stop | Out-Null
Write-Host "Registered '$name', every $IntervalMinutes minutes." -ForegroundColor Green

Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 20
$i = Get-ScheduledTaskInfo -TaskName $name
Write-Host ("LastTaskResult : " + $i.LastTaskResult + "  (this is the number of FAILING checks, not an error)")
Write-Host ("NextRunTime    : " + $i.NextRunTime)
