<#
check_status.ps1

Reads logs\running_pids.csv (written by run_all_datasets.ps1) and
reports, per dataset: whether the process is still running, and how
far it's gotten based on the log file (last line + fold count seen).

Usage:
    .\check_status.ps1
#>

$pidFile = "logs\running_pids.csv"
if (-not (Test-Path $pidFile)) {
    Write-Host "No logs\running_pids.csv found -- run .\run_all_datasets.ps1 first."
    exit 1
}

$records = Import-Csv $pidFile

Write-Host ("{0,-22} {1,-10} {2,-8} {3}" -f "Task", "Status", "PID", "Last log line")
Write-Host ("-" * 100)

foreach ($r in $records) {
    $proc = Get-Process -Id $r.PID -ErrorAction SilentlyContinue
    $status = if ($proc) { "RUNNING" } else { "finished/stopped" }

    $lastLine = ""
    if (Test-Path $r.LogOut) {
        $lastLine = (Get-Content $r.LogOut -Tail 1)
    }

    Write-Host ("{0,-22} {1,-10} {2,-8} {3}" -f $r.Task, $status, $r.PID, $lastLine)
}

Write-Host "`nTail a specific dataset live with, e.g.:"
Write-Host "    Get-Content logs\wine_quality.out.log -Wait -Tail 20"
Write-Host "`nKill a specific still-running dataset with:"
Write-Host "    Stop-Process -Id <PID>"
