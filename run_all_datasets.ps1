<#
run_all_datasets.ps1

Launches one FULLY DETACHED background process per dataset (Start-Process
-WindowStyle Hidden), each running the full 10-fold loop via run.py.
Unlike -NoNewWindow, these survive you closing the launching PowerShell
window -- they only stop if you kill them explicitly or shut the machine
down.

Usage:
    .\run_all_datasets.ps1
    .\run_all_datasets.ps1 -Seed 42 -NFolds 10
    .\run_all_datasets.ps1 -Tasks "wine_quality","yprop_4_1"   # subset only

This script does NOT block -- it launches everything and returns your
prompt immediately. Check status later (same window or a new one) with:
    .\check_status.ps1

Logs land in .\logs\<task>.out.log and .\logs\<task>.err.log
Predictions land in .\results\<task>\predictions_fold<N>.npy
Per-fold R^2 summaries land in .\results\<task>\predictions_r2_summary.npy
#>

param(
    [string[]]$Tasks = @(
        "bike_sharing_demand",
        "brazilian_houses",
        "superconductivity",
        "wine_quality",
        "yprop_4_1"
    ),
    [int]$Seed = 42,
    [int]$NFolds = 10,
    [string]$PythonExe = "python"   # change to full venv path if needed, e.g. ".\.venv\Scripts\python.exe"
)

# --- Pin thread counts BEFORE launching children, so every job inherits ---
# this and none of them (e.g. RandomForest's default n_jobs=-1) oversubscribe
# your 6 physical cores across 5 simultaneous jobs.
$env:OMP_NUM_THREADS        = "1"
$env:OPENBLAS_NUM_THREADS   = "1"
$env:MKL_NUM_THREADS        = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS    = "1"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$pidRecords = @()

foreach ($task in $Tasks) {
    $outDir  = "results\$task"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $outPath = "$outDir\predictions.npy"

    $logOut = "logs\$task.out.log"
    $logErr = "logs\$task.err.log"

    $argList = @(
        "run.py",
        "--task", $task,
        "--n-folds", $NFolds,
        "--seed", $Seed,
        "--output-path", $outPath
    )

    # -WindowStyle Hidden (NOT -NoNewWindow) is what makes this genuinely
    # detached: the child gets its own process group, so closing this
    # console window will NOT kill it.
    $proc = Start-Process -FilePath $PythonExe `
        -ArgumentList $argList `
        -RedirectStandardOutput $logOut `
        -RedirectStandardError $logErr `
        -PassThru -WindowStyle Hidden -WorkingDirectory (Get-Location)

    $pidRecords += [PSCustomObject]@{ Task = $task; PID = $proc.Id; LogOut = $logOut; LogErr = $logErr }
    Write-Host "Started '$task' (PID $($proc.Id)) -> log: $logOut"
}

$pidRecords | Export-Csv -Path "logs\running_pids.csv" -NoTypeInformation

Write-Host "`nAll $($pidRecords.Count) datasets launched as detached background processes."
Write-Host "You can safely close this window -- they'll keep running."
Write-Host "Check on them anytime with: .\check_status.ps1"
