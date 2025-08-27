# OSRS Bot Quickcheck Launcher
# Updated to use the new ilbot package structure

param(
    [Parameter(Mandatory=$true)]
    [string]$DataDir,
    [string]$Checkpoint = $null,
    [string]$Device = "cuda",
    [int]$Batches = 2,
    [int]$ValBatch = 8,
    [string]$OutDir = "quickcheck_out",
    [double]$TimeDiv = 1.0,
    [switch]$DisableAutoBatch
)

# Build the command with all parameters
$args = @(
    "--data_dir", $DataDir,
    "--batches", $Batches,
    "--val_batch", $ValBatch,
    "--out_dir", $OutDir,
    "--time_div", $TimeDiv
)

if ($Checkpoint) {
    $args += "--checkpoint", $Checkpoint
}

if ($Device -ne "cuda") {
    $args += "--device", $Device
}

if ($DisableAutoBatch) {
    $args += "--disable_auto_batch"
}

Write-Host "Starting OSRS Bot Quickcheck with ilbot package..." -ForegroundColor Green
Write-Host "Command: python -m apps.quickcheck $($args -join ' ')" -ForegroundColor Yellow

# Run the quickcheck using the new CLI wrapper
python -m apps.quickcheck @args
