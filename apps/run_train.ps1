# run_train.ps1 - PowerShell launcher for OSRS Imitation Learning Training
# Usage: .\run_train.ps1 [parameters]

param(
    [string]$DataDir = "data/recording_sessions/20250831_113719/06_final_training_data",
    [int]$Epochs = 10,
    [double]$LR = 2.5e-4,
    [int]$Batch = 64
)

$py = "py"
$script = "apps\train.py"

# Check if train.py exists
if (-not (Test-Path $script)) {
    Write-Error "Training script not found at: $script"
    Write-Host "Please ensure you have created apps/train.py with the proper CLI interface"
    exit 1
}

Write-Host "Starting OSRS Imitation Learning Training..." -ForegroundColor Green
Write-Host "Data Directory: $DataDir" -ForegroundColor Cyan
Write-Host "Epochs: $Epochs" -ForegroundColor Cyan
Write-Host "Learning Rate: $LR" -ForegroundColor Cyan
Write-Host "Batch Size: $Batch" -ForegroundColor Cyan

Write-Host "=" * 60 -ForegroundColor Yellow

# Run the training script with essential parameters
& $py $script `
  --data_dir $DataDir `
  --epochs $Epochs `
  --lr $LR `
  --batch_size $Batch

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host "Training completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Training failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}
