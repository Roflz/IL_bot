# run_train.ps1 - PowerShell launcher for OSRS Imitation Learning Training
# Usage: .\run_train.ps1 [parameters]

param(
    [string]$DataDir = "data/recording_sessions/20250827_040359/06_final_training_data",
    [int]$Epochs = 40,
    [double]$LR = 2.5e-4,
    [double]$WeightDecay = 1e-4,
    [int]$Batch = 32,
    [int]$StepSize = 8,
    [double]$Gamma = 0.5,
    [string]$TargetsVersion = "v2",
    [bool]$UseLog1pTime = $true,
    [double]$TimeDiv = 1000.0,
    [double]$LW_Time = 0.3,
    [double]$LW_X = 2.0,
    [double]$LW_Y = 2.0,
    [double]$LW_Button = 1.0,
    [double]$LW_KeyAction = 1.0,
    [double]$LW_KeyId = 1.0,
    [double]$LW_ScrollY = 1.0,
    [string]$ExtraArgs = ""
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
Write-Host "Targets Version: $TargetsVersion" -ForegroundColor Cyan
Write-Host "Use Log1p Time: $UseLog1pTime" -ForegroundColor Cyan
Write-Host "Time Division: $TimeDiv ms" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Yellow

# Run the training script with all parameters
& $py $script `
  --data_dir $DataDir `
  --epochs $Epochs `
  --lr $LR `
  --weight_decay $WeightDecay `
  --batch_size $Batch `
  --step_size $StepSize `
  --gamma $Gamma `
  --targets_version $TargetsVersion `
  --use_log1p_time $UseLog1pTime `
  --time_div_ms $TimeDiv `
  --lw_time $LW_Time --lw_x $LW_X --lw_y $LW_Y `
  --lw_button $LW_Button --lw_key_action $LW_KeyAction --lw_key_id $LW_KeyId --lw_scroll_y $LW_ScrollY `
  @ExtraArgs

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host "Training completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Training failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}
