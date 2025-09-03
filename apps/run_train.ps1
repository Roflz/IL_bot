# run_train.ps1 - PowerShell launcher for OSRS Imitation Learning Training
# 
# Usage Examples:
#   .\run_train.ps1                                    # Default settings (sequential model)
#   .\run_train.ps1 -Model "default"                   # Use ImitationHybridModel
#   .\run_train.ps1 -Model "sequential"                # Use SequentialImitationModel
#   .\run_train.ps1 -Epochs 20 -LR 1e-4               # Custom epochs and learning rate
#   .\run_train.ps1 -Model "default" -Epochs 15 -Batch 32 # Default model with custom settings
#
# Parameters:
#   -DataDir: Path to training data directory
#   -Epochs: Number of training epochs
#   -LR: Learning rate
#   -Batch: Batch size
#   -Model: Model type ("default" or "sequential")
#   -TargetsVersion: Target data version (v1 or v2)

param(
    [string]$DataDir = "data/recording_sessions/20250831_113719/06_final_training_data",
    [int]$Epochs = 40,
    [double]$LR = 2.5e-4,
    [int]$Batch = 64,
    [string]$Model = "sequential"  # "default" or "sequential"
)

$py = "py"
$script = "apps\train.py"

# Check if train.py exists
if (-not (Test-Path $script)) {
    Write-Error "Training script not found at: $script"
    Write-Host "Please ensure you have created apps/train.py with the proper CLI interface"
    exit 1
}

# Check if data directory exists
if (-not (Test-Path $DataDir)) {
    Write-Error "Data directory not found: $DataDir"
    Write-Host "Please ensure the data directory exists and contains the required .npy files"
    exit 1
}

# Check for required data files
$requiredFiles = @("action_input_sequences.npy", "action_targets.npy", "gamestate_sequences.npy")
foreach ($file in $requiredFiles) {
    $filePath = Join-Path $DataDir $file
    if (-not (Test-Path $filePath)) {
        Write-Error "Required data file not found: $filePath"
        Write-Host "Please ensure all required .npy files are present in the data directory"
        exit 1
    }
}

Write-Host "Starting OSRS Imitation Learning Training..." -ForegroundColor Green
Write-Host "Data Directory: $DataDir" -ForegroundColor Cyan
Write-Host "Epochs: $Epochs" -ForegroundColor Cyan
Write-Host "Learning Rate: $LR" -ForegroundColor Cyan
Write-Host "Batch Size: $Batch" -ForegroundColor Cyan


# Validate model parameter
if ($Model -notin @("default", "sequential")) {
    Write-Error "Invalid model type: $Model. Must be 'default' or 'sequential'"
    exit 1
}

# Display model information
if ($Model -eq "sequential") {
    Write-Host "Model: SequentialImitationModel (SequentialActionDecoder)" -ForegroundColor Yellow
} else {
    Write-Host "Model: ImitationHybridModel (ActionSequenceDecoder)" -ForegroundColor Yellow
}

Write-Host "=" * 60 -ForegroundColor Yellow

# Build command arguments
$cmdArgs = @(
    "--data_dir", $DataDir,
    "--epochs", $Epochs,
    "--lr", $LR,
    "--batch_size", $Batch
)

# Add model flag based on model type
if ($Model -eq "sequential") {
    $cmdArgs += "--use_sequential"
}

# Run the training script with all parameters
& $py $script @cmdArgs

# Check exit code and provide summary
if ($LASTEXITCODE -eq 0) {
    Write-Host "=" * 60 -ForegroundColor Green
    Write-Host "Training completed successfully!" -ForegroundColor Green
    Write-Host "Model trained with the following configuration:" -ForegroundColor Cyan
    Write-Host "  - Data Directory: $DataDir" -ForegroundColor White
    Write-Host "  - Epochs: $Epochs" -ForegroundColor White
    Write-Host "  - Learning Rate: $LR" -ForegroundColor White
    Write-Host "  - Batch Size: $Batch" -ForegroundColor White

    if ($Model -eq "sequential") {
        Write-Host "  - Model: SequentialImitationModel (SequentialActionDecoder)" -ForegroundColor White
        Write-Host "  - Features: Sequential generation, cumulative timing, natural sequence length" -ForegroundColor White
    } else {
        Write-Host "  - Model: ImitationHybridModel (ActionSequenceDecoder)" -ForegroundColor White
        Write-Host "  - Features: Parallel generation, timing-aware losses, cumulative timing" -ForegroundColor White
    }
    Write-Host "=" * 60 -ForegroundColor Green
} else {
    Write-Host "=" * 60 -ForegroundColor Red
    Write-Host "Training failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Please check the error messages above for details." -ForegroundColor Yellow
    Write-Host "=" * 60 -ForegroundColor Red
}
