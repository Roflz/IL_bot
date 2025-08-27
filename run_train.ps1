param(
  [string] $DataDir = "data\recording_sessions\20250827_040359\06_final_training_data",
  [int]    $Epochs = 40,
  [double] $LR = 0.00025,
  [double] $WeightDecay = 0.0001,
  [int]    $Batch = 32,
  [int]    $StepSize = 8,
  [double] $Gamma = 0.5,
  [switch] $NoClassWeights,
  [switch] $DisableAutoBatch,
  [double] $LW_Time = 0.3, [double] $LW_X = 2.0, [double] $LW_Y = 2.0,
  [double] $LW_Type = 1.0, [double] $LW_Btn = 1.0, [double] $LW_Key = 1.0,
  [double] $LW_SX = 1.0, [double] $LW_SY = 1.0,
  [double] $TimeDiv = 1000
)
$flags = @(
  "--data_dir", $DataDir,
  "--epochs", $Epochs,
  "--lr", $LR,
  "--weight_decay", $WeightDecay,
  "--batch_size", $Batch,
  "--step_size", $StepSize,
  "--gamma", $Gamma,
  "--lw_time", $LW_Time, "--lw_x", $LW_X, "--lw_y", $LW_Y,
  "--lw_type", $LW_Type, "--lw_btn", $LW_Btn, "--lw_key", $LW_Key,
  "--lw_sx", $LW_SX, "--lw_sy", $LW_SY,
  "--time_div", $TimeDiv
)
if ($NoClassWeights)   { $flags += "--no_class_weights" }
if ($DisableAutoBatch) { $flags += "--disable_auto_batch" }
Write-Host "Launching training with:" ($flags -join " ")
python train_model.py @flags
