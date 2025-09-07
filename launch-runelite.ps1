param(
  [int]$Count = 10,                 # how many instances to launch
  [int]$BasePort = 5500,            # first IPC port; instances use BasePort + i
  [string]$JarPath = "D:\bots\runelite\RuneLite.jar",  # path to RuneLite jar
  [string]$BaseDir = "D:\bots\instances",              # base dir to store per-instance homes
  [string]$ExportsBase = "D:\bots\exports"             # where your gamestate JSONs go
)

# ---- CONFIG you may want your plugins to read (names are examples) ----
# In your IpcInputPlugin/StateExporter2Plugin, read these:
#   System.getProperty("rl.instance") / ENV:RL_INSTANCE
#   System.getProperty("rl.ipc.port") / ENV:RL_IPC_PORT
#   System.getProperty("rl.export.dir") / ENV:RL_EXPORT_DIR

# Create base dirs
New-Item -ItemType Directory -Force -Path $BaseDir | Out-Null
New-Item -ItemType Directory -Force -Path $ExportsBase | Out-Null

# Track PIDs so we can stop cleanly later
$pidFile = Join-Path $BaseDir "runelite-pids.txt"
if (Test-Path $pidFile) { Remove-Item $pidFile -Force }

for ($i = 0; $i -lt $Count; $i++) {
  $inst = "inst_$i"
  $home = Join-Path $BaseDir $inst       # per-instance "user.home"
  $exp  = Join-Path $ExportsBase $inst   # per-instance gamestate export dir
  $port = $BasePort + $i

  New-Item -ItemType Directory -Force -Path $home | Out-Null
  New-Item -ItemType Directory -Force -Path $exp  | Out-Null

  # JVM system props (preferred; consistent across Windows)
  $jvmProps = @(
    "-Duser.home=$home",                 # isolates RuneLite config/cache
    "-Drl.instance=$i",                  # your plugin can log/label this
    "-Drl.ipc.port=$port",               # your IpcInput plugin listens here
    "-Drl.export.dir=$exp"               # your exporter writes JSONs here
    # Optional stability hints:
    # "-Dsun.java2d.d3d=false", "-Dsun.java2d.noddraw=true"
  ) -join ' '

  $args = "$jvmProps -jar `"$JarPath`""

  # Launch minimized; keep each PID so we can stop later
  $p = Start-Process -FilePath "javaw.exe" -ArgumentList $args -WindowStyle Minimized -PassThru
  "$($p.Id),$i,$port,$home" | Out-File -Append -Encoding ascii $pidFile

  Write-Host "Launched instance #$i  PID=$($p.Id)  Port=$port  Home=$home"
}

Write-Host "Done. PIDs stored in $pidFile"
