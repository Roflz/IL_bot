param(
  [int]$Count = 10,                 # how many instances to launch
  [int]$BasePort = 5500,            # first IPC port; instances use BasePort + i
  [string]$ProjectDir = "D:\IdeaProjects\runelite",    # path to RuneLite project
  [string]$ClassPathFile = "D:\repos\bot_runelite_IL\ilbot\ui\simple_recorder\rl-classpath.txt", # path to classpath file
  [string]$JavaExe = "C:\Program Files\Java\jdk-11.0.2\bin\java.exe", # Java executable
  [string]$BaseDir = "D:\bots\instances",              # base dir to store per-instance homes
  [string]$ExportsBase = "D:\bots\exports",             # where your gamestate JSONs go
  [string]$CredentialsDir = "D:\repos\bot_runelite_IL\credentials", # directory containing credential files
  [int]$DelaySeconds = 10,                              # delay between instance launches
  [string[]]$CredentialFiles = @()                      # specific credential files to use (optional)
)

# ---- CONFIG you may want your plugins to read (names are examples) ----
# In your IpcInputPlugin/StateExporter2Plugin, read these:
#   System.getProperty("rl.instance") / ENV:RL_INSTANCE
#   System.getProperty("rl.ipc.port") / ENV:RL_IPC_PORT
#   System.getProperty("rl.export.dir") / ENV:RL_EXPORT_DIR

# Pre-flight checks
if (-not (Test-Path $JavaExe)) { throw "Java not found: $JavaExe" }
if (-not (Test-Path $ProjectDir)) { throw "Project dir not found: $ProjectDir" }
if (-not (Test-Path $ClassPathFile)) { throw "Classpath file not found: $ClassPathFile" }
if (-not (Test-Path $CredentialsDir)) { throw "Credentials directory not found: $CredentialsDir" }

# Get available credential files
if ($CredentialFiles.Count -gt 0) {
  # Use specified credential files
  Write-Host "Using specified credential files: $($CredentialFiles -join ', ')"
  $selectedCredentialFiles = @()
  for ($i = 0; $i -lt $CredentialFiles.Count; $i++) {
    $credFile = $CredentialFiles[$i]
    $fullPath = if ([System.IO.Path]::IsPathRooted($credFile)) {
      $credFile
    } else {
      Join-Path $CredentialsDir $credFile
    }
    if (Test-Path $fullPath) {
      $selectedCredentialFiles += Get-Item $fullPath
      Write-Host "Found credential file: $($credFile)"
    } else {
      Write-Warning "Credential file not found: $fullPath"
    }
  }
  if ($selectedCredentialFiles.Count -eq 0) { throw "No valid credential files specified" }
  $credentialFiles = $selectedCredentialFiles
} else {
  # Auto-discover credential files
  $credentialFiles = Get-ChildItem -Path $CredentialsDir -Filter "*.properties" | Sort-Object Name
  if ($credentialFiles.Count -eq 0) { throw "No credential files found in: $CredentialsDir" }
  Write-Host "Auto-discovered $($credentialFiles.Count) credential files"


if ($credentialFiles.Count -lt $Count) { 
  Write-Warning "Only $($credentialFiles.Count) credential files available, but $Count instances requested. Some instances will reuse credentials."


# Build classpath
$deps = (Get-Content $ClassPathFile -Raw).Trim()
$classes = Join-Path $ProjectDir 'runelite-client\target\classes'
if (-not (Test-Path $classes)) {
  Write-Warning "Classes not found at $classes. Build first:`n  mvn -q -f `"$ProjectDir\pom.xml`" -pl runelite-client -am package -DskipTests"

$cpFull = "$classes;$deps"

# Create base dirs
New-Item -ItemType Directory -Force -Path $BaseDir | Out-Null
New-Item -ItemType Directory -Force -Path $ExportsBase | Out-Null

# Track PIDs so we can stop cleanly later
$pidFile = Join-Path $BaseDir "runelite-pids.txt"
if (Test-Path $pidFile) { Remove-Item $pidFile -Force }

for ($i = 0; $i -lt $Count; $i++) {
  $inst = "inst_$i"
  $instHome = Join-Path $BaseDir $inst       # per-instance "user.home"
  $exp  = Join-Path $ExportsBase $inst   # per-instance gamestate export dir
  $port = $BasePort + $i

  New-Item -ItemType Directory -Force -Path $instHome | Out-Null
  New-Item -ItemType Directory -Force -Path $exp  | Out-Null

  # Copy credentials for this instance BEFORE launching
  $credentialIndex = $i % $credentialFiles.Count  # Cycle through available credentials
  $sourceCredFile = $credentialFiles[$credentialIndex].FullName
  $targetCredFile = Join-Path $instHome "credentials.properties"
  
  Write-Host "Copying credentials: $($credentialFiles[$credentialIndex].Name) -> $targetCredFile"
  Copy-Item -Path $sourceCredFile -Destination $targetCredFile -Force
  Write-Host "✓ Credentials copied successfully"

  # JVM system props (preferred; consistent across Windows)
  $jvmProps = @(
    '-XX:TieredStopAtLevel=1',
    '-ea',
    '-Dsun.java2d.d3d=false',
    '-Dsun.java2d.noddraw=true',
    "-Duser.home=$instHome",                 # isolates RuneLite config/cache
    "-Drl.instance=$i",                  # your plugin can log/label this
    "-Drl.ipc.port=$port",               # your IpcInput plugin listens here
    "-Drl.export.dir=$exp"               # your exporter writes JSONs here
  ) -join ' '

  $args = "$jvmProps -cp `"$cpFull`" net.runelite.client.RuneLite --debug --developer-mode"

  # Debug output
  Write-Host "Launching with: $JavaExe $args"

  # Launch and capture output to see errors
  try {
    $logFile = Join-Path $instHome "runelite.log"
    $errorFile = Join-Path $instHome "runelite-errors.log"
    $p = Start-Process -FilePath $JavaExe -ArgumentList $args -WindowStyle Normal -PassThru -RedirectStandardOutput $logFile -RedirectStandardError $errorFile -ErrorAction Stop
    "$($p.Id),$i,$port,$instHome" | Out-File -Append -Encoding ascii $pidFile
    Write-Host "Launched instance #$i  PID=$($p.Id)  Port=$port  Home=$instHome"
    
    # Wait a moment and check if process is still running
    Start-Sleep -Seconds 3
    if (Get-Process -Id $p.Id -ErrorAction SilentlyContinue) {
      Write-Host "[OK] Instance #$i is still running"
    } else {
      Write-Host "[ERROR] Instance #$i crashed or exited"
      if (Test-Path $errorFile) {
        Write-Host "Check error file: $errorFile"
        Write-Host "Last few lines of error log:"
        Get-Content $errorFile -Tail 10
      }
      if (Test-Path $logFile) {
        Write-Host "Check log file: $logFile"
        Write-Host "Last few lines of log:"
        Get-Content $logFile -Tail 5
      }
    }
  } catch {
    $errorMsg = $_.Exception.Message
    Write-Host "[ERROR] Failed to launch instance #$i : $errorMsg"
  }

  # Wait between instances (except for the last one)
  if ($i -lt $Count - 1) {
    Write-Host "Waiting $DelaySeconds seconds before launching next instance..."
    Start-Sleep -Seconds $DelaySeconds
  }
}

Write-Host "Done. PIDs stored in $pidFile"
Write-Host "Credential mapping:"
for ($i = 0; $i -lt $Count; $i++) {
  $credentialIndex = $i % $credentialFiles.Count
  Write-Host "  Instance $i -> $($credentialFiles[$credentialIndex].Name)"
}

