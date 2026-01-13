param(
  [int]$Count = 10,                 # how many instances to launch
  [int]$BasePort = 5500,            # first IPC port; instances use BasePort + i
  [string]$ProjectDir = "D:\IdeaProjects\runelite",    # path to RuneLite project
  [string]$ClassPathFile = "D:\repos\bot_runelite_IL\ilbot\ui\simple_recorder\rl-classpath.txt", # path to classpath file
  [string]$JavaExe = "C:\Program Files\Java\jdk-11.0.2\bin\java.exe", # Java executable (will auto-switch to javaw.exe)
  [string]$BaseDir = "D:\bots\instances",              # base dir to store per-instance homes
  [string]$ExportsBase = "D:\bots\exports",            # where your gamestate JSONs go
  [string]$CredentialsDir = "D:\repos\bot_runelite_IL\credentials", # directory containing credential files
  [int]$DelaySeconds = 10,                              # delay between instance launches
  [string[]]$CredentialFiles = @(),                     # specific credential files to use (optional)
  [switch]$BuildMaven,                                  # whether to build Maven project
  [int]$DefaultWorld = 0                                 # specific world to use (0 = random from valid list)
)

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
if (-not (Test-Path $JavaExe)) { throw "Java not found: $JavaExe" }
if (-not (Test-Path $ProjectDir)) { throw "Project dir not found: $ProjectDir" }
if (-not (Test-Path $CredentialsDir)) { throw "Credentials directory not found: $CredentialsDir" }

# Prefer javaw.exe (no console window)
if ($JavaExe -match '\\java\.exe$') {
  $JavaExe = $JavaExe -replace '\\java\.exe$', '\javaw.exe'
}
if (-not (Test-Path $JavaExe)) { throw "javaw.exe not found: $JavaExe" }

# -----------------------------------------------------------------------------
# Win32 P/Invoke loader
# -----------------------------------------------------------------------------
function Ensure-NativeMethods {
  $t = [AppDomain]::CurrentDomain.GetAssemblies() |
       ForEach-Object { $_.GetTypes() } |
       Where-Object { $_.FullName -eq 'Win32Native' } |
       Select-Object -First 1
  if ($t) { Write-Host "[PInvoke] Win32Native already loaded." -ForegroundColor DarkGray; return $true }

  Write-Host "[PInvoke] Loading Win32Native..." -ForegroundColor DarkGray
  try {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class Win32Native {
  [DllImport("user32.dll", SetLastError=true)]
  public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
}
'@
    Write-Host "[PInvoke] Win32Native loaded successfully." -ForegroundColor Green
    return $true
  } catch {
    Write-Host "[PInvoke] Add-Type failed: $($_.Exception.Message)" -ForegroundColor Red
    return $false
  }
}

# -----------------------------------------------------------------------------
# Helpers to locate the REAL RuneLite client window (not the launcher splash)
# -----------------------------------------------------------------------------

# WMI wrapper to get ParentProcessId etc.
function Get-ProcessCimById {
  param([int]$Pid)
  try { Get-CimInstance Win32_Process -Filter "ProcessId=$Pid" -ErrorAction Stop } catch { $null }
}

# Find the child javaw.exe that owns the "RuneLite" window for this launch
function Find-ClientProcess {
  param(
    [Parameter(Mandatory=$true)][System.Diagnostics.Process]$LauncherProc,
    [int]$TimeoutSec = 25,
    [string]$Tag = ""
  )
  $deadline = (Get-Date).AddSeconds($TimeoutSec)

  do {
    Start-Sleep -Milliseconds 300

    # 1) Prefer a direct child of the launcher PID
    $kids = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($LauncherProc.Id) AND Name='javaw.exe'" -ErrorAction SilentlyContinue
    if ($kids) {
      # Map to .NET processes to read MainWindow
      foreach ($kid in $kids) {
        $p = Get-Process -Id $kid.ProcessId -ErrorAction SilentlyContinue
        if ($p) {
          $p.Refresh()
          if ($p.MainWindowHandle -ne 0 -and $p.MainWindowTitle -eq 'RuneLite') {
            Write-Host ("[WinFind] Found child client window 0x{0:X} (PID {1}) Title='{2}' {3}" -f $p.MainWindowHandle, $p.Id, $p.MainWindowTitle, $Tag) -ForegroundColor Green
            return $p
          }
        }
      }
    }

    # 2) Fallback: any javaw with exact title 'RuneLite' started after the launcher
    $cands = Get-Process -Name javaw -ErrorAction SilentlyContinue | Where-Object {
      $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -eq 'RuneLite' -and $_.StartTime -ge $LauncherProc.StartTime
    } | Sort-Object StartTime
    if ($cands) {
      $p = $cands | Select-Object -First 1
      Write-Host ("[WinFind] Found fallback client window 0x{0:X} (PID {1}) Title='{2}' {3}" -f $p.MainWindowHandle, $p.Id, $p.MainWindowTitle, $Tag) -ForegroundColor Green
      return $p
    }

  } while ((Get-Date) -lt $deadline)

  Write-Host "[WinFind] Timed out waiting for client window 'RuneLite' $Tag" -ForegroundColor Yellow
  return $null
}

function Maximize-Window {
  param(
    [Parameter(Mandatory=$true)][System.Diagnostics.Process]$Process,
    [string]$Tag = ""
  )
  if (-not (Ensure-NativeMethods)) { Write-Host "[Max] Cannot maximize: P/Invoke not available." -ForegroundColor Red; return $false }
  $SW_MAXIMIZE = 3
  try {
    [Win32Native]::ShowWindow($Process.MainWindowHandle, $SW_MAXIMIZE) | Out-Null
    Write-Host ("[Max] Maximized window 0x{0:X} (PID {1}) Title='{2}' {3}" -f $Process.MainWindowHandle, $Process.Id, $Process.MainWindowTitle, $Tag) -ForegroundColor Green
    return $true
  } catch {
    Write-Host "[Max] ShowWindow failed: $($_.Exception.Message)" -ForegroundColor Red
    return $false
  }
}

# -----------------------------------------------------------------------------
# Collect credentials (specified or autodiscover)
# -----------------------------------------------------------------------------
if ($CredentialFiles.Count -gt 0) {
  Write-Host "Using specified credential files: $($CredentialFiles -join ', ')"
  $selected = @()
  foreach ($credFile in $CredentialFiles) {
    $fullPath = if ([System.IO.Path]::IsPathRooted($credFile)) { $credFile } else { Join-Path $CredentialsDir $credFile }
    if (Test-Path $fullPath) {
      $selected += Get-Item $fullPath
      Write-Host "Found credential file: $(Split-Path $fullPath -Leaf)"
    } else {
      Write-Warning "Credential file not found: $fullPath"
    }
  }
  if ($selected.Count -eq 0) { throw "No valid credential files specified" }
  $credentialFiles = $selected
} else {
  $credentialFiles = Get-ChildItem -Path $CredentialsDir -Filter "*.properties" | Sort-Object Name
  if ($credentialFiles.Count -eq 0) { throw "No credential files found in: $CredentialsDir" }
  Write-Host "Auto-discovered $($credentialFiles.Count) credential files"
}

if ($credentialFiles.Count -lt $Count) {
  Write-Warning "Only $($credentialFiles.Count) credential files available, but $Count instances requested. Some instances will reuse credentials."
}

# -----------------------------------------------------------------------------
# Build (optional) and regenerate classpath
# -----------------------------------------------------------------------------
$classes = Join-Path $ProjectDir 'runelite-client\target\classes'

if ($BuildMaven) {
  Write-Host "Building RuneLite project..."
  Write-Host "Running: mvn -q -f `"$ProjectDir\pom.xml`" -pl runelite-client -am package -DskipTests"
  & mvn -q -f "$ProjectDir\pom.xml" -pl runelite-client -am package -DskipTests
  if ($LASTEXITCODE -ne 0) { throw "Maven build failed with exit code $LASTEXITCODE" }
} else {
  Write-Host "Skipping Maven build (BuildMaven = false)"
}

Write-Host "Regenerating classpath file..."
$mvnArgs = @("-q", "-f", "$ProjectDir\pom.xml", "-pl", "runelite-client", "dependency:build-classpath", "-Dmdep.outputFile=$ClassPathFile")
$mvnOutput = & mvn @mvnArgs
if ($LASTEXITCODE -ne 0) {
  Write-Host "Basic command failed, trying with different parameters..."
  $mvnArgs = @("-q", "-f", "$ProjectDir\pom.xml", "-pl", "runelite-client", "dependency:build-classpath", "-Dmdep.outputFile=$ClassPathFile", "-Dmdep.includeScope=compile")
  $mvnOutput = & mvn @mvnArgs
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Still failed, trying without -q flag to see error..."
    $mvnArgs = @("-f", "$ProjectDir\pom.xml", "-pl", "runelite-client", "dependency:build-classpath", "-Dmdep.outputFile=$ClassPathFile")
    $mvnOutput = & mvn @mvnArgs
    if ($LASTEXITCODE -ne 0) {
      throw "Failed to generate classpath file. Maven output: $mvnOutput"
    }
  }
}

if (Test-Path $ClassPathFile) {
  $deps = (Get-Content $ClassPathFile -Raw).Trim()
  $cpFull = "$classes;$deps"
} else {
  throw "Classpath file missing after generation: $ClassPathFile"
}

# -----------------------------------------------------------------------------
# Create base dirs
# -----------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $BaseDir | Out-Null
New-Item -ItemType Directory -Force -Path $ExportsBase | Out-Null

# Track PIDs so we can stop cleanly later
$pidFile = Join-Path $BaseDir "runelite-pids.txt"
if (Test-Path $pidFile) { Remove-Item $pidFile -Force }

Write-Host "Starting to launch $Count instances..."

# -----------------------------------------------------------------------------
# Launch loop
# -----------------------------------------------------------------------------
for ($i = 0; $i -lt $Count; $i++) {

  Write-Host "Starting instance $i..."
  $inst = "inst_$i"
  $instHome = Join-Path $BaseDir $inst              # per-instance "user.home"
  $exp      = Join-Path $ExportsBase $inst          # per-instance export dir
  $port     = $BasePort + $i

  New-Item -ItemType Directory -Force -Path $instHome | Out-Null
  New-Item -ItemType Directory -Force -Path $exp      | Out-Null

  # Copy credentials for this instance BEFORE launching
  $credentialIndex = $i % $credentialFiles.Count
  $sourceCredFile = $credentialFiles[$credentialIndex].FullName
  if (-not $sourceCredFile) { $sourceCredFile = $credentialFiles[$credentialIndex] }
  $targetCredFile = Join-Path (Join-Path $instHome ".runelite") "credentials.properties"

  $credName = $credentialFiles[$credentialIndex].Name
  if (-not $credName) { $credName = Split-Path $credentialFiles[$credentialIndex] -Leaf }

  Write-Host "Copying credentials: $credName -> $targetCredFile"
  $configDir = Join-Path $instHome ".runelite"
  if (-not (Test-Path $configDir)) { New-Item -ItemType Directory -Force -Path $configDir | Out-Null }
  if ($sourceCredFile -and (Test-Path $sourceCredFile)) {
    Copy-Item -Path $sourceCredFile -Destination $targetCredFile -Force
    Write-Host "Credentials copied successfully"
  } else {
    Write-Warning "Source credential file not found or is null: $sourceCredFile"
  }

  # Clear profiles2 dir (fresh start)
  $instanceProfiles2Dir = Join-Path $configDir "profiles2"
  Write-Host "Clearing instance profiles2 directory: $instanceProfiles2Dir"
  if (Test-Path $instanceProfiles2Dir) {
    Remove-Item -Path $instanceProfiles2Dir -Recurse -Force
  }

  # Determine world for this instance
  $worldForInstance = $DefaultWorld
  if ($DefaultWorld -eq 0) {
    # Random selection from valid F2P worlds (excluding skill total, Grid Master, Fresh Start)
    $validWorlds = @(308, 316, 326, 379, 383, 398, 417, 437, 455, 469, 483, 497, 499, 537, 552, 553, 554, 555, 571)
    $worldForInstance = $validWorlds | Get-Random
    Write-Host "  Random world selected: $worldForInstance"
  }

  # Write settings.properties
  $instanceSettingsFile = Join-Path $configDir "settings.properties"
  $settingsContent = @"
  
runelite.ipcinputplugin=true
ipcinput.port=$port
ipcinput.mode=AWT
ipcinput.hoverDelayMs=10
defaultworld.lastWorld=$worldForInstance
defaultworld.useLastWorld=true
runelite.logouttimerplugin=true
logouttimer.idleTimeout=25
menuentryswapper.swapQuick=true
menuentryswapper.swapGEItemCollect=DEFAULT
menuentryswapper.swapBait=false
menuentryswapper.swapJewelleryBox=false
runelite.menuentryswapperplugin=true
menuentryswapper.npcShiftClickWalkHere=true
menuentryswapper.swapAdmire=true
menuentryswapper.swapHomePortal=HOME
menuentryswapper.shopBuy=OFF
menuentryswapper.swapDepositItems=false
menuentryswapper.swapArdougneCloak=WEAR
menuentryswapper.swapTan=false
menuentryswapper.swapStairsShiftClick=CLIMB
menuentryswapper.swapTrade=true
menuentryswapper.swapBoxTrap=true
menuentryswapper.swapPay=true
menuentryswapper.swapBones=false
menuentryswapper.groundItemShiftClickWalkHere=true
menuentryswapper.swapEssenceMineTeleport=false
menuentryswapper.shiftClickCustomization=true
menuentryswapper.swapTemporossLeave=false
menuentryswapper.swapHerbs=false
menuentryswapper.bankDepositShiftClick=OFF
menuentryswapper.objectLeftClickCustomization=true
menuentryswapper.bankWithdrawShiftClick=OFF
menuentryswapper.swapMorytaniaLegs=WEAR
menuentryswapper.swapHarpoon=false
menuentryswapper.swapBanker=true
menuentryswapper.swapDesertAmulet=WEAR
menuentryswapper.swapPick=false
menuentryswapper.shopSell=OFF
menuentryswapper.swapHelp=true
menuentryswapper.swapAbyssTeleport=true
menuentryswapper.swapAssignment=true
menuentryswapper.objectShiftClickWalkHere=true
menuentryswapper.swapFairyRing=LAST_DESTINATION
menuentryswapper.leftClickCustomization=true
menuentryswapper.swapTravel=true
menuentryswapper.swapExchange=true
menuentryswapper.swapKaramjaGloves=WEAR
menuentryswapper.swapRadasBlessing=EQUIP
menuentryswapper.swapGEAbort=false
menuentryswapper.swapPortalNexus=false
menuentryswapper.swapPrivate=false
menuentryswapper.swapTeleToPoh=false
menuentryswapper.swapBirdhouseEmpty=true
menuentryswapper.swapChase=true
menuentryswapper.swapStairsLeftClick=CLIMB
menuentryswapper.swapTeleportItem=false
menuentryswapper.npcLeftClickCustomization=true
menuentryswapper.removeDeadNpcMenus=false
keyremapping.f10=48\:0
keyremapping.f12=61\:0
keyremapping.f11=45\:0
keyremapping.left=65\:0
runelite.keyremappingplugin=true
keyremapping.f1=49\:0
keyremapping.f3=51\:0
keyremapping.f2=50\:0
keyremapping.cameraRemap=true
keyremapping.fkeyRemap=false
keyremapping.esc=27\:0
keyremapping.control=0\:128
keyremapping.f9=57\:0
keyremapping.f8=56\:0
keyremapping.f5=53\:0
keyremapping.f4=52\:0
keyremapping.f7=55\:0
keyremapping.f6=54\:0
keyremapping.up=87\:0
keyremapping.space=32\:0
keyremapping.down=83\:0
keyremapping.right=68\:0
"@
  Write-Host "Writing settings to: $instanceSettingsFile"
  $settingsContent | Out-File -FilePath $instanceSettingsFile -Encoding UTF8

  Write-Host "Configured RuneLite settings for instance #$i"
  Write-Host "  IpcInput: port=$port, enabled=true"
  Write-Host "  StateExporter2: dir=$exp, enabled=true"
  Write-Host "  World Selection: world=$worldForInstance, auto-select=true"
  Write-Host "  Logout Timer: time=25 mins, enabled=true"

  Write-Host "Waiting 5 seconds for files to be fully written..."
  Start-Sleep -Seconds 5

  # JVM system props
  $jvmProps = @(
    '-ea',
    '-XX:TieredStopAtLevel=1',
    '-Dsun.java2d.d3d=false',
    '-Dsun.java2d.noddraw=true',
    "-Duser.home=$instHome",
    "-Drl.instance=$i"
  ) -join ' '

  # Build Java args; launching RuneLite main class directly
  $args = "$jvmProps -cp `"$cpFull`" net.runelite.client.RuneLite ==debug --developer-mode"

  # Launch and capture the process (this is the LAUNCHER)
  try {
    $logFile = Join-Path $instHome "runelite.log"
    $errorFile = Join-Path $instHome "runelite-errors.log"

    $launcher = Start-Process -FilePath $JavaExe -ArgumentList $args -PassThru -WorkingDirectory $ProjectDir -ErrorAction Stop

    "$($launcher.Id),$i,$port,$instHome" | Out-File -Append -Encoding ascii $pidFile
    Write-Host ("[RuneLite] Launched instance #{0}  PID={1}  Port={2}  Home={3}" -f $i, $launcher.Id, $port, $instHome)

    # Debug tag to correlate logs
    $tag = "port:$port inst:$i"

    # Quick health check (let launcher/splash run; client window spawns shortly)
    Start-Sleep -Seconds 3
    if (Get-Process -Id $launcher.Id -ErrorAction SilentlyContinue) {
      Write-Host "[OK] Instance #$i is still running"

      # Find actual client window titled EXACTLY 'RuneLite' related to this launcher
      $clientProc = Find-ClientProcess -LauncherProc $launcher -TimeoutSec 25 -Tag $tag
      if ($clientProc -and $clientProc.MainWindowHandle -ne 0) {
        $null = Maximize-Window -Process $clientProc -Tag $tag
      } else {
        Write-Host "[Max] Could not locate 'RuneLite' client window to maximize $tag" -ForegroundColor Yellow
      }
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

  # Delay between instances (except the last)
  if ($i -lt $Count - 1) {
    Write-Host "Waiting $DelaySeconds seconds before launching next instance..."
    Start-Sleep -Seconds $DelaySeconds
  }
}

Write-Host "Done. PIDs stored in $pidFile"
Write-Host "Configuration Summary:"
Write-Host "  Default World: $DefaultWorld"
Write-Host "  Auto World Selection: Enabled"
Write-Host "Credential mapping:"
for ($i = 0; $i -lt $Count; $i++) {
  $credentialIndex = $i % $credentialFiles.Count
  $credName = $credentialFiles[$credentialIndex].Name
  if (-not $credName) { $credName = Split-Path $credentialFiles[$credentialIndex] -Leaf }
  Write-Host "  Instance $i -> $credName"
}
