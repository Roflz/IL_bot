#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Launch multiple RuneLite instances with IPC support.

.DESCRIPTION
    Launches multiple RuneLite client instances, each with its own configuration,
    credentials, and IPC port. Supports auto-detection of paths and configuration
    via JSON file.

.PARAMETER Count
    Number of instances to launch (default: 10)

.PARAMETER BasePort
    First IPC port; instances use BasePort + i (default: 17000)

.PARAMETER DelaySeconds
    Delay between instance launches (default: 0)

.PARAMETER CredentialFiles
    Specific credential files to use (optional, otherwise auto-discovers)

.PARAMETER BuildMaven
    Whether to build Maven project before launching

.PARAMETER DefaultWorld
    Specific world to use (0 = random from valid list)

.PARAMETER ConfigFile
    Path to configuration JSON file (default: launch-config.json in script directory)

.EXAMPLE
    .\launch-runelite.ps1 -Count 5 -BasePort 17000
    .\launch-runelite.ps1 -ConfigFile "custom-config.json" -Count 3
#>

param(
  [int]$Count = 10,
  [int]$BasePort = 17000,
  [int]$DelaySeconds = 0,
  [string[]]$CredentialFiles = @(),
  [switch]$BuildMaven,
  [int]$DefaultWorld = 0,
  [string]$ConfigFile = ""
)

$ErrorActionPreference = "Continue"

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ScriptDir) {
    $ScriptDir = $PSScriptRoot
}
if (-not $ScriptDir) {
    $ScriptDir = Get-Location
}

# -----------------------------------------------------------------------------
# Configuration loading and auto-detection
# -----------------------------------------------------------------------------

function Get-ScriptDirectory {
    return $ScriptDir
}

function Load-Configuration {
    param([string]$ConfigPath)
    
    $defaultConfig = @{
        version = "1.0"
        autoDetect = @{
            enabled = $true
            java = $true
            maven = $true
            projectDir = $true
            credentialsDir = $true
            baseDir = $true
            exportsBase = $true
            classPathFile = $true
        }
        paths = @{
            projectDir = ""
            classPathFile = ""
            javaExe = ""
            baseDir = ""
            exportsBase = ""
            credentialsDir = ""
        }
        launch = @{
            basePort = 17000
            delaySeconds = 0
            defaultWorld = 0
            buildMaven = $true
        }
    }
    
    if (Test-Path $ConfigPath) {
        try {
            $configContent = Get-Content $ConfigPath -Raw | ConvertFrom-Json
            $config = @{
                version = $configContent.version
                autoDetect = @{
                    enabled = if ($configContent.autoDetect.enabled -ne $null) { $configContent.autoDetect.enabled } else { $true }
                    java = if ($configContent.autoDetect.java -ne $null) { $configContent.autoDetect.java } else { $true }
                    maven = if ($configContent.autoDetect.maven -ne $null) { $configContent.autoDetect.maven } else { $true }
                    projectDir = if ($configContent.autoDetect.projectDir -ne $null) { $configContent.autoDetect.projectDir } else { $true }
                    credentialsDir = if ($configContent.autoDetect.credentialsDir -ne $null) { $configContent.autoDetect.credentialsDir } else { $true }
                    baseDir = if ($configContent.autoDetect.baseDir -ne $null) { $configContent.autoDetect.baseDir } else { $true }
                    exportsBase = if ($configContent.autoDetect.exportsBase -ne $null) { $configContent.autoDetect.exportsBase } else { $true }
                    classPathFile = if ($configContent.autoDetect.classPathFile -ne $null) { $configContent.autoDetect.classPathFile } else { $true }
                }
                paths = @{
                    projectDir = if ($configContent.paths.projectDir) { $configContent.paths.projectDir } else { "" }
                    classPathFile = if ($configContent.paths.classPathFile) { $configContent.paths.classPathFile } else { "" }
                    javaExe = if ($configContent.paths.javaExe) { $configContent.paths.javaExe } else { "" }
                    baseDir = if ($configContent.paths.baseDir) { $configContent.paths.baseDir } else { "" }
                    exportsBase = if ($configContent.paths.exportsBase) { $configContent.paths.exportsBase } else { "" }
                    credentialsDir = if ($configContent.paths.credentialsDir) { $configContent.paths.credentialsDir } else { "" }
                }
                launch = @{
                    basePort = if ($configContent.launch.basePort) { $configContent.launch.basePort } else { 17000 }
                    delaySeconds = if ($configContent.launch.delaySeconds -ne $null) { $configContent.launch.delaySeconds } else { 0 }
                    defaultWorld = if ($configContent.launch.defaultWorld -ne $null) { $configContent.launch.defaultWorld } else { 0 }
                    buildMaven = if ($configContent.launch.buildMaven -ne $null) { $configContent.launch.buildMaven } else { $true }
                }
            }
            Write-Host "[Config] Loaded configuration from: $ConfigPath" -ForegroundColor Green
            return $config
        } catch {
            Write-Host "[Config] Failed to load config file, using defaults: $($_.Exception.Message)" -ForegroundColor Yellow
            return $defaultConfig
        }
    } else {
        Write-Host "[Config] Config file not found, using defaults: $ConfigPath" -ForegroundColor Yellow
        return $defaultConfig
    }
}

function Find-JavaExecutable {
    # Check JAVA_HOME
    if ($env:JAVA_HOME) {
        $javaExe = Join-Path $env:JAVA_HOME "bin\java.exe"
        if (Test-Path $javaExe) {
            return $javaExe
        }
        $javawExe = Join-Path $env:JAVA_HOME "bin\javaw.exe"
        if (Test-Path $javawExe) {
            return $javawExe
        }
    }
    
    # Check common installation paths (JDK 11 preferred)
    $commonPaths = @(
        "C:\Program Files\Java\jdk-11*",
        "C:\Program Files (x86)\Java\jdk-11*",
        "$env:ProgramFiles\Java\jdk-11*",
        "${env:ProgramFiles(x86)}\Java\jdk-11*",
        "C:\Program Files\Java\jdk-*",
        "C:\Program Files (x86)\Java\jdk-*"
    )
    
    foreach ($pathPattern in $commonPaths) {
        $paths = Get-ChildItem -Path $pathPattern -ErrorAction SilentlyContinue | Sort-Object Name -Descending
        if ($paths) {
            $javaExe = Join-Path $paths[0].FullName "bin\java.exe"
            if (Test-Path $javaExe) {
                return $javaExe
            }
            $javawExe = Join-Path $paths[0].FullName "bin\javaw.exe"
            if (Test-Path $javawExe) {
                return $javawExe
            }
        }
    }
    
    # Check PATH
    try {
        $javaCmd = Get-Command java -ErrorAction Stop
        return $javaCmd.Source
    } catch {
        # Java not in PATH
    }
    
    return $null
}

function Find-MavenExecutable {
    try {
        $mvnCmd = Get-Command mvn -ErrorAction Stop
        return $mvnCmd.Source
    } catch {
        return $null
    }
}

function Find-RuneLiteProject {
    $scriptDir = Get-ScriptDirectory
    
    # Check common locations relative to script
    $possiblePaths = @(
        Join-Path $scriptDir "..\..\IdeaProjects\runelite",
        Join-Path $scriptDir "..\runelite",
        Join-Path $env:USERPROFILE "IdeaProjects\runelite",
        "D:\IdeaProjects\runelite",
        "C:\IdeaProjects\runelite"
    )
    
    foreach ($path in $possiblePaths) {
        $resolvedPath = [System.IO.Path]::GetFullPath($path)
        if (Test-Path (Join-Path $resolvedPath "pom.xml")) {
            return $resolvedPath
        }
    }
    
    return $null
}

function AutoDetect-Paths {
    param([hashtable]$Config)
    
    $scriptDir = Get-ScriptDirectory
    $detected = @{}
    
    # Java
    if ($Config.autoDetect.java -and -not $Config.paths.javaExe) {
        $javaExe = Find-JavaExecutable
        if ($javaExe) {
            $detected.javaExe = $javaExe
            Write-Host "[AutoDetect] Found Java: $javaExe" -ForegroundColor Green
        } else {
            Write-Host "[AutoDetect] Java not found, please configure manually" -ForegroundColor Yellow
        }
    } else {
        $detected.javaExe = $Config.paths.javaExe
    }
    
    # Maven
    if ($Config.autoDetect.maven) {
        $mvnExe = Find-MavenExecutable
        if (-not $mvnExe) {
            Write-Host "[AutoDetect] Maven not found in PATH" -ForegroundColor Yellow
        }
    }
    
    # RuneLite Project
    if ($Config.autoDetect.projectDir -and -not $Config.paths.projectDir) {
        $projectDir = Find-RuneLiteProject
        if ($projectDir) {
            $detected.projectDir = $projectDir
            Write-Host "[AutoDetect] Found RuneLite project: $projectDir" -ForegroundColor Green
        } else {
            Write-Host "[AutoDetect] RuneLite project not found, please configure manually" -ForegroundColor Yellow
        }
    } else {
        $detected.projectDir = $Config.paths.projectDir
    }
    
    # Credentials directory (relative to script)
    if ($Config.autoDetect.credentialsDir -and -not $Config.paths.credentialsDir) {
        $credDir = Join-Path $scriptDir "credentials"
        if (Test-Path $credDir) {
            $detected.credentialsDir = $credDir
            Write-Host "[AutoDetect] Found credentials directory: $credDir" -ForegroundColor Green
        } else {
            Write-Host "[AutoDetect] Credentials directory not found: $credDir" -ForegroundColor Yellow
        }
    } else {
        $detected.credentialsDir = $Config.paths.credentialsDir
    }
    
    # Base directory (relative to script or user profile)
    if ($Config.autoDetect.baseDir -and -not $Config.paths.baseDir) {
        $baseDir = Join-Path $scriptDir "instances"
        $detected.baseDir = $baseDir
        Write-Host "[AutoDetect] Using base directory: $baseDir" -ForegroundColor Green
    } else {
        $detected.baseDir = $Config.paths.baseDir
    }
    
    # Exports base (relative to script or user profile)
    if ($Config.autoDetect.exportsBase -and -not $Config.paths.exportsBase) {
        $exportsBase = Join-Path $scriptDir "exports"
        $detected.exportsBase = $exportsBase
        Write-Host "[AutoDetect] Using exports directory: $exportsBase" -ForegroundColor Green
    } else {
        $detected.exportsBase = $Config.paths.exportsBase
    }
    
    # Classpath file (relative to script)
    if ($Config.autoDetect.classPathFile -and -not $Config.paths.classPathFile) {
        $classPathFile = Join-Path $scriptDir "rl-classpath.txt"
        $detected.classPathFile = $classPathFile
        Write-Host "[AutoDetect] Using classpath file: $classPathFile" -ForegroundColor Green
    } else {
        $detected.classPathFile = $Config.paths.classPathFile
    }
    
    return $detected
}

# Load configuration
if (-not $ConfigFile) {
    $ConfigFile = Join-Path $ScriptDir "launch-config.json"
}

$config = Load-Configuration -ConfigPath $ConfigFile

# Override with command-line parameters
if ($BasePort -ne 17000) { $config.launch.basePort = $BasePort }
if ($DelaySeconds -ne 0) { $config.launch.delaySeconds = $DelaySeconds }
if ($DefaultWorld -ne 0) { $config.launch.defaultWorld = $DefaultWorld }
if ($BuildMaven) { $config.launch.buildMaven = $true }

# Auto-detect paths
$paths = AutoDetect-Paths -Config $config

# Use detected or configured paths
$ProjectDir = $paths.projectDir
$ClassPathFile = $paths.classPathFile
$JavaExe = $paths.javaExe
$BaseDir = $paths.baseDir
$ExportsBase = $paths.exportsBase
$CredentialsDir = $paths.credentialsDir

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
if (-not $JavaExe -or -not (Test-Path $JavaExe)) {
    throw "Java not found: $JavaExe. Please install Java JDK 11 or configure the path in launch-config.json"
}
if (-not $ProjectDir -or -not (Test-Path $ProjectDir)) {
    throw "Project dir not found: $ProjectDir. Please configure the path in launch-config.json"
}
if (-not $CredentialsDir -or -not (Test-Path $CredentialsDir)) {
    throw "Credentials directory not found: $CredentialsDir. Please configure the path in launch-config.json"
}

# Prefer javaw.exe (no console window)
if ($JavaExe -match '\\java\.exe$') {
    $JavaExe = $JavaExe -replace '\\java\.exe$', '\javaw.exe'
}
if (-not (Test-Path $JavaExe)) {
    Write-Host "[Warning] javaw.exe not found, using java.exe" -ForegroundColor Yellow
    $JavaExe = $JavaExe -replace '\\javaw\.exe$', '\java.exe'
    if (-not (Test-Path $JavaExe)) {
        throw "Java executable not found: $JavaExe"
    }
}

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
# Build same as normal RuneLite: full clean install from root so runelite-api,
# injected-client, and runelite-client are all built and installed in one go.
# Then classpath resolution uses that same install; no version mismatch.
# -----------------------------------------------------------------------------
$classes = Join-Path $ProjectDir 'runelite-client\target\classes'

if ($config.launch.buildMaven) {
    Write-Host "Building RuneLite (full clean install, same as upstream)..."
    $mvnExe = Find-MavenExecutable
    if (-not $mvnExe) {
        throw "Maven not found. Please install Maven or disable BuildMaven option."
    }
    Write-Host "Running: $mvnExe -q -f `"$ProjectDir\pom.xml`" clean install -DskipTests"
    & $mvnExe -q -f "$ProjectDir\pom.xml" clean install -DskipTests
    if ($LASTEXITCODE -ne 0) { throw "Maven build failed with exit code $LASTEXITCODE" }
} else {
    Write-Host "Skipping Maven build (BuildMaven = false)"
}

Write-Host "Regenerating classpath file..."
$mvnExe = Find-MavenExecutable
if (-not $mvnExe) {
    throw "Maven not found. Please install Maven to generate classpath."
}
$mvnArgs = @("-q", "-f", "$ProjectDir\pom.xml", "-pl", "runelite-client", "dependency:build-classpath", "-Dmdep.outputFile=$ClassPathFile")
$mvnOutput = & $mvnExe @mvnArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Basic command failed, trying with different parameters..."
    $mvnArgs = @("-q", "-f", "$ProjectDir\pom.xml", "-pl", "runelite-client", "dependency:build-classpath", "-Dmdep.outputFile=$ClassPathFile", "-Dmdep.includeScope=compile")
    $mvnOutput = & $mvnExe @mvnArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Still failed, trying without -q flag to see error..."
        $mvnArgs = @("-f", "$ProjectDir\pom.xml", "-pl", "runelite-client", "dependency:build-classpath", "-Dmdep.outputFile=$ClassPathFile")
        $mvnOutput = & $mvnExe @mvnArgs
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
# Don't delete existing PID file - append to it so multiple launches are tracked
$pidFile = Join-Path $BaseDir "runelite-pids.txt"
# Note: We append to the file instead of overwriting, so multiple separate launches are tracked

Write-Host "Starting to launch $Count instances..."

# -----------------------------------------------------------------------------
# Launch loop
# -----------------------------------------------------------------------------
for ($i = 0; $i -lt $Count; $i++) {

    Write-Host "Starting instance $i..."
    $inst = "inst_$i"
    $instHome = Join-Path $BaseDir $inst
    $exp      = Join-Path $ExportsBase $inst
    $port     = $config.launch.basePort + $i

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
    $worldForInstance = $config.launch.defaultWorld
    if ($config.launch.defaultWorld -eq 0) {
        # Random selection from valid F2P worlds
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
        $delay = $config.launch.delaySeconds
        Write-Host "Waiting $delay seconds before launching next instance..."
        Start-Sleep -Seconds $delay
    }
}

Write-Host "Done. PIDs stored in $pidFile"
Write-Host "Configuration Summary:"
Write-Host "  Default World: $($config.launch.defaultWorld)"
Write-Host "  Auto World Selection: Enabled"
Write-Host "Credential mapping:"
for ($i = 0; $i -lt $Count; $i++) {
    $credentialIndex = $i % $credentialFiles.Count
    $credName = $credentialFiles[$credentialIndex].Name
    if (-not $credName) { $credName = Split-Path $credentialFiles[$credentialIndex] -Leaf }
    Write-Host "  Instance $i -> $credName"
}
