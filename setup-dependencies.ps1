#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Setup script for installing all dependencies required to run the RuneLite launcher.

.DESCRIPTION
    This script automatically installs or verifies the following dependencies:
    - Java JDK 11 (via Chocolatey or manual download)
    - Apache Maven (via Chocolatey or manual download)
    - Chocolatey (if not already installed)

.PARAMETER Force
    Force reinstallation of dependencies even if they're already installed.

.PARAMETER SkipChocolatey
    Skip Chocolatey installation and use manual methods only.

.EXAMPLE
    .\setup-dependencies.ps1
    .\setup-dependencies.ps1 -Force
    .\setup-dependencies.ps1 -SkipChocolatey
#>

param(
    [switch]$Force,
    [switch]$SkipChocolatey
)

$ErrorActionPreference = "Continue"

# Colors for output
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

Write-Info "=========================================="
Write-Info "RuneLite Launcher Dependency Setup"
Write-Info "=========================================="
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Warning "Not running as Administrator. Some installations may require elevation."
    Write-Warning "If installations fail, try running PowerShell as Administrator."
    Write-Host ""
}

# -----------------------------------------------------------------------------
# Check/Install Chocolatey
# -----------------------------------------------------------------------------
function Install-Chocolatey {
    if ($SkipChocolatey) {
        Write-Info "Skipping Chocolatey installation (SkipChocolatey flag set)"
        return $false
    }
    
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Success "Chocolatey is already installed"
        return $true
    }
    
    Write-Info "Chocolatey not found. Installing Chocolatey..."
    Write-Warning "This requires Administrator privileges."
    
    if (-not $isAdmin) {
        Write-Error "Cannot install Chocolatey without Administrator privileges."
        Write-Info "Please run this script as Administrator, or install Chocolatey manually:"
        Write-Info "  https://chocolatey.org/install"
        return $false
    }
    
    try {
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        Write-Success "Chocolatey installed successfully"
        return $true
    } catch {
        Write-Error "Failed to install Chocolatey: $($_.Exception.Message)"
        return $false
    }
}

# -----------------------------------------------------------------------------
# Check/Install Java JDK 11
# -----------------------------------------------------------------------------
function Install-JavaJDK11 {
    Write-Info "Checking for Java JDK 11..."
    
    # Check if Java is already installed
    $javaPath = $null
    $javaVersion = $null
    
    # Check JAVA_HOME
    if ($env:JAVA_HOME) {
        $javaExe = Join-Path $env:JAVA_HOME "bin\java.exe"
        if (Test-Path $javaExe) {
            try {
                $versionOutput = & $javaExe -version 2>&1 | Out-String
                if ($versionOutput -match 'version "11\.') {
                    $javaPath = $javaExe
                    $javaVersion = ($versionOutput -split "`n")[0]
                }
            } catch {
                # Version check failed, continue searching
            }
        }
    }
    
    # Check common installation paths
    if (-not $javaPath) {
        $commonPaths = @(
            "C:\Program Files\Java\jdk-11*",
            "C:\Program Files (x86)\Java\jdk-11*",
            "$env:ProgramFiles\Java\jdk-11*",
            "${env:ProgramFiles(x86)}\Java\jdk-11*"
        )
        
        foreach ($pathPattern in $commonPaths) {
            $paths = Get-ChildItem -Path $pathPattern -ErrorAction SilentlyContinue | Sort-Object Name -Descending
            if ($paths) {
                $javaExe = Join-Path $paths[0].FullName "bin\java.exe"
                if (Test-Path $javaExe) {
                    try {
                        $versionOutput = & $javaExe -version 2>&1 | Out-String
                        if ($versionOutput -match 'version "11\.') {
                            $javaPath = $javaExe
                            $javaVersion = ($versionOutput -split "`n")[0]
                            break
                        }
                    } catch {
                        # Version check failed, continue searching
                    }
                }
            }
        }
    }
    
    # Check PATH
    if (-not $javaPath) {
        try {
            $javaExe = Get-Command java -ErrorAction Stop
            try {
                $versionOutput = & $javaExe.Source -version 2>&1 | Out-String
                if ($versionOutput -match 'version "11\.') {
                    $javaPath = $javaExe.Source
                    $javaVersion = ($versionOutput -split "`n")[0]
                }
            } catch {
                # Version check failed
            }
        } catch {
            # Java not in PATH
        }
    }
    
    if ($javaPath -and -not $Force) {
        Write-Success "Java JDK 11 found: $javaPath"
        Write-Info "  Version: $javaVersion"
        return $true
    }
    
    if ($Force) {
        Write-Info "Force flag set, reinstalling Java JDK 11..."
    } else {
        Write-Warning "Java JDK 11 not found. Installing..."
    }
    
    # Try Chocolatey installation
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Info "Installing Java JDK 11 via Chocolatey..."
        try {
            if ($isAdmin) {
                choco install openjdk11 -y --force=$Force
                Write-Success "Java JDK 11 installed via Chocolatey"
                
                # Refresh environment
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
                
                # Try to find the newly installed Java
                $javaExe = Get-Command java -ErrorAction SilentlyContinue
                if ($javaExe) {
                    try {
                        $versionOutput = & $javaExe.Source -version 2>&1 | Out-String
                        $versionLine = ($versionOutput -split "`n")[0]
                        Write-Info "  Installed version: $versionLine"
                    } catch {
                        Write-Info "  Java installed but version check failed"
                    }
                }
                return $true
            } else {
                Write-Warning "Chocolatey installation requires Administrator privileges."
                Write-Info "Please run this script as Administrator, or install manually:"
                Write-Info "  choco install openjdk11 -y"
            }
        } catch {
            Write-Error "Failed to install Java via Chocolatey: $($_.Exception.Message)"
        }
    }
    
    # Manual installation instructions
    Write-Warning "Automatic installation failed or skipped."
    Write-Info "Please install Java JDK 11 manually:"
    Write-Info "  1. Download from: https://adoptium.net/temurin/releases/?version=11"
    Write-Info "  2. Install the JDK (not JRE)"
    Write-Info "  3. Set JAVA_HOME environment variable to the JDK installation directory"
    Write-Info "  4. Add %JAVA_HOME%\bin to your PATH"
    Write-Info ""
    Write-Info "Or use Chocolatey (as Administrator):"
    Write-Info "  choco install openjdk11 -y"
    
    return $false
}

# -----------------------------------------------------------------------------
# Check/Install Maven
# -----------------------------------------------------------------------------
function Install-Maven {
    Write-Info "Checking for Apache Maven..."
    
    # Check if Maven is already installed
    try {
        $mvnExe = Get-Command mvn -ErrorAction Stop
        $versionOutput = & $mvnExe.Source -version 2>&1
        if ($versionOutput -match 'Apache Maven') {
            Write-Success "Maven found: $($mvnExe.Source)"
            Write-Info "  Version: $($versionOutput[0])"
            if (-not $Force) {
                return $true
            }
        }
    } catch {
        # Maven not found
    }
    
    if ($Force) {
        Write-Info "Force flag set, reinstalling Maven..."
    } else {
        Write-Warning "Maven not found. Installing..."
    }
    
    # Try Chocolatey installation
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Info "Installing Maven via Chocolatey..."
        try {
            if ($isAdmin) {
                choco install maven -y --force=$Force
                Write-Success "Maven installed via Chocolatey"
                
                # Refresh environment
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
                
                # Verify installation
                $mvnExe = Get-Command mvn -ErrorAction SilentlyContinue
                if ($mvnExe) {
                    $versionOutput = & $mvnExe.Source -version 2>&1
                    Write-Info "  Installed version: $($versionOutput[0])"
                }
                return $true
            } else {
                Write-Warning "Chocolatey installation requires Administrator privileges."
                Write-Info "Please run this script as Administrator, or install manually:"
                Write-Info "  choco install maven -y"
            }
        } catch {
            Write-Error "Failed to install Maven via Chocolatey: $($_.Exception.Message)"
        }
    }
    
    # Manual installation instructions
    Write-Warning "Automatic installation failed or skipped."
    Write-Info "Please install Maven manually:"
    Write-Info "  1. Download from: https://maven.apache.org/download.cgi"
    Write-Info "  2. Extract to a directory (e.g., C:\Program Files\Apache\maven)"
    Write-Info "  3. Add the bin directory to your PATH"
    Write-Info "  4. Set M2_HOME environment variable to the Maven installation directory"
    Write-Info ""
    Write-Info "Or use Chocolatey (as Administrator):"
    Write-Info "  choco install maven -y"
    
    return $false
}

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
Write-Info "Starting dependency installation..."
Write-Host ""

$chocoInstalled = Install-Chocolatey
Write-Host ""

$javaInstalled = Install-JavaJDK11
Write-Host ""

$mavenInstalled = Install-Maven
Write-Host ""

# Summary
Write-Info "=========================================="
Write-Info "Installation Summary"
Write-Info "=========================================="
Write-Host ""

if ($chocoInstalled) {
    Write-Success "[OK] Chocolatey"
} else {
    Write-Warning "[SKIP] Chocolatey (not required if dependencies are installed manually)"
}

if ($javaInstalled) {
    Write-Success "[OK] Java JDK 11"
} else {
    Write-Error "[FAIL] Java JDK 11 - Please install manually"
}

if ($mavenInstalled) {
    Write-Success "[OK] Apache Maven"
} else {
    Write-Error "[FAIL] Apache Maven - Please install manually"
}

Write-Host ""

if ($javaInstalled -and $mavenInstalled) {
    Write-Success "All required dependencies are installed!"
    Write-Info "You can now run the RuneLite launcher."
} else {
    Write-Warning "Some dependencies are missing. Please install them manually and run this script again."
    Write-Info "Or run this script as Administrator to attempt automatic installation."
    exit 1
}
