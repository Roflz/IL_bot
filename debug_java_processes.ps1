# PowerShell script to show detailed information about all java.exe processes
# Useful for debugging RuneLite window maximization timing

# Add Windows API types once (before the loop)
Add-Type @"
    using System;
    using System.Runtime.InteropServices;
    using System.Text;
    public class Win32 {
        [DllImport("user32.dll")]
        public static extern int EnumWindows(EnumWindowsProc enumProc, int lParam);
        public delegate bool EnumWindowsProc(IntPtr hWnd, int lParam);
        [DllImport("user32.dll")]
        public static extern int GetWindowThreadProcessId(IntPtr hWnd, out int lpdwProcessId);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        public static extern int GetClassName(IntPtr hWnd, StringBuilder lpClassName, int nMaxCount);
        [DllImport("user32.dll")]
        public static extern bool IsWindowVisible(IntPtr hWnd);
    }
"@ -ErrorAction SilentlyContinue

# Function to get windows for a specific process ID
function Get-ProcessWindows {
    param([int]$ProcessId)
    
    $windows = @()
    $script:currentTargetProcessId = $ProcessId  # Use script-level variable
    
    $enumCallback = [Win32+EnumWindowsProc]{
        param($hWnd, $lParam)
        $processId = 0
        [Win32]::GetWindowThreadProcessId($hWnd, [ref]$processId)
        if ($processId -eq $script:currentTargetProcessId) {
            $title = New-Object System.Text.StringBuilder 512
            $className = New-Object System.Text.StringBuilder 256
            [Win32]::GetWindowText($hWnd, $title, 512) | Out-Null
            [Win32]::GetClassName($hWnd, $className, 256) | Out-Null
            $isVisible = [Win32]::IsWindowVisible($hWnd)
            $script:windows += @{
                HWND = $hWnd
                Title = $title.ToString()
                ClassName = $className.ToString()
                Visible = $isVisible
            }
        }
        return $true
    }
    
    $script:windows = @()
    [Win32]::EnumWindows($enumCallback, 0) | Out-Null
    return $script:windows
}

Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "Java.exe Process Information" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host ""

$javaProcesses = Get-Process -Name "java" -ErrorAction SilentlyContinue

if (-not $javaProcesses) {
    Write-Host "No java.exe processes found." -ForegroundColor Yellow
    exit
}

Write-Host "Found $($javaProcesses.Count) java.exe process(es):" -ForegroundColor Green
Write-Host ""

foreach ($proc in $javaProcesses) {
    Write-Host ("=" * 80) -ForegroundColor Yellow
    Write-Host "Process ID (PID): $($proc.Id)" -ForegroundColor White
    Write-Host "Process Name: $($proc.ProcessName)" -ForegroundColor White
    Write-Host "Start Time: $($proc.StartTime)" -ForegroundColor White
    Write-Host "CPU Time: $($proc.CPU)" -ForegroundColor White
    Write-Host "Memory (Working Set): $([math]::Round($proc.WorkingSet64 / 1MB, 2)) MB" -ForegroundColor White
    Write-Host "Memory (Private): $([math]::Round($proc.PrivateMemorySize64 / 1MB, 2)) MB" -ForegroundColor White
    
    # Get parent process
    try {
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)" | Select-Object -ExpandProperty ParentProcessId
        if ($parent) {
            $parentProc = Get-Process -Id $parent -ErrorAction SilentlyContinue
            if ($parentProc) {
                Write-Host "Parent Process: $($parentProc.ProcessName) (PID: $parent)" -ForegroundColor White
            } else {
                Write-Host "Parent Process ID: $parent" -ForegroundColor White
            }
        }
    } catch {
        Write-Host "Parent Process: Could not determine" -ForegroundColor Gray
    }
    
    # Get command line arguments
    try {
        $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
        if ($cmdLine) {
            Write-Host "Command Line: $cmdLine" -ForegroundColor Cyan
        }
    } catch {
        Write-Host "Command Line: Could not retrieve" -ForegroundColor Gray
    }
    
    # Get window information using Windows API
    Write-Host ""
    Write-Host "Window Information:" -ForegroundColor Magenta
    
    $windows = Get-ProcessWindows -ProcessId $proc.Id
    
    if ($windows.Count -eq 0) {
        Write-Host "  No windows found for this process" -ForegroundColor Gray
    } else {
        foreach ($win in $windows) {
            Write-Host "  HWND: $($win.HWND)" -ForegroundColor White
            Write-Host "  Title: '$($win.Title)'" -ForegroundColor $(if ($win.Title) { "Green" } else { "Gray" })
            Write-Host "  Class: $($win.ClassName)" -ForegroundColor White
            Write-Host "  Visible: $($win.Visible)" -ForegroundColor $(if ($win.Visible) { "Green" } else { "Yellow" })
            Write-Host ""
        }
    }
    
    Write-Host ""
}

Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  Total java.exe processes: $($javaProcesses.Count)" -ForegroundColor White
Write-Host "  PIDs: $($javaProcesses.Id -join ', ')" -ForegroundColor White
Write-Host ("=" * 80) -ForegroundColor Cyan
