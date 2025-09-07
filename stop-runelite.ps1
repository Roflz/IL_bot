param(
  [string]$BaseDir = "C:\bots\instances"
)
$pidFile = Join-Path $BaseDir "runelite-pids.txt"
if (-not (Test-Path $pidFile)) {
  Write-Host "No pid file found at $pidFile"
  exit 0
}

Get-Content $pidFile | ForEach-Object {
  $parts = $_.Split(',')
  if ($parts.Length -ge 1) {
    $pid = [int]$parts[0]
    try {
      Stop-Process -Id $pid -Force -ErrorAction Stop
      Write-Host "Killed PID $pid"
    } catch {
      Write-Host "PID $pid already stopped."
    }
  }
}
Remove-Item $pidFile -Force
Write-Host "All instances stopped."
