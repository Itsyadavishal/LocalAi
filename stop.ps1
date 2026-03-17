$ErrorActionPreference = "Stop"

$pidFilePath = "data\localai.pid"
$header = "----------------------------------------"

Write-Host $header
Write-Host "LocalAi — Stopping"
Write-Host $header

if (-not (Test-Path $pidFilePath)) {
    Write-Host "LocalAi is not running (no PID file found)"
    exit 0
}

$localAiPid = (Get-Content $pidFilePath -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
$proc = Get-Process -Id $localAiPid -ErrorAction SilentlyContinue
if (-not $proc) {
    Remove-Item -Path $pidFilePath -Force -ErrorAction SilentlyContinue
    Write-Host "LocalAi was not running (stale PID file cleaned up)"
    exit 0
}

Write-Host "Stopping LocalAi (PID: $localAiPid)..."
& taskkill /PID $localAiPid 2>$null | Out-Null

$waited = 0
while ($waited -lt 10) {
    $stillRunning = Get-Process -Id $localAiPid -ErrorAction SilentlyContinue
    if (-not $stillRunning) {
        break
    }
    Start-Sleep -Seconds 1
    $waited++
}

if (Get-Process -Id $localAiPid -ErrorAction SilentlyContinue) {
    Write-Host "Graceful stop timed out. Force killing..."
    & taskkill /PID $localAiPid /F 2>$null | Out-Null
    Start-Sleep -Seconds 1
}

Remove-Item -Path $pidFilePath -Force -ErrorAction SilentlyContinue
Write-Host "LocalAi stopped."
