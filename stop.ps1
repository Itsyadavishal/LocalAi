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

try {
    Invoke-RestMethod -Uri "http://127.0.0.1:8080/localai/shutdown" `
        -Method POST -ErrorAction Stop | Out-Null
} catch {
    Write-Host "Warning: Could not reach shutdown endpoint. Falling back to force kill."
    & taskkill /PID $localAiPid /F 2>$null | Out-Null
    Remove-Item -Path $pidFilePath -Force -ErrorAction SilentlyContinue
    Write-Host "LocalAi stopped."
    exit 0
}

$waited = 0
while ($waited -lt 15) {
    $stillRunning = Get-Process -Id $localAiPid -ErrorAction SilentlyContinue
    if (-not $stillRunning) {
        break
    }
    Start-Sleep -Seconds 1
    $waited++
}

$stillRunning = Get-Process -Id $localAiPid -ErrorAction SilentlyContinue
if ($stillRunning) {
    Write-Host "Graceful shutdown timed out. Force killing..."
    & taskkill /PID $localAiPid /F 2>$null | Out-Null
}

Remove-Item -Path $pidFilePath -Force -ErrorAction SilentlyContinue
Write-Host "LocalAi stopped."
