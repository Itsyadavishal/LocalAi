$ErrorActionPreference = "Stop"

$pidFilePath = "data\localai.pid"
$healthUrl = "http://127.0.0.1:8080/health"
$apiUrl = "http://127.0.0.1:8080/v1"
$header = "----------------------------------------"

if (-not (Test-Path $pidFilePath)) {
    Write-Host "LocalAi: not running"
    exit 0
}

$localAiPid = (Get-Content $pidFilePath -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
$proc = Get-Process -Id $localAiPid -ErrorAction SilentlyContinue
if (-not $proc) {
    Remove-Item -Path $pidFilePath -Force -ErrorAction SilentlyContinue
    Write-Host "LocalAi: not running"
    exit 0
}

$health = Invoke-RestMethod -Uri $healthUrl -Method GET -ErrorAction SilentlyContinue
if (-not $health) {
    Write-Host $header
    Write-Host "LocalAi: running (PID: $localAiPid) but not responding"
    Write-Host "Check logs/localai.log"
    Write-Host $header
    exit 0
}

Write-Host $header
Write-Host "LocalAi: running"
Write-Host "PID:          $localAiPid"
Write-Host "API:          $apiUrl"
Write-Host "Model loaded: $($health.model_loaded)"
Write-Host "VRAM used:    $($health.vram_used_mb) MB"
Write-Host "VRAM total:   $($health.vram_total_mb) MB"
Write-Host "Queue depth:  $($health.queue_depth)"
Write-Host $header
