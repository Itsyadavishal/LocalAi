$ErrorActionPreference = "Stop"

$projectRoot = "C:\LocalAi"
$pidFilePath = "data\localai.pid"
$healthUrl = "http://127.0.0.1:8080/health"
$apiUrl = "http://127.0.0.1:8080/v1"
$logPath = "logs/localai.log"
$header = "----------------------------------------"

Write-Host $header
Write-Host "LocalAi — Starting"
Write-Host $header

$currentPath = [System.IO.Path]::GetFullPath((Get-Location).Path).TrimEnd("\")
if ($currentPath -ine $projectRoot) {
    Write-Host "Error: start.ps1 must be run from C:\LocalAi"
    exit 1
}

if (Test-Path $pidFilePath) {
    $existingPid = (Get-Content $pidFilePath -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($existingPid) {
        $existingProcess = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
        if ($existingProcess) {
            Write-Host "LocalAi is already running (PID: $existingPid)"
            exit 0
        }
    }
    Remove-Item -Path $pidFilePath -Force -ErrorAction SilentlyContinue
}

$portInUse = Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "Error: Port 8080 is already in use. Stop the process using that port before starting LocalAi."
    exit 1
}

if (-not (Test-Path ".\venv\Scripts\Activate.ps1") -or -not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "Error: venv not found. Run: python -m venv venv"
    exit 1
}

foreach ($arg in $args) {
    switch ($arg) {
        "--fast" {
            $env:LOCALAI_FAST = "1"
        }
        "--no-autoload" {
            $env:LOCALAI_NO_AUTOLOAD = "1"
        }
        default {
            Write-Host "Error: Unknown flag: $arg"
            exit 1
        }
    }
}

& .\venv\Scripts\Activate.ps1

$pythonExe = (Resolve-Path ".\venv\Scripts\python.exe").Path
$process = Start-Process -FilePath $pythonExe `
    -ArgumentList "-m", "server.main" `
    -NoNewWindow `
    -PassThru

$process.Id | Set-Content -Path $pidFilePath

Start-Sleep -Seconds 3
$runningProcess = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
if (-not $runningProcess) {
    Remove-Item -Path $pidFilePath -Force -ErrorAction SilentlyContinue
    Write-Host "Error: LocalAi failed to start. Check logs/localai.log for details."
    exit 1
}

$ready = $false
for ($attempt = 0; $attempt -lt 15; $attempt++) {
    $healthResponse = Invoke-RestMethod -Uri $healthUrl -Method GET -ErrorAction SilentlyContinue
    if ($healthResponse) {
        $ready = $true
        break
    }
    Start-Sleep -Seconds 1
}

if (-not $ready) {
    Write-Host "Warning: Server started but /health did not respond. Check logs/localai.log"
}

Write-Host $header
Write-Host "LocalAi is running"
Write-Host "PID:    $($process.Id)"
Write-Host "API:    $apiUrl"
Write-Host "Health: $healthUrl"
Write-Host "Logs:   $logPath"
Write-Host $header
