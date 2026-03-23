Write-Host "----------------------------------------"
Write-Host "LocalAi - Add Model"
Write-Host "----------------------------------------"

$modelId = Read-Host "Model ID (folder name, no spaces, e.g. qwen3.5-4b-q4_k_m)"
$displayName = Read-Host "Display name (e.g. Qwen 3.5 4B Q4_K_M)"
$ggufFilename = Read-Host "GGUF filename (e.g. qwen3.5-4b-q4_k_m.gguf)"
$hasVision = Read-Host "Does this model have a vision encoder? (y/n)"
$mmproj = ""
if ($hasVision -eq "y") {
    $mmproj = Read-Host "mmproj filename (e.g. mmproj-BF16.gguf)"
}
$ctxSize = Read-Host "Context size (default: 4096)"
if (-not $ctxSize) { $ctxSize = "4096" }
$gpuLayers = Read-Host "GPU layers (default: 28)"
if (-not $gpuLayers) { $gpuLayers = "28" }
$capabilities = Read-Host "Capabilities (comma-separated: text,vision,code - default: text)"
if (-not $capabilities) { $capabilities = "text" }

if ($modelId -match "\s") {
    Write-Host "Error: Model ID must not contain spaces."
    exit 1
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$modelRoot = Join-Path $repoRoot "models\$modelId"
if (Test-Path $modelRoot) {
    Write-Host "Error: Model '$modelId' already exists."
    exit 1
}

$null = New-Item -ItemType Directory -Path $modelRoot
$null = New-Item -ItemType Directory -Path (Join-Path $modelRoot "weights")
$null = New-Item -ItemType Directory -Path (Join-Path $modelRoot "logs")
$null = New-Item -ItemType Directory -Path (Join-Path $modelRoot "metrics")
if ($hasVision -eq "y") {
    $null = New-Item -ItemType Directory -Path (Join-Path $modelRoot "vision")
}

$capabilityList = @($capabilities.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })

$config = [ordered]@{
    model_id = $modelId
    display_name = $displayName
    gguf_filename = $ggufFilename
    mmproj_filename = $(if ([string]::IsNullOrWhiteSpace($mmproj)) { $null } else { $mmproj })
    ctx_size = [int]$ctxSize
    gpu_layers = [int]$gpuLayers
    max_tokens = 2048
    capabilities = $capabilityList
    vram_estimate_mb = 0
}

$config | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $modelRoot "model.config.json") -Encoding utf8

$readme = @"
# $displayName

## Model Info
- **Model ID:** $modelId
- **GGUF file:** weights/$ggufFilename
- **Vision encoder:** $(if ($mmproj) { "vision/$mmproj" } else { "None" })

## Next Steps
1. Place the model weights in `weights/`.
2. If this is a vision model, place the encoder in `vision/`.
3. Generate checksum sidecar files after placing the binaries.
"@

$readme | Set-Content -Path (Join-Path $modelRoot "model.README.md") -Encoding utf8

Write-Host "----------------------------------------"
Write-Host "Model '$modelId' registered successfully."
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Place your GGUF file here:"
Write-Host "   models\$modelId\weights\$ggufFilename"
Write-Host ""
if ($mmproj) {
    Write-Host "If vision model, place mmproj here:"
    Write-Host "   models\$modelId\vision\$mmproj"
    Write-Host ""
}
Write-Host "2. Generate checksum after placing files:"
Write-Host ('   .\venv\Scripts\python.exe -c "from server.utils.checksum import write_checksum_file; write_checksum_file(''models/{0}/weights/{1}'')"' -f $modelId, $ggufFilename)
Write-Host ""
Write-Host "3. Restart LocalAi to discover the new model:"
Write-Host "   .\stop.ps1"
Write-Host "   .\start.ps1"
Write-Host ""
Write-Host "4. Verify registration:"
Write-Host "   curl http://localhost:8080/v1/models"
Write-Host "----------------------------------------"
