Write-Host "----------------------------------------"
Write-Host "LocalAi - Model Validation"
Write-Host "----------------------------------------"

$activateScript = Join-Path $PSScriptRoot "..\venv\Scripts\Activate.ps1"
$pythonExe = Join-Path $PSScriptRoot "..\venv\Scripts\python.exe"

if (-not (Test-Path $activateScript) -or -not (Test-Path $pythonExe)) {
    Write-Host "Error: venv not found. Run: python -m venv venv"
    exit 1
}

. $activateScript

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$pythonCode = @"
import sys
from pathlib import Path

sys.path.insert(0, r"$repoRoot")

from server.utils.checksum import verify_checksum
from server.utils.logger import get_logger, setup_logging

setup_logging("info", "logs")
logger = get_logger("validate_models")

models_dir = Path("models")
results = []

for model_dir in sorted(models_dir.iterdir()):
    if not model_dir.is_dir():
        continue
    config_file = model_dir / "model.config.json"
    if not config_file.exists():
        continue

    for subdir in ["weights", "vision"]:
        weight_dir = model_dir / subdir
        if not weight_dir.exists():
            continue
        for gguf in sorted(weight_dir.glob("*.gguf")):
            ok, msg = verify_checksum(str(gguf))
            if ok:
                status = "PASS"
            elif msg.startswith("No checksum file found:"):
                status = "MISSING"
            else:
                status = "FAIL"
            results.append((model_dir.name, gguf.name, status, msg))
            print(f"  [{status}] {model_dir.name}/{subdir}/{gguf.name} - {msg}")

passed = sum(1 for r in results if r[2] == "PASS")
failed = sum(1 for r in results if r[2] == "FAIL")
missing = sum(1 for r in results if r[2] == "MISSING")

print()
print(f"Results: {passed} passed, {failed} failed, {missing} missing checksum files")

if failed > 0:
    sys.exit(1)
"@

& $pythonExe -c $pythonCode
$exitCode = $LASTEXITCODE

if ($exitCode -eq 1) {
    Write-Host "Validation failed. Check output above."
    exit 1
}

Write-Host "All models validated successfully."
exit 0
