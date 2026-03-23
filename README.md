<div align="center">
  <h1>LocalAi</h1>
  <p><strong>Windows-first local inference infrastructure with an OpenAI-compatible API</strong></p>
  <p>Config-validated | Queue-backed | VRAM-aware | Graceful shutdown</p>
  <p><code>127.0.0.1:8080</code> API | <code>llama-server.exe</code> backend | <code>Python 3.11+</code></p>
</div>

## Overview

LocalAi is a standalone local inference server for Windows systems using `llama.cpp`. It exposes an OpenAI-compatible HTTP surface so local clients can switch to a local `base_url` without changing request structure.

Current implemented capabilities:

- validated startup config with Pydantic v2
- structured console and file logging
- NVIDIA GPU detection and live VRAM reporting
- model discovery from `models/` with three installed variants: `qwen3.5-4b-q4_k_m`, `qwen3.5-4b-q8_0`, `qwen3.5-9b`
- OpenAI-compatible `/v1/models`, `/v1/chat/completions`, and `/v1/completions`
- async request queueing in front of `llama-server.exe`
- SSE streaming for chat and completions
- health monitoring and metrics collection
- admin endpoints for model load, unload, status, metrics, and shutdown
- graceful shutdown through `POST /localai/shutdown`

## Quick Start In 60 Seconds

1. Open the project in `C:\LocalAi`.
2. Activate the project environment first:

```powershell
.\venv\Scripts\activate
```

1. If `venv` does not exist yet:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

1. Place `llama-server.exe` in `C:\LocalAi\bin\`.
2. Start the server:

```powershell
.\start.ps1
```

1. Check status:

```powershell
.\status.ps1
```

1. Stop the server:

```powershell
.\stop.ps1
```

## Current Status

| Area | Status |
| --- | --- |
| Phase 1 — Foundation | ✅ Complete |
| Phase 2 — Core API | ✅ Complete |
| Phase 3 — Intelligence | ✅ Complete |
## Requirements

| Component | Requirement |
| --- | --- |
| OS | Windows 11 |
| Python | 3.11+ |
| GPU | NVIDIA GPU supported by `pynvml` |
| Inference backend | `llama-server.exe` from `llama.cpp` |
| RAM | 8 GB minimum, 16 GB+ recommended |

## Current Vs Planned Endpoints

| Method | Path | State | Notes |
| --- | --- | --- | --- |
| GET | `/health` | Live | Returns status, VRAM, queue depth |
| GET | `/v1/models` | Live | Lists discovered models |
| GET | `/v1/models/{model_id}` | Live | Supports fuzzy model matching |
| POST | `/v1/chat/completions` | Live | Supports non-streaming and SSE streaming |
| POST | `/v1/completions` | Live | Supports non-streaming and SSE streaming |
| POST | `/localai/shutdown` | Live | Used by `stop.ps1` |
| POST | `/localai/models/load` | Live | Loads an installed model into llama-server |
| POST | `/localai/models/unload` | Live | Stops the current model cleanly |
| GET | `/localai/status` | Live | Runtime summary endpoint |
| GET | `/localai/metrics` | Live | Metrics snapshot endpoint |

## OpenAI Compatibility Notes

LocalAi is aiming for drop-in compatibility by changing only `base_url`.

Current behavior:

- OpenAI-style request models for chat and completions
- OpenAI-style error envelope:

```json
{
  "error": {
    "message": "...",
    "type": "...",
    "code": "..."
  }
}
```

- unknown request fields are preserved and forwarded to `llama-server`
- `None` values are stripped before queueing so `llama-server` does not reject the payload
- streaming requests are proxied as SSE to the client

## Model Naming Convention

Use one folder per discoverable model variant.

Current naming pattern:

- `qwen3.5-4b-q8_0`
- `qwen3.5-4b-q4_k_m`
- `qwen3.5-9b`

Recommended rule:

- use architecture and size as the base model name
- add quantization suffix only when multiple variants of the same size exist
- keep `model_id`, folder name, and `model.config.json` aligned exactly

## Current Model Layout

Each discovered model must have its own folder with:

- `model.config.json`
- `weights\<model>.gguf`
- `vision\<mmproj>.gguf` when vision is required

Current installed models:

- `qwen3.5-4b-q8_0`
- `qwen3.5-4b-q4_k_m`
- `qwen3.5-9b`

Example structure:

```text
models\
├── qwen3.5-4b-q8_0\
│   ├── model.config.json
│   ├── weights\Qwen3.5-4B-Q8_0.gguf
│   └── vision\mmproj-4B-BF16.gguf
├── qwen3.5-4b-q4_k_m\
│   ├── model.config.json
│   ├── weights\Qwen3.5-4B-Q4_K_M.gguf
│   └── vision\mmproj-4B-BF16.gguf
└── qwen3.5-9b\
    ├── model.config.json
    ├── weights\Qwen3.5-9B-Q4_K_M.gguf
    └── vision\mmproj-9B-BF16.gguf
```

Important note:

- the two 4B variants share the same `mmproj-4B-BF16.gguf` payload through a hard link, so Explorer shows two entries but disk allocation is not duplicated
- discovery is per model folder, not per `.gguf` file

## Configuration Defaults

The default runtime config in [localai.config.json](/c:/LocalAi/localai.config.json) is:

- API host: `127.0.0.1`
- API port: `8080`
- llama-server port: `8081`
- request timeout: `120` seconds
- max queue depth: `20`
- VRAM safety margin: `300 MB`
- runtime overhead estimate: `200 MB`
- log directory: `logs`

## Project Structure

```text
C:\LocalAi\
├── server\
│   ├── api\        OpenAI-compatible endpoints, admin endpoints, router registration
│   ├── config\     Pydantic schemas and config loading
│   ├── core\       Engine lifecycle, model discovery, queueing, VRAM decisions
│   └── utils\      Logging and GPU helpers
├── models\         Model folders and per-model configs
├── bin\            `llama-server.exe` location
├── logs\           Runtime logs
├── data\           PID file and runtime state
├── scripts\        PowerShell helper scripts
├── start.ps1        Start LocalAi in the background
├── stop.ps1         Graceful shutdown through the local API
└── status.ps1       Health and runtime summary
```

## Current Limitations

- a model must be loaded before inference requests can succeed
- Windows is the primary target environment right now

## Development Notes

- always use the project virtual environment
- always start from `C:\LocalAi`
- use `uvicorn.Config(app=app, ...)`, not string-form app loading, when running via `python -m`
- models and `.gguf` files are intentionally ignored by Git

## License

MIT
