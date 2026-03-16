# LocalAi
Local AI inference infrastructure for Windows.

## Overview
LocalAi is a standalone local inference server for Windows systems running local models through `llama.cpp`. Any project on the machine connects to it through an OpenAI-compatible REST API instead of managing inference directly. The server is responsible for model lifecycle, VRAM safety, request routing, and operational visibility. Currently consumes: AUREX, OpenClaw.

## System Requirements
| Component | Requirement |
| --- | --- |
| OS | Windows 11 |
| Python | 3.11+ |
| GPU | NVIDIA (CUDA) — RTX 4050 6GB minimum tested |
| Inference | llama.cpp (llama-server.exe) |
| RAM | 8GB minimum, 16GB+ recommended |

## Project Structure
```text
C:\LocalAi\
├── server\         Core service code, API surface, config loading, and utilities
│   ├── api\        OpenAI-compatible and LocalAi administrative endpoints
│   ├── core\       Inference orchestration, model state, VRAM, health, and metrics
│   ├── config\     Configuration schemas and config loading
│   └── utils\      Shared logging, checksum, GPU, and process helpers
├── models\         Local model storage root
├── bin\            llama.cpp binary directory
├── clients\        Client-specific access and preference configuration
├── logs\           Runtime log output
├── data\           Runtime state and process metadata
└── scripts\        Operational PowerShell helper scripts
```

## Quick Start
1. Clone or set up the repository in `C:\LocalAi\`.
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `.\venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Place `llama-server.exe` in `bin\`
6. Add a model to `models\`
7. Edit `localai.config.json`
8. Run: `.\start.ps1`

## API Reference
| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Service health status |
| GET | `/v1/models` | List available models |
| GET | `/v1/models/{model_id}` | Get details for a specific model |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completion endpoint |
| POST | `/v1/completions` | OpenAI-compatible text completion endpoint |
| POST | `/localai/models/load` | Load a model into serving state |
| POST | `/localai/models/unload` | Unload a model from serving state |
| GET | `/localai/status` | Get LocalAi runtime status |
| GET | `/localai/metrics` | Get LocalAi metrics snapshot |

## Configuration
LocalAi applies configuration at three levels. `localai.config.json` defines global server, inference, VRAM, model, and logging defaults. `models/{model}/model.config.json` is reserved for per-model settings that override global defaults when model-specific behavior is needed. Per-request API parameters apply only to the current request and should be used for transient inference controls.

## Development Status
| Phase | Status |
| --- | --- |
| Phase 1 — Foundation | 🔧 In Progress |
| Phase 2 — Core API | ⏳ Pending |
| Phase 3 — Intelligence | ⏳ Pending |

## License
MIT
