# LocalAi — Development Rules

## Mandatory Environment Rule

- Always use the project virtual environment for this repository
- Never install dependencies globally
- Never execute project dependencies from the global Python environment
- The first step when opening this project is: `.\venv\Scripts\activate`

## 1. Python Standards

- Python 3.11+ only
- Every file has a module-level docstring (purpose, part of, status)
- Every function has a docstring (what it does, args, returns, raises)
- No file exceeds 400 lines — split by responsibility if approaching limit
- Type hints on every function signature — no bare untyped code
- No hardcoded paths, ports, or values — everything from config
- No magic numbers — use named constants at top of file

## 2. Folder & File Naming

- Folders: lowercase, underscore_separated
- Python files: lowercase, underscore_separated
- Config files: lowercase, dot.separated.json
- No abbreviations unless industry standard (e.g. vram, api, gpu are fine)
- One responsibility per file — if a file is doing two things, split it

## 3. Configuration Rules

- All config loaded and validated with Pydantic on startup
- Fail fast: if config is invalid, server does not start — error is printed clearly
- Never read config values at import time — always pass through dependency injection
- localai.config.json is the only file a user should ever need to edit

## 4. Error Handling

- Every error is logged with full context before being raised
- User-facing API errors use standard HTTP status codes with JSON body
- No silent failures — if something goes wrong, it is logged
- External process failures (llama-server) are caught and surfaced cleanly

## 5. Logging

- Use the logger from server/utils/logger.py only — no print() statements in server code
- Log level follows localai.config.json setting
- Structured JSON format for all log entries
- Never log sensitive data

## 6. Git Commit Rules

### Format

<type>(<scope>): <short description>

[optional body — what and why, not how]

### Types

- feat     — new feature or capability
- fix      — bug fix
- refactor — code restructure, no behavior change
- docs     — documentation only
- config   — configuration file changes
- test     — adding or updating tests
- chore    — tooling, dependencies, project setup

### Scopes (use these exact scope names)

- scaffold    — folder structure, project setup
- config      — configuration system
- api         — API endpoints
- core        — core engine components
- vram        — VRAM management
- inference   — llama.cpp process management
- queue       — request queue
- logger      — logging system
- health      — health monitoring
- metrics     — metrics collection
- docs        — documentation files
- deps        — dependencies

### Rules

- Subject line max 72 characters
- Use imperative mood: "add", "fix", "remove" — not "added", "fixes"
- Never commit directly to main — always use a feature branch
