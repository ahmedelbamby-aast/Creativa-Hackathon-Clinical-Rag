$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it from https://docs.astral.sh/uv/ first."
}

uv sync --all-groups
Write-Host "Development environment installed. Run: uv run pytest"

