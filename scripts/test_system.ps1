$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
uv run python scripts/system_consistency.py --live-gemini --live-gradio @args
