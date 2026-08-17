#!/usr/bin/env sh
set -eu
export PYTHONUTF8=1
exec uv run python scripts/system_consistency.py --live-gemini --live-gradio "$@"
