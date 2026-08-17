#!/usr/bin/env sh
set -eu
exec uv run pytest "$@"

