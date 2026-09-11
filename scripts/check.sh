#!/usr/bin/env bash
# The one command that must be green before any commit.
# Order matters: cheap and deterministic first, so a formatting slip
# is reported before a slow test run.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== ruff format --check"
uv run ruff format --check src tests
echo "== ruff check"
uv run ruff check src tests
echo "== mypy"
uv run mypy
echo "== pytest"
uv run pytest
echo "== all green"
