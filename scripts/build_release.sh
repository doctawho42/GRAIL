#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/grail_release.XXXXXX")"
PYTHON_BIN="${PYTHON:-python}"

cleanup() {
  rm -rf "$STAGE_DIR"
}

trap cleanup EXIT

rsync -a \
  --exclude '.git' \
  --exclude '.github/.DS_Store' \
  --exclude '.idea' \
  --exclude '.pytest_cache' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'artifacts' \
  --exclude 'dist' \
  --exclude 'grail_metabolism/data' \
  --exclude 'grail_metabolism/lib' \
  --exclude 'grail_metabolism/venv' \
  --exclude 'notebooks' \
  --exclude '*.pyc' \
  "$ROOT_DIR/" \
  "$STAGE_DIR/"

cd "$STAGE_DIR"
"$PYTHON_BIN" -m build
"$PYTHON_BIN" -m twine check dist/*

mkdir -p "$ROOT_DIR/dist"
cp dist/* "$ROOT_DIR/dist/"

echo "Release artifacts copied to $ROOT_DIR/dist"
