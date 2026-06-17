#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export MPLCONFIGDIR="${ROOT_DIR}/.matplotlib_cache"
mkdir -p "${MPLCONFIGDIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

cd "${ROOT_DIR}"

echo "[ci] Formatting check (black)"
black --check python tests/python

echo "[ci] Linting (flake8)"
flake8 python tests/python

echo "[ci] Running pytest"
PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} --cache-clear" python -m pytest tests/python -q

echo "[ci] All checks passed"
