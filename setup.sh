#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-.venv}

echo "==> Creating virtual environment (${VENV_DIR})..."
if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Virtual environment already exists. Skipping creation."
fi

ACTIVATE="${VENV_DIR}/bin/activate"
if [ ! -f "${ACTIVATE}" ]; then
  echo "Activation script not found: ${ACTIVATE}"
  exit 1
fi

echo "==> Installing project dependencies..."
source "${ACTIVATE}"
pip install --upgrade pip
pip install -r requirements.txt

cat <<'INFO'

==> Setup complete! Configure environment variables next:

    source .venv/bin/activate
    export DASHSCOPE_API_KEY="your_dashscope_key"
    export VECTOR_DATA_DIR="/path/to/faiss_data"   # optional

Redis defaults to localhost:6379; adjust REDIS_HOST/PORT in .env if needed.
For OCR support, please install Tesseract and ensure it is on the PATH.

INFO
