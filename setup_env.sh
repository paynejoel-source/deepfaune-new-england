#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
PYTORCH_CPU_INDEX_URL="https://download.pytorch.org/whl/cpu"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install \
  --index-url "${PYTORCH_CPU_INDEX_URL}" \
  torch \
  torchvision \
  torchaudio
python -m pip install -r requirements.txt
python -m pip install \
  --force-reinstall \
  --no-deps \
  --index-url "${PYTORCH_CPU_INDEX_URL}" \
  torch \
  torchvision \
  torchaudio

echo "Virtual environment created at ${VENV_DIR} and dependencies installed."
echo "Activate it with: source ${VENV_DIR}/bin/activate"
