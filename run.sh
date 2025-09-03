#!/usr/bin/env bash
set -euo pipefail

# Ejecuta el asistente usando un venv local
BASE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$BASE_DIR"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi

# Dispositivo ALSA por defecto para aplay (puede ser sobrescrito vía entorno)
export APLAY_DEVICE="${APLAY_DEVICE:-hw:2,0}"

python assistant.py

