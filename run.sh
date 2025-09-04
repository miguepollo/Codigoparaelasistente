#!/usr/bin/env bash
set -euo pipefail

# Ejecuta el asistente usando un venv local
BASE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$BASE_DIR"

# 1) Si ya hay un entorno virtual ACTIVO, úsalo tal cual
if [ -n "${VIRTUAL_ENV:-}" ]; then
  # Dispositivo ALSA por defecto para aplay (puede ser sobrescrito vía entorno)
  export APLAY_DEVICE="${APLAY_DEVICE:-hw:2,0}"
  python assistant.py
  exit 0
fi

# 2) Prioriza ./venv si existe (compat con setup_orangepi.sh), luego .venv
if [ -d venv ]; then
  . venv/bin/activate
elif [ -d .venv ]; then
  . .venv/bin/activate
else
  # 3) Si no existe ninguno, crea ./venv y instala deps
  python3 -m venv venv
  . venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
fi

# Asegura dependencias (idempotente)
pip install -r requirements.txt

# Dispositivo ALSA por defecto para aplay (puede ser sobrescrito vía entorno)
export APLAY_DEVICE="${APLAY_DEVICE:-hw:2,0}"

python assistant.py

