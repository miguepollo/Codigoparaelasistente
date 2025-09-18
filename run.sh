#!/usr/bin/env bash
set -euo pipefail

# Ejecuta el asistente usando un venv local
BASE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$BASE_DIR"

# 1) Si ya hay un entorno virtual ACTIVO, úsalo tal cual
if [ -n "${VIRTUAL_ENV:-}" ]; then
  # Deja que APLAY_DEVICE venga del entorno si está definido; si no, no lo fuerces
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

# Dispositivo ALSA: no forzar, solo si el usuario lo define
# export APLAY_DEVICE=plughw:2,0    # ejemplo: hw:X,Y / plughw:X,Y / default
export APLAY_BUFFER_US=900000       # 900 ms de buffer
export APLAY_PERIOD_US=100000       # 100 ms de periodo
export APLAY_MIN_CHUNK_BYTES=32768  # tamaño mínimo de flush al pipe

python assistant.py

