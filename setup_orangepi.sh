#!/usr/bin/env bash
set -euo pipefail

# Este script prepara la Orange Pi (RK3588) en modo usuario (sin sudo):
# - Instala dependencias Python en $HOME
# - Descarga modelo Vosk (español pequeño)
# - Descarga voz Piper es_ES-emilia-low
# - Instala binario de Ollama (arm64) en ~/.local/bin (best-effort)

echo "[1/7] Creando entorno virtual en $HOME/assistant/venv"
mkdir -p "$HOME/assistant"
python3 -m venv "$HOME/assistant/venv"
source "$HOME/assistant/venv/bin/activate"
pip install --upgrade pip
echo "[2/7] Instalando dependencias en venv"
pip install vosk sounddevice ollama piper-tts onnxruntime numpy

echo "[3/7] Asegurando PATH de usuario (~/.local/bin)"
mkdir -p "$HOME/.local/bin"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) : ;;
  *) echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc" ;;
 esac

echo "[4/7] Descargando modelo de Vosk (es pequeño) si no existe..."
mkdir -p "$HOME/assistant/models"
if [ ! -f "$HOME/assistant/models/.vosk_downloaded" ]; then
  mkdir -p /tmp/vosk_dl
  cd /tmp/vosk_dl
  wget -q https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip -O model.zip
  unzip -q model.zip
  SRC_DIR=$(find . -maxdepth 1 -type d -name 'vosk-model-small-es-*' | head -n1)
  if [ -n "$SRC_DIR" ]; then
    rm -rf "$HOME/assistant/models/vosk"
    mv "$SRC_DIR" "$HOME/assistant/models/vosk"
  fi
  touch "$HOME/assistant/models/.vosk_downloaded"
fi

echo "[5/7] Descargando voz de Piper (es_ES) si no existe..."
mkdir -p "$HOME/assistant/voices"
cd "$HOME/assistant/voices"
if [ ! -f es_ES-sharvard-low.onnx ] || [ ! -f es_ES-sharvard-low.onnx.json ]; then
  base_hf="https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/es_ES-sharvard-low"
  wget -q "${base_hf}/es_ES-sharvard-low.onnx" -O es_ES-sharvard-low.onnx || true
  wget -q "${base_hf}/es_ES-sharvard-low.onnx.json" -O es_ES-sharvard-low.onnx.json || true
fi

echo "[6/7] Instalando Ollama (binario arm64 en usuario) si no existe..."
if ! command -v ollama >/dev/null 2>&1; then
  cd "$HOME/.local/bin"
  wget -q "https://github.com/ollama/ollama/releases/latest/download/ollama-linux-arm64" -O ollama || true
  chmod +x ollama || true
fi

echo "[7/7] Intentando iniciar servidor de Ollama y predescargar gemma3:1b (best-effort)"
if command -v ollama >/dev/null 2>&1; then
  pgrep -x ollama >/dev/null 2>&1 || nohup ollama serve >/dev/null 2>&1 &
  sleep 3 || true
  ollama pull gemma3:1b || true
fi

echo "Setup completado en modo usuario. Si faltan codecs ALSA u otros, instala con 'sudo nala install alsa-utils sox ffmpeg portaudio19-dev'"

