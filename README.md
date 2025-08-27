## Asistente IA (Vosk + Piper + Ollama)

### Requisitos (en Orange Pi 5 Ultra)
- Python 3.11+
- Micro y altavoz funcionando (ALSA)
- Ollama instalado y modelo `gemma3:1b` descargado

### Instalación rápida en Orange Pi (sin sudo)
1. Copia la carpeta al Orange Pi en `~/assistant`.
2. Ejecuta el setup:
```bash
cd ~/assistant
chmod +x setup_orangepi.sh
./setup_orangepi.sh
```
3. Verifica que exista `~/assistant/models/vosk` y `~/assistant/voices/es_ES-sharvard-low.onnx(.json)`.
4. Si `ollama pull gemma3:1b` falla, inicia el servicio y vuelve a intentarlo:
```bash
ollama serve &
sleep 3
ollama pull gemma3:1b
```

### Ejecutar
```bash
./run.sh
```

El asistente espera la palabra de activación "asistente" y luego responde con Piper.

