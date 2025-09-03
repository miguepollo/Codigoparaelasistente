## Asistente IA (Vosk + Piper + Ollama)

![Last commit](https://img.shields.io/github/last-commit/miguepollo/Codigoparaelasistente?style=for-the-badge)
![activity](https://img.shields.io/github/commit-activity/m/miguepollo/Codigoparaelasistente?style=for-the-badge)
![github stars](https://img.shields.io/github/stars/miguepollo/Codigoparaelasistente?style=for-the-badge)

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

## Configuración opcional

### Host de Ollama personalizado:
```bash
export OLLAMA_HOST="http://127.0.0.1:11434"
./run.sh
```

O si lo ejecutas como servicio de systemd, añade en el servicio:
```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

### Prompt personalizado del asistente:
```bash
export OLLAMA_PROMPT="Eres un experto en tecnología y programación. Responde en español de manera técnica y detallada."
./run.sh
```

O modifica directamente la variable `OLLAMA_PROMPT` en el código de `assistant.py`.

Ver `custom_prompt_example.txt` para ejemplos de prompts personalizados según diferentes roles/profesiones.

## Diagnóstico completo

### 🎯 Script maestro (recomendado - verifica TODO):
```bash
python3 diagnostico_completo.py
```

Este script verifica automáticamente:
- ✅ Sintaxis de archivos Python
- ✅ Archivos de voz existen y son válidos
- ✅ Piper TTS (CLI y librería Python)
- ✅ Conexión con Ollama

Si todo está OK, ¡el asistente está listo para usar!

### Errores comunes y soluciones:

**Error "flush of closed file" en CLI:**
- ✅ Ya corregido: usa `text=False` y `flush()` apropiadamente

**Error "memoryview: a bytes-like object is required, not 'AudioChunk'" en Python:**
- ✅ Ya corregido: maneja objetos AudioChunk y los convierte a bytes

**Error "cannot convert 'AudioChunk' object to bytes":**
- ✅ Ya corregido: acceso correcto a `chunk.pcm` para obtener los bytes de audio

### Si los tests pasan pero el asistente no habla:

1. Verifica que el asistente esté usando el entorno virtual correcto:
```bash
source ~/assistant/venv/bin/activate
python3 assistant.py
```

2. Comprueba que los logs del asistente muestren:
```
[TTS] Intentando sintetizar: 'respuesta...'
[TTS] Procesados X chunks de audio
[TTS] WAV generado: Y bytes
[TTS] Reproducción exitosa con aplay
```

3. Si ves "Omite TTS" en los logs, los archivos de voz están corruptos o faltan.

Los tests te dirán exactamente qué está fallando y cómo solucionarlo.

El asistente espera la palabra de activación "asistente", luego escucha tu comando, lo procesa con el modelo IA y responde con voz sintetizada por Piper.

Flujo de funcionamiento:
1. Di "asistente" para activar
2. El asistente confirma activación y espera tu comando
3. Habla tu pregunta/instrucción
4. El asistente procesa con IA y responde por voz
5. Después de 2 segundos de cooldown, vuelve a esperar "asistente"

Los logs muestran claramente cada paso del proceso para facilitar el debugging.

## Agradecimientos
A @rhasspy por su proyecto  [PiperTTS](https://github.com/OHF-Voice/piper1-gpl) muchas gracias!!!

A @alphacep por su speech-to-text [Vosk](https://github.com/alphacep/vosk-api) me ha servido de mucha utilidad vosk y sobre todo los ejemplos :D !!!
