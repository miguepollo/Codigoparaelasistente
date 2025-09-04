## Asistente IA (Vosk + Piper + Ollama)

### Requisitos
- Python 3.11+
- ALSA funcional (micrófono y altavoz)
- `ollama` instalado y modelo `llama3.2:1b` (o el que prefieras) descargado
- Recomendada la utilidad `sox` para máxima compatibilidad de audio

### Instalación rápida
1. Copia la carpeta al Orange Pi en `~/assistant`.
2. Ejecuta el setup:
```bash
cd ~/assistant
chmod +x setup_orangepi.sh
./setup_orangepi.sh
```
3. Verifica que exista `~/assistant/models/vosk` y `~/assistant/voices/es_ES-sharvard-*.onnx(.json)`.
4. Si `ollama pull llama3.2:1b` falla, inicia el servicio y vuelve a intentarlo:
```bash
ollama serve &
sleep 3
ollama pull llama3.2:1b
```

Dependencias del sistema recomendadas (si no las tienes):
```bash
sudo nala install alsa-utils sox
```

### Ejecutar
```bash
./run.sh
```

Por defecto se usará el dispositivo ALSA en `APLAY_DEVICE` si está definido (ej. `hw:2,0` o `plughw:2,0`). Si no, usará el predeterminado del sistema.

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

### Opciones del modelo (temperatura, top_p, etc.)
Puedes ajustar las opciones de inferencia via JSON:
```bash
export OLLAMA_OPTIONS='{"temperature":0.3,"top_p":0.9,"repeat_penalty":1.05,"num_predict":256}'
./run.sh
```

### Dispositivo de audio de salida (ALSA)
Listar dispositivos:
```bash
aplay -l | cat
```
Probar un dispositivo específico:
```bash
aplay -D hw:2,0 /usr/share/sounds/alsa/Front_Center.wav
aplay -D plughw:2,0 /usr/share/sounds/alsa/Front_Center.wav
```
Fijar para el asistente (usado por `aplay`):
```bash
export APLAY_DEVICE=hw:2,0   # o plughw:2,0
./run.sh
```

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

El asistente espera la palabra de activación (por defecto `hola`), luego escucha tu comando, lo procesa con el modelo IA y responde con voz sintetizada por Piper.

Flujo de funcionamiento:
1. Di la wake word (por defecto `hola`) para activar
2. Sonará un beep de inicio (canal abierto) y el asistente escuchará tu comando
3. Habla tu pregunta/instrucción; al terminar, sonará un beep de fin
4. El asistente procesa con IA y responde con voz; la respuesta se reproduce en streaming por frases (empieza a hablar mientras el modelo sigue generando)
5. Tras ~2s de cooldown, vuelve a esperar la wake word

Los logs muestran claramente cada paso del proceso para facilitar el debugging.

### Notas de audio (Piper/ALSA)
- Si tu `hw:*` no acepta el formato nativo, el asistente usa `sox` para convertir a 48kHz/16-bit/estéreo antes de `aplay`.
- Si oyes cortes, prueba `APLAY_DEVICE=plughw:X,Y` o instala `sox` (ver arriba).
- Los tonos/beeps de inicio/fin pueden ajustarse en `play_earcon` (frecuencia, duración, volumen).

## Agradecimientos
A @rhasspy por su proyecto  [PiperTTS](https://github.com/OHF-Voice/piper1-gpl) muchas gracias!!!

A @alphacep por su speech-to-text [Vosk](https://github.com/alphacep/vosk-api) me ha servido de mucha utilidad vosk y sobre todo los ejemplos :D !!!
