# Explicación del proyecto: Asistente de voz (Vosk + Piper + Ollama)

Este documento describe la arquitectura, flujo de trabajo y componentes principales del proyecto. El objetivo es ofrecer un asistente por voz en español con activación por palabra clave, reconocimiento de voz (STT), respuesta de un modelo LLM y síntesis de voz (TTS), además de una pequeña interfaz web de configuración.

## Arquitectura general
- Entrada de audio por ALSA con `sounddevice` (micrófono).
- Detección de palabra de activación (wake word) y posterior reconocimiento continuo con Vosk (offline STT).
- Clasificación de intención con LLM (Ollama) y lógica específica para clima/hora.
- Respuesta del LLM vía streaming, segmentada por frases, y sintetizada en tiempo real con Piper TTS (salida por `aplay`).
- Servidor web (Flask) para configurar claves/ubicación de clima y zona horaria, con reinicio automático del proceso tras guardar (cuando se usa integrado en `assistant.py`).

## Componentes principales

### 1) `assistant.py`
Archivo principal que orquesta todo el flujo.

- Configuración y rutas:
  - `BASE_DIR`, `VOSK_MODEL_DIR`, `VOICES_DIR`, `CONFIG_PATH`.
  - Rutas de Piper por defecto: `PIPER_MODEL` y `PIPER_CONFIG` (pueden ser sustituidas por variables de entorno `PIPER_MODEL`/`PIPER_CONFIG`).
  - Descarga automática del modelo de Vosk si falta (por defecto español grande 0.42) mediante `_download_and_setup_vosk_model()`.

- Audio y dispositivos:
  - Se fija `sd.default.device` a `'rockchip,es8388'` (hardware objetivo del proyecto). Puede cambiarse por entorno/sistema.
  - Parámetros: `SAMPLE_RATE` (autoajustado según dispositivo), `BLOCKSIZE`, umbral/duración de silencio y `WAKE_WORD` (por defecto "hola").

- LLM (Ollama):
  - Modelo por defecto `OLLAMA_MODEL = "llama3.2:3b"` y opciones (`OLLAMA_OPTIONS`) que se pueden ajustar con `OLLAMA_OPTIONS` (JSON en entorno).
  - `OLLAMA_PROMPT` personaliza el rol/tono, se puede definir por entorno.
  - Nota: En el código actual `OLLAMA_HOST` está definido como constante con una IP; si deseas que lea desde variable de entorno, habría que adaptar el código (ahora no se usa `os.getenv` para el host).

- Configuración de usuario (clima y hora):
  - `config.json` (raíz del proyecto) y variables de entorno.
  - `load_config()` carga campos: `owm_api_key`, `city`, `lat`, `lon`, `timezone`.
  - `save_config()` guarda de forma atómica.

- Interfaz web de configuración (Flask) integrada:
  - `start_config_server()` lanza un servidor en segundo plano (0.0.0.0:5000).
  - Plantilla mínima embebida para editar API key de OpenWeather, ciudad/lat/lon y zona horaria.
  - Al guardar, programa un reinicio suave del proceso de `assistant.py` con `os.execv`.

- Reconocimiento de voz (STT):
  - `ensure_paths()` valida/descarga el modelo Vosk y carga `vosk.Model` en memoria.
  - `create_wake_recognizer()` crea un reconocedor con gramática limitada a la wake word.
  - `wait_for_wake_word()` escucha en bucle hasta detectar la palabra de activación.
  - `create_recognizer()` y `listen_command()` capturan el comando completo hasta silencio/timeout, con pitidos de inicio/fin.

- Clasificación de intención y comandos nativos:
  - `classify_intent_via_llm()` pide al LLM un JSON `{"intent":"weather|time|other","when":"now|today|tomorrow|none"}`; si falla, aplica heurística (`detect_intent`).
  - `handle_weather_command()` consulta OpenWeather y genera un breve resumen con el LLM (no stream; habla tras tener el resultado).
  - `handle_time_command()` genera un payload de hora local y lo resume con el LLM (una frase).

- Síntesis de voz (TTS):
  - `speak(text)`: ruta no streaming; intento 1 con CLI de Piper → `aplay` (conversión `sox` si hace falta), con reintentos (`default`/`plughw`/`hw` + `sox`), y fallback 2 con librería `piper-tts` generando WAV en memoria. Fallback opcional a `espeak` si está habilitado por entorno.
  - `AudioPipeline`: canal de audio persistente hacia `aplay` (RAW o pasando por `sox` si el destino es `hw:*`).
  - `stream_and_speak_from_ollama(messages)`: flujo streaming del LLM, segmenta por frases (puntuación/heurísticas) y sintetiza cada segmento con Piper (ideal para respuestas largas, empieza a hablar mientras el LLM sigue generando).
  - Tonos/beeps: `play_earcon()` y `play_startup_beep()` generan WAV estéreo en memoria y reproducen por `aplay`.

- Bucle principal (`main()`):
  1. Valida archivos de voz de Piper y asegura rutas/modelos.
  2. Carga config y arranca la UI web en background.
  3. Reproduce beep de inicio.
  4. Bucle: espera wake word → escucha comando → clasifica intención → maneja `weather`/`time` o manda al LLM en streaming → reproduce respuesta → cooldown breve.

### 2) `config_server.py`
Servidor Flask simple (modo standalone) para editar `config.json`.
- Rutas: `GET /` muestra formulario, `POST /save` guarda.
- Si se ejecuta directamente, levanta en `0.0.0.0:5000`.
- El `assistant.py` ya incluye su propia versión integrada con reinicio automático.

### 3) `run.sh`
Script de arranque idempotente:
- Usa un entorno virtual existente (`venv` o `.venv`) o lo crea la primera vez.
- Instala `requirements.txt` si hace falta.
- Exporta ajustes de ALSA por defecto: `APLAY_DEVICE`, `APLAY_BUFFER_US`, `APLAY_PERIOD_US`, `APLAY_MIN_CHUNK_BYTES`.
- Ejecuta `assistant.py` con el entorno preparado.

### 4) `diagnostico_completo.py`
Script maestro de verificación:
- Chequea sintaxis de `assistant.py`.
- Valida existencia/tamaño de archivos de voz Piper.
- Verifica Piper (CLI y Python) y la conexión a Ollama.
- Muestra un resumen claro (OK/ERROR) y sugerencias.

### 5) `download_piper_voice.py`
Descarga una voz de Piper (`es_ES-sharvard-low.onnx` + JSON) en `voices/` desde GitHub. Útil si quieres una voz ligera. El proyecto por defecto apunta a la variante "medium" en varios scripts.

### 6) `setup_orangepi.sh`
Facilita una preparación en una Orange Pi (arm64) en modo usuario:
- Crea `~/assistant/venv`, instala dependencias Python y descarga Vosk pequeño (español) y voz Piper `es_ES-sharvard-medium`.
- Intenta instalar `ollama` en `~/.local/bin` y hacer `pull` de un modelo compacto.
- Imprime al final recomendaciones de paquetes del sistema (usar `sudo nala install ...`).

## Flujo de funcionamiento resumido
1. El asistente arranca, valida modelos y reproduce un beep de inicio.
2. Espera la palabra de activación (por defecto `hola`).
3. Al activarse, emite beep de inicio de escucha y captura tu comando hasta silencio.
4. Clasifica la intención:
   - Clima/hora: construye y lee un resumen breve.
   - Otro: manda al LLM en streaming y va hablando por frases.
5. Tras responder, breves 2s de cooldown y vuelve a esperar la wake word.

## Configuración y variables de entorno
- Clima/hora: `config.json` con `owm_api_key`, `city` o `lat`/`lon`, `timezone`.
- LLM:
  - `OLLAMA_PROMPT`: cadena para el rol del asistente.
  - `OLLAMA_OPTIONS`: JSON con opciones (ej.: `{"temperature":0.3,"num_predict":256}`).
  - Host: en el código actual `OLLAMA_HOST` es constante; si quieres leer de entorno, ajusta `assistant.py`.
- Audio (ALSA):
  - `APLAY_DEVICE` (ej. `hw:2,0`, `plughw:2,0` o `default`).
  - `APLAY_BUFFER_US`, `APLAY_PERIOD_US`, `APLAY_MIN_CHUNK_BYTES` para tuning de latencia/fluidez.
  - Para beeps y TTS se usa `aplay`; `sox` se usa opcionalmente para convertir a 48k/16-bit/2ch cuando el dispositivo es `hw:*`.
- Voces Piper: `PIPER_MODEL` y `PIPER_CONFIG` pueden fijarse por entorno si los archivos por defecto no existen o se desea otra voz.

## Dependencias
- Python: ver `requirements.txt` (Vosk, sounddevice, numpy, ollama, piper-tts, onnxruntime, Flask, requests).
- Paquetes del sistema recomendados:
  - Usa Nala en lugar de APT según preferencia: `sudo nala install alsa-utils sox`

## Resolución de problemas (rápido)
- Ejecuta: `python3 diagnostico_completo.py` para una verificación integral.
- Si no habla y ves "Omite TTS": faltan/dañados archivos de voz en `voices/`.
- Si `aplay` falla con `hw:*`, prueba `APLAY_DEVICE=plughw:X,Y` o instala `sox`.
- Si Ollama no responde, asegúrate de que el servicio esté corriendo y que el host en `assistant.py` sea accesible desde tu equipo.

## Diagnóstico de audio (ALSA) y cambios recientes

Cambios recientes para robustez del TTS:
- `play_earcon()` ahora reintenta automáticamente con `-D default` si falla el dispositivo configurado.
- `AudioPipeline` detecta si `aplay` cae al inicio y hace fallback automático al dispositivo `default` (tanto en modo RAW como en la ruta `sox → aplay` cuando el destino es `hw:*`).
- `run.sh` ya no fuerza `APLAY_DEVICE`; si no está definido, se usa el dispositivo por defecto del sistema.

Comprobaciones útiles en tu sistema:
- Listar dispositivos HW: `aplay -l`
- Listar targets ALSA (incluye `default`, `plughw`, mezclas/pulses): `aplay -L | head -n 100`
- Probar salida básica (default): `speaker-test -t sine -f 1000 -l 1` (si no suena, revisa mezcla/volumen ALSA/PulseAudio/PipeWire).
- Probar un beep con `aplay` directo: `python - << 'PY'\nimport sys, wave, io, math\nsr=48000; d=0.2; f=1000; buf=io.BytesIO();\nwith wave.open(buf,'wb') as w:\n  w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)\n  for n in range(int(sr*d)):\n    v=int(32767*0.3*math.sin(2*math.pi*f*n/sr)); w.writeframesraw((v.to_bytes(2,'little',signed=True))*2)\nopen('beep.wav','wb').write(buf.getvalue())\nPY\n&& aplay -q beep.wav`

Variables de entorno relevantes:
- `APLAY_DEVICE`: destino para `aplay` (ej. `default`, `plughw:2,0`, `hw:0,0`). Si no se define, se usa el predeterminado del sistema.
- `APLAY_BUFFER_US`, `APLAY_PERIOD_US`, `APLAY_MIN_CHUNK_BYTES`: tuning de latencia/fluidez.

Sugerencias si “dice que suena pero no suena”:
- Comprueba que `APLAY_DEVICE` apunte a una salida válida. Si ves logs como "Enviando RAW directo a aplay en plughw:2,0" y no hay audio, prueba sin `APLAY_DEVICE` (usando `default`) o cambia a `plughw:X,Y` en vez de `hw:X,Y`.
- Instala `sox` (`sudo nala install sox`) para permitir conversión automática cuando usas `hw:*`.
- Revisa mezcladores/volúmenes: `alsamixer` y que no esté en mute. Si usas PulseAudio/PipeWire, verifica rutas en el control de volumen del entorno.

## Extender el asistente (nuevas intenciones)
1. Añade una rama en `main()` tras la clasificación de intención.
2. Implementa un `handle_tu_intencion(...)` que obtenga datos externos si hace falta y genere un resumen breve con el LLM.
3. Si deseas streaming, usa `stream_and_speak_from_ollama(messages)`; si no, usa `speak(text)` tras construir la respuesta.

---
Este documento resume cómo está montado el asistente, dónde configurar cada pieza y cómo depurarlo o extenderlo.


