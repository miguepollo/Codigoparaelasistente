import os
import sys
import json
import time
import queue
import subprocess
import threading
import re
import io
import wave
import shutil
import math
from typing import Optional, Tuple, Literal
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from flask import Flask, request, redirect, url_for, render_template_string

import numpy as np
import sounddevice as sd
import vosk
import ollama


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOSK_MODEL_DIR = os.path.join(BASE_DIR, "models", "vosk")
PIPER_MODEL = os.path.join(BASE_DIR, "voices", "es_ES-sharvard-medium.onnx")
PIPER_CONFIG = os.path.join(BASE_DIR, "voices", "es_ES-sharvard-medium.onnx.json")
VOICES_DIR = os.path.join(BASE_DIR, "voices")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
# Permitir override por variables de entorno
env_model = os.getenv("PIPER_MODEL")
env_config = os.getenv("PIPER_CONFIG")
if env_model:
    PIPER_MODEL = env_model
if env_config:
    PIPER_CONFIG = env_config


# Configuración de audio basada en ejemplo probado
sd.default.device = 'rockchip,es8388'
SAMPLE_RATE = 16000
BLOCKSIZE = 8000
WAKE_WORD = "hola"
SILENCE_THRESHOLD = 2000  # RMS ~ energía. Ajustar si hace falta
SILENCE_MS = 2000  # fin por silencio
MAX_COMMAND_SECS = 12
OLLAMA_MODEL = "llama3.2:1b"
# Permite configurar el endpoint de Ollama, p.ej.: OLLAMA_HOST="http://127.0.0.1:11434"
OLLAMA_HOST = "http://127.0.0.1:11434"
# Prompt del sistema. Busca respuestas directas, sin saludos ni autorreferencias
OLLAMA_PROMPT = os.getenv(
    "OLLAMA_PROMPT",
    (
        "Eres un asistente llamado Kubik. Responde SIEMPRE en español, de forma directa y concisa. "
        "PROHIBIDO saludar o presentarte (no digas 'hola', 'soy Kubik', etc.). A menos que te pregunten por tu nombre, entonces responde 'Soy Kubik, tu asistente virtual.' "
        "Cuando te pregunten 'quién/qué/cuándo/dónde/por qué/cómo', responde con 2-3 frases informativas y nada más. "
        "Usa tono neutro, sin muletillas ni disculpas."
    ),
)

# Opciones de inferencia para Ollama (se puede sobrescribir con OLLAMA_OPTIONS JSON)
def _load_ollama_options() -> dict:
    raw = os.getenv("OLLAMA_OPTIONS", "")
    if raw.strip():
        try:
            opts = json.loads(raw)
            if isinstance(opts, dict):
                return opts
        except Exception:
            pass
    # Valores por defecto sobrios para respuestas más precisas y sin desvíos
    return {
        "temperature": 0.5,
        "top_p": 0.9,
        "repeat_penalty": 1.05,
        "num_predict": 256,
    }

OLLAMA_OPTIONS = _load_ollama_options()

def build_ollama_messages(user_message: str) -> list:
    """Construye los mensajes para enviar a Ollama incluyendo el prompt del sistema."""
    messages = []
    if OLLAMA_PROMPT.strip():
        messages.append({"role": "system", "content": OLLAMA_PROMPT.strip()})
    messages.append({"role": "user", "content": user_message})
    return messages

_piper_voice = None  # Lazy init para fallback Python
_vosk_model: Optional[vosk.Model] = None  # Reutilizar modelo en memoria
_config: dict = {}

IntentType = Literal["weather", "time", "other"]


def load_config() -> dict:
    """Carga configuración desde config.json y variables de entorno.
    Campos: owm_api_key, city, lat, lon, timezone (IANA).
    """
    cfg = {
        "owm_api_key": os.getenv("OWM_API_KEY", ""),
        "city": os.getenv("OWM_CITY", ""),
        "lat": os.getenv("OWM_LAT", ""),
        "lon": os.getenv("OWM_LON", ""),
        "timezone": os.getenv("TIMEZONE", "Europe/Madrid"),
    }
    try:
        if os.path.isfile(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                disk = json.load(fh) or {}
                if isinstance(disk, dict):
                    cfg.update({k: (disk.get(k) or cfg.get(k)) for k in cfg.keys()})
    except Exception:
        pass
    # Normalizar espacios y tipos
    for k in ["owm_api_key", "city", "timezone"]:
        if isinstance(cfg.get(k), str):
            cfg[k] = cfg[k].strip()
    for k in ["lat", "lon"]:
        v = cfg.get(k)
        if isinstance(v, str):
            cfg[k] = v.strip()
    return cfg


def get_config() -> dict:
    global _config
    if not _config:
        _config = load_config()
    return _config


def save_config(new_cfg: dict) -> None:
    """Guarda configuración en disco de forma atómica y actualiza memoria."""
    tmp_path = CONFIG_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(new_cfg, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, CONFIG_PATH)
    # Actualizar config en memoria
    global _config
    _config = new_cfg.copy()


def detect_intent(text: str) -> Tuple[IntentType, dict]:
    """[LEGADO] Detección por palabras clave (fallback)."""
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    extras: dict = {}
    if re.search(r"\b(hora|qué hora|que hora|hora actual)\b", t):
        return "time", extras
    if re.search(r"\b(tiempo|clima|temperatura|pron[oó]stico|lluev|viento|humedad|nubes?)\b", t):
        if re.search(r"ma[ñn]ana|mañana", t):
            extras["when"] = "tomorrow"
        elif re.search(r"hoy|ahora|actual", t):
            extras["when"] = "now"
        return "weather", extras
    return "other", extras


def classify_intent_via_llm(text: str) -> Tuple[IntentType, dict]:
    """Pide a la IA que clasifique la intención y el marco temporal.
    Devuelve (intent, extras) donde intent ∈ {weather,time,other} y extras puede incluir 'when'.
    """
    system = (
        "Eres un clasificador de intenciones para un asistente de voz. "
        "Dado el texto del usuario en español, responde SOLO un JSON en una sola línea con esta forma exacta: "
        "{\"intent\":\"weather|time|other\",\"when\":\"now|today|tomorrow|none\"}. "
        "Elige intent=weather si pregunta por clima/tiempo/temperatura/lluvia/viento/humedad/nubes/pronóstico. "
        "Elige intent=time si pregunta la hora. Si no aplica, other. "
        "'when': now o today si es sobre ahora/hoy, tomorrow si menciona mañana, si no se deduce usa none. "
        "No añadas texto adicional ni explicaciones."
    )
    user = text.strip()
    resp = _ollama_chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    intent: IntentType = "other"
    extras: dict = {}
    # Intentar parsear JSON
    try:
        data = json.loads(resp)
        val = str(data.get("intent", "other")).strip().lower()
        if val in {"weather", "time", "other"}:
            intent = val  # type: ignore[assignment]
        when = str(data.get("when", "none")).strip().lower()
        if when in {"now", "today", "tomorrow", "none"}:
            if when == "today":
                when = "now"
            if when != "none":
                extras["when"] = when
    except Exception:
        # Fallback: si el modelo devolvió texto plano
        low = (resp or "").strip().lower()
        if "weather" in low or "clima" in low or "tiempo" in low:
            intent = "weather"  # type: ignore[assignment]
        elif "time" in low or "hora" in low:
            intent = "time"  # type: ignore[assignment]
    # Si sigue en other, usar heurística básica como último recurso
    if intent == "other":
        intent, heur = detect_intent(text)
        extras.update(heur)
    return intent, extras


def _ollama_chat(messages: list) -> str:
    """Llama a Ollama de forma no streaming y devuelve el texto completo."""
    try:
        if OLLAMA_HOST:
            client = ollama.Client(host=OLLAMA_HOST)
            resp = client.chat(model=OLLAMA_MODEL, messages=messages, options=OLLAMA_OPTIONS)
        else:
            resp = ollama.chat(model=OLLAMA_MODEL, messages=messages, options=OLLAMA_OPTIONS)
        out = (resp or {}).get("message", {}).get("content", "")
        return (out or "").strip()
    except Exception as exc:
        return f"Error consultando el modelo: {exc}"


def _summarize_weather_json(json_payload: dict, location_label: str) -> str:
    """Construye prompt para resumir JSON meteorológico en 2-3 frases claras."""
    system = (
        "Eres un asistente metereológico. Resume en 2-3 frases, en español, de forma concreta, "
        "sin adornos ni saludos. Incluye temperatura, sensación térmica, estado general, y si hay lluvia/viento relevante. No añadas ninguna otra información. No pongas asteriscos. No digas 24Cº solo di 24 grados."
    )
    user = (
        "Resume el siguiente JSON de clima actual para el usuario. Usa unidades SI y 24h. "
        f"Ubicación: {location_label}. JSON:\n" + json.dumps(json_payload, ensure_ascii=False)
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _ollama_chat(messages)


def _summarize_time_json(json_payload: dict) -> str:
    """Construye prompt para resumir la hora local en 1 frase."""
    system = (
        "Eres un asistente de hora. Responde en una sola frase clara, en español, sin saludos." 
        "Usa formato 24h con ceros y menciona la zona horaria abreviada."
    )
    user = "Resume brevemente estos datos de hora local en una frase: " + json.dumps(json_payload, ensure_ascii=False)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _ollama_chat(messages)


def _fetch_openweather(cfg: dict, when: str = "now") -> Tuple[Optional[dict], Optional[str]]:
    """Obtiene datos de OpenWeatherMap para clima actual. Devuelve (json, error)."""
    api_key = (cfg.get("owm_api_key") or "").strip()
    lat = (cfg.get("lat") or "").strip()
    lon = (cfg.get("lon") or "").strip()
    city = (cfg.get("city") or "").strip()
    if not api_key:
        return None, "Falta la API key de OpenWeather. Configúrala en la interfaz web."
    params = {"appid": api_key, "units": "metric", "lang": "es"}
    url = "https://api.openweathermap.org/data/2.5/weather"
    if lat and lon:
        params.update({"lat": lat, "lon": lon})
        location_label = f"lat {lat}, lon {lon}"
    elif city:
        params.update({"q": city})
        location_label = city
    else:
        return None, "Falta ubicación (ciudad o lat/lon). Configúrala en la interfaz web."
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None, f"OpenWeather devolvió {r.status_code}: {r.text[:200]}"
        data = r.json()
        data["_location_label"] = location_label
        return data, None
    except Exception as exc:
        return None, f"Error consultando OpenWeather: {exc}"


def handle_weather_command(original_text: str, when: Optional[str] = None) -> str:
    cfg = get_config()
    if not when:
        when = detect_intent(original_text)[1].get("when", "now")
    data, err = _fetch_openweather(cfg, when=when)
    if err:
        return err
    assert data is not None
    location_label = data.pop("_location_label", cfg.get("city") or "")
    summary = _summarize_weather_json(data, location_label or "")
    return summary or "No pude generar el resumen del clima."


def handle_time_command() -> str:
    cfg = get_config()
    tz_name = cfg.get("timezone") or "Europe/Madrid"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("Europe/Madrid")
        tz_name = "Europe/Madrid"
    now = datetime.now(tz)
    offset_total_seconds = tz.utcoffset(now).total_seconds() if tz.utcoffset(now) else 0
    offset_hours = int(offset_total_seconds // 3600)
    offset_minutes = int((abs(offset_total_seconds) % 3600) // 60)
    sign = "+" if offset_total_seconds >= 0 else "-"
    offset_str = f"UTC{sign}{abs(offset_hours):02d}:{offset_minutes:02d}"
    payload = {
        "timezone": tz_name,
        "iso": now.isoformat(),
        "time_24h": now.strftime("%H:%M"),
        "date": now.strftime("%Y-%m-%d"),
        "weekday": now.strftime("%A"),
        "utc_offset": offset_str,
    }
    summary = _summarize_time_json(payload)
    return summary or f"Son las {payload['time_24h']} ({payload['timezone']})."


# =====================
# Interfaz Web (Flask)
# =====================

_flask_app: Optional[Flask] = None

_TEMPLATE = """
<!doctype html>
<html lang=\"es\">
  <head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>Config Asistente</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial; max-width: 880px; margin: 20px auto; padding: 0 16px; }
      .card { border: 1px solid #e3e3e3; border-radius: 10px; padding: 18px; margin: 12px 0; }
      label { display: block; font-weight: 600; margin-top: 10px; }
      input[type=text], input[type=password] { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px; }
      .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
      .actions { margin-top: 16px; }
      button { background: #0d6efd; color: white; border: none; border-radius: 8px; padding: 10px 14px; cursor: pointer; }
      .note { color: #555; font-size: 0.95em; }
    </style>
  </head>
  <body>
    <h2>Configuración del asistente</h2>
    <form method=\"post\" action=\"{{ url_for('cfg_save') }}\">
      <div class=\"card\">
        <h3>OpenWeatherMap</h3>
        <label>API Key</label>
        <input type=\"password\" name=\"owm_api_key\" value=\"{{ cfg.get('owm_api_key','') }}\" placeholder=\"tu_api_key\" />
        <div class=\"row\">
          <div>
            <label>Ciudad</label>
            <input type=\"text\" name=\"city\" value=\"{{ cfg.get('city','') }}\" placeholder=\"Madrid\" />
            <div class=\"note\">Puedes dejar vacío si usas lat/lon</div>
          </div>
          <div></div>
        </div>
        <div class=\"row\">
          <div>
            <label>Latitud</label>
            <input type=\"text\" name=\"lat\" value=\"{{ cfg.get('lat','') }}\" placeholder=\"40.4168\" />
          </div>
          <div>
            <label>Longitud</label>
            <input type=\"text\" name=\"lon\" value=\"{{ cfg.get('lon','') }}\" placeholder=\"-3.7038\" />
          </div>
        </div>
      </div>

      <div class=\"card\">
        <h3>Zona horaria</h3>
        <label>Timezone IANA</label>
        <input type=\"text\" name=\"timezone\" value=\"{{ cfg.get('timezone','Europe/Madrid') }}\" placeholder=\"Europe/Madrid\" />
        <div class=\"note\">Ejemplos: Europe/Madrid, America/Mexico_City, America/Bogota</div>
      </div>

      <div class=\"actions\">
        <button type=\"submit\">Guardar</button>
      </div>
    </form>
    <p class=\"note\">Tras guardar, el asistente se reiniciará.</p>
  </body>
  </html>
"""


def _schedule_restart(delay_sec: float = 0.4) -> None:
    def _do_restart() -> None:
        try:
            time.sleep(delay_sec)
            python = sys.executable or "python3"
            os.execv(python, [python, os.path.abspath(__file__)])
        except Exception:
            os._exit(0)
    threading.Thread(target=_do_restart, daemon=True).start()


def _create_flask_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def cfg_index():
        return render_template_string(_TEMPLATE, cfg=load_config())

    @app.post("/save")
    def cfg_save():
        cfg = load_config()
        for key in ["owm_api_key", "city", "lat", "lon", "timezone"]:
            val = request.form.get(key, "")
            if isinstance(val, str):
                val = val.strip()
            cfg[key] = val
        save_config(cfg)
        # Programar reinicio del proceso tras responder
        _schedule_restart(0.4)
        return redirect(url_for("cfg_index"))

    return app


def start_config_server() -> None:
    global _flask_app
    if _flask_app is not None:
        return
    _flask_app = _create_flask_app()
    def _run():
        try:
            _flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


def ensure_paths() -> None:
    if not os.path.isdir(VOSK_MODEL_DIR):
        raise FileNotFoundError(f"No se encontró el modelo de Vosk en {VOSK_MODEL_DIR}")
    if not os.path.isfile(PIPER_MODEL) or not os.path.isfile(PIPER_CONFIG):
        raise FileNotFoundError(
            f"No se encontraron los archivos de voz de Piper en {os.path.dirname(PIPER_MODEL)}"
        )
    # Detectar sample rate real del dispositivo de entrada
    global SAMPLE_RATE
    try:
        device_info = sd.query_devices(sd.default.device, "input")
        SAMPLE_RATE = int(device_info.get("default_samplerate", SAMPLE_RATE))
    except Exception:
        pass
    global _vosk_model
    if _vosk_model is None:
        _vosk_model = vosk.Model(VOSK_MODEL_DIR)

def _rms_int16(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def speak(text: str) -> None:
    if not text:
        return

    print(f"[TTS] Intentando sintetizar: '{text[:100]}...'")
    _discover_and_set_piper_voice()  # Asegurar que la voz esté configurada

    # Validar ficheros de voz Piper
    if not _validate_piper_files():
        print("[TTS] Archivos de voz Piper ausentes o corruptos. Omite TTS.")
        return
    # 1) Streaming con CLI de Piper → aplay (empieza a sonar de inmediato)
    try:
        cmd = ["piper", "-m", PIPER_MODEL, "-c", PIPER_CONFIG, "-f", "-"]
        print(f"[TTS] Lanzando Piper CLI streaming: {' '.join(cmd)}")
        proc_tts = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
        )
        assert proc_tts.stdin is not None and proc_tts.stdout is not None

        def build_aplay_cmd(use_plug: bool) -> list:
            base = ["aplay", "-q", "-t", "wav", "-"]
            aplay_dev = os.getenv("APLAY_DEVICE")
            if aplay_dev:
                dev = aplay_dev
                if use_plug and aplay_dev.startswith("hw:"):
                    dev = "plughw:" + aplay_dev.split(":", 1)[1]
                return ["aplay", "-q", "-D", dev, "-t", "wav", "-"]
            return base

        # Si es dispositivo hw:* y tenemos sox, insertar conversión a 48k/16-bit/estéreo
        aplay_dev_env = os.getenv("APLAY_DEVICE", "")
        have_sox = shutil.which("sox") is not None
        if aplay_dev_env.startswith("hw:") and have_sox:
            sox_cmd = [
                "sox",
                "-t", "wav", "-",  # entrada WAV desde Piper
                "-r", "48000",
                "-b", "16",
                "-c", "2",
                "-t", "wav", "-",  # salida WAV hacia aplay
            ]
            print(f"[TTS] Insertando conversión con sox: {' '.join(sox_cmd)}")
            proc_sox = subprocess.Popen(
                sox_cmd,
                stdin=proc_tts.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # aplay leerá desde la salida de sox
            aplay_cmd = build_aplay_cmd(use_plug=False)
            print(f"[TTS] Encadenando aplay streaming: {' '.join(aplay_cmd)}")
            proc_play = subprocess.Popen(
                aplay_cmd,
                stdin=proc_sox.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        else:
            # Intento 1: usar dispositivo tal cual (hw/plughw/default)
            aplay_cmd = build_aplay_cmd(use_plug=False)
            print(f"[TTS] Encadenando aplay streaming: {' '.join(aplay_cmd)}")
            proc_play = subprocess.Popen(
                aplay_cmd,
                stdin=proc_tts.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        # Escribir el texto a Piper para que empiece a generar
        proc_tts.stdin.write((text.strip() + "\n").encode("utf-8"))
        proc_tts.stdin.flush()
        proc_tts.stdin.close()
        # No cerrar proc_tts.stdout aquí; es la tubería hacia aplay

        ret_aplay = proc_play.wait()
        if ret_aplay != 0:
            err_play1 = b""
            try:
                err_play1 = proc_play.stderr.read() or b""
            except Exception:
                pass
            # Terminar Piper para evitar BrokenPipe masivo
            try:
                proc_tts.kill()
            except Exception:
                pass

            # Reintentar con plughw si el dispositivo era hw
            aplay_dev_env = os.getenv("APLAY_DEVICE", "")
            # Intento alternativo: 'default' puro
            if aplay_dev_env and aplay_dev_env != "default":
                aplay_cmd_def = ["aplay", "-q", "-D", "default", "-t", "wav", "-"]
                print(f"[TTS] Reintentando con dispositivo 'default': {' '.join(aplay_cmd_def)}")
                proc_tts = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                    bufsize=0,
                )
                assert proc_tts.stdin is not None and proc_tts.stdout is not None
                proc_play = subprocess.Popen(
                    aplay_cmd_def,
                    stdin=proc_tts.stdout,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                proc_tts.stdin.write((text.strip() + "\n").encode("utf-8"))
                proc_tts.stdin.flush()
                proc_tts.stdin.close()
                ret_aplay2 = proc_play.wait()
                if ret_aplay2 == 0:
                    _ = proc_tts.wait(timeout=60)
                    print("[TTS] Streaming Piper → aplay (default) finalizado correctamente")
                    return
                else:
                    try:
                        err_play_def = proc_play.stderr.read() or b""
                        if err_play_def:
                            print(f"[TTS] aplay (default) error: {err_play_def.decode(errors='ignore').strip()}")
                    except Exception:
                        pass

            if aplay_dev_env.startswith("hw:"):
                aplay_cmd2 = build_aplay_cmd(use_plug=True)
                print(f"[TTS] Reintentando con plughw: {' '.join(aplay_cmd2)}")
                # Relanzar Piper para un stream nuevo
                proc_tts = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                    bufsize=0,
                )
                assert proc_tts.stdin is not None and proc_tts.stdout is not None
                proc_play = subprocess.Popen(
                    aplay_cmd2,
                    stdin=proc_tts.stdout,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                proc_tts.stdin.write((text.strip() + "\n").encode("utf-8"))
                proc_tts.stdin.flush()
                proc_tts.stdin.close()
                ret_aplay = proc_play.wait()
                if ret_aplay == 0:
                    # Esperar Piper a terminar y salir
                    _ = proc_tts.wait(timeout=60)
                    print("[TTS] Streaming Piper → aplay (plughw) finalizado correctamente")
                    return
                else:
                    try:
                        err_play2 = proc_play.stderr.read() or b""
                    except Exception:
                        err_play2 = b""
                    print(f"[TTS] aplay (plughw) error: {err_play2.decode(errors='ignore').strip()}")
                    # Si tenemos sox, último intento: Piper → sox (48k/2ch) → aplay hw
                    if have_sox:
                        sox_cmd = [
                            "sox",
                            "-t", "wav", "-",
                            "-r", "48000",
                            "-b", "16",
                            "-c", "2",
                            "-t", "wav", "-",
                        ]
                        print(f"[TTS] Intentando conversión final con sox hacia hw: {' '.join(sox_cmd)}")
                        proc_tts = subprocess.Popen(
                            cmd,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=False,
                            bufsize=0,
                        )
                        assert proc_tts.stdin is not None and proc_tts.stdout is not None
                        proc_sox = subprocess.Popen(
                            sox_cmd,
                            stdin=proc_tts.stdout,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )
                        aplay_cmd_hw = ["aplay", "-q", "-D", aplay_dev_env, "-t", "wav", "-"]
                        print(f"[TTS] Encadenando aplay (hw) tras sox: {' '.join(aplay_cmd_hw)}")
                        proc_play = subprocess.Popen(
                            aplay_cmd_hw,
                            stdin=proc_sox.stdout,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                        )
                        proc_tts.stdin.write((text.strip() + "\n").encode("utf-8"))
                        proc_tts.stdin.flush()
                        proc_tts.stdin.close()
                        ret_aplay3 = proc_play.wait()
                        if ret_aplay3 == 0:
                            _ = proc_tts.wait(timeout=60)
                            print("[TTS] Streaming Piper → sox → aplay (hw) finalizado correctamente")
                            return
                        else:
                            try:
                                err_ap3 = proc_play.stderr.read() or b""
                                print(f"[TTS] aplay (hw tras sox) error: {err_ap3.decode(errors='ignore').strip()}")
                            except Exception:
                                pass

            # Si sigue mal, leer stderr de Piper para registro
            try:
                err_tts = proc_tts.stderr.read() or b""
                if err_play1:
                    print(f"[TTS] aplay error: {err_play1.decode(errors='ignore').strip()}")
                if err_tts:
                    print(f"[TTS] Piper CLI stderr: {err_tts.decode(errors='ignore').strip()}")
            except Exception:
                pass
            raise RuntimeError("aplay falló")
        else:
            # Esperar Piper a terminar y salir
            ret_tts = proc_tts.wait(timeout=60)
            if ret_tts == 0:
                print("[TTS] Streaming Piper → aplay finalizado correctamente")
                return
            else:
                try:
                    err_tts = proc_tts.stderr.read() or b""
                    if err_tts:
                        print(f"[TTS] Piper CLI stderr: {err_tts.decode(errors='ignore').strip()}")
                except Exception:
                    pass
                raise RuntimeError("Piper CLI falló")
    except Exception as exc:
        print(f"[TTS] Error en streaming Piper CLI: {exc}. Probando fallback Python...")

    # 2) Fallback: piper-tts → WAV en memoria → aplay (mayor compatibilidad)
    try:
        print("[TTS] Inicializando fallback piper-tts (WAV en memoria)…")
        global _piper_voice
        if _piper_voice is None:
            print(f"[TTS] Cargando modelo: {PIPER_MODEL}")
            from piper.voice import PiperVoice  # type: ignore
            _piper_voice = PiperVoice.load(PIPER_MODEL, PIPER_CONFIG)
            print(f"[TTS] Modelo cargado correctamente, sample rate: {_piper_voice.config.sample_rate}")

        pcm_iter = _piper_voice.synthesize(text)

        # Construir WAV completo en memoria (PCM16 mono)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(_piper_voice.config.sample_rate)
            for chunk in pcm_iter:
                try:
                    if hasattr(chunk, '__class__') and 'AudioChunk' in str(chunk.__class__):
                        if hasattr(chunk, 'pcm'):
                            if isinstance(chunk.pcm, (bytes, bytearray)):
                                data = chunk.pcm
                            elif hasattr(chunk.pcm, 'tobytes'):
                                data = chunk.pcm.tobytes()
                            else:
                                data = bytes(chunk.pcm)
                        elif hasattr(chunk, 'data'):
                            data = chunk.data
                        else:
                            data = bytes(chunk)
                    elif isinstance(chunk, (bytes, bytearray)):
                        data = chunk
                    elif hasattr(chunk, 'tobytes'):
                        data = chunk.tobytes()
                    else:
                        data = bytes(chunk)
                    if data:
                        wf.writeframes(data)
                except Exception:
                    continue
        wav_bytes = buf.getvalue()

        def build_wav_cmd(device: str | None) -> list:
            if device is None:
                return ["aplay", "-q", "-t", "wav", "-"]
            return ["aplay", "-q", "-D", device, "-t", "wav", "-"]

        # Intentar con APLAY_DEVICE
        device_env = os.getenv("APLAY_DEVICE")
        tried = []
        for dev in [device_env, "default", None]:
            if dev in tried:
                continue
            tried.append(dev)
            cmd_ap = build_wav_cmd(dev)
            try:
                p = subprocess.Popen(cmd_ap, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                assert p.stdin is not None
                p.stdin.write(wav_bytes)
                p.stdin.close()
                rc = p.wait()
                if rc == 0:
                    print("[TTS] Reproducción exitosa con WAV en memoria")
                    return
                else:
                    err = p.stderr.read() or b""
                    print(f"[TTS] aplay (WAV) código {rc} con dispositivo {dev or 'por defecto'}: {err.decode(errors='ignore').strip()}")
            except Exception as exc_:
                print(f"[TTS] Error lanzando aplay (WAV) con dispositivo {dev or 'por defecto'}: {exc_}")
    except Exception as exc:
        print(f"[TTS] Error en fallback Piper-tts (streaming): {exc}")


def _generate_beep_wav_bytes(frequency_hz: int, duration_ms: int, volume: float = 0.25) -> bytes:
    sr = 48000
    channels = 2
    total_samples = max(1, int(sr * duration_ms / 1000.0))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        two_pi_f_over_sr = 2.0 * math.pi * float(frequency_hz) / float(sr)
        peak = int(32767 * max(0.0, min(1.0, volume)))
        # Pequeño fade de entrada/salida para evitar clics
        fade_samples = max(1, int(0.004 * sr))
        for n in range(total_samples):
            sample = int(math.sin(two_pi_f_over_sr * n) * peak)
            if n < fade_samples:
                sample = int(sample * (n / fade_samples))
            tail = total_samples - 1 - n
            if tail < fade_samples:
                sample = int(sample * (tail / fade_samples))
            # Estéreo: duplicar muestra
            wf.writeframesraw(sample.to_bytes(2, byteorder="little", signed=True) * channels)
    return buf.getvalue()


def play_earcon(kind: str) -> None:
    try:
        if kind == "start_listen":
            wav_bytes = _generate_beep_wav_bytes(1200, 250, volume=0.55)
        elif kind == "end_listen":
            wav_bytes = _generate_beep_wav_bytes(1200, 180, volume=0.55)
        else:
            wav_bytes = _generate_beep_wav_bytes(1200, 200, volume=0.55)

        aplay_cmd = ["aplay", "-q", "-t", "wav", "-"]
        dev = os.getenv("APLAY_DEVICE")
        if dev:
            aplay_cmd = ["aplay", "-q", "-D", dev, "-t", "wav", "-"]
        p = subprocess.run(aplay_cmd, input=wav_bytes, capture_output=True)
        if p.returncode != 0 and dev and dev.startswith("hw:"):
            # Reintentar con 'default' si hw falla
            subprocess.run(["aplay", "-q", "-D", "default", "-t", "wav", "-"], input=wav_bytes)
    except Exception:
        # No romper el flujo por un beep
        pass


class AudioPipeline:
    """Tubería de audio persistente para enviar PCM16 mono a aplay,
    opcionalmente pasando por sox para convertir a parámetros que el hw acepte.
    """

    def __init__(self, input_rate: int) -> None:
        self.input_rate = int(input_rate)
        self.proc_sox: Optional[subprocess.Popen] = None
        self.proc_play: Optional[subprocess.Popen] = None
        self.stdin = None
        self._start_pipeline()

    def _start_pipeline(self) -> None:
        device = os.getenv("APLAY_DEVICE", "")
        have_sox = shutil.which("sox") is not None
        # Preferir conversión con sox si el destino es hw:*
        if device.startswith("hw:") and have_sox:
            print(f"[TTS-Pipeline] Usando sox → aplay (hw: conversión 48k/16bit/2ch) en {device}")
            sox_cmd = [
                "sox",
                "-t", "raw",
                "-r", str(self.input_rate),
                "-e", "signed",
                "-b", "16",
                "-c", "1",
                "-L",
                "-",
                "-r", "48000",
                "-b", "16",
                "-c", "2",
                "-t", "wav", "-",
            ]
            self.proc_sox = subprocess.Popen(
                sox_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            assert self.proc_sox.stdin is not None and self.proc_sox.stdout is not None
            self.stdin = self.proc_sox.stdin
            play_cmd = ["aplay", "-q", "-D", device, "-t", "wav", "-"] if device else ["aplay", "-q", "-t", "wav", "-"]
            self.proc_play = subprocess.Popen(
                play_cmd,
                stdin=self.proc_sox.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        else:
            # Directo a aplay en RAW (usa plughw si así está en APLAY_DEVICE)
            target = device if device else "(por defecto)"
            print(f"[TTS-Pipeline] Enviando RAW directo a aplay en {target} @ {self.input_rate}Hz mono S16_LE")
            play_cmd = [
                "aplay", "-q",
                "-t", "raw",
                "-f", "S16_LE",
                "-c", "1",
                "-r", str(self.input_rate),
                "-",
            ]
            if device:
                play_cmd = [
                    "aplay", "-q",
                    "-D", device,
                    "-t", "raw",
                    "-f", "S16_LE",
                    "-c", "1",
                    "-r", str(self.input_rate),
                    "-",
                ]
            self.proc_play = subprocess.Popen(
                play_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            assert self.proc_play.stdin is not None
            self.stdin = self.proc_play.stdin

    def write(self, data: bytes) -> None:
        if not data:
            return
        if self.stdin is None:
            return
        try:
            self.stdin.write(data)
            try:
                self.stdin.flush()
            except Exception:
                pass
        except Exception:
            # Intentar no romper la app si el dispositivo desaparece
            pass

    def close(self) -> None:
        try:
            if self.stdin:
                try:
                    self.stdin.flush()
                except Exception:
                    pass
                try:
                    self.stdin.close()
                except Exception:
                    pass
        finally:
            if self.proc_play is not None:
                # Esperar a que termine de reproducir todo el audio
                try:
                    self.proc_play.wait()
                except Exception:
                    pass
            if self.proc_sox is not None:
                try:
                    self.proc_sox.wait()
                except Exception:
                    pass


def _looks_like_sentence_end(buffer: str) -> bool:
    if not buffer:
        return False
    # Corta en finales de oración o salto de línea
    if re.search(r"[\.\?!¡!:\u2026]\s*$", buffer):
        return True
    if "\n" in buffer:
        return True
    # Evita segmentos demasiado largos
    if len(buffer) >= 200 and buffer.endswith(" "):
        return True
    # Evitar cortes agresivos en coma; requerir más longitud
    if len(buffer) >= 140 and buffer.strip().endswith(","):
        return True
    return False


def stream_and_speak_from_ollama(messages: list) -> str:
    print("[Streaming IA] Iniciando stream con Ollama y TTS en frases…")
    text_queue: "queue.Queue[Optional[str]]" = queue.Queue()
    full_reply: str = ""
    # Usar piper-tts directamente sobre una tubería continua para evitar cortes
    global _piper_voice
    if _piper_voice is None:
        from piper.voice import PiperVoice  # type: ignore
        _piper_voice = PiperVoice.load(PIPER_MODEL, PIPER_CONFIG)
    print(f"[TTS-Pipeline] Inicializando canal continuo a { _piper_voice.config.sample_rate } Hz")
    pipeline = AudioPipeline(input_rate=_piper_voice.config.sample_rate)

    def tts_worker() -> None:
        while True:
            segment = text_queue.get()
            try:
                if segment is None:
                    return
                seg = segment.strip()
                if not seg:
                    continue
                print(f"[TTS-Pipeline] Sintetizando segmento ({len(seg)} chars)…")
                # 1) Intento: extraer PCM directamente del iterador de piper-tts
                def to_bytes(obj) -> bytes:
                    if obj is None:
                        return b""
                    if isinstance(obj, (bytes, bytearray)):
                        return bytes(obj)
                    try:
                        return memoryview(obj).tobytes()  # type: ignore[arg-type]
                    except Exception:
                        pass
                    if hasattr(obj, 'tobytes'):
                        try:
                            return obj.tobytes()
                        except Exception:
                            pass
                    if hasattr(obj, 'astype'):
                        try:
                            return obj.astype('<i2').tobytes()
                        except Exception:
                            pass
                    try:
                        return bytes(obj)
                    except Exception:
                        return b""

                pcm_bytes_total = 0
                first_chunk_info = None
                try:
                    for idx, chunk in enumerate(_piper_voice.synthesize(seg)):
                        try:
                            data = b""
                            if hasattr(chunk, 'pcm'):
                                data = to_bytes(getattr(chunk, 'pcm'))
                            elif hasattr(chunk, 'data'):
                                data = to_bytes(getattr(chunk, 'data'))
                            else:
                                data = to_bytes(chunk)
                            if idx < 3 and first_chunk_info is None and data == b"":
                                first_chunk_info = f"tipo={type(chunk)} attrs={dir(chunk)[:6]}"
                            if data:
                                pcm_bytes_total += len(data)
                                pipeline.write(data)
                        except Exception:
                            continue
                except Exception:
                    pass

                # 2) Si no llegó PCM, usar CLI de Piper y extraer frames WAV
                used_cli = False
                if pcm_bytes_total == 0:
                    try:
                        cmd = ["piper", "-m", PIPER_MODEL, "-c", PIPER_CONFIG, "-f", "-"]
                        proc = subprocess.run(cmd, input=(seg + "\n").encode('utf-8'), capture_output=True)
                        used_cli = True
                        wav_bytes = proc.stdout
                        if wav_bytes:
                            rdr = wave.open(io.BytesIO(wav_bytes), 'rb')
                            frames = rdr.readframes(rdr.getnframes())
                            rdr.close()
                            if frames:
                                pipeline.write(frames)
                                pcm_bytes_total = len(frames)
                    except Exception:
                        pcm_bytes_total = 0

                if first_chunk_info and pcm_bytes_total == 0:
                    print(f"[TTS-Pipeline] Diagnóstico primer chunk vacío: {first_chunk_info}")
                print(f"[TTS-Pipeline] Segmento enviado ({pcm_bytes_total} bytes){' [cli]' if used_cli else ''}")
            finally:
                text_queue.task_done()

    worker_thread = threading.Thread(target=tts_worker, daemon=True)
    worker_thread.start()

    buffer: str = ""
    try:
        if OLLAMA_HOST:
            client = ollama.Client(host=OLLAMA_HOST)
            stream = client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
                options=OLLAMA_OPTIONS,
            )
        else:
            stream = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
                options=OLLAMA_OPTIONS,
            )

        for chunk in stream:
            try:
                piece = chunk.get("message", {}).get("content", "")
            except Exception:
                piece = ""
            if not piece:
                continue
            full_reply += piece
            buffer += piece
            # Emitir por frases
            if _looks_like_sentence_end(buffer):
                text_queue.put(buffer)
                buffer = ""

        # Vaciar lo que quede
        if buffer.strip():
            text_queue.put(buffer)
            buffer = ""
    except Exception as exc:
        print(f"[Streaming IA] Error durante streaming: {exc}")
    finally:
        # Señal de fin
        text_queue.put(None)
        text_queue.join()
        try:
            worker_thread.join(timeout=0.2)
        except Exception:
            pass
        pipeline.close()
        print("[TTS-Pipeline] Canal de audio cerrado")

    return full_reply.strip()


def create_recognizer() -> vosk.KaldiRecognizer:
    assert _vosk_model is not None
    rec = vosk.KaldiRecognizer(_vosk_model, SAMPLE_RATE)
    rec.SetWords(True)
    return rec


def _validate_piper_files() -> bool:
    try:
        if not os.path.isfile(PIPER_MODEL):
            print(f"[TTS] Modelo no encontrado: {PIPER_MODEL}")
            return False
        if not os.path.isfile(PIPER_CONFIG):
            print(f"[TTS] Config no encontrada: {PIPER_CONFIG}")
            return False
        model_size = os.path.getsize(PIPER_MODEL)
        cfg_size = os.path.getsize(PIPER_CONFIG)
        print(f"[TTS] Tamaños -> model: {model_size} bytes, config: {cfg_size} bytes")
        # Umbrales razonables (onnx suele ser > 1MB, json > 100 bytes)
        if model_size < 1_000_000:
            print(f"[TTS] Modelo demasiado pequeño (posible descarga incompleta): {PIPER_MODEL}")
            return False
        if cfg_size < 100:
            print(f"[TTS] Config demasiado pequeña (posible descarga incompleta): {PIPER_CONFIG}")
            return False
        with open(PIPER_CONFIG, "r", encoding="utf-8") as fh:
            json.load(fh)
        return True
    except Exception as exc:
        print(f"[TTS] Error leyendo JSON de config {PIPER_CONFIG}: {exc}")
        return False


def _discover_and_set_piper_voice() -> None:
    """Si los paths actuales no son válidos, intenta detectar cualquier voz válida en voices/."""
    global PIPER_MODEL, PIPER_CONFIG
    try:
        if not os.path.isdir(VOICES_DIR):
            return
        candidates = []
        for name in os.listdir(VOICES_DIR):
            if name.endswith(".onnx"):
                base = name[:-5]
                cfg = os.path.join(VOICES_DIR, base + ".onnx.json")
                if not os.path.isfile(cfg):
                    cfg = os.path.join(VOICES_DIR, base + ".json")
                model_path = os.path.join(VOICES_DIR, name)
                if os.path.isfile(model_path) and os.path.isfile(cfg):
                    # Filtra por tamaños mínimos para evitar HTML/descargas vacías
                    try:
                        if os.path.getsize(model_path) >= 1_000_000 and os.path.getsize(cfg) >= 100:
                            candidates.append((model_path, cfg))
                    except Exception:
                        continue
        # Prefiere voces de español
        candidates.sort(key=lambda t: (0 if "/es_" in t[0] or "es_" in os.path.basename(t[0]) else 1, t[0]))
        for model_path, cfg in candidates:
            try:
                with open(cfg, "r", encoding="utf-8") as fh:
                    json.load(fh)
                PIPER_MODEL = model_path
                PIPER_CONFIG = cfg
                print(f"[TTS] Voz Piper detectada: {PIPER_MODEL}, {PIPER_CONFIG}")
                return
            except Exception:
                continue
    except Exception:
        return


def create_wake_recognizer() -> vosk.KaldiRecognizer:
    assert _vosk_model is not None
    grammar = json.dumps([WAKE_WORD])
    recognizer = vosk.KaldiRecognizer(_vosk_model, SAMPLE_RATE, grammar)
    recognizer.SetWords(False)
    return recognizer


def wait_for_wake_word() -> None:
    q: "queue.Queue[bytes]" = queue.Queue()
    recognizer = create_wake_recognizer()

    def callback(indata, frames, t, status):
        if status:
            pass
        q.put(bytes(indata))

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCKSIZE,
            dtype="int16",
            channels=1,
            callback=callback,
            device=sd.default.device,
        ):
            print("Escuchando wake word...")
            while True:
                try:
                    data = q.get(timeout=1.0)  # Timeout para evitar bloqueo
                except:
                    continue

                if recognizer.AcceptWaveform(data):
                    res = json.loads(recognizer.Result())
                    txt = res.get("text", "").lower().strip()
                    if txt == WAKE_WORD:
                        print(f"Wake word detectada: '{txt}'")
                        # Pequeña pausa para evitar interferencia de audio residual
                        time.sleep(0.25)
                        return
    except Exception as exc:
        print(f"Error en wait_for_wake_word: {exc}")
        # Reintentar después de un breve delay
        time.sleep(1.0)
        return wait_for_wake_word()


def listen_command(recognizer: vosk.KaldiRecognizer) -> str:
    q: "queue.Queue[bytes]" = queue.Queue()
    last_voice_ts = time.time()
    start_ts = time.time()

    def callback(indata, frames, t, status):
        nonlocal last_voice_ts
        if status:
            pass
        audio = np.frombuffer(indata, dtype=np.int16)
        if _rms_int16(audio) > SILENCE_THRESHOLD:
            last_voice_ts = time.time()
        q.put(bytes(indata))

    # Beep de inicio de escucha
    try:
        play_earcon("start_listen")
    except Exception:
        pass

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        dtype="int16",
        channels=1,
        callback=callback,
        device=sd.default.device,
    ):
        transcript = ""
        while True:
            # Fin por silencio o timeout máximo
            if (time.time() - last_voice_ts) * 1000 > SILENCE_MS:
                # Obtener resultado final acumulado
                try:
                    final = json.loads(recognizer.FinalResult())
                    return final.get("text", "").strip()
                except Exception:
                    return transcript
            if time.time() - start_ts > MAX_COMMAND_SECS:
                try:
                    final = json.loads(recognizer.FinalResult())
                    return final.get("text", "").strip() or transcript
                except Exception:
                    return transcript

            try:
                data = q.get(timeout=0.2)
            except Exception:
                continue

            if recognizer.AcceptWaveform(data):
                res = json.loads(recognizer.Result())
                part = res.get("text", "").strip()
                if part:
                    transcript = (transcript + " " + part).strip()
            else:
                # opcional: usar parcial para feedback
                pass
    # Beep de fin de escucha
    try:
        play_earcon("end_listen")
    except Exception:
        pass


def main() -> None:
    _validate_piper_files()
    
    ensure_paths()
    # Cargar config de usuario (OpenWeather/ubicación/zone)
    global _config
    _config = load_config()
    # Lanzar siempre la UI de configuración en segundo plano
    start_config_server()
    print("Asistente listo. Di 'asistente' para activar.")
    cooldown_end_ts = 0.0

    while True:
        print("[Esperando palabra de activación]")

        # Evitar re-disparo inmediato por cooldown
        now = time.time()
        if now < cooldown_end_ts:
            time.sleep(max(0.0, cooldown_end_ts - now))

        # Esperar wake word
        wait_for_wake_word()
        print("[Wake word] detectada - cambiando a modo comando")

        # Crear nuevo recognizer para el comando
        command_recognizer = create_recognizer()
        print("[Escuchando comando] (habla ahora)")
        command = listen_command(command_recognizer)
        print(f"[Comando recibido]: '{command}'")

        if not command:
            print("[No se detectó comando] - volviendo a esperar wake word")
            cooldown_end_ts = time.time() + 1.0
            continue

        # Detección de intención con IA (fallback a heurística si falla)
        intent, _extras = classify_intent_via_llm(command)
        if intent == "weather":
            print("[Intent] Consulta de clima detectada")
            reply = handle_weather_command(command, when=_extras.get("when"))
            print(f"[IA resumen clima]: '{reply[:200]}...'")
            speak(reply)
        elif intent == "time":
            print("[Intent] Consulta de hora detectada")
            reply = handle_time_command()
            print(f"[IA resumen hora]: '{reply[:200]}...'")
            speak(reply)
        else:
            # Procesar con IA por defecto (streaming con síntesis por frases)
            messages = build_ollama_messages(command)
            print(f"[Procesando] Enviando a {OLLAMA_MODEL}: '{command[:100]}...'")
            print(f"[Prompt] Usando prompt del sistema: {OLLAMA_PROMPT[:100]}...")
            try:
                reply = stream_and_speak_from_ollama(messages)
            except Exception as exc:
                error = f"Hubo un error consultando el modelo: {exc}"
                print(f"[Error IA]: '{error}'")
                reply = error

        print(f"[Respuesta IA]: '{reply[:300]}...'")  # Primeros 300 chars

        # Cooldown antes de volver a esperar wake word
        cooldown_end_ts = time.time() + 2.0
        print("[Cooldown] Listo para nueva activación en 2 segundos")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSaliendo...")

