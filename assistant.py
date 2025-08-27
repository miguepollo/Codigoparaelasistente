import os
import sys
import json
import time
import math
import queue
import subprocess
import io
import wave
from typing import Optional

import numpy as np
import sounddevice as sd
import vosk
import ollama


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOSK_MODEL_DIR = os.path.join(BASE_DIR, "models", "vosk")
PIPER_MODEL = os.path.join(BASE_DIR, "voices", "es_ES-sharvard-medium.onnx")
PIPER_CONFIG = os.path.join(BASE_DIR, "voices", "es_ES-sharvard-medium.onnx.json")
VOICES_DIR = os.path.join(BASE_DIR, "voices")
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
WAKE_WORD = "asistente"
SILENCE_THRESHOLD = 300  # RMS ~ energía. Ajustar si hace falta
SILENCE_MS = 650  # fin por silencio
MAX_COMMAND_SECS = 12
OLLAMA_MODEL = "gemma3:1b"

_piper_voice = None  # Lazy init para fallback Python
_vosk_model: Optional[vosk.Model] = None  # Reutilizar modelo en memoria


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
    _discover_and_set_piper_voice()
    # Validar ficheros de voz Piper
    if not _validate_piper_files():
        print("[TTS] Archivos de voz Piper ausentes o corruptos. Omite TTS.")
        return
    # 1) Intento con CLI (rápido y ligero) + aplay (evita PortAudio)
    try:
        cmd = ["piper", "-m", PIPER_MODEL, "-c", PIPER_CONFIG, "-f", "-"]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write((text.strip() + "\n").encode("utf-8"))
        proc.stdin.close()
        wav_bytes = proc.stdout.read()
        stderr_bytes = proc.stderr.read() if proc.stderr else b""
        proc.wait(timeout=30)
        if wav_bytes:
            # Reproducir con aplay para evitar PortAudio
            try:
                aplay_cmd = ["aplay", "-q", "-t", "wav", "-"]
                aplay_dev = os.getenv("APLAY_DEVICE")
                if aplay_dev:
                    aplay_cmd = ["aplay", "-q", "-D", aplay_dev, "-t", "wav", "-"]
                p = subprocess.Popen(aplay_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                assert p.stdin is not None
                p.stdin.write(wav_bytes)
                p.stdin.close()
                p.wait()
                return
            except Exception as exc_play:
                print(f"[TTS] Error reproduciendo con aplay: {exc_play}")
        else:
            if stderr_bytes:
                print(f"[TTS] Piper CLI stderr: {stderr_bytes.decode(errors='ignore').strip()}")
            print("[TTS] Piper no devolvió audio por CLI. Probando fallback Python...")
    except Exception as exc:
        print(f"[TTS] Error en Piper CLI: {exc}. Probando fallback Python...")

    # 2) Fallback: librería piper-tts (ONNXRuntime)
    try:
        global _piper_voice
        if _piper_voice is None:
            from piper.voice import PiperVoice  # type: ignore
            _piper_voice = PiperVoice.load(PIPER_MODEL, PIPER_CONFIG)
        pcm_iter = _piper_voice.synthesize(text)
        # Construir WAV en memoria desde el generador PCM16 mono
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(_piper_voice.config.sample_rate)
            for chunk in pcm_iter:
                wf.writeframes(chunk)
        wav_bytes = buf.getvalue()
        try:
            aplay_cmd = ["aplay", "-q", "-t", "wav", "-"]
            aplay_dev = os.getenv("APLAY_DEVICE")
            if aplay_dev:
                aplay_cmd = ["aplay", "-q", "-D", aplay_dev, "-t", "wav", "-"]
            p = subprocess.Popen(aplay_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            assert p.stdin is not None
            p.stdin.write(wav_bytes)
            p.stdin.close()
            p.wait()
            return
        except Exception as exc_play:
            print(f"[TTS] Error reproduciendo con aplay (fallback): {exc_play}")
    except Exception as exc:
        print(f"[TTS] Error en fallback Piper-tts: {exc}")


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
    if _validate_piper_files():
        return
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
    print(f"grammar: {grammar}")
    recognizer = vosk.KaldiRecognizer(_vosk_model, SAMPLE_RATE, grammar)
    print("Despues de crear el recognizer")
    recognizer.SetWords(False)
    print("Despues de setear el recognizer")
    return recognizer


def wait_for_wake_word() -> None:
    q: "queue.Queue[bytes]" = queue.Queue()
    recognizer = create_wake_recognizer()

    def callback(indata, frames, t, status):
        if status:
            pass
        q.put(bytes(indata))
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        dtype="int16",
        channels=1,
        callback=callback,
        device=sd.default.device,
    ):

        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                res = json.loads(recognizer.Result())
                txt = res.get("text", "").lower()
                print(f"txt: {txt}")
                if txt == WAKE_WORD:
                    return


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


def main() -> None:
    ensure_paths()
    print("Asistente listo. Di 'asistente' para activar.")
    cooldown_end_ts = 0.0

    while True:
        print("[Esperando palabra de activación]")
        # Evitar re-disparo inmediato
        now = time.time()
        if now < cooldown_end_ts:
            time.sleep(max(0.0, cooldown_end_ts - now))
        wait_for_wake_word()
        print("[Wake word] detectada")
        # Nuevo recognizer para el siguiente enunciado
        command_recognizer = create_recognizer()
        command = listen_command(command_recognizer)
        print(f"command: {command}")
        if not command:
            cooldown_end_ts = time.time() + 1.0
            continue
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": command}],
            )
            reply = response["message"]["content"].strip()
        except Exception as exc:
            reply = f"Hubo un error consultando el modelo: {exc}"
        print(f"[IA] {reply}")
        speak(reply)
        # Pequeño cooldown para evitar re-disparos con ruido residual
        cooldown_end_ts = time.time() + 1.5


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSaliendo...")

