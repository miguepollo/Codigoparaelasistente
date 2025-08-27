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
PIPER_MODEL = os.path.join(BASE_DIR, "voices", "es_ES-sharvard-low.onnx")
PIPER_CONFIG = os.path.join(BASE_DIR, "voices", "es_ES-sharvard-low.onnx.json")

SAMPLE_RATE = 16000
BLOCKSIZE = 4096  # Latencia menor que 8000
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
    # Validar ficheros de voz Piper
    if not _validate_piper_files():
        print("[TTS] Archivos de voz Piper ausentes o corruptos. Omite TTS.")
        return
    # 1) Intento con CLI (rápido y ligero)
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
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            if sample_width != 2:
                raise RuntimeError("Formato no soportado (se espera PCM16)")
            audio = np.frombuffer(frames, dtype=np.int16)
            if num_channels > 1:
                audio = audio.reshape(-1, num_channels)
            sd.play(audio, samplerate=sample_rate)
            sd.wait()
            return
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
        pcm = _piper_voice.synthesize(text)
        audio = np.frombuffer(pcm, dtype=np.int16)
        sd.play(audio, samplerate=_piper_voice.config.sample_rate)
        sd.wait()
    except Exception as exc:
        print(f"[TTS] Error en fallback Piper-tts: {exc}")


def create_recognizer() -> vosk.KaldiRecognizer:
    assert _vosk_model is not None
    rec = vosk.KaldiRecognizer(_vosk_model, SAMPLE_RATE)
    rec.SetWords(True)
    return rec


def _validate_piper_files() -> bool:
    try:
        if not os.path.isfile(PIPER_MODEL) or os.path.getsize(PIPER_MODEL) < 1024:
            return False
        if not os.path.isfile(PIPER_CONFIG) or os.path.getsize(PIPER_CONFIG) < 20:
            return False
        with open(PIPER_CONFIG, "r", encoding="utf-8") as fh:
            json.load(fh)
        return True
    except Exception:
        return False


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
    print("Antes de crear el recognizer")
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        print("Antes del while")
        while True:
            print("Antes de get")
            data = q.get()
            print("Despues de get")
            if recognizer.AcceptWaveform(data):
                print("Antes de Result")
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
        print("Aquí funciona?")
        command = listen_command(command_recognizer)
        print(f"command: {command}")
        if not command:
            cooldown_end_ts = time.time() + 1.0
            continue
        print(f"[Usuario] {command}")
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

