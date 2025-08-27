import os
import sys
import urllib.request


DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
VOICE_ONNX = "es_ES-sharvard-low.onnx"
VOICE_JSON = "es_ES-sharvard-low.onnx.json"

URL_BASES = [
    "https://github.com/rhasspy/piper/releases/latest/download",
]


def download(url: str, dest: str) -> bool:
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as exc:
        print(f"Fallo descargando {url}: {exc}")
        if os.path.exists(dest):
            try:
                os.remove(dest)
            except Exception:
                pass
        return False


def main() -> int:
    os.makedirs(DEST_DIR, exist_ok=True)
    onnx_ok = os.path.isfile(os.path.join(DEST_DIR, VOICE_ONNX))
    json_ok = os.path.isfile(os.path.join(DEST_DIR, VOICE_JSON))

    if onnx_ok and json_ok:
        print("Voz ya presente.")
        return 0

    for base in URL_BASES:
        if not onnx_ok:
            if download(f"{base}/{VOICE_ONNX}", os.path.join(DEST_DIR, VOICE_ONNX)):
                onnx_ok = True
        if not json_ok:
            if download(f"{base}/{VOICE_JSON}", os.path.join(DEST_DIR, VOICE_JSON)):
                json_ok = True
        if onnx_ok and json_ok:
            break

    if not (onnx_ok and json_ok):
        print("No se pudo descargar la voz es_ES-sharvard-low. Descárgala manualmente.")
        return 1

    print("Voz descargada correctamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

