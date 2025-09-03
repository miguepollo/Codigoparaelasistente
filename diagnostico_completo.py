#!/usr/bin/env python3
"""
DIAGNÓSTICO COMPLETO DEL ASISTENTE IA
Un solo script que verifica TODO
"""
import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE_DIR, "voices", "es_ES-sharvard-medium.onnx")
CONFIG = os.path.join(BASE_DIR, "voices", "es_ES-sharvard-medium.onnx.json")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.1.162:8080")

def check_syntax():
    """Verifica sintaxis de archivos Python"""
    print("📝 VERIFICANDO SINTAXIS...")

    files = ["assistant.py"]
    all_ok = True

    for filename in files:
        filepath = os.path.join(BASE_DIR, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    compile(f.read(), filepath, 'exec')
                print(f"  ✓ {filename}")
            except SyntaxError as e:
                print(f"  ✗ {filename}: ERROR linea {e.lineno}")
                all_ok = False
        else:
            print(f"  - {filename}: no encontrado")
            all_ok = False

    return all_ok

def check_files():
    """Verifica archivos de voz"""
    print("📁 VERIFICANDO ARCHIVOS...")

    ok = True

    if not os.path.exists(MODEL):
        print("  ✗ Modelo no encontrado")
        ok = False
    else:
        size = os.path.getsize(MODEL)
        print(f"  ✓ Modelo: {size} bytes")

    if not os.path.exists(CONFIG):
        print("  ✗ Config no encontrada")
        ok = False
    else:
        size = os.path.getsize(CONFIG)
        print(f"  ✓ Config: {size} bytes")

    return ok

def check_piper():
    """Verifica Piper (CLI y Python)"""
    print("🎤 VERIFICANDO PIPER...")

    # Primero intentar CLI
    try:
        cmd = ["piper", "-m", MODEL, "-c", CONFIG, "-f", "-"]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False
        )

        proc.stdin.write(b"Test.")
        proc.stdin.flush()
        proc.stdin.close()

        stdout, stderr = proc.communicate(timeout=10)

        if proc.returncode == 0 and len(stdout) > 0:
            print("  ✓ Piper CLI funciona")
            return True
    except Exception as e:
        print("  ⚠ Piper CLI no disponible (probando Python)")

    # Intentar Python
    try:
        from piper.voice import PiperVoice
        voice = PiperVoice.load(MODEL, CONFIG)

        text = "Hola."
        pcm_iter = voice.synthesize(text)
        chunks = list(pcm_iter)

        if len(chunks) > 0:
            print("  ✓ Piper Python funciona")
            return True
        else:
            print("  ✗ Piper Python no genera audio")
            return False

    except Exception as e:
        print(f"  ✗ Piper error: {str(e)[:50]}...")
        return False

def check_ollama():
    """Verifica conexión con Ollama"""
    print("🤖 VERIFICANDO OLLAMA...")

    try:
        import ollama

        if OLLAMA_HOST:
            client = ollama.Client(host=OLLAMA_HOST)
            response = client.list()
        else:
            response = ollama.list()

        if response:
            print("  ✓ Ollama conectado")
            return True
        else:
            print("  ✗ Ollama no responde")
            return False

    except Exception as e:
        print(f"  ✗ Error Ollama: {str(e)[:50]}...")
        return False

def main():
    print("🔍 DIAGNÓSTICO COMPLETO DEL ASISTENTE IA")
    print("=" * 50)

    results = {}

    # 1. Sintaxis
    results['syntax'] = check_syntax()

    # 2. Archivos
    results['files'] = check_files()

    # 3. Piper
    results['piper'] = check_piper()

    # 4. Ollama
    results['ollama'] = check_ollama()

    # Resumen
    print("\n" + "=" * 50)
    print("RESUMEN:")

    status = [
        ("Sintaxis", results.get('syntax', False)),
        ("Archivos", results.get('files', False)),
        ("Piper", results.get('piper', False)),
        ("Ollama", results.get('ollama', False)),
    ]

    all_ok = True
    for name, ok in status:
        icon = "✅" if ok else "❌"
        print(f"{icon} {name}: {'OK' if ok else 'ERROR'}")
        if not ok:
            all_ok = False

    print("\n" + "=" * 50)
    if all_ok:
        print("🎉 ¡TODO LISTO! El asistente está funcionando.")
        print("\nPara ejecutar:")
        print("  python3 assistant.py")
    else:
        print("⚠️  HAY PROBLEMAS:")
        print("- Copia todos los archivos a ~/assistant/")
        print("- Ejecuta: ./setup_orangepi.sh")
        print("- Verifica que Ollama esté ejecutándose")

    return all_ok

if __name__ == "__main__":
    success = main()
    print(f"\nEstado final: {'✅ OK' if success else '❌ ERROR'}")
    sys.exit(0 if success else 1)
