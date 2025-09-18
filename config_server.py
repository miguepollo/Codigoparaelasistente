import os
import json
from typing import Dict

from flask import Flask, request, redirect, url_for, render_template_string

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


def load_config() -> Dict[str, str]:
    cfg = {
        "owm_api_key": "",
        "city": "",
        "lat": "",
        "lon": "",
        "timezone": "Europe/Madrid",
    }
    try:
        if os.path.isfile(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
                if isinstance(data, dict):
                    cfg.update({k: data.get(k, cfg[k]) for k in cfg.keys()})
    except Exception:
        pass
    return cfg


def save_config(cfg: Dict[str, str]) -> None:
    tmp_path = CONFIG_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, CONFIG_PATH)


app = Flask(__name__)


TEMPLATE = """
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Config Asistente</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, "Apple Color Emoji", "Segoe UI Emoji"; max-width: 880px; margin: 20px auto; padding: 0 16px; }
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
    <form method="post" action="{{ url_for('save') }}">
      <div class="card">
        <h3>OpenWeatherMap</h3>
        <label>API Key</label>
        <input type="password" name="owm_api_key" value="{{ cfg.get('owm_api_key','') }}" placeholder="tu_api_key" />
        <div class="row">
          <div>
            <label>Ciudad</label>
            <input type="text" name="city" value="{{ cfg.get('city','') }}" placeholder="Madrid" />
            <div class="note">Puedes dejar vacío si usas lat/lon</div>
          </div>
          <div></div>
        </div>
        <div class="row">
          <div>
            <label>Latitud</label>
            <input type="text" name="lat" value="{{ cfg.get('lat','') }}" placeholder="40.4168" />
          </div>
          <div>
            <label>Longitud</label>
            <input type="text" name="lon" value="{{ cfg.get('lon','') }}" placeholder="-3.7038" />
          </div>
        </div>
      </div>

      <div class="card">
        <h3>Zona horaria</h3>
        <label>Timezone IANA</label>
        <input type="text" name="timezone" value="{{ cfg.get('timezone','Europe/Madrid') }}" placeholder="Europe/Madrid" />
        <div class="note">Ejemplos: Europe/Madrid, America/Mexico_City, America/Bogota</div>
      </div>

      <div class="actions">
        <button type="submit">Guardar</button>
      </div>
    </form>
    <p class="note">Tras guardar, reinicia el asistente si está corriendo.</p>
  </body>
  </html>
"""


@app.get("/")
def index():
    return render_template_string(TEMPLATE, cfg=load_config())


@app.post("/save")
def save():
    cfg = load_config()
    for key in ["owm_api_key", "city", "lat", "lon", "timezone"]:
        val = request.form.get(key, "").strip()
        cfg[key] = val
    save_config(cfg)
    return redirect(url_for("index"))


if __name__ == "__main__":
    # Escucha en todas las interfaces para acceso desde la red local
    app.run(host="0.0.0.0", port=5000, debug=False)


