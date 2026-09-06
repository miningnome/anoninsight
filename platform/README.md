# Video Intelligence Platform (`vip`)

> Any camera. Local AI. Modular vision. Extensible intelligence.

Aplicacion independiente dentro de este monorepo (ver
`docs/video-intelligence-platform/ARCHITECTURE.md` para el analisis y la
arquitectura completa). No depende de `server/`; para el MVP usa el paquete
`insightface` ya vendido en `python-package/` de este mismo repositorio.

## Estado actual: Fase 2

- Fase 1: conectarse a una camara (RTSP, USB o fichero de video),
  decodificar los frames y mostrarlos en vivo en una pagina web.
- Fase 2: deteccion facial (SCRFD via `insightface`) integrada en el
  Frame Processing Pipeline, con bounding boxes dibujados en vivo sobre el
  video. Sin reconocimiento todavia (llega en la Fase 3/4).

Por defecto el motor de vision es `mock` (no requiere modelos ni red) para
que la plataforma arranque sin dependencias pesadas; ver "Activar
deteccion facial real" mas abajo. Las fases posteriores (reconocimiento,
eventos, persistencia, multicamara, ONVIF, etc.) se describen en la
seccion 11 de `ARCHITECTURE.md`.

## Instalacion (desarrollo)

```bash
cd platform
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Activar deteccion facial real

```bash
pip install -e ".[vision]"
```

Y en `config/settings.toml` cambia `[vision] engine = "insightface"`. La
primera vez que arranque descargara el paquete de modelos (`buffalo_l` por
defecto) si hay conexion a internet; si no, colocalo manualmente en
`~/.insightface/models/`.

## Configuracion

Copia `config/settings.example.toml` a `config/settings.toml` y ajusta al
menos una entrada `[[cameras]]` con la URL RTSP real, o usa
`source_type = "file"` apuntando a un video local para probar sin hardware.

Variable de entorno `VIP_CONFIG_FILE` para apuntar a un fichero de
configuracion en otra ruta (por defecto `config/settings.toml`).

## Ejecutar

```bash
uvicorn vip.api.app:app --host 0.0.0.0 --port 8088 --app-dir src
```

Abre `http://localhost:8088/` para ver el listado de camaras configuradas y
el video en vivo de cada una (`/api/cameras/{id}/stream.mjpg`).

## Tests

```bash
pytest
```

Los tests de camaras no requieren hardware RTSP real: inyectan una fabrica
de captura falsa (mismo patron que usa `server/` para sus propios tests).
