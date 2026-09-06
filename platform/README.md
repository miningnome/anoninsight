# Video Intelligence Platform (`vip`)

> Any camera. Local AI. Modular vision. Extensible intelligence.

Aplicacion independiente dentro de este monorepo (ver
`docs/video-intelligence-platform/ARCHITECTURE.md` para el analisis y la
arquitectura completa). No depende de `server/`; para el MVP usa el paquete
`insightface` ya vendido en `python-package/` de este mismo repositorio.

## Estado actual: Fase 1

Conectarse a una camara (RTSP, USB o fichero de video), decodificar los
frames y mostrarlos en vivo en una pagina web, sin analisis de IA todavia.
Las fases posteriores (deteccion facial, reconocimiento, eventos,
persistencia, multicamara, ONVIF, etc.) se describen en la seccion 11 de
`ARCHITECTURE.md`.

## Instalacion (desarrollo)

```bash
cd platform
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

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
