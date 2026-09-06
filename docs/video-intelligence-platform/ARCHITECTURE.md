# Video Intelligence Platform - Analisis de arquitectura (Fase 0)

> "Any camera. Local AI. Modular vision. Extensible intelligence."

Estado: **propuesta pendiente de validacion**. Este documento responde a los 12
puntos solicitados como primera tarea del proyecto. No se ha implementado
ningun codigo de aplicacion todavia; solo se documenta el analisis y la
arquitectura propuesta, tal y como se pidio explicitamente.

---

## 0. Hallazgo previo: este repositorio no parte de cero

Antes de proponer nada hay que dejar constancia de un hecho que condiciona
todo el resto del analisis: `anoninsight` es un fork del repositorio oficial
de InsightFace, y ya contiene en `server/` un producto llamado **"InsightFace
Server"** (anadido recientemente en el historial: commits `575ffbc` y
`7fadd42`) que es mucho mas que un prototipo. Un analisis de codigo (agente de
exploracion dedicado) confirma que ya implementa, con tests de concurrencia
extensos:

- **Ingesta RTSP con reconexion**: `server/backend/insightface_server/services/rtsp.py`.
  Un `MonitorSession` por camara, con threads de lectura e inferencia
  separados, backoff exponencial (1s -> 30s) en la reconexion, deteccion de
  perdida/recuperacion de senal, y throttling de FPS de analisis
  independiente del FPS de captura.
- **Motor de inferencia facial real**: `inference/onnx_engine.py`, con SCRFD
  (deteccion multi-resolucion) + ArcFace (embeddings) sobre ONNX Runtime,
  seleccion explicita de proveedor CPU/CUDA (sin fallback silencioso),
  auditoria de que los kernels realmente corren en GPU, y validacion de
  hardware (compute capability, version de driver).
- **Busqueda vectorial**: capa `search/` con backend nativo en C++/CUDA
  (cuantizacion FP32/FP16/BF16/INT8) y un backend Python de referencia,
  sincronizados con SQLite mediante un patron *outbox* (`search_changes`).
- **Persistencia**: SQLite con migraciones numeradas, esquema
  `Collection -> Person -> FaceSample`, cifrado de credenciales RTSP en
  reposo (AES-GCM).
- **API REST** bajo `/v1` (deteccion, comparacion, enrolamiento, busqueda
  1:N, CRUD de "Monitors" RTSP, eventos paginados, preview MJPEG) con
  autenticacion por API key.
- **Web UI** con paginas de Dashboard, Collections, People, Detect, Compare,
  Search, **Camera monitoring (RTSP)**, System y Help, multi-idioma.
- **Dockerizacion** completa (imagenes CPU y CUDA12 separadas, no-root,
  read-only, healthchecks) y un SDK Python tipado.

Sin embargo, ese sistema esta **deliberadamente acoplado al dominio "cara"**:

- No existe una abstraccion `CameraSource`: la apertura de video es
  `cv2.VideoCapture` directo sobre una URL, y `source_type` esta fijado en
  BD con `CHECK(source_type = 'rtsp')`. No hay ruta para camara USB, fichero
  de video u ONVIF sin migrar el esquema.
- El bucle de inferencia por frame llama directamente a
  `FaceService.search_all_faces()`. No existe una interfaz `VisionEngine`
  generica: `InferenceEngine.analyze()` devuelve siempre
  `list[FaceObservation]` (landmarks de 5 puntos, pose facial, embedding
  ArcFace). Anadir deteccion de objetos, OCR o tracking generico exigiria
  una interfaz paralela, no una extension de esta.
- La busqueda vectorial esta nombrada por dominio (`search_persons`,
  `person_numeric_id`) y el backend nativo tiene la dimension fijada a 512
  (ArcFace).
- **No hay tabla de eventos persistente**: los eventos de "Monitor" viven en
  un buffer en memoria (`deque`) y se pierden al reiniciar el proceso. El
  requisito del MVP de "registrar los eventos de deteccion en la base de
  datos" **no esta cubierto hoy** para el caso RTSP.
- Autenticacion de una sola API key global de servicio, sin usuarios, roles
  ni auditoria persistente.

**Conclusion practica**: `server/` es un producto solido, testeado y
versionado del proyecto InsightFace para el caso de uso "reconocimiento
facial self-hosted". La vision del usuario (plataforma multi-motor,
multi-camara, con eventos genericos y modelo de identidad/inteligencia
independiente del proveedor) es mas amplia que ese producto. Reescribir el
interior de `server/` para generalizarlo seria cirugia invasiva sobre un
componente ya publicado, con su propio ciclo de release, licenciamiento de
modelos y documentacion en 9 idiomas.

Por eso la recomendacion de este documento (ver seccion 2.1) es **construir
la Video Intelligence Platform como una aplicacion nueva e independiente**,
en un directorio propio del monorepo, que para el MVP reutiliza el paquete
`insightface` ya vendido en `python-package/` (SCRFD + ArcFace) en lugar del
interior de `server/`, y que deja la puerta abierta a integrar `server/`
mas adelante como un backend `FaceEngine` alternativo de alto rendimiento
(via su API REST/SDK) cuando el volumen de identidades o el uso de GPU a
escala lo justifique. Esta decision se plantea explicitamente al usuario en
la seccion 12 antes de tocar codigo.

---

## 1. Problemas tecnicos identificados

1. **Confusion de alcance con el codigo existente** (ver seccion 0): sin
   este analisis se habria empezado a duplicar reconexion RTSP, gestion de
   proveedores ONNX y busqueda vectorial ya resueltas en `server/`.
2. **GIL de Python y tiempo real**: decodificar N streams RTSP + correr
   inferencia + servir API + servir video en un unico proceso Python puede
   saturar el GIL. Mitigacion: la decodificacion via OpenCV/FFmpeg libera el
   GIL durante las llamadas nativas; la inferencia ONNX Runtime tambien
   libera el GIL durante el forward pass; aun asi, cada camara debe vivir en
   su propio hilo (o proceso) y el numero maximo de camaras debe ser
   configurable y limitado explicitamente (`max_cameras`), igual que hace
   `server/` con `rtsp_max_streams`.
3. **Contencion de GPU multi-motor**: si en el futuro corren a la vez
   FaceEngine, ObjectDetectionEngine y OCREngine sobre la misma GPU, hace
   falta un limitador de concurrencia de inferencia compartido (inspirado en
   `InferenceConcurrencyLimiter` de `server/`), no un semaforo por motor.
4. **Frecuencia de analisis vs. frecuencia de captura**: no se debe analizar
   cada frame. Hace falta desacoplar "FPS de captura" (lo que entrega la
   camara) de "FPS de analisis" (lo que se envia al VisionEngine) y de "FPS
   de visualizacion" (lo que ve el usuario en el dashboard), cada uno
   configurable de forma independiente.
5. **Reconexion de camaras**: RTSP sobre redes locales corta con frecuencia
   (reinicio de la camara, saturacion de red, timeouts del servidor RTSP).
   Hace falta backoff exponencial con techo, distincion entre "reconectando"
   y "error persistente", y eventos propios (`camera_online`,
   `camera_offline`) para que el dashboard y el Event Engine lo reflejen.
6. **Consistencia entre tracking en memoria y eventos persistidos**: si el
   tracking (confirmar una persona tras N frames consecutivos, expirar tras
   ausencia) vive solo en memoria y el proceso se reinicia, se pueden perder
   eventos "exit" o duplicar "enter". Hace falta decidir que parte del
   estado de tracking es efimera (aceptable) y que parte debe sobrevivir a
   un reinicio (los eventos ya emitidos, no el estado de tracking en si).
7. **Versionado de embeddings/modelos**: si se cambia el modelo de
   reconocimiento facial (p. ej. de `buffalo_l` a otro), los embeddings
   antiguos dejan de ser comparables. Hace falta guardar `model_id` +
   `model_version` junto a cada embedding y bloquear comparaciones cruzadas
   entre modelos incompatibles (patron que `server/` ya aplica bien y que se
   debe copiar conceptualmente).
8. **Privacidad y datos biometricos**: los embeddings faciales son datos
   biometricos sensibles (categoria especial bajo RGPD). Hace falta cifrado
   en reposo de credenciales de camara, separacion logica entre imagen,
   embedding, identidad y metadatos (tal como pide el enunciado), y que el
   procesamiento nunca salga de la red local en esta fase.
9. **Ambiguedad en el matching**: un `match_threshold` fijo global no es
   suficiente; hace falta permitir threshold por camara o por caso de uso
   (zona de alta seguridad vs. zona publica), y siempre devolver el score
   crudo (no redondeado) para evitar falsos positivos/negativos cerca del
   umbral.
10. **Escalado multi-camara**: el motor de inferencia debe ser un recurso
    compartido y con cupo (no una instancia por camara), para que anadir
    camaras no multiplique el consumo de VRAM/RAM linealmente sin control.
11. **Dependencia de fabricante**: usar directamente `cv2.VideoCapture(url)`
    en toda la aplicacion acopla el codigo a RTSP. Hace falta la abstraccion
    `CameraSource` desde el primer commit, aunque el MVP solo implemente
    `RTSPCamera`.
12. **SQLite vs. Postgres**: ver justificacion en la seccion 2.3. Hay que
    decidirlo ahora porque afecta al ORM y a las migraciones desde el
    principio.

---

## 2. Arquitectura propuesta

### 2.1 Decision de integracion con `server/`

Recomendacion (a validar, ver seccion 12): la Video Intelligence Platform
("VIP" en este documento) se construye como **aplicacion independiente**,
en un nuevo directorio de primer nivel `platform/`, con su propio
`pyproject.toml`. Para el MVP:

- El `FaceEngine` de VIP usa directamente el paquete `insightface` que ya
  vive en `python-package/` de este mismo monorepo (SCRFD + ArcFace via
  `insightface.app.FaceAnalysis`), no el interior de `server/`. Esto evita
  acoplarse a las restricciones de produccion de `server/` (auditoria
  estricta de CUDA, licenciamiento firmado de modelos, ABI nativo de
  busqueda) que son apropiadas para un producto comercial pero frenan la
  velocidad de un MVP en fase de validacion.
- La busqueda vectorial del MVP usa **FAISS** (`IndexFlatIP` sobre
  embeddings L2-normalizados, equivalente a similitud coseno), tal como
  pide el enunciado, en lugar del backend nativo C++/CUDA de `server/`.
- `server/` queda intacto. Mas adelante (fuera del MVP, ver Fase 9+ en la
  hoja de ruta) se puede anadir un segundo `FaceEngine` que delegue en la
  API REST/SDK de `server/` como backend de alto rendimiento (50M+
  vectores, cuantizacion INT8) para despliegues que lo necesiten, sin haber
  tenido que diseñar VIP en torno a esa restriccion desde el dia uno.

Esta decision se plantea explicitamente al usuario mas abajo porque es una
bifurcacion real con costes distintos (ver seccion 12).

### 2.2 Diagrama de capas

```
                         ┌─────────────────────────────┐
                         │        Web Dashboard        │
                         │  (video HUD, eventos, admin)│
                         └───────────────┬─────────────┘
                                         │ REST + SSE/WS
                         ┌───────────────▼─────────────┐
                         │            API               │
                         │  FastAPI, auth, schemas      │
                         └───────────────┬─────────────┘
                                         │
                         ┌───────────────▼─────────────┐
                         │      Event Engine             │
                         │ dedup, cooldown, persistencia │
                         └───────────────┬─────────────┘
                                         │
                         ┌───────────────▼─────────────┐
                         │  Identity / Intelligence      │
                         │  matching vectorial (FAISS)   │
                         └───────────────┬─────────────┘
                                         │
        ┌───────────────────────────────▼───────────────────────────────┐
        │                        Vision Engine                          │
        │  FaceEngine | ObjectDetectionEngine* | OCREngine* | Tracking*  │
        │                    (* futuro, no en MVP)                      │
        └───────────────▲───────────────────────────────────────────────┘
                         │
        ┌────────────────┴──────────────┐
        │   Frame Processing Pipeline    │
        │ decode -> sample -> dispatch   │
        └────────────────▲──────────────┘
                         │
        ┌────────────────┴──────────────┐
        │      Video Ingestion Layer     │
        │  CameraSource: RTSP|USB|File   │
        └────────────────▲──────────────┘
                         │
                     Camara IP (RTSP)
```

Cada capa se comunica con la de abajo solo a traves de interfaces
(`CameraSource`, `VisionEngine`, `VectorIndex`), nunca con implementaciones
concretas. Esto es lo que permite sustituir InsightFace por otro motor, o
RTSP por ONVIF, sin rediseñar el resto.

### 2.3 SQLite vs. PostgreSQL (justificacion)

**Decision para el MVP: SQLite.**

Razones:
- Despliegue de un solo nodo, un unico proceso escritor (igual que
  `server/`, que ya adopto SQLite con WAL para el mismo caso de uso y
  documenta por que funciona bien a esa escala).
- Cero infraestructura adicional: encaja con "Local AI" y con probar el
  MVP en un portatil o una Raspberry Pi/mini-PC junto a las camaras.
- Los patrones de acceso del MVP (una escritura por evento/frame analizado,
  lecturas ocasionales desde la API) no requieren un servidor de base de
  datos separado.
- El ORM se abstrae con SQLAlchemy desde el principio, y las migraciones se
  escriben en SQL plano numerado (mismo patron que `server/migrations`),
  de forma que migrar a PostgreSQL en una fase posterior (multi-tenant,
  multiples nodos de procesamiento, Fase 12) sea un cambio de cadena de
  conexion y de tipos especificos de columna, no una reescritura.

Postgres se adopta cuando aparezca cualquiera de estas condiciones: mas de
un proceso escribiendo eventos a la vez (multi-nodo), necesidad de
replicacion/alta disponibilidad, o multi-tenant real con aislamiento por
esquema.

---

## 3. Estructura de repositorio propuesta

```
platform/
├── pyproject.toml
├── README.md
├── config/
│   └── settings.example.toml
├── src/
│   └── vip/
│       ├── core/
│       │   ├── config.py            # Settings (env + toml), device policy
│       │   └── logging.py           # logging estructurado (structlog/json)
│       ├── cameras/
│       │   ├── base.py              # CameraSource (ABC), Frame
│       │   ├── rtsp.py              # RTSPCamera
│       │   ├── usb.py               # USBCamera
│       │   ├── video_file.py        # VideoFileCamera
│       │   └── onvif.py             # placeholder, Fase 8
│       ├── pipeline/
│       │   ├── ingestion.py         # CameraWorker: hilo lectura + reconexion
│       │   ├── sampling.py          # control de FPS de analisis
│       │   └── processor.py         # orquestador frame -> VisionEngine(s)
│       ├── vision/
│       │   ├── base.py              # VisionEngine (ABC), Detection
│       │   ├── face/
│       │   │   ├── base.py          # FaceEngine (ABC)
│       │   │   └── insightface_engine.py
│       │   ├── objects/             # placeholder, Fase 9
│       │   ├── ocr/                 # placeholder, Fase 9
│       │   └── tracking/            # placeholder, Fase 9
│       ├── identity/
│       │   ├── vector_index.py      # VectorIndex (ABC) + FaissVectorIndex
│       │   └── matcher.py           # IdentityMatcher
│       ├── events/
│       │   ├── models.py            # Event (pydantic)
│       │   └── engine.py            # EventEngine: dedup/cooldown/persistencia
│       ├── storage/
│       │   ├── database.py          # engine SQLAlchemy, sesiones
│       │   ├── models.py            # Camera, Person, FaceEmbedding, Event...
│       │   ├── migrations/          # SQL numerado, igual patron que server/
│       │   └── repository.py
│       ├── api/
│       │   ├── app.py               # FastAPI app factory
│       │   ├── deps.py              # auth, DB session, settings
│       │   ├── routes/
│       │   │   ├── cameras.py
│       │   │   ├── persons.py
│       │   │   ├── events.py
│       │   │   └── stream.py        # MJPEG/WS de video en vivo
│       │   └── schemas.py
│       └── web/
│           └── (frontend, Fase 5)
├── tests/
│   ├── unit/
│   └── integration/
├── docker/
│   ├── Dockerfile.cpu
│   └── Dockerfile.cuda
└── docs/
    └── ARCHITECTURE.md -> (este documento se movera aqui o se enlazara)
```

`server/` no se modifica. `platform/` es un paquete Python independiente
(`vip`), instalable por separado, que puede depender de `python-package/`
(insightface) via ruta relativa o como dependencia editable.

---

## 4. Componentes y responsabilidades

| Componente | Responsabilidad | No responsable de |
|---|---|---|
| `cameras.CameraSource` | Abrir/leer/cerrar una fuente de video, reconectar | Decidir que hacer con el frame |
| `pipeline.CameraWorker` | Hilo de vida de una camara: lectura continua, reconexion, publicar "ultimo frame" | Inferencia |
| `pipeline.processor` | Tomar el ultimo frame a la cadencia de `analysis_fps`, invocar los `VisionEngine` configurados | Persistencia, matching |
| `vision.VisionEngine` | Convertir un frame en `Detection[]` (bbox, score, embedding opcional, atributos) | Saber que camara lo genero, ni si hay match |
| `identity.IdentityMatcher` | Comparar un embedding contra el indice vectorial de personas registradas | Generar el evento final |
| `identity.VectorIndex` | Insertar/buscar vectores (FAISS en MVP) | Persistencia relacional |
| `events.EventEngine` | Decidir si una deteccion se convierte en evento (dedup, cooldown, confirmacion por N frames), persistirlo | Servir la API |
| `storage.Repository` | CRUD sobre SQLite (camaras, personas, embeddings, eventos) | Logica de negocio |
| `api` | Exponer REST (+ stream en vivo) autenticado | Logica de vision o matching |
| `web` | Mostrar video, overlays, listado de eventos, alta de personas | Cualquier logica de servidor |

---

## 5. Interfaces / clases principales

```python
# cameras/base.py
@dataclass(frozen=True)
class Frame:
    camera_id: str
    image: np.ndarray          # BGR, HxWx3
    timestamp: datetime
    sequence: int

class CameraSource(ABC):
    camera_id: str

    @abstractmethod
    def open(self) -> None: ...
    @abstractmethod
    def read(self) -> Frame | None: ...
    @abstractmethod
    def close(self) -> None: ...
    @property
    @abstractmethod
    def is_open(self) -> bool: ...


# vision/base.py
@dataclass(frozen=True)
class BoundingBox:
    x1: float; y1: float; x2: float; y2: float   # pixel-space

@dataclass(frozen=True)
class Detection:
    kind: str                    # "face" | "object" | "text" | ...
    bbox: BoundingBox
    score: float
    embedding: np.ndarray | None
    label: str | None            # p.ej. clase de objeto
    attributes: dict[str, Any]

class VisionEngine(ABC):
    name: str

    @abstractmethod
    def warmup(self) -> None: ...
    @abstractmethod
    def process(self, frame: Frame) -> list[Detection]: ...
    @abstractmethod
    def close(self) -> None: ...


# vision/face/base.py
class FaceEngine(VisionEngine):
    """process() siempre devuelve Detection(kind="face", embedding=...)."""


# identity/vector_index.py
@dataclass(frozen=True)
class VectorMatch:
    person_id: str
    similarity: float            # coseno crudo, sin redondear

class VectorIndex(ABC):
    @abstractmethod
    def add(self, person_id: str, embedding: np.ndarray) -> None: ...
    @abstractmethod
    def remove(self, person_id: str) -> None: ...
    @abstractmethod
    def search(self, embedding: np.ndarray, top_k: int = 1) -> list[VectorMatch]: ...


# identity/matcher.py
class IdentityMatcher:
    def __init__(self, index: VectorIndex, threshold: float): ...
    def match(self, embedding: np.ndarray) -> VectorMatch | None: ...


# events/models.py
class Event(BaseModel):
    id: str
    event_type: str               # "face_detected" | "person_matched" |
                                   # "unknown_person" | "camera_offline" | ...
    camera_id: str
    timestamp: datetime
    person_id: str | None
    confidence: float | None
    bounding_box: list[float] | None
    metadata: dict[str, Any] = {}
```

`event_type` es un `str` libre (no un `Enum` cerrado) desde el primer dia,
justo para que anadir `object_detected`, `vehicle_detected`,
`text_detected`, `restricted_zone`, etc. en fases futuras no requiera tocar
el esquema ni el modelo Pydantic, solo registrar el nuevo tipo en la
documentacion y, si aplica, anadir logica de negocio en el `EventEngine`.

---

## 6. Dependencias iniciales

`platform/pyproject.toml` (solo MVP, sin anadir nada que la Fase 1-6 no
necesite):

- `fastapi`, `uvicorn[standard]` - API.
- `pydantic` v2 - schemas/config.
- `sqlalchemy` - acceso a SQLite (y Postgres el dia que se migre).
- `opencv-python-headless` - decodificacion/lectura RTSP y overlays.
- `insightface` (dependencia local a `python-package/` de este mismo repo)
  - trae `onnxruntime` (CPU) como dependencia transitiva.
- `onnxruntime-gpu` - **opcional**, instalado solo en la imagen Docker CUDA
  (mismo patron que `server/requirements.cpu.lock` /
  `requirements.cuda12.lock`, ver seccion 7).
- `faiss-cpu` (MVP) / `faiss-gpu` opcional para instalaciones con CUDA.
- `numpy`.
- `python-multipart` - subida de fotos para registrar personas.
- `structlog` (o `logging` + `python-json-logger`) - logging estructurado.
- `pyyaml` o `tomli` - configuracion.
- Dev: `pytest`, `pytest-asyncio`, `httpx`, `ruff`, `mypy`.

Deliberadamente **no** se incluye todavia: FFmpeg como binding Python
dedicado (se usa via OpenCV/`CAP_FFMPEG` como hace `server/`), GStreamer
(se evalua en Fase 8 si ONVIF/RTP lo requiere), librerias de OCR/objetos
(Fase 9), ONVIF (`python-onvif-zeep`, Fase 8), ni nada de OSINT (excluido
explicitamente por el enunciado).

---

## 7. Gestion GPU / CPU

Principio: **configuracion explicita, nunca fallback silencioso** (mismo
espiritu que `server/`, con menos rigidez porque el MVP tambien debe poder
arrancar en un portatil sin GPU para desarrollo):

- `VIP_DEVICE=auto|cpu|cuda` (env var), default `auto`.
  - `auto`: si `onnxruntime.get_available_providers()` incluye
    `CUDAExecutionProvider`, se usa; si no, `CPUExecutionProvider`. Se loguea
    explicitamente que proveedor se eligio y por que.
  - `cuda`: fuerza `CUDAExecutionProvider`; si no esta disponible, el
    arranque **falla** con un mensaje claro (nunca degrada a CPU en
    silencio si el operador pidio GPU explicitamente).
  - `cpu`: fuerza CPU incluso si hay GPU (util para pruebas comparativas).
- `VIP_MAX_INFERENCE_CONCURRENCY`: cupo global de inferencia concurrente
  compartido entre todas las camaras y la API (mismo patron que
  `InferenceConcurrencyLimiter` de `server/`), para que anadir camaras no
  sature la GPU/CPU de forma descontrolada.
- FAISS: `faiss-cpu` por defecto; si `VIP_DEVICE=cuda` y `faiss-gpu` esta
  instalado, se usa un indice GPU. La interfaz `VectorIndex` oculta esta
  decision al resto del sistema.
- Imagenes Docker separadas `Dockerfile.cpu` / `Dockerfile.cuda`, replicando
  el patron ya validado en `server/docker/`: la imagen CUDA fija version
  exacta de CUDA/cuDNN y valida en el entrypoint que el runtime funciona
  antes de aceptar trafico.
- La adopcion de la auditoria estricta de kernels GPU (`audit_cuda_profile`
  en `server/`) se pospone a cuando GPU sea un requisito duro de produccion
  (no bloquea el MVP), pero queda documentada aqui como mejora conocida a
  incorporar.

---

## 8. Gestion de camaras RTSP

- `RTSPCamera(CameraSource)` encapsula `cv2.VideoCapture(url, cv2.CAP_FFMPEG)`
  con `CAP_PROP_OPEN_TIMEOUT_MSEC` / `CAP_PROP_READ_TIMEOUT_MSEC`
  configurables.
- Soporta opcionalmente una `secondary_url` (sub-stream de menor resolucion)
  para IA, dejando el stream principal solo para visualizacion cuando el
  operador la configure; si no se configura, se usa el mismo stream para
  ambos usos (igual que hace `server/` hoy).
- `pipeline.ingestion.CameraWorker` (uno por camara, hilo daemon propio):
  - Bucle de lectura continuo que publica "el frame mas reciente" protegido
    por un lock ligero (no una cola FIFO sin limite, para no acumular
    retraso si la inferencia va mas lenta que la camara).
  - Reconexion con backoff exponencial (base configurable, techo 30s, reset
    al reconectar con exito), replicando la politica ya probada de
    `server/services/rtsp.py::_reader_loop`.
  - Emite eventos `camera_online` / `camera_offline` al `EventEngine` en las
    transiciones de estado.
  - Redaccion de credenciales en cualquier URL expuesta en logs/API
    (usuario/contraseña embebidos en la URL RTSP nunca se loguean ni se
    devuelven en la API).
- `pipeline.sampling` decide, de forma independiente por camara, cada
  cuanto se envia el "frame mas reciente" al `VisionEngine` segun
  `analysis_fps` (config), sin intentar "recuperar" frames atrasados si el
  motor va lento (misma politica anti-catch-up que `server/`).
- Configuracion por camara (no solo global): `analysis_fps`,
  `display_fps`, `resolution` (si se fuerza downscale antes de inferencia),
  `match_threshold`, `enabled`.
- `USBCamera` y `VideoFileCamera` comparten la misma interfaz
  `CameraSource`, permitiendo desarrollar y testear sin hardware RTSP real.
- `ONVIFCamera` queda como modulo placeholder (Fase 8): descubrimiento,
  negociacion de perfil de stream y PTZ se añaden sin tocar el resto de la
  plataforma porque todo el resto solo conoce `CameraSource`.

---

## 9. Esquema inicial de base de datos (SQLite)

```sql
-- cameras: una fuente de video registrada
CREATE TABLE cameras (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    location        TEXT,
    source_type     TEXT NOT NULL CHECK (source_type IN ('rtsp','usb','file')),
    url_ciphertext  BLOB NOT NULL,       -- cifrado, nunca en claro
    secondary_url_ciphertext BLOB,
    enabled         INTEGER NOT NULL DEFAULT 1,
    analysis_fps    REAL NOT NULL DEFAULT 5.0,
    display_fps     REAL NOT NULL DEFAULT 15.0,
    match_threshold REAL NOT NULL DEFAULT 0.5,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

-- persons: identidad registrada voluntariamente
CREATE TABLE persons (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    external_id  TEXT UNIQUE,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

-- face_embeddings: representacion biometrica, separada de la identidad
CREATE TABLE face_embeddings (
    id           TEXT PRIMARY KEY,
    person_id    TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    embedding    BLOB NOT NULL,          -- float32[], L2-normalizado
    model_id     TEXT NOT NULL,
    model_version TEXT NOT NULL,
    quality_json TEXT NOT NULL DEFAULT '{}',
    crop_image   BLOB,                   -- opcional, imagen del rostro
    created_at   TEXT NOT NULL
);

-- events: modelo de evento generico y persistente
CREATE TABLE events (
    id             TEXT PRIMARY KEY,
    event_type     TEXT NOT NULL,        -- libre: face_detected, person_matched...
    camera_id      TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    person_id      TEXT REFERENCES persons(id) ON DELETE SET NULL,
    confidence     REAL,
    bounding_box_json TEXT,
    metadata_json  TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL
);
CREATE INDEX idx_events_camera_time ON events(camera_id, created_at);
CREATE INDEX idx_events_type_time   ON events(event_type, created_at);

-- api_keys: autenticacion de servicio (MVP: una o varias claves de servicio)
CREATE TABLE api_keys (
    id         TEXT PRIMARY KEY,
    label      TEXT NOT NULL,
    salt       BLOB NOT NULL,
    digest     BLOB NOT NULL,
    active     INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

-- audit_log: trazabilidad minima razonable para un MVP
CREATE TABLE audit_log (
    id         TEXT PRIMARY KEY,
    actor      TEXT NOT NULL,            -- api_key.id o "system"
    action     TEXT NOT NULL,            -- "camera.create", "person.enroll"...
    target     TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
```

Separacion explicita pedida por el enunciado: **identidad** (`persons`),
**imagen** (`crop_image`, opcional y borrable independientemente),
**embedding** (`face_embeddings.embedding`), **metadatos** (`metadata_json`
en cada tabla) y **eventos** (`events`) viven en tablas distintas con
borrado en cascada acotado (borrar una persona borra sus embeddings, pero
`events.person_id` se pone a `NULL`, no se borra el evento historico).

---

## 10. Flujo de datos completo

```
1. Camara IP (rtsp://...)
2. CameraWorker.read()               -> Frame (BGR + timestamp)
3. sampling: ¿toca analizar este frame segun analysis_fps? -> si/no
4. FaceEngine.process(frame)         -> Detection[] (bbox, score, embedding)
5. IdentityMatcher.match(embedding)  -> VectorMatch | None (via FAISS)
6. EventEngine.evaluate(detection, match)
     - aplica confirmacion por N frames / cooldown por persona+camara
     - si procede, construye Event
7. Repository.save(event)            -> INSERT en `events` (SQLite)
8. API expone:
     - REST: GET /cameras, /persons, /events (filtrable), POST /persons (enrolar)
     - Stream en vivo: MJPEG o WebSocket con overlay de bounding boxes
     - Server-Sent Events / WebSocket de eventos en tiempo real para el dashboard
9. Web Dashboard:
     - reproduce el stream de la camara
     - dibuja bounding box + nombre + score + timestamp sobre el video (HUD)
     - lista el feed de eventos recientes
     - permite dar de alta una persona subiendo fotos
```

Modelo de concurrencia: un hilo `CameraWorker` por camara (I/O-bound,
libera el GIL en la lectura FFmpeg); un pool acotado de hilos/`asyncio`
para la inferencia, limitado por `VIP_MAX_INFERENCE_CONCURRENCY`; la API
FastAPI corre en su propio event loop asincrono y solo lee del
`Repository`/cachés en memoria, nunca bloquea a las camaras.

---

## 11. Roadmap de implementacion

Se conserva la numeracion de fases propuesta por el usuario, con el
detalle tecnico de que implica cada una y, donde aplica, la relacion con
`server/`.

- **Fase 0 (este documento)**: arquitectura, estructura de `platform/`,
  interfaces principales. Sin codigo de aplicacion.
- **Fase 1**: `RTSPCamera` + `CameraWorker` + endpoint/pagina que muestra el
  video en vivo (sin IA todavia). Valida ingestion, reconexion y
  visualizacion antes de meter inferencia.
- **Fase 2**: `FaceEngine` (InsightFace, SCRFD) integrado en el
  `Frame Processing Pipeline`; overlay de bounding boxes en el HUD, sin
  reconocimiento (solo deteccion).
- **Fase 3**: registro de personas (`persons` + `face_embeddings`), subida
  de fotos voluntaria, calculo y almacenamiento del embedding.
- **Fase 4**: `IdentityMatcher` + `VectorIndex` (FAISS), reconocimiento
  facial local sobre el stream, `EventEngine` con confirmacion/cooldown.
- **Fase 5**: Dashboard web (HUD completo: bounding box + identidad + score
  + timestamp, tal como el ejemplo del enunciado).
- **Fase 6**: persistencia de eventos en SQLite (tabla `events`), API de
  consulta de eventos con filtros por camara/persona/rango de fechas.
- **Fase 7**: multicamara real (varias `CameraWorker` concurrentes,
  `max_cameras` configurable, cupo de inferencia compartido verificado bajo
  carga).
- **Fase 8**: ONVIF (descubrimiento, info de dispositivo, negociacion de
  stream, PTZ) como nueva implementacion de `CameraSource`, sin tocar capas
  superiores.
- **Fase 9**: `ObjectDetectionEngine`, `OCREngine`, `TrackingEngine` como
  nuevos `VisionEngine`; aqui es tambien el punto natural para evaluar si
  conviene anadir un `FaceEngine` alternativo respaldado por `server/` (via
  su API/SDK) para instalaciones que necesiten busqueda a escala de
  millones de identidades con cuantizacion INT8.
- **Fase 10**: motor de reglas y alertas (zonas, listas de vigilancia,
  combinaciones de tipos de evento) construido sobre la tabla `events` ya
  generica.
- **Fase 11**: Visual Search / OSINT como modulo independiente y opcional,
  sujeto a las restricciones legales que ya senala el enunciado; no se
  disena todavia.
- **Fase 12**: producto comercial on-premise / edge appliance;
  aqui es razonable evaluar si `platform/` y `server/` convergen en un unico
  producto empaquetado, reutilizando el trabajo ya hecho en `server/docker/`
  y `server/deploy/` como referencia de hardening de contenedores.

---

## 12. Decision que requiere validacion antes de escribir codigo

Antes de empezar la Fase 1 hace falta que el usuario confirme (o corrija)
la decision de la seccion 2.1: construir `platform/` como aplicacion nueva
que usa `insightface` + FAISS directamente para el MVP, dejando `server/`
como una posible integracion futura opcional, en lugar de acoplarse desde
ya a los internos de `server/` o de tratarlo como un microservicio del que
depende el MVP desde el primer commit.

Tambien queda pendiente de confirmar el nombre del paquete Python
(`vip` se usa aqui como marcador de posicion) y si el repositorio de
codigo debe vivir en este mismo monorepo bajo `platform/` o en un
repositorio separado.
