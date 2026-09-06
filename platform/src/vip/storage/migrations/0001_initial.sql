-- Fase 3: identidades registradas voluntariamente y sus representaciones
-- biometricas.
--
-- Identidad (persons), embedding (face_embeddings.embedding), imagen
-- (face_embeddings.crop_image) y metadatos (metadata_json) viven separados
-- a proposito: borrar la imagen no obliga a borrar la identidad, y el
-- embedding puede regenerarse sin tocar el resto. Los eventos llegan en la
-- Fase 6 con su propia migracion.

CREATE TABLE persons (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    external_id   TEXT UNIQUE,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE face_embeddings (
    id                  TEXT PRIMARY KEY,
    person_id           TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    embedding           BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    -- Un embedding solo es comparable con otros del mismo modelo: se guarda
    -- con que modelo se genero para poder rechazar comparaciones invalidas.
    model_id            TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    detection_score     REAL NOT NULL,
    bounding_box_json   TEXT NOT NULL,
    crop_image          BLOB,
    crop_media_type     TEXT,
    created_at          TEXT NOT NULL
);

CREATE INDEX idx_face_embeddings_person ON face_embeddings(person_id);
