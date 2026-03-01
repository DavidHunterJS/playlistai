"""
Database initialization and connection management.
Creates all tables defined in the PlaylistAI schema.
"""

import sqlite3
from contextlib import contextmanager
from config import DB_PATH

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS songs (
    song_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path           TEXT    UNIQUE NOT NULL,
    folder_date         TEXT,                       -- "YYYY-WW" or "YYYY-MM-DD" from folder name
    genre_tag           TEXT,                       -- resolved genre (any source)
    genre_source        TEXT,                       -- "id3"|"folder"|"yamnet"|"similarity"|"unknown"
    genre_confidence    REAL,
    artist              TEXT,
    title               TEXT,
    artist_source       TEXT,                       -- "id3"|"filename"
    title_source        TEXT,                       -- "id3"|"filename"
    duration_seconds    REAL,
    metadata_quality    TEXT,                       -- "full"|"partial"|"none"
    is_explored         BOOLEAN NOT NULL DEFAULT 0,
    embedding           BLOB,                       -- 1024-dim float32 array
    yamnet_top_class    TEXT,
    yamnet_confidence   REAL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_songs_genre_source  ON songs(genre_source);
CREATE INDEX IF NOT EXISTS idx_songs_is_explored   ON songs(is_explored);
CREATE INDEX IF NOT EXISTS idx_songs_genre_tag     ON songs(genre_tag);

CREATE TABLE IF NOT EXISTS playlists (
    playlist_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    source_file TEXT,
    song_count  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS playlist_songs (
    playlist_id INTEGER NOT NULL REFERENCES playlists(playlist_id),
    song_id     INTEGER NOT NULL REFERENCES songs(song_id),
    PRIMARY KEY (playlist_id, song_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    song_id       INTEGER NOT NULL REFERENCES songs(song_id),
    score         REAL    NOT NULL,
    model_version TEXT    NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (song_id, model_version)
);

CREATE TABLE IF NOT EXISTS feedback (
    song_id  INTEGER NOT NULL REFERENCES songs(song_id),
    rating   INTEGER NOT NULL CHECK(rating IN (0, 1)),
    rated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (song_id)
);

CREATE TABLE IF NOT EXISTS genre_centroids (
    genre       TEXT PRIMARY KEY,
    centroid    BLOB NOT NULL,   -- mean embedding, float32 array
    song_count  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS training_manifest (
    song_id     INTEGER NOT NULL REFERENCES songs(song_id),
    label       INTEGER NOT NULL CHECK(label IN (0, 1)),
    split       TEXT    NOT NULL CHECK(split IN ('train', 'val')),
    PRIMARY KEY (song_id)
);
"""


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db(db_path: str = DB_PATH):
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    print(f"Database initialized: {db_path}")


if __name__ == "__main__":
    init_db()
