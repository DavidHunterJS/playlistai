"""
Phase 1, Steps 1.2–1.4: File system scan with ID3 extraction, folder-name
genre parsing, and filename metadata extraction.

Run from the seedbox (or wherever the music collection is mounted):
    python phase1_scan.py

Environment variables:
    PLAYLISTAI_MUSIC_ROOT   — root of the music collection
    PLAYLISTAI_DB_PATH      — path to the SQLite database
    PLAYLISTAI_EXPLORED_CUTOFF — date cutoff for is_explored flag
"""

import os
import re
import sqlite3
import logging
from datetime import date, datetime
from pathlib import Path

import mutagen
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.oggvorbis import OggVorbis
from mutagen._util import MutagenError
from tqdm import tqdm

from config import (
    MUSIC_ROOT,
    DB_PATH,
    AUDIO_EXTENSIONS,
    EXPLORED_CUTOFF,
    EXPLORED_FOLDER_NAMES,
    FOLDER_GENRE_MAP,
)
from db import init_db, get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("phase1_scan.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Date parsing ───────────────────────────────────────────────────────────────

# Patterns for folder dates in the path segments.
# Covers: "2019", "2019-12", "2019-12-25", "2019_W34", "week34_2019",
#         "2019-W34", "Week 34 2019", etc.
_DATE_PATTERNS = [
    # ISO week:  2019-W34 or 2019_W34 or 2019W34
    (re.compile(r"\b(\d{4})[-_]?W(\d{1,2})\b", re.I), "week"),
    # Full date: 2019-12-25 or 2019.12.25
    (re.compile(r"\b(\d{4})[-./](\d{1,2})[-./](\d{1,2})\b"), "full"),
    # Year-month: 2019-12 or 2019.12
    (re.compile(r"\b(\d{4})[-./](\d{1,2})\b"), "yearmonth"),
    # Year only: 2019
    (re.compile(r"\b((?:19|20)\d{2})\b"), "year"),
]


def _parse_folder_date(path: str) -> str | None:
    """Extract the most specific date string from any segment of the path."""
    for segment in reversed(Path(path).parts):
        for pattern, kind in _DATE_PATTERNS:
            m = pattern.search(segment)
            if not m:
                continue
            if kind == "week":
                return f"{m.group(1)}-W{int(m.group(2)):02d}"
            if kind == "full":
                return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            if kind == "yearmonth":
                return f"{m.group(1)}-{int(m.group(2)):02d}"
            if kind == "year":
                return m.group(1)
    return None


def _is_explored(file_path: str, folder_date: str | None, cutoff: str) -> bool:
    """Return True if the song is in a known-explored folder or predates the cutoff."""
    # Any path segment matching an explored folder name wins unconditionally
    parts = {p.lower() for p in Path(file_path).parts}
    if parts & {name.lower() for name in EXPLORED_FOLDER_NAMES}:
        return True
    # Fall back to date comparison
    if folder_date is None:
        return False
    try:
        return folder_date < cutoff
    except TypeError:
        return False


# ── ID3 extraction ─────────────────────────────────────────────────────────────

def _load_tags(file_path: str) -> dict:
    """
    Attempt to read ID3/audio tags. Returns a dict with keys:
      genre, artist, title, duration_seconds
    All values default to None on failure.
    """
    result = {"genre": None, "artist": None, "title": None, "duration_seconds": None}
    ext = Path(file_path).suffix.lower()

    try:
        if ext == ".mp3":
            audio = MP3(file_path)
            result["duration_seconds"] = audio.info.length
            try:
                tags = EasyID3(file_path)
                result["genre"] = tags.get("genre", [None])[0]
                result["artist"] = tags.get("artist", [None])[0]
                result["title"] = tags.get("title", [None])[0]
            except Exception:
                pass

        elif ext == ".flac":
            audio = FLAC(file_path)
            result["duration_seconds"] = audio.info.length
            result["genre"] = (audio.get("genre") or [None])[0]
            result["artist"] = (audio.get("artist") or [None])[0]
            result["title"] = (audio.get("title") or [None])[0]

        elif ext in (".m4a", ".aac"):
            audio = MP4(file_path)
            result["duration_seconds"] = audio.info.length
            result["genre"] = str(audio.tags.get("\xa9gen", [None])[0]) if audio.tags else None
            result["artist"] = str(audio.tags.get("\xa9ART", [None])[0]) if audio.tags else None
            result["title"] = str(audio.tags.get("\xa9nam", [None])[0]) if audio.tags else None

        elif ext == ".ogg":
            audio = OggVorbis(file_path)
            result["duration_seconds"] = audio.info.length
            result["genre"] = (audio.get("genre") or [None])[0]
            result["artist"] = (audio.get("artist") or [None])[0]
            result["title"] = (audio.get("title") or [None])[0]

        else:
            # Generic fallback via mutagen
            audio = mutagen.File(file_path)
            if audio is not None:
                result["duration_seconds"] = getattr(audio.info, "length", None)

    except MutagenError as e:
        log.debug("mutagen error %s: %s", file_path, e)
    except Exception as e:
        log.debug("tag read error %s: %s", file_path, e)

    # Clean up empty strings
    for k in ("genre", "artist", "title"):
        if result[k] is not None and result[k].strip() == "":
            result[k] = None

    return result


# ── Folder genre parsing ───────────────────────────────────────────────────────

# Pre-sort by length descending so longer matches win (e.g. "deep house" > "house")
_FOLDER_KEYWORDS = sorted(FOLDER_GENRE_MAP.keys(), key=len, reverse=True)


def _folder_genre(file_path: str) -> str | None:
    """
    Walk every path segment above the filename and look for genre keywords.
    Returns the first (longest) match, or None.
    """
    parts = [p.lower() for p in Path(file_path).parts[:-1]]
    combined = " / ".join(parts)
    for keyword in _FOLDER_KEYWORDS:
        if keyword in combined:
            return FOLDER_GENRE_MAP[keyword]
    return None


# ── Filename metadata extraction ───────────────────────────────────────────────

_FILENAME_PATTERNS = [
    # 01 - Artist - Title
    re.compile(r"^\d+\s*[-–]\s*(.+?)\s*[-–]\s*(.+)$"),
    # Artist - Title
    re.compile(r"^(.+?)\s*[-–]\s*(.+)$"),
    # Artist_-_Title
    re.compile(r"^(.+?)_-_(.+)$"),
]


def _filename_meta(file_path: str) -> dict:
    """
    Attempt to extract artist and title from the filename stem.
    Also tries using the parent folder name as artist fallback.
    """
    stem = Path(file_path).stem
    # Replace underscores with spaces for matching
    stem_clean = stem.replace("_", " ").strip()

    for pat in _FILENAME_PATTERNS:
        m = pat.match(stem_clean)
        if m:
            groups = m.groups()
            if len(groups) == 2:
                # Disambiguate: track-num pattern has 3 groups; 2-group is artist/title
                return {"artist": groups[0].strip(), "title": groups[1].strip()}
            if len(groups) == 3:
                return {"artist": groups[1].strip(), "title": groups[2].strip()}

    # Single field — use filename as title, parent folder as artist candidate
    parent = Path(file_path).parent.name
    return {"artist": parent if parent else None, "title": stem_clean}


# ── Metadata quality assessment ────────────────────────────────────────────────

def _quality(genre, artist, title) -> str:
    filled = sum(x is not None for x in (genre, artist, title))
    if filled == 3:
        return "full"
    if filled >= 1:
        return "partial"
    return "none"


# ── File walker ────────────────────────────────────────────────────────────────

def _iter_audio_files(root: str):
    """Yield absolute paths of all audio files under root."""
    for dirpath, _dirs, files in os.walk(root, followlinks=True):
        for fname in files:
            if Path(fname).suffix.lower() in AUDIO_EXTENSIONS:
                yield os.path.join(dirpath, fname)


# ── Batch insert helpers ───────────────────────────────────────────────────────

_INSERT_SONG = """
INSERT OR IGNORE INTO songs
    (file_path, folder_date, genre_tag, genre_source, genre_confidence,
     artist, title, artist_source, title_source,
     duration_seconds, metadata_quality, is_explored)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_UPDATE_FOLDER_GENRE = """
UPDATE songs
SET genre_tag = ?, genre_source = 'folder', genre_confidence = 1.0
WHERE song_id = ? AND genre_source IS NULL
"""

_UPDATE_FILENAME_META = """
UPDATE songs
SET artist = ?, title = ?,
    artist_source = CASE WHEN artist IS NULL THEN 'filename' ELSE artist_source END,
    title_source  = CASE WHEN title  IS NULL THEN 'filename' ELSE title_source  END
WHERE song_id = ?
  AND (artist IS NULL OR title IS NULL)
"""

BATCH_SIZE = 500


# ── Main scan ──────────────────────────────────────────────────────────────────

def run_scan(music_root: str = MUSIC_ROOT, cutoff: str = EXPLORED_CUTOFF):
    init_db(DB_PATH)
    conn = get_connection(DB_PATH)

    log.info("Step 1.2 — Scanning %s for audio files …", music_root)

    # Collect all paths first so tqdm can show a total
    log.info("Counting files …")
    all_files = list(_iter_audio_files(music_root))
    log.info("Found %d audio files", len(all_files))

    errors = 0
    batch = []

    for file_path in tqdm(all_files, desc="Step 1.2 ID3 scan", unit="song"):
        try:
            tags = _load_tags(file_path)
        except Exception as e:
            log.warning("Unhandled error %s: %s", file_path, e)
            tags = {"genre": None, "artist": None, "title": None, "duration_seconds": None}
            errors += 1

        folder_date = _parse_folder_date(file_path)
        explored = _is_explored(file_path, folder_date, cutoff)

        genre = tags["genre"]
        artist = tags["artist"]
        title = tags["title"]

        genre_source = "id3" if genre else None
        genre_confidence = 1.0 if genre else None
        artist_source = "id3" if artist else None
        title_source = "id3" if title else None
        quality = _quality(genre, artist, title)

        batch.append((
            file_path, folder_date, genre, genre_source, genre_confidence,
            artist, title, artist_source, title_source,
            tags["duration_seconds"], quality, int(explored),
        ))

        if len(batch) >= BATCH_SIZE:
            conn.executemany(_INSERT_SONG, batch)
            conn.commit()
            batch.clear()

    if batch:
        conn.executemany(_INSERT_SONG, batch)
        conn.commit()
        batch.clear()

    total = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    log.info("Step 1.2 complete — %d songs in DB, %d errors", total, errors)

    # ── Step 1.3: Folder name genre extraction ─────────────────────────────────
    log.info("Step 1.3 — Folder name genre extraction …")
    no_genre = conn.execute(
        "SELECT song_id, file_path FROM songs WHERE genre_source IS NULL"
    ).fetchall()
    log.info("%d songs lack genre — running folder parser", len(no_genre))

    folder_hits = 0
    batch_folder = []
    for row in tqdm(no_genre, desc="Step 1.3 folder genre", unit="song"):
        genre = _folder_genre(row["file_path"])
        if genre:
            batch_folder.append((genre, row["song_id"]))
            folder_hits += 1
        if len(batch_folder) >= BATCH_SIZE:
            conn.executemany(_UPDATE_FOLDER_GENRE, batch_folder)
            conn.commit()
            batch_folder.clear()

    if batch_folder:
        conn.executemany(_UPDATE_FOLDER_GENRE, batch_folder)
        conn.commit()

    log.info("Step 1.3 complete — folder genre assigned to %d songs", folder_hits)

    # ── Step 1.4: Filename metadata extraction ─────────────────────────────────
    log.info("Step 1.4 — Filename metadata extraction …")
    missing_meta = conn.execute(
        "SELECT song_id, file_path FROM songs WHERE artist IS NULL OR title IS NULL"
    ).fetchall()
    log.info("%d songs lack artist/title — running filename parser", len(missing_meta))

    batch_fn = []
    for row in tqdm(missing_meta, desc="Step 1.4 filename meta", unit="song"):
        meta = _filename_meta(row["file_path"])
        batch_fn.append((meta["artist"], meta["title"], row["song_id"]))
        if len(batch_fn) >= BATCH_SIZE:
            conn.executemany(_UPDATE_FILENAME_META, batch_fn)
            conn.commit()
            batch_fn.clear()

    if batch_fn:
        conn.executemany(_UPDATE_FILENAME_META, batch_fn)
        conn.commit()

    log.info("Step 1.4 complete")
    conn.close()
    log.info("Phase 1 scan finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PlaylistAI Phase 1 file scan")
    parser.add_argument("--music-root", default=MUSIC_ROOT, help="Root music directory")
    parser.add_argument("--cutoff", default=EXPLORED_CUTOFF,
                        help="Explored cutoff date (YYYY-MM-DD or YYYY-WW)")
    args = parser.parse_args()

    run_scan(args.music_root, args.cutoff)
