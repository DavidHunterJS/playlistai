"""
Phase 1, Step 1.5: Parse all M3U/M3U8 playlist files and populate
the playlists and playlist_songs tables.

Run after phase1_scan.py has populated the songs table.

    python phase1_playlists.py --playlist-dir /path/to/playlists
"""

import os
import re
import logging
from pathlib import Path

from tqdm import tqdm

from config import DB_PATH, PLAYLIST_DIR, MUSIC_ROOT
from db import init_db, get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("phase1_playlists.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

PLAYLIST_EXTENSIONS = {".m3u", ".m3u8"}


def _find_playlists(playlist_dir: str) -> list[str]:
    found = []
    for dirpath, _dirs, files in os.walk(playlist_dir, followlinks=True):
        for fname in files:
            if Path(fname).suffix.lower() in PLAYLIST_EXTENSIONS:
                found.append(os.path.join(dirpath, fname))
    return found


def _parse_m3u(m3u_path: str, music_root: str) -> list[str]:
    """
    Parse an M3U file and return a list of path strings.
    Absolute paths are returned as-is.
    Relative paths are returned as-is (resolution against music_root
    sub-directories is handled at lookup time in run_playlist_import).
    Handles both UTF-8 and Latin-1 encodings.
    """
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with open(m3u_path, encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        log.warning("Could not decode %s — skipping", m3u_path)
        return []

    paths = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Normalize Windows path separators
        paths.append(line.replace("\\", "/"))

    return paths


def _playlist_name(m3u_path: str) -> str:
    """Derive a human-readable playlist name from the M3U filename."""
    stem = Path(m3u_path).stem
    # Strip leading numbers/underscores
    name = re.sub(r"^[\d_\-\.\s]+", "", stem).strip()
    return name if name else stem


def run_playlist_import(
    playlist_dir: str = PLAYLIST_DIR,
    music_root: str = MUSIC_ROOT,
):
    init_db(DB_PATH)
    conn = get_connection(DB_PATH)

    m3u_files = _find_playlists(playlist_dir)
    if not m3u_files:
        log.warning("No M3U files found in %s", playlist_dir)
        return

    log.info("Found %d M3U files", len(m3u_files))

    # Build a lookup: file_path → song_id  (for fast matching)
    log.info("Loading song path index from DB …")
    rows = conn.execute("SELECT song_id, file_path FROM songs").fetchall()
    path_to_id: dict[str, int] = {r["file_path"]: r["song_id"] for r in rows}
    log.info("Indexed %d songs", len(path_to_id))

    total_linked = 0
    total_missing = 0

    for m3u_path in tqdm(m3u_files, desc="Parsing playlists", unit="playlist"):
        name = _playlist_name(m3u_path)
        song_paths = _parse_m3u(m3u_path, music_root)

        if not song_paths:
            log.warning("Empty or unreadable playlist: %s", m3u_path)
            continue

        # Insert playlist record
        cur = conn.execute(
            "INSERT OR IGNORE INTO playlists (name, source_file, song_count) VALUES (?, ?, ?)",
            (name, m3u_path, len(song_paths)),
        )
        conn.commit()

        # Fetch the playlist_id (handles both fresh insert and pre-existing row)
        playlist_id = conn.execute(
            "SELECT playlist_id FROM playlists WHERE source_file = ?", (m3u_path,)
        ).fetchone()["playlist_id"]

        # Link songs
        linked = 0
        missing_paths = []
        insert_pairs = []

        for sp in song_paths:
            # Try the path as-is (absolute), then prepend known sub-roots
            candidates = [
                sp,
                f"{music_root}/{sp}",
                f"{music_root}/mp3/{sp}",
                f"{music_root}/soundcheck/{sp}",
            ]
            sid = next((path_to_id[c] for c in candidates if c in path_to_id), None)
            if sid is None:
                missing_paths.append(sp)
                continue
            insert_pairs.append((playlist_id, sid))
            linked += 1

        if insert_pairs:
            conn.executemany(
                "INSERT OR IGNORE INTO playlist_songs (playlist_id, song_id) VALUES (?, ?)",
                insert_pairs,
            )
            conn.commit()

        # Update song_count to actual linked count
        conn.execute(
            "UPDATE playlists SET song_count = ? WHERE playlist_id = ?",
            (linked, playlist_id),
        )
        conn.commit()

        total_linked += linked
        total_missing += len(missing_paths)

        if missing_paths:
            log.warning(
                "Playlist '%s': %d entries not found in DB (moved/renamed?)",
                name,
                len(missing_paths),
            )
            for mp in missing_paths[:5]:
                log.debug("  Missing: %s", mp)
            if len(missing_paths) > 5:
                log.debug("  … and %d more", len(missing_paths) - 5)

        log.info("Playlist '%s': %d/%d songs linked", name, linked, len(song_paths))

    # Summary
    distinct = conn.execute(
        "SELECT COUNT(DISTINCT song_id) FROM playlist_songs"
    ).fetchone()[0]
    log.info(
        "Step 1.5 complete — %d distinct playlisted songs, %d unresolved paths",
        distinct,
        total_missing,
    )
    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PlaylistAI Phase 1 playlist import")
    parser.add_argument("--playlist-dir", default=PLAYLIST_DIR)
    parser.add_argument("--music-root", default=MUSIC_ROOT)
    args = parser.parse_args()

    run_playlist_import(args.playlist_dir, args.music_root)
