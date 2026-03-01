"""
Phase 3: Embedding Extraction + YAMNet Genre Classification

For every song in the training_manifest, extracts a 1024-dim YAMNet
embedding and (where genre is missing) a YAMNet genre classification.
Runs on RunPod GPU. Audio files are batch-downloaded from the seedbox
via rclone since FUSE mounts are unavailable in Docker containers.

Setup on RunPod before running:
    pip install tensorflow tensorflow-hub librosa numpy pandas tqdm mutagen
    # rclone must be configured with a [seedbox] SFTP remote

Usage:
    python phase3_extract.py [--batch-size 500] [--clip-seconds 60]

The script is fully resumable — already-processed songs (embedding IS NOT
NULL) are skipped automatically on restart.
"""

import os
import logging
import argparse
import subprocess
import shutil
import tempfile
from pathlib import Path

import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

from config import (
    DB_PATH,
    CLIP_SECONDS,
    CHECKPOINT_EVERY,
    YAMNET_GENRE_MAP,
    YAMNET_CONFIDENCE_THRESHOLD,
)
from db import init_db, get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("phase3_extract.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16000          # YAMNet requires 16kHz mono
RCLONE_REMOTE = "seedbox"    # rclone remote name
# Seedbox music root — must match MUSIC_ROOT used during Phase 1 scan
SEEDBOX_MUSIC_ROOT = "/home/hd17/bytesizor/media/Audio"


# ── YAMNet class names ─────────────────────────────────────────────────────────

def _load_yamnet_class_names() -> list[str]:
    """Download and parse YAMNet class map CSV."""
    import urllib.request
    import csv
    import io
    url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    with urllib.request.urlopen(url) as r:
        content = r.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    return [row["display_name"] for row in reader]


# ── Audio loading ──────────────────────────────────────────────────────────────

def _load_clip(file_path: str, clip_seconds: int) -> np.ndarray | None:
    """
    Load audio file at 16kHz mono and return the middle clip_seconds.
    Returns None on failure.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        total = len(y)
        clip_samples = clip_seconds * SAMPLE_RATE
        if total <= clip_samples:
            return y.astype(np.float32)
        start = (total - clip_samples) // 2
        return y[start: start + clip_samples].astype(np.float32)
    except Exception as e:
        log.debug("Audio load failed %s: %s", file_path, e)
        return None


# ── Embedding + genre extraction ───────────────────────────────────────────────

def _extract(waveform: np.ndarray, model, class_names: list[str], needs_genre: bool):
    """
    Run YAMNet on a waveform.
    Returns (embedding_bytes, yamnet_top_class, yamnet_confidence, genre_tag)
    genre_tag is None if needs_genre is False or confidence is below threshold.
    """
    scores, embeddings, _ = model(waveform)

    embedding = np.mean(embeddings.numpy(), axis=0).astype(np.float32)
    embedding_bytes = embedding.tobytes()

    yamnet_top_class = None
    yamnet_confidence = None
    genre_tag = None

    if needs_genre:
        mean_scores = scores.numpy().mean(axis=0)
        top_indices = np.argsort(mean_scores)[::-1]
        for idx in top_indices:
            name = class_names[idx]
            if name in YAMNET_GENRE_MAP:
                yamnet_top_class = name
                yamnet_confidence = float(mean_scores[idx])
                if yamnet_confidence >= YAMNET_CONFIDENCE_THRESHOLD:
                    genre_tag = YAMNET_GENRE_MAP[name]
                break

    return embedding_bytes, yamnet_top_class, yamnet_confidence, genre_tag


# ── rclone batch download ──────────────────────────────────────────────────────

def _seedbox_path_to_rclone(file_path: str) -> str:
    """Convert an absolute seedbox path to a rclone remote path."""
    rel = os.path.relpath(file_path, SEEDBOX_MUSIC_ROOT)
    return f"{RCLONE_REMOTE}:{SEEDBOX_MUSIC_ROOT}/{rel}"


def _download_batch(file_paths: list[str], tmp_dir: str) -> dict[str, str]:
    """
    Download a list of seedbox file paths into tmp_dir using rclone copy.
    Returns a dict mapping original seedbox path → local temp path.
    """
    mapping = {}

    # Group by parent directory to minimise rclone calls
    by_dir: dict[str, list[str]] = {}
    for fp in file_paths:
        parent = str(Path(fp).parent)
        by_dir.setdefault(parent, []).append(fp)

    for remote_dir, paths in by_dir.items():
        rel_dir = os.path.relpath(remote_dir, SEEDBOX_MUSIC_ROOT)
        local_dir = os.path.join(tmp_dir, rel_dir)
        os.makedirs(local_dir, exist_ok=True)

        filenames = [Path(p).name for p in paths]
        # Write a filter file so rclone only downloads the files we need
        filter_file = os.path.join(tmp_dir, "_filter.txt")
        with open(filter_file, "w") as f:
            for name in filenames:
                f.write(f"+ {name}\n")
            f.write("- *\n")

        cmd = [
            "rclone", "copy",
            f"{RCLONE_REMOTE}:{remote_dir}",
            local_dir,
            "--filter-from", filter_file,
            "--transfers", "8",
            "--quiet",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.warning("rclone error for %s: %s", remote_dir, result.stderr[:200])

        for fp in paths:
            local_path = os.path.join(local_dir, Path(fp).name)
            if os.path.exists(local_path):
                mapping[fp] = local_path

    return mapping


# ── Main extraction loop ───────────────────────────────────────────────────────

def run_extraction(
    db_path: str = DB_PATH,
    batch_size: int = 500,
    clip_seconds: int = CLIP_SECONDS,
):
    init_db(db_path)
    conn = get_connection(db_path)

    log.info("Loading YAMNet from TF Hub …")
    model = hub.load(YAMNET_URL)
    class_names = _load_yamnet_class_names()
    log.info("YAMNet loaded. %d classes.", len(class_names))

    # Songs to process: in training_manifest, embedding not yet extracted
    rows = conn.execute(
        """
        SELECT s.song_id, s.file_path, s.genre_source
        FROM training_manifest tm
        JOIN songs s ON s.song_id = tm.song_id
        WHERE s.embedding IS NULL
        ORDER BY s.song_id
        """
    ).fetchall()

    total = len(rows)
    log.info("%d songs to process (embedding missing)", total)
    if total == 0:
        log.info("Nothing to do — all embeddings already extracted.")
        conn.close()
        return

    processed = 0
    errors = 0

    for batch_start in range(0, total, batch_size):
        batch_rows = rows[batch_start: batch_start + batch_size]
        file_paths = [r["file_path"] for r in batch_rows]

        log.info(
            "Batch %d–%d / %d — downloading %d files …",
            batch_start + 1, min(batch_start + batch_size, total), total, len(batch_rows),
        )

        with tempfile.TemporaryDirectory(prefix="playlistai_") as tmp_dir:
            local_map = _download_batch(file_paths, tmp_dir)
            log.info("  Downloaded %d / %d files", len(local_map), len(batch_rows))

            updates = []

            for row in batch_rows:
                local_path = local_map.get(row["file_path"])
                if local_path is None:
                    log.warning("  File not downloaded: %s", row["file_path"])
                    errors += 1
                    continue

                waveform = _load_clip(local_path, clip_seconds)
                if waveform is None:
                    log.warning("  Audio load failed: %s", row["file_path"])
                    errors += 1
                    continue

                needs_genre = row["genre_source"] is None or row["genre_source"] == "unknown"

                try:
                    emb_bytes, top_class, confidence, genre_tag = _extract(
                        waveform, model, class_names, needs_genre
                    )
                except Exception as e:
                    log.warning("  YAMNet error %s: %s", row["file_path"], e)
                    errors += 1
                    continue

                updates.append((emb_bytes, top_class, confidence, genre_tag, row["song_id"]))
                processed += 1

            # Batch write to DB
            if updates:
                conn.executemany(
                    """
                    UPDATE songs SET
                        embedding         = ?,
                        yamnet_top_class  = ?,
                        yamnet_confidence = ?,
                        genre_tag         = CASE
                            WHEN ? IS NOT NULL AND genre_source IS NULL THEN ?
                            ELSE genre_tag
                        END,
                        genre_source      = CASE
                            WHEN ? IS NOT NULL AND genre_source IS NULL THEN 'yamnet'
                            ELSE genre_source
                        END,
                        genre_confidence  = CASE
                            WHEN ? IS NOT NULL AND genre_source IS NULL THEN ?
                            ELSE genre_confidence
                        END
                    WHERE song_id = ?
                    """,
                    [
                        (e, t, c, g, g, g, g, c, sid)
                        for e, t, c, g, sid in updates
                    ],
                )
                conn.commit()

        log.info(
            "Progress: %d / %d processed, %d errors",
            processed, total, errors,
        )

    # Final summary
    total_embedded = conn.execute(
        "SELECT COUNT(*) FROM songs WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    yamnet_genre = conn.execute(
        "SELECT COUNT(*) FROM songs WHERE genre_source = 'yamnet'"
    ).fetchone()[0]

    print("\n" + "=" * 60)
    print("PLAYLISTAI — PHASE 3 EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Processed this run :  {processed:>8,}")
    print(f"  Errors             :  {errors:>8,}")
    print(f"  Total embedded     :  {total_embedded:>8,}")
    print(f"  YAMNet genre fills :  {yamnet_genre:>8,}")
    print("=" * 60)

    log.info("Phase 3 complete.")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlaylistAI Phase 3 embedding extraction")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Songs per rclone download batch (default: %(default)s)")
    parser.add_argument("--clip-seconds", type=int, default=CLIP_SECONDS,
                        help="Middle N seconds of audio to use (default: %(default)s)")
    args = parser.parse_args()

    run_extraction(batch_size=args.batch_size, clip_seconds=args.clip_seconds)
