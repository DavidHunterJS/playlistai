"""
Phase 5: Discovery Pipeline — Embedding Extraction → Genre Fallback → Inference → Playlists

Runs four steps in sequence on RunPod after phase4_train.py completes:

  Step 1  Embedding extraction   — batch-rclone unexplored songs → YAMNet embeddings + genre
  Step 2  Layer-4 genre fallback — cosine similarity to centroids for still-untagged songs
  Step 3  Inference              — score all unexplored songs, write to predictions table
  Step 4  Playlist generation    — write M3U files ranked by model score

Step 1 takes ~70–100 hrs for 124K songs and is fully resumable (songs with existing
embeddings are skipped). Steps 2–4 run in minutes and are always re-run unless suppressed.

Setup (same as Phase 3 — already done on RunPod):
    pip install tensorflow tensorflow-hub librosa numpy tqdm
    # rclone must be configured with a [seedbox] SFTP remote

Usage:
    python phase5_inference.py                          # full run
    python phase5_inference.py --skip-extraction        # skip step 1 (use existing embeddings)
    python phase5_inference.py --skip-extraction \\
                               --skip-inference         # regenerate playlists only
"""

import os
import csv
import io
import logging
import argparse
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from collections import defaultdict

import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

from config import (
    DB_PATH,
    CLIP_SECONDS,
    YAMNET_GENRE_MAP,
    YAMNET_CONFIDENCE_THRESHOLD,
    SIMILARITY_MIN,
    SIMILARITY_MARGIN,
)
from db import init_db, get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("phase5_inference.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

YAMNET_URL        = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE       = 16000
RCLONE_REMOTE     = "seedbox"
SEEDBOX_MUSIC_ROOT = "/home/hd17/bytesizor/media/Audio"

# Playlist definitions: (display_name, top_n, genre_filter, score_min, score_max)
# genre_filter=None → all genres; genre_filter="__ungenred__" → NULL genre_tag only
PLAYLIST_DEFS = [
    ("Discover Master",         200, None,             0.0,  1.0),
    ("Discover House",           50, "House",           0.0,  1.0),
    ("Discover Rap",             50, "Rap",             0.0,  1.0),
    ("Discover Rock-Metal",      50, "Rock/Metal",      0.0,  1.0),
    ("Discover Pop",             50, "Pop",             0.0,  1.0),
    ("Discover EDM-Electronic",  50, "EDM/Electronic",  0.0,  1.0),
    ("Discover Mystery",         50, "__ungenred__",    0.0,  1.0),
    ("Discover Borderline",      50, None,              0.45, 0.55),
]


# ── Shared audio helpers (mirror of phase3_extract.py) ─────────────────────────

def _load_yamnet_class_names() -> list[str]:
    url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    with urllib.request.urlopen(url) as r:
        content = r.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    return [row["display_name"] for row in reader]


def _load_clip(file_path: str, clip_seconds: int) -> np.ndarray | None:
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


def _extract(waveform: np.ndarray, yamnet_model, class_names: list[str], needs_genre: bool):
    scores, embeddings, _ = yamnet_model(waveform)
    embedding = np.mean(embeddings.numpy(), axis=0).astype(np.float32)
    embedding_bytes = embedding.tobytes()

    yamnet_top_class = yamnet_confidence = genre_tag = None
    if needs_genre:
        mean_scores = scores.numpy().mean(axis=0)
        for idx in np.argsort(mean_scores)[::-1]:
            name = class_names[idx]
            if name in YAMNET_GENRE_MAP:
                yamnet_top_class = name
                yamnet_confidence = float(mean_scores[idx])
                if yamnet_confidence >= YAMNET_CONFIDENCE_THRESHOLD:
                    genre_tag = YAMNET_GENRE_MAP[name]
                break

    return embedding_bytes, yamnet_top_class, yamnet_confidence, genre_tag


def _download_batch(file_paths: list[str], tmp_dir: str) -> dict[str, str]:
    mapping = {}
    by_dir: dict[str, list[str]] = {}
    for fp in file_paths:
        by_dir.setdefault(str(Path(fp).parent), []).append(fp)

    for remote_dir, paths in by_dir.items():
        rel_dir = os.path.relpath(remote_dir, SEEDBOX_MUSIC_ROOT)
        local_dir = os.path.join(tmp_dir, rel_dir)
        os.makedirs(local_dir, exist_ok=True)

        filter_file = os.path.join(tmp_dir, "_filter.txt")
        with open(filter_file, "w") as f:
            for name in [Path(p).name for p in paths]:
                f.write(f"+ {name}\n")
            f.write("- *\n")

        result = subprocess.run(
            ["rclone", "copy", f"{RCLONE_REMOTE}:{remote_dir}", local_dir,
             "--filter-from", filter_file, "--transfers", "8", "--quiet"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.warning("rclone error for %s: %s", remote_dir, result.stderr[:200])

        for fp in paths:
            local_path = os.path.join(local_dir, Path(fp).name)
            if os.path.exists(local_path):
                mapping[fp] = local_path

    return mapping


# ── Step 1: Embedding extraction ───────────────────────────────────────────────

def _step_extract(conn, batch_size: int, clip_seconds: int):
    """Extract YAMNet embeddings (+ genre) for all unexplored songs lacking embeddings."""
    rows = conn.execute(
        """
        SELECT song_id, file_path, genre_source
        FROM songs
        WHERE is_explored = 0
          AND embedding IS NULL
        ORDER BY song_id
        """
    ).fetchall()

    total = len(rows)
    log.info("Step 1 — %d unexplored songs need embeddings", total)
    if total == 0:
        log.info("Step 1 — nothing to do, all unexplored songs have embeddings.")
        return

    log.info("Loading YAMNet …")
    yamnet_model = hub.load(YAMNET_URL)
    class_names  = _load_yamnet_class_names()
    log.info("YAMNet loaded (%d classes).", len(class_names))

    processed = errors = 0

    for batch_start in range(0, total, batch_size):
        batch_rows  = rows[batch_start: batch_start + batch_size]
        file_paths  = [r["file_path"] for r in batch_rows]

        log.info(
            "Step 1 — batch %d–%d / %d  downloading %d files …",
            batch_start + 1, min(batch_start + batch_size, total), total, len(batch_rows),
        )

        with tempfile.TemporaryDirectory(prefix="playlistai_") as tmp_dir:
            local_map = _download_batch(file_paths, tmp_dir)
            log.info("  Downloaded %d / %d", len(local_map), len(batch_rows))

            updates = []
            for row in batch_rows:
                local_path = local_map.get(row["file_path"])
                if local_path is None:
                    log.warning("  Not downloaded: %s", row["file_path"])
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
                        waveform, yamnet_model, class_names, needs_genre
                    )
                except Exception as e:
                    log.warning("  YAMNet error %s: %s", row["file_path"], e)
                    errors += 1
                    continue

                updates.append((emb_bytes, top_class, confidence, genre_tag, row["song_id"]))
                processed += 1

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
                    [(e, t, c, g, g, g, g, c, sid) for e, t, c, g, sid in updates],
                )
                conn.commit()

        log.info("Step 1 — progress: %d / %d processed, %d errors", processed, total, errors)

    log.info("Step 1 complete — %d embeddings extracted, %d errors", processed, errors)


# ── Step 2: Layer-4 genre fallback (embedding similarity) ──────────────────────

def _step_genre_fallback(conn) -> int:
    """
    For unexplored songs with embeddings but no genre_tag, compute cosine similarity
    to per-genre centroids and assign genre if similarity is high and unambiguous.
    """
    centroid_rows = conn.execute(
        "SELECT genre, centroid FROM genre_centroids"
    ).fetchall()

    if not centroid_rows:
        log.warning("Step 2 — no genre centroids in DB, skipping Layer-4 fallback.")
        return 0

    genres = [r["genre"] for r in centroid_rows]
    raw_centroids = np.stack(
        [np.frombuffer(r["centroid"], dtype=np.float32) for r in centroid_rows]
    )
    # L2-normalise centroids for cosine similarity
    norms = np.linalg.norm(raw_centroids, axis=1, keepdims=True) + 1e-8
    centroid_matrix = raw_centroids / norms   # (n_genres, 1024)

    rows = conn.execute(
        """
        SELECT song_id, embedding
        FROM songs
        WHERE is_explored = 0
          AND embedding IS NOT NULL
          AND genre_source IS NULL
        ORDER BY song_id
        """
    ).fetchall()

    log.info("Step 2 — %d unexplored songs need genre fallback", len(rows))
    if not rows:
        return 0

    updates = []
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        norm = np.linalg.norm(emb) + 1e-8
        emb_n = emb / norm

        sims = centroid_matrix @ emb_n   # (n_genres,)
        sorted_idx = np.argsort(sims)[::-1]
        best_sim   = float(sims[sorted_idx[0]])
        second_sim = float(sims[sorted_idx[1]]) if len(sims) > 1 else 0.0

        if best_sim >= SIMILARITY_MIN and (best_sim - second_sim) >= SIMILARITY_MARGIN:
            updates.append((genres[sorted_idx[0]], best_sim, row["song_id"]))

    if updates:
        conn.executemany(
            """
            UPDATE songs
            SET genre_tag = ?, genre_source = 'similarity', genre_confidence = ?
            WHERE song_id = ? AND genre_source IS NULL
            """,
            updates,
        )
        conn.commit()

    log.info("Step 2 complete — genre assigned to %d songs via similarity", len(updates))
    return len(updates)


# ── Step 3: Inference ───────────────────────────────────────────────────────────

def _step_inference(conn, model_dir: str, infer_batch: int = 4096) -> str:
    """
    Load trained model + norm stats, score all unexplored songs with embeddings,
    write results to predictions table.  Returns the model_version string.
    """
    version_path = os.path.join(model_dir, "model_version.txt")
    if not os.path.exists(version_path):
        raise FileNotFoundError(
            f"model_version.txt not found in {model_dir}. Run phase4_train.py first."
        )
    model_version = Path(version_path).read_text().strip()
    log.info("Step 3 — model version: %s", model_version)

    model_path = os.path.join(model_dir, "playlistai_model.keras")
    norm_path  = os.path.join(model_dir, "norm_stats.npz")
    for p in (model_path, norm_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run phase4_train.py first.")

    model = tf.keras.models.load_model(model_path)
    norm  = np.load(norm_path)
    norm_mean, norm_std = norm["mean"], norm["std"]
    log.info("Step 3 — model + norm stats loaded from %s", model_dir)

    # Songs to score: unexplored, has embedding, not yet predicted for this version
    rows = conn.execute(
        """
        SELECT s.song_id, s.embedding
        FROM songs s
        LEFT JOIN predictions p
               ON s.song_id = p.song_id AND p.model_version = ?
        WHERE s.is_explored = 0
          AND s.embedding IS NOT NULL
          AND p.song_id IS NULL
        ORDER BY s.song_id
        """,
        (model_version,),
    ).fetchall()

    total = len(rows)
    log.info("Step 3 — %d unexplored songs to score", total)
    if total == 0:
        log.info("Step 3 — all songs already scored for version %s.", model_version)
        return model_version

    scored = 0
    for batch_start in range(0, total, infer_batch):
        batch = rows[batch_start: batch_start + infer_batch]
        embeddings = np.stack(
            [np.frombuffer(r["embedding"], dtype=np.float32) for r in batch]
        )
        embeddings_n = (embeddings - norm_mean) / norm_std
        scores = model.predict(embeddings_n, verbose=0).ravel()

        conn.executemany(
            "INSERT OR REPLACE INTO predictions (song_id, score, model_version) VALUES (?, ?, ?)",
            [(r["song_id"], float(s), model_version) for r, s in zip(batch, scores)],
        )
        conn.commit()
        scored += len(batch)
        log.info("Step 3 — scored %d / %d", scored, total)

    log.info("Step 3 complete — %d predictions written (version %s)", scored, model_version)
    return model_version


# ── Step 4: Playlist generation ────────────────────────────────────────────────

def _step_generate_playlists(conn, model_version: str, playlist_dir: str):
    """Write one M3U file per playlist definition, ranked by model score."""
    Path(playlist_dir).mkdir(parents=True, exist_ok=True)

    total_predictions = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE model_version = ?",
        (model_version,),
    ).fetchone()[0]

    if total_predictions == 0:
        log.error(
            "Step 4 — no predictions for version '%s'. Run inference first.", model_version
        )
        return

    log.info(
        "Step 4 — generating playlists from %d predictions (version %s) …",
        total_predictions, model_version,
    )

    results = {}
    for name, top_n, genre_filter, score_min, score_max in PLAYLIST_DEFS:
        if genre_filter == "__ungenred__":
            genre_clause = "AND s.genre_tag IS NULL"
            params = (model_version, score_min, score_max)
        elif genre_filter is not None:
            genre_clause = "AND s.genre_tag = ?"
            params = (model_version, score_min, score_max, genre_filter)
        else:
            genre_clause = ""
            params = (model_version, score_min, score_max)

        query = f"""
            SELECT s.file_path, s.artist, s.title, s.duration_seconds, p.score
            FROM predictions p
            JOIN songs s ON s.song_id = p.song_id
            WHERE p.model_version = ?
              AND s.is_explored = 0
              AND p.score BETWEEN ? AND ?
              {genre_clause}
            ORDER BY p.score DESC
            LIMIT {top_n}
        """
        # Re-order params: for genre_filter= query, genre param goes at end
        if genre_filter and genre_filter != "__ungenred__":
            songs = conn.execute(query, (model_version, score_min, score_max, genre_filter)).fetchall()
        else:
            songs = conn.execute(query, (model_version, score_min, score_max)).fetchall()

        safe_name = name.replace("/", "-").replace(" ", " ")
        out_path  = Path(playlist_dir) / f"{safe_name}.m3u"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("#EXTM3U\n")
            for row in songs:
                duration = int(row["duration_seconds"] or -1)
                artist   = row["artist"] or ""
                title    = row["title"] or Path(row["file_path"]).stem
                display  = f"{artist} - {title}" if artist else title
                f.write(f"#EXTINF:{duration},{display}\n")
                f.write(f"{row['file_path']}\n")

        results[name] = len(songs)
        log.info("  %-30s  %d songs → %s", name, len(songs), out_path)

    log.info("Step 4 complete — %d playlists written to %s", len(results), playlist_dir)
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def run_phase5(
    db_path: str = DB_PATH,
    model_dir: str = "model",
    playlist_dir: str = "playlists",
    batch_size: int = 500,
    clip_seconds: int = CLIP_SECONDS,
    skip_extraction: bool = False,
    skip_inference: bool = False,
):
    init_db(db_path)
    conn = get_connection(db_path)

    # ── Step 1 ─────────────────────────────────────────────────────────────────
    if skip_extraction:
        log.info("Step 1 skipped (--skip-extraction).")
    else:
        _step_extract(conn, batch_size, clip_seconds)

    # ── Step 2 ─────────────────────────────────────────────────────────────────
    similarity_hits = _step_genre_fallback(conn)

    # ── Step 3 ─────────────────────────────────────────────────────────────────
    if skip_inference:
        log.info("Step 3 skipped (--skip-inference).")
        version_path = os.path.join(model_dir, "model_version.txt")
        if not os.path.exists(version_path):
            raise FileNotFoundError(
                f"Cannot skip inference without {version_path}. Run phase4_train.py first."
            )
        model_version = Path(version_path).read_text().strip()
        log.info("Using existing predictions for model version %s.", model_version)
    else:
        model_version = _step_inference(conn, model_dir)

    # ── Step 4 ─────────────────────────────────────────────────────────────────
    playlist_results = _step_generate_playlists(conn, model_version, playlist_dir)

    # ── Summary ────────────────────────────────────────────────────────────────
    total_unexplored = conn.execute(
        "SELECT COUNT(*) FROM songs WHERE is_explored = 0"
    ).fetchone()[0]
    total_embedded = conn.execute(
        "SELECT COUNT(*) FROM songs WHERE is_explored = 0 AND embedding IS NOT NULL"
    ).fetchone()[0]
    total_predicted = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE model_version = ?",
        (model_version,),
    ).fetchone()[0]
    score_dist = conn.execute(
        """
        SELECT
            SUM(CASE WHEN p.score >= 0.7 THEN 1 ELSE 0 END) AS high,
            SUM(CASE WHEN p.score >= 0.5 AND p.score < 0.7 THEN 1 ELSE 0 END) AS medium,
            SUM(CASE WHEN p.score >= 0.45 AND p.score < 0.55 THEN 1 ELSE 0 END) AS borderline,
            SUM(CASE WHEN p.score < 0.5 THEN 1 ELSE 0 END) AS low
        FROM predictions p
        JOIN songs s ON s.song_id = p.song_id
        WHERE p.model_version = ? AND s.is_explored = 0
        """,
        (model_version,),
    ).fetchone()

    print("\n" + "=" * 60)
    print("PLAYLISTAI — PHASE 5 SUMMARY")
    print("=" * 60)
    print(f"  Model version        :  {model_version}")
    print(f"  Unexplored songs     :  {total_unexplored:>8,}")
    print(f"  With embeddings      :  {total_embedded:>8,}")
    print(f"  Predicted            :  {total_predicted:>8,}")
    print(f"  Genre fallback fills :  {similarity_hits:>8,}")
    if score_dist:
        print(f"  Score ≥ 0.70 (strong):  {score_dist['high']:>8,}")
        print(f"  Score 0.50–0.70      :  {score_dist['medium']:>8,}")
        print(f"  Score 0.45–0.55 (borderline): {score_dist['borderline']:>4,}")
        print(f"  Score < 0.50 (skip)  :  {score_dist['low']:>8,}")
    print()
    if playlist_results:
        print("  Playlists generated:")
        for name, count in playlist_results.items():
            print(f"    {name:<30}  {count:>4} songs")
    print("=" * 60)
    print(f"\n  M3U files written to: {playlist_dir}")
    print()

    log.info("Phase 5 complete.")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlaylistAI Phase 5 discovery pipeline")
    parser.add_argument(
        "--db-path", default=DB_PATH,
        help="Path to SQLite database (default: %(default)s)",
    )
    parser.add_argument(
        "--model-dir", default="model",
        help="Directory containing trained model + norm_stats.npz (default: %(default)s)",
    )
    parser.add_argument(
        "--playlist-dir", default="playlists",
        help="Directory to write M3U playlist files (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=500,
        help="Songs per rclone download batch (default: %(default)s)",
    )
    parser.add_argument(
        "--clip-seconds", type=int, default=CLIP_SECONDS,
        help="Middle N seconds of audio to use for YAMNet (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip step 1 (embedding extraction) and use embeddings already in DB",
    )
    parser.add_argument(
        "--skip-inference", action="store_true",
        help="Skip step 3 (inference) and regenerate playlists from existing predictions",
    )
    args = parser.parse_args()

    if args.skip_inference and not args.skip_extraction:
        parser.error("--skip-inference requires --skip-extraction")

    run_phase5(
        db_path=args.db_path,
        model_dir=args.model_dir,
        playlist_dir=args.playlist_dir,
        batch_size=args.batch_size,
        clip_seconds=args.clip_seconds,
        skip_extraction=args.skip_extraction,
        skip_inference=args.skip_inference,
    )
