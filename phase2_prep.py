"""
Phase 2: Training Set Preparation

Assembles positive examples (playlisted songs, label=1) and stratified
negative examples (explored-but-not-playlisted songs, label=0), then
writes an 80/20 train/val split to the training_manifest table.

Run after phase1_scan.py and phase1_playlists.py:
    python phase2_prep.py [--neg-size 8000] [--seed 42]

Environment variables:
    PLAYLISTAI_DB_PATH — path to the SQLite database
"""

import random
import logging
import collections

from config import DB_PATH, NEGATIVE_SAMPLE_SIZE, RANDOM_SEED, TRAIN_VAL_SPLIT
from db import init_db, get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("phase2_prep.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def _bucket(genre_tag):
    """Normalise a genre_tag to a bucket name; None → 'Unknown'."""
    return genre_tag if genre_tag else "Unknown"


def _stratified_sample(neg_pool, pos_counts, target, rng):
    """
    Sample `target` song_ids from neg_pool, stratified to mirror the genre
    distribution of positives.

    neg_pool  : list of (song_id, genre_tag)
    pos_counts: Counter of genre_bucket → positive count
    target    : total negatives to sample
    rng       : seeded random.Random instance
    """
    known = set(pos_counts.keys())
    total_pos = sum(pos_counts.values())

    # Group negatives into the same buckets as positives.
    # Any negative genre not in the positive set → 'Unknown'.
    buckets = collections.defaultdict(list)
    for song_id, genre_tag in neg_pool:
        b = _bucket(genre_tag)
        buckets[b if b in known else "Unknown"].append(song_id)

    # Per-bucket targets proportional to positive distribution.
    targets = {g: round(target * c / total_pos) for g, c in pos_counts.items()}
    # Fix rounding drift so targets sum exactly to `target`.
    diff = target - sum(targets.values())
    if diff:
        largest = max(targets, key=targets.get)
        targets[largest] += diff

    sampled = []
    used = set()
    shortage = 0

    for genre, n in targets.items():
        pool = buckets.get(genre, [])
        rng.shuffle(pool)
        take = pool[:n]
        sampled.extend(take)
        used.update(take)
        shortage += max(0, n - len(pool))

    # Fill any shortfall from leftover negatives across all buckets.
    if shortage > 0:
        leftovers = [sid for sids in buckets.values() for sid in sids if sid not in used]
        rng.shuffle(leftovers)
        sampled.extend(leftovers[:shortage])

    return sampled[:target]


def run_training_prep(
    db_path: str = DB_PATH,
    neg_size: int = NEGATIVE_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
    val_fraction: float = 1.0 - TRAIN_VAL_SPLIT,
):
    init_db(db_path)
    conn = get_connection(db_path)
    rng = random.Random(seed)

    # ── Step 2.1: Positive examples ────────────────────────────────────────────
    log.info("Step 2.1 — Loading positive examples (playlisted songs) …")
    pos_rows = conn.execute(
        """
        SELECT DISTINCT s.song_id, s.genre_tag
        FROM songs s
        JOIN playlist_songs ps ON s.song_id = ps.song_id
        """
    ).fetchall()

    if not pos_rows:
        log.error("No playlisted songs found — run phase1_playlists.py first.")
        conn.close()
        return

    pos_song_ids = [r["song_id"] for r in pos_rows]
    pos_counts = collections.Counter(_bucket(r["genre_tag"]) for r in pos_rows)

    log.info(
        "Step 2.1 complete — %d positives across %d genre buckets",
        len(pos_song_ids),
        len(pos_counts),
    )
    for genre, count in pos_counts.most_common():
        log.info("  %-20s %d  (%.1f%%)", genre, count, 100 * count / len(pos_song_ids))

    # ── Step 2.2: Stratified negative sample ───────────────────────────────────
    log.info("Step 2.2 — Sampling %d negatives from explored pool …", neg_size)

    neg_pool_rows = conn.execute(
        """
        SELECT song_id, genre_tag
        FROM songs
        WHERE is_explored = 1
          AND song_id NOT IN (SELECT song_id FROM playlist_songs)
        """
    ).fetchall()
    neg_pool = [(r["song_id"], r["genre_tag"]) for r in neg_pool_rows]
    log.info("Negative pool: %d songs available", len(neg_pool))

    if len(neg_pool) < neg_size:
        log.warning(
            "Pool (%d) smaller than target (%d) — using all available.",
            len(neg_pool),
            neg_size,
        )
        neg_size = len(neg_pool)

    neg_song_ids = _stratified_sample(neg_pool, pos_counts, neg_size, rng)
    log.info("Step 2.2 complete — %d negatives sampled", len(neg_song_ids))

    # ── Step 2.3: Combine, shuffle, split ──────────────────────────────────────
    log.info("Step 2.3 — Creating %.0f/%.0f train/val split …",
             100 * (1 - val_fraction), 100 * val_fraction)

    manifest = [(sid, 1) for sid in pos_song_ids] + [(sid, 0) for sid in neg_song_ids]
    rng.shuffle(manifest)

    val_count = round(len(manifest) * val_fraction)
    val_ids = {sid for sid, _ in manifest[:val_count]}

    rows = [
        (song_id, label, "val" if song_id in val_ids else "train")
        for song_id, label in manifest
    ]
    train_count = len(rows) - val_count

    log.info(
        "Split: %d train / %d val  (%.0f%% / %.0f%%)",
        train_count, val_count,
        100 * train_count / len(rows),
        100 * val_count / len(rows),
    )

    # ── Write to DB (idempotent — clears previous run first) ───────────────────
    log.info("Writing %d rows to training_manifest …", len(rows))
    conn.execute("DELETE FROM training_manifest")
    conn.executemany(
        "INSERT INTO training_manifest (song_id, label, split) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()

    # ── Summary ────────────────────────────────────────────────────────────────
    t_pos = conn.execute(
        "SELECT COUNT(*) FROM training_manifest WHERE label=1 AND split='train'"
    ).fetchone()[0]
    t_neg = conn.execute(
        "SELECT COUNT(*) FROM training_manifest WHERE label=0 AND split='train'"
    ).fetchone()[0]
    v_pos = conn.execute(
        "SELECT COUNT(*) FROM training_manifest WHERE label=1 AND split='val'"
    ).fetchone()[0]
    v_neg = conn.execute(
        "SELECT COUNT(*) FROM training_manifest WHERE label=0 AND split='val'"
    ).fetchone()[0]

    print("\n" + "=" * 60)
    print("PLAYLISTAI — PHASE 2 TRAINING MANIFEST SUMMARY")
    print("=" * 60)
    print(f"  Total rows      :  {len(rows):>8,}")
    print(f"  Train positives :  {t_pos:>8,}")
    print(f"  Train negatives :  {t_neg:>8,}")
    print(f"  Val   positives :  {v_pos:>8,}")
    print(f"  Val   negatives :  {v_neg:>8,}")
    print("=" * 60)

    log.info("Phase 2 complete.")
    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PlaylistAI Phase 2 training prep")
    parser.add_argument(
        "--neg-size", type=int, default=NEGATIVE_SAMPLE_SIZE,
        help="Number of negative examples to sample (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed for reproducibility (default: %(default)s)",
    )
    args = parser.parse_args()

    run_training_prep(neg_size=args.neg_size, seed=args.seed)
