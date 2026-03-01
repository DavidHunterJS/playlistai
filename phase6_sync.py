"""
Phase 6B: Sync Subsonic Stars → Feedback Table

Pulls all starred songs from Subsonic and records them as positive feedback
(rating=1) in the local database.  Run this periodically on the seedbox after
you've listened to and starred songs from the discovery playlists.

Usage:
    python phase6_sync.py [--db-path ~/playlistai/playlistai.db]

When you have enough new feedback (suggested: 50+ new stars), retrain:
    1. scp ~/playlistai/playlistai.db root@RUNPOD_IP:/workspace/playlistai/
    2. On RunPod: python phase2_prep.py            (rebuild training manifest)
    3. On RunPod: python phase3_extract.py         (extract embeddings for any new songs)
    4. On RunPod: python phase4_train.py           (retrain model)
    5. On RunPod: python phase5_inference.py --skip-extraction  (new predictions + playlists)
    6. scp -r root@RUNPOD_IP:/workspace/playlistai/playlists ~/playlistai/playlists
    7. python phase6_push.py                       (push updated playlists to Subsonic)

Environment variables (or set in .env):
    SUBSONIC_URL, SUBSONIC_USER, SUBSONIC_PASS, SUBSONIC_MUSIC_ROOT
"""

import json
import logging
import argparse
import urllib.request
import urllib.parse
from pathlib import Path

from config import (
    DB_PATH,
    SUBSONIC_URL, SUBSONIC_USER, SUBSONIC_PASS,
    SUBSONIC_API_VERSION, SUBSONIC_CLIENT,
    SUBSONIC_MUSIC_ROOT,
)
from db import init_db, get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("phase6_sync.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Subsonic API ───────────────────────────────────────────────────────────────

def _api(endpoint: str, **params) -> dict:
    base = SUBSONIC_URL.rstrip("/")
    qs = urllib.parse.urlencode({
        "u": SUBSONIC_USER,
        "p": SUBSONIC_PASS,
        "v": SUBSONIC_API_VERSION,
        "c": SUBSONIC_CLIENT,
        "f": "json",
        **params,
    })
    url = f"{base}/rest/{endpoint}?{qs}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = json.loads(resp.read())
    outer = data["subsonic-response"]
    if outer["status"] != "ok":
        raise RuntimeError(f"Subsonic error on {endpoint}: {outer.get('error')}")
    return outer


# ── Sync logic ─────────────────────────────────────────────────────────────────

def _get_starred_paths() -> list[str]:
    """
    Fetch all starred songs from Subsonic.
    Returns list of absolute file paths (SUBSONIC_MUSIC_ROOT + relative path).
    """
    data = _api("getStarred2")
    songs = data.get("starred2", {}).get("song", [])
    root = SUBSONIC_MUSIC_ROOT.rstrip("/")
    return [f"{root}/{s['path']}" for s in songs if "path" in s]


def sync_feedback(db_path: str = DB_PATH) -> None:
    if not SUBSONIC_URL:
        raise RuntimeError("SUBSONIC_URL is not set. Check your .env file.")

    init_db(db_path)
    conn = get_connection(db_path)

    log.info("Fetching starred songs from Subsonic …")
    starred_paths = _get_starred_paths()
    log.info("Found %d starred songs on Subsonic", len(starred_paths))

    if not starred_paths:
        log.info("No starred songs found — nothing to sync.")
        conn.close()
        return

    # Look up song_ids for the starred paths
    placeholders = ",".join("?" * len(starred_paths))
    rows = conn.execute(
        f"SELECT song_id, file_path FROM songs WHERE file_path IN ({placeholders})",
        starred_paths,
    ).fetchall()

    matched = {r["file_path"]: r["song_id"] for r in rows}
    unmatched = [p for p in starred_paths if p not in matched]

    if unmatched:
        log.warning(
            "%d starred paths not found in DB (songs scanned after Phase 1?): "
            "first few: %s",
            len(unmatched), unmatched[:3],
        )

    # Count existing feedback before update
    existing_feedback = conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE rating = 1"
    ).fetchone()[0]

    # Upsert: insert new feedback, keep existing
    if matched:
        conn.executemany(
            """
            INSERT INTO feedback (song_id, rating) VALUES (?, 1)
            ON CONFLICT(song_id) DO UPDATE SET rating = 1, rated_at = CURRENT_TIMESTAMP
            """,
            [(sid,) for sid in matched.values()],
        )
        conn.commit()

    new_feedback = conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE rating = 1"
    ).fetchone()[0]
    new_this_run = new_feedback - existing_feedback

    # Detailed breakdown: how many of the feedback songs have predictions
    feedback_with_predictions = conn.execute(
        """
        SELECT COUNT(DISTINCT f.song_id)
        FROM feedback f
        JOIN predictions p ON f.song_id = p.song_id
        WHERE f.rating = 1
        """
    ).fetchone()[0]

    print("\n" + "=" * 60)
    print("PLAYLISTAI — PHASE 6B FEEDBACK SYNC SUMMARY")
    print("=" * 60)
    print(f"  Starred on Subsonic      :  {len(starred_paths):>6,}")
    print(f"  Matched in DB            :  {len(matched):>6,}")
    print(f"  Unmatched (not in DB)    :  {len(unmatched):>6,}")
    print(f"  New feedback this run    :  {new_this_run:>6,}")
    print(f"  Total feedback (rating=1):  {new_feedback:>6,}")
    print(f"  Of those, were predicted :  {feedback_with_predictions:>6,}")
    print("=" * 60)

    if new_feedback >= 50:
        print(f"\n  {new_feedback} starred songs in feedback table.")
        print("  Ready to retrain! Follow the steps in phase6_sync.py docstring.")
    else:
        remaining = 50 - new_feedback
        print(f"\n  {remaining} more stars needed before retraining is worthwhile.")

    print()
    log.info("Feedback sync complete — %d total positive feedback rows.", new_feedback)
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Subsonic stars to feedback table")
    parser.add_argument(
        "--db-path", default=DB_PATH,
        help="Path to SQLite database (default: %(default)s)",
    )
    args = parser.parse_args()
    sync_feedback(args.db_path)
